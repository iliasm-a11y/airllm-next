# Lecture CPU des poids Q8 mappés dans Metal — patch expérimental

Le patch `patches/llama-metal-mapped-cpu-read.patch` modifie uniquement
`ggml_backend_sched_buffer_supported` et ses helpers dans `ggml-backend.cpp`.
Après vérification d'application, il a été compilé dans le runtime isolé
de la nouvelle campagne. Le backend de production reste inchangé.

## Justification vérifiée dans le code épinglé

`ggml-metal.cpp` définit `_Mapped.is_host=false`. Le scheduler considère donc
ce buffer incompatible avec le CPU et peut allouer/copier les poids utilisés
par une opération de préfill CPU. Dans `ggml-metal-device.m`, le mapping stocke
pourtant `all_data=ptr`, `is_shared=true`, `owned=false`, puis crée un buffer
Metal `newBufferWithBytesNoCopy` en mémoire partagée. Son `get_base` retourne
l'adresse d'origine. Le mmap des poids est créé avec `PROT_READ` dans
`llama-mmap.cpp`.

Cela justifie d'essayer une lecture CPU directe de ces mêmes octets immuables.
Une particularité oblige à vérifier plus que le nom : le type `_Mapped` peut
également allouer un buffer partagé propriétaire **modifiable**. Ce cas doit
rester exclu.

## Admission et portée

La nouvelle compatibilité exige simultanément :

- `AIRLLM_METAL_CPU_MAPPED_WEIGHTS=1` et `AIRLLM_GLM_Q8_ONLY=1`, exactement;
- backend destinataire identifié comme CPU;
- type de buffer exactement `MTL0_Mapped`, associé au device GPU `MTL0`;
- usage `WEIGHTS`, type de poids `Q8_0`, feuille constante ou alias de feuille;
- absence de flags input/output/compute sur la chaîne d'alias;
- adresses non nulles et plage du tenseur entièrement contenue dans le buffer;
- protections VM Darwin `READ` sans `WRITE` pour toute la plage, limitée à
  1 Gio et 64 régions VM inspectées. Un échec de requête refuse l'exception.

Les allocations `_Private`, les types partagés ordinaires, le type `_Mapped`
propriétaire modifiable et les buffers d'activations sont refusés. Aucun
`is_host` global n'est changé. Les autres copies et synchronisations du
scheduler restent intactes. Seules des feuilles de poids immuables peuvent
éviter leur copie; aucun résultat GPU mutable n'obtient cette exception.
L'admission est recalculée et ne conserve pas d'adresses en cache entre
libérations/réutilisations.

Les options sont désactivées par défaut. Ce patch est limité au chemin
expérimental Q8 fidèle et au device zéro du Mac cible. Les autres workspaces
et coûts du préfill restent présents; cette admission ne remplace aucun
garde-fou RSS, swap ou pression mémoire.

## Fixture réelle et limites de preuve

`mapped_cpu_fixture.cpp` utilise le vrai API GGML de mapping Metal sur un
fichier local de **68 octets de poids Q8_0**, dans une page mémoire en lecture
seule. Le helper extrait de ce patch est compilé avec les bibliothèques
épinglées. La fixture contrôle notamment les deux options, le pointeur de
base d'origine, l'usage activation, les flags mutables, les dépassements de
plage, un buffer propriétaire `_Mapped` modifiable et un buffer partagé
ordinaire. Elle exécute ensuite un petit produit sur **CPU**, directement
à partir des poids mappés Metal, et le compare bit à bit à une copie CPU.
Aucun modèle complet ni produit matriciel GPU n'est exécuté.

Le premier préflight réel a réussi **13 contrôles** et confirmé la base
originale et les deux sorties CPU identiques. L'initialisation du device Metal
et de ses bibliothèques a néanmoins lieu; cette fixture doit être isolée des
mesures de mémoire/I/O concurrentes.

Cette fixture vérifie l'admission et la possibilité de lecture CPU; à elle
seule, elle ne prouve **pas** que le scheduler supprime les copies dans GLM.
Son champ `scheduler_copy_elimination` reste `not_tested`; aucune preuve
ultérieure n'a été réécrite dans ce résultat initial.

## Vérification ultérieure dans le modèle complet

Le premier placement sélectif sur cinq couches, sans cette exception, a
réservé un workspace CPU de **677,50 Mio** et a été rejeté par le garde-fou
de croissance du swap. L'essai est conservé avec son statut d'échec.
Le même placement avec lecture CPU directe a ensuite réservé
**11,0011 Mio** de workspace CPU. Le placement sur douze couches conserve
la même taille de workspace CPU. Ces valeurs sont les lignes réelles
`CPU compute buffer size` des journaux du runtime; elles ne représentent
ni toute la mémoire CPU ni tout le RSS.

Les essais cinq et douze couches à quatre étapes passent les 619 520 logits
et toutes les familles capturées bit à bit. Le dernier contrôle complet douze couches
à seize étapes passe **2 478 080 logits**, 7 056 éléments de routes ordonnées,
254 016 éléments de routeurs et 11 698 176 éléments de sorties KDA, tous
identiques en bits. Son pic RSS échantillonné est de 6 354 485 248 octets,
sans croissance échantillonnée du swap système. Les limites restent
8,5 Gio RSS et 32 Mio de croissance du swap, avec pression critique
interdite. Une quantité de swap système préexistait; croissance nulle
ne signifie pas swap absolu nul.

Ces preuves se trouvent dans les essais bruts
`quant-tail5-fidelity4`, `quant-tail5-mapped-fidelity4` et
`quant-tail12-mapped-fidelity16-timing`. Les empreintes des neuf bibliothèques
natives du dernier essai sont conservées. Le diagnostic à seize étapes
dure 89,385 s avec captures et sérialisation : ce temps ne constitue pas
un benchmark de débit. La fidélité établie concerne cette trajectoire
forcée et les tenseurs capturés; les performances soutenues, les contextes
longs et les états non capturés exigent leurs preuves distinctes.

Un durcissement ultérieur vérifie aussi les flags de la feuille atteinte à
la limite de seize alias. Le runtime livré contenant cette garde passe
20 contrôles de la fixture réelle, mais possède une nouvelle empreinte
native. Le diagnostic CPU64 lancé ensuite s'est arrêté au garde-fou de
swap après 18 sorties. Il n'existe donc aucune nouvelle preuve complète
de fidélité sélective ou de débit pour ce build durci; les résultats à
seize étapes ci-dessus appartiennent aux bibliothèques antérieures
explicitement identifiées dans leurs rapports.

Tests :

```sh
python -m pytest -q research/glm53_gpu_v2/test_mapped_cpu_read.py
AIRLLM_TEST_METAL_MAPPED=1 python -m pytest -q \
  research/glm53_gpu_v2/test_mapped_cpu_read.py
```

Le second lancement initialise explicitement une petite ressource Metal.
Les chemins source/bibliothèques se règlent avec
`AIRLLM_PROFILE_RUNTIME_SOURCE` et `AIRLLM_PROFILE_RUNTIME_LIBRARIES`; les
répertoires de travail voisins sont utilisés sinon. Les tests de portée,
d'application et de compilation ne démarrent aucune inférence.
