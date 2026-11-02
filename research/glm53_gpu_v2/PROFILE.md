# Profilage CPU : opérations, résidence des poids et compteurs macOS

`native_profile.cpp` reprend le diagnostic `native_trace.cpp` et ajoute une
observation de résidence **sans lecture des poids**. Il ne modifie pas le
runtime épinglé et ne change aucune opération du modèle. Les deux options
restent désactivées par défaut :

- `AIRLLM_PROFILE_OPS=1` : temps mur et compteurs par opération, comme le
  diagnostic existant.
- `AIRLLM_PROFILE_PAGES=1` : inspection des pages avant chaque multiplication
  CPU sur un buffer de poids du modèle. Cette option active également le
  journal des opérations. Seules les valeurs `0` et `1` sont acceptées.

## Inspection bornée

Le helper vérifie que `src0` possède un buffer hôte de type `WEIGHTS`, que sa
plage appartient à ce buffer et que les tailles restent bornées. Les produits
sur activations/états intermédiaires, ou sur un buffer GPU, sont explicitement
marqués non applicables.

Pour `MUL_MAT`, la plage du tenseur de poids est inspectée. Pour `MUL_MAT_ID`,
le helper lit seulement le petit tenseur I32 `src2` déjà calculé, respecte ses
strides, valide les IDs, déduplique les experts réellement utilisés dans le
microbatch et inspecte leurs plages exactes de longueur `src0.nb[2]`. Il ne
parcourt pas toute la banque d'experts. Les pages de bord partagées sont
fusionnées pour éviter de les compter deux fois dans une opération.

Les plages logiques et l'union des pages inspectées sont limitées à **1 Gio**;
le vecteur de pages est limité à **1 Mio**. La requête Darwin `mincore` reçoit
les adresses et ne déréférence pas les octets des poids. L'inspection et son
temps propre sont terminés **avant** le départ du chronomètre et des snapshots
CPU/I/O de l'opération. Son coût figure séparément dans `inspection_seconds`.
Elle peut néanmoins modifier l'ordonnancement et prendre des verrous VM : ce
profilage n'est pas une mesure de débit de production.

Chaque ligne de `operations.jsonl` ajoute `weight_pages_before` avec :
`logical_selected_weight_bytes`, `selected_expert_count`, `page_size`,
`inspected_pages`, `resident_pages`, `nonresident_pages`, statut, raison,
`errno` et temps d'inspection. Les compteurs de pages sont **null** lorsqu'ils
sont inconnus, non applicables, non disponibles ou en erreur. Un échec ne
produit jamais de faux zéro de résidence. Les échecs restent des diagnostics;
ils ne changent ni le calcul ni les garde-fous mémoire du modèle.

## Résumé reproductible

Après un essai surveillé exécuté par le lanceur habituel :

```sh
python -m research.glm53_gpu_v2.profile_summary \
  /path/to/capture/operations.jsonl --output /path/to/profile-summary.json
```

Le résumé donne les totaux par opération, couche, étape, phase et combinaison
opération/couche/étape. Dans ce helper, l'étape zéro correspond au préfill du
prompt; les étapes suivantes décodent un token. Les métriques manquantes
restent nulles. Les erreurs et non-applicabilités de résidence sont comptées
explicitement. Un journal vide est refusé.

Ces colonnes répondent à des questions différentes :

- Le temps mur d'une opération inclut calcul, accès mémoire, attentes de défauts
  de page et ordonnancement. Il ne se décompose pas automatiquement en un temps
  I/O et un temps de calcul.
- Le temps CPU est sommé entre les threads. Son ratio au temps mur peut dépasser
  un et ne représente pas un pourcentage de temps passé en calcul.
- Les octets logiques des experts sélectionnés représentent les plages requises,
  sans prouver un trafic disque. Les sommes de pages sont des **snapshots répétés**
  et ne comptent pas des pages uniques à l'échelle de l'essai.
- Une page non résidente peut être satisfaite par le cache de fichiers macOS.
  Elle ne démontre pas une lecture NAND. Les deltas de lectures OS par processus
  et les compteurs de pageins sont conservés séparément.

Les captures sérialisent le graphe et écrivent des fichiers entre les opérations.
Ces écritures sont hors du chronomètre d'opération, mais peuvent perturber l'essai.
Une attribution quantitative à un SSD Fanxiang nécessite sa présence effective
et un protocole séparé; ce helper ne la suppose pas.

## Validation du mécanisme

`test_profile_pages.py` comprend cinq tests. La fixture C++ mappe un vrai
fichier avec `PROT_NONE`, puis interroge ses pages : une lecture accidentelle
ferait échouer le processus. Elle vérifie également la fusion de plages,
le refus au-delà de 1 Gio et une erreur `ENOSYS` injectée **uniquement dans la
compilation de test**, qui doit produire des métriques de pages nulles.
Un second test natif crée un vrai buffer GGML `CPU_Mapped` sur ce mapping
protégé, puis vérifie le parcours complet des `MUL_MAT_ID` avec IDs stridés et
dupliqués, un produit dense et un ID invalide. Dans le runtime épinglé, les
types `CPU` et `CPU_Mapped` exposent `device=NULL`; seuls ces deux noms exacts
sont admis avec un buffer hôte. Les buffers Metal partagés restent exclus.
Chaque observation enregistre le nom/type du buffer, son device et son usage.
Les tests Python vérifient les agrégations, erreurs et métriques inconnues.
Aucune inférence complète n'est exécutée par ces tests.

Compilation de la fixture indépendante :

```sh
clang++ -std=c++17 -O2 -DAIRLLM_PROFILE_RESIDENCY_TEST \
  research/glm53_gpu_v2/native_profile.cpp -o /path/to/profile-pages-fixture
```

Pour le helper complet, utiliser les mêmes includes, bibliothèques et rpath
que `native_trace.cpp`, puis activer explicitement les options de profilage
lors du lancement surveillé. Les politiques CPU/Metal, captures de fidélité,
limites de contexte, threads, tokens et mémoire restent celles du diagnostic.

Le premier essai complet `cpu-profile16` utilisait une garde trop stricte qui
exigeait un device CPU enregistré et a donc marqué toutes les résidences non
applicables. Ses métriques de pages sont inconnues; ses compteurs opérationnels
restent des données distinctes. La correction est vérifiée par la fixture GGML
réelle avant le nouvel essai matériel. Elle ne transforme pas l'ancien essai
en preuve de résidence.

Le test de tensor GGML utilise `AIRLLM_PROFILE_RUNTIME_SOURCE` et
`AIRLLM_PROFILE_RUNTIME_LIBRARIES`, ou les répertoires de travail voisins
`llama-v2` et `runtime-v2-build/bin`; il est ignoré si le runtime épinglé
n'est pas disponible. `AIRLLM_PROFILE_DETAIL=1` permet les captures détaillées
de la couche 40; ce détail supplémentaire est désactivé par défaut.
