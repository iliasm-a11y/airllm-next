# Décodage spéculatif n-gram, GLM-5.3 / M5

Prototype de recherche distinct de la production, basé sur la PR #25
(`195f7745afc3d0ca3fc8aca776287264e16f3a2b`). Un seul chargement du checkpoint
Q2_K_S original, un seul contexte physique, CPU10/GPU0, contexte512,
batch/ubatch8. Aucun second modèle, aucune conversion des poids.

Le proposeur est **`common_ngram_simple_draft` original**, compilé depuis
`common/ngram-map.cpp` de llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c`. Il n'est pas réécrit. Le petit
`log_bridge.cpp` désactive uniquement ses messages de diagnostic ; les calculs
du modèle restent dans les bibliothèques V4 existantes et vérifiées.

## Contrat expérimental

Le chemin proposé vérifie deux ou quatre tokens anticipés, donc **trois ou
cinq entrées** avec le token déjà connu. Seule la forme exacte ayant passé
l'audit est autorisée. Les propositions plus courtes et la fin du budget
repassent en génération séquentielle ; aucune forme intermédiaire n'hérite
d'une qualification. Les historiques utilisés par le proposeur appartiennent
à la requête courante ; aucun historique privé n'est partagé entre agents.

Le modèle contient des états récurrents KDA, des caches d'attention et un
indexeur DSA. Une simple troncature des positions ne suffit pas. Avant une
vérification, le prototype conserve un snapshot complet borné. Après un rejet,
il restaure ce snapshot puis rejoue séquentiellement les entrées acceptées.
Il utilise `llama_state_get_data` / `llama_state_set_data` et non une hypothèse
sur le support de `seq_rm` pour ce modèle. Le coût des snapshots, restaurations,
rejeux, propositions et copies des logits appartient au chronométrage.

## Qualification avant mesure

`--mode audit` ne mesure **aucun débit**. Il construit des fixtures de diagnostic
à partir de vrais tokens GLM, puis force trois situations : proposition correcte,
rejet immédiat et rejet partiel. Ces propositions oracle servent uniquement au
contrôle numérique ; ce ne sont pas des propositions n-gram réussies.

Chaque situation compare, sur les mêmes entrées et le même contexte restauré :

- la référence séquentielle et la vérification en lot ;
- tous les logits, bit à bit avec le critère CPU existant ;
- les routes ordonnées, les sorties de routeur et les sorties KDA par token ;
- la couverture des écritures de caches et leur finitude ;
- le hash de la sérialisation complète de séquence en fin de bloc ;
- le retour arrière et l'état final du préfixe accepté après rejeu.

Le premier échec interdit le chronométrage spéculatif. Le reçu est lié au binaire,
aux bibliothèques, à la forme de bloc et au prompt. Le passage de ces fixtures
reste une qualification ciblée, pas une preuve générale pour tous les contextes.

`--mode reference` puis `--mode ngram --qualification .../report.json` produisent
des trajectoires indépendantes. Elles doivent être égales, EOS compris. Effectuer
des paires alternées A/B/B/A avant de conclure à un gain. Comparer le temps complet
de génération après préfill et premier token, pas seulement les appels de vérification.
Les appels et positions rejoués sont explicitement comptés. Les compteurs OS ne
mesurent pas des octets NAND.

## Protections

Admission inchangée : 5,5 Gio disponibles, pression normale, réserve disque32 Gio,
secteur et capot ouvert. Surveillance du processus possédé, RSS8,5 Gio,
croissance globale du swap32 Mio et continuité des horloges. Le verrou UID/modèle
est acquis avant le lancement et hérité par l'auxiliaire natif. Un essai
concurrent est refusé ; aucun fichier de verrou n'est supprimé.

Les snapshots natifs sont plafonnés à256 Mio et les valeurs de référence d'un
bloc à64 Mio, avec vérification du RSS avant les nouvelles allocations. Le RSS
global échantillonné et le pic natif de durée de vie complètent ces budgets ;
ils ne constituent pas une mesure exhaustive de mémoire physique ou de swap
par processus.

Après l'arrêt d'un moteur, `prepare.py` peut appliquer `MADV_DONTNEED` à des
mappings en lecture seule du checkpoint vérifié, sous le verrou UID. Il ne
parcourt pas les poids et ne lance pas de modèle. Cette préparation peut rendre
des pages de fichiers récupérables par Darwin ; elle ne garantit ni un cache
froid ni un volume de RAM physique libérée. Le lancement suivant doit toujours
passer séparément l'admission inchangée de5,5 Gio et tous les garde-fous. En
cas de comparaison chronométrée, appliquer la même préparation à chaque bras
et conserver ses reçus ; ne pas comparer un bras préparé à un bras non préparé.
Le helper attend une fenêtre normale d'au moins2 secondes au-dessus de5,5 Gio,
avec une limite de60 secondes. Il ne relance aucun modèle pendant cette attente.

## Reproduction

Depuis la racine du dépôt, `PYTHONPATH=.:air_llm`, avec Python disposant de
NumPy, psutil et pytest. Tous les dossiers de sortie doivent être nouveaux.

```sh
python -m research.glm53_ngram.build \
  --source /chemin/llama.cpp-3f8af15 \
  --runtime /chemin/runtime-v4/bin --output /chemin/nouveau-build

GLM53_NGRAM_PROBE=/chemin/nouveau-build/ngram-probe \
  python -m pytest -q research/glm53_ngram

python -m research.glm53_ngram.run \
  --model /chemin/GLM-5.3-Flash-Q2_K_S --receipt /chemin/recu-integrite.json \
  --runtime /chemin/runtime-v4/bin --probe /chemin/nouveau-build/ngram-probe \
  --output /chemin/audit2 --mode audit --draft-tokens 2

python -m research.glm53_ngram.run \
  --model /chemin/GLM-5.3-Flash-Q2_K_S --receipt /chemin/recu-integrite.json \
  --runtime /chemin/runtime-v4/bin --probe /chemin/nouveau-build/ngram-probe \
  --output /chemin/temoin32 --mode reference --tokens 32

python -m research.glm53_ngram.run \
  --model /chemin/GLM-5.3-Flash-Q2_K_S --receipt /chemin/recu-integrite.json \
  --runtime /chemin/runtime-v4/bin --probe /chemin/nouveau-build/ngram-probe \
  --output /chemin/ngram2-32 --mode ngram --tokens 32 --draft-tokens 2 \
  --qualification /chemin/audit2/report.json
```

Transmettre la même valeur initiale à `--campaign-swap-used-bytes` pour toutes
les étapes d'une campagne. Elle ne peut qu'abaisser le budget restant.
Répéter l'audit séparément pour quatre tokens anticipés. Un changement de prompt
requiert son propre audit et ses comparaisons complètes.

`campaign.py` exécute une seule suite A/B/B/A, prépare le cache de la même
manière pour chaque bras, conserve une origine commune de swap et arrête
l'enchaînement au premier refus ou échec. Il vérifie aussi les empreintes des
sources avant chaque bras. Exemple après un audit2 réussi :

```sh
python -m research.glm53_ngram.campaign \
  --model /chemin/GLM-5.3-Flash-Q2_K_S --receipt /chemin/recu-integrite.json \
  --runtime /chemin/runtime-v4/bin --probe /chemin/nouveau-build/ngram-probe \
  --qualification /chemin/audit2/report.json \
  --output /chemin/nouvelle-campagne32 --tokens 32 --draft-tokens 2 --ngram-size 2
```

`opportunities.py` rejoue des IDs historiques sans ouvrir le modèle. Ses résultats
permettent de choisir des tâches ; ils ne sont jamais une mesure d'acceptation
réelle, de fidélité spéculative ou de débit.

Les preuves sélectionnées sont dans `evidence/selection.json`. Les journaux
`.log` et `.jsonl` sont publiés en gzip, sans réécriture des espaces du journal ;
les chemins de la machine sont remplacés par des alias. Le manifeste distingue
les empreintes des fichiers sources, des textes expurgés et des fichiers gzip.
