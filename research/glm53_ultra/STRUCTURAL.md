# Accélérations structurelles exactes : décision pour cette campagne

La piste ngram est close **pour les trajectoires mesurées dans cette campagne**.
Le rejeu de leurs IDs offre trop peu de continuations concordantes pour justifier
un nouveau chemin natif et son coût mémoire. Aucun moteur spéculatif n'a été
exécuté ; aucun gain de débit spéculatif n'est annoncé. D'autres charges,
notamment du texte très répétitif, pourraient justifier une nouvelle étude.

## Entrées et reproduction

Le bundle [structural-inputs](evidence/structural-inputs/manifest.json) contient
environ 21 Ko : quatre copies exactes des seuls `prompt_tokens.json`,
`tokens.jsonl` et `complete.json`, un extrait explicite des IDs de douze requêtes
serveur closes, et les dimensions nécessaires au calcul KDA. Il ne contient
ni tenseurs capturés, ni poids. Le prompt est public :
« La capitale de la France est ».

Les noms de fichiers sources sont relatifs au répertoire de travail de la
campagne précédente et leurs SHA-256 sont conservés. Les douze extraits de
benchmark restent identifiés comme des données dérivées, avec hash du JSON
source complet ; ils ne deviennent pas de fausses captures natives.
Les IDs du prompt des benchmarks viennent de la capture native du même texte
et du même checkpoint, sans nouvelle tokenisation. Le bundle permet de
recalculer les opportunités à partir de ces extraits, pas d'authentifier à lui
seul les JSON complets conservés dans la campagne précédente.

Depuis la racine du dépôt, vers un fichier de résultat encore inexistant :

```sh
python3 -m research.glm53_ultra.structural \
  --bundle research/glm53_ultra/evidence/structural-inputs \
  --output /tmp/glm53-structural-replay.json
```

Ce programme lit uniquement les petites métadonnées, vérifie le manifeste et
refuse une capture incomplète ou un token d'entrée forcé différent du token
généré. Les requêtes restent séparées. Le
[résultat livré](evidence/structural-analysis.json) utilise des chemins d'entrée
relatifs au bundle. Les tests portables sont dans `test_structural.py` ; ils
n'ont pas été lancés pendant la campagne de débit concurrente.

## MTP disponible dans le code, absent de ce checkpoint

L'inventaire sauvegardé du GGUF énumère **1 383 tenseurs**, avec un indice de
bloc maximal de **44**. Aucun nom ne contient `mtp`, `nextn` ou `blk.45`.
La métadonnée déclare 45 blocs et aucune couche NextN supplémentaire.
Cette vérification porte sur l'inventaire des headers déjà sauvegardé ; aucun
nouveau parcours des 110 Go de poids n'a été effectué pour cette analyse.
Le fichier `model-dimensions.json` conserve le hash de l'inventaire source.

Le runtime épinglé à
`3f8af15a76ac8675812e856da45454bafc90f12c` possède un
[graphe MTP GLM](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/src/models/glm5-next.cpp),
qui exige une couche NextN chargée. Son existence ne crée pas les poids
manquants. Activer MTP sur cette variante ne constitue donc pas une option
utilisable en conservant exactement le checkpoint demandé. Aucun modèle de
draft externe n'est substitué ou téléchargé.

## Ngram : proposition bon marché, vérification complète obligatoire

Le [lookup natif simple](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/common/ngram-map.cpp)
recherche dans les tokens déjà connus une continuation d'un motif récent.
Il ne demande pas de second modèle. Il ignore la position zéro, recherche
en ordre inverse, impose une histoire assez longue, et refuse une proposition
plus courte que le motif recherché. Ainsi **N=2/M=1 ne propose jamais de token**
dans cette implémentation. Ces règles sont reproduites dans `simple_draft`.

Le rejeu inspecte chaque position de la trajectoire séquentielle existante.
Les opportunités peuvent se recouvrir : leurs comptes ne sont pas un nombre
de passages réellement évités et ne doivent pas être additionnés comme tel.

| Taille du motif N / proposition M | Propositions par trajectoire | Premier token concordant | Proposition entière concordante |
|---|---:|---:|---:|
| 12 / 48, valeurs natives par défaut | 0 | 0 | 0 |
| 4 / 4 | 0 | 0 | 0 |
| 2 / 2 | 1 | 0 | 0 |
| 1 / 1 | 2 | 1 | 1 |
| 2 / 1 | 0 | 0 | 0 |

Le tableau est identique sur chacune des quatre captures de seize sorties et
sur chacun des douze préfixes de huit sorties. Ces répétitions concernent les
**mêmes IDs et le même prompt** ; elles ne constituent pas seize charges
indépendantes. Avec N=M=2, l'unique proposition `[12089, 13]`, après l'index de
sortie 4, diverge dès son premier token. Avec N=M=1, la proposition après
l'index 3 concorde et celle après l'index 4 échoue.

Ces concordances historiques ne sont pas des taux d'acceptation d'un moteur
spéculatif. La vérification doit encore exécuter le vrai GLM, et les produits
multi-colonnes ainsi que les scans KDA peuvent emprunter une autre arithmétique
que le décodage séquentiel. Logits, routes ordonnées, routeurs, KDA, états après
rejet et annulation exigeraient une nouvelle qualification stricte avant toute
mesure de performance.

## Rollback KDA et coût mémoire

Le [runtime déclare GLM5-Next compatible avec le rollback récurrent borné](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/src/llama-arch.cpp).
Cependant,
[`common_params_speculative::need_n_rs_seq()`](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/common/common.h)
réserve automatiquement ces snapshots seulement pour certaines méthodes avec
drafter, dont MTP ; les méthodes ngram renvoient zéro. La conversion des
[paramètres de contexte](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/common/common.cpp)
transmet cette valeur au moteur.

Le [serveur](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/tools/server/server-context.cpp)
prévoit alors un checkpoint pour les contextes ne permettant pas le retrait
partiel nécessaire. Après un rejet, il peut restaurer ce checkpoint puis
rejouer le préfixe accepté. Une simple activation ngram peut donc augmenter
le travail complet du modèle. Aucun patch n'est développé dans cette campagne.

Le journal natif du contrôle V4 observe `n_rs_seq=0`, une cellule physique, des
états F32, et **145,56 Mio de buffer R+S**. L'extrait exact, ses numéros de lignes
et le hash du journal complet sont dans `model-dimensions.json`.
Les [formules des dimensions](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/src/llama-hparams.cpp)
et l'[allocation des états](https://github.com/ggml-org/llama.cpp/blob/3f8af15a76ac8675812e856da45454bafc90f12c/src/llama-memory-recurrent.cpp)
donnent, avec les dimensions sauvegardées :

| Quantité | Valeur déduite |
|---|---:|
| Couches récurrentes | 34 |
| Éléments de convolution R par couche | `3 × (4−1) × 64 × 128` |
| Éléments S par couche | `128 × 128 × 64` |
| R, une cellule F32 | 10 027 008 octets |
| S, une cellule F32 | 142 606 336 octets |
| R+S | **152 633 344 octets, soit 145,5625 Mio** |

Le calcul concorde avec l'affichage arrondi observé. Il **ne mesure pas la
taille sérialisée d'un checkpoint**, ni son temps de copie/restauration.
KV, indexer, métadonnées, logits et activations sont exclus. À paramètres F32
et cellule inchangés, deux snapshots supplémentaires impliqueraient environ
**291,125 Mio de R+S additionnels**, sans présumer l'admission mémoire. La
copie CPU, les nouveaux scans et la perte éventuelle de pages de poids doivent
aussi être mesurés.

Le critère économique d'un futur essai serait conditionnel : coût du draft,
de la vérification, des snapshots et des rejeux inférieur au coût séquentiel
des tokens effectivement validés. Aucun de ces coûts n'est mesuré ici ; aucune
estimation de tokens/seconde n'est produite.

## Préfixes partagés, batching et 600 agents

Le serveur possède déjà la recherche du préfixe commun exact et la restauration
de checkpoints. Pour une mémoire hybride/récurrente, il peut revenir au
recalcul complet lorsqu'aucun état compatible n'est disponible. Activer
`cache_prompt` ne garantit donc pas une réutilisation utile. L'intégration
actuelle désactive ce cache par défaut et indique que les sessions logiques ne
conservent pas chacune un KV physique.

Une réutilisation future doit porter sur les tokens réellement identiques,
le même modèle/runtime et un état KDA/indexer exact, avec isolation des suffixes
et invalidation après annulation. Il ne faut pas modifier l'ordre des
instructions des agents pour fabriquer artificiellement un préfixe commun.
Son bénéfice potentiel concerne le préfill et le TTFT ; il ne prouve aucune
accélération du décodage soutenu.

Allouer un état physique F32 R+S à chacun des 600 agents demanderait environ
**85,29 Gio** pour ces seules catégories, avant KV et poids. Les identités,
instructions et historiques logiques doivent donc rester indépendants d'un
tel stockage physique. Un moteur persistant et un seul slot sont conservés.
Le batching supérieur reste rejeté pour la configuration livrée tant que sa
fidélité et son avantage réel n'ont pas été démontrés. Les anciens essais à
deux slots divergents ne deviennent pas qualifiés par cette analyse.
