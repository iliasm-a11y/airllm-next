# Poids denses : réutilisation et priorité d'optimisation

L'[inventaire réconcilié](evidence/dense-inventory.json) contient **684 matrices fixes distinctes, 9 444 458 496 octets par étape de décodage**. Les trois étapes décodées de `cpu-pages4` comptent chacune exactement 684 MUL_MAT et le même multiensemble de tailles, pour ce total exact. L'identité de chaque source n'était pas journalisée par cette ancienne sonde : le rapprochement des noms repose sur le graphe propriétaire et l'inventaire GGUF, pas sur une déduplication de pointeurs mesurés.

Le calcul inclut les 22 projections MLA K/V BF16 tridimensionnelles. Il exclut la table d'embeddings entière, dont une seule ligne est sélectionnée au décodage, et les 11 `indexer_compressor_ape.weight`, utilisés par ADD (`glm5-next.cpp`, `indexer_kpool_ape`). Les deux projections HC par couche portent le même nom de sortie de diagnostic `hc_mixes`, mais sont **deux poids distincts** ; les dédupliquer par ce nom serait erroné.

| Famille fixe | Octets par token |
|---|---:|
| KDA Q/K/V, 34 couches | 3 636 461 568 |
| Projections de sortie d'attention, 45 couches | 1 996 488 704 |
| Trois matrices des experts partagés, 42 couches | 1 123 024 896 |
| Tête de sortie Q8 | 674 037 760 |
| Neuf matrices FFN des trois couches denses initiales | 481 296 384 |
| MLA K/V BF16, 22 matrices | 369 098 752 |

Il n'y a donc pas de réutilisation dense intra-token démontrée qui supprimerait plusieurs lectures du même poids. La réutilisation principale se trouve **entre tokens**. Garder toutes les matrices denses simultanément représenterait environ 8,80 GiB, avant états, activations et runtime ; un pool qui force ces octets à rester résidents dépasserait déjà la garde RSS de 8,5 GiB. Le cache de fichiers macOS, non explicitement retenu, relève d'un autre périmètre et ne peut pas être présumé équivalent à ce pool.

Les octets MoE sélectionnés sont 2 774 532 096 par token dans cette architecture top-8, soit 12 218 990 592 octets de MUL_MAT/MUL_MAT_ID au total. Dans le seul modèle conditionnel où chaque octet évité économise le même temps d'I/O et où le calcul ne coûte rien, retenir 1 GiB réduit ce temps de 8,79 %, donc augmenterait le débit de 9,63 %. Atteindre +20 % demanderait alors au moins 2 036 498 432 octets de rétention utile. Ce sont des plafonds de ce modèle linéaire, pas une preuve que les coûts physiques par octet sont uniformes, ni un résultat de cache réel. Les duplications, la mémoire système comprimée et les copies de staging peuvent en réduire le bénéfice.

Dans la trace sérialisée `cpu-pages4`, les MUL_MAT denses prennent en moyenne **4,350 s** par token décodé et comptent environ **9,442 Go de lectures OS par PID**, proches des 9,444 Go logiques sélectionnés. Le plus grand groupe temporel est celui des 136 matrices de 35 651 584 octets, les quatre projections KDA de chaque couche : environ **1,944 s/token** sur ces trois étapes. Suivent notamment les matrices d'experts partagés de 8 912 896 octets, environ 0,482 s/token, et les projections MLA K/V de 16 777 216 octets, environ 0,361 s/token. Le fichier JSON conserve les groupes bruts par taille.

Ce rapprochement soutient une investigation des mouvements de données. Il ne transforme pas les compteurs OS en lectures NAND : readahead et cache de fichiers interviennent, les pageins ne sont pas tous attribuables à une lecture SSD, et le temps de MUL_MAT comprend calcul, défauts de pages et ordonnancement. Le profil ne sépare pas encore ces attentes du calcul réellement chaud.

La piste retenue est donc [l'anticipation mmap d'une matrice](DENSE_LOOKAHEAD.md), désactivée par défaut, avec un seul worker et une fenêtre initiale de 32 MiB. Elle vise le moment et la granularité des lectures, sans créer une copie anonyme ni réclamer un gros pool de poids persistants. Le conseil actuel PR19 arrive au début de la multiplication du poids courant ; il ne prouve aucun chevauchement avec le calcul précédent. La nouvelle piste doit démontrer des appels effectifs, une réduction de durée globale et une stabilité mémoire, avec le même volume de travail et une nouvelle preuve de fidélité.

Le [diagnostic de matrice complète](DENSE_COLD_WARM.md) est préparé pour confronter première utilisation et répétitions du même produit Q8 CPU10, avec lectures OS/pageins mesurés. Il sert à quantifier le coût chaud accessible, sans prétendre qu'une première phase est froide ou qu'une répétition est libre d'I/O. Aucun gain de modèle ni objectif de 20 % n'est démontré par l'inventaire, le préflight des sources ou la fixture d'anticipation.
