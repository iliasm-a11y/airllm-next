# Notes de lecture des preuves finales

Ces notes ont été préparées pendant les campagnes chronométrées, par lecture
des seules métadonnées et des rapports déjà clos. Aucun test, build, modèle,
kernel GPU ou accès aux poids n'a été lancé pour cette relecture. Les résultats
des campagnes en cours sont volontairement absents.

## Lecture des mêmes 1 Gio de matrices denses

Les trois sondes lisent les mêmes plages dans le même ordre, en 166 appels,
avec le même SHA-256 du flux obtenu. Le payload de lecture temporaire est
8 Mio. Le Fanxiang est absent ; les fichiers sont internes.

| Mode observé | Octets logiques | Lectures OS du PID | Lectures du contrôleur global | Temps cumulé dans read | Temps total avec hash et observations |
|---|---:|---:|---:|---:|---:|
| `F_NOCACHE` demandé | 1 073 741 824 | 1 074 589 696 | 1 087 467 520 | 0,185338 s | 0,888573 s |
| Normal, première sonde | 1 073 741 824 | 1 086 775 296 | 1 094 934 528 | 0,129104 s | 0,843793 s |
| Normal, répétition immédiate | 1 073 741 824 | 270 336 | 6 594 560 | 0,045976 s | 0,750389 s |

Les lectures PID sont une comptabilité OS, incluant éventuellement readahead
et autres I/O du processus. Les octets contrôleur sont système et incluent
d'autres processus. Les premières sondes montrent des lectures OS et une
activité de bloc proches du volume demandé ; la répétition montre leur forte
baisse sur les mêmes octets. Aucun taux exact de hits, volume NAND ou trafic
physique attribué au seul PID n'est disponible.

Les quotients logiques par temps d'appels valent respectivement 5,79, 8,32 et
23,35 Go/s. Ce ne sont **pas** des vitesses SSD démontrées, ni des plafonds
matériels à réutiliser dans les bornes de débit. Le temps total inclut beaucoup
de hash et d'observation ; son inverse n'est pas davantage une mesure pure du
disque. Aucun des trois essais ne prouve un cache physiquement froid. Le succès
du conseil `F_NOCACHE` sur un descripteur ne démontre pas non plus le
comportement du pager mmap du moteur complet.

Sources closes : `work/v3/storage-dense-{nocache-v2,normal,repeat}.json`.
Les trois sondes se terminent sans croissance globale du swap et sans
modification des statistiques du fichier source.

## Première utilisation et répétitions d'une vraie matrice Q8

La matrice de 35 651 584 octets est `blk.0.attn_q.weight`. Son activation vient
de la couche 40 et correspond en largeur, sans reproduire une opération
originale de couche 0. Le produit CPU10 prend 15,783 ms avec 35 672 064 octets
de lectures OS et 2 177 pageins du PID ; les répétitions prennent 0,322,
0,349 et 0,314 ms, sans lecture OS ni pagein attribué au PID. Les 8 192
sorties restent identiques en bits. Voir [preuve de composant](DENSE_COLD_WARM.md).

L'écart inclut résidence, caches matériels, première allocation éventuelle du
workspace et ordonnancement. Il ne mesure pas exactement un temps SSD et ne
prouve pas que 98 % du décodage complet soit de l'I/O. Une seule matrice chaude
ne démontre pas la résidence simultanée des 9,444 Go denses.

## Compression sans perte : résultat et décision

Le pilote Zstandard niveau 3 utilise huit échantillons totalisant 50 200 576
octets, tous reconstruits exactement. La décompression chaude mesurée inclut
la création d'un décompresseur et du résultat, sans lecture disque dans la
fenêtre chronométrée. Il ne s'agit ni d'une représentation complète du modèle,
ni d'une mesure de décompression intégrée au graphe.

| Échantillons | Taille comprimée / originale | Décompression originale mesurée |
|---|---:|---:|
| Trois plages Q8_0 | 0,91829–0,92855 | 1,544–1,564 Go/s |
| Deux plages BF16 | 0,77823–0,78255 | 1,368–1,431 Go/s |
| Gate/up/down Q2_K d'un expert | 0,97732–0,97831 | 1,550–1,555 Go/s |

La matrice Q2 compressée économise seulement environ 2,2 % dans ces
échantillons. Les gains BF16 sont plus grands, sur une fraction minoritaire
des poids. Ces échantillons ne permettent aucun ratio certifié du checkpoint
entier ni aucune estimation de la compression réelle de macOS.

À titre de modèle séquentiel à 1 Go/s, le temps par Go original serait
`ratio / 1 Go/s + 1 / débit_de_décompression`. Pour battre une lecture brute,
le décompresseur devrait dépasser environ 12,2–14,0 Go/s sur les Q8 testés,
4,5–4,6 Go/s sur BF16 et 44–46 Go/s sur les Q2 testés. Les débits du pilote
sont bien inférieurs. Un pipeline parfaitement superposé changerait ce
calcul mais demanderait buffers et ressources CPU supplémentaires ; aucun
gain de ce pipeline n'a été mesuré. L'intégration de ce format Zstandard est
donc rejetée dans cette campagne, sans prétendre exclure tous les codecs ou
algorithmes parallèles possibles.

Source : `work/v3/lossless-samples.json`. Aucun fichier GGUF n'a été réécrit.

## Cache partiel : priorité selon le coût, pas seulement selon les octets

Le chiffre « 1 Gio donnerait +9,63 % » appartient uniquement au modèle où chaque
octet évité économise le même coût d'I/O. Ce n'est pas une borne universelle.
Un petit poids très coûteux par octet pourrait valoir davantage.

Le contrôle ci-dessous utilise les 684 MUL_MAT denses de chacune des trois
étapes décodées de l'ancien `cpu-pages4`, et le multiensemble réconcilié dans
`evidence/dense-inventory.json`. Les tailles sont toutes multiples de 65 536
octets. Les noms de sortie, l'ordre et les tailles des 684 sites sont stables
entre ces trois étapes. Ils ne constituent toutefois pas une identité GGUF
directement journalisée par cette ancienne sonde.

On optimise un sac à dos de matrices entières dans lequel **tout le temps de
l'opération disparaît** lorsqu'une matrice est retenue : calcul compris,
chargement du cache gratuit, aucun coût d'éviction ni supplément mémoire.
Cette hypothèse surestime fortement un vrai cache. Le temps de référence est
6,280404787 s/étape. Le gain affiché vaut `T / (T − temps_supprimé) − 1`.

| Budget | Modèle par coût moyen du bucket : temps supprimé / gain | Choix fixe des sites de sortie, coût moyen sur 3 étapes | Borne plus optimiste : oracle fractionnaire différent à chaque étape |
|---|---:|---:|---:|
| 256 Mio | 0,281981 s / +4,70 % | 0,340863 s / +5,74 % | 0,372128 s / +6,30 % |
| 512 Mio | 0,544548 s / +9,49 % | 0,597357 s / +10,51 % | 0,629435 s / +11,14 % |
| 1 024 Mio | 0,961352 s / +18,07 % | 0,981690 s / +18,53 % | 1,018193 s / +19,35 % |

Le premier modèle utilise le coût moyen par taille et respecte le nombre de
matrices de chaque taille. Le second aligne les sites stables, avec leur
temps moyen, sans attribuer de nom GGUF inconnu. Le troisième trie séparément
les 684 appels de chaque étape par temps/octet et autorise une fraction de
la dernière matrice : sa relaxation contient tous les choix entiers fixes.
La moyenne des trois optima est donc une borne optimiste des seules économies
directes de ces opérations, dans cet échantillon. Elle ne borne pas les effets
indirects du cache sur les défauts de pages d'autres opérations ou sur la
pression globale, ni les performances d'autres trajectoires.

Les buckets les plus coûteux par octet sont les poids de 512 Kio
(BF16/F32, 1,287 ms/Mo), 2 Mio (BF16/F32, 1,054 ms/Mo), puis 16 Mio
(MLA K/V BF16, 0,978 ms/Mo). Les grandes matrices Q8 KDA sont vers
0,401 ms/Mo. Un cache fondé uniquement sur la taille ignorerait ces différences,
mais ces temps comprennent le calcul : ils ne sont pas des attentes disque.

La sélection exacte du **modèle par moyenne de bucket**, qui ne choisit pas
un nom GGUF à l'intérieur des buckets mixtes, est :

| Budget | Nombre de matrices par taille en octets | Types possibles des buckets | Payload total |
|---|---|---|---:|
| 256 Mio | 44 × 524 288 ; 77 × 2 097 152 ; 5 × 16 777 216 | BF16/F32 pour les deux petits buckets ; BF16 pour 16 Mio | 268 435 456 o, 126 matrices |
| 512 Mio | 44 × 524 288 ; 77 × 2 097 152 ; 21 × 16 777 216 | Mêmes familles | 536 870 912 o, 142 matrices |
| 1 Gio | 45 × 524 288 ; 77 × 1 048 576 ; 90 × 1 572 864 ; 79 × 2 097 152 ; 11 × 4 194 304 ; 23 × 4 718 592 ; 11 × 12 582 912 ; 22 × 16 777 216 | Petits BF16/F32, HC F32, routeurs F32 et MLA BF16 | 1 073 741 824 o, 358 matrices |

Le meilleur choix fixe des sites, autorisant les différences de coût au sein
d'une même taille, retient respectivement 104, 191 et 344 sites pour
268 435 456, 536 870 912 et 1 073 610 752 octets. Ce sont des sites de
l'ancienne trace ; un zip par taille/position avec les nouveaux noms sources
serait une proposition d'identification, pas une preuve d'identité de pointeur
dans la capture ancienne.

Pour +20 % sur ce temps moyen, il faudrait supprimer 1,046734 s/étape.
Même la relaxation à 1 Gio reste légèrement en dessous, avec 1,018193 s.
Ce résultat justifie une faible priorité d'un gros cache privé dans cette
campagne contrainte ; il ne prouve pas une impossibilité universelle de +20 %.
Le coût de résidence et les duplications d'un cache réel réduiraient encore
son intérêt direct. Les captures originales et la formule permettent de
reproduire ces calculs sans accéder au checkpoint.

## Portée de l'union et du diagramme

La [preuve d'union](evidence/cpu-stream16-source-union-requests.json) ferme
l'identité des 684 matrices et 126 banques pour le graphe B16 capturé : noms,
codecs, formes, strides, alias, offsets, bornes et routes effectives. Elle
prouve 12 218 990 592 octets distincts d'opérandes ciblés par étape décodée.
Elle ne trace pas les instructions de chargement, ne mesure pas les caches
CPU/GPU ou NAND et n'établit pas une borne informationnelle pour tout autre
algorithme. Les normes, embeddings, états et copies intermédiaires ne font
pas partie de ce total.

Le diagramme de [faisabilité](FEASIBILITY-CLOSED.md) partitionne un ancien
diagnostic sérialisé : 4,350 s denses, 1,235 s experts, 0,696 s autres par
étape. Ces groupes sont intercalés dans les couches. Il ne faut pas transformer
leurs parts de temps en pourcentages purs d'I/O, de CPU ou de GPU, ni ajouter
les temps CPU multithreads au temps mur.

Les comparaisons d'états comptent 3 792 événements CPY/SET_ROWS observés, dont
1 088 événements vides. Certains premiers rapports utilisent `compared_bytes`
pour une portée de stockage de 2 672 730 112 octets ; le rapport plus récent
distingue 2 626 682 880 octets logiques. Le nombre de 2,67 Go ne doit pas être
présenté comme autant de valeurs logiques d'état différentes validées.

## Capacité : conclusion conditionnelle

Le cache des huit experts précédents par couche réutilise 47,1514 % des
demandes sur les quatorze transitions mesurées. Avec tous les poids denses
retenus, son scénario I/O idéal à 1 Go/s donne environ 0,682 tok/s, avant
calcul et autres coûts. Une génération à 0,5 tok/s n'est donc pas exclue par
la seule quantité de RAM nominale.

Mais ces rétentions représentent 12,219 Go bruts, soit 11,380 Gio, avant
système, compresseur, runtime et staging. Elles dépassent déjà le plafond
RSS du processus de 8,5 Gio. Au snapshot avant le pilote codec, wired vaut 2,813 Go
et le stockage physique du compresseur 3,939 Go. Un scénario conservateur
qui les conserve et alloue un nouveau cache privé brut de 12,219 Go
demanderait environ 18,971 Go, au-delà des 17,180 Go physiques, avant les
autres allocations. Ce scénario ne doit pas être confondu avec la résidence
du cache de fichiers existant, une redistribution future de la mémoire ou
la fermeture d'autres applications.

La conclusion autorisée porte sur la charge et les architectures évaluées :
une grande rétention privée brute n'est pas admissible ; aucun débit rapide
ne se déduit des composants. Les mesures finales de débit, stabilité,
réinitialisation et EOS doivent compléter ce diagnostic. Une conclusion
universelle « tout M5 16 Gio est physiquement incapable de 0,5 tok/s » serait
plus forte que les preuves disponibles.
