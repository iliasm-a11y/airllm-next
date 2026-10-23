# Stockage et résidence des poids : mesures et limites

Les politiques expérimentales `shared`, `expert`, `hybrid` et `batch-aware`
réutilisent exactement l'enveloppe de cache existante. Elles n'ajoutent ni
copie des poids ni allocation de cache supplémentaire. Le défaut reste
`slru`. Une éventuelle partition compacte reste soustraite du cache décodé.

- `shared` privilégie attention, normalisations, routeurs et tuiles de sortie.
- `expert` privilégie les matrices d'experts.
- `hybrid` arbitre selon réutilisation et coût de rechargement observés ; un
  bénéfice strictement égal favorise le poids partagé.
- `batch-aware` privilégie temporairement l'union réellement routée ; ses
  compteurs de multiplicité proviennent des positions observées. La priorité
  disparaît à la fin du groupe, y compris après une erreur.

La classification distingue les identités d'experts `(nom, id)` ou `nom#id`
des tuiles de sortie `('output', nom, début, fin)`. Ces dernières restent des
poids partagés. Les tests couvrent classement, évictions, baisse de budget,
vues NumPy conservant un gros tampon, métadonnées bornées, déclaration de
lot atomique, enveloppe décodée/compacte commune et sélection par la CLI.
Le groupe ciblé de tests passe : **61 tests**, avec les contrôles de cache
adaptatif et d'intégrité du sidecar.

## Inventaire réel du Granite officiel

Le checkpoint étudié possède 24 couches, 32 experts par couche et 8 experts
sélectionnés par position. Une paire de matrices d'expert occupe 3 MiB en
BF16, soit 6 MiB après décodage FP32.

| Groupe | Source BF16 | Décodage FP32 |
|---|---:|---:|
| Poids partagés hors embedding et sortie | 152 668 160 octets | 305 336 320 octets |
| Sortie liée à l'embedding | 100 669 440 octets | 201 338 880 octets |
| Tous les experts | 2 415 919 104 octets | 4 831 838 208 octets |

Les seuls poids partagés hors embedding/sortie occupent environ **291,2 MiB
en FP32**. Ils ne tiennent donc pas entièrement dans les enveloppes de
192 ou 256 MiB de la campagne. Une préférence pour le partagé ne garantit
pas sa résidence complète ni un gain de vitesse.

Sous l'hypothèse analytique de sélections top-8 uniformes indépendantes,
l'union attendue vaut `32 × (1 − (3/4)^T)`. Les corrélations des routes réelles
peuvent modifier fortement cette union.

| Positions T | Union attendue par couche | Source sans cache, MiB/position | Experts seuls avec partagé parfaitement résident, MiB/position |
|---:|---:|---:|---:|
| 1 | 8,000 | 817,60 | 576,00 |
| 2 | 14,000 | 624,80 | 504,00 |
| 4 | 21,875 | 454,15 | 393,75 |
| 8 | 28,796 | 289,37 | 259,17 |
| 16 | 31,679 | 157,66 | 142,56 |
| 32 | 31,997 | 79,54 | 71,99 |

Ces nombres décrivent des octets logiques et excluent les petites lectures
des lignes d'embedding. La colonne de résidence parfaite est une limite
analytique ; elle ne décrit pas le cache réel testé. Aucun débit de SSD ni
temps de génération ne découle de ce tableau.

## Compression et quantification : échantillons entraînés

Le microbenchmark lit 18 matrices complètes de vrais experts, dans les
couches 0, 12 et 23, experts 0, 7 et 31, et 3 matrices d'attention partagées.
Il ne convertit pas le modèle entier. Les alias officiels Granite
`input_linear.weight` et `output_linear.weight` sont désormais reconnus par
`convert_experts`. La conversion mixte conserve les routeurs originaux.

| Représentation des 21 matrices | Octets de payload | Réduction |
|---|---:|---:|
| Source BF16 | 34 603 008 | — |
| Compression lossless existante + index | 24 989 530 | 27,8 % |
| Affine 4 bits partout | 10 813 440 | 68,75 % |
| Experts 4 bits, attention 8 bits | 12 386 304 | 64,2 % |

Les reconstructions lossless sont identiques aux octets sources. Les
headers JSON et manifestes sont exclus de ce tableau, sauf l'index lossless
explicitement compté.

Pour les 18 matrices d'experts, la somme des médianes de reconstruction
lossless vaut **45,38 ms**, contre **1,57 ms** pour convertir directement le
BF16 original vers FP32. La reconstruction lossless ne comprend pas la
conversion BF16 vers FP32, qui reste nécessaire ensuite. Le temps est
mesuré en mémoire chaude sur CPU, sans lecture de fichier. Une réduction
de stockage ne prouve donc pas une accélération locale.

Les probes de précision appliquent chaque paire d'expert à 32 activations
gaussiennes déterministes, avant pondération par le routeur et résiduel.
L'erreur L2 relative de sortie MLP est :

| Précision des experts | Médiane sur 9 experts | Maximum |
|---|---:|---:|
| 4 bits | 15,399 % | 16,220 % |
| 8 bits | 0,888 % | 0,932 % |

Ces probes ne sont ni des états cachés de génération ni une mesure de
qualité du modèle. Aucune génération complète quantifiée, aucun noyau MLX
quantifié, aucun SSD externe et aucun modèle géant ne sont validés par ces
mesures. L'échantillon 4 bits ne justifie pas à lui seul une conversion
intégrale.

## Reproductibilité

- `benchmarks/benchmark_storage_multitoken.py` : matrices entraînées,
  compression, temps de reconstruction, probes de sortie.
- `benchmarks/analyze_storage_union_multitoken.py` : inventaire, théorie
  et relecture sans allocation de poids des traces réelles de routage.
- `evidence-multitoken/storage-micro.json` : mesures détaillées et hashes
  SHA-256 des matrices échantillonnées.
- `evidence-multitoken/storage-summary.json` : agrégats et limites.
- `evidence-multitoken/storage-union-theory.json` : inventaire et hypothèses.
- `evidence-multitoken/draft-provenance.json` : identité du draft SmolLM2
  local, vérifiée contre l'API officielle au commit épinglé, sans
  téléchargement des fichiers du modèle.

Le replay utilise les politiques de production avec un budget FP32 égal,
les tailles BF16 exactes et les routes réellement exportées. Il n'invente
pas de coûts de rechargement : sans timings enregistrés, les politiques
adaptatives emploient leur hypothèse de coût égal. Il n'est donc pas un
benchmark matériel supplémentaire.
