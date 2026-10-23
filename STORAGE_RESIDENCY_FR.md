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
Le groupe ciblé de cache/sidecar passe : **61 tests**. La vérification finale
du dépôt compte **916 tests réussis, 1 ignoré et 11 sous-tests réussis**
(`evidence-multitoken/tests-final-e2-f.log`).

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

## Union et résidence sur les traces définitives

Le replay lit les routes réelles de `batch-cpu-128.json`, campagne primaire
sur deux textes naturels prédéclarés, avec les mêmes positions pour le témoin
token par token et le regroupement par union. Les anciens essais continuant
après EOS sont conservés à part comme exploratoires.

| Positions | Union réelle moyenne par couche | Source de l'union, MiB/position sans cache |
|---:|---:|---:|
| 1 | 8,000 | 817,60 |
| 2 | 12,646 | 576,05 |
| 4 | 20,271 | 425,28 |
| 8 | 25,958 | 263,83 |
| 16 | 29,188 | 146,44 |
| 32 | 30,667 | 76,55 |

L'union réelle est plus petite que l'attente indépendante pour ces deux
textes. Les tailles ci-dessus excluent les lignes d'embedding. La différence
avec les octets logiques enregistrés par le runtime est exactement celle de
ces petites lectures : 2 048 octets par identifiant distinct dans chaque
bloc, soit 59 392 octets pour les blocs de 32 positions étudiés.

Le tableau suivant rejoue les 32 positions **token par token**, depuis un
cache vide, à enveloppe FP32 strictement égale pour chaque politique. Les
nombres sont des MiB sources manquants par position, moyennés sur les deux
textes. Aucun temps de rechargement n'est fourni au replay ; les modes
adaptatifs utilisent donc leur coût égal de secours.

| Politique | 128 MiB | 256 MiB | 512 MiB |
|---|---:|---:|---:|
| LRU | 817,60 | 817,60 | 817,60 |
| SLRU | 817,60 | 817,60 | 817,60 |
| LFU | 799,00 | 722,49 | 705,69 |
| Adaptive | 773,26 | 761,65 | 658,91 |
| Shared | 776,60 | 771,60 | 582,85 |
| Expert | 806,93 | 790,68 | 762,45 |
| Hybrid | 773,26 | 761,65 | 658,91 |
| Batch-aware | 816,10 | 773,23 | 602,54 |

`shared` réduit ici les octets manquants de 11,54 % par rapport à `adaptive`
avec 512 MiB, mais les augmente de 1,31 % avec 256 MiB. `hybrid` produit les
mêmes octets que `adaptive` sur ces traces. Une préférence de classe ne
constitue donc pas un meilleur choix universel.

Pour un unique bloc union de 32 positions, tous les modes chargent les mêmes
76,55 MiB par position depuis le cache vide : chaque poids de l'union n'est
demandé qu'une fois dans ce bloc. Le bénéfice vient alors du regroupement,
avant toute différence de résidence entre blocs. Le replay comprend 12
unions observées et 576 combinaisons de budget/politique/exécution ; ce sont
des analyses d'octets, sans nouvelle inférence ni mesure de SSD.

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
qualité du modèle. Leur résultat ne suffit pas à valider une génération
complète ; celle-ci fait l'objet des mesures distinctes ci-dessous.

## Conversion complète et génération quantifiée

La conversion complète a produit 1 536 matrices Q4 pour `expert4`, puis
1 536 matrices Q4 et 96 matrices linéaires Q8 pour `mixed`, groupe affine 64.
Les originaux, routeurs, normalisations et embeddings sont conservés.

| Ensemble de poids utilisés | Payload effectif |
|---|---:|
| Original BF16 | 2 669 256 704 octets |
| Experts Q4, autres poids originaux | 1 008 312 320 octets |
| Experts Q4 et matrices admissibles Q8 | 942 252 032 octets |

Le « payload effectif » soustrait les poids remplacés et ajoute leur version
compacte. **Les overlays locaux conservent le checkpoint original** : ils
ajoutent physiquement 755 324 928 octets de fichiers pour `expert4` ou
840 281 472 pour `mixed`. Le dossier existant n'a donc pas été réduit à la
taille du payload effectif. Les manifestes et tous les fichiers sidecars
ont été vérifiés par SHA-256 avant et après les essais définitifs, sans
changement constaté.

La génération réelle utilise les noyaux MLX quantifiés, avec accumulation
FP32, budgets identiques, arrêt au premier EOS et comptage des tokens
effectivement émis. La confirmation Metal, sans le profilage linéaire
supplémentaire, porte sur quatre prompts et deux répétitions, soit 114
nouveaux tokens par variante. Le contrôle CPU porte sur deux prompts,
une répétition, soit 25 tokens par variante.

| Variante | Tokens/s agrégés Metal | Tokens/s agrégés CPU |
|---|---:|---:|
| Original | 4,228 | 4,913 |
| Experts Q4 | 4,728 | 4,114 |
| Mixte Q4/Q8 | 4,983 | 3,955 |

Ces débits incluent le préremplissage. Les charges CPU et Metal diffèrent :
les colonnes ne sont pas une comparaison directe de dispositifs. Les
variantes quantifiées produisent un contenu différent de l'original sur
6 générations Metal sur 8 ; les nombres observés ne prouvent donc pas
une accélération à résultat identique. Q4/Q8 améliore ici le débit agrégé
Metal, mais le dégrade dans le petit contrôle CPU. Les contrôles de budgets
et les audits finaux passent dans les deux confirmations.

## Qualité sur 512 cibles humaines tenues à l'écart de l'expérience

Une mesure indépendante utilise quatre fenêtres du [test WikiText-2 fourni
par le dépôt officiel PyTorch](https://github.com/pytorch/examples/blob/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/word_language_model/data/wikitext-2/test.txt), au commit
`d5678bc8ac0cdd79dbd5e44d4130271018bcec4e`. SHA-256 du fichier de
1 256 449 octets :
`d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`.
L'attribution WikiText/Salesforce et contributeurs Wikipédia, ainsi que les
licences indiquées par la [fiche officielle Salesforce](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
(CC-BY-SA-3.0 et GFDL), figurent
dans la provenance. Le texte reste dans `work/`, ignoré par Git.

Les fenêtres sont fixées aux milieux des quatre quartiles du flux tokenisé,
avant toute inférence. Chaque fenêtre possède 32 tokens humains de contexte
non notés et 128 cibles humaines ; aucun texte généré par le modèle ne sert
de référence. Le calcul de perplexité exponentie la NLL moyenne pondérée par
le nombre de cibles.

| Variante | NLL moyenne | PPL du sous-ensemble | Variation PPL | Accord top-1 avec l'original | Exactitude top-1 sur cibles humaines |
|---|---:|---:|---:|---:|---:|
| Original | 2,59409 | 13,38435 | — | 100 % | 51,758 % |
| Experts Q4 | 2,64953 | 14,14743 | +5,701 % | 87,109 % | 52,148 % |
| Mixte Q4/Q8 | 2,64526 | 14,08709 | +5,250 % | 86,719 % | 52,344 % |

La KL moyenne original→variante vaut 0,06989 pour Q4 et 0,06974 pour le
mixte. La légère hausse d'exactitude top-1 sur ce très petit ensemble
coexiste avec la hausse de NLL/PPL et ne démontre pas une meilleure qualité
générale.
Les 512 cibles, le contexte limité et la version prétraitée du corpus ne
constituent pas une PPL WikiText-2 standard ni une validation générale.
Le test est tenu à l'écart de cette expérience ; un chevauchement avec le
préentraînement de Granite reste inconnu.

Dans cette mesure, le pic KV atteint 25 690 112 octets et la réservation
conservatrice maximale des activations d'experts vaut 3 198 976 octets.
Les budgets, positions KV, fichiers originaux, tokenizer et identités des
sidecars sont vérifiés. Aucune mesure ci-dessus ne valide un modèle géant,
un SSD externe physique ou une qualité générale de corpus.

## Reproductibilité

- `benchmarks/benchmark_storage_multitoken.py` : matrices entraînées,
  compression, temps de reconstruction, probes de sortie.
- `benchmarks/analyze_storage_union_multitoken.py` : inventaire, théorie
  et relecture sans allocation de poids des traces réelles de routage.
- `evidence-multitoken/storage-micro.json` : mesures détaillées et hashes
  SHA-256 des matrices échantillonnées.
- `evidence-multitoken/storage-summary.json` : agrégats et limites.
- `evidence-multitoken/storage-union-theory.json` : inventaire et hypothèses.
- `evidence-multitoken/storage-replay-cpu128.json` : unions définitives et
  replays à 128/256/512 MiB.
- `evidence-multitoken/quantized-confirmation-metal.json` et
  `quantized-cpu.json` : générations quantifiées avec budgets vérifiés et
  audits finaux des sidecars.
- `evidence-multitoken/corpus-quality-metal.json` : NLL/PPL, KL et top-1
  sur les quatre fenêtres humaines prédéclarées ; provenance et token IDs.
- `evidence-multitoken/array-audit.json` : SHA-256 des 13 NPZ présents,
  correspondance des 9 hashes déclarés et 234 vérifications numériques des
  tableaux batch, y compris les archives exploratoires ; aucune erreur.
- `evidence-multitoken/draft-provenance.json` : identité du draft SmolLM2
  local, vérifiée contre l'API officielle au commit épinglé, sans
  téléchargement des fichiers du modèle.

Le replay utilise les politiques de production avec un budget FP32 égal,
les tailles BF16 exactes et les routes réellement exportées. Il n'invente
pas de coûts de rechargement : sans timings enregistrés, les politiques
adaptatives emploient leur hypothèse de coût égal. Il n'est donc pas un
benchmark matériel supplémentaire.
