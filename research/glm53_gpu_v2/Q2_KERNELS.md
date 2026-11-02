# Kernels Q2_K compatibles avec le CPU ARM — expérience M5

Neuf variantes exécutables sont ajoutées dans ce dossier, sans changer les
poids, la quantification GGUF ni les fichiers de la PR précédente. Les
activations Q8_K temporaires reproduisent celles du CPU. Une préparation
explicite permet de réutiliser ces activations entre projections; les huit
experts sélectionnés sont traités dans un même lancement, dans l'ordre fourni.
Aucune sélection ni somme pondérée des experts n'est réalisée ici.

## Variantes réalisées

- `serial32` : référence SIMD32 par ligne, ordre ARM des blocs conservé.
- `rowtile2`, `rowtile4` : deux/quatre lignes par groupe SIMD; activations
  chargées une fois puis réutilisées entre lignes.
- `blocks4`, `blocks8` : sous-groupes de huit/quatre lanes calculant plusieurs
  blocs en parallèle; les résultats entiers sont ramenés dans l'ordre ARM.
- `twopass` : produits entiers indépendants par bloc, puis chaîne FMA ordonnée.
  L'intermédiaire comporte deux tableaux int32, pas une matrice de poids FP32.
- `cachedbytes`, `cachedbytes_tile2`, `cachedbytes_tile4` : chaque lane conserve
  les deux octets quantifiés de ses deux demi-blocs et en extrait les quatre
  plans de bits, plutôt que de les recharger à chaque plan; versions une,
  deux et quatre lignes.

Les additions entières ne débordent pas: la borne conservatrice du terme
principal par bloc est 256 × 15 × 3 × 128 = 1 474 560. Changer leur arbre de
réduction conserve donc exactement l'entier. Chaque variante applique ensuite
les mêmes deux FMA FP32 par bloc, minimum puis produit principal, et parcourt
les blocs dans l'ordre du CPU ARM. Il ne s'agit pas d'une promesse de parité
avec un autre backend CPU, ni avec un checkpoint non quantifié.

## Vérification sur des données réelles

Le rapport principal est
`evidence/q2-layer40-real-eight-experts-suite.json`.
Il utilise **l'activation réelle `ffn_norm-40`, étape 1**, capturée lors du
nouveau diagnostic CPU, et les routes ordonnées correspondantes :
`[141,183,116,112,154,156,21,98]`.
Les huit banques gate du bloc 40 sont lues sans conversion depuis le GGUF
Q2_K_S local, soit 22 020 096 octets de poids. Les empreintes de ces octets,
des captures, de l'index, des bibliothèques CPU, du timer et du code Metal
généré sont conservées dans le rapport. La provenance est vérifiée avant
l'import GPU; la capture brute n'est pas publiée par ce benchmark.

Les neuf variantes produisent **16 384 sorties finies, bit à bit identiques**
à l'oracle natif ARM, dans MLX et dans l'exécution Metal autonome. Les
activations Q8_K et leurs échelles correspondent également au CPU. Les
adaptateurs CPU1/CPU10 sont recoupés avec des appels indépendants au dot natif
sur seize lignes de chaque expert.

Huit tests couvrent les dimensions, les limites mémoire, les entrées invalides,
les activations partagées ou distinctes, les fins de tuiles et les largeurs
256, 768 et 4096. Ces tests exécutent notamment chaque variante avec un et
quatre groupes SIMD; les comparaisons sont bit à bit.

## Mesures et résultat

Médianes de sept répétitions, ordre aléatoire enregistré. Les buffers de poids
et les activations sont préparés et réutilisés. Les mesures de chaque colonne
ont leur propre périmètre :

| Variante | MLX, mur synchronisé | Metal, temps GPU réel | Metal, création/encodage/commit/attente |
|---|---:|---:|---:|
| serial32 | 0,737 ms | 0,551 ms | 0,748 ms |
| rowtile2 | 0,691 ms | 0,512 ms | 0,733 ms |
| rowtile4 | 0,684 ms | 0,488 ms | 0,680 ms |
| blocks4 | 1,182 ms | 0,976 ms | 1,157 ms |
| blocks8 | 1,170 ms | 0,946 ms | 1,124 ms |
| twopass | 1,513 ms | 1,184 ms | 1,383 ms |
| cachedbytes | 0,805 ms | 0,571 ms | 0,778 ms |
| cachedbytes_tile2 | 0,815 ms | 0,513 ms | 0,708 ms |
| cachedbytes_tile4 | 0,671 ms | 0,488 ms | 0,685 ms |

Références natives préparées : **CPU1 1,505 ms; CPU10 0,664 ms**. Le CPU10
emploie un pool persistant et un seul appel pour les huit matrices; aucun
appel Python par ligne n'entre dans ce temps. L'ordonnanceur reste différent
de celui du graphe GGML de production.

La meilleure variante MLX réduit le temps médian de la référence GPU de ce
même essai d'environ **9 %**. La meilleure médiane GPU réelle baisse d'environ
**11,5 %**. **Le CPU10 reste légèrement plus rapide sur le temps mur MLX** :
0,664 contre 0,671 ms, avec des distributions qui se chevauchent. Ces petits
écarts et cet effectif ne prouvent pas une amélioration générale. Il n'y a
**aucun gain de décodage GLM complet démontré par ce composant**.

La première préparation observée coûte 46,0 ms pour métadonnées/lecture des
experts, 3,11 ms pour l'import des poids et 3,39 ms pour import/quantification
de l'activation. Elle peut inclure initialisation/JIT. Ces coûts sont séparés
des répétitions préparées; le coût de quantification d'une nouvelle activation
par token doit être ajouté lors d'une intégration. L'état du cache de fichiers
macOS n'est pas contrôlé et aucune mesure Fanxiang n'est attribuée ici.

La charge explicite maximale admise est **46 563 840 octets**, sous 64 Mio.
Le pic MLX observé est 24,25 Mo et le pic RSS du processus Python 128,19 Mo.
Le timer Metal utilise un processus séparé et des buffers partagés bornés;
la phase Metal commence après libération des poids MLX. Ces comptes de payload
ne sont pas un plafond global de RSS. Les garde-fous du backend complet restent
nécessaires. Les 600 identités et le slot physique ne sont pas modifiés.

## Temps GPU et reproductibilité

`q2_metal_timer.mm` compile les mêmes corps de kernels que MLX. Il mesure
**`MTLCommandBuffer.GPUEndTime - GPUStartTime`** et refuse de substituer une
horloge CPU si ces timestamps manquent. Les neuf variantes partagent le même
processus, les mêmes buffers et une file Metal. Quatre cycles de chauffe sont
exclus; l'ordre de chaque cycle mesuré est aléatoire et conservé. Le temps mur
apparié inclut création/encodage du command buffer, commit et attente.
Les chemins de compilation et d'intégration MLX/Metal autonome restent
 distincts; leurs temps ne doivent pas être mélangés en un ratio.

Le premier pilote `q2-layer40-real-eight-experts.json` reste une preuve
exploratoire à six variantes, exécutées dans des processus Metal séparés.
Le rapport `...-suite.json` le remplace pour les comparaisons de kernels.

Compilation des adaptateurs, depuis la racine du dépôt :

```sh
clang++ -std=c++17 -O3 -dynamiclib research/glm53_gpu_v2/q2_native_matrix.cpp \
  -L/path/to/pinned-runtime/bin -lggml-cpu -Wl,-rpath,/path/to/pinned-runtime/bin \
  -o /path/to/libq2-group.dylib
clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation \
  research/glm53_gpu_v2/q2_metal_timer.mm -o /path/to/q2-metal-timer
```

`python -m research.glm53_gpu_v2.q2_benchmark --help` expose le lancement borné.
Fournir `--activation` et `--capture-metadata` pour vérifier la provenance
réelle; leur absence produit explicitement une activation synthétique.
`AIRLLM_GGML_LIBRARIES=/path/to/pinned-runtime/bin python -m pytest -q
research/glm53_gpu_v2/test_q2_kernels.py` active les tests indépendants CPU/GPU.
