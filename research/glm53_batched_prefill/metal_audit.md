# Préfill Metal multi-token : contrat et limites

Le composant `metal_q8.metal` traite un vrai produit `[M,K] x [K,N]`, avec
N=8/16/32/64/128 tokens dans **un seul dispatch de produit**. Un threadgroup
traite huit lignes de poids et huit colonnes de tokens. À chaque bloc de32,
les 256 octets quantifiés des poids sont chargés dans une tuile partagée puis
consommés par huit tokens. Une seconde tuile réutilise les activations entre
huit lignes de sortie. Les poids restent Q8_0 originaux ; seuls les états
temporaires d'activation sont quantifiés comme la référence ARM. Le produit
ne déquantifie pas une copie complète de la matrice.

Ce composant n'est **pas installé dans le moteur GLM**. La preuve sur le
dispatcher CPU10 explique pourquoi une extension directe serait prématurée.

## Arithmétique CPU réelle

Sources épinglées llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`,
plus patches déjà publiés PR17-21 :

- `ggml/src/ggml-cpu/arch/arm/quants.c:1153` : Q8_0 `nrc=2` additionne un dot
  entier de32 termes puis une FMA par bloc. `nrc=1` maintient plusieurs
  accumulateurs pairs/impairs et une autre réduction. Le shader mono-token
  PR19 reproduit le second contrat ; l'appliquer au préfill change les valeurs.
- `ggml/src/ggml-cpu/ggml-cpu.c:1349` et `:1447` : le dispatcher essaie
  LLAMAFILE avant le fallback vec_dot. Le build mesuré garde LLAMAFILE et BLAS
  Apple activés. Une égalité contre `vec_dot` seul ne prouve pas l'égalité
  contre ce dispatcher.
- `ggml/src/ggml-cpu/ggml-cpu.c:1517` : le fallback choisit nrc1 si les
  dimensions ou les partitions de travail ont une longueur impaire. Le choix
  dépend donc aussi de la forme et des threads.
- `ggml/src/ggml-cpu/tiled/tiled.cpp:1180` et `:1202-1244` : une voie tuilée
  K-quants existe, mais `:597-604` la désactive sur ARM : elle dépend ici
  d'AVX/AVX2/AVX512. Elle ne constitue donc pas le préfill tuilé du M5.
  Sur ce Mac, les boucles de fallback dense/MoE dans `ggml-cpu.c` conservent
  de petits blocs de16 et les dots natifs ARM. L'architecture conditionnelle
  doit être vérifiée avant d'attribuer une optimisation au CPU mesuré.
- `research/glm53_gpu_v2/patches/llama-q8-selective.patch:68` conserve
  explicitement `x->ne[1] == 1` pour le placement Metal compatible existant.
  Les produits multi-colonnes restent CPU. Changer seulement un paramètre de
  microbatch n'active donc pas un moteur de préfill GPU compatible.

`metal_cpu_graph.cpp` construit un vrai graphe GGML Q8_0/F32 et appelle le
dispatcher CPU10 de la bibliothèque mesurée. Il complète l'oracle natif
`ggml_vec_dot_q8_0_q8_0(...,nrc=2)` ; les deux comparaisons restent séparées.

## Résultats de composant M5

Quinze fixtures synthétiques, M=10/14/32, K=32/96/4096 et
N=8/16/32/64/128, comparent **13888 valeurs** aux fonctions natives.
La quantification des activations et toutes les sorties sont bit à bit
contre nrc2. Contre le **graphe CPU10**, les lots64/128 divergent pour les
matrices K96 et K4096. Pour M32/K4096 :1791 puis3638 sorties divergent,
erreur absolue maximale9,1552734375e-5 puis1,373291015625e-4. Les lots8/16/32
de ces seules petites fixtures sont identiques au graphe. Aucun résultat
de ce composant ne qualifie les logits, routes ou états du modèle complet.

L'empreinte RSS maximale du processus Python lors du second run est
50544640 octets. Le timer impose64Mio de buffers Metal au maximum ; les
fixtures ont une charge connue bien inférieure. Le premier shader a été
refusé à la compilation pour un cast demi-précision, puis corrigé. Les deux
premiers essais de l'oracle graphe ont épuisé son arène locale ; le helper
utilise maintenant le workspace planifié et borné séparément. Ces échecs
étaient des erreurs de fixtures sans chargement du modèle.

Les mesures initiales précèdent l'ajout des contrôles internes secteur,
pression1, RSS8,5Gio et croissance swap32Mio au runner. Les gardes du moteur
GLM sont indépendantes et ne sont pas modifiées. Les rapports initiaux ne
permettent pas de reconstruire une télémétrie mémoire manquante.

## Préfill et états restant CPU

`src/models/glm5-next.cpp:213` projette plusieurs tokens par `ggml_mul_mat`.
La convolution causale concatène la fenêtre précédente (`:219`), écrit les
fenêtres récurrentes (`:224-233`), puis applique `SSM_CONV` et SILU (`:237`).
Ce graphe comporte donc des dépendances temporelles et des écritures ; un
produit matriciel plus rapide ne suffit pas à qualifier la restauration.
Les indexeurs k-pool reçoivent leurs entrées par le cache hybride (`:251`).
Les poids MoE sont des tenseurs à experts (`:169-171`) ; leur sélection doit
précéder le streaming. Le composant actuel ne traite ni Q2_K ni MoE,
normalisation, mHC, KDA/DSA, attention et écritures d'état. Aucun temps de ces
opérations, d'attente SSD ou de synchronisation du modèle n'est inventé à
partir des timestamps du petit composant.

Le batch8 historique est une configuration prudente et codée explicitement
dans le probe (`research/glm53_gpu_v3/native_probe.cpp:260`). Ce choix ne
constitue pas une mesure de l'optimum. Les campagnes admissibles doivent
mesurer les prompts complets et conserver des témoins contemporains.

## Reproduction isolée

Le timer réutilisé est `research/glm53_gpu_v2/q2_metal_timer.mm` : il compile
avec `MTLMathModeSafe`, rend les erreurs Metal explicites et ne remplace
jamais un timestamp GPU absent par une durée CPU. Le runner écrit formes,
nombre de dispatchs, charge connue et empreintes du shader et de l'oracle.

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
  research/glm53_gpu_v2/q2_metal_timer.mm -o /absolute/work/metal-timer
xcrun clang++ -std=c++17 -O2 -dynamiclib \
  research/glm53_batched_prefill/metal_cpu_graph.cpp \
  -I /absolute/llama-source/ggml/include -L /absolute/runtime/bin \
  -lggml-base -lggml-cpu -Wl,-rpath,/absolute/runtime/bin \
  -o /absolute/work/metal-cpu-graph.dylib
python -m research.glm53_batched_prefill.metal_probe \
  --libraries /absolute/runtime/bin --timer /absolute/work/metal-timer \
  --graph-oracle /absolute/work/metal-cpu-graph.dylib \
  --work /absolute/work/probe --output /absolute/work/report.json
```

Ajouter `--gguf /absolute/model-shard.gguf --tensor output.weight --rows 32`
pour lire uniquement les32 premières lignes originales de cette matrice,
sous budget8Mio. Les activations restent synthétiques et sont identifiées
comme telles. La commande retourne3 lorsque la comparaison du graphe échoue.
Les timestamps décrivent des poids préparés en mémoire ; ils n'établissent
ni débit de préfill GLM ni gain sur SSD. Le composant ne modifie aucune option
de RONIN et aucune voie GPU n'est retenue en production.
