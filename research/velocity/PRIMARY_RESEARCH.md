# Sources primaires et portée — 23 octobre 2026

## Modèle déjà installé : priorité de validation système

Le magasin Ollama local contient `qwen3.6:latest`, figé par le digest de manifeste
`07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522`.
Le blob GGUF de poids occupe **23 938 321 664 octets**. Le total affiché par
Ollama, 23 938 333 577 octets, ajoute licence et paramètres. `/api/show` rapporte
Q4_K_M, l'architecture `qwen35moe`, 40 couches, 256 experts, top-8 et un expert
partagé. Les métadonnées comptent 35 951 822 704 paramètres avec la partie vision.

La [configuration Qwen officielle](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
correspond aux dimensions locales : 30 couches Gated DeltaNet et dix couches
d'attention complète, hidden 2048, dimension d'expert 512. La fiche annonce
35 milliards de paramètres texte et trois milliards actifs, sous Apache-2.0.
Le [registre Ollama](https://ollama.com/library/qwen3.6/tags) référence explicitement
`qwen3.6:35b-a3b-q4_K_M` au digest `07d35212591f`; le `latest` distant a changé.
Le test doit utiliser le modèle installé et éviter de remplacer ce checkpoint.

Un test via Ollama démontre une capacité du système local et de son moteur.
Il ne valide pas le backend natif AirLLM. Celui-ci ne contient ni lecteur GGUF
ni décodeur Q4_K/Q6_K; la disposition SSM GGUF signale également des têtes V
réordonnées. Une conversion BF16 complète, proche de 72 Go, dépasse l'espace
libre vérifié et serait inutile pour le premier test. Le téléchargement MLX
ci-dessous reste partiel et suspendu après la découverte du modèle installé.

## Modèle safetensors préparé pour l'adaptateur natif

Le checkpoint officiel [Qwen3-30B-A3B-MLX-6bit](https://huggingface.co/Qwen/Qwen3-30B-A3B-MLX-6bit/tree/b92024737fc38f0bce249aaefb7611dd22c38b58)
est figé à `b92024737fc38f0bce249aaefb7611dd22c38b58`. Ses cinq fichiers de poids
occupent **23 853 636 272 octets**, soit **23,854 Go décimaux / 22,215 Gio**.
Leur payload tensoriel déclaré est de 23 853 477 888 octets. Il dépasse les
17 179 869 184 octets de mémoire physique, mais reste en dessous de l'objectif
indicatif de 28 Go. L'alternative officielle 8 bits occupe 31 486 614 312 octets
(29,324 Gio), avec moins de marge de stockage et davantage d'E/S.

La [fiche du modèle source](https://huggingface.co/Qwen/Qwen3-30B-A3B)
annonce 30,5 milliards de paramètres au total et 3,3 milliards activés.
La configuration vérifiée contient 48 couches, 128 experts par couche, top-8,
32 têtes Q et quatre têtes KV de dimension 128, avec QK RMSNorm. Licence Apache-2.0.
Le checkpoint sélectionné contient des matrices U32 affines 6 bits, groupes de
128, et des échelles/biais BF16. Il quantifie également le routeur et les
embeddings. Ce checkpoint quantifié n'est pas équivalent sans perte au BF16
source. Son évaluation de qualité reste distincte de la validation du lecteur.

L'adaptateur expérimental explicite `ExperimentalQwen3MLXModel` lit les matrices,
les experts et les lignes d'embedding individuellement, garde les matrices
packées dans le cache et utilise le matmul quantifié MLX. Les équations suivent
[Qwen3 MoE dans Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_moe)
et [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_moe.py).
Ses tests CPU comparent des logits et plusieurs pas autorégressifs avec un
Transformers indépendant, après déquantification des mêmes matrices. Ces petites
fixtures aléatoires ne constituent pas une génération entraînée hors mémoire.

## Résultats publiés utiles, sans extrapolation au M5

- [Apple, LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language)
  étudie la réduction des transferts et les lectures contiguës via réutilisation
  des neurones et regroupement lignes/colonnes. Les accélérations publiées
  concernent ses modèles et références naïves. L'enseignement retenu est de
  mesurer les octets réellement transférés et leur granularité; les ratios du
  papier ne prédisent pas le débit de ce moteur.

- [SpecMoEOff, version 2](https://arxiv.org/abs/2508.21706v2) combine spéculation,
  analyse roofline, attention CPU par blocs et réglage automatique. Il annonce
  jusqu'à 2,5 fois le débit de décodage de ses références. La combinaison
  vérification multi-tokens/offload est donc antérieure à ce travail.

- [SP-MoE, version 2](https://arxiv.org/abs/2510.10302v2) utilise la correspondance
  structurelle draft/cible pour précharger les experts, borne la profondeur de
  préchargement et pipeline le calcul avec les transferts. Ses gains TPOT de
  1,07 à 3,5 fois sont des mesures de ses auteurs, pas une validation Metal.

- [FlashMoE, version du 5 octobre 2026](https://arxiv.org/abs/2601.17063v2)
  combine experts sur SSD et politique de cache entraînée à partir de récence et
  fréquence. Les auteurs rapportent jusqu'à 51 % d'amélioration du hit rate et
  2,6 fois d'accélération. Cela justifie une ablation du cache adaptatif, sans
  assimiler les hits à un gain de génération ni à des lectures NAND évitées.

- [DraftExpert](https://arxiv.org/html/2607.24434v1) entraîne un expert draft
  résident par couche et contrôle l'expansion de l'union d'experts. Le papier
  rapporte 1,45 fois le débit moyen sur RTX 4090 avec experts en RAM CPU et sur
  Hexagon avec experts en Flash. Les drafts top-r sans entraînement y sont plus
  lents que l'autorégression une fois la vérification et la réparation payées.
  Son drafter entraîné n'est pas fourni par le checkpoint Qwen sélectionné.

- [SpecFit, 11 octobre 2026](https://arxiv.org/html/2610.15417v1) optimise le
  rapport tokens acceptés/coût de l'union d'experts. Le même Qwen3-30B-A3B y
  progresse de 21,6 % et 7,0 % face à EAGLE-3 avec le même cache sur deux jeux
  limités. **Limite déterminante :** la section 4.6 utilise un atlas de routes
  exactes acquis hors du chronomètre. Un prédicteur déployable reste un travail
  futur, son coût de 0,1–0,2 ms étant estimé. Il serait incorrect de présenter le
  ratio comme un pipeline intégral reproductible sur notre Mac.

- Le [dépôt Flash-MoE original](https://github.com/danveloper/flash-moe)
  annonce 4,36 tokens/s pour Qwen3.5-397B sur M3 Max **48 Go**, avec un SSD annoncé
  à 17,5 Go/s. Son document décrit un routage **K=4** et des experts quantifiés.
  Le matériel, le nombre d'experts et la qualité doivent être contrôlés avant
  toute comparaison; ces chiffres ne certifient ni le M5 16 Go ni un routage
  identique à un autre checkpoint.

## Coût minimal et expérience suivante

Pour une séquence acceptée de longueur A, une borne optimiste est
`T_cycle >= max(B_miss / bandwidth_effective, compute_time) + nonoverlapped_overhead`.
Le débit est `A / (T_draft + T_verify + T_repair + T_sync)`; l'amélioration des
lectures seule ne suffit pas. Pour Qwen, un expert contient trois matrices
2048×768; à 6 bits et groupes 128, son payload source est de 3 686 400 octets
(3,516 Mio). Un token active huit experts par couche dans 48 couches, soit environ 1,318 Gio
d'experts en absence totale de réutilisation. Cette quantité est un calcul sur
les formes, pas un relevé d'E/S. Le cache, les routeurs, l'attention et la tête
de sortie modifient le coût effectif; la bande passante SSD réelle doit être
mesurée séparément.

L'étape suivante est une génération réelle de quatre tokens avec le checkpoint
entier vérifié par SHA256, puis 32/64/128 tokens si réserve et swap restent dans
les limites. Un fichier partiel ou préalloué ne remplit jamais cette condition.
