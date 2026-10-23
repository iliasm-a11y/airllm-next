# Sources primaires — backend GGUF natif AirLLM Next

Consultation : 23 octobre 2026. Ce document distingue les contrats de formats, les choix d'implémentation suggérés et les résultats locaux. Il ne constitue pas une mesure de vitesse ni une validation du checkpoint complet.

## Sources de formats

| Source primaire consultée | Utilisation |
|---|---|
| [Spécification GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | Entête, métadonnées typées, offsets relatifs, alignement et types de tokens |
| [Structures GGML](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) | Tailles et dispositions binaires des blocs quantifiés |
| [Déquantification C GGML](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) | Référence de calcul Q4_K et Q6_K |
| [Déquantification NumPy officielle](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py) | Comparateur indépendant de l'implémentation AirLLM |

Les offsets de tenseurs sont relatifs au début de la zone de données alignée. L'alignement par défaut vaut 32 si le champ correspondant est absent. Les tableaux de dimensions GGML décrivent la dimension contiguë en premier ; leur ordre doit être inversé pour une forme logique NumPy habituelle. Une lecture sélective d'expert doit utiliser sa tranche de données encodées, sans matérialiser le tenseur regroupant tous les experts. Ces propriétés viennent de la [spécification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

### Contrats quantifiés

`Q4_K_M` désigne une recette de quantification du modèle, pas un type de bloc autonome. Le lecteur doit sélectionner le décodeur d'après chaque descripteur de tenseur.

| Type de tenseur | Valeurs par bloc | Octets par bloc | Disposition |
|---|---:|---:|---|
| Q4_K | 256 | 144 | d FP16, dmin FP16, 12 octets de facteurs, 128 octets de valeurs compactées |
| Q6_K | 256 | 210 | 128 octets de bits bas, 64 octets de bits hauts, 16 facteurs int8, d FP16 |
| Q8_0 | 32 | 34 | d FP16, 32 valeurs int8 |

Ces tailles sont définies par les [structures GGML](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h). Q8_0 peut servir aux tests du lecteur ; l'index du checkpoint installé hérité de PR3 n'en contient pas.

Pour un élément `i` d'un bloc Q4_K, les groupes de facteurs comportent 32 valeurs. Le nibble bas d'un octet représente le premier groupe d'une paire, le nibble haut le second. La valeur reconstruite est `(d * facteur_groupe) * entier_4_bits - dmin * minimum_groupe`. Les facteurs et minima sont des entiers de 6 bits répartis dans 12 octets. Référence : `get_scale_min_k4` et `dequantize_row_q4_K` dans [GGML C](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).

Pour Q6_K, chaque moitié de bloc contient 128 valeurs. Les bits bas sont répartis en deux tableaux de 32 octets et deux nibbles ; les bits hauts utilisent successivement les décalages 0, 2, 4 et 6. L'entier reconstruit est recentré en soustrayant 32, puis multiplié par le facteur **signé** du groupe de 16 et l'échelle FP16. Une lecture unsigned des facteurs modifie silencieusement les poids. Référence : `dequantize_row_q6_K` dans [GGML C](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).

Les calculs de référence multiplient les échelles converties en FP32. Le passage intermédiaire par FP16 ou la fusion d'une multiplication et d'une addition dans un kernel peut changer l'arrondi. La validation doit distinguer déquantification exacte des blocs, tolérance des produits matriciels et identité des tokens produits.

## Architecture et conversion Qwen3.6

Le dépôt officiel [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) annonce `qwen3_5_moe`. Son [config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) définit 40 couches, 256 experts, 8 experts actifs, largeur 2048, FFN expert 512 et une attention complète toutes les quatre couches. Les autres couches utilisent GatedDeltaNet. Cette architecture ne peut pas être exécutée par un chemin Qwen3 MoE ne disposant que d'attention classique.

Les sources d'exécution comparables sont [MLX Qwen3.5](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py), [MLX Qwen3Next](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py), [MLX GatedDeltaNet](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py) et [llama.cpp qwen35moe](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp).

Points à préserver, vérifiés dans ces sources :

- Attention complète : projection Q et gate conjointe, séparation **à l'intérieur de chaque tête**, normalisation Q/K, RoPE partielle, attention causale, multiplication par sigmoid(gate), projection de sortie.
- Dimensions officielles : 16 têtes Q, 2 têtes KV, dimension de tête 256, dimension RoPE 64, base RoPE 10000000.
- GatedDeltaNet : convolution causale de largeur 4 sur QKV, SiLU, normalisation L2 de Q/K, état récurrent FP32, gate temporelle, correction delta, normalisation finale et gate SiLU.
- MoE : softmax du routeur, huit experts sélectionnés, probabilités sélectionnées renormalisées ; expert partagé ajouté avec sa gate sigmoid.

La [conversion officielle](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py) applique plusieurs transformations qu'un importeur doit reconnaître :

1. `A_log` devient `-exp(A_log)`.
2. Les normes ajoutent 1, sauf la norme de sortie de GatedDeltaNet.
3. Quand le nombre de têtes V diffère de celui de K, les têtes V sont réordonnées dans QKV, Z, alpha/beta, A/dt, convolution et projection de sortie.
4. La projection de sortie réordonne ses colonnes, les projections entrantes leurs lignes.

Le GGUF installé annonce explicitement des têtes V réordonnées. Pour reconstruire les poids attendus par MLX à partir de ce fichier, il faut inverser cette disposition. Les poids de `nn.RMSNorm` MLX attendent déjà les normes augmentées ; retrancher 1 n'est nécessaire que pour reconstruire les paramètres décalés de Transformers, ou pour une étape de sanitation qui ajouterait 1 à nouveau. La conversion Qwen3.5 normale n'effectue pas la permutation Q/K utilisée par certaines autres architectures.

### Référence des états récurrents

La récurrence [MLX officielle](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py) normalise Q/K avant l'appel du kernel et inclut le facteur de lecture `Dk**-0.5` dans Q. La gate temporelle vaut `exp(-exp(A_log) * softplus(a + dt_bias))`. À chaque position : décroissance de l'état, calcul de la mémoire KV, delta pondéré par sigmoid(b), mise à jour par produit extérieur avec K, puis lecture par Q. Ajouter une seconde normalisation ou un second facteur d'échelle dans le kernel est incorrect.

La version officielle MLX consultée contient elle-même une spécialisation Metal pour Dk=128. Une éventuelle spécialisation AirLLM doit donc être présentée comme une adaptation mesurée, et non comme une nouveauté algorithmique sans antériorité.

## Tokenizer et chat

Sources : [configuration du tokenizer](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/tokenizer_config.json), [vocabulaire llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp), [configuration de génération](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/generation_config.json).

Le pré-tokenizer `qwen35` utilise une regex différente de Qwen2 : les suites de lettres incluent les marques Unicode `\p{M}` et les chiffres sont découpés individuellement. La reconstruction doit appliquer cette séparation, suivie de l'encodage ByteLevel sans ajout d'espace initial, puis les merges BPE GGUF. Il ne faut pas réappliquer la regex ByteLevel par défaut. Aucun BOS n'est ajouté automatiquement ; le nettoyage des espaces à la détokenisation est désactivé.

Les tokens ajoutés doivent être indivisibles même lorsque leur flag `special` vaut false. C'est notamment le cas des balises de pensée et de certains outils/FIM. Les flags de normalisation, rognage d'espaces et restriction à un mot sont false. Le flag spécial réel doit rester distinct de l'appartenance à la liste des tokens ajoutés, particulièrement pour `skip_special_tokens`.

Pour le checkpoint officiel, `im_start` vaut 248045, `im_end` 248046 et `endoftext` 248044. La configuration de génération s'arrête sur **248046 ou 248044**, alors que le tokenizer expose `im_end` comme EOS. Il faut contrôler ces IDs depuis les métadonnées du fichier avant utilisation.

Le template officiel termine son préfixe assistant par une section de pensée vide lorsque `enable_thinking=false`. Un simple ajout du mot `assistant` sans cette section ne reproduit pas le prompt officiel. Les tests doivent couvrir texte français accentué et décomposé, chiffres, espaces, sauts de ligne, balises ajoutées et rendu exact du préfixe assistant.

## MLX et kernels quantifiés

La [documentation de `quantized_matmul`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul) décrit un format MLX à valeurs compactées dans des entiers uint32, avec facteurs et biais par groupe. Il ne s'agit pas directement de la disposition Q4_K/Q6_K GGML. Réinterpréter les octets GGUF comme des poids MLX quantifiés sans réorganisation est incorrect.

Deux chemins restent possibles : déquantifier seulement un bloc ou une tuile bornée, puis multiplier ; ou écrire un produit matriciel Metal lisant directement les blocs GGML. Une comparaison doit compter préparation, copies, allocations temporaires et synchronisations, et pas seulement le temps d'un kernel isolé.

La [documentation Metal MLX](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) recommande de créer le kernel une fois et de le réutiliser. `ensure_row_contiguous=true` peut entraîner une copie implicite. Le code ne doit pas supposer que la mémoire unifiée élimine toutes les copies de conversion ou de mise en forme. Les options de compilation mathématique doivent être vérifiées dans la version effectivement installée, puis la précision comparée à la référence.

## Antériorités et conséquences expérimentales

| Travail primaire | Mécanisme pertinent et limite de transposition |
|---|---|
| [LLM in a flash, 2023](https://arxiv.org/abs/2312.11514) | Réduction des octets et regroupement de lectures flash ; les gains publiés ne sont pas des mesures AirLLM ni M5 |
| [Fiddler, 2024](https://arxiv.org/abs/2402.07033) | Répartition CPU/GPU pour réduire les transferts ; son système avec RAM hôte et GPU distinct diffère de la mémoire unifiée |
| [AdapMoE, 2024](https://arxiv.org/abs/2408.10284) | Gating adaptatif et gestion d'experts ; réduire les experts modifie le calcul cible et ne remplace pas le chemin exact |
| [HOBBIT, 2024](https://arxiv.org/abs/2411.01433) | Précisions mixtes pour réduire les défauts de cache ; modification des poids, donc validation qualité séparée |
| [FlashMoE, version octobre 2026](https://arxiv.org/abs/2601.17063v2) | Experts sur SSD, remplacement combinant récence et fréquence ; antériorité directe aux caches adaptatifs d'experts |
| [Speculative Decoding, 2023](https://arxiv.org/abs/2211.17192) | Proposition peu coûteuse puis vérification cible ; l'accélération dépend du coût total et du nombre accepté |
| [Speculative Sampling, 2023](https://arxiv.org/abs/2302.01318) | Rejet corrigé pour conserver la distribution cible ; la seule égalité greedy ne valide pas un sampler stochastique |

Le code [qwen35moe officiel](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp) dispose désormais d'un chemin MTP à une couche. L'index local hérité de PR3 contient déjà les paramètres `mtp.*`, ce qui ouvre une piste de draft utilisant le même tokenizer sans nouveau checkpoint. Cette présence ne valide ni les paramètres MTP de ce fichier dans AirLLM, ni un gain : il faut d'abord établir un trunk correct, puis mesurer coût du draft, restauration des états récurrents, positions acceptées et vitesse de bout en bout.

## Limites de cette note

### Protocole proposé pour une référence indépendante bornée

La [CLI llama.cpp actuelle](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) expose `--no-repack`, `--load-mode mmap` et `--no-warmup`. L'[API publique](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) expose aussi le chargement sans allocation, la désactivation des buffers supplémentaires de repacking, le nombre de couches GPU et l'accès au dernier vecteur de logits. Ces capacités doivent être contrôlées sur la version locale avant usage ; aucun binaire llama-cli ou llama-server n'a été trouvé dans les chemins de commande inspectés pendant cette recherche.

Un lancement générique sur les 23,94 Go ne constitue pas un comparateur sûr : le [chargeur officiel](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp) permet une prélecture de la totalité du mapping. La variante de référence proposée doit désactiver explicitement cette prélecture, le repacking, l'offload GPU, le verrouillage RAM et le warmup. Elle conserve les kernels et les équations amont inchangés. Préflight sans allocation, contexte 64, batch 1, MTP désactivé, puis un seul vecteur de logits sous le superviseur AirLLM. L'initialisation et les allocations de travail doivent être inspectées avant autorisation du calcul. La sortie de 248320 logits FP32 représente environ 0,95 Mio.

Le mode lazy de l'API amont ne garantit pas un budget mémoire global : il vise seulement certains tenseurs marqués par l'architecture. Ni mmap ni un arrêt a posteriori ne remplacent une admission avant allocation. Si ce comparateur ne respecte pas les seuils, conserver le résultat négatif et utiliser uniquement les preuves plus limitées disponibles : blocs officiels GGML, produits par tuiles et modèle miniature indépendant. Aucune de ces preuves locales limitées ne doit être appelée validation complète de Qwen3.6.

Les statistiques du fichier installé mentionnées ici proviennent de l'index conservé dans PR3 ; elles doivent être régénérées sur le fichier réellement testé. Les sources amont ont été relues en direct. Aucun poids complet n'a été déquantifié, aucun modèle massif n'a été téléchargé et aucun benchmark lourd n'a été lancé pour rédiger cette note. Les sources et antériorités ne prouvent aucun seuil de vitesse sur le MacBook Air M5.
