# Audit GLM-5.3-Flash Q2_K_S, 2 novembre 2026

Audit en lecture seule des poids et des dépôts. Aucune inférence, compilation ou modification du moteur effectuée par cet audit. Les conclusions sur les coûts sont des hypothèses de profilage, pas des mesures de latence.

## Identité locale et distante

La lecture des seuls en-têtes GGUF confirme `general.architecture=glm5-next`, 45 blocs, 4096 dimensions cachées, vocabulaire 154880 et contexte théorique 1048576. Le premier fragment fait 9430112 octets et contient les métadonnées/tokenizer, sans tenseur. Le second contient 1383 tenseurs et fait 110017922752 octets. Total des fichiers : 110027352864 octets. Ceci explique qu'un test de taille minimale supposant deux gros fragments serait incorrect.

L'API officielle Hugging Face a répondu pour la révision exacte `119034bc3d9d76eb41f91508e6f9fb19fde42672`; les tailles et SHA-256 annoncés correspondent au profil de vérification antérieur présent localement. L'audit n'a pas recalculé les 110 Go de SHA-256 : il ne s'agit pas d'une nouvelle validation complète de tous les octets. Réponse distante conservée dans `la réponse API officielle de révision` et lecture des en-têtes dans `../sources/gguf-headers.json`. Lecture reproductible : les seules métadonnées et descriptions de tenseurs GGUF ont été parcourues; aucun contenu de poids n’est inclus dans cet inventaire.

Source distante : [révision du modèle GGUF](https://huggingface.co/ggml-org/GLM-5.3-Flash-GGUF/tree/119034bc3d9d76eb41f91508e6f9fb19fde42672), [API de révision](https://huggingface.co/api/models/ggml-org/GLM-5.3-Flash-GGUF/revision/119034bc3d9d76eb41f91508e6f9fb19fde42672?blobs=true).

## Architecture à préserver

- 34 couches Kimi Delta Attention (KDA), 64 têtes de dimension 128, convolution courte de largeur 4, portes par canal bornées à -5, état récurrent matriciel. Ce n'est pas une attention Qwen interchangeable.
- 11 couches DeepSeek Sparse Attention (indices 3, 7, ..., 43), MLA sans RoPE dans les projections principales (`rope.dimension_count=0`), rang KV 512, rang Q 1536, dimensions QK/V originales 256.
- Indexeur 32 têtes de dimension 128, pools de 4 tokens, top-2048, conservation de la queue incomplète; les clés poolées complètes sont déjà mises à jour de manière incrémentale dans llama.cpp.
- Connexions résiduelles mHC : quatre flux, 20 itérations Sinkhorn, epsilon 1e-6.
- Trois premiers blocs denses; 42 blocs MoE ensuite, 288 experts routables par bloc, top-8 par token et un expert partagé. Routeur sigmoid, biais de correction des sélections, normalisation des poids top-k puis facteur 2.5; SwiGLU plafonné à 10. L'ordre et les règles exactes du routeur doivent être conservés.
- Le fichier `config.json` mentionne une couche NextN/MTP, mais le GGUF cible ne contient que les blocs 0 à 44 et aucun tenseur NextN/MTP. Une auto-spéculation MTP ne peut donc pas être activée à partir de ces seuls fragments. La présence du support MTP dans le moteur ne prouve pas la présence de ses poids.
- Aucun projecteur vision ni tenseur vision n'est présent dans les fragments inspectés. Le chemin validable ici est la génération texte; il ne faut pas annoncer une compatibilité multimodale complète à partir du `config.json`.

## Quantification réellement présente

Les octets ci-dessous sont les plages de données entre offsets de tenseurs (alignement inclus), indépendamment des métadonnées. L'alignement ne change pas le diagnostic de coût.

| Type | Nombre | Octets |
|---|---:|---:|
| Q2_K | 126 | 99883155456 |
| Q8_0 | 306 | 8953004032 |
| BF16 | 225 | 796917760 |
| F32 | 726 | 384758784 |

Les 126 Q2_K correspondent exactement aux trois banques gate/up/down des 42 blocs MoE. Q2_K utilise des blocs de 256 valeurs pour 84 octets, soit 2.625 bits/poids avec paramètres de déquantification. Chaque banque d'un bloc représente 792723456 octets pour 288 experts; un expert a 2752512 octets par projection, 8257536 octets pour les trois projections. Huit experts sélectionnés représentent donc 66060288 octets/bloc/token, et 2774532096 octets pour les 42 blocs, avant réutilisation, alignement de pages et cache.

Le reste des poids représente 10134680576 octets. Ce chiffre n'est pas une lecture SSD obligatoire à chaque token : embeddings partiels, réutilisation en cache, poids résidents et ordonnancement changent les lectures physiques. Il démontre toutefois pourquoi « top-8 sur 288 donc modèle actif minuscule » est une hypothèse insuffisante. Les matrices KDA/attention majoritairement Q8_0 et les experts partagés sont substantiels. Les octets d'experts sélectionnés sont un volume logique calculé sur les métadonnées, pas un débit mesuré.

Décomposition plus précise : embedding d'entrée 674037760 octets au total mais seulement 4352 octets Q8_0 par ligne de token (hors granulation des pages); tête de sortie 674037760 octets pour chaque projection complète du vocabulaire; trois FFN denses ensemble 481345536 octets. Tous les poids non experts hors matrice d'embedding d'entrée totalisent 9460642816 octets. Les trois banques top-8 des 42 blocs plus ce dernier total donnent 12235174912 octets de plages de poids potentiellement touchées par étape de décodage, avant réutilisation en cache. Ce total n'est ni une mesure de trafic mémoire ni un minimum physique démontré.

## Backend GGUF existant

Le dépôt llama.cpp local est au commit `3f8af15a76ac8675812e856da45454bafc90f12c`. Il contient déjà `src/models/glm5-next.cpp`, la mémoire hybride récurrente/MLA/indexeur et les opérations nécessaires au graphe texte de ce GGUF. Il n'est donc pas nécessaire de convertir ce modèle en runtime Qwen pour obtenir une intégration native correcte.

Le chemin CPU prend en charge les types et l'architecture. Le test antérieur montre seulement une génération courte fonctionnelle; il ne démontre ni une parité Transformers BF16 ni la conformité à toutes les longueurs de contexte.

Metal implémente déjà Q2_K/Q8_0/BF16/F32, `MUL_MAT_ID`, KDA/Gated Delta Net, convolution SSM, Lightning Indexer, masque top-k et les trois opérations mHC fusionnées. Les conditions matérielles et de formes sont vérifiées au démarrage. Pour KDA, la dimension 128 est divisible par 32; pour l'indexeur la dimension 128 est prévue. L'existence du code est une preuve de prise en charge potentielle, pas une preuve qu'une exécution donnée est restée entièrement sur GPU.

Les chemins fusionnés GDN, indexeur, mHC et masque top-k sont déjà activés/probés par défaut dans `llama-context.cpp`. Il faut lire les traces avant d'attribuer leur gain à une nouvelle optimisation.

## GPU, cache et mémoire

`--cpu-moe` conserve les banques d'experts en mémoire hôte/mmap; `--moe-cache-mib N` crée un cache GPU SLRU de taille bornée et conserve les octets quantifiés d'origine. Il regroupe les banques de disposition identique et publie hits/misses/octets chargés. Il ne modifie pas les experts choisis. Le cache exige un backend GPU et des blocs assignés à ce GPU. Son budget minimal doit contenir les experts d'un token; le code rejette les budgets insuffisants. Pour les microbatches de plus de 32 tokens, le chemin évite d'évincer le cache de décodage.

Le cache effectue une synchronisation GPU après récupération des IDs du routeur, puis les copies des experts manquants. Cette frontière constitue un coût à mesurer. Sur mémoire unifiée, un cache ajoute des copies et une réserve résidente, même sans bus PCIe. Un meilleur taux de hits ne garantit pas un meilleur temps/token.

Une configuration GPU intégrale avec experts CPU garderait encore environ 10 Go de poids non experts, auxquels s'ajoutent état, graphes, cache d'experts et applications. Sur 16 Go, c'est risqué. Commencer par `-ngl 4` ou `-ngl 8`, `--cpu-moe`, un petit cache (par exemple 128/256 Mio) et une surveillance réelle est plus raisonnable qu'un `-ngl 99` aveugle. Ces valeurs sont des points d'expérience, pas une configuration optimale mesurée. Le CPU mmap reste indispensable comme référence.

Ne pas utiliser mlock sur 110 Go; ne pas confondre mapping virtuel et résidence physique. Le mmap de llama.cpp applique déjà des conseils de lecture et dispose d'un helper de prefetch. Il n'existe pas ici de prefetch générique des futurs experts MoE : les usages actuels de `llama_prefetch_rows` concernent des embeddings par couche. Un prefetch exact après routage pourrait regrouper les plages gate/up/down choisies sans approximation, mais sa valeur reste à mesurer.

### Audit précis du chargement macOS

`llama-model.cpp:1916` appelle `init_mappings(true, ...)`. `llama-model-loader.cpp:1442` traduit ce booléen en préchargement `SIZE_MAX`; `llama-mmap.cpp:513` envoie ensuite `POSIX_MADV_WILLNEED` sur le complément des plages lazy, donc presque tout le fragment de 110 Go ici. Ce conseil peut déclencher du travail et de l'I/O avant le calcul; le code seul n'attribue pas les 40 s mesurées historiquement à cet appel. Le test historique inclut déjà `--no-warmup` et `--no-repack`, donc ces deux options ne sont pas de nouvelles pistes d'amélioration par rapport à sa référence.

`--lazy-mode on` ne suffit pas : seuls les tenseurs créés avec `TENSOR_READ_LAZY` entrent dans ces plages. Aucun tenseur GLM n'est marqué ainsi; les usages de ce drapeau dans le dépôt concernent les embeddings par couche de Gemma/Qwen/DeepSeek41. Tous les tenseurs experts de ce GGUF font moins de 4 Gio chacun, mais même modifier ce seuil ne suffirait pas sans le drapeau.

Un patch privé minimal, désactivé par défaut, est fourni sous `../patches/llama-mmap-demand.patch` dans le nouveau dépôt AirLLM. `AIRLLM_MMAP_DEMAND=1` supprime seulement le WILLNEED initial et journalise l'activation. Il n'annule pas tout read-ahead système. Il préserve le mapping et les opérations; sa performance doit être comparée sur le même binaire. Vérification d'application du patch réussie sur le commit épinglé; aucune compilation ni mesure de ce patch n'a été effectuée par cet audit.

Une expérience ultérieure pourrait marquer seulement les banques `ffn_*_exps.weight` en `MADV_RANDOM` tout en préchargeant les petites matrices denses. Les offsets/tailles sont disponibles dans `weights_map` au chargement et dans l'inventaire JSON. Il faut ajouter des plages de conseil sans changer la classification `is_lazy` ni les buffers des tenseurs, et mesurer les requêtes disque. Le caractère séquentiel à l'intérieur d'un expert de 2.75 Mo peut rendre RANDOM défavorable; il ne faut pas le conserver sur la seule intuition. [Apple décrit madvise comme un conseil de stratégie de pagination](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/madvise.2.html), pas une garantie de comportement ou de débit.

`MAP_RESILIENT_CODESIGN` n'est pas un réglage de read-ahead. Le `sys/mman.h` Apple précise que ces mappings peuvent fournir un contenu incorrect au lieu d'une exception. Aucune preuve de gain n'est trouvée; ce drapeau ne répond pas à l'exigence de fidélité et n'est pas inclus. `MAP_RESILIENT_MEDIA` peut substituer des zéros à un stockage indisponible : il est également exclu. Le fcntl `F_RDAHEAD` existe sur le SDK local mais ne remplace pas une expérience mesurée et ciblée; il n'est pas activé.

## MLX, MPS et M5

[MLX-VLM possède un modèle glm5_next](https://raw.githubusercontent.com/Blaizzy/mlx-vlm/main/mlx_vlm/models/glm5_next/README.md), avec KDA/DSA/MoE/mHC et des optimisations documentées (projections, indexeur incrémental, traitement des courts contextes). Cela fournit des idées à comparer. Son chargement de modèle Safetensors ne constitue pas un lecteur direct de ces banques GGUF Q2_K; remplacer les poids par une autre quantification MLX ne satisferait pas la contrainte de fidélité du modèle cible.

[Apple décrit les Neural Accelerators du M5 via Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/). Ils ciblent surtout les opérations denses à forte intensité de calcul, notamment le préfill. Q2_K est un format GGML avec échelles/minima personnalisés; le support d'un entier 2 bits natif ne rend pas automatiquement son layout compatible. Une déquantification/représentation adaptée doit conserver les valeurs et mesurer son coût. Le llama.cpp local contient déjà TensorOps conditionnellement au SDK et au test de compilation, activé par défaut sur M5 quand disponible. Les traces `has_tensor` et le chemin effectif des kernels doivent être conservés; aucune accélération M5 ne peut être déduite de la seule marque du GPU.

MPS fournit des matrices performantes mais ne résout pas à lui seul la lecture GGUF, le routage, la mémoire KDA, mHC ou le streaming des experts. Reconvertir tout en F16 exploserait mémoire/stockage. La voie la plus directe est l'intégration du backend GGUF déjà compatible et des expériences CPU/hybrides bornées.

## Trois goulots à tester en priorité

1. Défauts de page et stockage sur les poids non résidents : comparer lectures OS par processus, fautes majeures, temps mur et pages disponibles; distinguer relecture chaude des poids de froid relatif au processus. Ne pas appeler « froid physique » un simple redémarrage.
2. Calcul/déquantification des grandes projections Q8_0 et des Q2_K experts : comparer CPU 4/6/10 threads et offload partiel, avec tokens/logits identiques ou critères explicites de tolérance. Le petit routeur F32 peut déclencher beaucoup d'I/O sans être lui-même coûteux.
3. Synchronisations/copies et coût récurrent KDA/mHC : mesurer un petit cache MoE GPU, puis sa combinaison avec offload partiel; lire les statistiques. Les fusions existantes ne doivent pas être créditées comme nouvelles.

Il s'agit d'hypothèses classées par structure, pas d'un profil quantifié. Les kernels personnalisés ne sont justifiés qu'après attribution des temps. Une optimisation d'indexeur court contexte est envisageable (la sélection couvre tous les tokens), mais doit maintenir les caches pour le franchissement du seuil et les séquences multiples, et comparer les logits; le gain pourrait rester négligeable face au stockage.

## Critères de fidélité proposés

Référence : même GGUF, même révision native, CPU mmap, prompt tokenisé identiquement, seed fixe, température 0, absence de cache de réponse et de spéculation. Enregistrer IDs générés et logits complets sur une séquence forcée pour distinguer dérive numérique et amplification après divergence d'un token. Critère strict CPU : égalité des IDs et, lorsque seule l'intégration change, identité des logits. Metal peut nécessiter une tolérance explicite (max absolu/relatif, RMSE et accord top-1), approuvée comme tolérance plutôt que vendue comme bit-à-bit. Un test de 4 tokens n'est pas un décodage soutenu, et une paire ne qualifie pas la variabilité ni les objectifs de 1/3/5/10 token/s.

## Complément : projection de sortie seule sur Metal

Examen statique du même commit, sans compilation ni inférence supplémentaire par l'auditeur. La projection `output.weight` est Q8_0, de forme `[4096,154880]`, et représente 674037760 octets. Son transfert initial et sa résidence représentent donc une expérience beaucoup plus petite que l'offload de plusieurs blocs. Cela ne prouve pas un gain : la synchronisation, l'initialisation Metal, les autres lectures et les erreurs numériques doivent être mesurées.

Le nom du buffer CLI correct de ce moteur est **`MTL0`**, et non `Metal`. `GGML_METAL_NAME` vaut `MTL` dans `ggml-metal.cpp`, et le buffer partagé unifié ajoute l'indice zéro. `MTL0_Mapped` est un buffer interne créé lors du mapping; `parse_tensor_buffer_overrides` ne propose que les buffers par défaut des devices, donc il ne faut pas l'utiliser comme valeur CLI. Si `GGML_METAL_SHARED_BUFFERS_DISABLE` a été hérité, le buffer par défaut peut devenir `MTL0_Private`; cet environnement doit être contrôlé et les noms réellement exposés doivent être vérifiés dans les traces du pilote.

Variante stricte qui change seulement cette matrice, à ajouter aux options communes de la référence :

```sh
--device MTL0 --gpu-layers 0 --override-tensor '^output\.weight$=MTL0' --no-op-offload
```

`--device none` doit être retiré ou remplacé : le contexte initialise ses backends GPU depuis `model.devices`. Allouer un poids Metal tout en excluant ce backend du contexte est incohérent. `--no-op-offload` peut être conservé : dans `ggml-backend.cpp`, l'ordonnanceur choisit d'abord le backend du buffer d'un poids; ce drapeau désactive l'offload opportuniste des poids restés CPU, sans interdire l'opération attachée à un poids déjà Metal. La projection Q8_0 avec entrée F32 est supportée. Aucun cache MoE GPU n'est nécessaire pour cette variante.

Le mapping Metal par défaut reprend les plages exactes des tenseurs de son contexte via `get_mapping_ranges`, qui exclut les plages appartenant aux autres contextes. La présence de `output_norm.weight` à la fin du grand fragment n'oblige donc pas à mapper la totalité de l'intervalle de 110 Go entre tête et norme. Le backend demande la résidence de ses allocations avec les residency sets lorsqu'ils sont disponibles. Cette observation motive une diminution possible des relectures de la tête; elle ne remplace pas la mesure des lectures OS et de la mémoire réellement utilisée.

Une variante plus simple est **`--device MTL0 --gpu-layers 1 --no-op-offload`** : dans cette révision précise, `i_gpu_start = n_layer_all + 1 - n_gpu_layers` et la sortie est le groupe numéro `n_layer_all`. Ainsi, `-ngl 1` déplace le groupe sortie (tête et norme), aucun des 45 blocs. Cette convention ne doit pas être remplacée par une hypothèse héritée d'une ancienne version de llama.cpp. Cette variante n'est pas identique à l'override strict, car elle peut aussi déplacer la normalisation finale.

Correction des estimations d'admission : pour `ngl = 0`, poids GPU estimés = 0. Pour `1 <= ngl <= 46`, poids non experts GPU = 674054144 octets (tête + norme) + la somme des poids non experts des blocs d'indice `>= 46-ngl`. Les blocs 0 à 2 font chacun 313459712 octets; les blocs DSA 3,7,...,43 font 193376640; les autres font 184486016. Exemples : `ngl1=674054144`, `ngl6=1605374848`, `ngl8=1983237504`, `ngl46=9460642816` octets, avant états, alignement de pages, graphes et cache. La table de sommes des N derniers blocs reste une description correcte des blocs, mais ne correspond pas directement à l'argument `ngl` de cette révision.

Plan de validation : garder le même prompt, seed, contexte, threads et réglages de prefetch que la meilleure référence CPU; comparer séparément override strict et `ngl1`. Archiver le placement effectif des buffers/ops, la latence et les lectures de la requête hors chargement. Sur une trajectoire de tokens imposée par la référence, les routeurs et KDA CPU en amont doivent être comparés séparément des logits de tête Metal; une divergence naturelle de token en génération libre change ensuite l'entrée du modèle et peut modifier les routes sans démontrer un défaut du routeur. Aucun réglage de production n'est modifié avant ce pilote et ses critères numériques.
