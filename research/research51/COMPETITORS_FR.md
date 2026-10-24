# Audit concurrents — AirLLM Next 5.1

Date de consultation : 24 octobre 2026. Statut : inspection de sources primaires publiques et calculs analytiques. Aucun modèle lancé, aucun poids téléchargé, aucune mesure locale d'inférence. Les performances ci-dessous sont des résultats publiés par leurs auteurs, non reproduits sur le MacBook Air. Les fichiers de preuve consultés sont conservés dans `work/competitor-src/` avec leur identité Git.

## Identités et licences vérifiées

| Projet | Révision figée consultée | Licence constatée | Pertinence réelle |
|---|---|---|---|
| [Colibrì](https://github.com/JustVugg/colibri/tree/e0db083cdc9e563e0766cc6fc661ef23c32298e0) | `e0db083cdc9e563e0766cc6fc661ef23c32298e0`, 17-10-2026 | Apache-2.0, NOTICE et avis tiers | Streaming SSD C, Metal, cache et prélecture |
| [Pulsar SSD](https://github.com/giannisanni/pulsar/tree/18f990094e1091ee2ef3078f20be418f8c162f03) | `18f990094e1091ee2ef3078f20be418f8c162f03`, 01-09-2026 | Texte MIT; l'API GitHub classe `Other`, donc le texte LICENSE fait foi | Rust/CUDA, Linux io_uring; moteur macOS explicitement stub |
| [Pulsar Metal](https://github.com/loopai-hq/pulsar/tree/61f571f7abf95f1fca12df78ac575eca3fd6cdeb) | `61f571f7abf95f1fca12df78ac575eca3fd6cdeb`, 14-10-2026 | Apache-2.0 | Fork Splash 1.3.0, Qwen3.8-27B/Qwen3.6-35B-A3B, spéculation/kernels Metal |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/71255fefc092b0344f2acddb891388814ae961be) | `71255fefc092b0344f2acddb891388814ae961be`, 24-10-2026 | MIT | GGUF, Metal, chargement différé et spéculation |
| [PulsarMLX](https://github.com/MahdiHedhli/PulsarMLX/tree/2d388856593b96a136fa5a6036c0e973919cedff) | `2d388856593b96a136fa5a6036c0e973919cedff`, 03-10-2026 | Texte MIT avec attributions Pulsar/ds4/ggml | Recherche Apple MLX géante, cible pratique Flash, runtime natif inachevé |

Le mot « Pulsar » ne désigne donc pas un moteur unique. Ne pas attribuer le streaming Rust/CUDA au fork Metal Qwen. Toute réutilisation doit conserver notices/licences des fichiers concernés; cet audit ne copie aucun code dans AirLLM.

## GLM-5.3 complet : existence et géométrie confirmées

Le dépôt officiel [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/ba32f8c19c8738e2c57c71798c601d477a7bf844) existe. Révision HF figée `ba32f8c19c8738e2c57c71798c601d477a7bf844`. [config.json officiel](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json) : `GlmMoeDsaForCausalLM`, `glm_moe_dsa`, 78 couches, 3 premières denses, 256 experts routés, top-8, 1 partagé, hidden=6144, expert intermediate=2048, MLA/DSA, un bloc nextn déclaré. Dépôt annoncé environ 756 GB, poids FP8/BF16/F32; ce n'est pas un checkpoint de 16 GB. Licence des poids propre à GLM-5.3, distincte des moteurs.

Chez Colibrì, `c/colibri.c` traite le GLM-5.2/5.3 complet. `c/glm53.c` traite **GLM-5.3-Flash**, autre modèle de 321B avec architecture hybride. Chez llama.cpp, `src/models/glm-dsa.cpp` est pertinent au complet; `glm5-next.cpp` concerne Flash. Les chiffres Flash ne valident pas le complet.

### Borne analytique utile, pas un benchmark

Un expert routé contient trois projections de `6144*2048`, soit 37 748 736 poids. À int4 idéal, hors scales/alignements, cela représente 18 874 368 octets. Pour les 75 couches MoE et top-8 :

`75 * 8 * 3 * 6144 * 2048 * 0.5 = 11 324 620 800 octets/token = 11.325 GB/token`.

Il s'agit du volume des experts actifs **sans hit**, pas du trafic réellement mesuré ni du travail résident total. À FP8 idéal : 22.649 GB/token. Pour un hit rate pondéré par octets `h`, une borne favorable de débit I/O int4 est `bandwidth_GB_s / (11.325 * (1-h))`, avant temps de calcul/déquantification/autres poids. Exemple purement hypothétique : SSD effectif 3 GB/s, h=0 → plafond 0.265 tok/s. Ce calcul ne démontre aucune vitesse sur le SSD présent et n'autorise pas une comparaison de qualité FP8→int4.

## Techniques et limites observées

### Colibrì : l'antériorité est déjà substantielle

[Code GLM figé](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/c/colibri.c) : cache LRU par couche, index expert→slot, pin historique, buffers experts en slabs groupés, PIPE de chargement borné. Le code contient prélecture PILOT, canal de couplage inter-couches COUPLE et garde `PILOT_EVICT_GUARD` contre une éviction spéculative trop coûteuse. La RAM partagée peut être enregistrée pour Metal; prélecture et calcul se recouvrent. Ces familles d'optimisations ne constituent pas en elles-mêmes une nouveauté AirLLM.

[Politique de réglage](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/tuning.md) : pin d'historique et LFRU récent/fréquent avec hystérésis 25%, quatre remplacements max; PILOT annonce rappel top-8 71.6% contre 41.3% pour précédent token. Prédictions erronées dépensent bande passante et polluent le cache; la doc reconnaît des régressions sur disque saturé. `--topp`, CACHE_ROUTE et DEGRADE_ZERO modifient les experts utilisés : impropres au chemin « mêmes équations ». MTP annonce 2.2–2.8 tokens/forward quand il rapporte; le coût dépend du cache et d'un vrai draft adapté.

[Mesure Metal M5 Max](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/METAL-M5MAX-PERF-REPORT.md) : GLM-5.2 int4, M5 Max 128 GB, RSS ~97.9 GB, cache chaud 74–75% hits. Meilleur bras publié 2.24 tok/s contre 2.06 ancien bras; défaut rebased 1.25. Attente OpenMP passive + PIPE rétablit les performances. Une seule exécution par configuration, variation thermique non bornée; divergences greedy entre runs parallèles reconnues. Aucune extrapolation légitime vers Air M5/16 GB. Le zéro copie ne résout pas le volume SSD requis.

[Compression rANS](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/int4-rans256-g0.md) : codec exact des bytes int4, ratio annoncé ~0.76 en per-row et ~0.89 en g64. **Outils hors ligne uniquement** dans cette spécification, pas accélération moteur Metal livrée. Le ratio de taille ne constitue pas une mesure de vitesse. Pour un ratio `r` et décompression séquentielle au débit de sortie `D`, avantage théorique seulement si `r/B + 1/D < 1/B`; donc `D > B/(1-r)`. Les buffers de sortie et la mémoire temporaire doivent être compris dans le budget.

[Batch continu GLM](https://github.com/JustVugg/colibri/blob/e0db083cdc9e563e0766cc6fc661ef23c32298e0/docs/experiments/glm52-continuous-batching-2026-07-31.md) : sur 6×RTX5090/251GiB, sans lectures disque, débit agrégé 4.84/6.33/8.14/8.30 tok/s pour 1/2/4/8 sessions, contre ~8.90 tok/s single-stream hors mux. La bande passante experts CPU plafonne. Aucun multiplicateur de débit démontré; KV et placement experts se disputent la même capacité. Les logs complets cités sont sur le disque de l'auteur, pas tous embarqués dans le dépôt.

### Pulsar SSD Rust/CUDA

[Code StreamingStore](https://github.com/giannisanni/pulsar/blob/18f990094e1091ee2ef3078f20be418f8c162f03/crates/engine/src/lib.rs) : cache host LFU avec départage par récence; éviction échantillonnée de 64 entrées pour limiter le coût de scan; cache GPU de triplets gate/up/down; placement statique sur GPU secondaires; prédiction inter-couches avec compteurs de rappel. Les voies CPU/GPU utilisent les coûts de bande passante mesurés pour partager les misses. Bon exemple d'ordonnancement au coût, sans transfert direct de ces constantes vers Apple.

[Fetch Linux](https://github.com/giannisanni/pulsar/blob/18f990094e1091ee2ef3078f20be418f8c162f03/crates/stream/src/lib.rs) : io_uring, O_DIRECT, alignement 4096, queue depth 32, consommation au fil des complétions, transfert de propriété des slabs vers cache. C'est Linux-only; CUDA/PCIe est structurellement différent de Metal mémoire unifiée. Attention en cas d'adaptation : les complétions vérifient `res<0`, mais pas explicitement `res==disk_len`; il faut traiter lecture courte et durée de vie des buffers avant de réutiliser ce schéma.

[Résultats README](https://github.com/giannisanni/pulsar/blob/18f990094e1091ee2ef3078f20be418f8c162f03/README.md) : GLM-5.2 à 2.7–2.8 tok/s annoncé sur 2×GPU16GB + Ryzen9900X + RAM30GB + NVMeGen5, quantification ~211GB, n=64, cache chaud. La doc distingue n=32 et n=64 et avertit du warm state persistant. Ce n'est ni GLM-5.3 complet dans sa précision officielle, ni M5 Air. Les types IQ avec codebooks peuvent décoder plus lentement que K-quants plus volumineux, illustrant le besoin de mesurer déquantification et cache ensemble.

### Pulsar Metal Qwen

[Benchmarks publics](https://github.com/loopai-hq/pulsar/blob/61f571f7abf95f1fca12df78ac575eca3fd6cdeb/docs/BENCHMARKS.md) : comparaisons alternées ABBA, M5 Max40GPU/128GB, Qwen dense/MoE avec draft DFlash. Débit client calculé entre premier et dernier chunk; modèles/checkpoints, nombre de tokens et préfill diffèrent selon certains adversaires. La documentation précise que la qualité des réponses n'a pas été comparée dans une campagne et que les moteurs sans spéculation utilisent leurs défauts. Les centaines de tok/s ne sont donc pas une preuve pour GLM géant en streaming. Des sessions M5 Pro24GB rapportent du swap, ce qui interdit d'emprunter leur configuration sans admission locale.

[Code QwenTarget](https://github.com/loopai-hq/pulsar/blob/61f571f7abf95f1fca12df78ac575eca3fd6cdeb/runtime/model/QwenTarget.cpp) : batch de vérification avec géométries spécifiques et kernels quantifiés. [Residency](https://github.com/loopai-hq/pulsar/blob/61f571f7abf95f1fca12df78ac575eca3fd6cdeb/runtime/metal/Residency.hpp) conserve un set Metal chaud pendant une activité puis le libère. Adaptation possible seulement dans le budget dur : pinning n'est pas réduction de mémoire. Aucun support GLM complet démontré par ces sources.

### llama.cpp actuel

[Loader](https://github.com/ggml-org/llama.cpp/blob/71255fefc092b0344f2acddb891388814ae961be/src/llama-model-loader.cpp) : mmap, lecture différée de grands tenseurs avec plages exclues du préchargement, placement par buffer backend et modes de chargement explicites. Le seuil auto du lazy read est 4 GiB. [Mapping](https://github.com/ggml-org/llama.cpp/blob/71255fefc092b0344f2acddb891388814ae961be/src/llama-mmap.cpp) gère `WILLNEED`/`RANDOM`; O_DIRECT est gardé Linux. [Metal](https://github.com/ggml-org/llama.cpp/blob/71255fefc092b0344f2acddb891388814ae961be/ggml/src/ggml-metal/ggml-metal.cpp) expose buffers partagés/mappés. mmap ne garantit pas un cache expert borné, une absence de swap, ni un chargement GPU entièrement à la demande.

[Spéculation](https://github.com/ggml-org/llama.cpp/blob/71255fefc092b0344f2acddb891388814ae961be/docs/speculative.md) : draft séparé, EAGLE3, DFlash, n-gram et MTP selon architecture; [GLM DSA](https://github.com/ggml-org/llama.cpp/blob/71255fefc092b0344f2acddb891388814ae961be/src/models/glm-dsa.cpp) contient le bloc draft MTP avec assertions de géométrie. Cela prouve la présence du code, pas la qualité/speedup du checkpoint GLM-5.3 sur le Mac. Un [signalement CPU GLM-5.3](https://github.com/ggml-org/llama.cpp/issues/29010) décrit des `@` répétés sur une configuration/commit précis; c'est un cas de bug à reproduire, pas une preuve que toute version actuelle échoue.

### Autre moteur pertinent : PulsarMLX

[État public](https://github.com/MahdiHedhli/PulsarMLX/blob/2d388856593b96a136fa5a6036c0e973919cedff/README.md) : référence Python/MLX Flash181.9GB, M1Ultra128GB, cache experts60–70GB, 2.6–4.2tok/s annoncés; natif Rust Flash encore en développement. Qwen/GLM-5.2/Flash portent des niveaux de qualification différents. Un exemple réduit 25 152 lectures à4 pour ~0.4% de gain, car la déquantification CPU dominait; la vectorisation des décodeurs fait ensuite chuter les temps de composants. Cette observation motive le profilage du chemin AirLLM complet avant de choisir E/S ou déquantification. Rien ne prouve Air16GB ou GLM-5.3 complet à ces vitesses.

## Opportunités exactes et critères d'acceptation pour AirLLM

1. **Cache au coût économisé par octet résident.** Comparer LRU, LFU/LFRU et admission pondérée par latence réelle, bytes compressés, réutilisation et transition. Les tailles/debits hétérogènes rendent le hit ratio seul trompeur. Coût Python/locks compris; préserver les experts actuellement consommés. Gain nouveau à mesurer, aucun pourcentage promis.
2. **Demand I/O avant spéculation.** Budget commun des buffers, déduplication d'union, lecture groupée lorsque coût d'une lecture additionnelle dépasse le coût des bytes de gap. Ne jamais transformer une prélecture en admission obligatoire. Mesurer bytes inutiles, prévisions utiles/tardives/fausses, évictions dommageables et wait foreground. Sur Apple, F_NOCACHE n'est ni O_DIRECT ni preuve physique de froid.
3. **Quantifié→kernel sans matérialisation FP32 géante.** C'est le levier pouvant réduire RAM/bytes et déquant simultanément; nécessite format exact pris en charge et oracle numérique. Le regroupement de commandes Metal évite un submit par petite matrice. Gain du composant et génération restent distincts.
4. **Multi-séquence bornée.** Partager l'expert chargé et réutiliser son poids entre lignes, avec KV isolés, équité et limites d'attente. Pour routage uniforme indépendant, experts distincts attendus par couche : `256*(1-(1-8/256)^S)`. S=2 donne seulement 1.016× de réutilisation idéale I/O, S=4 : 1.048×, S=8 : 1.115×. Forte corrélation peut aider; 128 sessions exigent mesure mémoire/latence et n'ont aucune garantie.
5. **Spéculation après mesure des unions d'experts.** Une vérification de k tokens peut charger l'union de k routages, pas un ensemble gratuit. Mesurer acceptation, temps draft, bytes et buffers en plus du tok/s. Token-exact greedy nécessite ordre de calcul stable; distribution préservée à température>0 exige correction/rejet et rollback KV adaptés, pas uniquement comparaison top1.
6. **Compression sans perte en option expérimentale.** Valider bytes décodés, consommation GPU/temp et débit réel; la réduction de taille seule ne justifie pas activation. Le rANS concurrent est une antériorité publique, pas une preuve de gain AirLLM.

## Campagne comparative minimale honnête

Figer moteurs/commits/binaires/configs, même checkpoint et même quantification effective, tokenizer/template/EOS identiques, top-k exact, même stockage, même prompt/token budget, mêmes limites RAM/KV/swap, ordre alterné et plusieurs répétitions. Conserver préfill, TTFT, decode, débit bout-en-bout, sorties intégrales, hit pondéré, bytes logiques/physiques, temps expert, pic mémoire et delta swap. Froid OS et froid expert doivent être étiquetés séparément.

Colibrì utilise souvent des conteneurs safetensors int4-g64 alors qu'AirLLM GGUF utilise d'autres grilles/scales : des suffixes «4-bit» identiques ne suffisent pas pour un match de précision. Sans poids effectifs comparables et sans admission sûre GLM sur cette machine, seule une comparaison de composants et de traces est recevable. Aucun résultat trouvé ne démontre GLM-5.3 complet réellement exploitable sur M5 Air 16 GB avec qualité inchangée et swap borné. La conclusion actuelle est un obstacle de capacité résidente et de bytes nécessaires, à confirmer par le moteur et le stockage réels, pas une impossibilité mathématique générale.

Manifest des fichiers consultés : `work/competitor-source-manifest.json` (URL figée, longueur et SHA-256 local). Aucun dépôt concurrent n’a été compilé ou exécuté.
