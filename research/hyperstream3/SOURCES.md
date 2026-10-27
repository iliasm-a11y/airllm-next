# Sources primaires et portée de réutilisation

Consultation : 27 octobre 2026. Les caractéristiques et résultats externes ci-dessous
ne sont pas des mesures d'AirLLM Next sur ce Mac.

## Apple : Metal 4, TensorOps et précision

- [Guide officiel Metal Performance Primitives (16 mars 2026)](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) : tuilage GEMM, mémoire device/threadgroup, accumulation coopérative, nécessité de faire participer tous les threads de la portée.
- [WWDC26 : Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/) : formats quantifiés, plans de données/échelles et possibilité de déquantifier un format personnalisé avant TensorOps. Cela ne prouve pas que Q4_K/Q6_K sont directement compatibles avec un format natif Apple.
- [WWDC25 : Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/) : construction des tenseurs et portée des opérations.
- [Spécification officielle MSL](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Le SDK local a aussi été inspecté, sans compilation ni calcul GPU :

`/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/MPPTensorOpsMatMul2d.h`

Les lignes 25, 120-121 et 385-395 confirment respectivement `float × float → float`,
le compromis de précision autorisé par `relaxed_precision=true` et la signature.
Le chemin H3 règle cette option à **false**, conserve des opérandes FP32 et ne
réencode pas les poids. Le header d'implémentation
`__impl/MPPTensorOpsMatMul2dImpl.h` expose `is_valid_element` (lignes 5101-5118).
Les exemples en commentaires du header public utilisent encore `get_mask` ; cette
ancienne graphie n'est pas une preuve de compatibilité du compilateur actuel.
Les contraintes M/N aux lignes 6370-6374 acceptent une tuile 16×16×32.

Une API FP32 stricte ne garantit pas des résultats bit à bit identiques à une
autre association des additions. La compilation, la tolérance indépendante et
la fidélité gloutonne sur checkpoint réel restent des validations distinctes.

## MLX

Tag vérifié : `v0.32.4`, commit `36969c094354086e15d2a327a9c50feb20b9cf11`.

- [device.cpp à ce tag](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp) : compilation à l'exécution, Metal 4.1 sur macOS 27 et Metal 4.0 sur macOS 26.
- [Bindings fast.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/python/src/fast.cpp) : option `compile_options={"math_mode":"safe"}` de `mx.fast.metal_kernel`.
- [Custom kernel](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/custom_kernel.cpp) : construction puis évaluation GPU ; les erreurs peuvent être différées jusqu'à `mx.eval`.
- [Thin NAX GEMM](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_thin_nax.h) : exemple de tenseurs et d'accumulateurs coopératifs. Sa précision relâchée n'est pas reprise.
- [Licence MIT MLX](https://github.com/ml-explore/mlx/blob/main/LICENSE), Copyright © 2023 Apple Inc.

L'absence de l'exécutable `xcrun metal` ne démontre pas l'absence de compilation
MPP à l'exécution. La présence de MLX et du SDK ne prouve pas non plus qu'un
kernel particulier compile ou s'exécute correctement.

## llama.cpp et GGML

- [PR #29869 : few-row MMA mat-mul and batched copies for speculative decoding](https://github.com/ggml-org/llama.cpp/pull/29869), fusionnée le 5 octobre 2026 : chemin SIMD 8×8 pour 2 à 16 lignes et réutilisation de déquantification ; Q4_K/Q6_K compris. Ce chemin constitue un repli lorsque l'API Tensor est absente. Les mesures M3 Ultra de cette PR ne prédisent pas un gain M5 hors mémoire.
- [mul_mv_mma.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv_mma.metal).
- [mul_mm.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal) : chemin MPP avec déquantification bornée en mémoire threadgroup. Le réglage de précision relâchée de cet exemple n'est pas repris.
- [PR #30047 : few-row MMA kernels for MUL_MAT_ID](https://github.com/ggml-org/llama.cpp/pull/30047), encore ouverte à la consultation : sélection selon le nombre de tokens par expert ; un batch spéculatif ne garantit pas suffisamment de réutilisation dans chaque expert MoE.
- [Licence MIT llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE), Copyright (c) 2023-2026 The ggml authors.

H3 réutilise les décodeurs Q4_K/Q6_K déjà présents dans AirLLM Next et conserve
leur attribution à GGML. Les nouveaux corps de kernels sont une implémentation
locale fondée sur les API et dispositions documentées, sans import intégral des
kernels cités. Tout import ultérieur d'un corps de code MIT doit conserver le
copyright et la licence correspondants. Les documents Apple restent soumis aux
conditions Apple ; ils ne sont pas incorporés au dépôt.

## Validation et limites

`tests_hyperstream3/test_m5_kernels.py` sépare tests portables avec faux MLX et
tests matériels. Les seconds nécessitent `AIRLLM_M5_GPU_TEST=1`, puis
l'admission mémoire avant l'import MLX ; ils n'ont pas été exécutés sous pression
macOS de niveau 2. L'oracle scalaire indépendant est extrait par AST de
`tests_next/test_gguf_metal.py`, sans charger son conftest ou ses imports GPU.

Le dispatcher synchronise chaque appel expérimental pour pouvoir détecter une
erreur MLX différée et revenir à l'opération de référence. Cette synchronisation
peut réduire le chevauchement et doit compter dans le coût réel. Les timings de
calibration sont des temps hôte synchronisés, pas du temps GPU pur.
