# Metal 4 TensorOps : périmètre réellement établi

Le prototype TensorOps **sur des produits entiers** a compilé et exécuté trois cas bornés sur le M5 : 384 résultats int32 identiques à l'oracle indépendant. Une substitution directe des produits Q8/Q2 par une multiplication matricielle flottante TensorOps ne satisfait pas le contrat d'arithmétique CPU existant. Les kernels Q8/Q2 compatibles du moteur ne font actuellement aucun appel TensorOps ; le nouveau prototype est séparé. Aucun gain GLM ni taux d'utilisation des Neural Accelerators ne lui est attribué.

Cette note est une inspection des sources et du SDK, datée du 2 novembre 2026. Un prototype borné a ensuite été écrit ; les états de compilation et d'exécution sont consignés séparément ci-dessous. La préparation du prototype n'a lancé aucun kernel pendant la campagne modèle complète.

## Capacités disponibles et chemins effectivement concernés

Le runtime est épinglé au commit llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`. Les références de source ci-dessous désignent ce commit, avec les patches privés décrits dans ce dossier.

| Chemin | Ce que montre la source | Limite de l'observation |
|---|---|---|
| Initialisation Metal | `ggml-metal-device.m:1147-1217` teste la famille Metal 4, compile des sondes MPP F16/BF16 et expose `has_tensor`. Le journal réel `work/v2/trace2-gpu.guard.stderr.log` annonce `has tensor = true` sur Apple M5. | Capacité et compilation disponibles ; ce journal n'est pas une mesure d'utilisation matérielle par chaque opération du modèle. |
| `kernel_mul_mm` natif | Sous `GGML_METAL_HAS_TENSOR`, `kernels/mul_mm.metal:13-76` emploie `mpp::tensor_ops::matmul2d`. Les spécialisations Q8_0/Q2_K déquantifient les poids par tuiles en `half` avant cette multiplication. Le descripteur de ce chemin utilise `relaxed_precision=true`. | C'est bien un chemin TensorOps potentiel, mais il ne reproduit ni la quantification temporaire CPU des activations, ni son ordre d'accumulation. Le modèle ne sélectionne pas ce chemin pour tous ses MUL_MAT. |
| `kernel_mul_mm_id` et attention tensor native | `mul_mm.metal:623` et `fa_tensor.metal:120-125` emploient également MPP, avec leurs propres descripteurs et critères de sélection. | L'existence de ces fonctions ne prouve pas qu'une projection KDA ou notre kernel quantifié les utilise. |
| Petit préfill Q8 natif | `ggml-metal-common.cpp:53-154` admet Q8 avec 2 à 16 colonnes, sous les conditions de formes et de strides. `ggml-metal-ops.cpp:2732` choisit ce chemin avant MUL_MM. `mul_mv_mma.metal` appelle `simdgroup_multiply_accumulate`. | Pour la projection `[4096,8192]` et les six colonnes capturées, ces conditions prédisent le chemin SIMD-group MMA classique. C'est une déduction du dispatch ; la capture de tenseurs ne contient pas le nom de pipeline exécuté. |
| Nos kernels Q8 compatibles | `kernels_q8/` : entiers, masques/décalages, réductions SIMD, FMA explicites ; activation préparée, inline ou partagée dans 4 608 octets de threadgroup. Le patch natif choisit un kernel MUL_MV dédié mono-colonne. | Aucun `mpp::tensor_ops` ni `simdgroup_multiply_accumulate` dans ces kernels. Leur fidélité mesurée ne constitue pas une preuve d'usage des Neural Accelerators. |
| Nos variantes Q2 compatibles | `kernels_q2/` : décodage du format GGUF, produits/réductions entiers et FMA dans l'ordre CPU. | Même distinction : kernels Metal personnalisés, sans appel TensorOps. |
| Nouveau prototype entier borné | `kernels_tensorops/int8_partials.metal` appelle MPP avec les types exacts `int8_t×int8_t→int32_t`, trois exécutions réelles et 384 sorties exactes. | Cas synthétiques de sous-sommes uniquement ; aucune intégration au graphe GLM, réduction FP32 complète ou mesure d'utilisation des Neural Accelerators. |
| Référence CPU10/GPU0 | Calcul du graphe sur CPU. | Aucun calcul GPU à décrire comme une accélération TensorOps. |

Dans la politique expérimentale sélective `AIRLLM_GLM_Q8_ONLY=1`, la marche complète du graphe impose le CPU à toutes les opérations sauf les produits vectoriels Q8 éligibles dont le poids est déjà Metal. Ceux-ci utilisent notre kernel compatible. Les produits multi-colonnes du préfill, RMSNorm, HC, convolution/scan/états KDA, routage et Q2 restent CPU ; certaines projections Q8 KDA mono-colonne éligibles passent sur GPU au décodage. Cette politique n'active donc pas les chemins natifs MUL_MM/attention TensorOps par la simple présence de `has_tensor=true`. Les sondes de capacité à l'initialisation restent distinctes du calcul du graphe.

Apple distingue explicitement les matrices SIMD-group classiques des opérations TensorOps destinées aux Neural Accelerators intégrés aux cœurs GPU M5. Il faut donc conserver ces noms distincts dans les résultats. [Présentation Apple M5/A19](https://developer.apple.com/videos/play/tech-talks/111432/).

## Contrat du SDK installé

`xcrun --show-sdk-version` renvoie **27.0**. Les headers inspectés sont dans :

```text
/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk/System/Library/Frameworks/
  MetalPerformancePrimitives.framework/Versions/A/Headers/
    MPPTensorOpsMatMul2d.h
    __impl/MPPTensorOpsMatMul2dImpl.h
```

Le header public annonce notamment `int8_t × int8_t → int32_t`, `int8_t × int4b_format → int32_t` et `int8_t × int2b_format → int32_t`, ainsi que leurs variantes entièrement non signées. L'implémentation contient des spécialisations entières. Pour une portée SIMD-group, les contrôles statiques imposent M et N multiples de 8, dont au moins un multiple de 16 ; K doit être multiple de 16, ou de 32 pour les types sous-octet statiques. La tuile entière M=16, N=8, K=32 du prototype est conforme et a désormais compilé puis exécuté réellement.

`xcrun -f metal` échoue ici car le compilateur Metal hors ligne n'est pas installé dans les outils sélectionnés. Cela n'empêche pas la compilation à l'exécution : le runtime existant a réellement compilé ses sondes TensorOps F16/BF16, puis notre sonde entière a compilé bibliothèque et pipeline via `newLibraryWithSource` et `newComputePipelineStateWithFunction`.

Apple documente l'arrivée des types entiers 4/8 bits dans les mises à jour 26 et des entiers 2 bits dans 27. Les nouvelles échelles de plan auxiliaire E8M0 ne décrivent pas automatiquement les échelles et minima arbitraires des blocs GGUF. [Présentation officielle des formats et de TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/).

## Pourquoi une GEMM quantifiée complète ne suffit pas à la fidélité

Le [comparatif détaillé](evidence/trace40-detailed-comparison.json) du préfill hybride révèle d'abord `hc_flat_norm-40` différent en bits, maximum `9.5367431640625e-7`, puis la première sortie nommée hors tolérance, `kda_q_proj-40`, maximum `0.01267658919095993`. La normalisation CPU accumule en FP64 ; Metal utilise une réduction FP32. Les activations de la projection ont donc déjà pu diverger. En outre, le CPU transforme les activations en Q8_0 temporaire, alors que le chemin Metal standard consomme les activations flottantes. Pour ce préfill de six colonnes, le CPU peut employer `nrc=2`, différent de son produit vectoriel `nrc=1`.

Ces sources établissent plusieurs différences arithmétiques ; elles n'isolent pas la fraction d'erreur imputable à chacune. Remplacer la seule projection par TensorOps sans harmoniser les entrées et le contrat de réduction ne constitue pas une correction démontrée. Le [contrat Q8 qualifié](Q8_FIDELITY.md) vise le cas `nrc=1`, avec des preuves modèle distinctes pour la tête de sortie. Il ne qualifie pas le préfill multi-colonne.

### Q8_0 mono-colonne

Le CPU ARM quantifie chaque bloc de 32 activations avec le réciproque de l'échelle FP32 non arrondie, arrondit les entiers au plus proche pair et stocke l'échelle en FP16. Il conserve ensuite **quatre sommes FP32 par bloc pair et quatre par bloc impair**. Chaque somme entière de huit produits reçoit son échelle via une FMA ; les réductions finales suivent l'ordre pairwise ARM. Voir `ggml/src/ggml-cpu/arch/arm/quants.c`, `quantize_row_q8_0` et `ggml_vec_dot_q8_0_q8_0`.

TensorOps entier peut être candidat au calcul des **sommes entières partielles**, car ces petits produits ne débordent pas int32. Une borne conservatrice de huit produits int8 est `8×128×128=131072`, représentable exactement en FP32. Mais une seule somme de 32 produits suivie d'une FMA ne préserve pas les quatre arrondis CPU séparés. Une GEMM flottante sur toute la dimension K change aussi l'ordre des FMA. Il faut garder les quatre sous-sommes, les deux chaînes pair/impair et leur réduction finale.

Une traduction possible masque les activations dans quatre colonnes distinctes : la colonne p conserve seulement les positions `4p..4p+3` et `16+4p..19+4p` du bloc. Une multiplication entière donne alors les quatre sous-sommes requises. Le padding à huit colonnes et la récupération des résultats coopératifs introduisent du travail supplémentaire. L'intérêt net de cette traduction reste inconnu.

### Q2_K mono-colonne

Chaque bloc GGUF de 256 poids contient 16 paires de nibbles échelle/minimum, 64 octets de codes dans un ordre de bits spécifique, puis deux échelles FP16 : **84 octets**, et non une simple matrice Int2 dense avec une seule échelle. Le CPU utilise les activations temporaires Q8_K et calcule deux termes entiers, schématiquement :

```text
D = somme(scale_nibble × q2 × activation_int8)
M = somme(min_nibble × activation_int8)
acc = FMA(-activation_d × weight_dmin, M, acc)
acc = FMA( activation_d × weight_d,    D, acc)
```

Les deux FMA et l'ordre séquentiel des blocs sont conservés dans nos kernels. Les termes `scale_nibble×q2` sont compris entre 0 et 45 ; une représentation int8 intermédiaire est possible. Avec `|activation_int8|≤128`, les bornes par bloc sont `|D|≤1474560` et `|M|≤491520`, sans débordement int32 et exactement convertibles en FP32.

Deux produits TensorOps entiers pourraient fournir D/M avant les mêmes FMA. Cependant, il faut décoder ou réorganiser les plans de bits, distribuer les nibbles par groupes de 16, récupérer les résultats et préserver l'état séquentiel FP32. Le type Int2 natif ne réalise pas automatiquement l'affinité `d×scale×q − dmin×min`, ni cet ordre d'accumulation. Une déquantification FP16 générale ou un remplacement des échelles par E8M0 modifierait le contrat. Une conversion bornée par tuile en int8 peut préserver les poids d'origine, avec un coût supplémentaire à mesurer ; une expansion intégrale du modèle est exclue du protocole.

## Limites de performance et prochain test borné

Apple explique qu'une matrice multipliée par un unique vecteur a une faible intensité arithmétique et peut être limitée par les lectures mémoire, alors que des GEMM plus larges réutilisent davantage les données. Cette observation générale ne mesure pas la part I/O, CPU, déquantification ou GPU de notre GLM. TensorOps ne réduit pas le volume d'experts lus depuis le stockage. [Analyse Apple du préfill et du décodage](https://developer.apple.com/videos/play/tech-talks/111432/).

Le protocole d'extension proposé, **après libération du créneau de campagne**, est limité à :

1. Compiler à l'exécution une sonde MPP `int8×int8→int32`, `relaxed_precision=false`, portée SIMD-group et tuile 16×32 par 32×8. Échouer explicitement si cette combinaison n'est pas compilable, sans substituer un produit flottant. Cette première étape et les trois fixtures synthétiques sont désormais réalisées.
2. Utiliser 16 lignes originales Q8_0 et un seul bloc d'activations quantifié par l'oracle natif. Vérifier exactement les quatre sous-sommes entières masquées, puis reconstruire les FMA CPU inchangées. Étendre seulement ensuite aux blocs pairs/impairs, aux zéros et aux arrondis limites.
3. Réserver moins de 1 MiB de payload et de scratch explicites, conserver verrou unique, limites RSS/pression/swap et calcul de référence indépendant. Pour Q2, tester séparément un seul bloc de 256 poids, avec décodage int8 borné et vérification indépendante de D/M puis des deux FMA.
4. Distinguer temps GPU réel, temps mural de soumission/synchronisation et coût de préparation. Comparer aux kernels exacts actuels avec le même travail inclus. Un futur taux d'utilisation des Neural Accelerators nécessite des compteurs ou un profil GPU approprié ; un appel API compilé seul ne le prouve pas.
5. N'élargir à une matrice de composant réelle puis au modèle complet qu'après fidélité exacte, budget mémoire vérifié et intérêt de performance reproductible. Les routes, les octets GGUF, les seuils de fidélité, les 600 agents logiques et le slot physique adaptatif restent hors du périmètre de cette substitution de calcul.

Cette proposition teste les unités matricielles sans assimiler leurs réductions flottantes au CPU. Le prototype ne fournit aucun gain modèle ni aucune validation sur le SSD externe Fanxiang.

## Prototype concret et modes séparés

Le source [kernels_tensorops/int8_partials.metal](kernels_tensorops/int8_partials.metal) effectue le produit entier fixe via MPP, avec `relaxed_precision=false` et un SIMD-group complet. [tensorops_native.mm](tensorops_native.mm) construit la bibliothèque et le pipeline avec MSL 4.1 et le mode mathématique sûr. Son mode `--compile-only` s'arrête avant la création de toute queue de commandes. Son mode `--execute` alloue 1 280 octets de buffers Metal par fixture et soumet un seul threadgroup de 32 threads, puis écrit les 128 résultats int32.

[tensorops_probe.py](tensorops_probe.py) prépare trois fixtures : entiers signés complets avec seed 5305, activations nulles et extrêmes signés. Le produit de référence utilise des entiers Python et les quatre groupes ARM de huit positions, sans employer une deuxième implémentation MPP. Les quatre colonnes de padding doivent être exactement nulles. Le payload comptabilisé inclut les entrées, leurs copies device, la référence et la relecture des résultats, avec un plafond de 1 MiB. Le processus natif est limité en temps, RSS, mémoire disponible et croissance du swap. Ces limites ne prétendent pas compter le cache du compilateur système ou toute la mémoire graphique interne.

Les modes `prepare`, `compile` et `execute` écrivent un rapport JSON, y compris lors d'un échec. La compilation seule laisse `accepted=false`. Seules trois exécutions complètes, des sorties int32 de forme exacte et une égalité entière de toutes les valeurs autorisent `accepted=true`. Le verrou de campagne existant est pris avant toute compilation ou exécution native. Aucune instruction de repli ne remplace le produit entier par du flottant en cas d'indisponibilité du backend.

```sh
python -m pytest -q research/glm53_gpu_v2/test_tensorops.py

python -m research.glm53_gpu_v2.tensorops_probe --mode prepare \
  --directory /absolute/path/tensorops-fixtures --output /absolute/path/prepare.json

# À exécuter après libération du créneau de campagne uniquement.
python -m research.glm53_gpu_v2.tensorops_probe --mode compile \
  --directory /absolute/path/tensorops-fixtures --output /absolute/path/compile.json
python -m research.glm53_gpu_v2.tensorops_probe --mode execute \
  --directory /absolute/path/tensorops-fixtures --output /absolute/path/execute.json
```

L'exécution du prototype teste des sous-sommes entières synthétiques. Elle ne quantifie pas les activations FP32 du modèle, n'effectue aucune réduction flottante sur K, et ne qualifie ni un kernel Q8/Q2 complet, ni son intégration au moteur. Les durées d'un seul petit dispatch par fixture sont diagnostiques ; elles ne forment pas un benchmark comparatif.

Première vérification effectuée : **7 tests portables réussis en 0,09 s**, sans création de device Metal. [tensorops_prepare.json](tensorops_prepare.json) enregistre 9 312 octets de payload comptabilisé, `host_compiled=false`, `device_execution_attempted=false`, `accepted=false`.

Après libération explicite du créneau, la première tentative de compilation avec des tenseurs `char` a été rejetée par une spécialisation MPP « Unsupported type ». Le [rapport brut](tensorops_compile_direct_chars.json) et le [shader rejeté](kernels_tensorops/int8_partials_direct_char.metal) sont conservés. Remplacer `char` par le type exact `int8_t` a résolu ce problème de type du prototype : ce n'était pas une preuve d'indisponibilité du SDK ou du matériel. Aucune substitution flottante n'a été employée.

Le [rapport de compilation](tensorops_compile.json) valide ensuite bibliothèque et pipeline, sans queue ni commande GPU, avec `accepted=false`. Le [rapport d'exécution réelle](tensorops_execute.json) valide trois commandes MPP et les **384 résultats int32**, y compris les colonnes de padding, sans aucun écart. Tous les garde-fous passent : 9 312 octets de payload connus, aucune mémoire threadgroup statique déclarée, pic RSS échantillonné 10 797 056 octets et aucune croissance échantillonnée du swap système.

Les horodatages GPU disponibles donnent respectivement environ 6,875, 5,917 et 6,000 µs ; le premier temps mural de commande est 1,462 ms, contre environ 0,192 et 0,171 ms ensuite. Il s'agit d'une commande par fixture, avec effets de démarrage et sans bras de référence chronométré. Ces durées ne démontrent aucun avantage sur les kernels SIMD actuels ou sur le CPU10, ni une utilisation mesurée des Neural Accelerators. Aucun test de performance du modèle n'a été lancé dans ce créneau de prototype.
