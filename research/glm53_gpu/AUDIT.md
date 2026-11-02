# Audit du chemin GLM-5.3 GPU et du stockage

Audit du 2 novembre 2026. Référence AirLLM Next : PR #17, commit `0ab4ad0bd068ae8e4411d63b4c20e4c545c2a2a5`. Runtime natif : llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`, avec les patches de préchargement de cette PR. Les sections historiques réanalysent les archives et captures existantes. La section « Ablations nouvelles » réexamine indépendamment des captures produites par la nouvelle campagne ; cet auditeur n'a pas lancé lui-même ces inférences.

## Fidélité : divergence réelle sur une trajectoire déjà imposée

Le [comparatif historique](../glm53_m5/evidence/published/comparisons/fidelity-hybrid6-comparison.json.gz) oppose CPU4/GPU0 à CPU4/GPU6, cache MoE GPU de 256 Mio. Le candidat est déjà en mode `forced=true`. Les six tokens du prompt et les quatre entrées de décodage sont identiques. Les quatre argmax de sortie sont également identiques : `12089, 13, 4929, 9621`. Le changement des routes ne provient donc pas ici de l'injection de tokens générés différents.

Les critères annoncés avant comparaison étaient : routes top-8 ordonnées et IDs exacts, valeurs finies, `atol=0.001`, `rtol=0.0001` pour les flottants Metal. Les 619 520 logits ont un écart maximal de `0.5348455905914307`, une RMSE de `0.08897702251287894` et 612 177 valeurs hors tolérance. Le chemin échoue à ces critères ; l'identité des quatre argmax ne le qualifie pas.

La réanalyse trie les couches par numéro, et non par nom alphabétique :

| Premier écart observé | Étape | Tenseur | Résultat |
|---|---:|---|---|
| KDA différent en bits | préfill | `kda_scan_out-1` | max absolu `4.5704655e-7`, encore dans la tolérance |
| Routeur hors tolérance | préfill | `ffn_moe_logits-3` | max absolu `0.0120286942` |
| Ordre top-8 différent | préfill | `ffn_moe_topk-5`, colonne de token 4 | permutation des experts 155 et 104 |
| Ensemble top-8 différent | préfill | `ffn_moe_topk-15`, colonne de token 4 | expert 156 au lieu de 89 |
| KDA hors tolérance | préfill | `kda_scan_out-24` | max absolu `0.00110105984` |

Sur les 378 vecteurs top-8 canoniques (42 couches MoE fois 9 positions : six prompt et trois décodages), 163 ont un ordre différent. Parmi eux, 108 ne changent que l'ordre et 55 changent aussi l'ensemble choisi. Cela représente 364 positions différentes sur 3 024. L'ordre compte aussi pour la somme flottante des sorties ; conserver seulement l'ensemble des experts ne suffit pas.

Le comparateur historique signale en plus 15 alias `ffn_moe_topk-40..44 (reshaped)` absents d'un côté aux étapes 1 à 3. Il s'agit d'une différence de couverture de capture liée au graphe. La divergence des tenseurs top-8 canoniques et des logits subsiste indépendamment de ces alias ; ils n'expliquent pas l'échec numérique. Toute comparaison future doit distinguer couverture des alias et valeurs canoniques, sans ignorer silencieusement un tenseur de calcul manquant.

Sources : comparaison archivée ci-dessus ; captures locales complètes `stock-cpu4` et `hybrid6`. Les archives publiques sélectionnées omettent les sorties KDA brutes ; les résultats KDA ne sont donc pas entièrement recalculables à partir de ce seul sous-ensemble public. [Le script de réanalyse](audit_evidence.py) et [ses résultats](audit_evidence.json) identifient les sources publiques par SHA-256 et réexaminent facultativement les captures brutes.

## Un changement de graphe global confond l'expérience hybride

Les logs d'initialisation donnent un fait plus précis que « le GPU diverge » :

- CPU : `fused DeepSeek V4 HC post enabled`.
- Hybride : la couche 39 est CPU, mais son opération HC post est assignée à MTL0 ; `resolve_fused_ops` désactive ensuite HC post pour tout le graphe.

Le code du runtime confirme cette décision globale : `src/llama-context.cpp:620-657` inspecte les dispositifs des opérations fusionnées et met à `false` le booléen partagé si une seule couche est mal placée. `src/models/glm5-next.cpp:502-542` construit alors une suite MUL/ADD/CONCAT au lieu de DSV4_HC_POST pour toutes les couches. La référence et l'hybride n'ont donc pas le même graphe dans leurs premières couches CPU, alors que seuls les poids des cinq dernières couches et la sortie sont offloadés.

Le premier écart KDA à la couche 1 est compatible avec une différence introduite par HC post à la couche précédente. Le kernel CPU fusionné (`ggml/src/ggml-cpu/ops.cpp:11548-11621`) accumule `sum += rv * cv` dans une boucle, tandis que le graphe décomposé matérialise des multiplications séparées. Contraction FMA et arrondis peuvent différer. **Le rôle causal exact et l'ampleur de cette contribution ne sont pas établis sans ablation.** Les captures actuelles ne conservent pas les entrées/sorties de chaque HC post.

`op_offload=true` et `offload_kqv=true` sont aussi deux différences de politique dans le pilote historique. Ils ne prouvent pas, à eux seuls, une multiplication GPU dans les premières couches : la politique Metal pour les poids hôtes a un seuil de batch 32 par défaut (`ggml-metal-device.m:1285`), au-dessus des six tokens du prompt. Il faut enregistrer les opérations effectivement placées et l'état des fusions.

Le [patch expérimental de placement HC](patches/llama-layer-local-hc.patch) ajoute deux réglages désactivés par défaut, uniquement pour l'architecture GLM5_NEXT :

- `AIRLLM_GLM_LAYER_LOCAL_HC=1` assigne `hc_attn_post` et `l_out` au dispositif de leur couche, avec vérification du support de l'opération. L'objectif est d'éviter que la frontière CPU/GPU désactive HC post partout.
- `AIRLLM_GLM_HC_POST=0` désactive HC post fusionné sur CPU comme sur GPU pour une ablation à graphe comparable. Ce réglage change la trajectoire arithmétique de la référence et sert uniquement au diagnostic.

L'application du patch a été vérifiée par `git apply --check` sur le commit épinglé. Cela ne prouve ni sa fidélité ni un gain. Les réglages doivent être consignés dans chaque résultat, et ne doivent pas être activés sur l'application livrée avant comparaison. GDN est codé `fused_gdn_ar/ch=true`, `auto_fgdn=false` dans ce runtime ; il n'existe pas de paramètre public équivalent pour le désactiver. HC post n'avait pas non plus d'API publique de sélection avant ce patch.

Ablations utiles, sur les mêmes tokens forcés : CPU par défaut contre CPU HC post désactivé ; hybride historique contre hybride au placement HC local ; puis CPU et hybride tous deux HC post désactivé. Une tête seule avec `op_offload=false` et `offload_kqv=false` est un autre pilote de portée réduite. Les logits, routeurs et KDA doivent être comparés séparément, avant de mesurer la vitesse sans instrumentation détaillée.

## Ablations nouvelles : le changement HC explique le préfixe CPU observé

Le nouveau binaire compilé avec le patch, exécuté en CPU10 sans option HC, reproduit bit à bit la capture CPU4 historique sur les quatre étapes et toutes les familles comparées (vérification de la campagne principale). Le simple ajout du patch désactivé ne change donc pas ces résultats.

Une exécution CPU10 avec `AIRLLM_GLM_HC_POST=0`, sur les mêmes entrées forcées, dérive de la référence fusionnée (écart maximal de logits annoncé par le comparatif de la campagne : environ `0.546031`). Ce candidat reste un diagnostic non qualifié.

La comparaison indépendante de ce CPU HC-post-décomposé avec l'hybride GPU6 **historique** est plus probante pour l'origine des écarts précoces : tous les tenseurs capturés des couches 0 à 39 sont bit à bit identiques, soit 536 tenseurs/alias et 4 522 248 éléments (2 664 routes, 95 904 valeurs de routeur, 4 423 680 sorties KDA), sans valeur non finie. Le prompt et les entrées de décodage sont identiques. Le script [audit_hc_ablation.py](audit_hc_ablation.py) produit [audit_hc_ablation.json](audit_hc_ablation.json), avec empreintes des manifestes et contenus sélectionnés.

Cette ablation établit, pour cette trajectoire et ces tenseurs, que le chemin HC post décomposé sur CPU reproduit entièrement le préfixe divergent de l'hybride. Les différences précoces ne doivent pas être attribuées aux multiplications quantifiées GPU des couches 40 à 44. Elle ne qualifie pas les opérations non capturées, toutes les trajectoires, ni la fidélité GPU en aval.

Le pilote nouveau « tête seule », sans offload opportuniste ni KQV, conserve également routes/routeurs/KDA bit à bit selon le comparatif de la campagne, mais les logits de tête dépassent la tolérance (écart maximal annoncé d'environ `0.05558`). Il échoue donc à la fidélité globale. Cela isole une seconde source numérique dans la projection de sortie sans démontrer qu'un autre kernel la résout.

Le premier pilote hybride avec placement HC local et cache de 384 Mio (`hybrid6-local-hc384`) s'est arrêté par assertion native après les premiers logits : sélection des experts de la couche 40 dans le même segment de graphe que leur lookup de cache. Le rapport de surveillance enregistre `returncode=-6`, 19,415 s et aucun dépassement mémoire ; ce n'est ni une réussite de fidélité ni un arrêt mémoire. Les sorties brutes et cet essai échoué doivent être conservés.

L'ordonnanceur ne crée actuellement une nouvelle frontière sur un poids hôte incompatible que si le segment contient déjà des entrées copiées. Le callback du cache exige cependant que ses IDs aient été calculés dans un segment antérieur, y compris quand le segment précédent n'a pas d'entrée copiée. Le patch privé [llama-moe-cache-split.patch](patches/llama-moe-cache-split.patch) ajoute `AIRLLM_GLM_CACHE_SPLIT=1`, désactivé par défaut : il force une frontière non vide pour le GET_ROWS d'un poids hôte I32 `[1,N]` nommé `moe_cache.slot_map-*`, avec callback de copie et backend incompatible. Il conserve l'assertion de cache. C'est un correctif de dépendance à valider par compilation et nouvelle exécution ; son existence ne qualifie aucune performance ni la fidélité GPU.

### Nouvel hybride avec frontière de cache réparée, 384 Mio

La nouvelle exécution CPU10/GPU6, HC local activé et `AIRLLM_GLM_CACHE_SPLIT=1`, termine ses quatre étapes avec le cache de 384 Mio. Le garde-fou ne constate aucun dépassement : pic RSS échantillonné 5 087 985 664 octets, croissance du swap nulle. Les 31,762 s comprennent chargement, sérialisation et captures : **ce temps n'est pas un débit de décodage**. Le processus termine avec le code 0, mais le contrôle de fidélité rejette le résultat.

Les 536 tenseurs/alias des couches CPU 0 à 39 restent bit à bit identiques à la référence historique, soit 4 522 248 éléments. Les premiers écarts se trouvent maintenant dans la première couche GPU : `kda_scan_out-40` au préfill (max absolu `0.0000538854`) et `ffn_moe_logits-40` (max absolu `0.0102720261`, hors tolérance).

Sur l'ensemble des quatre étapes :

| Mesure | Nouveau candidat GPU6 |
|---|---:|
| Logits, écart absolu maximal | `0.1210167407989502` |
| Logits, RMSE | `0.02095468260868081` |
| Routeur, écart absolu maximal | `0.05374002456665039` |
| Positions top-8 différentes | 8 / 3 024 |
| Vecteurs top-8 à ordre différent | 4 / 378 |
| Vecteurs top-8 à ensemble différent | 0 / 378 |
| Sorties KDA hors ancienne tolérance Metal | 0 |
| Sorties KDA différentes en bits | 589 820 / 5 013 504 |

Les huit positions top-8 différentes correspondent à quatre permutations par paires : couches 41, 42 et 44 au préfill, couche 40 à l'étape 2. La première permute 157 et 26 à la couche 41, première colonne du prompt. L'absence de changement d'ensemble dans cette courte trajectoire ne satisfait pas l'ordre exact requis et ne qualifie pas d'autres trajectoires. Les critères n'ont pas été assouplis. Les erreurs de couverture des alias historiques sont consignées séparément.

Le cache rapporte 42 hits, 78 misses (35 %) et 614,25 Mio chargés. Il fonctionne donc avec réutilisation sur cet essai, sans que ce taux établisse un gain de temps. La capacité peut aussi modifier l'éligibilité du préfill selon les dimensions effectives du graphe ; cela doit être vérifié plutôt qu'inféré du seul budget. Les 120 accès de cache rapportés correspondent à cinq couches fois huit experts fois trois étapes de décodage et n'établissent pas une utilisation du cache pendant le préfill. Une répétition à 256 Mio, avec les mêmes autres options, est traitée séparément par la campagne principale.

Les résultats et empreintes de sources sont conservés dans [audit_gpu_retest384.json](audit_gpu_retest384.json), produits par [audit_retest.py](audit_retest.py). Le [contrôle indépendant du préfixe CPU](audit_hc_prefix_fixed.json) inclut aussi les empreintes des manifestes et contenus sélectionnés. Ces éléments montrent que le problème de graphe précoce est isolé et que l'assertion de dépendance est évitée pour ce scénario ; ils ne démontrent ni une fidélité GPU globale ni un nouveau gain de décodage.

## Q2_K et Q8_0 : les kernels fusionnés existent déjà

La quantification du fichier ne suffit pas à définir toute l'arithmétique d'exécution :

- Sur CPU, les traits GGML Q8_0 utilisent `ggml_vec_dot_q8_0_q8_0` et des activations converties en Q8_0 ; Q2_K utilise `ggml_vec_dot_q2_K_q8_K` et des activations converties en Q8_K (`ggml/src/ggml-cpu/ggml-cpu.c:280-316`, conversion vers le type de produit scalaire à partir de la ligne 1375).
- Le chemin Metal vectoriel `kernel_mul_mv_q8_0_f32_impl` consomme directement les activations F32 et applique les échelles aux entiers Q8_0. `kernel_mul_mv_q2_K_f32_impl` applique de même échelles/minima Q2_K aux activations F32 pendant l'accumulation (`ggml/src/ggml-metal/kernels/mul_mv.metal:514-585`, `1241-1330`). Il ne crée pas de matrice de poids F32 complète.
- Les variantes `MUL_MAT_ID` appellent déjà ces kernels avec les experts sélectionnés. Les variantes matricielles Q2_K utilisent aussi une déquantification par blocs et des calculs SIMD-group, avec des intermédiaires half pour certains chemins.

Ce sont des différences numériques de méthode identifiables dans le code, même si tous les octets des poids restent inchangés. Elles sont une autre cause candidate pour les couches effectivement GPU. Elles ne démontrent pas que le chemin Metal est incorrect par rapport aux poids mathématiques ; elles montrent pourquoi il ne reproduit pas automatiquement la référence CPU quantifiée choisie. Une nouvelle fusion doit préciser la méthode de référence, respecter les valeurs GGUF, comparer les arrondis d'activations et les réductions, puis qualifier les routes exactes. Remplacer simplement ces kernels par une multiplication F32 sur poids déquantifiés ne résout pas la parité CPU.

## Cache MoE : la configuration de 256 Mio peut tourner à vide

Le log hybride enregistre 252 Mio effectivement alloués, 0 hit, 120 misses, 945 Mio chargés. Un expert Q2_K gate/up/down occupe 8 257 536 octets, soit 7,875 Mio. La banque contient donc 32 slots, partagés entre cinq couches MoE de même disposition. Une seule passe de décodage touche 40 clés distinctes `(couche, expert)` si chaque couche conserve son top-8, soit 315 Mio. Un parcours cyclique dans un cache de 32 slots peut tout évincer avant le token suivant, même sans changement de sélection.

Cette explication structurelle est cohérente avec les compteurs observés, sans prouver que chaque miss avait cette cause. Un budget borné de 352/384 Mio, une répartition par couche, ou un petit sous-ensemble de couches admises mérite une ablation si les garde-fous mémoire l'autorisent. Un budget plus grand ne garantit ni un hit ni un gain. Pour le préfill de six tokens, le cache actuel est contourné quand `6*8 > n_slots`, avant même de calculer le nombre réel d'experts distincts (`llama-moe-cache.cpp:479-489`).

Le cache existant fait déjà des uploads asynchrones, mais récupère les IDs sélectionnés puis synchronise le backend avant le plan de remplacement (`llama-moe-cache.cpp:566-597`). Cela n'est pas une preuve de chevauchement I/O/calcul. Précharger les futurs experts avant leur sélection requerrait de la spéculation ; une voie exacte peut seulement charger les IDs déjà connus et chevaucher le travail indépendant, avec durée de vie des buffers et limite totale explicites.

## Coûts calcul et stockage : ce que les mesures permettent

Le profil détaillé CPU4 historique utilise `cb_eval` sur chaque opération et synchronise les backends. Pour les trois étapes de décodage, il additionne 46,753683 s de temps instrumenté :

| Opération | Temps instrumenté | Part | Octets de lecture comptés par macOS |
|---|---:|---:|---:|
| MUL_MAT | 36,712767 s | 78,52 % | 28 334 112 768 |
| MUL_MAT_ID | 9,086962 s | 19,44 % | 8 459 669 504 |
| Toutes opérations | 46,753683 s | 100 % | 36 848 259 072 |

MUL_MAT et MUL_MAT_ID concentrent ainsi 97,96 % du temps instrumenté. Leurs temps incluent les lectures par défauts de page et ne sont pas des chronométrages isolés de calcul/déquantification. Les copies de l'ordonnanceur et écritures des captures sont en dehors des buckets ; la synchronisation peut modifier fusions et chevauchement. Il est impossible d'en déduire « 78,5 % du temps est du calcul dense ». Le volume macOS moyen est 12,283 Go par étape dans ce profil, proche des 12,235 Go de plages de poids potentiellement touchées calculées sur les métadonnées, sans égalité démontrée avec les lectures physiques NAND.

Les mesures de génération non instrumentées de la PR #17 sont :

| Configuration | Prompt / sortie | Débit de décodage | TTFT hors chargement |
|---|---|---:|---:|
| CPU10 GPU0, répétitions 1/2 | 6 / 16 | 0,166421 / 0,150614 tok/s | 20,325 / 21,462 s |
| CPU10 GPU0, répétitions 1/2 | 15 / 32 | 0,108502 / 0,139161 tok/s | 53,057 / 56,124 s |
| CPU4 GPU6, pilote | 6 / 8 | 0,143209 tok/s | 34,384 s |
| CPU10 GPU1, groupe sortie, pilote | 6 / 8 | 0,154668 tok/s | 20,880 s |

Le pilote GPU6 n'est pas une comparaison CPU10 à threads constants et échoue à la fidélité. Le pilote de tête ne démontre pas de gain face au pilote CPU10 de même longueur (0,155483 tok/s), et n'a pas la portée d'un décodage soutenu. Les opérations fusionnées déjà présentes ne peuvent pas être créditées comme une nouvelle optimisation.

Les requêtes CPU10 de 16 tokens comptent respectivement 214,790 et 214,785 Go de lectures macOS en 110,459 et 121,057 s, préfill compris ; ces quotients (1,945 et 1,774 Go/s) ne mesurent pas la vitesse du SSD externe. Elles utilisent en moyenne 6,52 et 6,67 secondes CPU par seconde murale sur 10 threads, ce qui ne fournit pas non plus une part de temps I/O par simple soustraction. Le cache de fichiers n'est pas contrôlé. Les compteurs CPU historiques sont convertis avec la base de temps Mach vérifiée 125/3 ns par tick ; les anciens noms `*_ns` ne suffisent pas à interpréter les valeurs.

Sur un SSD réellement à 1 Go/s et sans réutilisation, **l'hypothèse** d'une lecture physique de 12,235 Go par token conduirait à environ 12,2 s de stockage par token. Ce calcul conditionnel n'est ni une mesure de Fanxiang ni un plafond démontré : résidence, réutilisation, read-ahead, mise en cache et accès réels changent les octets lus. La connexion fibre n'accélère pas la lecture locale des poids déjà téléchargés.

La qualification demandée requiert donc un même stockage explicitement identifié, des paires CPU10/GPU à ordre alterné, des lectures par processus et des compteurs du périphérique, l'état du cache décrit, puis des temps GPU/copies/synchronisations séparés. Une mesure effectuée sur le disque interne ou sous une autre génération concurrente ne doit pas être présentée comme un benchmark Fanxiang isolé.

## Reproduction et état du patch

Depuis la racine AirLLM Next :

```sh
python3 research/glm53_gpu/audit_evidence.py --output audit-public.json
python3 research/glm53_gpu/audit_evidence.py --captures /chemin/vers/captures --output audit-complet.json
```

La seconde commande requiert NumPy et les captures `stock-cpu4`/`hybrid6` complètes ; elle ne lit pas les 110 Go de poids. Pour le patch natif, partir d'une copie isolée de llama.cpp au commit épinglé, conserver les options de compilation de la référence et appliquer :

```sh
git apply --check /chemin/airllm-next/research/glm53_gpu/patches/llama-layer-local-hc.patch
git apply /chemin/airllm-next/research/glm53_gpu/patches/llama-layer-local-hc.patch
cmake -S . -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j 2
```

Les patches mmap/préchargement de PR #17 doivent être identiques dans les deux bras d'une comparaison qui les emploie. Utiliser ensuite la sonde native et la surveillance fournies par cette branche ; ajouter un drapeau d'environnement au binaire installé existant ne peut pas activer une modification absente de ce binaire. Ce document n'annonce aucun nouveau gain ni une résolution complète de la divergence. Les 600 identités logiques et le slot physique unique ne sont modifiés par aucun fichier de cet audit.

## Pilotes de génération sans instrumentation et cache exact

Après les diagnostics, une seule paire courte a été mesurée avec le serveur patché, CPU10 dans les deux bras, contexte 512, batch/ubatch 8, prompt 6 / sortie 8, même préchargement. CPU10/GPU0 : **0,17336794 tok/s**, TTFT 20,9648 s. CPU10/GPU6/cache384 avec placement HC et frontière du cache : **0,18267086 tok/s**, TTFT 20,9874 s. Les huit IDs sont identiques ; le swap système reste à 16 515 072 octets dans les deux essais. Les journaux et événements SSE sont conservés dans `evidence/`.

L'écart de 5,37 % est descriptif d'une paire, sans ordre alterné ni maîtrise du cache macOS, et le candidat échoue aux critères de logits/routes. Ce n'est donc ni un gain de décodage qualifié ni une accélération considérable. Les essais portent sur le disque interne ; aucune mesure du Fanxiang n'est obtenue.

À placement/graphes identiques, la comparaison I/O stricte des captures cache256/cache384 passe bit à bit sur 619 520 logits et toutes les familles capturées, y compris les alias communs. Cache256 : 0 hit / 120 misses, 945 Mio chargés ; cache384 : 42 hits / 78 misses, 614,25 Mio. Cela valide l'absence de changement arithmétique de cette variation de budget sur quatre étapes, pas le passage des critères CPU par l'hybride.
