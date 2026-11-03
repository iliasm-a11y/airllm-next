# Fidélité du runtime V4 et rétention dense de 256 Mio

Analyse des métadonnées closes A16/E16. Ces diagnostics à entrées imposées établissent une fidélité sur la trajectoire capturée. Leurs durées comprennent instrumentation, captures et hachages ; elles ne mesurent pas le débit de décodage. Les résultats ABBA E8 en cours restent hors de cette note.

## Runtime effectivement compilé

La recette complète a produit le runtime V4 depuis llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`. Le manifeste indique `status=built`, 16 patches, 3 auxiliaires du diagnostic et 59 commandes enregistrées. Il décrit 13 fichiers binaires réguliers, dont 10 bibliothèques dynamiques, et 16 liens symboliques. La compilation se termine le `2026-11-02T23:02:26.953090+00:00`. `benchmark_performed=false` et `model_loaded=false` décrivent la phase de construction ; les diagnostics distincts ci-dessous ont ensuite chargé le modèle.

Les 10 empreintes de bibliothèques et celle du diagnostic enregistrées dans A16 et E16 correspondent exactement aux artefacts du manifeste V4. Les 16 empreintes de patches du manifeste de construction correspondent à celles du manifeste source V4. Les fonctionnalités expérimentales sont désactivées par défaut dans la recette ; leur activation dans E16 est explicite.

| Objet | SHA-256 |
|---|---|
| `work/runtime-v4/manifest.json` | `7b0ac6dddabf542aae42d7fb8bb8762d6b98e1705dc423b09ad071112d0c8750` |
| Diagnostic `glm-native-fidelity` | `5cabde2cec89f1ed1bb71e962d25509ccb466eed3d5395a10d34caed841504b2` |
| Serveur `llama-server` | `a0c7632523ae199f576b8b63e423f0b0dc80068545a62d45d2132ceb765a9f83` |
| Diff des sources appliquées, champ du manifeste | `99b66feb027288d5ec6eaece5b91619a3757555fc9008c767b7df0047d7a908e` |
| Source du diagnostic, champ du manifeste | `589d10c505073d8bc5e4038b90a9db377d9db8b88533b65f9e299d701a2d9037` |
| `work/v3/v4-source-manifest.json` | `339b07abe8a9d76d664413d293dd35e2e77d6cac1f36f494be366013fd72e5b8` |

Le manifeste source porte sur les modifications suivies et les nouveaux en-têtes réellement présents dans le checkout isolé lors de la compilation : 13 fichiers natifs, dont `ggml-cpu-dense-lookahead.h` et `llama-dense-lock.h`. `dense_lock_header_matches_reviewed_source=true`. Ce relevé ferme la couverture des nouveaux en-têtes que le seul diff des fichiers suivis ne suffirait pas à décrire. Il ne constitue pas une affirmation de reproductibilité binaire indépendante.

| Source native modifiée ou ajoutée | SHA-256 |
|---|---|
| `ggml/src/ggml-backend.cpp` | `dce7b42ecdba028d7c7ee2b979eedcf2277a129b4f0d8738fcf42b929c3a00b9` |
| `ggml/src/ggml-cpu/ggml-cpu-dense-lookahead.h` | `cc1bf5055a33f306c3dc684c7921c854b4eb61dbcd1f83d1a13d71cb2c171685` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `4e2c75dedbf4e074e3d763d94dff956b97508c7c86bfba31e1cf5ddc0060b93e` |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `39e90d506a6aecf425a14d02054fb7d0a9920d997cd5d2be79c3498c9160efe2` |
| `ggml/src/ggml-metal/ggml-metal-context.m` | `9ecb6454ce7278ea904101100091d362dbfe29cf9a15a0f0a942af71bef0c631` |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | `c841b02306f755962f9c5e8424c9eac6862c1eef533ee079f5f3be187570cce6` |
| `ggml/src/ggml-metal/ggml-metal-device.m` | `c3b7cbe2d64f3ab01698c6ba1a5da7a932a9146e42b7f00a469bcb4a85f7484e` |
| `ggml/src/ggml-metal/kernels/mul_mv.metal` | `b36763bdb3534ff78581a933a8e21ec3c7a81062643867ac621df8eb3c67a185` |
| `src/llama-context.cpp` | `33d8cf6d489cce950bf35bc19979f90f87b2e987e9fc095875ba8cf7debedcb4` |
| `src/llama-dense-lock.h` | `8daee7fc71a4bc49db5885c1af1f6570ffa091bfe03ca2c3137867897fe9e482` |
| `src/llama-mmap.cpp` | `3615c7c7dd351d58222f5f9bc0923221105c3f0f697c065a57f4af5675942606` |
| `src/llama-model.cpp` | `a2c2d032e8416d9745fe237afc85c403f88eee34fe356ca0517e76101dcb3029` |
| `src/models/glm5-next.cpp` | `ab8ddd13fa5c28de863392580f443584fefdeb020976675445496ce655f0f60b` |

Les auxiliaires du diagnostic sont également identifiés :

| Auxiliaire | SHA-256 |
|---|---|
| `state_capture.inc` | `0b21a7ec4f1b58892a14fbe3d73417371afebdd71a07f43d3088a1b14cc5ca4f` |
| `source_capture.inc` | `b7f5e0484def857c7d2e97dff6be874cb71d733648346ea7635236e640fe6f9b` |
| `self_vm.inc` | `f8e352dc6d32db0867e89ba7626717a374f5e3ed8de4847bb231d9a30097eaf9` |

## Preuve numérique et mémoire

Les deux diagnostics utilisent 10 threads CPU, `gpu_layers=0`, un contexte de 512, un batch de 8 et une seule requête de 16 sorties à entrées imposées. A16 utilise la politique CPU. E16 active la sélection quantifiée Metal `quant-tail` sur la tête de sortie et les cinq derniers blocs sélectionnés, les poids mappés lisibles par le CPU, le chemin Q8 respectant l’arithmétique CPU, le préchargement dense anticipé de 32 Mio et le verrouillage dense de 256 Mio. Le cache MoE explicite est nul dans ces captures.

| Contrôle clos | Référence | Logits finis comparés | Erreurs de bits | États observés comparés | RSS maximale échantillonnée (octets) | Footprint maximale échantillonnée (octets) | Croissance maximale du swap système (octets) |
|---|---|---:|---:|---:|---:|---:|---:|
| A16 V4 CPU | A16 V3 `cpu-control16-isolated` | 2 478 080 | 0 | 3 792 | 6 426 984 448 | 428 680 080 | 0 |
| E16 V4 tail5 + lock256 | A16 V4 `v4-cpu-control16` | 2 478 080 | 0 | 3 792 | 6 301 024 256 | 450 027 928 | 0 |

Les champs RSS proviennent exclusivement de `guard.json:peak_sampled_rss_bytes`, avec un intervalle d’échantillonnage de 0,5 s ; ils ne garantissent pas l’absence de pointes entre échantillons. La RSS comprend les pages mappées résidentes du processus. La footprint est un compteur distinct et les deux valeurs ne se somment pas. Les gardes appliqués sont 8,5 Gio de RSS, 32 Mio de croissance du swap et 400 s. Les deux gardes se terminent normalement, sans violation. Zéro croissance observée ne signifie pas zéro swap déjà alloué sur le système.

Les comparaisons A16 V4→A16 V3 et E16→A16 V4 passent intégralement : mêmes tokens d’entrée et de sortie, mêmes clés de tenseurs, 16 vecteurs complets de 154 880 logits, tous finis, erreurs absolue maximale et RMSE nulles. La fidélité E16 vers le témoin V3 est ainsi transitive sur ces seules données capturées.

| Famille capturée, pour chacune des deux comparaisons | Tenseurs/vecteurs comparés | Éléments comparés | Écarts numériques / de bits |
|---|---:|---:|---:|
| Routes MoE | 672 | 7 056 | 0 / 0 |
| Router | 672 | 254 016 | 0 / 0 |
| KDA | 1 088 | 11 698 176 | 0 / 0 |
| `result_output` | 16 | 2 478 080 | 0 / 0 |
| Logits complets | 16 | 2 478 080 | 0 / 0 |

`result_output` et les logits complets sont des vues de preuve qui se recouvrent ; leurs nombres d’éléments ne doivent pas être additionnés comme des données indépendantes.

Chacune des deux comparaisons d’état valide les 3 792 événements ordonnés observés `CPY`/`SET_ROWS`, sans événement divergent : 2 626 682 880 octets logiques hachés, pour 2 672 730 112 octets de portée de stockage déclarée. La portée de stockage peut inclure des espacements ; elle ne désigne pas davantage de contenu logique validé. Cette preuve concerne les écritures observées, et ne certifie ni tous les états internes non capturés ni une génération arbitrairement longue.

## Rétention dense effectivement acquise par E16

Le plan `dense-lock-plan256.json` est accepté avant lancement et après exécution avec la même empreinte `ec8ea4cead78bce65fdf8f43b79cc9d51bf16e3ef3cd9236d097eb3adc6aabf6`. Le journal du runtime et sa validation rapportent :

| Mesure E16 | Valeur |
|---|---:|
| Tenseurs sélectionnés et retenus | 103 |
| Octets bruts des tenseurs sélectionnés | 266 862 592 |
| Taille de page | 16 384 octets |
| Union de pages et octets verrouillés | 268 255 232 |
| Budget | 268 435 456 octets (256 Mio) |
| Pages vérifiées à l’acquisition | 16 373 |
| Appels de verrouillage | 85 |
| Appels de déverrouillage avant le relevé final | 0 |
| Erreur de verrouillage / déverrouillage | 0 / 0 |

La différence entre octets bruts et union de pages vient de l’arrondi et du partage des pages. La vérification établit la résidence à l’acquisition et la conservation des verrous utilisateur jusqu’au relevé final. Elle ne mesure pas des octets NAND évités, un taux de cache global ou un gain de vitesse. Le nombre nul de déverrouillages est celui du relevé final avant destruction du contexte.

## Intégrité et limites

Les deux expériences rapportent des tailles et dates de modification du modèle inchangées. Aucun nouveau hachage intégral des 110 Go n’a été effectué. Le sélecteur dense est lié à un inventaire et à des SHA historiques vérifiés ; son propre rapport précise que device/ctime peuvent différer de l’identité historique. Il ne prouve pas une nouvelle identité complète à cinq champs ni un nouveau SHA intégral du modèle.

Ces contrôles établissent la fidélité du runtime V4 effectivement testé et la rétention dense bornée sur cette trajectoire. Ils ne qualifient pas encore le débit, la stabilité longue, un arrêt EOS, le SSD externe Fanxiang ou une intégration RONIN. Les durées diagnostiques ne sont volontairement pas converties en tokens/s. Aucun test, build ou accès aux poids du modèle n’a été exécuté pour rédiger cette note à partir des métadonnées closes.

## Empreintes des preuves closes

| Artefact | SHA-256 |
|---|---|
| `work/v3/v4-cpu-control16/experiment.json` | `0f30e48a06adcf6921052dba43c6d94bd7cdad063fda20ea002c3f0656cbd032` |
| `work/v3/v4-cpu-control16/comparison.json` | `2325dbf3daa8cef8af4686c486699e69555c836718a490fec082437e03cf8d1c` |
| `work/v3/v4-cpu-control16/guard.json` | `8043d1a5feac136efb0ad6366a784d2c731d804aa4123cddabdc9819b8cfd4d3` |
| `work/v3/v4-tail5-lock256-fidelity16/experiment.json` | `6771947d1c588f61ed2f126ca633358a56155baabd5cca912f4101ee6a1c555a` |
| `work/v3/v4-tail5-lock256-fidelity16/comparison.json` | `1a4219acbc78aec080d5641204f8c9cfd9a32a0b0fec81ccf783d2ca0c18fec1` |
| `work/v3/v4-tail5-lock256-fidelity16/guard.json` | `8b09a97b853a09a8a14d7456df7d03b07383ed90208e01682965a3b8a4558249` |

Les chemins ci-dessus sont relatifs à la racine du workspace de campagne, distincte du dépôt `work/airllm-next`.
