# GLM-5.3 Flash sur M5 : calcul quantifié fidèle et limites du streaming

Rapport d'ingénierie du 2 novembre 2026. Branche expérimentale `research/glm53-m5-faithful-kernels`, issue de la PR #18 (`7898eecd93dfbde4a823b0df034a04a5b2c5c688`). Runtime llama.cpp épinglé à `3f8af15a76ac8675812e856da45454bafc90f12c`. `main`, le backend de production RONIN, ses **600 identités logiques et son slot physique unique** ne sont pas modifiés.

Le runtime sélectif chronométré a reproduit bit à bit les logits, routes ordonnées, routeurs et sorties KDA capturés sur 16 étapes du modèle complet. Il exécute des produits Q8_0 sur Metal, avec l'arithmétique temporaire et l'ordre d'accumulation ARM de la référence. Ce résultat corrige la tête auparavant non fidèle ; il ne qualifie pas le chemin hybride général GPU6, toujours expérimental. Le bilan de débit figure dans la section campagne ci-dessous. Aucun palier de 0,5 / 1 / 3 / 5 tok/s n'est considéré comme acquis.

## Environnement et identité des mesures

MacBook Air M5, GPU huit cœurs, mémoire unifiée 16 Gio, macOS 27.2 ; SDK sélectionné 27.0. Modèle original GLM-5.3 Flash GGUF Q2_K_S : deux shards totalisant **110 027 352 864 octets**, dont le second contient les poids. Le premier shard de 9 430 112 octets ne représente pas leur volume. Les fichiers sont sur le **SSD interne APFS**.

Le Fanxiang PS1008 n'était pas monté. Le chemin `/Volumes/SSD` rencontré était un répertoire du volume système interne ; son nom ne constitue pas une preuve de disque externe. Tous les résultats de stockage sont internes. Aucun poids, aucun format GGUF et aucune quantification persistante n'ont été modifiés. L'identité des exécutions est vérifiée par tailles et dates de modification des shards, empreintes des octets de composants sélectionnés et des bibliothèques exécutées ; aucun nouveau hash intégral de 110 Go n'est revendiqué.

Les comparaisons utilisent CPU10/GPU0 comme référence. Les anciens 0,1085–0,1664 tok/s restent des observations historiques, avec d'autres conditions ; le gain du candidat doit être calculé contre les contrôles de sa propre campagne. Le système possédait déjà du swap. « Croissance nulle » signifie un delta observé nul pendant une exécution, pas une quantité absolue nulle.

## Origine et correction des divergences

La PR #18 avait déjà isolé la désactivation globale de la fusion mHC : une opération mal placée à la frontière CPU/GPU changeait aussi le graphe des premières couches CPU. Son correctif local rétablit le préfixe CPU bit à bit, mais GPU6 restait rejeté : maximum de logits **0,12101674**, huit positions de routes ordonnées différentes. L'ancien résultat PR #17 atteignait 0,5348455906. Garder le même ensemble d'experts ou les mêmes argmax ne suffit pas.

Les nouvelles captures détaillées de couche 40 montrent :

| Observation sur le préfill de six tokens | Écart absolu maximal | Interprétation |
|---|---:|---|
| `hc_flat-40` avant normalisation | 0 | Entrée capturée identique |
| `hc_flat_norm-40`, RMS_NORM | 9,536743e-7 | Premier écart en bits parmi les sorties nommées inspectées |
| `kda_q_proj-40`, MUL_MAT | 0,01267658919 | Première violation des tolérances ; 31 614 / 49 152 éléments hors seuil |

La source CPU somme en FP64 les carrés arrondis FP32 ; Metal emploie une réduction parallèle FP32, puis une autre séquence de racine/réciproque. La projection standard ne reproduit pas non plus la quantification Q8 temporaire des activations CPU et son dispatch ARM `nrc=2` pour le préfill. Ces différences sont établies dans les sources ; l'expérience ne répartit pas quantitativement l'erreur finale entre elles. Voir [analyse des traces](TRACE_FIDELITY.md).

Le nouveau mode garde la normalisation, mHC, routage, récurrences/convolutions/états KDA et tous les produits multi-colonnes sur CPU. Les seules opérations GPU admises sont les matvecs Q8_0 mono-colonne éligibles sur des poids déjà Metal, y compris certaines projections Q/K/V au décodage. Une marche finale du graphe couvre aussi les nœuds sans nom et les copies d'état. Les graphes de réservation et d'inférence reçoivent la même politique. Une configuration incompatible échoue explicitement.

Le kernel compatible reproduit le réciproque de l'échelle FP32 non arrondie, l'arrondi au plus proche pair des entiers, l'échelle temporaire FP16, les huit chaînes FMA pair/impair et leur réduction ARM. Les poids Q8_0 restent leurs octets originaux. Les options de calcul sûr sont activées uniquement avec le réglage expérimental. [Contrat et preuves Q8](Q8_FIDELITY.md).

## Placement et mémoire

Le meilleur placement étudié sélectionne **82 matrices, 2 592 538 624 octets** de poids Q8_0 des douze dernières couches et de la tête, sous un plafond statique de 3 Gio. Ce nombre ne borne pas toute la mémoire du processus.

Le premier tail5 a dépassé la limite de croissance du swap : le scheduler copiait les poids Metal vers un workspace CPU de **677,50 Mio** pour le préfill. Cette preuve négative est conservée. Un accès CPU direct aux poids mappés originaux a ensuite réduit ce workspace à **11,0011 Mio**. Le correctif n'annonce pas tous les buffers Metal comme hôtes. Il exige simultanément le mode sélectif, un poids Q8 constant, un buffer `MTL0_Mapped`, un alias dans les bornes, et des pages VM lisibles/non inscriptibles. Les activations et états mutables suivent les synchronisations originales. [Garde et fixture réelle](MAPPED_CPU_READ.md).

Les mesures modèle restent surveillées : RSS maximal admis **8,5 Gio**, croissance du swap **32 Mio**, arrêt sur pression critique, erreur de surveillance ou délai dépassé (400 s pour les diagnostics courts, 900 s pour les diagnostics au-delà de 16 sorties et les campagnes). Les benchmarks serveur imposent en plus un minimum de 256 Mio de mémoire disponible ; ce seuil supplémentaire n'est pas implémenté dans le superviseur natif des captures. Aucune limite mémoire n'a été relevée pour faire passer un candidat. Les expérimentations prennent le verrou unique existant ; les processus possédés sont arrêtés et nettoyés sans toucher aux autres applications.

## Fidélité établie

Tolérances historiques inchangées : flottants Metal `atol=0,001`, `rtol=0,0001`, valeurs finies, routes ordonnées et IDs exacts. Les nouveaux contrôles rapportés ci-dessous font mieux sur leur trajectoire : **zéro différence en bits**.

| Candidat | Étapes | Logits comparés | Autres sorties capturées | Résultat |
|---|---:|---:|---|---|
| Tête Q8 inline compatible | 4 puis 16 | 619 520 puis 2 478 080 | Routes, routeurs, KDA | Exact |
| Tête Q8 activation partagée | 4 | 619 520 | Mêmes familles | Exact, pas de gain modèle établi |
| Tail5 avec mapping sûr | 4 | 619 520 | Mêmes familles | Exact |
| Tail12 avec mapping sûr | 4 puis 16 | 619 520 puis 2 478 080 | Mêmes familles | Exact |
| Tail12, runtime chronométré avec timestamps, avant durcissement final des alias | 16 | 2 478 080 | 7 056 éléments de routes, 254 016 éléments de routeurs, 11 698 176 éléments KDA | Exact, RSS 6 354 485 248 octets, croissance du swap nulle |

Les captures suivent les mêmes entrées forcées que la référence ; elles ne remplacent pas des erreurs par des tokens de référence après comparaison. Les IDs générés sont aussi comparés. Les 89,385 secondes du dernier diagnostic complet comprennent chargement et sérialisation : **aucun débit n'en est déduit**. Cette preuve porte sur les tenseurs capturés, cette trajectoire et les empreintes natives enregistrées avant le durcissement final des alias ; elle n'affirme pas la parité du runtime final durci, de tout état récurrent non capturé, de tous les prompts ou des longs contextes.

## Campagne de débit complet

**Aucun cycle ABBA complet admissible ; aucun gain de décodage reproductible qualifié.** Les paliers de 0,5, 1, 3 et 5 tok/s ne sont pas atteints par une mesure acceptée.

| Tentative courte, limite 16 sorties | A0 CPU tok/s | B1 GPU tok/s | B2 GPU tok/s | A3 CPU | Décision |
|---|---:|---:|---:|---|---|
| 1 | 0,169122 | 0,194499 | Non lancé | Non lancé | Traces GPU masquées par le niveau INFO ; pilote rejeté |
| 2, sortie stderr réparée | 0,177137 | 0,196916 | 0,199422 | Arrêt au chargement | Croissance swap système 46 399 488 octets ; cycle rejeté |
| 3, nouvelle tentative complète | 0,170475 | 0,156443 | Arrêt après première sortie | Non lancé | Croissance swap système 46 530 560 octets ; cycle rejeté |

Ces débits sont les passages terminés, **pas des bras ABBA qualifiés**. Tous leurs seize IDs concordent avec la référence ; cela ne remplace pas la preuve complète de fidélité séparée. La tentative 2 semblait favorable, mais la tentative 3 donne au contraire −8,23 % sur la paire terminée. Choisir seulement la meilleure paire serait trompeur. Les comparaisons automatiques restent `comparison_valid_on_stated_scope=false` et `optimization_qualified=false`.

Les TTFT A/B1 étaient respectivement 21,409/18,769 s, 20,332/18,632 s et 23,473/21,203 s. Ces observations de préfill/latence ne démontrent pas un gain soutenu de décodage. Les passages admissibles courts n'ont pas fait croître le swap échantillonné, mais les cycles complets ont échoué aux garde-fous système. Leurs RSS échantillonnés restent en dessous du plafond ; la croissance globale du swap n'est pas attribuable avec certitude au seul modèle.

Après le durcissement des alias, un diagnostic CPU de 64 sorties demandé a produit **18 sorties puis s'est arrêté**, à 238,243 s, pour croissance du swap système **161 480 704 octets** (154 Mio). Pic RSS 5 800 722 432 octets. Il n'a pas de marqueur complet et n'est pas un benchmark de débit : capture et sérialisation y sont actives. La comparaison longue GPU/ABBA n'a pas été lancée après ces échecs répétés. Aucun EOS ni stabilité longue n'est qualifié. Les limites de 8,5 Gio / 32 Mio sont conservées.

Les mesures GPU disponibles sont réelles : par exemple, le premier B de la tentative 2 contient 2 121 command buffers terminés, union d'intervalles de 1,25355 s sur la vie du serveur. Cette somme ne compte pas toutes les attentes/transferts côté hôte et n'est pas le temps de décodage. Elle confirme l'exécution GPU, sans convertir l'utilisation matérielle en accélération modèle. Voir [consolidation de toutes les tentatives](evidence/campaign-consolidation.json) et [manifest des preuves brutes](evidence/selection-manifest-v2.json).


Protocole : serveurs neufs, ordre A/B/B/A, CPU10 dans les deux bras, contexte 512, batch/microbatch 8, prompt `La capitale de la France est` (six tokens), température 0, seed 42, réutilisation KV désactivée, EOS respecté, limite de sortie explicitement indiquée. A est CPU10/GPU0 ; B est tail12 compatible sans cache MoE GPU. Le débit de décodage observé vaut `(N−1)/(instant dernier token−instant premier token)` ; TTFT et chargement sont distincts. Les événements SSE, IDs, temps entre tokens, compteurs processus, mémoire et timestamps des command buffers sont sauvegardés.

L'état du cache de fichiers macOS et la charge d'arrière-plan ne sont pas contrôlés. ABBA limite certains biais sans les supprimer. Deux mesures par bras ne donnent pas une estimation fiable de variance générale ; aucune significativité ou généralisation n'est revendiquée. Les timestamps GPU proviennent de command buffers appartenant au moteur ; le pourcentage GPU IORegistry décrit tout le système.

Matrice demandée : B, GPU corrigé sans nouveau cache, et E, meilleure combinaison fidèle, correspondent ici au tail12 compatible ; D, kernels Q2 optimisés, est comparé en composant et reste CPU dans le moteur ; C, cache adaptatif plus GPU, n'a pas de qualification de débit complet, car le nouveau cache Python n'est pas intégré au graphe natif. Le cache natif GPU6 de la PR #18 reste attaché à un chemin numériquement rejeté. Aucun chiffre C n'est fabriqué à partir d'un replay.

## Q2_K, MLX et TensorOps réellement exécutés

Neuf variantes fusionnent déquantification et produit Q2_K × Q8_K temporaire : lignes groupées 2/4, sous-groupes de blocs, deux passes entières, conservation des octets quantifiés, réutilisation de l'activation et lancement groupé des huit experts dans leur ordre original. Le test principal utilise les vrais octets gate couche 40, l'activation CPU capturée et les routes `[141,183,116,112,154,156,21,98]`. Toutes les **16 384 sorties** par variante sont identiques au CPU ARM dans MLX et Metal autonome.

| Mesure préparée, huit experts | Référence GPU serial32 | Meilleure variante | CPU10 |
|---|---:|---:|---:|
| Temps GPU command buffer | 0,551 ms | 0,488 ms, rowtile4 | Sans objet |
| Temps mur MLX synchronisé | 0,737 ms | 0,671 ms, cachedbytes_tile4 | 0,664 ms |

Sept répétitions dans un même processus, ordre aléatoire conservé, quatre chauffes exclues. Le temps GPU baisse d'environ 11,5 % dans ce composant ; le temps complet MLX ne bat pas CPU10. Préparation/lecture/import et quantification de la nouvelle activation sont séparés, et doivent être payés à l'intégration. Pas de matrice de poids FP32 complète, payload connu maximal 46 563 840 octets. [Détails et reproduction](Q2_KERNELS.md). L'ancien 1,309 ms GPU concernait une autre expérience ; ce n'est pas un avant/après comparable à ce nouveau groupe réel de huit experts.

Une sonde Metal 4 TensorOps **entière int8×int8→int32** a réellement compilé puis exécuté trois fixtures : **384 résultats exacts** contre un oracle entier indépendant. Elle réserve 9 312 octets de payload connu et n'a pas fait croître le swap observé. La première erreur de type `char`, corrigée avec `int8_t`, reste archivée. Ce résultat ne mesure ni le débit GLM ni l'utilisation des Neural Accelerators. Les kernels Q8/Q2 intégrés/MLX compatibles actuels utilisent les ALU SIMD et FMA, pas TensorOps. [Contrat du SDK, limites et sonde](TENSOROPS.md).

## Cache MoE borné et préchargement

Le nouveau cache conserve les octets originaux, réserve ensemble résidents, éléments en vol et leases actifs, réduit sa capacité selon mémoire/pression et conserve le budget pendant toute lecture asynchrone. Le consommateur suit l'ordre des routes effectivement calculées ; les IDs du modèle ne sont jamais changés. Les plages strictement adjacentes peuvent être regroupées sans lire les trous. Les buffers résidents sont réutilisés ; un second pool de buffers non comptabilisé n'est pas créé.

Sur les 3 024 IDs capturés, à 128 Mio, le replay **métadonnées seulement** donne LRU 704 hits, SLRU 657, LFU 562 ; LRU reste le défaut pour continuité, sans preuve qu’il soit optimal au décodage. La ventilation par phase montre que ses **704 hits sont tous au préfill**, puis **0 hit / 1 008 misses au décodage** ; SLRU fait 15 / 993 au décodage, soit seulement 1,49 % de hits sur ce court replay. Ce n'est pas une amélioration du précédent 35 % à 384 Mio : capacités et périmètres diffèrent. Le vrai test de deux tokens couche 40 réutilise six buffers pour six experts communs, lit 26,25 Mio gate au total et libère toutes les réservations après réduction à zéro. Le test réel des quatre politiques sur douze demandes distinctes a zéro hit dans les quatre cas. Aucun gain de token/s ni chevauchement I/O/calcul n'est déduit. [API et preuves](IO_CACHE.md).

Une alternative native CPU d'abandon des pages après usage (`MADV_DONTNEED`) a été implémentée, bornée et testée sur fixture réelle. Le test GLM complet a été interrompu après 269,329 s avec seulement deux sorties terminées. Il n'a ni marqueur de fin ni preuve de fidélité complète ; il est rejeté comme candidat de performance. Le profil ponctuel montre entre autres des calculs BF16, attentes workers, conseils mémoire et vérifications Mach VM ; il ne permet pas d'attribuer tout le ralentissement aux seules barrières. [Preuve négative](EXPERT_DROP.md).

## Diagnostic quantitatif : stockage, calcul et synchronisation

Le profil CPU16 sérialisé compte 118 608 opérations, 105,019 s de temps mur d'opération et 701,899 s CPU sommées entre threads. Sur ses quinze étapes de décodage : **91,115 s mur, 677,178 s CPU et 183,800 Go de lectures OS**, soit **12,253 Go/token**. Le ratio CPU/mur de 7,43 n'est pas une fraction « calcul pur » ; des workers peuvent tourner ou attendre pendant les défauts de page.

La première version du profil refusait les buffers `CPU_Mapped` sans device enregistré : ses données de résidence sont **inconnues**, pas nulles au sens zéro. La correction a été validée sur un fichier `PROT_NONE` puis dans un vrai tenseur GGML à IDs stridés/dédupliqués. Le profil CPU4 corrigé inspecte 2 430 multiplications de poids au décodage : **36,657 Go logiques pour trois étapes**, soit **12,219 Go/token**, et 36,760 Go OS. Parmi les octets logiques, les produits denses représentent **9,444 Go/token (77,3 %)**, les experts sélectionnés **2,775 Go/token (22,7 %)**. Le cache MoE seul ne cible donc pas la majorité du volume demandé. Seulement **0,143 %** des snapshots de pages inspectées sont marqués résidents à cet instant. Il s'agit de snapshots répétés par opération, pas de pages uniques ni du taux global de hits du cache de fichiers. L'inspection coûte 0,725 s, séparées des 18,841 s d'opérations de décodage. [Profil, données par couche/opération/token et limites](PROFILE.md).

Le compteur contrôleur système du profil16 rapporte 204,631 Go de lectures pendant 106,652 s, environ **1,92 Go/s**. Il englobe le chargement, le préfill et toute activité système ; ce n'est ni un compteur NAND ni une attribution exacte au seul processus. Le test indépendant de 1 Gio avec `F_NOCACHE` lit les octets logiques à 6,9 Go/s mais ne provoque que 866 390 016 octets au compteur contrôleur : l'état du cache n'est pas froid. **6,9 Go/s n'est donc pas une vitesse SSD démontrée.**

Ces observations établissent une forte pression de streaming des poids, en parallèle d'un travail CPU élevé. Elles ne donnent pas une décomposition exacte en pourcentages I/O, calcul et synchronisation. Il manque notamment une mesure exhaustive séparant les stalls de défaut de page, toutes les copies/allocations scheduler et les attentes CPU/GPU. Les command buffers GPU du candidat sont mesurés, mais ne couvrent pas tout le temps hôte et tous les transferts.

À titre de budget conditionnel, avec un trafic de 12,2 Go/token et un service effectif autour de 1,9 Go/s, l'I/O seule consommerait environ 6,4 s/token. Atteindre 0,5 tok/s au même débit demanderait de ramener les lectures à environ 3,8 Go/token, ou de changer le service/chevauchement. Ce calcul explique la priorité à la réutilisation réellement résidente et aux poids denses épinglés ; **ce n'est pas une borne physique universelle**, car les compteurs proviennent de périmètres distincts et les accès peuvent se chevaucher. Aucun chiffre Fanxiang n'est extrapolé.

## Reproduction et configuration RONIN

Construire un runtime isolé sur Apple Silicon, sans modifier le checkout source ni charger le modèle :

```sh
python -m research.glm53_gpu_v2.build_runtime \
  --source /absolute/path/to/pinned-llama-checkout \
  --destination /absolute/path/to/new-empty-runtime --jobs 3
```

Les treize patches sont appliqués dans l'ordre déclaré et consignés avec leurs empreintes. Les bibliothèques, outils, SDK, commandes et logs sont conservés. Le helper de recette ne garantit pas des Mach-O identiques entre compilations ; une nouvelle construction doit repasser la fidélité et fournir ses propres empreintes avant campagne.

```sh
python -m research.glm53_gpu_v2.run_native \
  --model /absolute/path/to/first-shard.gguf \
  --probe /absolute/path/to/runtime/bin/glm-native-fidelity \
  --runtime-libraries /absolute/path/to/runtime/bin \
  --output /absolute/path/to/new-fidelity \
  --reference /absolute/path/to/complete-cpu-capture --tokens 16 \
  --policy quant-tail --tail-layers 12 --mapped-cpu-weights --metal-timing

python -m research.glm53_gpu_v2.campaign \
  --binary /absolute/path/to/runtime/bin/llama-server \
  --model /absolute/path/to/first-shard.gguf \
  --output /absolute/path/to/new-abba \
  --candidate-fidelity /absolute/path/to/new-fidelity/experiment.json \
  --candidate quant-tail --tail-layers 12 --tokens 16 --cycles 1
```

Le pilote refuse un répertoire existant, une identité de modèle changée, un diagnostic incomplet/non exact, des bibliothèques différentes, une absence de commandes GPU observées ou une défaillance des garde-fous. `comparison_valid_on_stated_scope` et `optimization_qualified` sont distincts ; le second n'est jamais activé automatiquement par la seule égalité des tokens chronométrés.

Le durcissement final ajoute uniquement le contrôle des flags de la feuille à la limite de seize alias. Il est compilé et passe la fixture matérielle ; les preuves complètes 16 étapes et les chronométrages précédents identifient leurs bibliothèques antérieures par SHA-256. Le contrôle CPU64 post-durcissement est interrompu par le garde-fou système ; aucune nouvelle qualification complète du runtime durci ou de son débit n’est déduite de la seule identité des sources arithmétiques.

**Configuration recommandée pour RONIN livré : CPU10/GPU0 existant, 600 agents logiques, un slot physique partagé/adaptatif, garde-fous existants.** Le candidat reste sur la branche de recherche ; il n'est pas installé dans l'application. L'essai tail12 exige le mode Q8 compatible, la politique complète de placement et l'accès mappé sûr ensemble. Activer seulement `gpu_layers` ou le cache GPU6 ne donne pas ce chemin fidèle. Cache adaptatif Python, DROP et TensorOps restent désactivés dans le backend livré.

## Tests, preuves et limites restantes

**152 tests et 22 sous-tests réussis**, sans échec ni skip dans la sélection finale, en 3,27 s. Elle couvre les nouveaux modules, les expériences GPU précédentes, les garde-fous/comparateurs GLM et l'intégration RONIN. Les tests Q8 et neuf variantes Q2 exécutent Metal avec un oracle natif indépendant ; les fixtures de mapping, pages protégées et abandon mémoire sont compilées/exécutées sur ce Mac. La garde d'alias durcie passe 20 contrôles réels. Les tests de campagne/lifecycle vérifient notamment rejet des hashes divergents, étapes rejetées, annulation possédée et croissance finale du swap.

Les tests qui avaient des noms non standards ont été renommés `test_*.py` pour être découverts par défaut. Le premier passage de 120 tests n'incluait pas ces 28 cas ; le passage final les inclut, ainsi que le nouveau test de ventilation par phase et les trois contrôles supplémentaires de fermeture forcée/annulation. Aucun résultat ignoré n'est présenté comme exécuté.

Les treize patches s'appliquent au commit épinglé dans un checkout isolé, sans erreur de whitespace. Les neuf sources natives obtenues correspondent exactement au source compilé après durcissement : diff SHA-256 `39b1a83e1a528bdcf9aec2d05d7375be5c5b6bdc6210c77a314cf1c410540be1`. La recette et la construction manuelle effectivement mesurée diffèrent par quelques choix explicites de compilateur/SDK/rpath ; la parité des sources ne constitue pas une preuve de binaires identiques. [Vérification et flags](evidence/patch-series-validation.json), [empreintes du runtime final](evidence/runtime-final-fingerprint.json).

Reproduction de la sélection, avec MLX/NumPy/psutil/pytest et le runtime épinglé disponibles :

```sh
AIRLLM_GGML_LIBRARIES=/absolute/path/runtime/bin \
AIRLLM_TEST_METAL_MAPPED=1 \
AIRLLM_EXPERT_DROP_FIXTURE=/absolute/path/expert-drop-fixture \
python -m pytest -q research/glm53_gpu_v2 research/glm53_gpu \
  research/glm53_m5/test_compare_fidelity.py \
  research/glm53_m5/test_run_guarded_probe.py \
  research/glm53_m5/test_benchmark_two_slots.py integrations/ronin/tests
```


Les rapports sélectionnés, logs de construction/diagnostic/campagne, échecs et empreintes sont dans `evidence/`. La sélection publique masque les chemins personnels et conserve les hashes des originaux ; l'archive locale comprend les captures binaires et opérations détaillées, sans aucun poids du modèle. Les tests matériels ne sont pas présentés comme exécutés sur CI. Les résultats appartiennent au M5 accessible, au checkpoint et aux trajectoires indiqués. Les paliers de débit, le Fanxiang, les contextes longs, l'ensemble des états récurrents et le déploiement RONIN restent limités par les preuves effectivement produites.
