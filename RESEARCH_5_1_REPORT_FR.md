# AirLLM Next 5.1 — résultats de recherche sur MacBook Air M5

24 octobre 2026. **Recherche publiée sur une branche dédiée ; aucune fusion dans main, aucune version stable 5.1 annoncée.**

## Résultat vérifié

Sur le checkpoint Qwen3.6-35B-A3B GGUF authentique de 23,94 Go, le décodage réellement restitué passe de **1.424 à 1.545 token/s (+8.51 %)**, à budgets identiques. Le débit comprenant le processus complet passe de **0.791 à 0.842 token/s (+6.55 %)**. Les six paires sont favorables en décodage et de bout en bout.

Les douze générations atteignent EOS. Entrées, sorties, empreintes des vecteurs complets de logits, routes et positions concordent à chaque pas. Chaque variante produit 372 tokens, dont 366 après le premier token de chaque processus. Aucun ajout de swap, pression élevée ou dépassement échantillonné n'est observé. La vérification indépendante depuis les fichiers bruts retrouve les mêmes résultats sans importer l'analyseur principal.

**Ce résultat concerne Qwen et ce protocole. GLM-5.3 complet n'est pas exécuté ni validé ; une victoire contre Colibrì n'est pas démontrée. Le volume logique de poids lus par token ne baisse pas.**

## 1. État du dépôt et conservation des travaux

La PR #6 est vérifiée sur son commit exact `4ed1a58b460f8b93432e6cabc786c4ce9fc732c7`, séparément des optimisations ultérieures. La branche Throughput `ab59677` est plus avancée pour l'inférence GGUF réelle ; elle sert de référence de débit, réexécutée aujourd'hui. La nouvelle branche `research/airllm-next-5.1-2026-10-24` fusionne les deux historiques, puis corrige les défauts. Les branches E/S, native-GGUF, multi-token et PRISME sont inspectées. L'ancien historique PRISME ne remplace pas le moteur récent.

Commits principaux : `f82a064` intègre PR #6, ownership et lecture directe ; `6625211` ajoute backpressure Metal et protocole contrôlé ; **`cdd426e` est le commit candidat mesuré**. Les preuves figent toutes les sources du moteur et les scripts/tâches réellement exécutés. Les ajouts de rapports et d'outils d'analyse ultérieurs ne changent pas ces sources ; cette égalité est vérifiée par le manifeste des sources livrées. `main` est conservé à `9453681973ddfc93d66c6144c8eca2d5a39a8f1c`.

Le code conserve les chemins historiques et les modèles précédemment pris en charge. Les trois nouvelles options restent facultatives. Cette campagne ne qualifie ni vision, ni MTP, ni toute la gamme de checkpoints.

## 2. Audit réel de la PR #6 et corrections

La suite de la PR #6 exacte termine avec **1 166 tests réussis, 3 ignorés et 11 sous-tests réussis**. Cela ne suffisait pas : sa nouvelle option sans copie échoue sur le vrai checkpoint pendant le préfill, avec `Packed buffer budget exhausted`, alors que le chemin historique termine le même prompt. L'arrêt est conservé comme échec, sans violation du superviseur ni augmentation du swap.

Trois mécanismes sont corrigés :

1. Le finalizer attaché seulement à l'ndarray ne suivait pas une référence conservée au buffer bytes sous-jacent. Un propriétaire de buffer PEP 688, compatible Python 3.12 déjà requis, conserve la charge à travers les vues et memoryviews. Le chemin historique reste writable.
2. `AdaptiveWeightCache.retained_size` créait une closure récursive conservant sa liste d'arrays jusqu'au GC cyclique, même après rejet du cache. Le calcul devient itératif. Des tests avec GC cyclique désactivé prouvent la libération immédiate ; le moteur ne force pas de GC pour masquer le défaut.
3. Les acquisitions/libérations de budget et les erreurs de conversion/allocation sont validées. Les exceptions conservant un traceback gardent aussi la charge des buffers réellement vivants.

La nouvelle suite ciblée reproduit **21 échecs sur 54 cas** avec le code de PR #6. Les corrections passent les cas de possession, plafonds, lectures courtes et concurrence. Les compteurs et les deux échecs du modèle avant correction restent dans l'archive.

L'import DLPack a révélé une autre limite : le dernier buffer peut rester loué par une commande Metal après évaluation. Avant une allocation qui dépasserait le budget, le moteur synchronise puis revérifie ce budget. Un alias encore vivant provoque un refus. Aucun plafond relevé, aucun cache utile vidé et aucune application tierce fermée.

## 3. Algorithme retenu : lire dans l'allocation finale

`gguf_packed_io.read_packed_rows_into` valide sélection, offset, taille maximale, budget E/S et propriétaire avant allocation. Il remplit l'ndarray NumPy final par `preadv`, ou sur ce Mac par `libc.pread` 64 bits. Le verrou du descripteur empêche une fermeture concurrente ; lectures partielles, EOF, EINTR et EAGAIN sont traités avec reprises bornées. Le finalizer est installé dès l'allocation, avant les opérations faillibles.

Ce chemin supprime les deux copies hôtes explicites du chargement historique. Son buffer writable permet l'import DLPack existant ; succès et fallback sont comptés. Les copies de regroupement des experts et le transfert système vers le buffer ne disparaissent pas : il ne s'agit pas d'une promesse de trajet SSD→GPU intégralement sans copie.

Le parsing groupé des tableaux numériques de métadonnées reste borné à 64 Kio par morceau. Il restitue les mêmes valeurs. Le regroupement des experts, les kernels Q4_K/Q6_K, le routage, les poids et l'ordre d'addition numérique sont conservés.

La comparaison principale mesure **la combinaison lecture directe + import DLPack + métadonnées groupées**, pas leurs effets causaux séparés. Les options restent désactivées par défaut ; le preset validé est décrit dans [les commandes de reproduction](research/research51/README.md).

## 4. Campagne sur checkpoint authentique

Machine vérifiée : MacBook Air `Mac17,3`, Apple M5, 16 Gio. Aucun SSD externe physique connecté ; `/Volumes/SSD` est un lien vers `/`. Les mesures concernent le SSD interne. Le fichier de modèle pèse **23 938 321 664 octets**, soit plus que la RAM installée. AirLLM le lit directement ; aucun service Ollama ne produit les réponses.

SHA-256 intégral recalculé en lecture seule **après** les générations : `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`. Les 23,938,321,664 octets, inode, taille et date restent cohérents. Cette vérification complète s'ajoute aux identités de fichier enregistrées pendant la campagne ; elle ne prétend pas avoir recalculé le hash avant chaque token.

Cache 64 Mio, E/S 64 Mio, états/KV 256 Mio, travail 128 Mio, lecture individuelle 16 Mio ; tuiles 8 192, préfill 16, contexte 1 280, LRU, experts groupés, concaténation. Greedy, vrai tokenizer/template/EOS, FP32 et TF32 désactivé. Le pipeline est désactivé. Trois prompts, deux répétitions, ordre opposé par répétition et blocs mélangés avec graine publiée ; douze processus frais.

| Prompt | Décodage référence, tok/s | Candidat, tok/s | TTFT moteur médian réf. / cand., s |
|---|---:|---:|---:|
| explanation | 1.433 | 1.559 | 20.37 / 19.93 |
| long_context | 1.375 | 1.483 | 66.81 / 64.86 |
| short | 1.413 | 1.489 | 13.74 / 12.84 |

Les sommes de durées de décodage sont 257.061 et 236.890 s ; celles des processus supervisés 470.479 et 441.559 s. Le facteur médian apparié de décodage vaut 1.0674×, quartiles 1.0568–1.0800, étendue 1.0524–1.1166. Il diffère du facteur agrégé car les longueurs de réponse pondèrent ce dernier. Le TTFT n'est pas meilleur dans toutes les paires.

Le débit principal inclut restitution et journaux entre tokens. Le débit des pas seuls, distinct, est 1.425 / 1.547 tok/s. Le débit de bout en bout comprend démarrage Python, imports, préparation, génération et supervision. Le TTFT moteur exclut imports, tokenizer et initialisation, mesurés séparément. Les empreintes et journaux perturbent les temps mais sont présents des deux côtés.

### Mémoire et E/S

| Mesure primaire | Référence | Candidat |
|---|---:|---:|
| Pic RSS échantillonné, Mio | 516.12 | 486.73 |
| Pic allocateur MLX, Mio | 167.33 | 167.36 |
| Minimum disponible système, Gio | 4.179 | 4.065 |
| Croissance maximale du swap | 0 | 0 |
| Lectures logiques de décodage, Go/token | 1.964699 | 1.964699 |
| Lectures stockage OS du processus, Mo/token décodé | 191.00 | 196.79 |
| Hits / misses du cache logiciel | 46 / 853010 | 46 / 853010 |

Les octets logiques sont identiques. Aucun gain de trafic physique n'est établi ; le compteur stockage du candidat est même légèrement supérieur ici. Le progrès vient du transport et de l'orchestration, pas d'une réduction démontrée des poids nécessaires. Le cache de fichiers macOS n'est pas purgé ni contrôlé. `ri_diskio_bytesread` est un compteur OS du processus, pas une mesure NAND. RSS, MLX et cache OS se recouvrent ; ils ne s'additionnent pas en consommation totale de mémoire unifiée.

### Mesures de composants et stabilité, séparées

Sur 51 répétitions randomisées chaudes d'un payload synthétique de 1,179,648 octets : copies historiques **57.21 µs**, readonly **26.42 µs**, direct writable **26.67 µs**. Les empreintes d'octets sont identiques. Pic tracemalloc d'une lecture : 2,360,825 contre 1,181,784 octets. Le direct n'est pas établi plus rapide que le readonly ; son intérêt est aussi d'être writable pour DLPack. Ceci n'est pas un benchmark modèle ni SSD froid.

Un profil séparé de la réponse courte observe **23,961 chargements de tuiles d'experts**, 16.292 Go logiques et 5.865 s dans le helper de chargement. Il comprend préfill et génération, cache, ownership, lectures et éventuelle synchronisation ; il ne devient pas du temps SSD pur. Ce profil supplémentaire n'entre pas dans les douze mesures primaires.

Le test à longueur fixe produit **256 tokens**, 1.574 tok/s, TTFT 21.56 s, génération 183.56 s, RSS supervisé 420.25 Mio, aucune croissance du swap. Arrêt volontaire `max_new_tokens` : la réponse est tronquée, sans EOS, et n'est pas une réussite de réponse complète. Trois minutes ne constituent pas une validation thermique de 30 minutes ou quotidienne.

### Portée de la qualité

Les réponses intégrales ne sont pas réécrites : [sorties primaires](research/research51/results/primary-analysis/responses.md). La réponse d'explication comporte 106 mots pour 100 demandés et décrit la possibilité générique du swap ; cela ne décrit pas le chemin sélectif sans croissance de swap observé ici. Le prompt nommé `long_context` ne contient que 233 tokens avec template : il teste un contexte relativement plus long, pas une capacité de très long contexte. Les erreurs et limites identiques restent présentes dans les deux versions. La campagne historique plus large de conversation n'est pas requalifiée par ces tests de débit.

## 5. Cinq axes de recherche : résultats conservés et limites

| Axe | Travail exécuté | Mesure ou décision |
|---|---|---|
| A — cache et prédiction | Correction lifetime ; nouveaux replays LRU/LFU/admission et prédictions causales sur traces historiques dédupliquées | Transition 50,68 % de rappel sur 6 transitions/1 court tour, 947 mauvaises prédictions. Pas de prélecture activée. LFU ou LRU gagne selon le budget abstrait. |
| B — streaming | Lecture directe bornée, owner réel, métadonnées groupées, backpressure et import DLPack | Gain de combinaison +8,51 % en décodage réel ; aucun octet logique économisé par token. |
| C — multi-séquences | Nouveau replay OLMoE teacher-forced, union de catégories ; revue PRISME | Réductions idéales de demandes 1,077× / 1,240× / 1,592× pour 2/4/8 flux de traces. Aucun débit d'inférences simultanées mesuré, aucun support «128 agents» revendiqué. |
| D — quantification | Même checkpoint Q4_K/Q6_K ; vérifications numériques et codecs héritées réexécutées | Aucun nouveau changement de précision ni gain de qualité annoncé. Réduction de taille et vitesse restent distinctes. Compression alternative non activée. |
| E — décodage | Kernels groupés existants conservés ; réanalyse du draft causal historique et des unions d'experts | Aucun nouveau gain spéculatif de génération mesuré. Draft, coût cible et rollback récurrent restent à valider ensemble. |

Le [rapport de réemploi](research/research51/EXPERT_REUSE_FR.md) distingue explicitement nouveaux calculs et anciennes générations. Il corrige les surinterprétations PRISME : une borne d'indépendance n'est pas une obligation de lire le modèle entier, des entropies marginales ne prouvent pas une impossibilité universelle de compression, et le coût du rollback n'est pas négligeable sans mesure.

Le pilote direct sans DLPack n'a pas démontré de bénéfice supplémentaire ; les essais initiaux sont uniques, soumis au cache OS, et le dernier s'est arrêté avant correction. Ils ne sont pas mélangés au résultat primaire. Pipeline, SIMD élargi, prélecture prédictive et spéculation restent désactivés dans le preset retenu. Aucun défaut observé n'est masqué en augmentant mémoire, swap ou durée.

## 6. Colibrì et faisabilité GLM-5.3

L'[audit sourcé des concurrents](research/research51/COMPETITORS_FR.md) inspecte Colibrì `e0db083`, Pulsar SSD `18f9900`, Pulsar Metal `61f571f`, PulsarMLX `2d38885` et llama.cpp `71255fe`. Il identifie les licences, l'antériorité du streaming/cache/prédiction, les voies Apple et les chiffres non reproduits. Aucun code concurrent nouveau n'est copié dans le moteur ; les attributions GGML existantes sont conservées.

La mesure Colibrì citée de 2,24 tok/s concerne GLM-5.2 int4 sur **M5 Max 128 Go avec environ 98 Go de RSS**, un cache chaud et une seule exécution par réglage. Les résultats Flash concernent une autre architecture que GLM-5.3 complet. Ils ne constituent pas un adversaire mesuré sur ce M5 Air 16 Go. Les suffixes «int4» et «Q4» ne garantissent pas les mêmes poids effectifs.

GLM-5.3 complet officiel est bien identifié (`glm_moe_dsa`, révision HF `ba32f8c19c8738e2c57c71798c601d477a7bf844`, environ 756 Go de dépôt). Il n'est pas disponible localement ; le stockage libre est de l'ordre de 52 Gio et aucun externe n'est connecté. Les poids int4 d'un autre moteur demanderaient aussi stockage, licence, format et fidélité vérifiés. Télécharger un checkpoint géant dans ces conditions n'est pas admissible.

Calcul analytique, pas mesure : 75 couches MoE × 8 experts × 3 projections × 6144 × 2048 × 0,5 octet donnent **11 324 620 800 octets d'experts actifs par token en int4 idéal**, sans hit, hors scales, poids partagés, attention et calcul. À FP8 idéal ce volume double. Un cache peut réduire ce trafic ; ni son hit réel GLM ni la bande passante physique froide du stockage présent n'ont été mesurés. Ce calcul explique la priorité des octets évités et de la résidence ; il ne prouve ni une impossibilité générale ni un débit GLM.

**La comparaison strictement identique avec Colibrì est non exécutée : checkpoint/quantification communs et stockage admissible manquent. L'objectif GLM-5.3 complet exploitable sur 16 Go reste ouvert.** Le prochain obstacle vérifiable est d'obtenir un stockage suffisant et un format GLM complet pris en charge, puis de mesurer vrai routage, coût des experts et qualité à mémoire constante. Le gain Qwen ne se transpose pas automatiquement à GLM.

## 7. Validation et livrables

La dernière suite pertinente, avec lectures du vrai GGUF et capture locale historique du routeur activées, donne **1 398 tests réussis, 1 test NVIDIA ignoré et 20 sous-tests réussis**. La capture contenant des poids reste locale. Le journal, le XML JUnit et le superviseur sont archivés. Compilation Python et vérification des diffs complètent les contrôles ; aucune CI distante n'est supposée réussie par défaut.

Garde-fous des modèles : 2 Gio RSS, réserve 3 Gio, croissance swap 128 Mio, 600 s par essai, pression normale, seuil de 32 processus et surveillance 50 ms. Les échantillons ne garantissent pas l'absence de pic entre deux lectures. Toutes les générations primaires, le profil, le test à longueur fixe et le hash ont terminé sans violation. Les échecs PR #6 et du screening initial restent publiés comme tels.

Livraison : code et tests sur la branche, [protocole](research/research51/PROTOCOL_FR.md), [reproduction](research/research51/README.md), [résumé JSON](research/research51/results/primary-analysis/summary.json), [CSV](research/research51/results/primary-analysis/benchmarks.csv), [vérification indépendante](research/research51/results/independent-primary-final.json), réponses complètes, notes d'audit et archive brute avec manifeste. Les poids, captures binaires privées et secrets sont exclus. Aucun push sur main ni fusion automatique.

La restauration a vérifié les **105 fichiers bruts** (2 331 290 octets) dans un nouveau dossier. Une nouvelle analyse des preuves restaurées retrouve exactement les agrégats et les six paires. Le manifeste et le résultat de restauration sont livrés.
