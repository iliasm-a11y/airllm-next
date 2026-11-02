# GLM-5.3-Flash Q2_K_S sur Apple M5 : intégration et résultats mesurés

L’intégration RONIN avec runtime natif et moniteur entièrement embarqués est fonctionnelle sur deux générations de quatre tokens avec moteur réutilisé, puis sur une génération de 32 tokens lancée et consultée dans l’interface réelle. **La stabilité prolongée et les réponses jusqu’à EOS restent ouvertes** : un essai Low plus long a été arrêté par la garde de swap. Sur les benchmarks natifs distincts, la configuration CPU retenue atteint **0,1506 à 0,1664 token/s** sur les deux réponses courtes de 16 tokens, contre **0,0671 à 0,0696 token/s** pour la référence : **2,318 fois le débit médian**. Une seconde tâche de 32 tokens donne un rapport de 2,175 entre la médiane optimisée et l’unique référence disponible. Ces résultats qualifient la meilleure configuration testée dans cette campagne, pas une vitesse maximale universelle. Les objectifs de **1, 3, 5 et 10 tokens/s ne sont pas atteints**.

## Modèle, matériel et code livré

La cible reste exclusivement `ggml-org/GLM-5.3-Flash-GGUF`, révision `119034bc3d9d76eb41f91508e6f9fb19fde42672`, deux fragments Q2_K_S totalisant 110 027 352 864 octets. Aucun modèle de substitution ni nouvelle quantification n’intervient. Machine mesurée : MacBook Air Apple M5, identifiant Mac17,3, 17 179 869 184 octets de mémoire unifiée, macOS 27.2, build 26B118.

L’inspection du GGUF confirme 45 blocs : 34 KDA, 11 DSA, connexions mHC, trois blocs denses puis 42 blocs MoE avec huit experts sélectionnés parmi 288 et un expert partagé. Le format commercial Q2_K_S contient aussi Q8_0, BF16 et F32. Le GGUF ne contient ni poids MTP/NextN ni projecteur vision utilisables ; aucune capacité correspondante n’est annoncée.

La branche dédiée est `research/glm53-ronin-m5`, construite à partir de la PR14 sans modifier ses preuves historiques ni fusionner dans `main`. Publication : [Code et preuves](https://github.com/iliasm-a11y/airllm-next/tree/research/glm53-ronin-m5). Les changements comprennent :

- `air_llm/airllm/nextgen/glm_native.py` et le backend `next-glm-native` : serveur natif persistant, tokenizer/template du GGUF, événements et IDs natifs, annulation, contrôle des fichiers et file d’attente bornée.
- Quatre patches privés sur llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c` : suppression optionnelle du préchargement global initial, conseils de préchargement des experts déjà sélectionnés et des matrices Q8_0, extension expérimentale bornée du nombre de colonnes. Aucun changement arithmétique ni choix de routeur n’est ajouté.
- `integrations/ronin/` et `research/glm53_m5/` : application macOS séparée, backend embarqué, reconstruction isolée du runtime, mesures surveillées et comparateur numérique. Les optimisations fusionnées déjà présentes dans llama.cpp ne sont pas créditées à cette contribution.

La configuration retenue emploie 10 threads CPU, `mmap`, batch/ubatch 8, un slot physique, GPU désactivé, et les trois options de préchargement activées ; limite de colonnes denses conservée à 1. Le modèle reste sur stockage, avec pages mappées récupérables par macOS. La construction finale possède un manifeste vérifiant l’exécutable **et ses bibliothèques**, pas uniquement son petit lanceur.

## Benchmarks avant/après

Les processus démarrent à froid ; l’état du cache de fichiers macOS n’est pas contrôlé. Les répétitions partagent le processus avec `cache_prompt=false`. Température 0, seed 42, même GGUF, aucune spéculation. Le décodage observé exclut le premier token et repose sur les arrivées des tokens suivants. Le TTFT ci-dessous exclut le chargement du processus, présenté séparément.

| Configuration | Capacité contexte ; entrée → sortie | Essais | Chargement, s | TTFT, s | Préfill, tokens/s | Décodage, tokens/s | Requête complète, s |
|---|---|---:|---:|---:|---:|---:|---:|
| Référence CPU4 | 512 ; 6 → 16 | 2 | 57,33 | 36,25–36,77 | 0,1636–0,1655 | 0,06712–0,06965 | 251,61–260,26 |
| Final CPU10 | 512 ; 6 → 16 | 2 | 9,30 | 20,33–21,46 | 0,2815–0,2953 | 0,15061–0,16642 | 110,46–121,06 |
| Référence CPU4 | 1 024 ; 15 → 32 | 1 | 57,46 | 67,83 | 0,2211 | 0,05693 | 612,39 |
| Final CPU10 | 1 024 ; 15 → 32 | 2 | 12,94 | 53,06–56,12 | 0,2690–0,2827 | 0,10850–0,13916 | 278,89–338,77 |

**1 024 désigne la capacité configurée : l’entrée ne comporte que 15 tokens. Cette campagne ne qualifie pas un prompt de 1 024 tokens.** Les réponses atteignent leur budget sans EOS. Les petits effectifs, l’ordre des essais et le cache OS empêchent une estimation robuste de variabilité ou une attribution indépendante de chaque gain.

Sources : `cpu4-native-baseline.json`, `final-cpu10-16x2.json`, `long-reference-cpu4-32.json`, `long-final-cpu10-32x2.json`. Les pilotes séparés demand/expert/dense, les combinaisons et les variantes 4/6/8/10 threads sont conservés ; les pilotes isolés ne sont pas des répétitions qualifiées de toute la campagne.

## Mémoire, CPU et I/O

Valeurs en Go décimaux ; RSS et empreinte physique attribuée sont deux compteurs distincts.

| Campagne | Pic RSS | Pic empreinte | Mémoire disponible minimale | Croissance swap système |
|---|---:|---:|---:|---:|
| Référence courte | 6,412 Go | 0,888 Go | 3,567 Go | 2 949 120 octets |
| Final courte | 5,856 Go | 0,921 Go | 3,578 Go | 0 |
| Référence 32 tokens | 5,618 Go | 0,781 Go | 2,575 Go | 0 |
| Final 32 tokens | 5,590 Go | 1,240 Go | 3,096 Go | 0 |

Les compteurs CPU Mach sont convertis avec la base de temps mesurée 125/3 ns par tick ; les anciens champs de référence nommés `user_ns/system_ns` contiennent des ticks bruts. Les échantillons donnent environ 6,40 cœurs occupés en moyenne sur la campagne courte finale et 6,62 sur celle de 32 tokens. Ces observations ne constituent pas des temps de kernels individuels.

Trois freins restent documentés : pagination/I/O des poids non résidents ; calcul et déquantification des projections denses et experts ; copies/synchronisations du chemin hybride. La référence courte comptabilise environ 215,21 Go de lectures OS par requête, contre 214,79 Go pour la configuration finale, avec environ 13,2 millions de pageins dans les deux cas. Le gain ne repose donc pas sur une disparition comparable de ce volume comptabilisé.

Ces lectures `libproc` incluent pagination et read-ahead. **Les lectures logiques GGUF et les lectures physiques NAND ne sont pas mesurées.** Aucun plafond SSD n’est déduit. L’utilisation GPU IORegistry est globale au système et inclut le bureau. Les temps séparés de déquantification, routeur, synchronisation et allocation restent non attribués. Les sondes disponibles ne signalent pas d’alerte thermique ; elles ne démontrent pas l’absence de toute limitation thermique.

L’admission remplace le seuil forfaitaire de 6 Gio par réserve, espace de travail et budget GPU explicite. Les gardes RSS 9 Gio, swap +64 Mio et pression critique restent actives. La pression d’avertissement 2 est contextualisée : le démarrage reste refusé, mais un moteur chargé peut continuer avec marge disponible et swap borné. Les valeurs et décisions sont exposées dans `pressure_context`.

## Fidélité et expériences rejetées

La comparaison CPU finale passe **bit à bit** sur quatre tokens : 619 520 logits, 3 024 éléments de routes ordonnées, 108 864 valeurs de routeur et 5 013 504 éléments des sorties capturées `kda_scan_out`, vues comprises et strides respectés. Les états internes récurrents complets ne sont pas comparés. Le diagnostic instrumenté est séparé des benchmarks. Cette preuve concerne la séquence capturée et la référence native du même GGUF ; ce n’est ni une preuve générale sur toutes les sorties ni un oracle du checkpoint FP8 original.

L’hybride Metal à six couches, experts CPU et cache de 256 Mio, échoue sous préfixes forcés identiques : écart maximal de logits **0,5348455906**, 364 éléments de routes différents. Les quatre tokens restent identiques, ce qui ne suffit pas. Son cache rapporte zéro hit, 120 misses et 945 Mio chargés. Aucun chemin GPU n’est qualifié ; `GPU_FIDELITY_UNQUALIFIED` bloque son activation sans activation expérimentale explicite. Ces écarts ne permettent pas une conclusion générale sur tous les calculs Metal.

Le pilote deux slots augmente le débit utile de **49,98 %** : 0,09159 → 0,13737 token/s sur l’ensemble des deux tâches. Mais la seconde sortie et les top-8 divergent ; les deux contrôles stricts échouent. L’expérience est rejetée pour la livraison. L’ordre séquentiel puis concurrent et le cache OS sont aussi des facteurs confondants. RONIN conserve **un moteur, un slot physique et 600 identités logiques**, pas 600 inférences simultanées.

## Validation RONIN et livraison

Le trajet vérifié de 128 tokens utilise le **backend Python extrait de l’application, avec le runtime natif et ses bibliothèques encore externes** dans `outputs/native-runtime/`. Il produit **128 tokens en 892,878 s**, avec un décodage natif de **0,152175 token/s**, manifeste runtime vérifié et 600 agents enregistrés. L’arrêt est `limit`, pas EOS : cela vérifie le trajet RONIN → moteur → stockage des preuves, pas une réponse achevée à « OK », ni la stabilité du paquet final entièrement embarqué.

Deux tentatives précédentes avaient été arrêtées par `MEMORY_PRESSURE_SWAP`, avec hausses système d’environ 39 et 32 Mo. Dans le trajet de 128 tokens surveillé avec psutil 7.2.2 sur Python 3.12, le swap reste à **80 281 600 octets**, avec au moins 2 794 143 744 octets disponibles et un pic RSS de 5 632 425 984 octets. L’observateur externe a démarré tard : ses compteurs ne couvrent pas tout le lancement. Aucune cause unique n’est attribuée aux différences entre tentatives.

**Un essai Low de 64 tokens utilise bien le runtime natif, ses bibliothèques et psutil embarqués dans l’application finale. Il échoue après 428,335 s avec `SWAP_LIMIT`.** Le swap système passe de 80 281 600 à 559 874 048 octets, soit **+479 592 448 octets** ; la dernière mesure conserve pourtant 3 549 773 824 octets disponibles. Les gardes n’ont pas été relâchées. Le compteur de swap est global : cet échec ne démontre pas à lui seul une fuite du modèle, une causalité du moniteur ou un plafond physique.

Le template réellement appliqué confirme `Reasoning Effort: Low` pour l’option explicite et `Reasoning Effort: Max` sans option. **Le défaut Max reste inchangé.** Aucun job ne s’est terminé dans cet essai de 64 tokens ; il reste un échec de stabilité, sans attribution causale exclusive au modèle ou au moniteur.

Un essai entièrement embarqué ultérieur termine **deux requêtes de quatre tokens en 197,824 s** au total, une en Low et l’autre sans changement du défaut Max. Le même PID est conservé : **un chargement, deux requêtes, huit tokens**, manifeste runtime vérifié et 600 agents logiques. Le pic RSS est de 5 304 893 440 octets, la mémoire disponible minimale de 2 915 532 800 octets. Le swap système passe de 559 874 048 à 551 485 440 octets sans dépasser sa valeur initiale. Cette preuve qualifie l’intégration courte et la réutilisation du moteur. Les deux arrêts restent `limit` et les textes sont des fragments de raisonnement, sans réponse finale ni EOS. Elle ne supprime pas l’échec prolongé précédent.

Sources : `ronin-e2e-traced-psutil.json`, son suivi de ressources, `ronin-embedded-low-and-reuse.json` et `ronin-embedded-4x2.json`.

L’application se trouve à **`outputs/RONIN-GLM/RONIN GLM.app`**, le runtime reconstruit séparé à `outputs/native-runtime/`, les sources à `integrations/ronin/` et `research/glm53_m5/`. Le paquet final embarque le runtime natif et son moniteur ; Python et les 110 Go du modèle restent externes. L’ancienne application n’est pas remplacée. Depuis la racine de cette livraison :

```sh
open 'outputs/RONIN-GLM/RONIN GLM.app'
```

Le point d’entrée CLI est livré sous `Contents/Resources/integrations/ronin/ronin_glm.py`, avec `runtime-config.json`. L’accueil, les réglages, le modèle exact et l’état **Chargé** après clic sur Charger GLM ont été contrôlés visuellement. Une mission lancée dans l’interface produit ensuite **32 tokens en 321,881 s** : préfill de 18 tokens en 50,295 s, soit 0,357885 token/s ; décodage de 31 intervalles en 271,574 s, soit **0,114150 token/s**. L’interface affiche **Terminée** ; Preuves → Réponse GLM locale montre les IDs exacts, timings et `stop_type=limit` enregistrés. La sortie reste du raisonnement : aucun entier final ni EOS n’est obtenu. Le budget explicite de 32 tokens et le défaut Max restent inchangés. Preuve : `ronin-gui-generation.json`.

Le rafraîchissement automatique omettait l’état `waiting_model` et nécessitait de sélectionner à nouveau la mission pour voir sa fin. Ce défaut a été corrigé, Swift recompilé et la signature stricte revérifiée (`app-signature-release-v2.json`). Après réouverture, la mission persistée reste Terminée et le moteur est déchargé, sans processus natif restant. **Aucune seconde génération après cette correction ne valide encore le rafraîchissement automatique à l’achèvement.**

La signature locale ad hoc passe `codesign --verify --deep --strict` ; le paquet n’est pas notarié. Le dernier rerun portable comptabilise **622 tests réussis, 64 ignorés et 92 sous-tests, en 6,10 s**. Ces contrôles ne qualifient pas les performances matérielles ou la fidélité générale. Trois échecs initiaux de capture pytest 9.2 ont aussi été reproduits sur la PR14 propre ; `--capture=sys` permet le rerun sans modifier les preuves historiques. Huit tests du constructeur incluent le refus d’un exécutable ou d’une bibliothèque altérés avant copie/signature. La vérification réelle du paquet source contrôle 29 artefacts, soit 25 608 224 octets. Reçus : `portable-controls-release.json`, `ronin-controls-release.json`, `packaging-source-integrity-release.json`, `app-signature-release-v2.json` et `research/glm53_m5/evidence/pytest-capture-note.json`.

Les sorties restent des propositions non vérifiées, sans exécution d’outils à partir de leur texte. Restent ouverts : stabilité prolongée de la livraison entièrement embarquée, réponse jusqu’à EOS, rafraîchissement automatique après une nouvelle génération, reprise après incident, contextes réellement longs, corpus varié, répétitions équilibrées, parité longue, mesure physique I/O et attribution précise des kernels. Aucun gain natif ne transforme ces validations manquantes en résultats acquis.
