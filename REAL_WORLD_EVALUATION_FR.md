# AirLLM Next — évaluation d’un assistant local en conditions réelles

**Le moteur fonctionne comme prototype d’assistant local : il programme, rédige, calcule et conserve une conversation réelle. L’expérience quotidienne reste limitée par une forte latence, des réponses parfois incomplètes ou trop affirmatives, et une génération à 1 024 tokens non validée.**

Sur les 21 tours principaux : **16 réussites, 2 réponses partielles, 2 échecs de critère, 1 cas indéterminé**. Une lecture plus souple de la contrainte de phrase de E5 porterait les réussites à 17 ; cette sensibilité est déclarée. Le décodage vaut **0,917 token/s** et la réponse complète médiane **109,16 s**, sans les pauses humaines. Les paliers 128/256/512 atteignent chacun leur plafond configuré ; celui de 1 024 est arrêté à 244 tokens par une croissance de swap système de **162,56 Mio**. Il serait donc faux de qualifier toute cette mission de « sans croissance de swap ».

Le réemploi du contexte réduit, sur quatre tours appariés, la somme des délais avant premier token de **536,02 à 84,24 s**, avec les cinq réponses identiques. Une consigne de concision permet de livrer les deux résultats de calcul et un choix budgétaire initialement coupés, sans modifier les poids. Un véritable arrêt du routeur est corrigé, puis la demande exacte est rejouée jusqu’à EOS.

Les réponses, y compris les échecs et les fragments, sont celles du modèle réellement exécuté par AirLLM Next. Codex les évalue ; il ne les remplace pas par ses propres réponses. Aucun message réel n’est envoyé, aucun service n’est modifié par les tâches proposées au modèle. Toutes leurs données sont synthétiques.

## Matériel, modèle et périmètre

Les réponses proviennent du backend GGUF natif d’AirLLM Next, exécuté par MLX/Metal sur le MacBook Air M5 à 16 Go de mémoire unifiée et SSD interne. Ollama n’a pas servi à générer les réponses. Son magasin contient le fichier déjà installé ; lire ce fichier avec AirLLM n’est pas exécuter Ollama.

Le tag local `qwen3.6:latest` désigne le GGUF Q4_K_M de 23 938 321 664 octets, à architecture `qwen35moe` / Qwen3.6-35B-A3B : 40 couches, 256 experts, 8 experts activés par token, dimension cachée 2 048. L’identité et les métadonnées lues sont conservées dans [model-provenance.json](research/real_world/model-provenance.json) et [environment.json](research/real_world/environment.json). Le nom commercial 36,0B du manifeste n’est pas une mesure indépendante du nombre de paramètres.

Python 3.12.13, MLX 0.32.4, Transformers 5.20.2, tokenizers 0.23.3 ; Torch 2.14.1 sert uniquement de référence CPU au petit routeur lorsque le chemin natif rencontre une égalité au seuil. Aucun modèle complet n’est chargé dans Torch.

L’empreinte complète a été recalculée après toutes les inférences, par lecture seule en blocs de 4 Mio : **SHA-256 `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`**, sur les 23 938 321 664 octets, avec identité et métadonnées du fichier inchangées avant/après. Le garde termine sans violation ni hausse de swap. Les métriques brutes gardent honnêtement leur champ « empreinte précédemment vérifiée » ; cette vérification actuelle est une preuve séparée : [model-sha256-current.json](evidence-real-world/model-sha256-current.json). Un premier essai du wrapper de hash a échoué avant toute lecture parce qu’il refusait le journal créé par son propre superviseur ; cet échec est conservé sous `model-sha256-wrapper-preflight.*`, puis le wrapper local est corrigé. Aucun moteur ni poids n’est modifié par cette étape.

## Configuration et protections

Les demandes utilisent le tokenizer du GGUF, le template officiel vérifié, un message système français, les rôles réels et l’historique effectivement généré. Décodage greedy, température 0, thinking désactivé, EOS actifs : `im_end` 248046 et `endoftext` 248044. Le moteur ne reçoit ni réponses attendues ni tests de correction.

Configuration principale : cache de poids 64 Mio, KV 256 Mio, contexte local 1 280 tokens, préremplissage par blocs de 16, lignes de poids par blocs de 2 048, calcul FP32 sur GPU, kernel Metal, lectures `pread`, préchargement désactivé. Les quatre paliers longs utilisent cache 32 Mio, KV 288 Mio et contexte 2 048 ; ce changement est déclaré, et leurs vitesses ne sont pas attribuées au seul plafond de génération.

Chaque inférence est exécutée dans un enfant supervisé : RSS maximal 2 Gio, réserve système 3 Gio, seuil d’arrêt pour croissance du swap de 128 Mio, pression mémoire normale requise, échantillonnage 50 ms. Une première admission à 1 Gio a été refusée avant lancement faute de marge ; les budgets de cache ont été réduits, sans diminuer les protections. Deux lancements de la série longue ont ensuite été refusés avant création du processus après baisse de la mémoire disponible. Les preuves d’admission sont conservées ; la suite attend ensuite la réserve, le payload prévu et une marge supplémentaire de 256 Mio avant chaque nouvelle tentative, avec les gardes inchangés. Le swap déjà présent au départ est distinct de sa croissance pendant l’essai. Le palier 1 024 a toutefois été arrêté à 244 tokens par une croissance système observée de 170 459 136 octets (162,56 Mio), au-delà du seuil ; RSS échantillonné 429,28 Mio et pression mémoire normale. L’échantillonnage détecte un dépassement et ne garantit pas une borne d’allocation stricte. Cette hausse ne peut pas être attribuée au seul modèle ou aux seules autres applications. Les essais suivants attendent une fenêtre de 60 secondes sans croissance du swap ; le palier 1 024 n’est pas relancé et reste non validé.

## Lecture des mesures

Les JSON donnent les prompts, rôles, historiques, IDs d’entrée/sortie, textes complets, arrêts, configuration, initialisation, temps par token et ressources. [ANALYSIS_FR.md](research/real_world/ANALYSIS_FR.md) et [RESPONSES_FR.md](research/real_world/RESPONSES_FR.md) les présentent de façon lisible.

L’ouverture du moteur en environ 0,4–0,5 s n’est pas le chargement complet des 23,94 Go : les poids sont lus sélectivement à la demande, et ce coût reste dans le préremplissage et le décodage. Le TTFT moteur commence juste avant le premier passage du modèle. Le délai de première requête inclut en plus lancement de l’enfant, tokenizer et ouverture du moteur ; l’analyse fournit un intervalle conservateur pour les petits coûts de préparation restants. La durée jusqu’à la réponse complète est mesurée directement. Le champ brut `cold_first_request` désigne la première requête du processus ; il n’est pas assimilé à un SSD froid : aucun vidage des caches système n’a été effectué.

Le débit de décodage est `(tokens générés − 1) / (fin − premier token)`, EOS compris s’il est généré. L’agrégat est un ratio de sommes. La sauvegarde de preuves à chaque token fait partie du temps mesuré ; le prototype peut fournir un callback par token, mais l’évaluation n’est pas un test d’une nouvelle interface graphique.

Les lectures logiques correspondent aux octets demandés par le lecteur natif. Les octets de stockage disponibles viennent des compteurs du processus macOS ; ils ne constituent pas une mesure directe des lectures NAND du SSD. Les caches système et l’activité de la machine restent des facteurs non contrôlés. Le RSS échantillonné ne mesure pas à lui seul toute la mémoire unifiée, les caches de fichiers et l’occupation GPU ; la réserve et la pression système complètent la surveillance.

## Corrections et diagnostic

Le contrat texte du tokenizer a été confronté aux assets officiels : 9 encodages, 18 décodages et 64 rendus/encodages conversationnels concordent dans le périmètre testé. L’ancien template omettait l’ouverture de pensée dans un mode et gérait incorrectement certains historiques. L’appel simple sans thinking était déjà correct. Une divergence Hindi avec l’adaptateur AutoTokenizer installé vient de sa reconstruction de l’expression de pré-tokenisation ; AirLLM et l’asset `tokenizer.json` officiel concordent sur le cas conservé. Ces vérifications ne prouvent pas la qualité générale du modèle.

Un contrôle CLI réel a répondu « 4 » à la demande conversationnelle « Combien font 2+2 ? Réponds uniquement par le nombre. », en 15,368 s tokenizer compris, avec deux IDs dont EOS. Cette sortie est conservée avec sa [revue séparée](evidence-real-world/baseline_review.json). Elle ne remplace pas l’ancien essai brut `2+2=` : les prompts diffèrent, donc aucune correction de qualité à prompt identique n’en est déduite.

Le nouvel adaptateur [GGUFChatSession](air_llm/airllm/nextgen/gguf_chat.py) conserve les messages réels, contrôle le contexte avant exécution et réutilise le KV uniquement si le préfixe de tokens est strictement identique. Une divergence force une reconstruction sûre. Un échec ne valide pas un tour inachevé dans l’historique. Tous les logits sont contrôlés finis avant sélection.

Un défaut réel de l’implémentation a interrompu D2 après 77 tokens : deux experts avaient des logits GPU FP32 égaux au seuil top-8. Le replay a reproduit les 77 tokens et capturé les valeurs originales. Le routeur CPU complet distingue ces experts. Le correctif recalcule seulement la petite gate, sur la forme de chunk entière, puis remplace uniquement les lignes affectées. Il garde le refus générique par défaut et applique les budgets avant import/allocation. Le replay D2 corrigé termine sur EOS à 79 tokens, avec le même préfixe de 77 tokens. Voir [ROUTER_TIE_FIX_FR.md](research/real_world/ROUTER_TIE_FIX_FR.md).

Le repli D2 directement instrumenté coûte 0,673 s, dont 0,659 s d’import Torch. La différence de temps global entre réponse interrompue et réponse achevée n’est pas un surcoût causal : leurs points d’arrêt et états de cache diffèrent. Le correctif ne prétend pas fournir une identité bit à bit du modèle entier avec Transformers, ni un ordre universel des égalités sur tous les backends.

La sortie D2 indique « 8 avril » pour « date actuelle », au lieu du 22 mai attendu pour la formation. Le 8 avril est la date de version du dossier : la formulation originale est ambiguë. L’écart au critère prévu reste visible, mais son attribution à une erreur de compréhension du modèle est incertaine. Le contrôle à consigne clarifiée est présenté séparément.

Sur les 13 tours de la reprise corrigée, 51 200 contrôles de frontière totalisent 12,994 s ; les deux replis CPU totalisent 0,751 s, import compris. Ces temps instrumentés ne sont donc pas le goulot principal des 1 964,49 s de requêtes de cette phase. Aucun pourcentage causal de l’ensemble du backend n’est déduit des seuls compteurs.

La sauvegarde des preuves est devenue atomique après revue : fichier temporaire puis remplacement, sans changer les réponses. Les versions originales et leurs empreintes sont archivées. La campagne principale contient donc une phase avant correctif et une phase après correctif, distinguées dans les résultats ; elle n’est pas présentée comme 21 réponses produites par une unique version finale.

## Résultats principaux : 21 demandes réellement exécutées

Les 21 tours fonctionnels couvrent 17 sessions, dont une conversation de cinq tours. La lecture indépendante compte **16 réussites selon les critères retenus, 2 réponses partielles, 2 échecs de critère et 1 cas indéterminé**. Une réussite exige les résultats essentiels et les contraintes interprétées du prompt ; un défaut de clarté n’invalide pas automatiquement une solution exacte. A1 est accepté après seconde lecture de ses parenthèses, toutes vraies. E5 reste partiel selon la convention stricte d’une phrase pour toute la réponse, mais serait réussi si cette contrainte porte seulement sur la solution : le total pourrait alors être 17/21. D2 reste indéterminé. Ce petit échantillon ne constitue pas un taux de qualité représentatif.

La colonne qualité donne les moyennes des **cinq axes séparés** sur 5, dans cet ordre : exactitude / instructions / cohérence / complétude / utilité. Les notes et justifications par réponse sont dans [independent_review.json](evidence-real-world/independent_review.json), liées aux textes par SHA-256. La vitesse comprend ici la médiane du temps jusqu’à réponse complète et le débit de décodage agrégé ; les TTFT et coûts de première requête sont détaillés plus loin.

| Domaine | Tâches réussies | Qualité : cinq axes /5 | Vitesse | Problèmes |
| --- | --- | --- | --- | --- |
| Raisonnement | 2/3 | 5,0 / 4,3 / 4,0 / 4,3 / 4,0 | 164,5 s ; 0,96 tok/s | Clarté de A1 discutée ; A2 coupé avant le reste du budget. |
| Programmation | 2/2 | 5,0 / 5,0 / 5,0 / 5,0 / 5,0 | 119,9 s ; 0,94 tok/s | Deux fonctions exactes sur les cas exécutés ; pas une preuve générale de correction. |
| Français | 2/2 | 5,0 / 5,0 / 5,0 / 5,0 / 5,0 | 88,1 s ; 0,97 tok/s | Deux exercices courts ; style partiellement subjectif. |
| Documents | 1/2 | 4,5 / 4,5 / 5,0 / 4,5 / 4,5 | 189,5 s ; 0,86 tok/s | D2 : date ambiguë ; arrêt du routeur corrigé et rejoué. |
| Conversation | 4/5 | 5,0 / 4,8 / 5,0 / 5,0 / 5,0 | 79,6 s ; 0,90 tok/s | Contexte corrigé conservé ; E5 en deux phrases au lieu d’une. |
| Tâches complexes | 1/2 | 5,0 / 3,0 / 4,5 / 3,0 / 3,5 | 290,7 s ; 0,88 tok/s | F1 exact ; F2 interrompu avant les totaux et le choix. |
| Fiabilité | 4/5 | 4,4 / 4,4 / 4,6 / 4,8 / 4,2 | 82,1 s ; 0,92 tok/s | G3 tranche arbitrairement le propriétaire du rapport. |

Les calculs de référence ont été vérifiés indépendamment par fractions exactes, décimaux ou énumération : [GROUND_TRUTH_FR.md](research/real_world/GROUND_TRUTH_FR.md). Les deux fonctions Python sont exécutées **sans correction** dans un environnement restreint : **8/8 cas prévus**, plus **91/91 contrôles supplémentaires dans le domaine annoncé**. B1 retourne une nouvelle liste sans mutation et conserve l’ordre ; B2 calcule correctement les sommes, dont nombres négatifs, zéros et grands entiers. Les trois essais liste/tuple échouent dans une extension hors contrat et sont publiés séparément, sans diminuer artificiellement la réussite du prompt d’origine.

Les problèmes importants restent visibles : A2 donne 403,20 € mais omet le reste de 46,80 € ; F2 élimine correctement l’offre de huit places mais ne livre aucun choix ; G3 attribue arbitrairement le rapport à Julie par une règle grammaticale non fondée, tout en reconnaissant l’ambiguïté de la réunion. Les trois sorties A2/F2/G3 atteignent le plafond de 256 tokens. Pour G3, le défaut affirmatif apparaît bien avant la coupure : ce n’est pas seulement une phrase finale manquante.

G3 part d’une session neuve, sans KV réutilisé, avec template vérifié et logits finis. Cela écarte les problèmes évidents de contexte ancien et de prompt brut. Sans génération complète de référence aux mêmes poids, il reste impossible d’attribuer définitivement cette erreur au modèle, à la quantification ou aux différences numériques du backend. Aucune correction fictive n’est annoncée.

## Attente réelle sur les demandes principales

**1 865 tokens** sont produits en **3 080,66 s** cumulées de requêtes, soit **0,61 token/s** quand le préremplissage est inclus. Le décodage seul atteint **0,917 token/s**. La médiane du premier token moteur est **45,80 s** (18,03 à 189,83 s) ; celle de la réponse complète est **109,16 s** (57,31 à 369,87 s), hors coût de lancement ajouté aux premières requêtes.

Les deux ouvertures de processus principales ajoutent environ 1,9 s et 1,8 s avant la première demande, tokenizer compris. La première réponse A1 prend ainsi 166,39 s depuis le lancement, et la première D2 de la reprise 283,40 s. Les demandes suivantes gardent le moteur ouvert ; elles ne sont pas exemptées du préremplissage de leur nouveau contexte.

Les compteurs totalisent **4 163,63 Gio de lectures logiques** et **1 279,98 Gio de lectures de stockage du processus** pour ces 21 tours. Ils ne mesurent pas directement le trafic NAND. Le travail de préremplissage, les lectures répétées et le décodage à environ un token par seconde sont les goulots visibles pour l’utilisateur. Ces observations ne suffisent pas à attribuer un pourcentage causal à chaque composant.

Sur les 21 tours retenus, RSS maximal échantillonné **557,09 Mio** et croissance maximale de swap **0 Mio**. Le garde global de la reprise observe aussi une disponibilité système minimale de **3 558 768 640 octets**, au-dessus de la réserve de 3 Gio ; le swap préexistant reste de plusieurs Gio. Les maxima ne sont pas des garanties d’allocation ni d’absence de pics entre échantillons.

## Amélioration mesurée : conserver le contexte exact

Les cinq tours E ont été rejoués avec reconstruction complète du KV à chaque demande, sur le même code final et les mêmes budgets 64/256 Mio, contexte 1 280. **Les cinq prompts rendus et les cinq suites de tokens générés sont exactement identiques** entre les deux modes. Aucun résultat correct n’est remplacé par une sortie plus courte.

| Tour | Entrée complète | Entrée ajoutée avec cache | TTFT reconstruction s | TTFT réemploi s | Sortie identique |
| --- | --- | --- | --- | --- | --- |
| E1 | 116 | 116 | 42,72 | 44,64 | oui |
| E2 | 198 | 52 | 69,80 | 20,53 | oui |
| E3 | 268 | 40 | 100,34 | 18,03 | oui |
| E4 | 378 | 67 | 143,00 | 23,25 | oui |
| E5 | 548 | 62 | 222,88 | 22,44 | oui |

Pour E2–E5, la somme des TTFT passe de **536,02 s à 84,24 s**, soit un facteur **6,36** sur cet indicateur et **451,78 s économisées**. La somme des réponses complètes passe de **840,99 s à 382,16 s** (facteur **2,20**). E1 est exclu de ce gain, car il n’a aucun préfixe à réemployer et son rang de processus diffère entre passages.

Le contrôle reconstruit termine avec garde `exit=0`, RSS maximal échantillonné 453,70 Mio et croissance de swap observée nulle. Cette comparaison comporte un seul passage par mode, dans cet ordre, sans maîtrise des caches OS ni de la charge de la machine : elle démontre le résultat observé dans ce petit historique, pas un facteur universel ou un essai randomisé. Le réemploi exige toujours une égalité stricte des IDs consommés ; aucun historique n’est supprimé pour accélérer.

Preuves : [conversation-rebuild.json](evidence-real-world/conversation-rebuild.json), [garde](evidence-real-world/conversation-rebuild.guard.json), [passage avec réemploi](evidence-real-world/campaign-resumed.json).

## Amélioration de livraison : consignes de concision

Deux diagnostics postérieurs aux échecs de livraison conservent les questions utilisateur, le modèle, les budgets et le plafond 256. Seule une contrainte de concision est ajoutée au système. Les demandes et résultats d’origine restent dans les 21 tours principaux ; aucun échec n’est effacé.

| Demande | Principal : tokens / attente complète | Contrôle : tokens / attente depuis ouverture | Résultat du contrôle |
| --- | --- | --- | --- |
| A2, remise et livraison | 256 / 312,33 s | 51 / 117,04 s | 403,20 € et reste 46,80 €, tous deux exacts ; une ligne au lieu des deux exigées |
| F2, choix sous contraintes | 256 / 369,87 s | 97 / 193,03 s | A=460 €, B=460 €, C=390 € théoriques mais capacité 8 ; choix B ; quatre lignes respectées |

A2 ne devient pas plus rapide au premier token : TTFT moteur 43,62 → 61,00 s. F2 passe de 74,36 → 77,59 s. Le prompt système plus long et le lancement distinct font partie du contrôle. Le gain concerne la livraison d’une réponse utile plus courte, pas la vitesse du kernel, un changement des poids ou une amélioration démontrée de tous les premiers tokens.

Le contrôle A2 produit exactement :

```text
Total : 18 * 24 * 0.85 + 36 = 403.20 €. Reste : 450 - 403.20 = 46.80 €.
```

Le contrôle F2 produit exactement :

```text
Option A : 380 + 80 = 460 € (capacité 10, valide).
Option B : 440 + 20 = 460 € (capacité 12, valide).
Option C : 350 + 40 = 390 € (capacité 8, invalide).
Choix : Option B, car coût égal à A mais capacité supérieure.
```

Ces contrôles montrent qu’une consigne de style peut livrer les résultats dans 256 tokens. Ils n’établissent pas ce que les prompts initiaux auraient produit au-delà de leur plafond. Les deux gardes terminent sans violation et avec croissance de swap observée nulle. A2 reste partiel pour son format dans la revue diagnostique ; F2 satisfait la consigne concise. Les évaluations détaillées et leurs cinq axes restent séparés dans [diagnostic_review.json](evidence-real-world/diagnostic_review.json).

Preuves : [A2 concis](evidence-real-world/concision-control.json), [F2 concis](evidence-real-world/f2-concision-control.json).

## Diagnostic documentaire : clarifier la date visée

Le contrôle séparé remplace seulement la formulation ambiguë par « date retenue pour la journée de formation […] sans les confondre avec la date de version du dossier ». Le dossier et les critères numériques restent identiques. Le modèle répond exactement :

```text
La date retenue est le 22 mai, l’ancien créneau du 15 mai ayant été annulé.
Le total des dépenses prévues s’élève à 1 632 € (salle, repas et intervenante), soit une marge de 168 € sur le budget de 1 800 €.
Les deux points non confirmés sont la confirmation écrite de l’intervenante et les détails des allergies alimentaires.
Inès est responsable des repas avec une échéance de commande avant le 13 mai.
```

Les quatre lignes et les calculs sont corrects : 116 tokens, EOS, TTFT moteur 193,56 s, réponse depuis ouverture 326,08 s, croissance de swap observée nulle. L’attente de précaution avant ouverture ajoute 120,98 s à cette fenêtre opérationnelle. Ce contrôle de compréhension est plus long que D2 initial ; ce n’est pas une optimisation de vitesse. Il renforce le diagnostic d’une consigne initiale ambiguë, sans effacer son résultat ni démontrer une absence générale d’erreurs du modèle.

Preuve : [d2-clarified.json](evidence-real-world/d2-clarified.json), avec les notes séparées dans [diagnostic_review.json](evidence-real-world/diagnostic_review.json).

## Générations longues : trois paliers achevés, 1 024 non validé

Les quatre demandes partagent exactement le même texte, qui sollicite un guide de 900–1 100 mots en huit parties. Les plafonds limitent donc volontairement la portion observable. Un plafond atteint n’est pas un EOS prématuré ni une preuve de tâche achevée. EOS reste actif à tous les paliers.

| Plafond | Tokens réellement produits | Arrêt | TTFT moteur s | Réponse depuis ouverture s | Décodage tok/s | RSS max Mio | Swap Δ Mio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 128 | plafond | 160,92 | 293,39 | 0,973 | 421,81 | 0,00 |
| 256 | 256 | plafond | 180,00 | 462,39 | 0,909 | 386,95 | 0,00 |
| 512 | 512 | plafond | 178,00 | 743,80 | 0,906 | 391,95 | 0,00 |
| 1024 | 244 | garde : croissance swap | indisponible | interrompu après 433,66 s de processus | non calculé | 429,28 | 162,56 |

**Le palier 1 024 n’est pas validé.** Le garde l’arrête à 244 tokens lorsque le swap système augmente de **170 459 136 octets (162,56 Mio)**, dépassant son seuil de 128 Mio. La mémoire disponible minimale est 3 905 142 784 octets et la pression reste normale : ces autres indicateurs ne suffisent donc pas à annoncer une absence de pagination. Le pic RSS du processus ne représente pas toute la mémoire du système. L’origine de cette croissance globale est indéterminée ; ni le modèle ni les autres applications ne sont isolés causalement. Le palier n’a pas été relancé pour contourner cet arrêt.

Le JSON enfant conserve `status=started`, car SIGTERM a empêché sa clôture ; le garde final `exit=-15` et `violation=system_swap_growth` fait autorité. L’analyse l’affiche comme interrompu. Les 244 IDs et le fragment brut restent inchangés, avec notes conditionnelles distinctes, et aucun débit ou temps de réponse complète fictif n’est dérivé de ce fragment.

Les préfixes sont identiques : 128/128 entre L128 et L256, 256/256 entre L256 et L512, et 244/244 pour le fragment L1024 face à L512. Aucune boucle, aucun token spécial parasite ni logit non fini n’est observé sur les portions produites. Cela ne valide pas les tokens non générés.

L128 donne 77 mots ; L256, 170 ; L512, 346. Le guide demeure incomplet : L512 développe situation et objectifs, puis amorce seulement le budget. Les objectifs ajoutés — sensibilisation, rentabilité — peuvent être un cadrage proposé, dont le statut est peu balisé ; ils ne sont pas qualifiés automatiquement de faits inventés certains. Les réserves réelles sur salle, dons et réparabilité restent présentes.

Sur L512, la durée moyenne des 64 premières passes de décodage après le premier token vaut 1,076 s, contre 1,065 s pour les 64 dernières. Ces durées synchronisées excluent le callback de sauvegarde, contrairement au débit global. Elles ne montrent pas de ralentissement progressif marqué sur ce passage ; ni température, ni puissance, ni endurance thermique ne sont validées. Les débits globaux 0,973 / 0,909 / 0,906 diffèrent aussi selon le cache système et les tokens traités, sans répétitions contrôlées.

Les cinq axes détaillés, limites et fragments sont dans [long_review.json](evidence-real-world/long_review.json). Les textes intégraux sont dans [RESPONSES_FR.md](research/real_world/RESPONSES_FR.md) et les fichiers `evidence-real-world/long-*.json` avec leurs gardes.

## Contrôle difficile supplémentaire : probabilités conditionnelles

A4 est annoncé prospectivement après les premières observations, avant sa génération ; il reste hors des 21 demandes principales. Le modèle doit suivre deux tests successifs, leurs effectifs conditionnels et un seuil de 90 %. Sa réponse intégrale est :

```text
1) A+ : 900 D et 900 non-D.
2) A+B+ : 720 D et 450 non-D.
3) P(D|A+) = 0,5 (50 %) ; P(D|A+B+) = 720/1170 ≈ 0,6154 (61,54 %).
4) Non, le seuil de 90 % n'est pas atteint.
```

Tous les effectifs et calculs sont exacts : 0,5 vaut 1/2 et 720/1170 vaut 8/13. La seconde fraction n’avait pas à être réduite. Les quatre lignes et la décision de seuil sont correctes. La revue attribue **5 / 4 / 5 / 5 / 5** ; le verdict formel reste partiel parce que la première probabilité n’est pas écrite en fraction a/b. Si une décimale exacte est acceptée, une réussite est défendable : aucune erreur de Bayes n’est observée.

107 tokens, EOS, premier token moteur 97,69 s, réponse depuis ouverture 215,22 s ; croissance de swap observée nulle. Preuves : [hard-reasoning.json](evidence-real-world/hard-reasoning.json), [supplementary_review.json](evidence-real-world/supplementary_review.json).

## Attentes de précaution avant ouverture

Après l’arrêt du palier 1 024, l’orchestrateur de recherche attend une marge mémoire supplémentaire de 256 Mio et 60 secondes de swap non croissant. Ces pauses précèdent le lancement et s’ajoutent aux temps moteur. Elles ne sont ni cachées dans le décodage, ni présentées comme un coût universel de l’API native. Le tableau donne le délai jusqu’à la fin du premier tour ; pour l’ablation, les quatre tours suivants viennent ensuite.

| Essai | Attente avant lancement s | Lancement → première réponse complète s | Somme opérationnelle s |
| --- | --- | --- | --- |
| Conversation reconstruite, E1 | 60.47 | 77.97 | 138.43 |
| A2 concis | 70.59 | 117.04 | 187.63 |
| F2 concis | 60.52 | 193.03 | 253.55 |
| D2 clarifié | 120.98 | 326.08 | 447.06 |
| A4 probabilités | 70.59 | 215.22 | 285.81 |

Ces sommes additionnent deux fenêtres mesurées et excluent de petites opérations de validation entre elles. Elles ne sont pas un chronométrage d’interface utilisateur. Le journal complet des admissions et de l’unique échec conservé est [campaign-orchestration-final.json](evidence-real-world/campaign-orchestration-final.json). Les délais des premières requêtes du début de campagne, sans cette politique ajoutée, restent ceux décrits dans leurs données brutes.

## Validation, provenance et reproduction

La suite corrigée termine avec **1 166 tests réussis, 1 ignoré et 11 sous-tests**, en 21,87 s de pytest : [journal](evidence-real-world/validation-fixed.log), [garde](evidence-real-world/validation-fixed.guard.json). Le test ignoré exige CUDA/bitsandbytes sur matériel NVIDIA. Le garde observe 577,75 Mio de RSS maximal et aucune hausse de swap ; le fichier XML reste local.

Après cette suite, le test de capture réelle du routeur devient explicitement optionnel par variable d’environnement, avec taille et SHA bornés ; le moteur n’est pas changé. Le contrôle final du fichier courant passe **13/13 tests**, capture locale comprise : [router-current-tests.json](evidence-real-world/router-current-tests.json). Le contrôle public sans capture passait 12 tests et en ignorait un. Les neuf tests de sauvegarde atomique sont inclus dans la suite complète.

Les deux fonctions produites par le modèle ont été réexécutées sans correction après clôture des fichiers : [code-checks.json](evidence-real-world/code-checks.json). L’ancien contrôle, réalisé alors que la campagne évoluait encore, reste dans `code-checks-in-progress.json` ; les textes par réponse concordent, et le contrôle final lie également le fichier de campagne complet. Les contrôles supplémentaires sont dans [adversarial-code-checks.json](evidence-real-world/adversarial-code-checks.json). L’isolation testée refuse le réseau, la création de sous-processus et les écritures hors du dossier temporaire ; Python utilise un AST et des builtins restreints, une limite CPU de 3 s, des fichiers limités à 1 Mio et un suivi RSS parent à 256 Mio. Darwin n’applique pas ici RLIMIT_AS/DATA : ce suivi échantillonné ne devient pas une borne absolue d’allocation. Ce dispositif est spécifique à ces petites fonctions, pas un environnement pour n’importe quel programme.

Au total, **36 sorties achevées et 2 fragments originaux** ont une lecture et cinq notes, liées au texte exact par SHA : [review_bindings.json](evidence-real-world/review_bindings.json). Les 36 comprennent 21 tours principaux, 3 paliers longs, 5 tours d’ablation, 3 contrôles ciblés, A4, 2 sorties pilotes et le contrôle CLI. Les deux fragments sont D2 interrompu et L1024. La capture D2 reproduite à l’identique n’est pas comptée une deuxième fois ; les préflights à zéro token ne sont pas des réponses évaluables. Les pilotes à plafond court sont conservés et l’amendement à 256 tokens est déclaré avant la campagne retenue. Aucune sélection a posteriori ne remplace les résultats principaux.

Le corpus, les critères et amendements figurent dans [PROTOCOL_FR.md](research/real_world/PROTOCOL_FR.md) et [tasks.json](research/real_world/tasks.json). Les commandes exactes, configurations, versions et empreintes du moteur sont dans chaque JSON/garde. Les sources antérieures au correctif du routeur sont archivées pour expliquer la phase initiale. Le fichier binaire de capture contient des poids : il reste local, ignoré par Git, avec seulement son empreinte publiée. Aucun poids complet, asset tokenizer téléchargé, secret ni document personnel n’est ajouté.

Pour recalculer les tableaux à partir des preuves déjà enregistrées, sans modèle ni réseau :

```sh
python3 benchmarks/analyze_real_world.py --help
python3 benchmarks/analyze_real_world.py
```

Les arguments de l’évaluateur et du vérificateur de code sont accessibles avec `--help`. Toute nouvelle inférence doit conserver le superviseur et ses seuils ; le palier 1 024 arrêté n’est pas relancé dans cette mission. L’analyseur est contrôlé indépendamment dans [ANALYZER_AUDIT_FR.md](research/real_world/ANALYZER_AUDIT_FR.md). Un générateur de figures est fourni, mais les figures ne sont pas produites dans l’environnement courant faute de matplotlib ; aucun graphique inexistant n’est annoncé.

L’inventaire de publication et les SHA-256 sont dans [publication_manifest.json](research/real_world/publication_manifest.json), avec les exclusions dans [RELEASE_AUDIT_FR.md](research/real_world/RELEASE_AUDIT_FR.md). Le manifeste ne s’auto-hache pas. Cette branche dépend de la PR #4 et cible `research/native-gguf-2026-10-23`, depuis `ac93b5d084dca4e72ec11092629bc4ebf7ac8a25` ; elle ne fusionne ni ne modifie `main`.

## Limites de conclusion


Le jeu est petit, synthétique et ciblé. Il porte sur le décodage greedy sans mode de pensée ; ni le sampling ni les autres modes de génération n’ont été comparés. Les notes sur cinq axes proviennent d’une lecture indépendante par Codex, avec des contrôles objectifs quand ils sont possibles ; elles ne sont pas des évaluations humaines en aveugle ni un classement universel. Les poids n’ont pas été modifiés. Une réponse fausse ne suffit pas à distinguer modèle, quantification et différences numériques sans comparaison supplémentaire.

Les champs conservateurs `full_qwen_checkpoint_validated=false` et `metal_validated=false` des métriques restent inchangés : l’exécution GPU/Metal observée ici n’est pas une concordance numérique complète du checkpoint avec une référence indépendante.

Cette mission ne valide ni un modèle BF16 complet, ni 300–700B, ni le SSD externe non reçu, ni un contexte de 262 144 tokens malgré la métadonnée du modèle, ni une interface grand public, ni une endurance thermique certifiée. Température physique, puissance et consommation d’énergie ne sont pas mesurées. Les observations de swap concernent uniquement les fenêtres enregistrées ; une croissance nette nulle ne démontre pas une absence d’activité de pagination entre deux échantillons.

## Priorités et réponse à la question de la mission

**Oui pour un prototype réel d’assistant local supervisé ; pas encore pour une expérience quotidienne fluide et une fiabilité générale démontrée.** Les sorties utiles et le contexte réel sont prouvés sur ce corpus ; environ une minute avant le premier token médian et près de deux minutes jusqu’à la réponse médiane restent pénalisantes. Le palier de 512 tokens aboutit en environ 12 min 24 s, sans achever le guide demandé. Celui de 1 024 échoue sous la garde conservatrice.

Les priorités suivantes découlent des observations : conserver le réemploi exact du KV déjà mesuré ; livrer plus de contenu utile dans le budget de sortie par une concision explicitement demandée ; travailler sur le coût de préremplissage et les lectures répétées avec comparaisons appariées ; élargir ensuite les cas difficiles et les ambiguïtés sans confondre concordance numérique et vérité. Le repli rare du routeur apporte surtout la stabilité, avec un coût mesuré trop petit pour expliquer la latence globale. Aucune accélération future, meilleure quantification ou performance sur SSD externe n’est annoncée avant mesure.
