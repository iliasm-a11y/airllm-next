# Évaluation pratique d’AirLLM Next — protocole synthétique

Ce document décrit le jeu d’essai, pas des résultats. Les réponses évaluées doivent être produites par le modèle réellement exécuté dans AirLLM Next. Les réponses de référence présentes dans `tasks.json` servent à contrôler les sorties ; elles ne doivent jamais être transmises au modèle ni remplacer ses réponses.

## Périmètre du jeu

Toutes les personnes, organisations, documents et données des prompts sont fictifs. Le jeu ne dépend ni d’un accès Internet ni de connaissances récentes. Il mesure une utilité pratique sur un petit ensemble ciblé, sans prétendre estimer la qualité générale du modèle ni un score de benchmark représentatif.

| Domaine | Tours | But |
|---|---:|---|
| A — raisonnement | 3 | Logique à solution unique, remise et frais fixes, contradiction |
| B — programmation | 2 | Fonction Python, correction d’un bug ; quatre cas exécutables par fonction |
| C — français | 2 | Accords et futur, rédaction brève et naturelle |
| D — documents | 2 | Note courte, dossier plus long avec statut et calcul |
| E — conversation | 5 | Mémoire, correction, parenthèse, retour au contexte, demande contradictoire |
| F — tâches complexes | 2 | Indicateurs commerciaux, arbitrage sous contraintes |
| G — fiabilité | 5 | Inconnu, donnée manquante, ambiguïté, JSON strict, fausse prémisse |
| L — génération longue | 4 | Même prompt, plafonds 128, 256, 512 et 1 024 tokens |
| Total | 25 | 21 tours fonctionnels et 4 diagnostics de longueur |

**Historique — paramètres initiaux du pilote, non courants.** Il y a 21 sessions : chaque tâche est indépendante, sauf les cinq tours de la session E. Les plafonds des tâches principales vont de 64 à 160 nouveaux tokens ; leur somme est 2 096 tokens. Les quatre paliers ajoutent au maximum 1 920 tokens, soit 4 016 tokens générés au maximum pour le jeu complet. Un EOS naturel peut réduire ce total. À titre de planification uniquement, 4 016 tokens à une vitesse constante hypothétique de 1 token/s représentent environ 67 minutes de décodage ; ce calcul exclut initialisation, préremplissage et surcoûts, et n’est pas une mesure de performance.

**Paramètres courants de la campagne principale, après amendement du 23 octobre 2026 :** les 21 demandes fonctionnelles ont chacune un plafond de 256 nouveaux tokens, soit 5 376 tokens au maximum. Les quatre paliers longs ajoutent 1 920 tokens au maximum, pour un total prévu de 7 296 tokens hors pilote et contrôles additionnels. Les paramètres initiaux ci-dessus sont conservés pour retracer le changement ; ils ne décrivent pas la campagne principale exécutée.

## Schéma et séparation des données

`tasks.json` est une liste de sessions :

- `id` : identifiant stable de session ;
- `domain` : famille de tâche ;
- `system` : message système effectivement utilisé ;
- `turns` : messages utilisateur, chacun avec `id`, `prompt`, `max_new_tokens` et `checks` ;
- `diagnostic: true` : session de la montée en longueur ; `ladder_cap` rappelle son plafond.

`checks` contient un objectif, la réponse attendue et, selon le cas, des critères textuels, un objet `json_exact`, des `python_tests` et une `rubric` de lecture. Les critères textuels sont des indices : l’absence d’une chaîne littérale ne démontre pas seule une erreur si le sens est correct, et sa présence ne démontre pas seule une réponse correcte. Les critères ne sont pas des consignes supplémentaires pour le modèle.

Seuls `system` puis les `prompt` alimentent la conversation. Les réponses de référence, tests exécutables, critères et notes restent hors du contexte. En E, conserver successivement tous les prompts et toutes les réponses réellement générées ; ne pas remplacer une réponse fausse par une réponse corrigée. Réinitialiser l’historique entre sessions et paliers L.

## Préparation du moteur

Avant le premier résultat de qualité, consigner et vérifier :

1. Le chemin du modèle, sa taille et son empreinte, la révision de code, le backend, le matériel, les limites mémoire et la configuration effective.
2. Le véritable tokenizer du modèle, son template de conversation, l’encodage des rôles système/utilisateur/assistant et les identifiants spéciaux d’arrêt.
3. Le prompt rendu et ses identifiants de tokens, ou leurs artefacts exacts ; l’assistant doit commencer au bon emplacement du template.
4. Une conversation de plusieurs tours correctement sérialisée et un arrêt naturel sur les tokens de fin pertinents.
5. La capacité de contexte totale : prompt rendu, historique et réserve de génération. Compter les tokens avec ce tokenizer avant exécution. Ne pas tronquer silencieusement un document ou un historique pour faire rentrer une tâche.

Utiliser une configuration fixe et enregistrée, de préférence greedy pour ce premier diagnostic. Le choix du template, des paramètres propres au modèle et des tokens d’arrêt doit venir de sa configuration réelle ; ne pas inventer un format. Les plafonds de `tasks.json` ne sont pas des longueurs obligatoires. Garder EOS actif. Ne pas ajouter de pénalité, de prolongation ou de réécriture non déclarée pour obtenir une réponse plus flatteuse.

Si un plafond dépasse les ressources ou le contexte effectivement pris en charge, conserver le motif de non-exécution ou d’interruption. Il ne constitue pas une réponse du modèle. Ne pas prétendre avoir validé ce palier à partir d’un essai plus court.

## Ordre et conditions de mesure

Exécuter les tâches principales dans l’ordre du fichier, puis les paliers L dans l’ordre croissant. Distinguer la première requête après ouverture du moteur des requêtes suivantes. Une relance après erreur reste une relance distincte, avec sa raison et son numéro d’essai ; ne pas écraser l’échec.

Pour chaque tour, enregistrer : prompt exact, historique complet, template rendu, paramètres, réponse intégrale, tokens de sortie, motif d’arrêt, temps d’initialisation, temps jusqu’au premier token, durée de réponse, temps de décodage, débit de décodage, mémoire, swap, compteurs de lecture disponibles et erreurs. En E, noter le nombre de tokens d’entrée à chaque tour pour suivre le coût de l’historique.

Définir les horodatages avant de calculer les débits. Distinguer soumission → premier token, premier → dernier token et soumission → fin. Indiquer si le débit inclut le premier token ou le préremplissage. Un débit calculé sur un seul token ne décrit pas un décodage soutenu. Conserver aussi le délai ressenti d’ouverture du moteur jusqu’à la première réponse terminée.

Étiqueter les compteurs disponibles selon leur portée : processus, système, octets logiques ou lectures physiques. Une lecture via mmap et un cache système chaud ne prouvent pas un débit physique SSD. Sans mesure fiable, noter « indisponible ». Mesurer l’état initial, les pics observés et l’état final de mémoire/swap ; ne pas confondre mémoire réservée, mémoire résidente et occupation des caches.

## Contrôles objectifs

Les résultats mathématiques ont été calculés indépendamment pour préparer les références :

| Tour | Résultat contrôlable |
|---|---|
| A1 | Unique solution : Nora lundi, Lina mardi, Marc mercredi |
| A2 | 432 × 0,85 + 36 = 403,20 € ; reste 46,80 € |
| D2 | 480 + 24 × 18 + 720 = 1 632 € ; marge 168 € |
| E4–E5 | 8 × 14 + 12 = 124 € ; dépassement 4 € sur 120 € |
| F1 | CA 620 € ; marge 264 € ; résultat 174 € ; contributions A 120 €, B 144 € |
| F2 | A 460 €, B 460 €, C 390 € ; C exclu ; B choisi à coût égal |
| G4 | JSON exact : projet Iris, total numérique 51, confirme booléen false |
| G5 | 7 × 8 = 56 |
| L | Dépenses 550 €, marge 100 € ; 12 créneaux de 30 min hors pause, capacité théorique 48 limitée à 40 visiteurs |

A1 a été vérifié par énumération des six permutations. Les deux fonctions B doivent être exécutées sur les cas de `python_tests`, sans correction manuelle préalable. Tester également l’absence de mutation de l’argument. Chaque fonction est vérifiée dans un processus isolé avec ses quatre cas : le délai mural maximal de deux secondes couvre le démarrage et l’ensemble de ces cas, sans délai séparé de deux secondes par cas. Une limite de trois secondes CPU est appliquée séparément. Une exception, un mauvais résultat, une mutation ou un dépassement observé fait échouer ce contrôle. Exécuter les sorties dans un environnement isolé et temporaire, sans donnée personnelle ni secret, en conservant le code brut et le résultat des tests. Une extraction automatique de blocs Markdown peut être enregistrée pour diagnostiquer un code exécutable, mais le non-respect de la demande « sans Markdown » reste un défaut de format séparé ; ne pas faire passer un correctif de code pour une simple extraction.

Pour G4, analyser la sortie complète comme JSON et imposer les trois clés exactes. Le type de `total_euros` doit être numérique, celui de `confirme` booléen. Un parseur Python ne doit pas accepter `0` comme `false` par simple égalité, ni `true` comme un nombre. Les espaces de présentation sont libres ; aucun texte extérieur n’est permis.

Pour C2, compter les mots selon une règle explicite et fixe, par exemple les segments séparés par espaces. Examiner séparément les faits imposés, la proposition non confirmée et l’absence de raison inventée. Une appréciation du ton reste partiellement subjective.

## Lecture indépendante de chaque réponse

Lire la sortie complète et attribuer cinq notes distinctes de 0 à 5 : exactitude factuelle, respect des instructions, cohérence, complétude, utilité réelle. Ajouter les résultats objectifs et une justification courte liée à des passages de la réponse. Une chaîne attendue présente dans un raisonnement contradictoire ne mérite pas automatiquement une bonne note.

Repères : 0 = inutilisable ou entièrement contraire au critère ; 1 = défauts majeurs ; 2 = résultat très incomplet ou plusieurs erreurs ; 3 = utile avec limite réelle ; 4 = correct avec défaut mineur ; 5 = satisfait pleinement le critère dans ce périmètre. Ne pas fabriquer de précision objective pour une appréciation de style.

Conserver les échecs et les réponses tronquées. Distinguer : réponse terminée naturellement, plafond atteint, contexte insuffisant, arrêt de protection et erreur d’exécution. Aux paliers L, évaluer répétitions, cohérence, dérive et respect des faits sur la portion réellement produite ; la complétude du guide est séparée. Un plafond atteint ne démontre pas un arrêt prématuré du modèle. Un EOS observé avant le plafond ne démontre pas seul un défaut : lire si la réponse est terminée.

## Diagnostic et limites d’attribution

Une mauvaise réponse constate une défaillance de la configuration testée. Elle n’identifie pas automatiquement le responsable. Séparer les hypothèses : modèle, tokenizer/template, génération, historique, KV cache, poids/quantification, chargement et exécution CPU/GPU. Attribuer une cause seulement après une comparaison contrôlée ou une preuve spécifique ; sinon la laisser ouverte.

Les quatre paliers L réutilisent exactement le même prompt et ne diffèrent que par leur plafond. Si le décodage est déterministe et le moteur réinitialisé de façon équivalente, comparer leurs préfixes communs et signaler toute divergence ; ne pas lui attribuer d’emblée une cause. Chaque palier conserve ses propres mesures et son motif d’arrêt.

Une campagne réussie sur ces tâches et cette machine ne prouve ni la qualité sur d’autres modèles, ni le comportement d’un SSD externe, ni une absence de swap hors des fenêtres observées, ni une endurance thermique longue. Le rapport doit distinguer ce qui a été exécuté, ce qui a été vérifié et ce qui reste non mesuré.

## Amendement du 23 octobre 2026 après le pilote

Le premier passage, conservé dans `evidence-real-world/pilot-short-caps.json` avec ses réponses et sa revue `pilot_review.json`, a montré deux coupures au plafond de 96 tokens : A1 donnait le bon calendrier sans achever sa justification ; A2 n’avait encore fourni aucun des deux résultats finaux. Ces observations ne suffisent pas à conclure à une erreur du modèle. Elles montrent que les plafonds initiaux de la campagne limitaient aussi sa capacité à terminer une présentation pourtant pertinente.

La campagne principale reprend donc exactement les 21 demandes fonctionnelles avec un plafond uniforme de **256 nouveaux tokens par tour**, sans changement du fond des prompts, des critères de référence ou du message système. Les paliers L restent 128, 256, 512 et 1 024. Les anciens paramètres sont conservés dans `pilot-short-caps-tasks.json`. Le plafond principal est désormais 5 376 tokens et celui du jeu principal avec paliers 7 296 tokens, hors sorties du pilote. Cela représente environ 122 minutes de décodage à la vitesse hypothétique constante de 1 token/s, toujours hors autres coûts.

Cet ajustement est postérieur aux sorties A1 et A2 et n’est pas présenté comme un choix prédéfini à l’aveugle. Les réponses du pilote restent visibles et ne sont pas remplacées par celles de la reprise. La revue principale identifie les nouvelles réponses par fichier et empreinte ; aucun score n’est calculé pour un tour dépourvu de réponse achevée. Le pilote et la reprise doivent être distingués dans toute synthèse de qualité et de performance.

### Précisions de correction avant lecture de D2, G3 et B1

D2 demande deux points non confirmés sans imposer qu’ils soient opérationnels. La confirmation écrite de l’intervenante et les détails des allergies forment la référence principale, mais le format de bilan encore non décidé constitue aussi une réponse étayée par le dossier. Accepter deux points réellement non déterminés cités fidèlement ; ne pas transformer la référence principale en liste exclusive. L’échéance absente pour Ana peut être signalée comme information manquante, mais ne doit pas être présentée comme un retard constaté.

Pour G3, Julie et Emma sont les deux référents saillants ; la phrase ne permet pas de décider. Une demande de clarification fidèle est correcte sans liste exhaustive des possesseurs possibles. Ne pas pénaliser une réponse qui signale également qu’un tiers serait possible hors contexte.

Pour B1, la consigne « nouvelle liste » impose une liste résultat différente de l’objet liste d’entrée. Les comparaisons de valeurs et la vérification d’absence de mutation ne suffisent pas à établir cette propriété ; contrôler aussi `resultat is not entree` sans modifier le code du modèle. Aucun clonage profond des éléments n’est exigé par le prompt.

Pour F2, C à 390 € est le calcul théorique pour les dix participants, malgré sa capacité insuffisante. C à 320 € est également défendable si la réponse précise qu’il s’agit de huit places seulement (8 × 35 + 40) et exclut C pour le groupe de dix. Le choix B demeure unique après application des critères. Une interprétation explicitée de cette ambiguïté ne constitue pas une erreur arithmétique.


## Contrôle additionnel A4 déclaré avant exécution

Déclaration enregistrée le 2026-10-23T19:01:43.890383+00:00 (UTC), avant toute réponse du moteur sur A4. Le fichier séparé `hard-reasoning-tasks.json` ajoute un seul problème conditionnel de Bayes à la campagne : session `A4_bayes`, tour `A4`, domaine A, même message système générique, plafond de 256 nouveaux tokens, quatre lignes maximum et sans préambule. Les fixtures principales ne sont pas modifiées.

Les effectifs et les fractions de référence ont été calculés exactement avec `fractions.Fraction` : A+ contient 900 pièces défectueuses et 900 non défectueuses ; A+B+ contient 720 défectueuses et 450 non défectueuses ; les probabilités demandées sont 1/2 et 8/13 (environ 61,54 %). Le seuil de 90 % n’est pas atteint. Les taux du second test sont conditionnels à A+, et aucune indépendance n’est supposée.

Ce contrôle est ajouté après les premières observations de la campagne, avec son statut explicitement additionnel. Il sera conservé et évalué quelle que soit sa réussite ou son échec ; il ne remplace ni A2 ni D2 et ne sera pas intégré rétroactivement au nombre initial de tâches prédéclarées. Aucune inférence n’a été exécutée lors de sa préparation.

### Limite de formulation de D2 reconnue après relecture

Le critère prédéclaré de D2 attend la date de formation du 22 mai. Toutefois, la question originale demandait « date actuelle », tandis que le dossier commençait par « version du 8 avril ». Cette formulation est réellement ambiguë : citer le 8 avril reprend une donnée vraie du texte et ne démontre pas une invention du modèle. La référence au créneau annulé favorise l’intention de calendrier, sans supprimer entièrement cette ambiguïté. La revue doit donc distinguer l’écart au critère prévu de l’attribution incertaine de cet écart au modèle ou au protocole. Le diagnostic `D2_clarified`, préparé séparément, explicite la date de la formation ; il ne remplace pas la réponse originale.


### Correction descriptive du délai des contrôles Python — 23 octobre 2026

La version initiale de ce protocole décrivait à tort un délai de deux secondes « par cas ». Le vérificateur effectivement exécuté applique ce délai mural au processus complet de chaque fonction, démarrage et quatre cas inclus, avec une limite CPU distincte de trois secondes. La description ci-dessus est harmonisée avec cette mise en œuvre. Les huit cas connus ont été exécutés dans ces conditions ; les preuves brutes ne sont pas modifiées.

### Verdicts descriptifs pour le décompte des tâches

Chaque réponse achevée reçoit, en plus de ses cinq notes distinctes, un verdict explicite : `reussi` si les résultats essentiels et contraintes sont satisfaits ; `partiel` si un résultat essentiel utile est livré mais qu’un autre résultat, une justification ou une contrainte manque ; `echec_critere` si le résultat essentiel est faux ou si aucun résultat essentiel n’est livré pour une demande non ambiguë ; `indetermine` si l’ambiguïté du protocole ou une preuve insuffisante empêche un jugement univoque. Les exceptions et absences de réponse restent des statuts d’exécution distincts. Ces catégories ne sont ni une note composite ni un remplacement des cinq axes. Le décompte final distingue strictement réussites, partiels, écarts certains et indéterminés ; D2 ambigu ne devient pas une erreur certaine du modèle.


### Contrôle de concision F2 déclaré après observation

Le 2026-10-23T19:30:50.740404+00:00 (UTC), un contrôle séparé `F2_concis` est préparé dans `f2-concision-tasks.json`, après observation de la présentation très développée du F2 principal. Il conserve exactement le même message utilisateur et le plafond de256 tokens ; seule la phrase système « Réponds sans préambule, en quatre lignes maximum. » est ajoutée. Aucune inférence n’est lancée lors de cette préparation.

La réponse principale sera conservée et évaluée intégralement. Le contrôle cherche l’effet d’une contrainte de concision sur la livraison sous plafond ; il ne prouve pas à lui seul une capacité ou incapacité arithmétique et ne remplace jamais F2 dans les21 demandes principales. Son résultat sera évalué dans la revue de diagnostics, réussite ou échec.
