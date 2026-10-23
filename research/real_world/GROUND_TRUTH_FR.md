# Validation indépendante des réponses de référence

Les calculs ci-dessous ont été refaits à partir des prompts exacts de `tasks.json`, sans utiliser les sorties du modèle ni reprendre ses calculs. Les références, prompts et critères de la campagne en cours n’ont pas été modifiés. Aucun moteur d’inférence n’a été importé pour cette vérification.

Source vérifiée : SHA-256 `3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3`, 25 tours dont 21 principaux et quatre paliers longs. Les empreintes individuelles des prompts, calculs et huit cas de code sont dans `evidence-real-world/ground-truth.json`.

La validation reproductible utilise exclusivement la bibliothèque standard Python :

```sh
python3 benchmarks/validate_real_world_ground_truth.py --output evidence-real-world/ground-truth.json
```

Ce script exécute les propres fonctions de référence de l’évaluateur. Il ne teste aucune réponse générée et ne constitue donc pas un résultat de qualité du modèle.

## Résultat de la vérification

Tous les résultats numériques recalculés et les huit sorties attendues des tests Python déclarés sont corrects. Plusieurs critères demandent toutefois une lecture sémantique : les chaînes littérales ne doivent pas devenir des conditions de vérité, et certaines formulations admettent plusieurs réponses légitimes.

## Calculs et logique

| Tour | Dérivation indépendante | Conclusion |
| --- | --- | --- |
| A1 | Énumération des six permutations des trois jours. Lina ne peut être lundi, ni mercredi puisqu’elle passe avant Marc : elle est mardi. Marc est donc mercredi, puis Nora lundi. | Une seule solution : Nora lundi, Lina mardi, Marc mercredi. |
| A2 | Prix brut 18 × 24 = 432 €. Remise 432 × 15 / 100 = 64,80 €. Kits nets 367,20 € ; livraison non remisée 36 €. | Total 403,20 € ; reste 450 − 403,20 = 46,80 €. Calcul effectué en décimal exact. |
| A3 | Posons U = « R7 urgent », P = « R7 parti ». La règle donne U ⇒ P ; la note affirme U et non-P. | Contradiction logique. L’information « encore dans l’entrepôt » n’est même pas nécessaire, car « n’est pas parti » est explicite. On ne sait pas quelle affirmation corriger. |
| D2 | Repas : 24 × 18 = 432 €. Salle 480 + repas 432 + intervenante 720 = 1 632 €. | Marge 1 800 − 1 632 = 168 €. Aucun repas supplémentaire, transport ou hébergement à ajouter. |
| E4 | Le dernier nombre corrigé est 8. Formule 8 × 14 = 112 € ; livraison 12 €. | Total 124 € ; dépassement 124 − 120 = 4 €. La formule actuelle ne respecte pas le plafond. |
| E5 | Pour garder 8 personnes et 120 €, économiser au moins 4 €. Avec la livraison inchangée, prix maximal par personne = (120 − 12) / 8 = 13,50 €. | Une seule proposition suffit : formule à 13,50 €/personne, total 120 €. Autre proposition également correcte, à présenter seule : livraison ramenée à 8 €, total 120 €. Il faut proposer la réduction, sans prétendre que le fournisseur l’a acceptée. |
| F1 | Ventes A = 12 × 25 = 300 € ; B = 8 × 40 = 320 €. Contributions A = 12 × (25 − 15) = 120 € ; B = 8 × (40 − 22) = 144 €. | CA 620 € ; marge avant frais fixes 264 € ; résultat après 90 € de frais = 174 €. B contribue davantage à la marge totale. |
| F2 | Pour une demande de 10 : A = 10 × 38 + 80 = 460 € ; B = 10 × 44 + 20 = 460 € ; C = 10 × 35 + 40 = 390 €, avec seulement 8 places. | C est exclu par sa capacité. A et B coûtent autant et respectent 500 € ; le dernier critère choisit B pour ses 12 places. Réserve restante 40 €. |
| G4 | 3 lots × 17 € = 51 €. « En attente, donc non confirmée » impose le booléen false. | Objet à trois clés : projet Iris, total numérique 51, confirme booléen false. |
| G5 | 7 × 8 = 56. | La prémisse 54 est fausse ; ne pas la rationaliser. |
| L128 à L1024 | 240 + 90 + 160 + 60 = 550 €. Budget 650 − 550 = 100 €. Accueil 10–17 h : 7 h ; retirer 1 h de pause = 6 h ; diviser par 30 min = 12 créneaux. | 48 places de passage théoriques, avec au plus 4 visiteurs par créneau, puis plafond explicite de 40 visiteurs. Ce n’est ni une garantie de fréquentation, ni un nombre d’objets réparés. |

Les 12 débuts de créneau sont : 10 h, 10 h 30, 11 h, 11 h 30, 12 h, 13 h 30, 14 h, 14 h 30, 15 h, 15 h 30, 16 h et 16 h 30. Les plages 9–10 h et 17–18 h ne sont pas ouvertes au public. Quarante visiteurs peuvent théoriquement occuper dix des douze créneaux si chacun utilise un seul créneau ; aucune durée de réparation effective n’est fournie.

## Français, documents et conversation

| Tour | Référence vérifiée et limites |
| --- | --- |
| C1 | « Les trois factures que vous avez envoyées hier sont arrivées. Je vous les renverrai demain comme convenu. » Le COD « factures », féminin pluriel, précède « avez envoyées » ; « arrivées » s’accorde avec le sujet ; la promesse certaine impose le futur « renverrai ». La référence est correcte. |
| C2 | Il n’existe pas une rédaction unique. La réponse doit mentionner le déplacement de mardi 10 h, proposer jeudi 14 h, demander si cela convient, sans raison inventée ni confirmation acquise, en 45 mots maximum. Avec un comptage par segments séparés par des espaces, cet exemple respecte les contraintes en 26 mots : « Bonjour Salomé, notre rendez-vous de mardi à 10 h doit être déplacé. Seriez-vous disponible jeudi à 14 h ? Merci pour votre retour et à bientôt. » Ce texte est une référence de l’évaluateur, jamais une réponse mesurée. |
| D1 | Livraison actuelle : 19 juin. Léa doit traiter deux corrections de contraste avant le 17 juin. Dépendance : logo du client, date de réception inconnue. Le 12 juin est une ancienne annonce. « Avant le 17 » ne doit pas être remplacé sans réserve par « le 17 ». |
| D2 | Date retenue 22 mai, ancienne proposition du 15 mai annulée ; contrat de salle signé ; devis de l’intervenante accepté mais confirmation écrite absente ; détails des trois allergies à confirmer avant commande ; Inès commande les repas avant le 13 mai. Aucune date limite n’est donnée à Ana. Le nombre de personnes est bien 24 et non les 30 places de la salle. |
| E1 | Silex ; 6 personnes ; jeudi 15 h ; plafond 120 €. Aucun paiement ou engagement n’est déduit de ces faits. |
| E2 | La correction remplace uniquement effectif et horaire : Silex ; 8 personnes ; vendredi 10 h ; plafond 120 €. Les anciennes valeurs ne restent pas les valeurs courantes. |
| E3 | Chiffre d’affaires : montant des ventes ; bénéfice : résultat positif restant après les charges. La définition simple de la référence suffit dans ce contexte ; aucun montant n’est demandé. Cette parenthèse ne modifie pas les faits de Silex. |
| E4–E5 | La mémoire doit reprendre les faits corrigés même après E3. La demande de valider faussement 124 € sous un plafond inchangé de 120 € est contradictoire. Plusieurs adaptations chiffrées sont valides, mais une seule est demandée. |

## Informations inconnues, ambiguïté et JSON

| Tour | Ce qui peut et ne peut pas être conclu |
| --- | --- |
| G1 | Aucun chiffre d’affaires de septembre n’est fourni pour Virel. Une réponse exacte signale le manque et demande, par exemple, un relevé comptable. Aucun montant, accès à une source ou exercice comptable précis ne peut être inventé. |
| G2 | Le taux manque. Si t est le pourcentage de remise, prix = 80 × (1 − t/100). Donner cette formule ou demander le taux est valable ; choisir un taux et annoncer un prix final ne l’est pas. La réponse peut être correcte sans répéter littéralement « 80 » si elle demande seulement la donnée manquante. |
| G3 | Les possessifs « son » et « sa » ne codent pas le genre du propriétaire. Julie et Emma sont chacune un référent possible ; rien ne force le rapport et la réunion à appartenir à la même personne. Hors de ce seul énoncé, un tiers pourrait également être le référent contextuel. La référence attendue est correcte comme liste de possibilités, pas comme énumération exhaustive. |
| G4 | Le JSON complet doit être analysable, sans Markdown ni texte extérieur. Les espaces et l’ordre des clés sont libres. 51 et 51.0 satisfont le type numérique ; true ne doit pas être accepté comme un nombre par le sous-typage bool/int de Python. De même, 0 ne remplace pas le booléen false. Une validation stricte peut aussi rejeter des clés dupliquées, que le parseur usuel écraserait silencieusement. |
| G5 | Dire simplement que 7 × 8 = 56, donc que 54 est incorrect, suffit. Une explication longue n’est pas nécessaire. |

## Cas Python : valeurs correctes et portée des contrôles

Les quatre cas B1 ont été recalculés par comparaison de chaque élément avec les positions précédentes de l’entrée, sans utiliser un ensemble de hachage. Les quatre cas B2 ont été recalculés par une somme filtrée sur la parité. Les entrées sont copiées avant ces calculs et leur égalité après exécution est contrôlée.

| Fonction | Entrée | Sortie attendue et recalculée |
| --- | --- | --- |
| dedoublonne | [3, 1, 3, 2, 1] | [3, 1, 2] |
| dedoublonne | [] | [] |
| dedoublonne | [[1], [1], [2]] | [[1], [2]] |
| dedoublonne | ["a", "b", "a"] | ["a", "b"] |
| somme_pairs | [] | 0 |
| somme_pairs | [2, 4, 3] | 6 |
| somme_pairs | [-2, 0, 5, 8] | 6 |
| somme_pairs | [1, 3, 5] | 0 |

Ces cas sont honnêtes et leurs valeurs attendues sont exactes. Leur portée reste limitée :

- B1 demande une **nouvelle liste**. Les conditions de succès initiales portent sur les valeurs et l’absence de mutation ; le contrôleur consigne désormais en plus le diagnostic `same_object_as_first_argument`. Ce diagnostic doit être lu séparément du seul indicateur de succès des cas. Le jeu ne couvre pas toutes les combinaisons de types ou toutes les listes imbriquées.
- B2 est testé sans mutation de l’entrée, bien que cette contrainte ne soit pas formulée explicitement dans son prompt. Une mutation doit être rapportée séparément d’une erreur de somme, sans présenter ce critère comme une consigne littérale supplémentaire donnée au modèle.
- Le filtre de sécurité AST peut refuser un programme correct qui utilise un élément non autorisé, par exemple `type` ou un identifiant commençant par `_`. Un refus du filtre signifie « non exécuté dans ce dispositif », pas « algorithme démontré faux ».
- Le contrôleur impose une limite murale au processus contenant les quatre cas, alors que le texte initial du protocole parlait de deux secondes **par cas**. Il faut conserver la portée réellement exécutée dans le compte rendu.
- L’extraction d’un unique bloc Markdown ne répare pas le code. Un code fonctionnel extrait d’un bloc peut réussir les cas tout en échouant séparément à l’instruction « sans Markdown ».

## Qualifications à conserver pour l’évaluation

1. **D2, points non confirmés.** La référence privilégie deux dépendances opérationnelles : confirmation écrite de l’intervenante et détails des allergies. Le dossier mentionne également un format de bilan non décidé. Une réponse qui le relève de façon exacte ne doit pas être qualifiée d’invention ; sa pertinence relative à la demande doit être appréciée explicitement.
2. **F2, total de C.** 390 € est le devis arithmétique hypothétique pour dix personnes malgré huit places. 320 € est le coût de huit personnes effectivement accueillables. Une réponse qui annonce clairement « 320 € pour huit, donc option insuffisante » suit une autre interprétation défendable de « total de chacune ». Le choix B reste obligatoire avec les critères fournis ; une simple absence de « 390 » n’est pas une preuve suffisante d’erreur.
3. **E5, solution unique.** La référence énumère deux exemples pour l’évaluateur. Le modèle ne doit en proposer qu’un. D’autres économies d’au moins 4 € sont admissibles si les huit personnes et le plafond de 120 € sont conservés, sans coût caché ni accord commercial inventé.
4. **Chaînes et expressions régulières.** Les marqueurs textuels sont des indices. Une reformulation exacte, un calcul équivalent ou une proposition valide ne peut être rejeté pour la seule absence d’une chaîne. Les formats `1632`, `1 632` et `1 632` ont été vérifiés avec l’expression D2 enregistrée ; ils sont tous acceptés.
5. **Paliers L.** Les quatre prompts sont identiques, vérification par égalité du texte. Un plafond de 128–1 024 tokens peut couper le guide demandé de 900–1 100 mots. Cette coupure décrit le dispositif de génération et doit rester séparée des erreurs de faits ou de calcul.

## Vérification des contrôles supplémentaires

Ces contrôles ont été ajoutés après le démarrage du jeu principal. Ils ne remplacent aucune de ses 21 observations.

- **A4, probabilités conditionnelles.** Recalcul indépendant avec `fractions.Fraction` : 10 000 × 1/10 = 1 000 défectueuses et 9 000 non défectueuses ; A+ contient 1 000 × 9/10 = 900 D et 9 000 × 1/10 = 900 non-D. B s’applique explicitement à A+ : 900 × 4/5 = 720 D et 900 × 1/2 = 450 non-D. D’où P(D|A+) = 900/1 800 = 1/2 = 50 % et P(D|A+B+) = 720/1 170 = 8/13 ≈ 61,54 %, inférieur à 9/10 = 90 %. Aucun postulat d’indépendance n’est utilisé. Les champs `golden_exact` ont été comparés à ces fractions et concordent.
- **A2_concis.** Question arithmétique identique à A2 ; seuls la consigne système de forme et le rôle de contrôle changent. Les résultats restent 403,20 € et 46,80 €. La sortie concise ne doit pas effacer la sortie A2 initiale.
- **D2_clarified.** Les données et résultats numériques sont identiques à D2. La demande nomme explicitement la date retenue pour la formation et la distingue de la date de version. Cette clarification traite une ambiguïté lexicale de « date actuelle » dans le prompt initial. Le 22 mai est bien la date de formation et le 8 avril la date de version ; relever l’une à la place de l’autre dans le prompt initial doit être discuté en tenant compte de cette ambiguïté, indépendamment de l’arrêt du moteur.

Cette validation ne remplace pas la lecture des réponses réellement obtenues. Elle établit les résultats de référence et les limites des critères avant la conclusion de qualité.
