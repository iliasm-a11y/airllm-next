# Audit final des calculs de l’analyseur

Audit arrêté au 2026-10-23T23:12:25+02:00. Les comptes et durées ont été recalculés séparément à partir des JSON bruts, avec les seuls entiers Python et `math.fsum`, puis comparés aux fonctions de l’analyseur à une tolérance relative de 10⁻¹². Aucun import Torch, MLX, NumPy ou Matplotlib, aucune inférence et aucune modification du moteur ou du harness n’ont eu lieu pendant cet audit.

Le résultat machine et les SHA-256 des entrées figurent dans [analyzer-audit.json](../../evidence-real-world/analyzer-audit.json). Le SHA-256 de l’analyseur audité est `895093d03490169093fcea13d92f5932d864ef0572170f40fc64c45bb451f0fb`. L’ancien instantané à 14 tours est conservé dans [analyzer-audit-initial14.json](../../evidence-real-world/analyzer-audit-initial14.json).

## Conclusion et périmètre

Aucun défaut de calcul n’a été constaté sur les résultats finaux vérifiés. La campagne principale sélectionne **21 IDs uniques achevés**, puis A4 reste un supplément distinct ; la ligne combinée est explicitement à 22. Les 35 observations achevées des différents essais ont leurs débits de décodage recoupés. Les 38 annotations sont liées au bon texte exact et au bon fichier. Les fragments interrompus sont diagnostiques et exclus des agrégats de résultats achevés.

Cette vérification valide les calculs et les rattachements des preuves. Elle ne transforme pas une note qualitative en vérité objective et ne démontre pas un gain causal universel de performance.

## Trois recoupements numériques indépendants

### 1. Campagne principale, rapports des sommes

Sélection : les huit tours achevés A1–D1 de `campaign.json`, puis les 13 tours achevés D2–G5 de `campaign-resumed.json`. Le D2 initial échoué et `campaign-before-router-fix.json` ne sont pas comptés deux fois.

| Calcul | Recalcul indépendant | Concordance |
| --- | --- | --- |
| Tours / tokens de sortie | 21 / 1865 | Oui |
| Somme des durées murales de requête | 3 080,655897377968 s | Oui |
| Débit global, préremplissage compris | 1865 / 3 080,655897377968 = 0,6053905604 token/s | Oui |
| Tokens de décodage, premier token exclu à chaque tour | 1865 − 21 = 1844 | Oui |
| Somme des durées moteur après le premier token | 2 011,648530169041 s | Oui |
| Débit agrégé de décodage | 1844 / 2 011,648530169041 = 0,9166611226 token/s | Oui |

La moyenne non pondérée des 21 débits individuels donnerait 0,9099334488 token/s. Cette moyenne n’est pas utilisée pour l’agrégat : le rapport des sommes vaut 0,9166611226 token/s.

Avec A4, les **22** tâches produisent 1972 tokens en 3 293,986053877969 s murales de requête, soit 0,5986667726 token/s ; le décodage vaut 1950 / 2 127,288216294037 = 0,9166599923 token/s. Le pilote et les contrôles diagnostiques ne remplacent aucune réponse principale.

### 2. Domaine A, trois tâches initiales

| Calcul | Recalcul indépendant | Concordance |
| --- | --- | --- |
| Tokens A1 + A2 + A3 | 118 + 256 + 73 = 447 | Oui |
| Temps mural total | 602,731052540992 s | Oui |
| Débit total | 447 / 602,731052540992 = 0,7416243084 token/s | Oui |
| Décodage | 444 / 463,54445320899 = 0,9578369387 token/s | Oui |

A4 est présenté séparément. Ses notes liées au texte exact sont 5/4/5/5/5 ; le 4 en respect des instructions concerne l’absence de forme a/b pour la première probabilité, tandis que le calcul est correct.

### 3. Intervalles TTFT et durée complète

Les trois cas suivants ont été recalculés : premier tour du processus initial, premier tour du processus de reprise, puis tour avec réutilisation KV.

| Tour | TTFT moteur (s) | Lancement → prêt ajouté (s) | Borne basse (s) | Borne haute (s) | Lancement → prêt + requête entière (s) |
| --- | --- | --- | --- | --- | --- |
| A1 | 46,329511208001 | 1,903273417003 | 48,232784625005 | 48,233681292004 | 166,392913709002 |
| D2 | 189,825855082992 | 1,804422249988 | 191,630277332981 | 191,632191748984 | 283,398243915988 |
| E2 | 20,5293465 | 0 | 20,5293465 | 20,530880458988 | 57,311456999989 |

La borne basse ajoute le lancement → prêt au TTFT moteur seulement pour la première requête du processus. La borne haute ajoute encore `max(0, response_wall_seconds − total_seconds)`, attribuant par prudence au temps avant premier token tous les frais hors fenêtre moteur, y compris ceux réellement situés après génération. Le calcul concorde dans les trois cas.

Cet encadrement concerne la disponibilité du premier token synchronisé dans le périmètre enregistré. Il ne mesure ni rendu à l’écran, ni réseau, ni gestes humains. La somme « lancement → prêt + requête » ne réintroduit pas les petits intervalles entre ces deux fenêtres et n’est pas un temps humain de bout en bout.

## Durées, stockage et mémoire

- Les 35 débits par tour concordent avec `(nombre de tokens − 1) / (total_seconds − ttft_seconds)`. Un token EOS effectivement échantillonné est compté ; les tokens ne sont pas des mots visibles.
- `total_seconds − ttft_seconds` inclut les callbacks de sauvegarde postérieurs au premier token : le débit est instrumenté. `token_seconds`, utilisé par les figures optionnelles, exclut ces callbacks et décrit donc un autre périmètre.
- Le jeu principal totalise **4,470,667,232,384 octets logiques** et **1,374,366,089,216 octets de stockage imputés au processus**. Ces sommes concordent avec l’analyseur. Avec A4 : 4,756,434,514,048 et 1,482,789,134,336 octets respectivement.
- « Logique » compte les demandes du lecteur ; « stockage » est le compteur OS du processus entier. Le second ne certifie pas le trafic NAND ni un SSD particulier. Un zéro n’établit pas à lui seul l’absence de lecture physique ; une donnée absente reste absente. La couverture est vérifiée avant agrégation.
- MiB et GiB utilisent les puissances de 1 024. Les pics RSS et les croissances maximales du swap sont pris au maximum, jamais additionnés entre tours. Le swap est système et peut inclure d’autres applications.
- « Premier processus » ne signifie pas cache de poids ou cache SSD froid.

## Versions de D2 et notes liées aux réponses

Le D2 initial est `failed`, avec 77 IDs partiels. Le D2 de reprise est `completed`, 79 IDs, arrêt EOS ; ses 77 premiers IDs sont exactement identiques au fragment initial. Seule la version achevée est retenue dans les 21 tâches. La première ligne reste « Date actuelle : 8 avril ; ancien créneau annulé : 15 mai. » : la réparation du routeur ne résout pas l’ambiguïté de « date actuelle » entre date de version et date de formation. L’analyseur ne déduit pas un jugement de compréhension de la seule réussite technique.

Les **38 liaisons** ont été vérifiées avec le fichier source, l’ID, la clé `text` ou `partial_text`, le texte intégral et son SHA-256 : 21 principales, 2 pilote, 4 longues dont un fragment, 8 diagnostiques, 1 A4, 1 baseline CLI et 1 fragment D2. La baseline et les notes conditionnelles du fragment 1 024 restent en annexes distinctes. La revue principale actuelle porte les verdicts 16 réussis, 2 partiels, 2 échecs de critère et 1 indéterminé ; ces catégories sont celles de la revue humaine/agent liée, pas un score calculé par regex.

Le contrôle de code canonique final est lié aux textes exacts et à la source courante : sandbox vérifiée, **2/2 fonctions et 8/8 cas réussis**. Son SHA-256 est `4568e4ade5aeb0b7d1b611987eb9990e7898acc74831575a4f5b92d57322f146`. L’ancienne sortie intermédiaire est conservée sous `code-checks-in-progress.json` et n’est pas substituée à cette version finale.

## Ablation E : mêmes entrées et sorties

Les cinq paires E1–E5 ont les mêmes IDs d’entrée et de sortie. E1 reste visible, mais seules E2–E5 entrent dans le gain agrégé de réutilisation KV.

| Tour | Réutilisation KV, requête (s) | Reconstruction, requête (s) | Reconstruction / KV |
| --- | --- | --- | --- |
| E1 | 79,610939709 | 76,119203291 | 0,956139992 |
| E2 | 57,311457 | 103,73404125 | 1,810005306 |
| E3 | 64,838504709 | 146,39937775 | 2,257907989 |
| E4 | 139,95432525 | 269,086739166 | 1,922675406 |
| E5 | 120,052082792 | 321,769212333 | 2,680246813 |

Sur E2–E5 : 840,98937049901 / 382,156369750985 = **2,2006420331×** pour les sommes des durées murales, en accord avec l’analyseur. Le ratio individuel et la réduction `100 × (1 − KV / reconstruction)` sont également recoupés. L’ordre des essais et les états des caches de poids diffèrent : ce rapport est une observation appariée, avec ces facteurs de confusion, et non une estimation causale isolant universellement le KV.

## Paliers longs et garde autoritaire

Les paliers 128, 256 et 512 sont achevés au plafond, avec égalité exacte des préfixes pour les trois comparaisons possibles. Le palier demandé à 1 024 n’est **pas validé** : le brut reste `started` après SIGTERM, mais le garde final impose l’état effectif `interrompu_garde` (code -15, violation `system_swap_growth`). Il contient 244 tokens partiels ; leur préfixe est exactement celui des 244 premiers tokens du palier 256.

La croissance système du swap est de 170459136 octets, supérieure à la limite de 134217728 octets. L’analyseur conserve ce fragment et le statut brut, affiche l’interruption explicite, et n’invente ni durée complète ni débit complet à partir du fragment. Les notes du fragment sont conditionnelles et ne comptent pas comme une réponse achevée. L’observation des n-grammes répétés demeure descriptive ; elle ne produit aucun score automatique de qualité.

## Empreintes et stabilité des sources

Les 12 essais nommés ont été relus. La phase initiale et le pilote déclarent chacun 45 sources : 42 concordent avec les fichiers actuels ; les trois autres concordent exactement avec les archives `sources_before_router_fix/gguf_qwen.py`, `qwen_moe.py` et `evaluate_real_world.py`. Pour les dix essais après correction, les 45 sources sur 45 concordent avec les fichiers courants. La copie `campaign-before-router-fix.json` n’est pas traitée comme une exécution supplémentaire.

Les empreintes de toutes les entrées réellement lues et les détails numériques figurent dans le JSON d’audit. Le helper n’a chargé aucun poids et n’a pas recalculé l’empreinte du fichier modèle volumineux : cette vérification est distincte. Les sources d’inférence, leurs preuves JSON et leurs gardes sont restés inchangés pendant l’audit. Les ajouts au générateur concernent seulement l’affichage des annexes et des statuts ; aucun calcul n’a été changé sans défaut prouvé.
