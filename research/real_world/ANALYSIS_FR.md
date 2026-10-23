# Analyse des évaluations réelles — AirLLM Next

Instantané généré le 23/10/2026 à 23:14:34 CEST. Lecture de fichiers existants ; aucune inférence déclenchée.

## Couverture des preuves

| Fichier | État effectif | État JSON brut | Tours achevés | Échecs | Interrompus / incomplets | En cours | Erreur |
| --- | --- | --- | --- | --- | --- | --- | --- |
| campaign.json | failed | failed | 8 | 1 | 0 | 0 | ValueError: Qwen router top-k boundary tie: reference tie-breaking is not validated |
| campaign-resumed.json | completed | completed | 13 | 0 | 0 | 0 |  |
| hard-reasoning.json | completed | completed | 1 | 0 | 0 | 0 |  |
| concision-control.json | completed | completed | 1 | 0 | 0 | 0 |  |
| f2-concision-control.json | completed | completed | 1 | 0 | 0 | 0 |  |
| d2-clarified.json | completed | completed | 1 | 0 | 0 | 0 |  |
| pilot-short-caps.json | interrompu_processus | started | 2 | 0 | 1 | 0 |  |
| long-128.json | completed | completed | 1 | 0 | 0 | 0 |  |
| long-256.json | completed | completed | 1 | 0 | 0 | 0 |  |
| long-512.json | completed | completed | 1 | 0 | 0 | 0 |  |
| long-1024.json | interrompu_garde | started | 0 | 0 | 1 | 0 |  |
| conversation-rebuild.json | completed | completed | 5 | 0 | 0 | 0 |  |

Seuls les tours dont `status=completed` entrent dans les calculs. Un garde final signalant un arrêt fait autorité pour l’affichage : un JSON resté `started` après SIGTERM est présenté comme interrompu, sans être réécrit ni admis dans les agrégats achevés. Une sortie arrêtée par `max_new_tokens` est intégralement conservée, mais sa complétude n’est pas présumée. Les diagnostics longs, le pilote et l’ablation sont séparés de la campagne principale. Le pilote A1/A2 à 96 tokens a motivé un amendement : reprise des 21 mêmes demandes à 256 tokens par tour. Ce choix est postérieur au pilote, pas un paramètre fixé à l’aveugle.

La campagne principale combine les tours achevés de `campaign.json` avant l’arrêt D2 et de `campaign-resumed.json` après correction du routeur, à raison d’une observation par ID. La copie `campaign-before-router-fix.json` reste une archive et n’ajoute aucune observation. Le fragment D2 échoué est conservé comme diagnostic ; seul D2 achevé de la reprise entre dans les 21 tours.

| Périmètre | Observations achevées | Cible annoncée |
| --- | --- | --- |
| Principal, IDs uniques | 21 | 21 |
| Supplément A4, raisonnement conditionnel | 1 | 1 |
| Principal + A4, en explicitant le supplément | 22 | 22 |

IDs principaux encore sans réponse achevée : aucun.

## Définitions et limites de mesure

- TTFT moteur : délai de la première passe de modèle jusqu’au premier token synchronisé ; tokenisation exclue.
- TTFT perçue encadrée : borne basse = TTFT moteur plus lancement/imports/initialisation pour la première requête du processus. La borne haute ajoute tous les frais hors temps moteur : max(0, temps mural requête − temps moteur). Cet intervalle couvre la préparation omise de la borne basse sans prétendre mesurer exactement son partage avant/après le premier token.
- Total perçu enregistré : lancement→prêt plus temps mural de la requête pour la première requête ; temps mural de la requête seule ensuite. Le périmètre est cette somme, hors gestes humains et temps entre requêtes.
- « Premier processus » désigne la première requête de ce lancement ; cela ne prouve pas un cache SSD froid.
- Temps total par tour : temps mural instrumenté de la requête, incluant sauvegarde des tokens et échantillonnage. Le débit de décodage exclut le premier token, mais inclut le callback de sauvegarde dans son dénominateur.
- Débits agrégés : rapports des sommes ; aucune moyenne non pondérée des débits individuels.
- Lectures logiques : octets demandés par le lecteur. Stockage : compteur du processus entier, réadahead inclus, sans certification de trafic NAND ni attribution à un fichier ou un SSD.
- RSS : maximum observé par échantillonnage, environ 50 ms. Swap : croissance maximale du swap système durant le tour ; elle peut provenir d’autres applications et ne s’additionne pas entre tours.

## Mesures par tour achevé

| Run | Domaine | Tour | Tokens sortie | TTFT moteur s | Rang processus | TTFT perçue basse s | TTFT perçue haute s | Total s | Total perçu s | Decode tok/s | Logique GiB | Stockage GiB | RSS MiB | Swap Δ MiB | Arrêt | Logits finis vérifiés |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| campaign | A_raisonnement | A1 | 118 | 46,33 | premier processus | 48,23 | 48,23 | 164,49 | 166,39 | 0,99 | 247,27 | 61,70 | 491,14 | 0,00 | eos | oui |
| campaign | A_raisonnement | A2 | 256 | 43,62 | suivant | 43,62 | 43,62 | 312,33 | 312,33 | 0,95 | 502,86 | 109,48 | 436,42 | 0,00 | max_new_tokens | oui |
| campaign | A_raisonnement | A3 | 73 | 49,24 | suivant | 49,24 | 49,24 | 125,91 | 125,91 | 0,94 | 165,60 | 51,60 | 436,86 | 0,00 | eos | oui |
| campaign | B_programmation | B1 | 78 | 50,94 | suivant | 50,94 | 50,94 | 130,59 | 130,59 | 0,97 | 177,00 | 52,27 | 424,50 | 0,00 | eos | oui |
| campaign | B_programmation | B2 | 51 | 54,09 | suivant | 54,09 | 54,09 | 109,16 | 109,16 | 0,91 | 131,44 | 55,74 | 425,25 | 0,00 | eos | oui |
| campaign | C_francais | C1 | 26 | 45,80 | suivant | 45,80 | 45,80 | 72,17 | 72,17 | 0,95 | 79,70 | 38,17 | 445,48 | 0,00 | eos | oui |
| campaign | C_francais | C2 | 53 | 51,11 | suivant | 51,11 | 51,11 | 104,01 | 104,01 | 0,98 | 131,43 | 46,75 | 435,39 | 0,00 | eos | oui |
| campaign | D_documents | D1 | 33 | 61,61 | suivant | 61,61 | 61,61 | 97,50 | 97,50 | 0,89 | 109,44 | 59,69 | 435,23 | 0,00 | eos | oui |
| campaign-resumed | D_documents | D2 | 79 | 189,83 | premier processus | 191,63 | 191,63 | 281,59 | 283,40 | 0,85 | 275,58 | 157,99 | 506,81 | 0,00 | eos | oui |
| campaign-resumed | E_multitour | E1 | 31 | 44,64 | suivant | 44,64 | 44,64 | 79,61 | 79,61 | 0,86 | 87,98 | 41,22 | 554,42 | 0,00 | eos | oui |
| campaign-resumed | E_multitour | E2 | 31 | 20,53 | suivant | 20,53 | 20,53 | 57,31 | 57,31 | 0,82 | 71,54 | 25,83 | 557,09 | 0,00 | eos | oui |
| campaign-resumed | E_multitour | E3 | 44 | 18,03 | suivant | 18,03 | 18,03 | 64,84 | 64,84 | 0,92 | 91,21 | 20,49 | 489,64 | 0,00 | eos | oui |
| campaign-resumed | E_multitour | E4 | 109 | 23,25 | suivant | 23,25 | 23,25 | 139,95 | 139,95 | 0,93 | 217,81 | 48,68 | 529,41 | 0,00 | eos | oui |
| campaign-resumed | E_multitour | E5 | 88 | 22,44 | suivant | 22,44 | 22,44 | 120,05 | 120,05 | 0,89 | 177,26 | 39,55 | 526,77 | 0,00 | eos | oui |
| campaign-resumed | F_donnees_budget | F1 | 140 | 61,00 | suivant | 61,00 | 61,00 | 211,52 | 211,52 | 0,92 | 300,23 | 83,02 | 498,08 | 0,00 | eos | oui |
| campaign-resumed | F_donnees_budget | F2 | 256 | 74,36 | suivant | 74,36 | 74,37 | 369,87 | 369,87 | 0,86 | 520,73 | 129,75 | 491,34 | 0,00 | max_new_tokens | oui |
| campaign-resumed | G_fiabilite | G1 | 38 | 41,70 | suivant | 41,70 | 41,70 | 81,77 | 81,77 | 0,92 | 96,94 | 38,48 | 468,72 | 0,00 | eos | oui |
| campaign-resumed | G_fiabilite | G2 | 31 | 39,24 | suivant | 39,24 | 39,24 | 75,11 | 75,11 | 0,84 | 83,14 | 36,43 | 469,80 | 0,00 | eos | oui |
| campaign-resumed | G_fiabilite | G3 | 256 | 43,68 | suivant | 43,68 | 43,68 | 314,69 | 314,69 | 0,94 | 496,29 | 92,14 | 482,70 | 0,00 | max_new_tokens | oui |
| campaign-resumed | G_fiabilite | G4 | 32 | 47,54 | suivant | 47,54 | 47,54 | 82,14 | 82,14 | 0,90 | 94,90 | 48,39 | 473,00 | 0,00 | eos | oui |
| campaign-resumed | G_fiabilite | G5 | 42 | 40,02 | suivant | 40,02 | 40,02 | 86,04 | 86,04 | 0,89 | 105,29 | 42,61 | 496,38 | 0,00 | eos | oui |
| hard-reasoning | A_raisonnement | A4 | 107 | 97,69 | premier processus | 99,57 | 99,58 | 213,33 | 215,22 | 0,92 | 266,14 | 100,98 | 504,11 | 0,00 | eos | oui |
| concision-control | A_raisonnement | A2_concis | 51 | 61,00 | premier processus | 63,07 | 63,07 | 114,97 | 117,04 | 0,93 | 137,05 | 59,25 | 463,77 | 0,00 | eos | oui |
| f2-concision-control | F_donnees_budget | F2_concis | 97 | 77,59 | premier processus | 79,42 | 79,42 | 191,20 | 193,03 | 0,85 | 231,83 | 89,56 | 527,25 | 0,00 | eos | oui |
| d2-clarified | D_documents | D2_clarified | 116 | 193,56 | premier processus | 195,42 | 195,42 | 324,22 | 326,08 | 0,88 | 348,29 | 171,49 | 428,14 | 0,00 | eos | oui |
| pilot-short-caps | A_raisonnement | A1 | 96 | 47,72 | premier processus | 49,52 | 49,52 | 146,35 | 148,15 | 0,96 | 207,01 | 58,74 | 476,34 | 0,00 | max_new_tokens | oui |
| pilot-short-caps | A_raisonnement | A2 | 96 | 53,24 | suivant | 53,24 | 53,24 | 151,76 | 151,76 | 0,96 | 210,12 | 60,26 | 412,16 | 0,00 | max_new_tokens | oui |
| long-128 | L_generation_longue | L128 | 128 | 160,92 | premier processus | 162,82 | 162,82 | 291,49 | 293,39 | 0,97 | 361,61 | 159,41 | 421,81 | 0,00 | max_new_tokens | oui |
| long-256 | L_generation_longue | L256 | 256 | 180,00 | premier processus | 181,80 | 181,81 | 460,59 | 462,39 | 0,91 | 595,82 | 193,97 | 386,95 | 0,00 | max_new_tokens | oui |
| long-512 | L_generation_longue | L512 | 512 | 178,00 | premier processus | 179,91 | 179,91 | 741,89 | 743,80 | 0,91 | 1 064,24 | 252,02 | 391,95 | 0,00 | max_new_tokens | oui |
| conversation-rebuild | E_multitour | E1 | 31 | 42,72 | premier processus | 44,57 | 44,57 | 76,12 | 77,97 | 0,90 | 88,10 | 41,44 | 453,70 | 0,00 | eos | oui |
| conversation-rebuild | E_multitour | E2 | 31 | 69,80 | suivant | 69,80 | 69,80 | 103,73 | 103,73 | 0,88 | 111,55 | 66,21 | 416,11 | 0,00 | eos | oui |
| conversation-rebuild | E_multitour | E3 | 44 | 100,34 | suivant | 100,34 | 100,35 | 146,40 | 146,40 | 0,93 | 156,38 | 88,01 | 443,28 | 0,00 | eos | oui |
| conversation-rebuild | E_multitour | E4 | 109 | 143,00 | suivant | 143,00 | 143,01 | 269,09 | 269,09 | 0,86 | 305,22 | 143,93 | 409,28 | 0,00 | eos | oui |
| conversation-rebuild | E_multitour | E5 | 88 | 222,88 | suivant | 222,88 | 222,88 | 321,77 | 321,77 | 0,88 | 317,30 | 183,49 | 388,67 | 0,00 | eos | oui |

## Agrégats par exécution et domaine

| Run | Domaine | Tours | Tokens | Σ total s | Σ tokens/Σ total | Σ decode/Σ durée decode | Σ logique GiB | Σ stockage GiB | Couverture stockage | RSS max MiB | Swap Δ max MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRINCIPAL_21_ID_uniques | GLOBAL | 21 | 1865 | 3 080,66 | 0,61 | 0,92 | 4 163,63 | 1 279,98 | 21/21 | 557,09 | 0,00 |
| PRINCIPAL_21_ID_uniques | A_raisonnement | 3 | 447 | 602,73 | 0,74 | 0,96 | 915,73 | 222,77 | 3/3 | 491,14 | 0,00 |
| PRINCIPAL_21_ID_uniques | B_programmation | 2 | 129 | 239,75 | 0,54 | 0,94 | 308,44 | 108,01 | 2/2 | 425,25 | 0,00 |
| PRINCIPAL_21_ID_uniques | C_francais | 2 | 79 | 176,19 | 0,45 | 0,97 | 211,13 | 84,93 | 2/2 | 445,48 | 0,00 |
| PRINCIPAL_21_ID_uniques | D_documents | 2 | 112 | 379,09 | 0,30 | 0,86 | 385,02 | 217,68 | 2/2 | 506,81 | 0,00 |
| PRINCIPAL_21_ID_uniques | E_multitour | 5 | 303 | 461,77 | 0,66 | 0,90 | 645,81 | 175,78 | 5/5 | 557,09 | 0,00 |
| PRINCIPAL_21_ID_uniques | F_donnees_budget | 2 | 396 | 581,39 | 0,68 | 0,88 | 820,96 | 212,77 | 2/2 | 498,08 | 0,00 |
| PRINCIPAL_21_ID_uniques | G_fiabilite | 5 | 399 | 639,74 | 0,62 | 0,92 | 876,56 | 258,04 | 5/5 | 496,38 | 0,00 |
| SUPPLEMENT_A4 | GLOBAL | 1 | 107 | 213,33 | 0,50 | 0,92 | 266,14 | 100,98 | 1/1 | 504,11 | 0,00 |
| SUPPLEMENT_A4 | A_raisonnement | 1 | 107 | 213,33 | 0,50 | 0,92 | 266,14 | 100,98 | 1/1 | 504,11 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | GLOBAL | 22 | 1972 | 3 293,99 | 0,60 | 0,92 | 4 429,77 | 1 380,95 | 22/22 | 557,09 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | A_raisonnement | 4 | 554 | 816,06 | 0,68 | 0,95 | 1 181,87 | 323,75 | 4/4 | 504,11 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | B_programmation | 2 | 129 | 239,75 | 0,54 | 0,94 | 308,44 | 108,01 | 2/2 | 425,25 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | C_francais | 2 | 79 | 176,19 | 0,45 | 0,97 | 211,13 | 84,93 | 2/2 | 445,48 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | D_documents | 2 | 112 | 379,09 | 0,30 | 0,86 | 385,02 | 217,68 | 2/2 | 506,81 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | E_multitour | 5 | 303 | 461,77 | 0,66 | 0,90 | 645,81 | 175,78 | 5/5 | 557,09 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | F_donnees_budget | 2 | 396 | 581,39 | 0,68 | 0,88 | 820,96 | 212,77 | 2/2 | 498,08 | 0,00 |
| PRINCIPAL_plus_A4_cible22 | G_fiabilite | 5 | 399 | 639,74 | 0,62 | 0,92 | 876,56 | 258,04 | 5/5 | 496,38 | 0,00 |
| campaign | GLOBAL | 8 | 688 | 1 116,16 | 0,62 | 0,95 | 1 544,73 | 475,40 | 8/8 | 491,14 | 0,00 |
| campaign | A_raisonnement | 3 | 447 | 602,73 | 0,74 | 0,96 | 915,73 | 222,77 | 3/3 | 491,14 | 0,00 |
| campaign | B_programmation | 2 | 129 | 239,75 | 0,54 | 0,94 | 308,44 | 108,01 | 2/2 | 425,25 | 0,00 |
| campaign | C_francais | 2 | 79 | 176,19 | 0,45 | 0,97 | 211,13 | 84,93 | 2/2 | 445,48 | 0,00 |
| campaign | D_documents | 1 | 33 | 97,50 | 0,34 | 0,89 | 109,44 | 59,69 | 1/1 | 435,23 | 0,00 |
| campaign-resumed | GLOBAL | 13 | 1177 | 1 964,49 | 0,60 | 0,90 | 2 618,90 | 804,58 | 13/13 | 557,09 | 0,00 |
| campaign-resumed | D_documents | 1 | 79 | 281,59 | 0,28 | 0,85 | 275,58 | 157,99 | 1/1 | 506,81 | 0,00 |
| campaign-resumed | E_multitour | 5 | 303 | 461,77 | 0,66 | 0,90 | 645,81 | 175,78 | 5/5 | 557,09 | 0,00 |
| campaign-resumed | F_donnees_budget | 2 | 396 | 581,39 | 0,68 | 0,88 | 820,96 | 212,77 | 2/2 | 498,08 | 0,00 |
| campaign-resumed | G_fiabilite | 5 | 399 | 639,74 | 0,62 | 0,92 | 876,56 | 258,04 | 5/5 | 496,38 | 0,00 |
| hard-reasoning | GLOBAL | 1 | 107 | 213,33 | 0,50 | 0,92 | 266,14 | 100,98 | 1/1 | 504,11 | 0,00 |
| hard-reasoning | A_raisonnement | 1 | 107 | 213,33 | 0,50 | 0,92 | 266,14 | 100,98 | 1/1 | 504,11 | 0,00 |
| concision-control | GLOBAL | 1 | 51 | 114,97 | 0,44 | 0,93 | 137,05 | 59,25 | 1/1 | 463,77 | 0,00 |
| concision-control | A_raisonnement | 1 | 51 | 114,97 | 0,44 | 0,93 | 137,05 | 59,25 | 1/1 | 463,77 | 0,00 |
| f2-concision-control | GLOBAL | 1 | 97 | 191,20 | 0,51 | 0,85 | 231,83 | 89,56 | 1/1 | 527,25 | 0,00 |
| f2-concision-control | F_donnees_budget | 1 | 97 | 191,20 | 0,51 | 0,85 | 231,83 | 89,56 | 1/1 | 527,25 | 0,00 |
| d2-clarified | GLOBAL | 1 | 116 | 324,22 | 0,36 | 0,88 | 348,29 | 171,49 | 1/1 | 428,14 | 0,00 |
| d2-clarified | D_documents | 1 | 116 | 324,22 | 0,36 | 0,88 | 348,29 | 171,49 | 1/1 | 428,14 | 0,00 |
| pilot-short-caps | GLOBAL | 2 | 192 | 298,11 | 0,64 | 0,96 | 417,14 | 119,00 | 2/2 | 476,34 | 0,00 |
| pilot-short-caps | A_raisonnement | 2 | 192 | 298,11 | 0,64 | 0,96 | 417,14 | 119,00 | 2/2 | 476,34 | 0,00 |
| long-128 | GLOBAL | 1 | 128 | 291,49 | 0,44 | 0,97 | 361,61 | 159,41 | 1/1 | 421,81 | 0,00 |
| long-128 | L_generation_longue | 1 | 128 | 291,49 | 0,44 | 0,97 | 361,61 | 159,41 | 1/1 | 421,81 | 0,00 |
| long-256 | GLOBAL | 1 | 256 | 460,59 | 0,56 | 0,91 | 595,82 | 193,97 | 1/1 | 386,95 | 0,00 |
| long-256 | L_generation_longue | 1 | 256 | 460,59 | 0,56 | 0,91 | 595,82 | 193,97 | 1/1 | 386,95 | 0,00 |
| long-512 | GLOBAL | 1 | 512 | 741,89 | 0,69 | 0,91 | 1 064,24 | 252,02 | 1/1 | 391,95 | 0,00 |
| long-512 | L_generation_longue | 1 | 512 | 741,89 | 0,69 | 0,91 | 1 064,24 | 252,02 | 1/1 | 391,95 | 0,00 |
| conversation-rebuild | GLOBAL | 5 | 303 | 917,11 | 0,33 | 0,88 | 978,55 | 523,07 | 5/5 | 453,70 | 0,00 |
| conversation-rebuild | E_multitour | 5 | 303 | 917,11 | 0,33 | 0,88 | 978,55 | 523,07 | 5/5 | 453,70 | 0,00 |

## D2 — arrêt initial, reprise et sémantique de la date

| Mesure | Observation |
| --- | --- |
| Tokens avant arrêt initial | 77 |
| Tokens de la reprise achevée | 79 |
| Préfixe initial exactement conservé dans la reprise | oui |
| Prompt utilisateur identique | oui |
| Arrêt initial | ValueError: Qwen router top-k boundary tie: reference tie-breaking is not validated |
| Arrêt de la reprise | eos |

Première ligne du fragment initial, reproduite exactement :

```text
Date actuelle : 8 avril ; ancien créneau annulé : 15 mai.
```

Première ligne de la reprise achevée :

```text
Date actuelle : 8 avril ; ancien créneau annulé : 15 mai.
```

La date retenue pour la formation est le **22 mai** ; le **8 avril** est la date de version du dossier. La résolution de l’arrêt du routeur et l’interprétation de cette date sont deux résultats distincts. La consigne initiale « date actuelle » peut prêter à une lecture de la date de version ; la référence visait la date de formation. Ces extraits et la revue indépendante permettent de discuter cette ambiguïté sans attribuer automatiquement une erreur de compréhension au moteur. La formulation ambiguë ne permet pas de conclure à une erreur certaine du modèle sur cette seule réponse.

La première ligne est inchangée après reprise ; le correctif de routage n’a pas réécrit ce passage antérieur à l’arrêt. Cela ne tranche pas l’ambiguïté de la consigne.

Contrôle lexical séparé `D2_clarified`, première ligne exacte :

```text
La date retenue est le 22 mai, l’ancien créneau du 15 mai ayant été annulé.
```

Ce contrôle explicite la date de formation dans la consigne. Il modifie la formulation et ne remplace pas D2 dans les 21 tâches principales.

## Repli de la gate du routeur sur CPU — compteurs enregistrés

Les compteurs des poids sont cumulatifs dans chaque processus. Les deltas ci-dessous sont calculés entre tours achevés successifs de la même exécution ; le premier part de zéro. Un tour échoué entre deux observations peut donc contribuer au delta suivant. Une absence de compteur sur l’ancien moteur n’est pas transformée en mesure nulle. Lorsqu’une égalité déclenche le repli, la gate CPU complète est recalculée sur tout le chunk courant (F.linear → softmax → topk → normalisation), en FP32 dans la configuration évaluée ; seuls les indices et scores des lignes affectées remplacent ceux du routage natif. La capture D2 montre un écart FP32 entre Metal et Torch CPU dès les logits, avant le softmax : réordonner les seules probabilités natives ne reproduirait donc pas cette gate de référence. La référence est le Torch CPU local et versionné, sans garantie de départage identique entre versions ou avec CUDA. Les experts et le reste du modèle continuent d’utiliser le moteur natif.

| Run | Tour | Appels repli Δ | Appels repli cumul | Lignes repli cumul | Temps repli Δ s | Import réf cumul s | Version réf | Appareil réf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| campaign-resumed.json | D2 | 1 | 1 | 1 | 0,6732 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | E1 | 0 | 1 | 1 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | E2 | 0 | 1 | 1 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | E3 | 1 | 2 | 2 | 0,0778 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | E4 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | E5 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | F1 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | F2 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | G1 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | G2 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | G3 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | G4 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| campaign-resumed.json | G5 | 0 | 2 | 2 | 0,0000 | 0,6589 | 2.14.1 | cpu |
| hard-reasoning.json | A4 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| concision-control.json | A2_concis | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| f2-concision-control.json | F2_concis | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| d2-clarified.json | D2_clarified | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| long-128.json | L128 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| long-256.json | L256 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| long-512.json | L512 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| conversation-rebuild.json | E1 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| conversation-rebuild.json | E2 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| conversation-rebuild.json | E3 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| conversation-rebuild.json | E4 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |
| conversation-rebuild.json | E5 | 0 | 0 | 0 | 0,0000 | 0,0000 | — | — |

## Contrôle de concision — séparé de A2

Le contrôle A2_concis reprend la même question arithmétique avec une consigne système supplémentaire : deux lignes, 45 mots au maximum, sans préambule ni format Markdown/LaTeX. Il ne remplace pas la sortie A2 principale.

| Observation | Tokens | Mots séparés par espaces | Lignes non vides | Total s | Arrêt |
| --- | --- | --- | --- | --- | --- |
| A2 principal | 256 | 111 | 12 | 312,33 | max_new_tokens |
| A2_concis diagnostic | 51 | 20 | 1 | 114,97 | eos |

Question utilisateur exacte identique : oui. Les comptes de mots et de lignes sont descriptifs ; la présence des résultats corrects et le respect global des consignes restent examinés par la revue.

## Échelle de génération longue

Comparaison des IDs réellement générés. Les répétitions de fenêtres de 8 et 16 tokens sont des observations descriptives ; elles ne donnent aucune note automatique de qualité ou de cohérence.

| Plafond | Statut | Tokens observés | Mots observés | Arrêt | Total achevé s | Fenêtres 8 répétées | Fenêtres 16 répétées |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | achevé | 128 | 74 | max_new_tokens | 291,49 | 0/121 (0,0 %) | 0/113 (0,0 %) |
| 256 | achevé | 256 | 167 | max_new_tokens | 460,59 | 0/249 (0,0 %) | 0/241 (0,0 %) |
| 512 | achevé | 512 | 338 | max_new_tokens | 741,89 | 0/505 (0,0 %) | 0/497 (0,0 %) |
| 1024 | interrompu_garde — non validé | 244 | 158 | system_swap_growth | — | — | — |

Les fragments interrompus ne valident pas leur palier ; aucun temps de réponse complet, débit achevé ou comparaison de préfixe de réponse achevée n’est calculé à partir de ces fragments.

| Paire | Entrée IDs identique | Sources identiques | Longueur comparée | Préfixe sortie exact |
| --- | --- | --- | --- | --- |
| 128 → 256 | oui | oui | 128 | oui |
| 128 → 512 | oui | oui | 128 | oui |
| 256 → 512 | oui | oui | 256 | oui |

## Ablation de la conversation E — réemploi KV / reconstruction

Les deux exécutions doivent conserver les mêmes messages réellement générés. L’égalité est testée sur les IDs, et non sur une ressemblance du texte. Les gains ci-dessous sont des rapports observés ; l’ordre des tâches et l’état initial du cache de poids peuvent différer entre processus. Ils ne suffisent pas à attribuer causalement tout l’écart au seul KV.

| Tour | Entrée exacte | Sortie exacte | Préfixe KV réutilisé | KV total s | Rebuild total s | Rebuild/KV × | Réduction observée % | TTFT rebuild/KV × |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | oui | oui | 0 | 79,61 | 76,12 | 0,96 | -4,59 | 0,96 |
| E2 | oui | oui | 146 | 57,31 | 103,73 | 1,81 | 44,75 | 3,40 |
| E3 | oui | oui | 228 | 64,84 | 146,40 | 2,26 | 55,71 | 5,57 |
| E4 | oui | oui | 311 | 139,95 | 269,09 | 1,92 | 47,99 | 6,15 |
| E5 | oui | oui | 486 | 120,05 | 321,77 | 2,68 | 62,69 | 9,93 |

Paires E2–E5 à entrée et sortie identiques : 4/4. Rapport des sommes de temps rebuild/KV : 2,20 × (Σ KV 382,16 s ; Σ rebuild 840,99 s).

## Contrôles objectifs du code

Le succès des cas exécutés et le respect du format « code seul » sont séparés. Une réponse absente ou non exécutée n’est pas une preuve d’échec du modèle.

Résumé enregistré : {"sandbox_verified": true, "evaluated": 2, "executed": 2, "passed": 2}. Empreinte du texte vérifiée pour 2/2 résultat(s).

| Tour | SHA réponse concorde | Exécuté | Cas réussis | Code seul | Retourne le même objet | Erreur |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | oui | oui | oui | oui | non |  |
| B2 | oui | oui | oui | oui | non |  |

## Évaluation indépendante

État enregistré : **completed**. Lecture intégrale des seules réponses dont status=completed ; notes de qualité distinctes des contrôles objectifs. Aucun contenu corrigé ne remplace la sortie brute. Évaluateur Codex distinct du moteur testé, sans prétention de reproductibilité inter-évaluateurs.

21 tour(s) annoté(s), dont 21 avec texte source complet et empreinte concordante. Les notes sont celles de l’évaluateur enregistré, sur 5 ; elles ne sont pas déduites des métriques de répétition.

| Tour | Source | SHA concorde | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | campaign.json | oui | 5 | 5 | 4 | 5 | 4 |
| A2 | campaign.json | oui | 5 | 3 | 4 | 3 | 3 |
| A3 | campaign.json | oui | 5 | 5 | 4 | 5 | 5 |
| B1 | campaign.json | oui | 5 | 5 | 5 | 5 | 5 |
| B2 | campaign.json | oui | 5 | 5 | 5 | 5 | 5 |
| C1 | campaign.json | oui | 5 | 5 | 5 | 5 | 5 |
| C2 | campaign.json | oui | 5 | 5 | 5 | 5 | 5 |
| D1 | campaign.json | oui | 5 | 5 | 5 | 5 | 5 |
| D2 | campaign-resumed.json | oui | 4 | 4 | 5 | 4 | 4 |
| E1 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| E2 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| E3 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| E4 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| E5 | campaign-resumed.json | oui | 5 | 4 | 5 | 5 | 5 |
| F1 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| F2 | campaign-resumed.json | oui | 5 | 1 | 4 | 1 | 2 |
| G1 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 4 |
| G2 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| G3 | campaign-resumed.json | oui | 2 | 2 | 3 | 4 | 2 |
| G4 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |
| G5 | campaign-resumed.json | oui | 5 | 5 | 5 | 5 | 5 |

### A1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Les trois jours et les deux relations temporelles citées sont exacts. La parenthèse peut laisser entendre une causalité insuffisante pour exclure lundi, mais ce lien causal n’est pas explicitement affirmé et l’exclusion de lundi était donnée dans le prompt.
- respect_instructions (5/5) : Donne les trois jours et une justification en une phrase achevée.
- coherence (4/5) : Le calendrier respecte toutes les contraintes. Les parenthèses peuvent brouiller la distinction entre contrainte donnée et justification déduite ; cette réserve de clarté n’est pas une contradiction factuelle démontrée.
- completude (5/5) : La solution et sa justification sont achevées avant un EOS naturel.
- utilite_reelle (4/5) : Le bon calendrier est exploitable ; la justification mérite une correction mineure.
- Contrôle : Affectation des jours ; résultat : pass ; preuve : Lina mardi ; Marc mercredi ; Nora lundi.
- Contrôle : Justification achevée ; résultat : pass ; preuve : La phrase termine par la place laissée à Nora ; EOS à 118 tokens.
- Contrôle : Clarté des liens logiques exposés ; résultat : interpretive_clarity_reservation ; preuve : La parenthèse après l’exclusion de lundi peut suggérer une causalité insuffisante ; elle contient toutefois un fait vrai et ne formule pas explicitement cette implication.
- Limite relevée : Les parenthèses peuvent être rédigées plus clairement, mais elles ne suffisent pas à établir une erreur de logique. Cette réserve ne fait pas échouer la rubrique prédéclarée, qui exigeait l’absence de contradiction.

### A2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : max_new_tokens.

- exactitude_factuelle (5/5) : Tous les calculs présents sont exacts : 432 € brut, 64,80 € de remise, 367,20 € après remise et 403,20 € livraison comprise. Le reste n’est pas évaluable car absent.
- respect_instructions (3/5) : Donne le total et son calcul, mais ne fournit pas le reste du budget demandé.
- coherence (4/5) : La progression de calcul est cohérente et distingue correctement la livraison ; la sortie se termine sur un séparateur après le premier des deux volets annoncés.
- completude (3/5) : Le premier résultat est complet ; le second, 46,80 € restant sur 450 €, manque au plafond de 256 tokens.
- utilite_reelle (3/5) : Le prix à payer est exploitable, mais le lecteur doit encore calculer son budget restant.
- Contrôle : Total de 403,20 € ; résultat : pass ; preuve : 367,20 € + 36 € = 403,20 €.
- Contrôle : Reste sur budget de 46,80 € ; résultat : missing ; preuve : Aucune soustraction sur le budget 450 € n’est produite dans la sortie intégrale.
- Contrôle : Remise appliquée aux seuls kits ; résultat : pass ; preuve : 432×0,15=64,80 avant ajout des 36 € de livraison.
- Limite relevée : Le texte s’arrête au plafond fixé ; l’absence de second résultat ne prouve pas une erreur arithmétique.

### A3 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Identifie correctement la contradiction entre tous les urgents partis et R7 urgent encore dans l’entrepôt.
- respect_instructions (5/5) : Réponse française en deux phrases, sans cause inventée.
- coherence (4/5) : La première phrase est claire ; la deuxième formule maladroitement l’idée du membre restant, mais ne change pas le diagnostic.
- completude (5/5) : Répond à la compatibilité et explique la contradiction, avec un EOS naturel.
- utilite_reelle (5/5) : Le lecteur obtient une conclusion juste et directement exploitable sans spéculation.
- Contrôle : Contradiction explicite de R7 ; résultat : pass ; preuve : « incompatibles » ; tous les urgents partis opposé à R7 urgent encore présent.
- Contrôle : Deux phrases et absence de cause inventée ; résultat : pass ; preuve : Deux phrases ; aucune désignation de l’affirmation fautive ni explication logistique inventée.

### B1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : À la lecture du code, conserve la première occurrence et l’ordre pour les entiers, chaînes et listes du domaine demandé ; la conversion des listes en tuples ne requiert aucun hachage.
- respect_instructions (5/5) : Produit uniquement la fonction Python dedoublonne, sans Markdown ni import, et construit une nouvelle liste.
- coherence (5/5) : Le traitement et les variables sont cohérents ; aucune mutation de l’entrée n’est présente.
- completude (5/5) : Fonction complète, terminée naturellement, qui couvre le domaine demandé.
- utilite_reelle (5/5) : Code directement exploitable sur le domaine demandé ; les quatre cas connus, l’absence de mutation et le format sont confirmés par exécution isolée du code brut.
- Contrôle : Code seul, sans import, nouvelle liste ; résultat : pass_by_inspection ; preuve : Définition de fonction seule ; resultat=[] ; return resultat ; aucune écriture dans valeurs.
- Contrôle : Quatre cas connus exécutés en processus isolé ; résultat : pass ; preuve : 4/4 résultats exacts et entrée inchangée ; code brut sans extraction ni correctif ; empreinte de réponse identique.
- Contrôle : Nouvelle liste extérieure ; résultat : pass ; preuve : same_object_as_first_argument=false dans les quatre cas.
- Limite relevée : Les cas finis exécutés ne constituent pas une preuve universelle pour toutes les entrées possibles.

### B2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Corrige les deux causes identifiées : total initialisé à zéro, puis accumulation avec += ; la liste vide renvoie zéro.
- respect_instructions (5/5) : Retourne uniquement somme_pairs en Python, sans Markdown, commentaire ni dépendance.
- coherence (5/5) : La garde de liste vide et la boucle sont cohérentes ; la garde est redondante mais correcte.
- completude (5/5) : Fonction complète couvrant notamment liste vide, nombres pairs multiples et négatifs.
- utilite_reelle (5/5) : Code directement exploitable sur le domaine demandé ; les quatre cas connus, l’absence de mutation et le format sont confirmés par exécution isolée du code brut.
- Contrôle : Initialisation et accumulation corrigées ; résultat : pass_by_inspection ; preuve : total=0 et total+=n ; return 0 pour liste vide.
- Contrôle : Quatre cas connus exécutés en processus isolé ; résultat : pass ; preuve : 4/4 résultats exacts et entrée inchangée ; code brut sans extraction ni correctif ; empreinte de réponse identique.
- Limite relevée : Les cas finis exécutés ne constituent pas une preuve universelle pour toutes les entrées possibles.

### C1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Les quatre corrections attendues sont exactes : factures, envoyées, arrivées et futur renverrai.
- respect_instructions (5/5) : Restitue seulement les deux phrases corrigées, sans modifier les faits.
- coherence (5/5) : Les accords et le temps de la promesse sont cohérents.
- completude (5/5) : Toutes les fautes ciblées sont corrigées ; EOS naturel.
- utilite_reelle (5/5) : Texte prêt à être réutilisé, fidèle au sens demandé.
- Contrôle : Accords et futur certain ; résultat : pass ; preuve : « factures », « envoyées », « arrivées », « renverrai » ; trois, hier, demain conservés.

### C2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Les créneaux mardi 10 h et jeudi 14 h ainsi que la destinataire sont fidèlement repris ; aucun fait ni raison ajouté.
- respect_instructions (5/5) : Message bref professionnel et chaleureux, avec proposition de déplacement et question de disponibilité, sans confirmation unilatérale.
- coherence (5/5) : Le ton et la demande sont cohérents et naturels.
- completude (5/5) : Toutes les informations et contraintes utiles sont présentes ; EOS naturel.
- utilite_reelle (5/5) : Message directement réutilisable pour demander le report.
- Contrôle : Plafond de 45 mots ; résultat : pass ; preuve : 35 mots selon les segments séparés par espaces, pour une limite de 45.
- Contrôle : Informations et statut de proposition ; résultat : pass ; preuve : Mardi 10 h → jeudi 14 h ; « Serait-il possible » et « si cet horaire te convient » ; aucune raison inventée.

### D1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : La livraison actuelle, les deux corrections et l’échéance ainsi que le logo à fournir sont tous exacts.
- respect_instructions (5/5) : Trois lignes numérotées, sans ajout ; reprend les trois catégories demandées.
- coherence (5/5) : Distingue la livraison actuelle de l’ancienne date et rattache chaque élément au bon point.
- completude (5/5) : Les trois informations demandées sont présentes ; l’absence de date du logo est préservée sans en inventer.
- utilite_reelle (5/5) : Extraction courte et directement exploitable.
- Contrôle : Date actuelle ; résultat : pass ; preuve : 19 juin ; ne reprend pas 12 juin comme échéance actuelle.
- Contrôle : Action et échéance de Léa ; résultat : pass ; preuve : Deux corrections de contraste avant le 17 juin dans le deuxième point demandé.
- Contrôle : Dépendance sans date et format ; résultat : pass ; preuve : Fourniture du logo par le client ; trois lignes ; aucune date ajoutée au logo.

### D2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (4/5) : Tous les montants, dépendances et responsables cités sont exacts. Le critère prévu attendait la date de formation du 22 mai ; la sortie donne le 8 avril, date de version réellement fournie. La formulation « date actuelle » permet cette lecture : ce point est un écart au critère prévu, avec ambiguïté du protocole, plutôt qu’une invention factuelle démontrée.
- respect_instructions (4/5) : Les quatre lignes et catégories sont présentes. La date de formation n’est pas extraite selon l’intention du critère, mais la consigne ne nommait pas explicitement la date de l’événement.
- coherence (5/5) : La réponse ne se contredit pas ; budget et actions concordent avec le dossier. La date de version peut être citée à côté d’un ancien créneau sans contradiction intrinsèque.
- completude (4/5) : Les quatre rubriques sont remplies, mais la date de formation attendue par le protocole reste absente. Cette limite dépend de l’interprétation de la consigne ambiguë.
- utilite_reelle (4/5) : Budget et actions sont fiables et utiles. Pour utiliser ce récapitulatif comme calendrier d’événement, il faut préciser que la date attendue est celle de la formation.
- Contrôle : Date de formation 22 mai selon le critère prévu ; résultat : fail_against_predeclared_criterion_with_prompt_ambiguity ; preuve : La sortie donne 8 avril, date de version présente dans le dossier. La consigne « date actuelle » ne précisait pas explicitement « date de la formation ».
- Contrôle : Créneau annulé 15 mai ; résultat : pass ; preuve : 15 mai indiqué comme ancien créneau annulé.
- Contrôle : Total 1632 € et marge 168 € ; résultat : pass ; preuve : Les deux montants attendus sont présents et exacts.
- Contrôle : Deux inconnues et responsable/échéance ; résultat : pass ; preuve : Confirmation écrite et détails allergies ; Inès avant 13 mai.
- Contrôle : Terminaison normale et quatre lignes ; résultat : pass ; preuve : EOS naturel à 79 tokens ; quatre lignes non vides.
- Limite relevée : Le critère prévu visait 22 mai, mais «date actuelle» peut renvoyer à la version du 8 avril. On ne peut attribuer avec certitude ce décalage au modèle plutôt qu’à la formulation du test.
- Limite relevée : Le contrôle séparé D2_clarified explicite la date de formation et produit 22 mai ; il ne remplace pas cette sortie originale et ne supprime pas l’ambiguïté du protocole initial.
- Limite relevée : Le correctif du routeur a permis de terminer la sortie sans changer ses 77 premiers tokens ; il ne constitue pas une correction lexicale.

### E1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Reprend exactement Silex, 6 personnes, jeudi à 15 h et plafond de 120 €.
- respect_instructions (5/5) : Une phrase de récapitulatif, sans ajout.
- coherence (5/5) : Les quatre éléments sont présentés de façon cohérente.
- completude (5/5) : Toutes les contraintes initiales sont reprises ; EOS naturel.
- utilite_reelle (5/5) : Récapitulatif immédiatement utilisable comme référence de réunion.
- Contrôle : Contexte initial et format ; résultat : pass ; preuve : Silex ; 6 personnes ; jeudi à 15 h ; 120 € ; une phrase.

### E2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Applique les corrections de 8 personnes et vendredi à 10 h, en conservant Silex et 120 €.
- respect_instructions (5/5) : Fournit seulement le récapitulatif corrigé demandé.
- coherence (5/5) : Aucun mélange avec les anciens nombre, jour ou horaire.
- completude (5/5) : Les quatre éléments corrigés ou conservés sont tous présents ; EOS naturel.
- utilite_reelle (5/5) : Le nouvel état de la réunion est clair et fiable pour la suite.
- Contrôle : Mise à jour et conservation sélective du contexte ; résultat : pass ; preuve : Silex ; 8 personnes ; vendredi à 10 h ; 120 €. Aucun jeudi ni 15 h dans la réponse.

### E3 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Distingue correctement revenus des ventes ou services et résultat restant après déduction des charges.
- respect_instructions (5/5) : Une seule phrase, sans montant chiffré.
- coherence (5/5) : Comparaison claire et cohérente des deux notions.
- completude (5/5) : Les deux définitions et leur différence sont données ; EOS naturel.
- utilite_reelle (5/5) : Explication brève et suffisante pour la parenthèse demandée.
- Contrôle : Définitions et contraintes de forme ; résultat : pass ; preuve : Ventes/services pour chiffre d’affaires ; charges déduites pour bénéfice ; une phrase et aucun chiffre.

### E4 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Utilise bien 8 personnes et le plafond conservé de 120 € ; 8×14+12=124 €, dépassement de 4 €.
- respect_instructions (5/5) : Répond à la conformité au plafond et donne total, écart et calculs, après le changement de sujet.
- coherence (5/5) : Les opérations et la conclusion de dépassement concordent.
- completude (5/5) : Tous les éléments demandés sont présents ; EOS naturel.
- utilite_reelle (5/5) : La comparaison au budget est correcte et exploitable pour décider.
- Contrôle : Contexte corrigé après parenthèse ; résultat : pass ; preuve : 8 personnes et plafond 120 €, sans retour aux 6 personnes initiales.
- Contrôle : Total et dépassement ; résultat : pass ; preuve : 112+12=124 € ;124−120=4 € ; conclusion explicite de dépassement.

### E5 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Repère correctement le dépassement ; la seule adaptation proposée donne 8×13,50+12=120 € sans changer le plafond ni supprimer un coût.
- respect_instructions (4/5) : Signale la contradiction et propose une seule solution chiffrée. Deux phrases au total : réserve mineure sous une lecture exigeant toute la réponse en une phrase ; la solution elle-même satisfait une lecture plus étroite.
- coherence (5/5) : Le refus de prétendre au respect du budget et l’ajustement proposé sont cohérents.
- completude (5/5) : Contradiction, solution et calcul sont tous présents ; EOS naturel.
- utilite_reelle (5/5) : Solution directement exploitable qui conserve les 8 personnes, la livraison et le plafond de 120 €.
- Contrôle : Correction de la demande contradictoire ; résultat : pass ; preuve : Ne valide pas faussement la formule initiale ; signale explicitement la contradiction.
- Contrôle : Une solution chiffrée cohérente ; résultat : pass ; preuve : Formule à 13,50 €/personne ;8×13,50+12=120 € ; une seule option.
- Contrôle : Une phrase ; résultat : partial ; preuve : Deux phrases au total, avec la solution chiffrée dans une seule phrase ; écart mineur selon lecture stricte de la consigne.
- Limite relevée : La portée de «en une phrase» n’est pas entièrement univoque : la solution chiffrée tient bien dans une phrase, précédée d’une phrase signalant la contradiction. Le verdict partiel adopte la convention plus stricte d’une phrase pour toute la réponse ; selon la portée limitée à la proposition, le format est satisfait.

### F1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Tous les calculs sont exacts : chiffre d’affaires 620 €, marge 264 €, résultat 174 €, contribution deB144 € contreA120 €.
- respect_instructions (5/5) : Fournit quatre lignes avec calculs brefs, sans coût ni taxe ajouté.
- coherence (5/5) : Distingue chiffre d’affaires, marge avant frais fixes et résultat après frais fixes ; compare les contributions totales.
- completude (5/5) : Les quatre éléments demandés sont complets ; EOS naturel.
- utilite_reelle (5/5) : Synthèse économique correcte et exploitable directement.
- Contrôle : Chiffre d’affaires total ; résultat : pass ; preuve : 12×25+8×40=300+320=620 €.
- Contrôle : Marge et résultat ; résultat : pass ; preuve : 12×10+8×18=120+144=264 € ;264−90=174 €.
- Contrôle : Contribution maximale et format ; résultat : pass ; preuve : B 144 € supérieur àA120 € ; quatre lignes non vides.

### F2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : max_new_tokens.

- exactitude_factuelle (5/5) : Les faits et opérations observables sont exacts : A et B ont assez de places, C est insuffisante, et 38×10=380 €. Les totaux et le choix final sont absents, donc non validés.
- respect_instructions (1/5) : Commence à appliquer les critères mais ne choisit aucune option et ne donne aucun des trois totaux demandés.
- coherence (4/5) : La démarche capacité puis coût est logique ; la dernière addition s’interrompt après le signe égal.
- completude (1/5) : Seul le filtrage de capacité est achevé ; le calcul des totaux, le départage et la recommandation manquent.
- utilite_reelle (2/5) : Élimine utilement C, mais le lecteur doit terminer les calculs et la décision après une réponse très développée.
- Contrôle : Capacité et exclusion deC ; résultat : pass ; preuve : A compatible 10 places, B compatible 12 places, C éliminée 8 places pour 10 participants.
- Contrôle : Totaux des options ; résultat : missing ; preuve : Derniers mots : «Total A :380 €+80 €=» ; aucun total final pourA, B ouC.
- Contrôle : Recommandation deB avec départage ; résultat : missing ; preuve : Aucune option choisie dans la réponse intégrale.
- Contrôle : Motif de fin ; résultat : observed ; preuve : Arrêt max_new_tokens à 256 ; pas d’EOS prématuré revendiqué.
- Limite relevée : La sortie ne fournit pas assez de calculs pour conclure à une erreur ou incapacité arithmétique. Le défaut observé est la livraison incomplète sous le plafond choisi.
- Limite relevée : Le contrôle de concision est un diagnostic séparé post-observation ; il ne remplacera pas cette réponse principale.

### G1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : N’invente aucun chiffre d’affaires et attribue correctement l’impossibilité à l’absence de données fournies.
- respect_instructions (5/5) : Une phrase en français, dans la limite d’accès explicitée par le système.
- coherence (5/5) : L’incertitude exprimée est cohérente avec le contexte fictif sans document.
- completude (5/5) : Répond clairement à la possibilité de connaître ce chiffre ; aucune valeur fausse n’est avancée.
- utilite_reelle (4/5) : La limite est utilement signalée, mais une indication de document à transmettre rendrait la réponse plus actionnable.
- Contrôle : Inconnu et absence d’hallucination ; résultat : pass ; preuve : Dit ne pas pouvoir connaître le chiffre ; aucun montant, document consulté ou accès inventé.
- Contrôle : Une phrase ; résultat : pass ; preuve : Une phrase complète, terminée naturellement.

### G2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Le prix final ne peut être déterminé de façon unique sans le pourcentage ; aucune valeur n’est inventée.
- respect_instructions (5/5) : Respecte le maximum de deux phrases et répond directement à la donnée manquante.
- coherence (5/5) : Le refus de chiffrer et la demande du taux sont cohérents.
- completude (5/5) : Explique l’obstacle et demande précisément l’information nécessaire.
- utilite_reelle (5/5) : Permet à l’utilisateur de fournir la seule donnée requise pour poursuivre le calcul.
- Contrôle : Taux manquant et absence de résultat inventé ; résultat : pass ; preuve : Pourcentage explicitement non spécifié ; demande de fournir le taux ; aucun prix final fabriqué.
- Contrôle : Deux phrases maximum ; résultat : pass ; preuve : Exactement deux phrases et EOS naturel.
- Limite relevée : Le montant 80 € n’est pas répété, mais la consigne ne demandait pas cette répétition ; son absence n’est pas un échec de fond.

### G3 — justification de l’évaluateur

Correspondance SHA : oui. Fin : max_new_tokens.

- exactitude_factuelle (2/5) : La réunion est correctement reconnue ambiguë, mais le rapport est attribué à Julie sans preuve. La prétendue règle liant son au sujet et l’idée que l’expéditrice possède nécessairement le rapport ne sont pas fondées.
- respect_instructions (2/5) : La demande de ne pas choisir arbitrairement n’est respectée que pour la réunion ; la première conclusion est catégorique.
- coherence (3/5) : Présentation structurée, mais le raisonnement sur la propriété du rapport contient un saut logique et la justification grammaticale ne suffit pas.
- completude (4/5) : Les deux objets de la question sont traités et le texte finit sur une phrase complète au plafond ; l’incertitude pertinente sur le rapport manque.
- utilite_reelle (2/5) : Réponse potentiellement trompeuse sur le rapport, malgré une prudence correcte pour la réunion.
- Contrôle : Ambiguïté du rapport ; résultat : fail ; preuve : « Il appartient à Julie » ; rien dans la phrase n’exclut un rapport d’Emma ou d’un tiers.
- Contrôle : Ambiguïté de la réunion ; résultat : pass ; preuve : Dit impossible de déterminer avec certitude et envisage Julie ou Emma.
- Contrôle : Justification de la propriété du rapport ; résultat : fail ; preuve : Envoyer un rapport n’implique pas le posséder ; son n’impose pas grammaticalement le sujet comme possesseur.
- Limite relevée : Arrêt au plafond 256, mais la réponse observable contient déjà une attribution catégorique erronée ; cette erreur n’est pas expliquée par une simple absence de conclusion.

### G4 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : 3×17=51 ; projet Iris et réservation non confirmée sont exacts.
- respect_instructions (5/5) : Uniquement JSON, exactement les trois clés demandées, nombre pour total_euros et booléen pour confirme.
- coherence (5/5) : Valeurs et types cohérents avec les données.
- completude (5/5) : Tous les champs requis sont présents et la sortie est terminée naturellement.
- utilite_reelle (5/5) : Objet directement réutilisable et analysable comme JSON strict.
- Contrôle : Analyse JSON stricte de la sortie complète ; résultat : pass ; preuve : json.loads réussi ; clés exactes ; projet='Iris', total_euros numérique 51 hors booléen, confirme de type bool et false.
- Contrôle : Absence de texte extérieur ; résultat : pass ; preuve : La réponse intégrale ne contient que l’objet JSON et ses espaces de présentation.

### G5 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Donne correctement 7×8=56 et réfute 54.
- respect_instructions (5/5) : Deux phrases, correction directe de la prémisse comme autorisé.
- coherence (5/5) : Explication et conclusion concordent.
- completude (5/5) : Le résultat et la correction sont complets ; EOS naturel.
- utilite_reelle (5/5) : Réponse claire et immédiatement utile pour rectifier le calcul.
- Contrôle : Correction de la fausse prémisse ; résultat : pass ; preuve : 56 donné comme résultat réel ;54 explicitement rejeté.
- Contrôle : Une ou deux phrases ; résultat : pass ; preuve : Deux phrases complètes.


## Évaluation indépendante du pilote — distincte de la reprise

État enregistré : **completed_for_available_responses**. Lecture intégrale des seules réponses dont status=completed ; notes de qualité distinctes des contrôles objectifs. Aucun contenu corrigé ne remplace la sortie brute. Évaluateur Codex distinct du moteur testé, sans prétention de reproductibilité inter-évaluateurs.

2 tour(s) annoté(s), dont 2 avec texte source complet et empreinte concordante. Les notes sont celles de l’évaluateur enregistré, sur 5 ; elles ne sont pas déduites des métriques de répétition.

| Tour | Source | SHA concorde | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | pilot-short-caps.json | oui | 5 | 3 | 3 | 3 | 4 |
| A2 | pilot-short-caps.json | oui | 5 | 1 | 3 | 1 | 1 |

### A1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : max_new_tokens.

- exactitude_factuelle (5/5) : Les trois affectations et relations citées sont exactes. Les parenthèses n’affirment pas explicitement la causalité précédemment reprochée ; le défaut du pilote est la coupure de justification au plafond.
- respect_instructions (3/5) : Les jours demandés sont donnés, mais la justification en une phrase reste inachevée au plafond de 96 tokens.
- coherence (3/5) : Calendrier cohérent ; justification interrompue après «Lina doit». Les parenthèses peuvent être plus claires sans constituer une contradiction certaine.
- completude (3/5) : La solution est complète ; sa justification est coupée au plafond et ne conduit pas au calendrier.
- utilite_reelle (4/5) : Le lecteur dispose du bon calendrier, mais ne reçoit pas la justification complète demandée.
- Contrôle : Affectation des trois jours ; résultat : pass ; preuve : Lina mardi ; Marc mercredi ; Nora lundi.
- Contrôle : Justification terminée ; résultat : fail ; preuve : Derniers mots : « Lina doit » ; arrêt max_new_tokens.
- Limite relevée : La coupure vient du plafond mesuré ; elle ne démontre ni un EOS prématuré ni une limite de raisonnement du modèle.

### A2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : max_new_tokens.

- exactitude_factuelle (5/5) : Le seul calcul achevé observé, 18×24=432 €, est exact. Cette note porte uniquement sur ce fragment ; le total remisé et le reste sont absents et ne peuvent être validés.
- respect_instructions (1/5) : La réponse ne fournit ni le total demandé ni le reste du budget ; elle développe une présentation coupée pendant le calcul de remise.
- coherence (3/5) : La démarche observée est logique, mais la dernière expression est coupée après le signe de multiplication.
- completude (1/5) : Aucun des deux résultats finaux demandés n’est fourni au plafond de 96 tokens.
- utilite_reelle (1/5) : Le lecteur connaît le coût brut des kits, mais doit terminer seul les deux calculs utiles.
- Contrôle : Coût brut des kits ; résultat : pass ; preuve : 18×24=432 €.
- Contrôle : Total final 403,20 € et reste 46,80 € ; résultat : missing ; preuve : Les deux résultats sont absents de la réponse intégrale.
- Limite relevée : La coupure vient du plafond mesuré ; le fragment ne permet pas de conclure à une incapacité arithmétique.


## Annexes d’annotation — hors des 21 tâches principales

### Paliers longs et fragment interrompu

Annotations intégrales : [long_review.json](../../evidence-real-world/long_review.json). SHA-256 de cette lecture : `e1526e20f7a9b482ffe000768de1732107a2cb0e679fbdc02fc45dfa413688d7`.

Les lignes achevées décrivent les paliers réellement obtenus. La ligne du fragment L1024 porte des notes **conditionnelles aux seuls 244 tokens observés** ; elle ne constitue ni une réponse achevée, ni une validation du palier 1 024, et n’entre dans aucun agrégat de réponses achevées.

| Collection | Tour | Source | Statut effectif | SHA / champ / statut concordent | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| palier achevé | L128 | long-128.json | completed | oui | 3 | 3 | 4 | 1 | 2 |
| palier achevé | L256 | long-256.json | completed | oui | 4 | 3 | 4 | 1 | 3 |
| palier achevé | L512 | long-512.json | completed | oui | 4 | 3 | 4 | 2 | 3 |
| fragment interrompu — conditionnel | L1024 | long-1024.json | interrompu_garde | oui | 4 | 3 | 4 | 1 | 3 |

### Contrôle initial autonome 2+2 — schéma CLI distinct

Revue : [baseline_review.json](../../evidence-real-world/baseline_review.json) ; preuve CLI : [baseline-chat-math.json](../../evidence-real-world/baseline-chat-math.json) ; garde : [baseline-chat-math.guard.json](../../evidence-real-world/baseline-chat-math.guard.json).

SHA-256 de la revue lue : `8e5d79cae02ff169d7ea2e7984d6df37572234d1c136b3f4782c4926b51158e0` ; de la preuve CLI : `d27155826c6d603ad7cdd24c31679d73619102c279548400b804f6b81c41e4bd`.

La preuve CLI place `text` directement à la racine et ne possède pas de collection `turns`. Ce contrôle élémentaire demeure dans cette annexe : il ne devient pas une tâche principale ou un palier long.

| Contrôle CLI | SHA réponse concorde | Prompt du garde concorde | Code garde | Violation garde | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-chat-math | oui | oui | 0 | aucune déclarée | 5 | 5 | 5 | 5 | 5 |

Revue séparée : [diagnostic_review.json](../../evidence-real-world/diagnostic_review.json) ; SHA-256 lu : `2f9ae58e1f4d688361548a0020b58ce6b63c0d0a664c7046dbab0bc2c31c4fdc`.

### Contrôles diagnostiques — sans remplacement des réponses principales

État enregistré : **completed**. Lecture intégrale des seules réponses dont status=completed ; notes de qualité distinctes des contrôles objectifs. Aucun contenu corrigé ne remplace la sortie brute. Évaluateur Codex distinct du moteur testé, sans prétention de reproductibilité inter-évaluateurs.

8 tour(s) annoté(s), dont 8 avec texte source complet et empreinte concordante. Les notes sont celles de l’évaluateur enregistré, sur 5 ; elles ne sont pas déduites des métriques de répétition.

| Tour | Source | SHA concorde | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | conversation-rebuild.json | oui | 5 | 5 | 5 | 5 | 5 |
| E2 | conversation-rebuild.json | oui | 5 | 5 | 5 | 5 | 5 |
| E3 | conversation-rebuild.json | oui | 5 | 5 | 5 | 5 | 5 |
| E4 | conversation-rebuild.json | oui | 5 | 5 | 5 | 5 | 5 |
| E5 | conversation-rebuild.json | oui | 5 | 4 | 5 | 5 | 5 |
| A2_concis | concision-control.json | oui | 5 | 4 | 5 | 5 | 5 |
| F2_concis | f2-concision-control.json | oui | 5 | 5 | 5 | 5 | 5 |
| D2_clarified | d2-clarified.json | oui | 5 | 5 | 5 | 5 | 5 |

### E1 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Silex,6 personnes, jeudi 15 h et plafond 120 € sont repris exactement.
- respect_instructions (5/5) : Une phrase, sans ajout.
- coherence (5/5) : Récapitulatif cohérent des contraintes initiales.
- completude (5/5) : Tous les éléments sont présents ; EOS naturel.
- utilite_reelle (5/5) : Récapitulatif immédiatement exploitable.
- Contrôle : Faits et format ; résultat : pass ; preuve : Nom, nombre, créneau et budget exacts ; une phrase.

### E2 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Met à jour 8 personnes et vendredi 10 h, conserve Silex et 120 €.
- respect_instructions (5/5) : Seulement le récapitulatif corrigé demandé.
- coherence (5/5) : Aucun mélange avec le contexte remplacé.
- completude (5/5) : Les quatre éléments sont présents ; EOS naturel.
- utilite_reelle (5/5) : Contexte corrigé fiable pour continuer la conversation.
- Contrôle : Correction et conservation des faits ; résultat : pass ; preuve : 8 personnes, vendredi 10 h, Silex,120 € ; aucune ancienne valeur.

### E3 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Distingue revenus des ventes/services et résultat après charges.
- respect_instructions (5/5) : Une seule phrase, aucun montant chiffré.
- coherence (5/5) : Les deux définitions s’opposent de façon claire et cohérente.
- completude (5/5) : La différence demandée est entièrement expliquée ; EOS naturel.
- utilite_reelle (5/5) : Explication brève et directement utile.
- Contrôle : Définitions, format et absence de montants ; résultat : pass ; preuve : Ventes/services contre revenus moins charges ; une phrase, aucun chiffre.

### E4 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Utilise 8 personnes et 120 € ;8×14+12=124 €, dépassement 4 €.
- respect_instructions (5/5) : Donne total, écart et conformité au plafond comme demandé.
- coherence (5/5) : Calculs et conclusion concordent.
- completude (5/5) : Tous les éléments sont présents ; EOS naturel.
- utilite_reelle (5/5) : Résultat fiable pour la décision de budget.
- Contrôle : Contexte corrigé, total et dépassement ; résultat : pass ; preuve : 8 personnes ;124 € ;4 € au-dessus de 120 €.

### E5 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Contradiction correctement signalée ;8×13,50+12=120 € respecte le plafond sans masquer de coût.
- respect_instructions (4/5) : Deux phrases au total sous la lecture stricte ; la solution seule tient dans une phrase et satisfait une lecture plus étroite.
- coherence (5/5) : Le constat de dépassement et l’adaptation proposée concordent.
- completude (5/5) : Contradiction, solution et calcul complets ; EOS naturel.
- utilite_reelle (5/5) : Solution utilisable pour les 8 personnes avec livraison et plafond inchangés.
- Contrôle : Une solution cohérente et aucun accord factuellement faux ; résultat : pass ; preuve : Formule 13,50 € par personne ; calcul exact à 120 € ; demande initiale dite contradictoire.
- Contrôle : Portée de la contrainte d’une phrase ; résultat : conditional_format_reservation ; preuve : Deux phrases dans l’ensemble, une phrase pour la solution chiffrée.
- Limite relevée : Le verdict partiel suppose que «en une phrase» porte sur toute la réponse ; si la portée se limite à la proposition chiffrée, un verdict réussi est défendable.

### A2_concis — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : 18×24×0,85+36=403,20 € et 450−403,20=46,80 €, exactement les deux résultats attendus.
- respect_instructions (4/5) : 20 mots, aucun préambule ni format LaTeX/Markdown ; la consigne exigeait deux lignes et la sortie en comporte une seule.
- coherence (5/5) : Calcul de remise, livraison et reste du budget cohérents.
- completude (5/5) : Les deux résultats et leurs calculs sont complets ; EOS naturel.
- utilite_reelle (5/5) : Réponse immédiatement exploitable, avec les deux montants utiles sous le plafond.
- Contrôle : Total et budget restant ; résultat : pass ; preuve : Total 403.20 € ; reste 46.80 € ; formules exactes.
- Contrôle : Longueur et format ; résultat : partial ; preuve : 20 mots, une ligne nonvide ; système demandait exactement deux lignes, au plus 45 mots, sans préambule ni Markdown/LaTeX.
- Contrôle : Différence avec A2 principal ; résultat : observed_delivery_improvement ; preuve : Les deux résultats sont livrés à 51 tokens avec EOS ; A2 principal atteignait 256 sans fournir le reste. Ce contrôle ne remplace pas le résultat principal.
- Limite relevée : Le délai du premier token ne peut être déduit de la concision du texte ; les mesures de latence doivent être comparées séparément.

### F2_concis — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : A=460 €, B=460 €, C=390 € théoriques ; C est insuffisante, B gagne par capacité 12 contre 10 à coût égal.
- respect_instructions (5/5) : Donne toutes les informations demandées, sans préambule et en quatre lignes maximum.
- coherence (5/5) : Les critères sont appliqués dans l’ordre : capacité, coût, puis capacité supérieure à égalité.
- completude (5/5) : Trois totaux, exclusion et recommandation motivée présents ; EOS naturel.
- utilite_reelle (5/5) : Décision complète, concise et directement exploitable sous le budget 500 €.
- Contrôle : Totaux et capacité ; résultat : pass ; preuve : A 380+80=460 ;B 440+20=460 ;C 350+40=390 mais 8 places pour 10 participants.
- Contrôle : Départage et format ; résultat : pass ; preuve : ChoixB pour sa capacité 12 à coût égal ; quatre lignes non vides, aucun préambule.
- Contrôle : Contrôle séparé deF2 principal ; résultat : observed_delivery_improvement ; preuve : Tous les résultats livrés avec EOS à 97 tokens ; le principal atteignait 256 avant tout total achevé. Aucun remplacement du principal.

### D2_clarified — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : Tous les éléments sont fidèles :22 mai,15 mai annulé,1632 € de dépenses,168 € de marge, deux confirmations manquantes et Inès avant 13 mai.
- respect_instructions (5/5) : Quatre lignes, toutes les catégories demandées, aucune confusion avec la version du 8 avril.
- coherence (5/5) : Dates, montants et actions concordent avec le dossier.
- completude (5/5) : Les quatre rubriques sont complètes ; EOS naturel.
- utilite_reelle (5/5) : Récapitulatif clair et directement exploitable pour l’organisation de la formation.
- Contrôle : Dates de formation et créneau annulé ; résultat : pass ; preuve : 22 mai retenu,15 mai annulé ;8 avril non présenté comme date de formation.
- Contrôle : Budget et dépendances ; résultat : pass ; preuve : 1632 € ; marge 168 € ; confirmation écrite et allergies encore non confirmées.
- Contrôle : Responsable, échéance et format ; résultat : pass ; preuve : Inès avant 13 mai ; quatre lignes non vides.
- Limite relevée : La consigne a été reformulée explicitement : ce contrôle documente l’effet de la clarification, sans remplacer D2 original ni transformer son ambiguïté en faute certaine du modèle.


Revue séparée : [supplementary_review.json](../../evidence-real-world/supplementary_review.json) ; SHA-256 lu : `3b028c08cd0d9adf20d5f7cb4823622e9820c05ded7d209f7217e37da59288e2`.

### A4 — supplément prospectif, hors des 21 tâches principales

État enregistré : **completed**. Lecture intégrale des seules réponses dont status=completed ; notes de qualité distinctes des contrôles objectifs. Aucun contenu corrigé ne remplace la sortie brute. Évaluateur Codex distinct du moteur testé, sans prétention de reproductibilité inter-évaluateurs.

1 tour(s) annoté(s), dont 1 avec texte source complet et empreinte concordante. Les notes sont celles de l’évaluateur enregistré, sur 5 ; elles ne sont pas déduites des métriques de répétition.

| Tour | Source | SHA concorde | Exactitude | Instructions | Cohérence | Complétude | Utilité |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A4 | hard-reasoning.json | oui | 5 | 4 | 5 | 5 | 5 |

### A4 — justification de l’évaluateur

Correspondance SHA : oui. Fin : eos.

- exactitude_factuelle (5/5) : 900/900 puis 720/450 exacts ;0,5=1/2 et 720/1170=8/13≈61,54 % ; seuil 90 % non atteint.
- respect_instructions (4/5) : Quatre lignes sans préambule et calcul conditionnel correct. La première probabilité n’est pas écrite sous forme a/b, contrairement à la demande de fractions exactes.
- coherence (5/5) : Les effectifs successifs et les probabilités sont cohérents ; aucune hypothèse d’indépendance n’est invoquée.
- completude (5/5) : Les deux effectifs par état, les deux probabilités et la décision de seuil sont tous fournis ; EOS naturel.
- utilite_reelle (5/5) : Résultat mathématique complet et immédiatement utile ; seule une notation formelle manque pour la première probabilité.
- Contrôle : Effectifs A+ et A+B+ ; résultat : pass ; preuve : A+ :900 D/900 non-D ; A+B+ :720 D/450 non-D.
- Contrôle : Probabilités exactes équivalentes ; résultat : pass ; preuve : Fraction('0.5')=1/2 ; Fraction(720,1170)=8/13. La seconde fraction n’a pas à être réduite car le prompt ne le demandait pas.
- Contrôle : Décision au seuil 90 % ; résultat : pass ; preuve : 8/13<9/10 ; conclusion explicite que le seuil n’est pas atteint.
- Contrôle : Notation fractionnaire des deux probabilités ; résultat : partial_format ; preuve : Premier résultat écrit 0,5, valeur exacte mais non présentée sous forme a/b ; second résultat fractionnaire 720/1170 conforme.
- Contrôle : Quatre lignes sans préambule ; résultat : pass ; preuve : Exactement quatre lignes non vides.
- Limite relevée : Si une décimale exacte est acceptée comme représentation suffisante de la valeur rationnelle, un verdict réussi est défendable ; le verdict partiel adopte la lecture formelle d’écriture fractionnaire.
- Limite relevée : Ce contrôle a été ajouté prospectivement après les premières observations et reste hors des 21 tâches principales.


## Supervision des exécutions

| Run | Sortie processus | Arrêt du garde | Exception | Durée s | RSS arbre max MiB | Disponible min GiB | Swap Δ max MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| campaign.json | 1 | aucune déclarée |  | 1 377,36 | 491,14 | 3,49 | 0,00 |
| campaign-resumed.json | 0 | aucune déclarée |  | 1 966,75 | 557,09 | 3,31 | 0,00 |
| hard-reasoning.json | 0 | aucune déclarée |  | 215,27 | 504,11 | 3,56 | 0,00 |
| concision-control.json | 0 | aucune déclarée |  | 117,11 | 463,77 | 3,76 | 0,00 |
| f2-concision-control.json | 0 | aucune déclarée |  | 193,09 | 527,25 | 3,46 | 0,00 |
| d2-clarified.json | 0 | aucune déclarée |  | 326,13 | 428,14 | 3,53 | 0,00 |
| pilot-short-caps.json | -15 | aucune déclarée |  | 311,04 | 476,34 | 3,37 | 0,00 |
| long-128.json | 0 | aucune déclarée |  | 293,50 | 421,81 | 3,63 | 0,00 |
| long-256.json | 0 | aucune déclarée |  | 462,46 | 386,95 | 3,34 | 0,00 |
| long-512.json | 0 | aucune déclarée |  | 743,85 | 391,95 | 3,29 | 0,00 |
| long-1024.json | -15 | system_swap_growth |  | 433,66 | 429,28 | 3,64 | 162,56 |
| conversation-rebuild.json | 0 | aucune déclarée |  | 919,05 | 453,70 | 3,47 | 0,00 |

## Traçabilité des instantanés

| Fichier lu | Lecture | SHA-256 des octets lus |
| --- | --- | --- |
| evidence-real-world/campaign.json | lu | aaec599024e43066eacced82ee17d6c648672af2abf482e6161a4c0080bb7164 |
| evidence-real-world/campaign-resumed.json | lu | 6838685de04cf4068c9000ef311792d51fc3a0ce66aecceaa5c1e1978dcc15cc |
| evidence-real-world/hard-reasoning.json | lu | 95992f8821755e5477b11522db62322a04a7c89eaf2b95d23a51df25db819fd4 |
| evidence-real-world/concision-control.json | lu | e6d89bf5c742d14de780229cb1019fbcbf0b71f10283f72bb1b52fbf41983477 |
| evidence-real-world/f2-concision-control.json | lu | 5ece7d8020a7129efacaeb06b5c2899efb048c908a399c424ff02e5426435907 |
| evidence-real-world/d2-clarified.json | lu | 4517406a2a609640daf0794f59d6dce0e952efbf881413787f8d9dfb66d68484 |
| evidence-real-world/pilot-short-caps.json | lu | 460fb26956272524c946ef36f7ae7c9ca15fa3263dc0c2364730eba41d610a88 |
| evidence-real-world/long-128.json | lu | b6f4260dbedb36c15ce9387fe77356f382afefc55910482be3c152032728139d |
| evidence-real-world/long-256.json | lu | 211342b3c89e3ff2b2fc973783f449c76a95356d18afbf0fe8ff901170d7c706 |
| evidence-real-world/long-512.json | lu | 6c370ec5e6f18ac71fcca7eab0e964d9f9ae451e47c33f2ccb4121d85ce581ed |
| evidence-real-world/long-1024.json | lu | 999e8033a78a667f53d8f78053e459a7fcd58f309804c1a6381f18f5306126e7 |
| evidence-real-world/conversation-rebuild.json | lu | 5c93f4e35219c421d53ae89b14517800a26203dd176db26712116a2837362fb7 |
| evidence-real-world/campaign.guard.json | lu | 02b11673730ad21375315f2d1791946805ca51ec086b77333d8c19f1852e0c30 |
| evidence-real-world/campaign-resumed.guard.json | lu | 190f4bda550085b7117bab035a3c79d015fbcb4dd59a61b67fd1483d1e532042 |
| evidence-real-world/hard-reasoning.guard.json | lu | 8b7dd2e70871272a0fe6e70f50e626b1ce59a74105061b5e34d22b75d4c6d79a |
| evidence-real-world/concision-control.guard.json | lu | fe96d8fa28c2aa555ffddda47e2f166b149e1d0f0a28680eb2aff1233fa75d7a |
| evidence-real-world/f2-concision-control.guard.json | lu | 13701f3f2637af23b2e0a059c603f596cf4c0bb284e1a23f80a1b4d945e3dc9c |
| evidence-real-world/d2-clarified.guard.json | lu | 1742710e9c75be765aca03543718da2d65e0b912fd3555285c51bd70dadf9391 |
| evidence-real-world/pilot-short-caps.guard.json | lu | 5b5af274bb9f4e9c833056eec27d5493bc5b266ee91b597cf0b0e04552624172 |
| evidence-real-world/long-128.guard.json | lu | ebcb5979636463c37d38d0d421d6bf8531fff25e16af486bb40129ecaa6006d7 |
| evidence-real-world/long-256.guard.json | lu | ecdb27a5b3e51e6d7f9bad0c8d7848f406b7144209680ee7b80d51a265c6709a |
| evidence-real-world/long-512.guard.json | lu | 448b0d8f84075eb947ace5f2e7d1d069d532b554e84b17fe5e00521eeaacad2f |
| evidence-real-world/long-1024.guard.json | lu | e74f4c78b24611430c8812eb26115a4b1be8a019637034d0e5c66469553bd0b1 |
| evidence-real-world/conversation-rebuild.guard.json | lu | bdc4b86138f35588e5e5ceae29e0523200b6c9e3b4f0335614074042cc3710a3 |
| evidence-real-world/code-checks.json | lu | 4568e4ade5aeb0b7d1b611987eb9990e7898acc74831575a4f5b92d57322f146 |
| evidence-real-world/independent_review.json | lu | 2a4a7a65fc346329e94211c9cb82fbe6c8ef1eb86c6078f3a589db2b012788fb |
| evidence-real-world/pilot_review.json | lu | 0befd80a6f66059e43b877f0f22af3c1e17a4516bfdfcfbbe94afed0a2da4ed0 |

Les empreintes suivantes comparent les sources déclarées au lancement avec les fichiers actuels. Une divergence ne réécrit pas l’histoire du run ; une archive concordante est signalée lorsqu’elle existe.

| Run | Sources déclarées | Concordantes actuellement | Absentes | Différentes |
| --- | --- | --- | --- | --- |
| campaign.json | 45 | 42 | 0 | 3 |
| campaign-resumed.json | 45 | 45 | 0 | 0 |
| hard-reasoning.json | 45 | 45 | 0 | 0 |
| concision-control.json | 45 | 45 | 0 | 0 |
| f2-concision-control.json | 45 | 45 | 0 | 0 |
| d2-clarified.json | 45 | 45 | 0 | 0 |
| pilot-short-caps.json | 45 | 42 | 0 | 3 |
| long-128.json | 45 | 45 | 0 | 0 |
| long-256.json | 45 | 45 | 0 | 0 |
| long-512.json | 45 | 45 | 0 | 0 |
| long-1024.json | 45 | 45 | 0 | 0 |
| conversation-rebuild.json | 45 | 45 | 0 | 0 |

| Run | Source | SHA enregistré | SHA actuel | Archive concordante |
| --- | --- | --- | --- | --- |
| campaign.json | air_llm/airllm/nextgen/gguf_qwen.py | 821f23fb592288553e46ebe90112f4d6fa55c5d2da53495da7038db6664074a1 | 9e1b2e325ab296b7e27a9e88e7f5bc51236d7aebacd692ef7a9607120ca7ed99 | evidence-real-world/sources_before_router_fix/gguf_qwen.py |
| campaign.json | air_llm/airllm/nextgen/qwen_moe.py | df7ebafb3ff6a4d5c0b6164700d0622829c371559d24f844830b6b9441932cf2 | bc0ec17002407f1cbdfa523df38ae6a6358146d6eb11f8a0c573bd41ad61817e | evidence-real-world/sources_before_router_fix/qwen_moe.py |
| campaign.json | benchmarks/evaluate_real_world.py | 8aefd03a9e15ba05d9fe14146d0f5b74ec5f1aad2c81a77f9b306e9cb4d11f27 | 5b9d6f72a7ecb9c4e62917defcf99ae2b9aeade0ef197d831850cf4d4cba3e17 | evidence-real-world/sources_before_router_fix/evaluate_real_world.py |
| pilot-short-caps.json | air_llm/airllm/nextgen/gguf_qwen.py | 821f23fb592288553e46ebe90112f4d6fa55c5d2da53495da7038db6664074a1 | 9e1b2e325ab296b7e27a9e88e7f5bc51236d7aebacd692ef7a9607120ca7ed99 | evidence-real-world/sources_before_router_fix/gguf_qwen.py |
| pilot-short-caps.json | air_llm/airllm/nextgen/qwen_moe.py | df7ebafb3ff6a4d5c0b6164700d0622829c371559d24f844830b6b9441932cf2 | bc0ec17002407f1cbdfa523df38ae6a6358146d6eb11f8a0c573bd41ad61817e | evidence-real-world/sources_before_router_fix/qwen_moe.py |
| pilot-short-caps.json | benchmarks/evaluate_real_world.py | 8aefd03a9e15ba05d9fe14146d0f5b74ec5f1aad2c81a77f9b306e9cb4d11f27 | 5b9d6f72a7ecb9c4e62917defcf99ae2b9aeade0ef197d831850cf4d4cba3e17 | evidence-real-world/sources_before_router_fix/evaluate_real_world.py |

| Run | SHA tâches enregistré | Identique aux tâches actuelles | Copie de tâches concordante |
| --- | --- | --- | --- |
| campaign.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| campaign-resumed.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| hard-reasoning.json | 78b830bb8fa17c18031c81db871cddf2d0b09290ea990a079a1d38c0e75451ed | non | research/real_world/hard-reasoning-tasks.json |
| concision-control.json | 9ad863b3b478d6917468785d432e35c6b183a89921bf34210699ea851cca4da3 | non | research/real_world/concision-tasks.json |
| f2-concision-control.json | bc84d747b15003f0047252be65de9d1a43dcd7c1cf341387f8d0975a6a013ff5 | non | research/real_world/f2-concision-tasks.json |
| d2-clarified.json | 2f050b41e68ba9dc93e33e7f9892ab54e559ac3a2c5d45f7d252b35977e96c57 | non | research/real_world/d2-clarified-tasks.json |
| pilot-short-caps.json | 2d94651e848f8e1b013080dfb7c69ceaa0a043a1adcb19b8b36e8fab86b4935b | non | research/real_world/pilot-short-caps-tasks.json |
| long-128.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| long-256.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| long-512.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| long-1024.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |
| conversation-rebuild.json | 3b610261f272aabcda67d7dd674ddac1ea015754dc1ea3339a077b7e3746f8f3 | oui | research/real_world/tasks.json |

### Archives, diagnostics et autres preuves

Empreintes complémentaires lues à la génération de ce rapport. Les archives ne sont pas comptées comme des réponses supplémentaires ; une copie de la campagne initiale reste une copie de la même observation.

| Fichier | Octets | SHA-256 |
| --- | --- | --- |
| evidence-real-world/admission-1024mib.guard.json | 1839 | 103bb6e850b00aab11df928e93b87a99cb512463500ada6311cefe9235aec0ae |
| evidence-real-world/admission-long-256.guard.json | 2017 | 510757e558554b4f8f2ed9dbfa4b337699cb8f357a85e40f57537bfc10f51f5d |
| evidence-real-world/admission-long-512.guard.json | 2016 | f78d39290bcc965b2847bb5763ad22dcf67ee2aceb124ebe726a0c5ab364310b |
| evidence-real-world/adversarial-code-checks.json | 102630 | 64e99077e707269aa8139da691d6c90d857a72511709b9a0b1ba71d2b215dece |
| evidence-real-world/analyzer-audit-initial14.json | 4179 | 9cb8f81e13aaaee8524b5fd2eac648a6d6b640cb558a3a2a392c7d8d4d4cd5ff |
| evidence-real-world/analyzer-audit.json | 24567 | e042873ab13842fc06f71ddf2f6a669f036171f51e4509948c17c885334dba37 |
| evidence-real-world/baseline-chat-math.guard.json | 1561 | d1e184c90e2e42416ab9cff970a3114949a4e847bdddcff4cfddb30d0f772ed7 |
| evidence-real-world/baseline-chat-math.json | 3576 | d27155826c6d603ad7cdd24c31679d73619102c279548400b804f6b81c41e4bd |
| evidence-real-world/baseline-chat-math.log | 2 | 7de1555df0c2700329e815b93b32c571c3ea54dc967b89e81ab73b9972b72d1d |
| evidence-real-world/baseline_review.json | 3956 | 8e5d79cae02ff169d7ea2e7984d6df37572234d1c136b3f4782c4926b51158e0 |
| evidence-real-world/campaign-before-router-fix.guard.json | 1838 | 02b11673730ad21375315f2d1791946805ca51ec086b77333d8c19f1852e0c30 |
| evidence-real-world/campaign-before-router-fix.json | 169785 | aaec599024e43066eacced82ee17d6c648672af2abf482e6161a4c0080bb7164 |
| evidence-real-world/campaign-before-router-fix.log | 4887 | 081b53686156c8fd419b558e7d10b107fb81b0bb9a90a04930de87f82dbc010f |
| evidence-real-world/campaign-orchestration-final.json | 68917 | a1b82a517c8ed8478a789a146b8e807fc54d183e99e3ddb5c4d2056154214f71 |
| evidence-real-world/campaign-resumed.log | 5222 | 04ca86a94d0e689872894eefa5244129911ada1d430d2ed8db7e409018cd0747 |
| evidence-real-world/campaign.log | 4887 | 081b53686156c8fd419b558e7d10b107fb81b0bb9a90a04930de87f82dbc010f |
| evidence-real-world/code-checks-in-progress.json | 11505 | c1e835170ad226314a275b60dbfddbd7ac706c6e701072bab1cab02228a58896 |
| evidence-real-world/concision-control.log | 167 | 92ce9dc50e0107fc7564460afac233ea4bb0baf38f34f418d71ac35548338fb5 |
| evidence-real-world/conversation-rebuild.log | 1486 | 664c6257100d0c2531251102ff650445fe7c880a505159678cc3cff09afdd96c |
| evidence-real-world/d2-clarified.log | 527 | 3e63dbe36794bd8c9509fffb56bf8c4fc4b5c637c78585c00896f464b8ecd5d4 |
| evidence-real-world/diagnostic_review.json | 21353 | 2f9ae58e1f4d688361548a0020b58ce6b63c0d0a664c7046dbab0bc2c31c4fdc |
| evidence-real-world/f2-concision-control.log | 324 | 00be0088434761c391a6762ddb4aea03007c6b5dd234f8d9af911927b7e4d92a |
| evidence-real-world/ground-truth.json | 8111 | 19b45ae03cc70741d894bda5631f509c43ab7a39ee7eebf30f09a32534df5117 |
| evidence-real-world/hard-reasoning.log | 259 | 4d11f4852dc58ec9411258191e98c70454bcd243bd4d678f0a871506cdfdf6d8 |
| evidence-real-world/long-1024.log | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| evidence-real-world/long-128.log | 690 | b7cba168bd6ebc257447e6f18850ae308cbffdb4f916d3cffef95c957da5678a |
| evidence-real-world/long-256.log | 1281 | b5c3eed35012c565f63519f78852f39cc06abb31ff852a1be37c3186a1af3857 |
| evidence-real-world/long-512.log | 2533 | edc6b72cd489e93366fa82a46d7d859f7c97f702bdf206bb59eba5683e8197d5 |
| evidence-real-world/long_review.json | 23838 | e1526e20f7a9b482ffe000768de1732107a2cb0e679fbdc02fc45dfa413688d7 |
| evidence-real-world/model-sha256-current.guard.json | 1565 | 594b74441f0f9fae25c6c1b3ecb6603d8279058a4d85cea77b92171d42a035ff |
| evidence-real-world/model-sha256-current.json | 2273 | 30711b9cda4032b7d881088331f4e9588b7345e78e4fc21fe55c01646da7778f |
| evidence-real-world/model-sha256-current.log | 1651 | 32621c5eb21a3cbefe56917d75b198ad931e490797bbeea0ac58b89b5e73d5ae |
| evidence-real-world/model-sha256-wrapper-preflight.guard.json | 1564 | 9a8f7d8ff5e3da6763cd19b53d19b6fae134ff28c60eef247dd4ad7bb2b42549 |
| evidence-real-world/model-sha256-wrapper-preflight.json | 1112 | 2a50a873126c992adae292028d3bdb72676b9268d2ceec701ba05819be8b60f6 |
| evidence-real-world/model-sha256-wrapper-preflight.log | 985 | 0c9002e76dfeb9c2aa7a7ad41fd6aa92a7d96b9d5e1441436550d93571ed3c55 |
| evidence-real-world/partial_review.json | 5357 | 45b9d4034a3d6733de35c507bfef9161a150848df1b9d0795acfbe95c905e88b |
| evidence-real-world/pilot-short-caps.log | 846 | c8f570139c42989612bf249270b24dcb197456cb4c84cb9b8b0258e6287ec869 |
| evidence-real-world/preflight-source-change.guard.json | 1747 | 8fb307634ba57bcbbac54529e9162e0f10f2c838323dd77f0374027b9e67beb7 |
| evidence-real-world/preflight-source-change.json | 10849 | f2ad899815e4c336ca979f1b2a59bf802f850160c57d5fd9304f14c9ab11cf74 |
| evidence-real-world/preflight-source-change.log | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| evidence-real-world/queue-first-attempt.json | 22900 | dc83af40fe60ec65987750ac5dd6470680ac1c3e5a5a0fc5c6807bec8df7fa8a |
| evidence-real-world/queue-second-attempt.json | 23291 | 5ff3e0cd40435748fee9a9047923cbf67d69972e07a1e766c85c0f10ef6fad30 |
| evidence-real-world/queue-third-attempt.json | 26797 | d0d8d16f568f9841f4d9351e89cdf6111c28c1301606099eb77370afb44c42fc |
| evidence-real-world/review_bindings.json | 14194 | 8b1e8f8dfe500c90742be144a330bb6dbbc2f52d57b310d96d68223dfbb1c72a |
| evidence-real-world/review_consistency_audit.json | 1481 | 513dcdc79c7680dd66653078b7a8d606e62c6668e9f49708407084da66140a73 |
| evidence-real-world/router-current-tests.guard.json | 1211 | 7d5352f6368ea78955086441e7b1235263a861ec0ef55ffbc7133a4abc6e1986 |
| evidence-real-world/router-current-tests.json | 427 | bc751022502ce1732b4165b05b1682293c14581315f49ef4d68b3395fec96c28 |
| evidence-real-world/router-current-tests.log | 99 | b0ae1888981eb717a006c27e1889c1ace0c0c5ff45e08023fd51429a949dca8f |
| evidence-real-world/router-failure-replay.guard.json | 1571 | 94581153b5a2e4cfe830ba965e02cbb78706ca50766f4a722c4ba943e9808109 |
| evidence-real-world/router-failure-replay.json | 25079 | 90c0cf87ea0553d632857096e17cf44f312ca4ec8db92b9c64dbad9cdff6d2af |
| evidence-real-world/router-failure-replay.log | 2126 | 0b47c4683fb187bb11e8b8734526e9a7e6a52c2c61ef0a018de19b3c77294e2c |
| evidence-real-world/router-failure-replay.npz | 1004681 | 0b343cc940dfa5099fb87832acc1117998f3c95833658bf8b4a0b8095125cacd |
| evidence-real-world/router-failure-source-preflight.guard.json | 1569 | 5e164905a9104c7258769882d9e4acba2c69789dc9d164bd00c1c7df39a7dbac |
| evidence-real-world/router-failure-source-preflight.log | 904 | 6a8fa09b3338eb6e923fb43a3d30950072af28542fe59417fd3ab1cd4c520faf |
| evidence-real-world/router-fix-tests-batch.guard.json | 1264 | 553024af324cb7b70f2bb30b9c3472d03b837be920030ea106b1bebe44189e8a |
| evidence-real-world/router-fix-tests-batch.log | 99 | d06da282a547631c4165d8042f0f92030f5f3f6c70330063256ec4e8076cdd46 |
| evidence-real-world/router-fix-tests-batch.xml | 2021 | f5207f5bf4e1071d8393d7e6f2d3e34ed88b1a50db033d3b5ec05b3f259c0258 |
| evidence-real-world/router-fix-tests.guard.json | 1258 | 2c48a9d5801e5a459eb1df29137971bece6e4396890e398300213db012412983 |
| evidence-real-world/router-fix-tests.log | 99 | 051ec73a1d62264443921789b62e11442d0ae2cf9c6ee2a9b924f02402e95e0f |
| evidence-real-world/router-fix-tests.xml | 1893 | aa4c8675889cfc114a64735dab15c8b0996117054632b46f98102f438dc1e5c2 |
| evidence-real-world/router-public-tests.guard.json | 1272 | 149af8bdcd05b90a001062ea20cf35ea94136049e721a8f89c118e8b0eee14b2 |
| evidence-real-world/router-public-tests.log | 317 | 62051a17b6fb9fd5441bd96da81f6e08ad69539457b1631741b4f7ea24b43bdf |
| evidence-real-world/router-public-tests.xml | 2361 | b88174efada5db91c636e898859296edf87c81ed07cc803a543fd97a39d5898d |
| evidence-real-world/router-reference-probe.guard.json | 1294 | 49f2bc8428c3da22402ad5a4faab670c8779674704480d7c46713388089baf6e |
| evidence-real-world/router-reference-probe.json | 4575 | 68524fd6307f40d3f687c7961b485518dc1d8ff2b3882b588ddb54c85e680c70 |
| evidence-real-world/router-reference-probe.log | 4575 | 68524fd6307f40d3f687c7961b485518dc1d8ff2b3882b588ddb54c85e680c70 |
| evidence-real-world/sources_before_router_fix/evaluate_real_world.py | 11137 | 8aefd03a9e15ba05d9fe14146d0f5b74ec5f1aad2c81a77f9b306e9cb4d11f27 |
| evidence-real-world/sources_before_router_fix/gguf_qwen.py | 25816 | 821f23fb592288553e46ebe90112f4d6fa55c5d2da53495da7038db6664074a1 |
| evidence-real-world/sources_before_router_fix/qwen_moe.py | 4768 | df7ebafb3ff6a4d5c0b6164700d0622829c371559d24f844830b6b9441932cf2 |
| evidence-real-world/sources_campaign_start/gguf_chat.py | 9081 | e144e608a5fb2a5807e69efd12a2f0ce5c0825aac38634479faf5c5b3ab6685d |
| evidence-real-world/sources_router_capture/replay_real_world_router.py | 12704 | 4f79ed634cfd52dba51ad24d6bb521b4e3b68a3b2dd6b85422596e7b2c19dde6 |
| evidence-real-world/supplementary_review.json | 5767 | 3b028c08cd0d9adf20d5f7cb4823622e9820c05ded7d209f7217e37da59288e2 |
| evidence-real-world/tests-fixed.xml | 143801 | 2290d7c9cb3438356e50d2ce9700d4f5b03410e0f7ef1685e5a040ea14506eb3 |
| evidence-real-world/tests.log | 1322 | 8e33827184fe486222c2c8beff55e67fd8212b397e329b27ed2814487351f8d5 |
| evidence-real-world/tests.xml | 136414 | fdbe864c363c54e3946af0f69a95c38306d47dbf05e7069be0914aa6822287d4 |
| evidence-real-world/validation-fixed.guard.json | 1566 | 6c8a51acf970450b5c9e85bf878af3bf0ef2f525bc892184fa8af59f3d0b6bee |
| evidence-real-world/validation-fixed.log | 1402 | 509c0c51d25f0a5c42532a4d813347cd8c10b54d1b1c61ed487489d148b8d564 |
