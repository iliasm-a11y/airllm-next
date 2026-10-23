# Notes de lancement

- Premier contrôle CLI avant correction du contrat : `baseline-chat-math.json`, réponse « 4 » au prompt conversationnel exact conservé ; cache 256 Mio, KV 192 Mio.
- `preflight-source-change.*` : premier lancement arrêté volontairement avant premier token après modification concurrente de la source de session. Le hash initial et sa copie sont conservés. Ce lancement ne compte pas comme réponse évaluée.
- `admission-1024mib.guard.json` : lancement refusé avant création du processus ; 4 059 447 296 octets disponibles pour 3 Gio de réserve + 1 Gio admis. Pression normale. Aucun seuil abaissé.
- Campagne : cache réduit à 64 Mio, KV 256 Mio, contexte 1280, en réponse à cette marge système. Les poids restent identiques. Les comparaisons de réemploi de contexte utilisent ces mêmes budgets.

- Amendement après deux réponses pilotes coupées à 96 tokens : `pilot-short-caps.*` et fixtures originales conservées. A1 fournit les affectations exactes mais une justification inachevée ; A2 est interrompu avant le total demandé. Toutes les 21 demandes principales sont rejouées, sans changer leurs prompts ni poids, avec plafond 256. Arrêt volontaire avant émission du premier token A3 ; aucune sélection des seuls résultats réussis. Les plafonds de la série longue restent 128/256/512/1024.

- Contrôle supplémentaire décidé après A2 : même message utilisateur et même plafond 256, ajout au message système d’une contrainte de deux lignes/45 mots/sans préambule ni LaTeX. Exécution séparée après campagne, avec budgets identiques. Il ne remplace pas A2 et ne constitue ni un gain matériel ni une correction des poids. Les deux résultats numériques restent exigés.
