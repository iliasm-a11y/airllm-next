# Protocole Throughput — GGUF natif réel

Le choix des variantes est exploratoire ; la campagne finale est séparée des pilotes. Le fichier `evidence-throughput/primary-balanced/plan.json` fige les prompts, paramètres, empreintes des sources, répétitions et ordre avant le premier résultat apparié.

## Référence et candidat

Référence : chemin numérique de la PR #5, nouvelles options désactivées, cache adaptatif 64 Mio et tuiles 2 048 lignes. Le même exécutable de mesure est utilisé pour les deux variantes ; ce contrôle ne doit pas être décrit comme un ancien binaire chronométré indépendamment.

Candidat retenu : experts déjà routés regroupés par nombre de positions, au plus huit experts par groupe ; concaténation de tuiles évaluées ; tuiles 8 192 lignes ; cache LRU 64 Mio. Le kernel conserve un groupe SIMD par ligne et l'ordre FP32 de chaque somme. Le pipeline asynchrone, l'import sans copie et les groupes SIMD élargis restent désactivés dans la campagne finale, car les pilotes ne justifient pas leur activation.

Budgets identiques : cache 64 Mio, E/S 64 Mio, états/KV 256 Mio, travail 128 Mio, lecture individuelle 16 Mio, préremplissage par blocs de 16 et contexte 1 280 tokens. GPU FP32/Metal, TF32 désactivé avant initialisation, pread, aucun préchargement prédictif. Tokenizer GGUF réel, template conversationnel français existant, greedy, thinking désactivé, EOS actifs.

## Plan et arrêts

Trois prompts synthétiques de longueurs différentes, deux répétitions par prompt et variante, soit douze processus frais et six paires. Chaque prompt a une paire référence-candidat et une paire candidat-référence ; l'ordre des paires est mélangé avec la graine publiée. Le plafond est conservé dans chaque tâche, mais une observation principale exige EOS : un plafond atteint reste incomplet et interrompt la suite.

Un premier plan purement aléatoire donnait cinq ordres référence-candidat pour un seul ordre inverse. Il a été arrêté pendant le premier préremplissage, avant tout résultat apparié, pour contrebalancer l'ordre. Le fragment, le garde et `primary/design-amendment.json` sont conservés ; aucun seuil mémoire n'avait été franchi.

Chaque enfant est supervisé : RSS de l'arbre 2 Gio, réserve système 3 Gio, admission 768 Mio, croissance maximale autorisée du swap système 128 Mio, pression normale exigée, échantillonnage 50 ms, délai maximal 900 s. Une violation ou un échec arrête la campagne sans relance. Le swap initial est conservé séparément ; l'activité d'autres applications peut provoquer un arrêt conservateur. Aucun cache système n'est vidé et aucune application personnelle n'est fermée.

## Mesures et fidélité

Le TTFT moteur commence avant le premier forward, après le rendu du template. Le débit de décodage est `(tokens générés - 1) / (fin - premier token)`, EOS compris. Le débit de bout en bout utilise tous les tokens et le temps total du superviseur, lancement et tokenizer inclus. Les agrégats sont des ratios de sommes ; les rapports de vitesse appariés sont également conservés.

Chaque forward enregistre une empreinte SHA-256 du vecteur complet de logits et des ensembles/comptes d'experts par couche, la position de l'état, les appels pread, les octets logiques, le compteur stockage du processus, le temps mural et le temps CPU. L'empreinte des routes ne représente pas les probabilités brutes du routeur. Les IDs d'entrée et de sortie doivent être identiques sur chaque paire. L'analyse refuse une divergence de ces empreintes. Les huit positions de captures séparées vérifient aussi les états récurrents/convolution/KV complets et les vecteurs de logits binaires ; ces captures ne sont pas assimilées à toutes les positions longues.

L'empreinte des logits et les petits journaux par token font partie du coût de mesure pour les deux variantes. La capture binaire et le hachage des états complets sont réservés aux contrôles de fidélité, hors campagne chronométrée.

Les octets logiques sont ceux demandés au lecteur natif, relectures comprises. `ri_diskio_bytesread` est un compteur macOS du processus, incluant lectures annexes et réadahead ; ce n'est ni une mesure NAND ni un compteur de cache OS par lecture. Les caches OS restent non contrôlés. L'ordre contrebalancé réduit ce biais sans le supprimer.

## Profilage et résultats négatifs

Les profils sont des expériences séparées. Les intervalles de temps mural hôte exclusifs ne se chevauchent pas ; les intervalles inclusifs sont imbriqués et ne doivent pas être additionnés. Le temps CPU du processus inclut les threads du runtime. Une attente `mx.eval` comprend soumission, exécution des graphes en attente et synchronisation ; elle n'est pas un temps GPU isolé. L'API MLX installée n'expose pas les timestamps GPU par commande et `xctrace` est absent. La déquantification Q4/Q6 est fusionnée au kernel, sans durée GPU séparée inventée.

Tous les pilotes, variantes moins rapides, essais interrompus et identités des sources sont conservés. Les contrôles de vérification à positions connues ne deviennent jamais des gains de génération. L'étude du draft par recherche dans le préfixe utilise uniquement le passé pour proposer et les tokens enregistrés pour évaluer après coup ; elle ne mesure ni vérification cible, ni rollback récurrent, ni accélération de génération.

Les objectifs 1,5 / 2 / 4 / 8 tokens/s restent des objectifs, indépendants des meilleurs petits pilotes. Les besoins de bande passante calculés à partir des lectures logiques sont conditionnels, sans preuve d'une limite physique du SSD ou du GPU. Aucun maintien thermique prolongé, SSD externe ou modèle de plusieurs centaines de milliards de paramètres n'est validé.
