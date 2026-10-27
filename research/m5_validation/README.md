# Banc autonome de validation Apple M5

Ce banc exécute les trois générations existantes dans des processus séparés et
séquentiels. Il ne ferme aucune application, ne purge aucun cache système et ne
télécharge aucun modèle. Codex peut être fermé après la préparation ; le processus
du terminal utilise directement l'interpréteur Python installé.

## Préparation après gel des sources

Les références doivent rester exactement H1 `c8f81e57463ed3d7fd3d7dba226a65500439a9ce`
et H2 `5e8234181e736c0319a64df25e6813e92dd3e6da`. Le candidat doit descendre de
la PR #12 `3b137621d69f3549a85b7d0cc20460bd07b11fb9`. La préparation capture le
commit, les sources Python, les scripts, les prompts, les limites et l'identité du
checkpoint. Un changement ultérieur impose une nouvelle préparation. Il faut donc
préparer la campagne définitive **après le dernier commit du code**.

Depuis la copie du candidat, avec les chemins présents sur ce Mac :

```sh
M5_PYTHON=/Users/macbook/Documents/Codex/2026-10-22/files-pasted-by-the-user-mission/work/validation-venv/bin/python
"$M5_PYTHON" benchmarks/m5_validation.py prepare \
  --model /Users/macbook/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d \
  --expected-sha256 f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d \
  --h1 /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/h1 \
  --h2 /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/h2 \
  --candidate /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/airllm-next \
  --pairs 3 --output /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/campagne-m5
```

Le SHA attendu désigne le checkpoint historique original ; son nom seul n'est pas
une preuve actuelle. Le banc exige une lecture SHA-256 complète sous surveillance
avant l'inférence, puis après chaque expérience terminée et à la fin. La préparation
ne lit pas les 23,94 Go du checkpoint et ne lance aucun worker.

## Exécuter et reprendre

Après avoir volontairement libéré les ressources nécessaires :

```sh
"$M5_PYTHON" benchmarks/m5_validation.py run \
  --campaign /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/campagne-m5
```

La même commande reprend la campagne. Le verrou interdit deux contrôleurs
simultanés. Chaque tentative possède son propre dossier ; aucun résultat antérieur
n'est écrasé. Les expériences dont les workers et les hashes avant/après sont
scellés sont conservées. Seule une expérience interrompue avant son hash final doit
être rejouée ; ses anciens fichiers restent des preuves brutes exclues des mesures
acceptées. Un hash final peut servir de hash initial à l'expérience suivante lorsque
l'identité du fichier reste inchangée. Les timestamps lient chaque worker à ce bloc.

Le défaut est le refus immédiat lorsque l'admission échoue, avec sortie `75` et
statut `deferred_resources`. Une observation bornée peut être demandée avec
`--resource-wait-seconds 60`. Elle exige cinq secondes continues de pression normale
et 256 Mio de marge supplémentaire, puis la garde refait sa propre admission. Il
n'existe aucune boucle d'attente illimitée. `--experiment h1_vs_h2` limite une session
à une expérience ; plusieurs options `--experiment` sont possibles. Une campagne
partielle ne passe pas le vérificateur final comme campagne complète.

Les limites de l'inférence restent : réserve système 3 Gio, RSS des processus
possédés 2 Gio, croissance du swap système 128 Mio, payload d'admission 768 Mio,
pression macOS normale, au plus 32 processus possédés. Elles sont vérifiées
sémantiquement en plus des hashes de configuration. Les kernels utilisent le même
contrôle global avec une empreinte plus petite : RSS 512 Mio, payload 256 Mio,
allocation MLX 128 Mio et cache MLX nul. Le hash utilise un payload de 64 Mio.
Les limites de pression, réserve et swap ne sont pas relevées pour ces petits cas.

Au lancement, les kernels sont d'abord compilés et comparés aux références sur la
grille complète de 160 cas Q4_K/Q6_K, groupés ou non, avec blocs de 2,16,32,64,128 et
plusieurs formes matricielles. Une compilation impossible, un rejet numérique ou
l'absence de gain observé exclut le backend concerné et conserve la référence. Une
petite campagne `--quick` ne satisfait pas ce préalable. Les essais H1/H2, transport
et spéculation peuvent continuer lorsque seul TensorOps est indisponible. Un rapport
kernel déjà collecté peut être fourni avec `prepare --kernel-report ...` ; son
fichier de garde adjacent et ses sources sont vérifiés, et une copie est conservée
dans la campagne.

Les interruptions normales, erreurs et SIGTERM entraînent le nettoyage des seuls
processus observés du contrôleur. Les PID sont associés à leur date de création ;
un PID réutilisé ne peut pas autoriser la terminaison d'un autre processus. Les
identités persistantes permettent également de nettoyer les descendants observés
au prochain démarrage après un crash. Comme tout programme utilisateur, le banc
ne peut garantir une action immédiate après SIGKILL ou une coupure du système,
notamment dans la fenêtre entre création d'un enfant et observation de son identité.

## Comparaisons et conditions d'acceptation

Les 24 expériences comprennent H1/H2, blocs fixes 16/32/64/128, TensorOps,
réutilisation 2/4/8/16, lectures groupées, transport historique/direct, import sans
copie, pipeline gate/up et profondeurs spéculatives 2/4/8/16. Les paires alternent
AB puis BA. La combinaison finale exige au préalable une fidélité acceptée et une
amélioration mesurée des latences de requête **et** du worker complet pour chaque
composant ; un composant plus lent donne une exclusion explicite de la combinaison.

Le prompt historique demandant `42` reste une référence de préfill et ne suffit pas
au décodage soutenu. Les copies synthétiques favorables au prompt lookup et le
guide de jardinage original sont des charges distinctes, jamais agrégées. Le guide
vise environ 180 mots, avec une borne de 384 tokens. Moins de 64 tokens effectivement
émis ne fournit pas une preuve de décodage soutenu. EOS et limite de tokens sont
enregistrés tels qu'observés ; une sortie tronquée n'est pas appelée réponse complète.

Pour chaque paire, le vérificateur standard-library retenu contrôle tous les logits
FP32 émis, les tokens, les positions, les routes d'experts et les arrêts. Les sorties
complètes et les fichiers logits `.f32` restent disponibles. Il contrôle aussi les
forwards spéculatifs rejetés et les préfixes acceptés ; cela ne certifie pas une
égalité directe de toutes les matrices récurrentes DeltaNet/KV cachées.

Les synthèses séparent initialisation, premier token, latence de requête, durée du
worker, durée du processus supervisé, décodage, CPU, lectures logiques et octets par
token émis. Les mesures actuelles sont instrumentées pour la fidélité. Il reste
nécessaire de corroborer les temps dans une campagne appariée sans observateur avant
une conclusion forte de performance. Les temps hôte synchronisés ne sont pas des
temps GPU purs ; ces derniers restent `null` en l'absence d'horodatage GPU exploitable.
Un worker frais ne prouve pas un SSD froid ; le cache système reste incontrôlé.
Aucun compteur macOS n'est présenté comme une mesure NAND certifiée.

## Vérification indépendante des fichiers

```sh
"$M5_PYTHON" benchmarks/verify_m5_validation.py \
  /Users/macbook/Documents/Codex/2026-10-27/files-pasted-by-the-user-mission-5/work/campagne-m5
```

Ce lecteur n'importe ni MLX, ni NumPy, ni le moteur, ni un gestionnaire de processus.
Il recalcule les hashes du manifeste, la fidélité des paires, les limites des gardes,
les hashes du checkpoint et les préalables des kernels à partir des preuves brutes.
Il refuse une campagne différée, partielle, modifiée ou non scellée. Une campagne
terminée avec exclusions peut être acceptée comme **preuves des essais effectués** ;
les exclusions restent explicites et ne deviennent jamais des gains matériels.
Le manifeste établit une cohérence des fichiers, pas l'authenticité externe des données.

Une copie du dossier peut être vérifiée ailleurs : les chemins absolus des commandes
historiques restent conservés, et les rapports sont résolus dans la copie locale.
La reprise de l'exécution doit en revanche utiliser le dossier d'origine ; déplacer
une campagne impose une nouvelle préparation pour toute nouvelle mesure.

## Contrôles légers du banc

```sh
"$M5_PYTHON" -m pytest --noconftest -q tests_hyperstream3/test_m5_validation.py
```

Ces tests vérifient les refus, verrous, processus factices légers, interruptions,
manifeste, politiques et preuves falsifiées. Ils n'exécutent ni MLX ni un modèle,
et leurs fixtures de mesure synthétiques ne constituent aucun résultat matériel.
