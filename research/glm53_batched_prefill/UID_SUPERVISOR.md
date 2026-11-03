# Verrou UID acquis avant le lancement du helper

Le lanceur `uid_supervisor.py` permet de conserver le fichier `native_run.py` et sa clé numérique inchangés tout en acquérant la propriété du modèle **avant** le démarrage du wrapper. Il est réservé aux diagnostics natifs `full` et `restore`.

## Contrat de propriété

Le superviseur prend le verrou exclusif non bloquant utilisé par `NativeGlmModel.load` : répertoire temporaire `airllm-native-locks-<UID>`, avec la même clé de modèle épinglé produite par `_model_lock_key()`. Un verrou occupé provoque un refus avant le lancement du wrapper ; aucun détenteur existant n'est signalé.

Un bootstrap Python exécute le fichier wrapper inchangé avec `runpy`. En mémoire, il ajoute le descripteur de verrou au paramètre `pass_fds` du seul `Popen` correspondant exactement au helper configuré, avec six arguments et le mode `full` ou `restore`. L'appel natif de métadonnées `--state-parameters` demeure autorisé. Une autre forme d'appel du même helper est refusée avant son lancement.

Le descripteur est ainsi détenu par le superviseur, le wrapper et le helper. Un helper survivant conserve le verrou après la disparition d'un parent. Le superviseur surveille les PID possédés, leurs dates de création et leurs exécutables, ainsi que les identités des fichiers. Une observation OS `lsof`, limitée au PID et au numéro de descripteur du helper, distingue l'héritage demandé d'un descripteur effectivement observé.

Le superviseur **ne prend pas `BENCHMARK_LOCK`**. Le wrapper inchangé continue à prendre son verrou de campagne ; les deux acquisitions sont non bloquantes. Le superviseur transmet une demande d'arrêt au wrapper possédé, dont le gestionnaire existant reste responsable de l'annulation et de l'arrêt natifs. Il ne signale pas directement un helper natif ou une application personnelle.

Les fichiers mesurés restent inchangés. Le reçu externe contient les empreintes du superviseur et du bootstrap, les événements de lancement et les observations de propriété. L'ajout de ce descripteur ne constitue aucune validation de fidélité ou de performance numérique.

## Reproduction depuis la racine du dépôt

`PYTHON` doit désigner un Python disposant de `psutil` et des dépendances du wrapper. Le modèle, le helper compilé, les prompts natifs et le paquet runtime vérifié restent externes. Adapter les chemins absolus aux artefacts correspondants et choisir des noms de sortie neufs.

```sh
$PYTHON -m research.glm53_batched_prefill.uid_supervisor \
  --wrapper research/glm53_batched_prefill/native_run.py \
  --receipt /absolute/results/full8.uid.json \
  -- \
  --config /absolute/cpu10-reference.json \
  --binary /absolute/build/native-prefill \
  --helper-source research/glm53_batched_prefill/native_prefill.cpp \
  --prompts /absolute/native-prompts.json \
  --runtime-libraries /absolute/runtime-v4/bin \
  --output /absolute/results/full8 \
  --microbatch 8 --mode full --diagnostic --max-seconds 5400
```

Pour le diagnostic de restauration, employer `--mode restore` avec un autre `--output` et un autre `--receipt`. Le reçu doit être **hors** du répertoire de sortie du wrapper, car celui-ci refuse un répertoire préexistant. Le superviseur conserve aussi stdout/stderr du wrapper et un flux d'événements à côté de son reçu.

Cette commande peut lancer une inférence réelle coûteuse. Les protections du wrapper restent applicables : secteur, couvercle, continuité d'horloge, intégrité, admission mémoire, RSS de 8,5 Gio et croissance du swap de 32 Mio au maximum. La durée est un plafond de campagne explicite ; elle ne relève aucun seuil mémoire. Aucun modèle n'a été lancé pour tester le nouveau lanceur.

Le `guard.json` produit par le wrapper reste la preuve des protections de l'inférence. Le reçu du superviseur documente uniquement la chronologie du verrou et l'héritage des descripteurs. Une qualification de snapshot nécessite séparément les captures numériques complètes et leur comparaison.

## Validation et limite historique

Cinq tests portables, exécutés sur des processus Python synthétiques sans modèle, vérifient :

- le refus d'un verrou UID déjà détenu ;
- l'héritage réel du descripteur par le descendant et l'absence de double verrouillage du verrou de campagne ;
- la conservation du verrou par le descendant après l'arrêt de son parent bootstrap ;
- le refus des modes non admis ;
- le placement du reçu hors du répertoire de sortie.

```sh
$PYTHON -m unittest research.glm53_batched_prefill.test_uid_supervisor -v
```

**Seul ce mécanisme synthétique a été testé.** Aucune campagne GLM réelle n'a encore été exécutée avec ce superviseur. Le run de référence `native-full8` avait commencé sans verrou UID transmis au helper. Son guardian externe a acquis le verrou seulement le 3 novembre 2026 à 13:52:42 Paris ; le helper n'avait pas hérité de ce descripteur. Le nouveau lanceur ne confère aucune couverture avant lancement rétroactive à cette référence.
