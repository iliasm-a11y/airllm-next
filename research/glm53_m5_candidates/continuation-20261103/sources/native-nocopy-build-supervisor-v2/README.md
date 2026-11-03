# Build NoCopy : superviseur dédié v2

Source seulement, non importée/non exécutée. Aucun test, AST, compilation,
metadata native ou GPU pendant C16. Ancien `guarded_build_once.py`, ancien
`native-noblas-nocopy-candidate` et package NoCopy clos restent inchangés.

Ce nouveau chemin résout le double verrouillage en utilisant **un seul
superviseur dédié**. Il ne faut pas le lancer sous `guarded_build_once.py` : le
superviseur acquiert lui-même BENCHMARK_LOCK puis le vrai verrou UID du modèle,
sans attente ni éviction. Il n'existe aucun flag/env « already guarded » permettant
de sauter ces étapes. Un verrou occupé fait refuser le build avant compilation.

La commande de compilation est reprise du reçu réellement compilé NoCopy, avec
le même compiler/SDK/flags/includes/libraries/rpath, en changeant seulement la
traduction source du helper approuvé et la nouvelle sortie isolée. Les mathlibs
ne sont pas reconstruites ou remplacées. Le nouveau reçu driver suit toutes les
sources, headers et bibliothèques avant/après, y compris ce superviseur, son
manifest et le candidat source ancien qu'il consomme en lecture seule.

Chaque phase (compiler, signature, cinq smokes paramètres et un smoke registre)
est un enfant Popen propre. Un bootstrap Python `-I -S` bloque avant exec : pas
de sitecustomize/startup utilisateur avant GO. Le superviseur observe le PID,
birth/exe et les deux FD BENCHMARK/UID réels par path/device/inode avant GO, puis
observe aussi l'exécutable cible et ses FD après exec. Les FD sont explicitement
hérités du parent ; aucun simple booléen de reçu ne remplace cette observation.
Un outil qui ferme ces FD ou sort trop vite pour l'observation fait refuser la
phase, sans transformer l'inconnu en succès. Aucune compilation ne peut commencer
avant la libération du GO par le propriétaire qui détient déjà les deux verrous.

La surveillance applique la politique native originale avec ses limites 8,5 GiB,
croissance swap 32 MiB, réserve/headroom et grâces inchangées. L'enveloppe de build
conserve un plafond plus strict de 2 GiB pour le coordinateur, l'arbre de phase
et l'assertion idle connue (somme conservatrice des RSS). Admission initiale
secteur/pression1 et 3 GiB disponibles, comme l'ancien build gardé. Deadline
globale 300 s ; compiler 120 s, codesign 30 s, chaque smoke 15 s. Les pauses
d'observation GO sont bornées. Les erreurs après lancement ne signalent que les
processus réellement possédés, après recontrôle birth/exe ; aucun détenteur de
verrou étranger n'est attendu ou tué.

Le nettoyage examine tous les descendants déjà observés même si le Popen
principal est sorti. Une transition exec du même PID/birth est réobservée et
consignée avant le recontrôle exe immédiatement préalable au signal. Tout
signal forcé, survivant, dépassement ou échec d'assertion rend la phase refusée ;
le nettoyage n'est jamais présenté comme une fin naturelle. Le garde conserve
le timestamp monotonic exact de chaque appel à la politique mémoire originale,
afin que le futur validateur puisse rejouer les grâces sans les approximer.

Le garde est autoritaire jusqu'au postflight, y compris l'intégrité, AC/horloge,
la fin des assertions idle, tous les retours enfants et la fermeture des FD
parent après reaping. Le fichier `driver-build.json` demeure `guard_pending`
pendant toutes les phases et est alors refusé par les validateurs existants.
Une erreur de postflight écrit un reçu `failed`, même si compiler et smokes
avaient réussi. La validation favorable de schéma n'est faite qu'en mémoire ;
aucun fichier accepté n'est publié avant la clôture réelle du garde. Le reçu
favorable est la dernière publication atomique et lie le SHA du garde clos,
de la source superviseur et du manifest. Il ne constitue aucune qualification
de modèle ou de performance.

Les smokes lisent les paramètres et les bibliothèques chargées, sans modèle et
sans matrice du bridge. L'initialisation historique Metal peut utiliser le
device : `bridge_matrix_commands_observed=0` ne prétend pas mesurer toute
activité interne du pilote. Aucune source CPP ou shader supplémentaire n'est
modifiée. Durée réelle, RSS maximale et résultat du build v2 restent inconnus.

Commande future **directe**, seulement après clôture C16, revue et signal root :

```text
/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/venv/bin/python /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/native-nocopy-build-supervisor-v2/supervised_build.py --destination /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/native-nocopy-build-supervisor-v2/build-run1 --plan /Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/native-nocopy-build-supervisor-v2/build-plan.json --execute
```

Sans `--execute`, le chemin ne fait qu'un plan de fichiers. Même ce plan n'a pas
été exécuté pendant C16. Sélection de source pour revue : `supervised_build.py`,
`source-manifest.json`, ce README et le reçu de revue indépendant futur. Aucun
binaire ni preuve numérique nouvelle n'est inclus.

La révision distincte `../native-noblas-nocopy-probe-compare-v3` sélectionne le
helper via le rôle hashé `input_roles.helper_source` : le suffixe du nom est
ambigu puisque le reçu suit aussi la source de géométrie CPU v4. Elle exige le
garde de build final réellement passé, son SHA, ses huit phases, leurs logs,
les observations FD avant/après GO, le postflight et la fermeture des capacités.
Ces preuves restent dans une enveloppe de provenance séparée ; la carte des
sources mathématiques du runner et sa clé ne sont pas réécrites. L'ancien
comparateur et tous les anciens gels restent intacts.
