# Preuves GLM-5.3-Flash Q2_K_S / Apple M5

Ces fichiers sont des preuves de mesures et de diagnostics. Ils ne contiennent
aucun poids du modèle, binaire d'exécution, base SQLite, dossier personnel,
fichier de clé API ni nonce d'authentification. Des commandes brutes peuvent
contenir les chemins de leurs entrées ou d'un fichier de clé; son contenu
n'est jamais inclus. Les échecs et refus de fidélité restent présents.

Les JSON de mesures et comparaisons complets sont compressés avec gzip
(mtime=0). La synthèse `summary-finished.json.gz` reprend les essais terminés;
les valeurs CPU historiquement nommées user_ns/system_ns sont des ticks Mach,
convertis avec le timebase 125/3 ns par tick. Les lectures macOS comptabilisées
par processus ne sont pas des octets NAND et les deltas par requête sont
séparés des lectures cumulées comprenant le chargement.

## Archives de capture sélectionnées

`capture-*-selected.zip` contient les logits complets, les tenseurs routeurs
et les top-k ordonnés capturés, ainsi que run/prompt/tokens/complete et un index.
Cette sélection **omet KDA et le tenseur result_output redondant**. Son
manifeste est nommé `tensors.selected.jsonl`, volontairement différent du
manifeste exigé par le comparateur complet. Ne pas exécuter le comparateur
de fidélité complète sur cette sélection et ne pas la présenter comme une
validation de toutes les opérations ou des états internes.

Les captures locales complètes, y compris tous les `kda_scan_out-*` observés,
sont conservées dans `Preuves-GLM-M5.zip`, sous `captures/`. Les sorties KDA
capturées ne constituent pas une capture de toutes les matrices d'état
récurrent interne. `full-local-manifest.json.gz` indique SHA-256 et taille de
chaque contenu local; le manifeste public donne aussi le hash de l'archive
locale. Les archives sélectionnées incluent le hash du manifeste de leur
capture complète afin de relier précisément les deux périmètres.

## Provenance du diagnostic

`probe/provenance.json` décrit les sources initiale et finale et les hashes
des binaires enregistrés par les gardes. La source initiale du diagnostic
stock est préservée: elle précède les contrôles renforcés de fichiers/logits
et de lecture de rusage. Les sources initiale et finale ne sont donc pas
identiques. Le graphe de calcul du modèle n'a pas été changé par ces contrôles;
les comparaisons de valeurs capturées sont un résultat distinct, consultable
dans les JSON complets, pas une conséquence supposée du hash de source.

Le profil `stock-cpu4/operations.jsonl.gz` utilise des callbacks et une
synchronisation par opération. Il sert à attribuer des coûts dans ce chemin
instrumenté; ses durées ne sont pas un débit normal, ni un temps de kernel GPU
isolé. Le swap et la pression sont des compteurs système; l'utilisation GPU
IORegistry inclut le bureau et les autres processus.

Une mesure RONIN encore en cours est exclue de la publication. L'archive
locale peut en contenir un instantané explicitement étiqueté; un nouvel
emballage après sa fin publie le résultat terminé sans modifier le brut.
Les dates ZIP sont fixes, les chemins des membres sont relatifs et la
compression niveau 1 utilise un seul processus. Le manifeste public ne
s'auto-hashe pas; il inventorie les autres fichiers de ce dossier.
