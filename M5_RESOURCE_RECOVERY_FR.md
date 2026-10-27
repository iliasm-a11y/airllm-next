# Libération automatique et première validation réelle du M5

27 octobre 2026. Ce rapport poursuit la PR #13 après l'autorisation explicite
d'arrêter proprement les processus non essentiels identifiés avec certitude.
Il remplace le constat « aucun GPU exécuté » de la session initiale pour les
essais décrits ci-dessous. Aucun résultat de la campagne complète n'est anticipé.

## Nettoyage effectivement réalisé

Quarante paires de services MCP Firebase/Playwright redondants ont été arrêtées
par SIGTERM : **80 processus arrêtés, aucun survivant, aucun SIGKILL**. Avant
chaque signal, le script a revérifié PID, date de création, UID, parent, exécutable,
arguments exacts, répertoire, descendants, absence de terminal, de connexion IP,
de navigateur et de document ouvert. Les chats associés étaient inactifs.
L'inactivité CPU seule n'a pas servi à décider d'un arrêt.

Les parents Codex/application, les processus système, les services non attribués
et les applications susceptibles de contenir du travail non enregistré ont été
préservés. Aucun fichier personnel, modèle, téléchargement ou application n'a
été supprimé. Les inventaires privés restent locaux ; les preuves publiables
omettent les arguments et données sans rapport avec la mesure.

| Mesure système | Avant | Après les arrêts |
|---|---:|---:|
| Disponible | 3 094 659 072 octets | 5 777 031 168 octets |
| Pression macOS | 2 | 1 |
| Swap utilisé | 3 450 208 256 octets | 3 223 453 696 octets |

Ce sont des instantanés système avant/après, pas une somme de mémoire physique
exclusive imputable aux processus. Les identités, raisons et signaux sont dans
`research/m5_validation/evidence/recovery-20261027/cleanup-actions.json`.

## Compilation et exécution GPU réelles

Un second lot de **8 processus** inutilisés de cette mission a ensuite été
revérifié et arrêté par SIGTERM, sans survivant. Le total est donc **88 arrêts
propres**, aucune terminaison forcée. Pour ce second lot, le disponible passe
de 5 596 774 400 à 5 947 326 464 octets, pression 1 et swap inchangé.

Sur **Apple M5 / MLX 0.32.4**, les **160 cas** de la grille complète ont été
compilés et exécutés. Le premier essai interrompu et son erreur sont conservés.
Le second a terminé en **5,14 s de supervision**, avec un pic RSS échantillonné
d'environ **67,3 Mo**, pression 1 et aucune croissance du swap observée.
Cette durée est celle du banc complet, pas un temps GPU pur.

Les **128 sorties de réutilisation** sont identiques bit pour bit aux sorties
du GPU historique. TensorOps compile sur ses 32 cas, mais des divergences
numériques empêchent sa qualification globale ; il reste désactivé.
Des désaccords existent aussi entre le kernel historique et l'oracle utilisant
une autre réduction FP32. Les entrées/sorties brutes, les réductions FP64 et le
décodeur C GGML permettent de les examiner : la déquantification des 32 jeux
est identique bit pour bit au C. Les tolérances n'ont pas été relâchées.

La politique v2 autorise explicitement la réutilisation sur la seule preuve
brute d'égalité exacte avec le chemin historique sur toute sa grille ; les
désaccords scalaires restent des résultats négatifs visibles. Cette exception
ne concerne pas TensorOps. Une comparaison indépendante des vrais logits,
tokens, routes et conditions d'arrêt reste exigée pour chaque paire réelle.
Ces miniatures ne certifient pas à elles seules le modèle complet.

## Le vrai checkpoint répond jusqu'à EOS

Un essai H1 sur le GGUF original de 23 938 321 664 octets a produit un guide
français de jardin partagé, **319 tokens générés jusqu'à EOS** :

| Mesure de cet essai | Résultat |
|---|---:|
| Initialisation | 0,712 s |
| Premier token | 29,431 s |
| Décodage | 1,46747 token/s |
| Génération totale | 246,130 s |
| Pic RSS supervisé | 484 278 272 octets |
| Croissance du swap observée | 0 |
| Pression maximale observée | 1 |

Les sources, le protocole et l'identité du fichier sont restés inchangés.
La réponse intégrale et les 319 vecteurs de logits sont conservés localement.
Cet essai fonctionnel n'est pas une comparaison AB/BA ni une preuve de gain
face aux **1,584 token/s historiques**. Son hash complet préalable n'ayant pas
été exécuté dans cette tentative exploratoire, il est séparé des futures
expériences scellées avant/après par SHA-256.

## Première paire H1/H2 : résultat négatif conservé

Le checkpoint a maintenant également été haché intégralement avant et après
la première paire réelle : les deux SHA-256 correspondent à la valeur attendue.
Pour le même prompt historique de 364 tokens, les deux moteurs répondent `42`
puis EOS. Le premier token passe de **158,071 à 53,476 secondes** dans cette
paire ; les lectures logiques de la génération passent de 139 447 912 832 à
40 928 599 936 octets. Il s'agit d'une seule paire, sans répétition BA achevée.

**Le gain n'est pas qualifié par le contrat strict de fidélité.** À la couche
30, position 154, les experts 193 et 104 échangent leurs rangs 7 et 8. Aucun
ensemble d'experts sélectionnés ne change parmi les 14 640 positions observées.
Les 744 960 logits respectent tous les tolérances inchangées, avec un écart
maximal de 1,13249e-5. Les scores du routeur n'ayant pas été enregistrés, un
effet de quasi-égalité reste une hypothèse, pas une cause démontrée.

La liste ordonnée n'est pas normalisée pour contourner le contrôle : le rejet
est conservé. La campagne peut sceller ce résultat négatif et poursuivre des
expériences indépendantes, sans lui attribuer de gain accepté ni autoriser une
combinaison qui en dépend. Les défauts de ressources, sources, artefacts,
schéma, argmax ou valeurs non finies restent des arrêts fatals.

## Campagne indépendante de Codex

`benchmarks/m5_autonomous_supervisor.py` vérifie les dépendances, fichiers,
identités et verrous, puis attend au plus **30 minutes** une pression 1 et
**4 Gio disponibles pendant cinq secondes**. Il lance une seule campagne,
avec un budget de fonctionnement de **8 heures**. La réserve système de 3 Gio,
le plafond RSS de 2 Gio, la croissance maximale du swap de 128 Mio et les
autres gardes héritées restent inchangés. Une réserve disque supplémentaire
de **5 Gio** est contrôlée avant et pendant les mesures.

Le superviseur est conçu pour un service utilisateur launchd, indépendant de
Codex et de ses workers. Les journaux et identités sont persistants. Il ne ferme
aucune application ; il peut seulement arrêter ses propres descendants si une
garde est violée. Chaque expérience complète est scellée par les hashes du
checkpoint avant/après ; une reprise conserve les expériences déjà scellées.

Les 24 comparaisons H1/H2/H3 et leurs 144 workers sont séquentiels. Les gains,
la fidélité réelle, les exclusions et l'état final seront lus dans les preuves
du superviseur et de la campagne ; « démarré » ne signifie pas « terminé ».
Codex peut rester ouvert : le nettoyage des auxiliaires a déjà rendu l'admission
possible, sans risquer le travail non enregistré de l'application principale.

Le premier service launchd a rencontré un démarrage retardé puis une erreur de
chemin d'origine non canonique ; sa propre inscription a été retirée pour
éviter des redémarrages inutiles. Une seconde méthode a réellement lancé et
terminé les deux workers H1/H2 : `nohup`, nouvelle session, entrée `/dev/null`
et journaux sur fichiers. Le contrôle indépendant est réexécuté avec des
chemins absolus et une attente de récupération bornée à **60 secondes entre
workers**. Les preuves distinguent ce détachement OS d'une démonstration de
survie après fermeture réelle de Codex, qui n'a pas été nécessaire ici.
