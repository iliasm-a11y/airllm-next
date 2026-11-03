# Comparateur de rejet de la fenêtre 128

Source préparée pendant le modèle CPU v4. Aucun import de ce nouveau comparateur,
test, AST, build, appel natif, GPU ou comparaison de payloads n'a été exécuté.
Les comparateurs complets existants restent inchangés et refusent toujours
`complete=false`.

`probe_compare.py` possède deux chemins futurs distincts : création d'un binding
à partir des **deux vrais builds**, puis comparaison des deux captures closes.
Aucun binding avec SHA nul, driver planifié ou simple feature/profile importé
n'est admissible. Le témoin est le helper CPU8 mesuré et son reçu exact ; le
candidat doit être le nouveau driver réellement compilé, lié aux mathlibs NoCopy
existantes, à leur reçu provider exact et au nouveau profil GPU complet. Les
sources des helpers, wrappers, headers/dépendances, bibliothèques, politiques et
modules délégués sont liées par empreintes, contrôlées sur disque et recontrôlées.
Un module en cache provenant d'un autre chemin est refusé.

Les 600 agents originaux sont imposés par le SHA du fichier JSON gelé
`f2b40d0c9f94a507ccaaff5e131fb57e698585ba640c0614fe2e69c12bd11fef`, les IDs
entiers uniques 1 à 600 et les listes complètes de tokens. Le lanceur hache et
parse les mêmes octets bornés en une seule lecture. Aucun champ `complete` n'est
inventé pour cette entrée : l'original n'en possède pas. Le comparateur exige
que le fichier transmis au natif soit exactement ce document plus la propre clé
de l'arm et la borne de scratch, et vérifie le SHA des IDs complets de request0.

Chaque clé est d'abord recomputée depuis **son propre** manifeste effectif.
L'égalité des modèles/tokenizer/template/autres paramètres n'est examinée
qu'après la validation indépendante des deux builds, sources, paramètres réels
du smoke et profils. La compatibilité locale retire uniquement les différences
explicitement validées : batch, politique v4/noBLAS, profil GPU et fenêtre de
diagnostic. Les sources/binaires sont remplacés par un marqueur local après
vérification séparée. Aucun reçu ni clé observée n'est modifié ou substitué.

Le témoin CPU8 doit posséder son garde clos original sans erreur. Le candidat
doit posséder `probe_only`, `complete=false`, retour 0, aucune violation et un
checkpoint explicite à 128. `probe_guard.py` est un validateur séparé pour cet
état partiel : il reprend les vérifications de ressources approuvées du
comparateur natif (seuils, finitude, baseline, réserve, pression 1/2 contextuelle,
swap et replay) et n'accepte que le statut partiel explicite. Il n'appelle aucun
comparateur complet avec un faux statut `completed`. Les preuves AC/horloge,
sortie naturelle, identités de fichiers et, pour le candidat, FD UID observé
avant GO/fermeture du natif sont requises. Le témoin historique ne possédait pas
le champ supplémentaire `runner_source_identity_unchanged` : son garde original,
`integrity_passed`, l'identité du wrapper suivi et son SHA exact sont conservés.

Le candidat doit aussi fournir la trace brute actuelle du provider : SHA
canonique/kernel, vraies commandes du N demandé, appariement soumission/fin/
publication, alias NoCopy vérifié, zéro copie de poids, budgets et gardes. Le
simple import d'un profil ou `actual_gpu=true` ne suffit pas. Toute exécution de
request1 ou au-delà du token127 dans les reçus candidat est refusée ; les fichiers
de continuation doivent être vides, et les logits artificiels absents.

Les 118 noms GLM originaux sont couverts exactement sur 0 à 127. Chaque ligne
doit correspondre à son lot réel, sans trou/chevauchement/duplication, avec forme,
type, axes et strides corrects. Le SHA et la taille des payloads sont vérifiés
avant une vue `tensor_view` respectant les strides ; seuls les éléments logiques
sont comparés. Les padding bytes ne deviennent pas des éléments sémantiques.
Le comparateur strict existant de séries est utilisé pour les bits CPU, sans son
lecteur complet. Les tolérances Metal originelles (atol 1e-3, rtol 1e-4, routes
entières exactes) sont calculées séparément comme diagnostic auxiliaire. Elles
ne convertissent pas un écart bit à bit ou un état opaque différent en réussite.
Le checkpoint128 est un hash opaque natif explicitement nommé, sans prétention
de fichier d'état final du prompt.

Résultats possibles : `invalid_probe`, `rejected_probe_window`, ou
`identical_probe_window`. Même le dernier garde `complete=false` et toutes les
qualifications complètes/modèle/snapshot/EOS/performance à false. Il s'agit d'une
barrière avant un éventuel vrai prompt complet, jamais d'un benchmark du prompt
521 ni d'un objectif A déclaré atteint.

Après revue, build réel et signal root uniquement, les formes de commande seront :

```text
PYTHON probe_compare.py --make-binding --output build-run1/probe-two-build-binding.json
PYTHON probe_compare.py --reference CPU8_CAPTURE --reference-guard CPU8_GUARD --candidate GPU128_CAPTURE --candidate-guard GPU128_GUARD --binding build-run1/probe-two-build-binding.json --output probe128-compare.json
```

Les chemins réels seront absolus dans les reçus. Le fichier de préparation
`probe-compare-prepare.json` ferme les sources de comparaison, pas une mesure.
Le plan de driver antérieur aux corrections de revue est marqué obsolète et ne
constitue aucun reçu de build ; le futur `build_driver.py` doit régénérer un plan
neuf et des smokes réels. Les 13 tests portables antérieurs ne sont pas présentés
comme une validation des nouveaux correctifs ; les nouveaux tests restent en
source, non exécutés pendant CPU v4.

Correctif de contre-revue : les checkpoints du candidat doivent couvrir exactement
la seule ligne `(request=0, token_end=128)`. Aucune ligne hors fenêtre n'est
ignorée. Les indices, positions, flags et tailles exigent des entiers exacts,
et le SHA exige 64 caractères hexadécimaux minuscules. La lecture exige aussi un
JSONL clos, terminé par un saut de ligne. Le témoin historique complet garde son
filtre séparé vers le checkpoint128 de request0. Ce correctif reste source
seulement, sans test ni AST exécuté pendant C16.
