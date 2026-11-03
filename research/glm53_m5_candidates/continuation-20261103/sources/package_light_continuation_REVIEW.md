# Livraison légère de continuation — source seulement

`package_light_continuation.py` est préparé, non importé, non testé et non exécuté par son auteur. Aucun fichier n’a été copié, compressé ou publié. Le parent reste responsable de l’approbation du périmètre final et de la publication GitHub par connecteur.

## Périmètre et résultats futurs

Entrée de base : `work/continuation-selection-compact.json`, uniquement ses `allowed_entries` explicites. Le chemin public ancien proposé dans cet inventaire est ignoré : le seul namespace permis est **`work/publication/research/glm53_m5_candidates/continuation-20261103`**, neuf et créé exclusivement. Aucun `frozen-work`, index ou archive de `f0acff6` n’est remplacé. Aucun checkout natif complet n’est parcouru ; pas de `rglob` de découverte.

- Code/docs/patchs (`.py/.cpp/.h/.inc/.mm/.metal/.patch/.md/.cmake`) et configs/manifests sources nommés dans les constantes sont copiés **sans modifier les bytes** sous `sources/<relatif-work>`.
- Reçus, logs et autres métadonnées explicitement sélectionnés sont conservés sans perte dans `evidence-continuation.zip` : un membre gzip par original, lui-même stocké sans double compression ZIP.
- `index.json` contient chemin original absolu et relatif, taille, SHA original, SHA des copies, chemins/membres gzip, tailles/SHA gzip et SHA de l’archive de preuves. L’identité de fichier du gel est conservée. Les références de reconstruction déjà publiques restent explicites.
- Le ZIP utilisateur futur `outputs/Sources-et-preuves-continuation-M5.zip` contient exactement ce namespace et les seuls documents finaux `outputs/` explicitement ajoutés. Ces derniers **ne vont pas dans la publication GitHub**. Un manifest extérieur donne le SHA du ZIP et les fichiers du namespace.

Les sources et reçus historiques restent des instantanés exacts, y compris leurs anciens chemins/états `source_only`, refus ou échecs. Ils ne sont ni réécrits pour paraître portables, ni requalifiés. La sélection finale révision 10 annonce 316 fichiers et de vrais SHA déjà gelés : ils doivent tous correspondre. Si un ancien inventaire contenait encore des SHA `null`, ils ne pourraient être calculés pendant un futur gel quiescent que sur acceptation explicite du parent ; cela identifierait les bytes exportés, **pas une provenance de build mesuré reconstituée rétroactivement**. L’option d’acceptation différée reste inutile et doit rester fausse pour la sélection finale entièrement épinglée.

## Liste finale explicite

Le gabarit est `work/extra_allowlist_final.example.json`. Le parent renseigne puis sauve **`work/extra_allowlist_final.json`** ; ce fichier n’est pas créé ici pour ne pas chevaucher la sélection finale du parent/Cam. Il épingle le SHA du JSON de base, atteste que les entrées sont fermées/gelées, et comporte :

1. `entries` : au plus 64 ajouts, avec `root` (`work` ou `outputs`), `relative_path`, `bytes`, vrai `sha256`, `delivery` (`source`, `evidence`, `output_only`), `role`, `closed_artifact:true`. Ajouter par leur **nom exact uniquement les fichiers absents de la base**. V5, binding réel, métadonnées finales de capture et audit C16 v2 sont déjà dans les 316 entrées de la révision 10 ; les extras attendus sont surtout ce packager, sa documentation et les rapports/métriques finaux `outputs`. Doublons et remplacements implicites d’entrées de base refusent ; un artefact modifié doit être versionné explicitement ou l’inventaire de base revu et repinné.
2. `closure_checks` : au moins un vrai reçu JSON clos, taille/SHA et conditions de champs exactes (chemins pointés, par exemple `cleanup.owned_server_stopped`). Les statuts admis sont completed/failed/cancelled/blocked ou le résumé C16 `functional_rejected_REQUEST_TIMEOUT`. Le parent peut et devrait épingler les reçus NoCopy et C16 distincts. Aucune sortie de code 0 n’est transformée en preuve d’EOS.
3. `approved_for_delivery:true`, `all_base_entries_frozen:true`, et, seulement si nécessaire, `accept_deferred_base_hashes:true`. Sans cette approbation et les pins réels, `--execute` refuse. L’invocation sans `--execute` ne fait qu’un plan et lit les deux JSON de sélection ; elle n’inspecte/hache aucun source sélectionné.

Les noms de rapports `outputs` admis sont fixés dans `OUTPUT_NAMES`, sans motif large. Un autre nom demande un changement source relu. Les configurations/manifests JSON nécessaires en clair sont fixés dans `SOURCE_JSON_NAMES`/`SOURCE_JSON_PATHS` ; les reçus JSON restants sont gzip. Une source qui dépend d’un chemin natif local reste un instantané de recherche, pas une installation autonome.

## Bornes et garde de quiescence

Avant tout hash de source, copie ou compression, le chemin exécuté acquiert **les mêmes verrous exclusifs non bloquants benchmark et UID/modèle**, avec contrôle de l’inode du descripteur. Le modèle ciblé reste GLM original révision `119034bc…`. Le chemin UID réel est épinglé par le parent dans la liste finale ; le fichier doit déjà exister, appartenir à l’UID et avoir la clé canonique attendue. Un moteur propriétaire empêche le packaging. Ces verrous ne prouvent pas l’absence d’une activité étrangère qui ne suit pas ce protocole ; le coordinateur doit confirmer la fenêtre matérielle quiescente.

Admission : réserve disque **32 Gio + budget d’écriture 32 Mio**. Le budget compte les extensions réelles de tous les fichiers générés, avant chaque écriture, y compris ZIP et manifests ; les écritures correctives d’en-têtes ZIP n’augmentent pas artificiellement leur taille comptée. Réserve 32 Gio contrôlée pendant et à la fin. Le namespace GitHub doit rester **strictement inférieur à 7 000 000 octets**. Limites d’entrée : 384 fichiers, 32 Mio par fichier, 96 Mio cumulés ; elles servent uniquement à permettre de gzip un garde JSON clos volumineux, pas à inclure du raw.

Les originaux sont ouverts sans suivre un lien symbolique, puis leur descripteur est comparé à l’identité gelée avant lecture. Taille/SHA et identité sont vérifiés à la copie, puis toutes les identités/pins de sélection sont recontrôlés avant commit. Les deux ZIP sont relus en streaming : ensembles complets et uniques, SHA des membres gzip, décompression bornée à la taille originale, SHA des bytes originaux et SHA de chaque membre du ZIP utilisateur. Les sorties sont d’abord dans un répertoire temporaire privé de `work/`. Le commit utilise créations/liens exclusifs, jamais un remplacement d’un fichier existant ; son rollback ne supprime que les inodes créés par cet appel. Aucun appel Git/GitHub n’existe dans le script.

Fichiers privés, DB/SQLite, poids, `.bin/.f32`, exécutables, bibliothèques, archives préexistantes et chemins de checkout/personnal-memory refusent. Texte UTF-8 et absence de NUL sont exigés ; certains marqueurs manifestes de clés privées/jetons entraînent un refus **sans redaction**. Ce contrôle ne remplace pas la revue humaine de l’allowlist et ne prétend pas détecter tout secret imaginable. Un chemin `--api-key-file` dans un argv historique n’est pas la valeur du secret ; le fichier de credential n’est jamais sélectionné ou ouvert.

## Disponibilité des preuves

Pas de matrices/tensors/logits/états bruts publics dans ce package. L’archive CPU8 de 1 585 638 687 B reste un livrable **local** déjà vérifié, non hébergé sur GitHub. Les grands captures v3/v4/NoCopy restent locaux et leurs gaps publics sont indexés. Les tableaux Q8-tail n’ont jamais été exportés : un reçu/hash n’invente pas leur disponibilité. **Aucun whole-raw archive d’environ 6 Go n’est créé ou promis.** La déduplication de fichiers n’accorde aucune qualification native.

Avant toute exécution, revue indépendante demandée à Cam. Les tests archivés dépendant d’anciens chemins ne doivent pas être lancés automatiquement ; aucune suite, build, modèle, GPU, compression ou copie n’est nécessaire pour cette revue source.
