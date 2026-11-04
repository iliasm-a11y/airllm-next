# RONIN GLM natif

Cette intégration utilise exclusivement les deux shards GLM-5.3-Flash Q2_K_S
épinglés à `119034bc3d9d76eb41f91508e6f9fb19fde42672`.
Elle remplace la frontière de moteur du candidat RONIN du 28 octobre 2026
par `NativeGlmModel`. Le moteur natif reste partagé entre les sessions.

Le runtime RONIN prépare les sources autorisées puis appelle réellement GLM.
La sortie est enregistrée comme `model_output_unverified` dans les preuves
SQLite. Elle ne peut pas exécuter des outils ni étendre le périmètre autorisé.
Le multimodal et les appels d’outils ne sont pas déclarés pris en charge.
L’effort de raisonnement peut être choisi explicitement par requête comme
conditionnement du template, selon le contrat détaillé ci-dessous.

## Configuration

Créer un fichier JSON, avec des chemins absolus existants :

```json
{
  "model_path": "/chemin/GLM-5.3-Flash-Q2_K_S",
  "runtime_binary": "/chemin/runtime/bin/llama-server",
  "runtime_manifest_file": "/chemin/runtime/manifest.json",
  "verification_file": "/chemin/verification-glm.json",
  "context_tokens": 512,
  "threads": 4,
  "batch_size": 8,
  "gpu_layers": 0
}
```

La configuration optimale mesurée et ses limites doivent être prises dans
les preuves GLM de cette branche, sans extrapoler les résultats Qwen de la PR14.
`verification_file` permet de réutiliser des hashes vérifiés lorsque
l’identité des fichiers est inchangée. Sinon le backend recalcule les hashes.
Le backend conserve les gardes de mémoire, pression, swap et durée.

Le manifeste local du constructeur doit accompagner le runtime distribué.
Il est vérifié avant lancement : révision source épinglée, état de construction,
SHA-256 de l’exécutable et de toutes les bibliothèques natives du dossier `bin/`,
et liens symboliques limités aux fichiers déclarés de ce même dossier.
Les identités des fichiers sont ensuite contrôlées avant les requêtes ; un
changement arrête le moteur. Les variables `DYLD_*` héritées sont supprimées
pour conserver les bibliothèques empaquetées. Les affirmations de provenance
source restent celles du manifeste de construction local, sans attestation distante.

Le chemin historique sans `runtime_manifest_file` reste disponible pour les
comparaisons explicites, mais son statut indique `runtime_artifacts_verified=false` :
le hash du petit exécutable seul ne qualifie pas ses dépendances mathématiques.

L’admission mémoire distingue le démarrage d’un moteur et une requête sur un
moteur déjà chargé. Le démarrage refuse toujours une pression macOS de niveau 2.
Une requête sur le moteur chargé peut être admise au niveau 2 si la mémoire
disponible couvre encore la réserve et le budget de travail, si le swap reste
sous les seuils et si aucune garde n’a arrêté le moteur. Cette correction vient
des mesures du même GLM sur ce M5 : pendant la référence CPU 4 threads,
contexte 512, batch 8, le niveau 2 a duré 230 secondes avec plus de 3 Go disponibles,
un pic de mémoire physique attribuée de 887 584 792 octets et une croissance du
swap d’environ 2,95 Mo. Le RSS de 6,41 Go inclut les pages de poids mappées,
réclamables par le système. Ces observations ne qualifient pas les autres charges.

La garde arrête immédiatement le moteur au niveau critique 4, au-delà de
9 Gio de RSS ou de 64 Mio de croissance du swap. Au niveau 2 pendant au moins
20 secondes, elle arrête aussi si la mémoire disponible descend sous
la réserve de 768 Mio plus 512 Mio, ou si le swap croît de plus de 32 Mio.
Une mémoire disponible sous la réserve seule provoque l’arrêt après 3 secondes,
même sans pression de niveau 2. Le statut `pressure_context` expose les valeurs,
la durée du signal, les seuils et la décision ; poursuivre sous surveillance
ne constitue pas une garantie générale de stabilité.

La configuration retenue utilise le CPU. L’essai hybride Metal à six couches,
experts CPU et cache MoE de 256 Mio a échoué aux critères de fidélité sous les
mêmes préfixes forcés : écart absolu maximal de logits de 0,5348455905914307 et
364 éléments de routes ordonnées différents sur les 3 024 comparés. Les quatre
tokens émis restent identiques dans cet essai ; cela ne suffit pas à qualifier
le GPU. Ces écarts concernent cette comparaison à la référence CPU et ne sont
pas une conclusion générale sur la précision de tous les chemins Metal.

Tout `gpu_layers>0` est donc refusé par défaut avant lancement, avec le code
`GPU_FIDELITY_UNQUALIFIED`. Une expérience volontaire peut préciser
`"allow_unqualified_gpu": true` dans la configuration JSON ; cette autorisation
ne retire ni la vérification du périphérique ni les gardes mémoire. Aucun
chemin GPU n’est déclaré qualifié. Le statut `numerical_qualification` rappelle
le périmètre CPU ciblé de quatre tokens, 619 520 logits et 3 024 éléments de routes,
sans promettre d’équivalence générale pour toutes les séquences et configurations.

Les expériences natives restent désactivées par défaut. Les options JSON
`mmap_demand`, `expert_prefetch` et `dense_prefetch` n’acceptent que des booléens.
Les équivalents d’environnement sont `RONIN_GLM_MMAP_DEMAND=1`,
`RONIN_GLM_EXPERT_PREFETCH=1` et `RONIN_GLM_DENSE_PREFETCH=1`.
Le premier supprime le conseil initial de préchargement de tout le fichier ;
les deux autres demandent les plages des experts Q2_K sélectionnés ou des
matrices denses Q8_0 lors du décodage CPU à un token. Les choix du routeur et
les opérations numériques restent inchangés. Ces options exigent le runtime
privé patché et une validation comparative ; leur présence ne prouve aucun gain.
Les variables `AIRLLM_*` héritées sont nettoyées avant le lancement, puis seules
les options explicites choisies sont transmises au processus natif.

Pour reconstruire le runtime et le diagnostic à partir du commit épinglé,
sans modifier le checkout source existant :

```sh
python3 research/glm53_m5/build_runtime.py --source CHECKOUT_LLAMA_EXISTANT --destination RUNTIME_VIDE --jobs 3
```

Sans `--source`, le script clone le dépôt officiel dans un dossier temporaire.
Il applique les quatre patches privés, compile Metal et le diagnostic avec les
mêmes en-têtes et bibliothèques, puis produit `bin/`, `manifest.json` et
`build.log`. Aucun modèle n’est chargé par cette commande.

Le quatrième patch fournit un plafond expérimental de colonnes denses pour les
pilotes natifs : `AIRLLM_DENSE_PREFETCH_MAX_COLS` accepte uniquement `2`, `4`
ou `8`, et conserve la valeur 1 dans les autres cas. Cette limite n’active pas
le préchargement à elle seule. Le backend RONIN conserve son unique slot et
nettoie cette variable héritée ; aucune option de production n’a été ajoutée.

Depuis la racine du dépôt :

```sh
python3 integrations/ronin/ronin_glm.py status --config CONFIG.json --home DONNEES_ISOLEES
python3 integrations/ronin/ronin_glm.py generate --config CONFIG.json --home DONNEES_ISOLEES --prompt 'La capitale de la France est' --max-tokens 4 --output generation-ronin.json
```

Le chemin d’exécution est `Runtime.dispatch -> Engine.stream ->
NativeGlmModel.stream_chat`. Le template du GGUF est appliqué par le runtime
natif. Les événements, tokens et timings natifs sont conservés. Cette commande
n’est donc pas directement comparable à une complétion native sans template.

L’option facultative `--reasoning-effort low|high|max` transmet exactement cette
valeur à `chat_template_kwargs.reasoning_effort` du moteur natif. Si elle est
omise, aucun paramètre de template n’est ajouté : le défaut du template GGUF,
actuellement `Max`, reste inchangé. Aucun défaut global n’est remplacé par `low`.
Exemple de choix volontaire pour une requête :

```sh
python3 integrations/ronin/ronin_glm.py generate --config CONFIG.json --home DONNEES_ISOLEES \
  --prompt 'Réponds uniquement : OK' --max-tokens 128 --reasoning-effort low --output generation-low.json
```

Le statut expose `reasoning_effort_scope=chat_template_conditioning_only`.
Cette option peut modifier la quantité et le contenu du raisonnement ; elle ne
prouve aucune équivalence de qualité entre efforts. Une réponse plus courte ne
constitue pas une accélération du débit de décodage. Les comparaisons de vitesse
doivent conserver le même effort et rapporter les tokens réellement produits.
Les valeurs et arguments inconnus sont rejetés avant chargement. Les tokens EOS,
les poids, les kernels et les gardes ne sont pas modifiés par cette option.

## Application macOS séparée

```sh
python3 integrations/ronin/build_app.py --destination LIVRAISON --config CONFIG.json
open 'LIVRAISON/RONIN GLM.app'
```

Pour embarquer uniquement le moniteur `psutil` utilisé pendant les benchmarks,
préciser son dossier de package installé et un Python externe compatible :

```sh
python3 integrations/ronin/build_app.py --destination LIVRAISON --config CONFIG.json \
  --python /chemin/venv/bin/python \
  --monitor-package /chemin/venv/lib/python3.12/site-packages/psutil \
  --embed-runtime
```

Le constructeur conserve le chemin absolu du lanceur Python, y compris son
lien symbolique de venv ; il ne le remplace pas par le chemin de l’interpréteur
de base. Python reste une dépendance externe et doit rester disponible à ce
chemin après la livraison. Le package `psutil`, son extension macOS et sa licence
disponible dans `psutil-VERSION.dist-info/LICENSE` sont les seuls ajouts de
dépendance. Le fichier `monitor-manifest.json` consigne la version, la provenance,
les tags de wheel disponibles, les hashes des sources et ceux des fichiers
embarqués après signature. Le wheel doit être compatible avec le Python externe
et l’architecture macOS retenus ; le constructeur n’exécute pas son extension.

Sans `--monitor-package`, aucune dépendance n’est copiée. Le backend conserve
son choix existant : `psutil` s’il est disponible dans le Python choisi, sinon
le moniteur de repli utilisant les outils système. Ce choix est explicite dans
le manifeste. Embarquer le moniteur des benchmarks réduit les appels externes
de mesure ; cela ne prouve pas que les arrêts mémoire précédents venaient du
moniteur. Les seuils, l’admission et les décisions des gardes sont inchangés.

Avec `--embed-runtime`, le constructeur copie le dossier `bin/` du runtime,
ses liens symboliques locaux, le manifeste et les licences dans
`Resources/native-runtime`. Il copie seulement le petit reçu JSON de
vérification des poids vers `Resources/model-verification.json` ; les deux
GGUF restent dans leur emplacement externe. Les trois chemins correspondants
de `runtime-config.json` sont relatifs au dossier de ce fichier. Le runtime
RONIN résout uniquement ses options de chemins connues depuis ce dossier ;
il ne dépend donc pas du dossier courant du processus pour trouver ces fichiers.
En l’absence de `state_dir`, une configuration utilise `DataHome/engine-state`.
Le mode embarqué retire l’ancien `state_dir` de la copie de configuration.

Après signature des fichiers natifs, le constructeur recalcule les hashes et
tailles du manifeste embarqué et le hash attendu de l’exécutable dans la
configuration. Les références d’origine sont conservées dans
`pre_bundle_artifacts` et `bundle_packaging`. Il met ensuite à jour le manifeste
du moniteur et signe l’enveloppe de l’application, sans modifier à nouveau les
fichiers natifs. Le manifeste de construction source reste intact.
Avant toute copie et signature, il vérifie les tailles et SHA de tous les
fichiers source, le hash d’exécutable attendu s’il est fourni, la liste exacte
des artefacts et les liens confinés au dossier `bin/`. Il vérifie ensuite la
copie contre ce même manifeste original. Le rafraîchissement post-signature
est réservé à ce flux ayant vérifié des entrées intactes ; il ne sert pas à
accepter un moteur altéré. Sans preuve de vérification source et copie dans
le manifeste de packaging, le helper de rafraîchissement refuse de réécrire
les hashes. La confiance dans la provenance du manifeste source reste locale.

Les helpers `bundle_runtime(config, resources, config_directory)` et
`refresh_runtime_manifest(resources)` sont également appelables sans compiler
l’application. Pour une copie manuelle : copier, signer avec `--deep`, rafraîchir
les manifestes du runtime et du moniteur, puis signer uniquement l’enveloppe
sans `--deep` et vérifier avec `--deep --strict`. Les données de recherche,
Python et les GGUF restent externes. L’application peut être déplacée sur le
même Mac tant que ces dépendances externes restent accessibles.

Le constructeur crée une application distincte `local.ronin.glm.prototype`,
signée ad hoc. Il ne remplace pas l’ancienne application RONIN. Le dossier
de données par défaut est `~/Library/Application Support/RONIN GLM`.
L’application dépend du Python indiqué au constructeur et, sans
`--embed-runtime`, du binaire natif indiqué dans sa configuration ; ils ne sont
pas téléchargés ni substitués.
Le catalogue communautaire historique n’est pas embarqué par ce constructeur.

Les réglages permettent de sélectionner le dossier, de charger et décharger
GLM. Une nouvelle mission lance la préparation puis la génération. Les
contrôles de pause et d’arrêt transmettent l’annulation au backend natif.
Cette génération reçoit actuellement le texte de la mission. Les sources
préparées et les preuves restent enregistrées dans RONIN, mais ne sont pas
encore synthétisées automatiquement dans le contexte envoyé au modèle.
Cette livraison vérifie donc le trajet d’inférence RONIN et son contrôle ;
elle ne constitue pas une investigation autonome complète fondée sur les sources.
Un appel réel du backend RONIN a émis 128 tokens avant d’atteindre sa limite
pendant le raisonnement, sans réponse finale ni EOS. Un essai long avec
le runtime entièrement embarqué a été arrêté par `SWAP_LIMIT`. Ensuite, deux
appels courts de quatre tokens chacun ont réussi avec le même moteur réutilisé,
en atteignant leur limite de tokens. Un appel lancé depuis l’interface graphique
a émis 32 tokens en 321,881 secondes et atteint la limite pendant le raisonnement.
Le rafraîchissement périodique inclut maintenant l’état de génération pour que
l’interface récupère son état final. Une génération complète jusqu’à EOS avec
cette livraison reste non validée. Les lanceurs
utilisateur conservent l’effort Max par défaut et les gardes existantes.

Les 600 identités sont des métadonnées persistantes ; le backend utilise une
file FIFO bornée et une seule inférence physique. Il peut suspendre les
admissions ou arrêter son propre processus lors d’une pression mémoire.
Il n’augmente pas la concurrence sans qualification matérielle.
Le verrou d’instance est commun à cette cible épinglée pour le même utilisateur,
indépendamment du chemin du modèle ou du dossier de travail. Il est transmis
au processus natif par un descripteur hérité.
Si le parent Python est arrêté brutalement par SIGKILL, le processus natif peut
survivre ; il conserve alors ce verrou et empêche le lancement d’un deuxième
moteur sur le même modèle. La fermeture normale et SIGTERM passent par le
nettoyage explicite. Cette livraison ne termine pas automatiquement un processus
orphelin inconnu et ne prétend pas couvrir une panne système.

L’admission Metal suit le placement exact du runtime épinglé : `gpu_layers=1`
place seulement la tête de sortie et la normalisation sur GPU (674 054 144 octets) ;
`gpu_layers=6` ajoute les cinq derniers blocs et totalise 1 605 374 848 octets
hors banques d’experts CPU. Le GGUF cible ne comporte pas de couche MTP.
Ces budgets ne remplacent pas la surveillance des allocations de travail.

Le compteur `requests_in_flight` inclut les requêtes en attente ; le compteur
`physical_inferences_active` provient du slot natif et reste compris entre 0 et 1.

## Vérification portable

L'[intégration source des accès denses et du préfill](../../research/glm53_gpu_v3/INTEGRATION-DENSE-PREFILL-FR.md)
ajoute des réglages explicites et un prototype natif de préfixe partagé.
Les options d'accélération restent désactivées par défaut et non qualifiées.
Cette intégration ne reconstruit ni ne remplace le runtime installé.
Les tests ci-dessous n'ont pas été exécutés pour cette intégration pendant
le travail de l'agent actif.

```sh
python3 -m unittest discover -s integrations/ronin/tests -v
```

Ces tests utilisent de faux événements pour vérifier le contrôle de flux,
les 600 sessions, l’isolation, l’annulation, les tokens et l’enregistrement des
preuves. Ils ne constituent pas une preuve d’inférence, de parité ou de débit.
Les générations réelles doivent être archivées séparément.
