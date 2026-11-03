# CPU → Metal Q8 par opération : candidat source, non compilé et non exécuté

Ce patch propose un provider CPU `extra_compute_forward/work_size` sur les
sources V4 exactes. Il ne place aucun poids de modèle de façon permanente sur
Metal : chaque produit conserve sa matrice CPU originale et copie une tuile
de lignes dans un espace Metal fixe. Ce n'est pas encore un moteur qualifié.
Le noyau Q8 seul a été mesuré auparavant ; le provider, ses barrières, sa
copie des poids et ses erreurs n'ont pas été exécutés.

`source/` est un overlay de sept fichiers. Le patch
`cpu-metal-q8-bridge.patch` est destiné exclusivement à la reconstruction
`work/metal-integration-v4-source/source`. Son reçu identifie les 16 patches V4
et les 13 sources concordantes. Aucun checkout ou exécutable mesuré n'a été
modifié et aucun build n'a été lancé pendant CPU16.

## Raccordement et valeur par défaut

Le CMake ajoute `GGML_CPU_METAL_Q8_BRIDGE`, OFF par défaut. Il ne compile
l'Objective-C++ que si l'option est explicitement activée sur Apple arm64,
pour une seule variante CPU. La bibliothèque annonce alors la feature
`CPU_METAL_Q8_BRIDGE_COMPILED=1` par l'API de features existante.

Même compilé, le provider est absent de la liste des extras tant que
`AIRLLM_CPU_METAL_Q8_BRIDGE=1` n'est pas défini au démarrage. La configuration
est figée avant la planification du graphe. Un SHA256 de profil GPU explicite
est obligatoire dans `AIRLLM_CPU_METAL_Q8_CANONICAL_SHA256` ; les valeurs de
flag inconnues et les clés absentes sont fatales.

La registration est un marker hook-only : elle réutilise les interfaces de
buffer CPU et interdit sa sélection comme nouveau format de poids. Les
poids éligibles restent exactement dans le buffer CPU original, standard et
marqué WEIGHTS, y compris CPU_Mapped. Le helper additif
`ggml_backend_cpu_buffer_has_plain_layout` reconnaît les deux identités de
type natives, sans admettre un nom de buffer usurpé, repack ou Metal. Il exige
de reconstruire ensemble la bibliothèque de base et la bibliothèque CPU.
Ni `ggml_cplan`, ni `ggml_compute_params`, ni leurs layouts ABI ne changent.
La registration C++ est interne à la version épinglée.

Les hooks réels `ggml_cpu_extra_compute_forward` (`ggml-cpu.c:1955`) et
`ggml_cpu_extra_work_size` (`:3062`) précèdent le mul_mat générique et ses deux
LLAMAFILE. Un refus avant soumission retourne la même décision à tous les
workers, qui reprennent alors le calcul CPU original. Q2_K, MUL_MAT_ID,
KDA, mHC et les opérations d'attention autres que les projections Q8 restent
sur leur chemin CPU.

## Domaine et workspace

Le domaine expérimental est Q8_0/F32 → F32, sources et sortie contiguës 2D,
K2048/4096, M multiple de 8 dans [8,8192], N8/16/32/64/128. Les queues N6/14,
N1 et les autres formes restent CPU. Les buffers non host, poids repackés,
vues de poids, sorties vues et entrées non finies sont refusés avant soumission.
Les échelles FP16 subnormales non nulles restent hors du domaine du provider.
Les grandes formes admises du provider ne sont pas toutes couvertes par la
fixture représentative déjà mesurée ; elles exigent des essais dédiés avant
la moindre qualification du graphe GLM.

Les workers quantifient par blocs avec `ggml_get_type_traits_cpu(Q8_0)->from_float`,
comme la référence CPU. Le workspace du graphe contient 64 octets de contrôle,
les activations Q8_0 natives et la sortie complète de l'opération en staging.
La décision est copiée en variable locale puis protégée par une barrière avant
tout retour CPU : le mul_mat générique ne peut pas écraser un contrôle encore
lu dans `wdata`.

Le coordinateur ith0 utilise un mutex de provider et quatre MTLBuffer Shared
alloués une fois puis réutilisés. Aucun poids FP32 complet ni cache croissant :

| Buffer fixe Metal | Octets |
|---|---:|
| Tuile de poids, au plus 1536×4096 Q8_0 | 6684672 |
| Activations int8, au plus 128×4096 | 524288 |
| Échelles FP16 natives | 32768 |
| Sortie d'une tuile | 786432 |
| **Total fixe** | **8028160** |

À M8192/K4096/N128, les activations CPU natives valent 557056 octets et le
staging de sortie 4194304 octets ; le besoin logique du provider est
4751424 octets. Il refuse une exécution si le workspace CPU réellement
fourni au graphe, ajouté aux buffers Metal, dépasse 64 Mio. Les traces
distinguent besoin logique du provider, workspace CPU du graphe, payload
Metal retenu, allocations, réutilisations et commandes soumises. Ce plafond
n'est pas un plafond RSS : bibliothèques, threadpool, pilote et compilation
Metal sont supplémentaires et restent sous le watchdog existant.

La routine fait une copie de poids par tuile et opération ; elle ne prétend
pas éviter toutes les relectures entre lots. Les compteurs de copie sont
des octets mémoire vers les buffers partagés, pas des lectures SSD physiques.
Le coût de copie, quantification, attente et soumission doit être mesuré dans
le futur témoin ; les timestamps GPU seuls n'établissent aucun gain.

## Commandes, erreurs et publication

Chaque tuile réutilise le noyau `q8_batch_dot` exactement mesuré. Le header
embarqué conserve le texte complet et son SHA ; le provider remplit ses
int8/échelles depuis les blocs quantifiés CPU sans nouvelle quantification GPU.
Les branches et les barrières du noyau ne sont pas réécrites. Chaque commande
utilise une sortie Metal détenue par le provider ; le graphe CPU n'est jamais
une destination GPU.

Le handler de completion signale la fin terminale. Un timeout de 60 secondes,
une erreur de commande, un timestamp invalide, une sortie non finie ou une
admission refusée après soumission sont fatals avec sortie non publiée. Il
n'existe aucun retour false/fallback après une soumission en cours. Les
ressources ne sont ni recyclées ni libérées avant fin terminale ; un timeout
termine le processus au lieu de laisser CPU et GPU écrire simultanément.

Les sorties de toutes les tuiles sont d'abord copiées dans le staging CPU.
`dst` n'est publié qu'une fois, après succès terminal de l'opération entière,
puis tous les workers franchissent les barrières de fin. Les gardes locales
secteur, RSS8,5 Gio et swap32 Mio complètent le watchdog natif existant.
La garde locale admet un sous-ensemble conservateur suffisant de la politique
contextuelle originale : pression1 avec disponible ≥768+128 Mio ; pression2
avec disponible ≥768+128+512 Mio et croissance swap ≤32 Mio. Les autres
niveaux sont refusés. Les grâces natives3s/20s ne sont pas réinterprétées :
ces bornes empêchent les conditions contextuelles de stop même après grâce.
Le watchdog externe conserve seul son autorité, sa source, ses limites et
son protocole. L'admission de démarrage demande toujours pression1.

Le disponible natif utilise (free_count+inactive_count)×page_size, comme
psutil7.2.2 installé, dont la version est exigée par ce runner. La définition
est vérifiable dans [la source officielle psutil7.2.2](https://github.com/giampaolo/psutil/blob/release-7.2.2/psutil/arch/osx/mem.c).
La baseline swap est celle du premier échantillon global du watchdog, transmise
explicitement par `AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES`, jamais reprise
à la première opération GPU. Elle figure dans l'enveloppe ressource et les
traces ; ce compteur dynamique ne fait pas partie de la clé mathématique.

`MTLDevice.currentAllocatedSize`, présent dans le SDK local MTLDevice.h:629,
est échantillonné après allocation, avant chaque commande et avant chaque
publication, avec un plafond distinct64 Mio. Les traces séparent valeur
courante et maximum observé échantillonné du payload logique8028160 octets
et du RSS. Cette API mesure les allocations de ressources du device ; elle
ne prouve ni un pic continu ni toute la résidence ou l'épinglage du pilote.
La résidence pilote non mesurée reste explicitement `null`.

## Métadonnées et étapes encore requises

`profile.py` construit une identité `native_cpu_with_metal_q8_bridge`, même
si `n_gpu_layers` vaut 0. Elle exige les SHA des nouvelles bibliothèques,
du patch, du noyau, du runner et de la politique de garde, ainsi qu'un reçu
de feature compilée. L'extension versionnée s'insère sous
`render_options.gpu_execution_profile` et ses knobs sous
`render_options.native_environment`, emplacements déjà acceptés et hachés par
le schéma 1 ; aucun nouveau champ racine. Elle ne remplace pas l'identité du
modèle et du contexte.
Les traces `actual_gpu=1`, allocations et soumissions restent obligatoires.
Une campagne sans commande GPU ne peut pas qualifier ce provider.

Le runner actif et son filtre d'environnement sont volontairement inchangés.
La copie isolée `native_gpu_run.py` consomme maintenant explicitement ce JSON,
les reçus de build du provider/driver et les empreintes réelles des fichiers.
Elle vérifie aussi les images natives réellement chargées, annoncées par dyld
dans le nouveau driver ; de simples fichiers présents dans un package ne
suffisent pas. `gpu_contract.py` rejette un backend incohérent, le mode CPU
legacy, une clé changée, une copie de poids permanente, des compteurs non
terminaux ou l'absence de commandes GPU. Une importation de profil seule ne
peut pas rendre la campagne réussie.

Cette copie reprend la candidate UID : même verrou du modèle et UID, pris
avant toute requête native de métadonnées, même FD hérité par le helper,
observation OS limitée à ce PID/FD et arrêt limité au child retenu. Ne pas
l'envelopper en plus dans le superviseur UID externe : cela tenterait de
reprendre le même verrou. Le verrou de benchmark, les limites RSS/swap,
réserves/grâces et classes d'échec natives sont conservés. Une réservation
additionnelle de 128 Mio est facturée : données64 + politique pilote/framework64,
sans affirmer un pic RSS du pilote mesuré. L'admission est répétée après les
métadonnées. Un nonzero/deadline/arrêt mémoire conserve sa classe même si la
trace GPU secondaire est incomplète.

`driver/` est une copie du helper `cpu8-tail-compatible-v1` figé, enrichie
seulement des descripteurs GPU, de la feature compilée et des images dyld.
La CLI garde ses six arguments et le traitement des queues CPU8 est conservé.
À N16+, le runner refuse le driver legacy divergent. La copie reste non
compilée/non exécutée ; la campagne mesurée et ses fichiers gelés sont intacts.

`--state-parameters` conserve les paramètres mathématiques à la racine et
ajoute seulement deux enveloppes réservées : `native_linkage` (images dyld)
et `native_resource_context` (baseline de garde). Le runner retire exactement
ces deux enveloppes après validation. Tous les autres champs constituent
`manifest.state_parameters`, identique à `report.state_parameters` vérifié
postflight. Les images réellement chargées sont vérifiées avant et après le
modèle ; la baseline doit rester la même. Le rapport absent ou incohérent
refuse un code0, sans remplacer une erreur native antérieure.

`build_package.py` écrit uniquement un plan par défaut. Le futur `--execute`
exige une destination neuve sous cette candidate et ne remplace aucune V4.
Le configure/build réutilise exactement les arguments historiques du manifeste
V4 (Release, clang/clang++ épinglés, SDK, Ninja, Metal embarqué, bibliothèques
partagées, cibles llama-server/llama-completion, jobs3). Seuls le bridge ON et
le compilateur identique pour le nouveau langage Objective-C++ sont ajoutés.
Le cache CMake historique V4 n'a pas été conservé ; le plan cite les arguments
réellement enregistrés, sans prétendre posséder ce cache. Les versions exactes
clang et CMake sont requises avant compilation ; le cache nouvellement généré
sera haché. Les délais bornés sont configure120s, build600s, helper120s ; ce
sont des plafonds, pas une durée prédite. Aucun de ces builds n'a été exécuté.

Les nouveaux reçus schéma2 suivent les sources originales et staged avant/après,
les sept overlays, le noyau/patch, les16 patches V4, les13 sources V4, CMake,
les quatre en-têtes du driver, snapshot.h et state_capture.inc. Le driver lie
le SHA du reçu provider et les mêmes SHA réels de dylibs. Ses smokes gardent le
stdout réel, les paramètres et les deux enveloppes pour chaque UB8..128. Le
runner revérifie ces entrées sur disque, les images dyld et la feature réellement
compilée ; une déclaration de JSON seule ne suffit pas.

Avant toute exécution GLM : revue indépendante du diff ; build isolé avec
empreintes ; fixtures natives multi-workers couvrant les grandes formes,
queues, données refusées et erreurs terminales ; mesures CPU/GPU complètes,
puis captures routes/logits/états. Un essai d'une projection est un contrôle de
raccordement, pas le moteur final ou une justification de campagne longue.
Le contrat bit à bit CPU et les tolérances Metal restent distincts ; aucun
critère PR19 ou du modèle n'est assoupli.
