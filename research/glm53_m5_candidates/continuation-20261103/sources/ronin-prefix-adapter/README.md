# Adaptateur RONIN de préfixe partagé — source isolée, non activable

Ce répertoire contient un candidat Python et des fixtures de protocole. Aucun
modèle, serveur, build ou GPU n’est lancé par import ou par les tests. Aucune
source mesurée ou de production n’est modifiée. **B8 a échoué au seuil de swap ;
il n’existe donc actuellement aucune admission réelle permettant ce cache.**

`make_model_class(NativeGlmModel)` fournit une sous-classe pour une future factory
de recherche injectée dans `runtime.engine._model_factory`. Ce code ne change
pas la factory de production. Il refuse par défaut : il faut le booléen explicite
`enable_shared_prefix=True`, un `NativePrefixGate`, un budget global créé avant le Runtime,
un répertoire de preuve neuf et un callback des gardes externes existantes.
`campaign.py` fournit maintenant un lanceur de recherche isolé. Son mode par
défaut affiche uniquement le plan, sans importer le Runtime ni lancer le
serveur. Le chemin d’exécution exige `--execute-research`, des chemins explicites
de configuration/preuves et une admission native complète réellement positive.
Ce chemin n’a pas été exécuté et reste bloqué avec les preuves actuelles.

`guards.py` raccorde maintenant le callback aux classes existantes `PowerGate`
et `NativeLimits` ainsi qu’à l’observation OS du seul PID/FD détenu. Avant le
chargement, son probe non bloquant du verrou UID est **libéré avant** l’appel à
`NativeGlmModel.load` ; le reçu signale une absence de conflit observé, sans
prétendre détenir un enfant ou le verrou natif. La course éventuelle est tranchée
par l’acquisition atomique du verrou dans le chargement natif existant. Pendant
l’intervalle où ce chargement détient déjà son FD mais n’a pas encore de PID,
le callback observe ce FD sans tenter de le reverrouiller. Après spawn, il exige
le même fichier/FD natif, l’héritage effectivement observé dans l’enfant,
son UID, parent, exécutable, naissance, génération et objet Popen inchangés.
Il ne conserve jamais un second flock autour du chargement. `_admit`, `_watch`,
`_check_identity`, le verrou modèle et le nettoyage du backend natif restent
hérités. Les limites existantes sont acceptées seulement si elles sont identiques
ou plus strictes, avec RSS ≤ 8,5 Gio, swap ≤ 32 Mio et délais bornés. Un niveau
de pression 2 admis par la politique native en phase chaude n’est pas rejeté au
seul motif de ce niveau. Une réserve supplémentaire conservatrice de 17 Mio est
exigée avant les opérations HTTP (16 Mio pour les transferts d’état natifs
épinglés, 1 Mio pour le hachage Python). Il s’agit d’une admission statique
supplémentaire, pas d’un pic HTTP mesuré ni d’un plafond d’allocation interne.

Avant le premier chargement, `NativePrefixGate.verify()` exige les deux captures
complètes full/restore du profil et relance leur comparateur strict sur les
**preuves closes**, sans imposer aucun nom de dossier. Le profil 8 exige full8/
restore8. Un futur profil 16 exige simultanément full16/restore16 stricts et le
vrai pont fullCPU8→full16 du comparateur de variante épinglé, avec ses binaires,
sources, builds réels et patch liés. Déclarer simplement 16 ne l’admet jamais.
Les trois modules de comparaison/clé/critères sont épinglés par SHA. Il exige
le même manifeste, les paramètres exacts du helper, 208 IDs natifs et le hash
`4c4d6f24…` de 156 739 588 octets. Aucune capture partielle, simple première
requête exacte ou réussite de mock ne satisfait cette admission.

Le runner, le gate natif et les modules délégués doivent provenir d’un seul
checkout cohérent. Dans l’agencement livré, utiliser
`/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/airllm-next`.
Le comparateur de variante épinglé `07939855…` dérive lui-même ce chemin depuis
son répertoire frère. Le pointer vers `work/publication` ne transfère pas sa
provenance et est refusé avant l’import des modules. Une nouvelle variante de
source ou de politique demanderait sa revue et de nouvelles preuves ; aucun
pin ni reçu mesuré n’est réécrit par cet adaptateur.
Le superviseur UID gelé est une exception de disposition explicitement liée :
il reste dans `work/uid_supervisor.py`, SHA256 `7857d7d0…c219`, chargé sous son
chemin réel avec vérification avant/après import et du checkout déclaré. Seule
sa fonction de lecture du FD détenu est appelée ; son CLI n’est pas exécuté.

Le serveur est unique, avec un slot physique. Les 600 identités restent celles
du Runtime/Engine RONIN ; le cache est une seule copie du préfixe commun et
aucun KV privé n’est enregistré par identité. Le serveur HTTP garde un profil 8
distinct, même si un futur profil natif 16 fournit ses preuves ; chaque profil
conserve sa propre clé. La sous-classe rend explicites
CPU10, batch8, contexte2048, swa_full, absence d’offload KV/op, absence de repack
(donc extra_bufts=false), absence de cache RAM/checkpoints/draft et slot-save-path.
Elle retire aussi les variables GGML/AIRLLM/DYLD héritées puis ne remet que les
options CPU du manifeste natif. Le drapeau `RONIN_PREFILL_DIAGNOSTIC=1` est
obligatoire dans les preuves natives ; son identité reste intégralement dans
le manifeste admis, mais ce drapeau de capture n’est pas transmis au serveur.
La clé HTTP déclare explicitement cette séparation. Le template brut est vérifié par `/props` ; les
messages complets passent par `/apply-template` et `/tokenize` natifs, sans
override low/max : le défaut natif Max est conservé. Les 208 IDs et le budget
contexte sont vérifiés pour chaque requête.
Le parseur V4 applique aussi des configurations automatiques avant la CLI.
L’adaptateur utilise donc un `XDG_CONFIG_HOME` privé vide pour ce seul enfant
et refuse la présence de `/etc/llama.cpp/config.ini` ou d’une configuration dans
ce répertoire privé, sans en lire le contenu ni modifier les fichiers utilisateur.
Le contrôle est répété avant spawn et pendant l’usage. L’audit source des 15
options ajoutées et de `--slot-save-path` est consigné dans `parser-audit.md`.

La préparation commune utilise honnêtement `n_predict=0` avec **un échantillon
produit, journalisé et écarté**, jamais ajouté à l’historique logique. Le slot
doit encore contenir exactement 208 tokens. La sauvegarde reste en quarantaine
jusqu’à validation stricte : fichier ordinaire privé, taille ≤ 256 Mio, format
texte pur sans annexe, IDs exacts et payload natif canonisé égal au checkpoint.
La validation est en flux, sans allocation Python du payload complet. Le fichier
admis devient lecture seule ; son identité et ses SHA sont revérifiés avant
chaque restauration. Une clé HTTP propre lie le manifeste natif qualifié, le
vrai serveur chargé, le nouveau wrapper, la politique héritée, les flags, les
paramètres, les limites, les sources du callback et ses dépendances. La clé du témoin natif n’est jamais réutilisée en
faisant disparaître la différence de backend ou de taille de lot.

Avant chaque suffixe : erase, restore, observation du slot, puis nouvelle
sauvegarde temporaire de l’état effectivement restauré et comparaison canonique.
Après génération : erase de l’état privé. Un échec, une annulation, un timeout,
un abandon du générateur, une copie corrompue ou un cache_n différent de 208
ferme le serveur, invalide les fichiers détenus et interdit le redémarrage
automatique de cet adaptateur. Le comportement des gardes natives n’est pas
redéfini. Un EOS vient exclusivement de l’événement natif.

Le budget global de 7200 s inclut qualification des preuves, chargement, startup
et préfixe commun. Un délai de 2700 s inclut toute la requête logique, dès avant
la file d’attente : rendu, tokenisation, admission, nettoyage, restauration,
vérification de l’état, traitement privé et génération. Les appels HTTP de
préparation/save/restore/erase et health, ainsi que les événements réels du flux,
sont comptés et journalisés séparément des demandes logiques. Le budget du
backend natif n’est jamais prolongé. Le monitor de délai conserve le mécanisme
de fermeture du backend existant et n’effectue aucune recherche globale de PID.

Limites avant toute intégration active : B8 positif absent ; chemin HTTP non
exécuté ; adéquation effective de tous les paramètres serveur et du scratch
à confirmer ; aucune preuve de fidélité globale des logits/routes/états de C ;
aucun EOS utile ni gain mesuré. L’API V4 n’a pas de plafond interne de fichier :
la taille prévisible doit être admise avant sauvegarde par le profil/superviseur.
La quarantaine et son contrôle après écriture ne remplacent pas ce plafond.
Les contrôles source du callback et du runner ont des mocks légers ; ils
demandent encore une revue indépendante et des reçus réels avant activation.

Le runner crée un état RONIN neuf, vérifie les 600 identités, injecte la factory
de recherche dans cette seule instance et charge une fois le serveur. Il crée
deux missions privées, puis réalise 1 → 600 → 1 en reprenant la première mission.
Le troisième prompt doit contenir la question et la sortie complète stockée
de la première décision de l’agent 1, sans marqueur de l’agent 600. Aucun résumé,
retrait ou remplacement par la sortie native avant la politique de texte RONIN
n’est admis. Les messages et la mémoire passent par les méthodes existantes.
`logical_scope()` démarre le délai avant `Runtime.dispatch`, donc inclut aussi
l’historique et l’appel Engine.load idempotent. La préparation commune est
comptée au budget global ; chaque restauration et son contrôle sont comptés
à la requête logique. Le journal `physical.jsonl`, horodaté depuis le début de
campagne, fournit les prompts rendus, IDs natifs et événements SSE sans appel
dupliqué. Les appels physiques et les trois générations logiques restent
distincts dans le rapport.

Le suivi continu à 0,5 s au plus conserve le secteur/lid/horloge, l’identité et
le verrou natif, sans rechercher d’autres PID ni signaler un PID non détenu.
Le nettoyage réutilise les méthodes durcies existantes, vérifie l’arrêt normal
du serveur et conserve les échecs/annulations comme tels. Un verdict fonctionnel
futur exige les trois EOS natifs utiles, les contrôles de réponse existants,
l’isolation et la reprise d’historique, les transcriptions complètes, les gardes,
l’intégrité et les sources inchangées. Même ce verdict laisse
`HTTP_numerically_qualified`, `exact_state_qualified` et `full_qualification`
à faux : le runner ne capture pas de nouveaux logits/routes/états globaux HTTP.
Une observation distincte conserve les deux premières réponses utiles avec EOS
effectivement enregistrées si le retour final à l’agent 1 échoue. Elle ne
qualifie ni toute la campagne, ni la reprise d’historique, ni la fidélité, ni
les gardes globales. Les résultats réellement acquis restent auditables.

Les tests positifs remplacent explicitement B8, la liaison du package et le
serveur par des doubles. Le payload de fixture fait quelques dizaines d’octets.
Ils vérifient uniquement les refus, le format et le contrôle de flux. Ils ne
qualifient jamais le cache, le runtime ou le modèle.

Le profil HTTP 16 qui porterait le gain du préfill corrigé n’est **pas implémenté
dans cette livraison** : `batch_size=16` est refusé. Il demanderait, après les
preuves natives réelles complètes indiquées plus haut, sa propre identité et
une étape privée supplémentaire seulement pour une queue impaire de plus de
8 tokens. Par exemple, pour 521 IDs : restaurer 208, préremplir jusqu’à 520 avec
un appel `n_predict=0`, compter/écarter son échantillon, puis envoyer les 521 IDs
complets pour la vraie génération. Pour 523, conserver 8+3 et non 10+1. Les deux
appels restent sous le même délai logique ; la première phase doit montrer
`cache_n=208` et la seconde `cache_n=520`, avec coûts/tokens physiques distincts.
L’échantillon de préparation n’est pas redécodé ni ajouté à l’historique ; la
nouvelle tâche doit repartir avec son sampler neuf. Cette extension est une
étape de source et de revue ultérieure, sans activation ni gain présumé ici.
