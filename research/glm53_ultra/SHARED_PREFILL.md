# Préfixe commun RONIN : existence, état natif et limites

**Un préfixe textuel commun existe sans changer les messages : 832 caractères,
soit 869 octets UTF-8, dans le premier message système des agents 1 et 600.**
La construction actuelle et les noms par défaut du registre isolé le rendent
commun aux 600 identités de ce probe. **Sa longueur commune exacte en tokens
n'est pas enregistrée dans les preuves disponibles.** Aucun tokenizer ni appel
au serveur n'a été lancé pour cette note.

Un seul checkpoint immuable de ce vrai préfixe pourrait constituer une piste
de mutualisation du préfill. Cette possibilité n'est pas implémentée ni
qualifiée. Activer simplement `cache_prompt` ne démontre pas cette
réutilisation, notamment à cause de l'état KDA et des positions auxquelles le
serveur crée ses checkpoints.

**Piste différée explicite dans cette livraison :** aucune activation de
`cache_prompt`, aucun réglage `cache_ram=0` et aucun changement des options de
checkpoints natifs n'est effectué. Cette note n'introduit aucune configuration
non testée dans le backend ou la production.

## Préfixe réellement retrouvé

[`AgentContexts.begin`](../airllm-next/integrations/ronin/runtime/agent_context.py)
construit le début du système dans cet ordre :

```text
IDENTITY

Identité logique : RONIN-<numéro> (agent <id>).
Mission isolée : <mission>.
Rôle : <rôle>.
Spécialisation : <spécialisation>
Instructions : <instructions>
... mémoire de travail, notices et historique ...
```

La constante `IDENTITY` de
[`ronin_runtime.py`](../airllm-next/integrations/ronin/runtime/ronin_runtime.py)
compte **805 caractères / 841 octets UTF-8**. Le préfixe exact commun est cette
constante suivie de `\n\nIdentité logique : RONIN-`. Il finit donc **avant les
chiffres** de `RONIN-001` et `RONIN-600`.

La comparaison a porté sur les deux systèmes effectivement sauvegardés dans
les décisions de la base isolée du probe, ouverte en lecture seule. Le
registre contient 600 identités, dont tous les noms correspondent au format
`RONIN-{id:03d}`. Les sources courantes `agent_context.py` et `ronin_runtime.py`
correspondent aux hashes du manifeste gelé de ce probe.

| Périmètre | Plus long préfixe textuel établi |
|---|---|
| Systèmes sauvegardés des agents 1 / 600 | 832 caractères / 869 octets UTF-8 |
| Les 600 noms par défaut de ce registre et cette construction | Même préfixe jusqu'à `RONIN-` |
| Deux noms voisins, par exemple 001 / 002 | Peuvent partager quelques chiffres de plus ; cela ne s'applique pas aux 600 ensemble |

Ce constat n'établit pas 600 inférences réelles. Il porte sur les messages et
les identités hôtes. Les rôles, spécialisations, instructions, UUID de mission,
mémoire et historique restent après la divergence ; ils doivent être
recalculés et isolés comme aujourd'hui. Un texte identique situé **après** une
divergence n'est pas un préfixe réutilisable : son état dépend déjà du contexte
antérieur.

Les changements de mémoire, d'historique ou de notice de tentative interrompue
n'altèrent pas ce début dans la construction inspectée. Un changement de
constante globale, de nommage, de gabarit ou d'options de rendu peut en revanche
modifier le préfixe et doit invalider la clé correspondante.

## Tokens déjà disponibles : ce qu'ils permettent et ne permettent pas

Le [préflight existant](../airllm-next/research/glm53_ultra/evidence/context-preflight.json)
rapporte **444 tokens** pour son agent 1 et **443** pour son agent 600, avec le
tokenizer/gabarit compagnon et `reasoning_effort=low`. Il conserve seulement
ces tailles pour les agents, **pas leurs listes d'IDs ni les prompts rendus**.
Ses missions aléatoires et son texte de mémoire sont aussi distincts de ceux
du probe réel ; ces tailles ne doivent pas être substituées à chaque requête
réelle.

À la clôture du [probe](agent-nocache-awake-v4-1/report.json), le terminal du premier agent fournit son prompt rendu
et le compteur natif **447 tokens d'entrée**, mais pas la liste de leurs IDs.
Le préfill natif rapporté est **937,168513 s** ; le temps hôte de la requête est
**947,229228 s**, avec chargement et autres frais inclus dans ce dernier
intervalle. L'unique ID produit est **154842**, `</think>`, avec arrêt `limit`
et décision `partial` : ce n'est ni une réponse finale à la demande ni une
mesure de débit de décodage. Le deuxième agent est annulé lors du changement
secteur → batterie observé par le [wrapper](agent-nocache-awake-v4-1-power.json), et le probe n'est pas qualifié comme deux requêtes
réussies. Cette clôture n'établit pas un incident mémoire.

Le prompt rendu du premier agent confirme textuellement l'en-tête `low`
attendu. La longueur totale de 447 ne fournit toutefois pas la longueur
tokenisée du préfixe commun, et les durées ci-dessus ne permettent pas de
prédire la part économisable par ce préfixe.

Les six IDs du prompt brut « La capitale de la France est » sont sauvegardés,
mais ne renseignent pas ce préfixe d'agent. La
[calibration native de chat court](chat-native-template.json) conserve bien
19 IDs et son rendu exact, mais son entrée contient seulement le message
utilisateur, sans les systèmes d'agents. Elle ne permet donc pas de mesurer le
plus long préfixe tokenisé des agents 1 et 600.

La lecture du gabarit compagnon existant confirme un en-tête commun pour
`low` sans outils : `[gMASK]<sop><|system|>Reasoning Effort: Low`, puis le rôle
système et le texte ci-dessus. Cela soutient l'existence d'un début de rendu
commun, **sans établir le nombre d'IDs du préfixe de 869 octets**. La
tokenisation peut fusionner des fragments à la frontière : compter des
caractères, ou tokeniser un préfixe séparément, ne suffit pas à certifier le
préfixe d'une séquence complète.

Un futur contrôle devrait conserver les IDs réellement produits par la voie
native pour les prompts complets, calculer leur plus long préfixe identique,
puis choisir une frontière de tokens à l'intérieur de celui-ci. Aucun nombre
de tokens communs, pourcentage de préfill évité ou gain de latence n'est
inventé ici.

## Contrat KDA : un état de préfixe, pas un KV simplement tronqué

Le runtime V4 est dérivé du commit llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c`, avec ses patches conservés. Les
sources inspectées exposent :

- `common_prompt_checkpoint::update_tgt/load_tgt` : taille native, copie de
  l'état sérialisé, vérification du nombre d'octets restaurés et positions
  associées.
- `llama_memory_hybrid_idx::state_write/state_read` : état d'attention, état
  récurrent et indexer. L'indexer doit adopter le layout des cellules
  d'attention ; les groupes KPool doivent rester cohérents.
- `llama_memory_recurrent::seq_rm` : le retrait partiel de fin peut être
  refusé. Le rollback borné ne couvre que les snapshots conservés de la
  dernière microbatch ; avec `n_rs_seq=0`, un état final ne permet pas de
  revenir arbitrairement au préfixe.
- `llama_memory_hybrid_idx::seq_cp` : seule la copie d'une séquence entière est
  admise ; copier arbitrairement une plage de préfixe est explicitement refusé.

Il faut donc capturer l'état **avant toute partie spécifique à un agent**.
Effacer le suffixe du KV d'un agent déjà traité ne reconstruit pas l'état KDA
commun. Une restauration partiellement échouée doit vider tous les caches
cohérents, sans poursuivre sur un mélange d'états.

Le checkpoint réutilisable devrait conserver les vrais IDs/positions du
préfixe et tout l'état hybride nécessaire. Il ne faut pas le fabriquer par un
chat autonome contenant seulement `IDENTITY` : son gabarit ajouterait ses
propres frontières et `<|assistant|><think>`, qui ne sont pas la continuation
du système réel à cet endroit.

La séparation préfixe/suffixe peut aussi changer les formes et frontières des
microbatches par rapport au préfill original. Une frontière compatible avec
les microbatches existantes est à rechercher **après** connaissance des IDs,
sans modifier les messages. L'équivalence des scans KDA, des routes et des
logits doit être vérifiée ; elle ne découle pas de l'identité du texte.

## Pourquoi `cache_prompt=True` ne suffit pas

L'adaptateur actuel appelle `stream_chat` sans option de cache ; la valeur par
défaut est `cache_prompt=False`, `id_slot=0` et un seul slot physique.
`session_kv_persistence=False` reste déclaré.

Le serveur sait comparer les préfixes et restaurer des checkpoints. Mais ses
checkpoints automatiques sont privilégiés aux débuts de messages utilisateur
et près de la fin du prompt. Le préfixe commun trouvé ici finit **au milieu du
premier message système**, avant l'identité numérique : aucun checkpoint
utile à cette frontière n'est attesté. À défaut d'état compatible, la voie
native revient au recalcul complet. Les preuves de reset de requêtes ou
d'isolation hôte existantes ne qualifient pas un nouveau chemin de préfixe.

Autre distinction importante : `cache_prompt=False` dans une requête ne
désactive pas tous les mécanismes de cache natifs. Les defaults de la source
sont notamment **8 192 Mio de limite de cache RAM**, **32 checkpoints de
contexte**, cache des slots inactifs activé et similarité de prompt à 0,1.
La commande RONIN inspectée ne fixe pas ces options. Le chemin
`get_available_slot` peut sauvegarder l'ancien prompt lorsque la fraction
conservée tombe sous 0,5, même avec un slot demandé explicitement, avant le
traitement du drapeau de cache de la requête.

Ces defaults sont des **limites/options**, pas la preuve que 8 Gio sont
alloués, que ce cache est peuplé dans le probe, ou qu'il a causé un arrêt.
Ils signifient qu'un futur budget « un checkpoint commun » doit aussi
contrôler les caches internes et copies temporaires, sans supposer que le
nombre de slots physiques les borne à un seul état. Aucun de ces paramètres
ou fichiers gelés n'a été changé pendant cet audit.

## Un checkpoint commun et un slot : mémoire et isolation

Les dimensions du modèle donnent, pour une cellule F32 et 34 couches KDA,
**152 633 344 octets de R+S**, soit **145,5625 Mio**. Cette allocation logique
concorde avec l'ancien diagnostic natif V4 à 145,56 Mio. Elle n'est pas une
mesure de taille sérialisée d'un checkpoint : attention, indexer, positions,
tokens et métadonnées s'y ajoutent selon les flags et cellules sauvegardées.

Un snapshot commun représenterait une copie d'état supplémentaire à budgéter,
en plus de l'état actif et des copies de restauration éventuelles. Six cents
copies des seuls R+S représenteraient environ **85,29 Gio** ; elles sont exclues
de cette piste. Les 600 historiques et mémoires logiques restent des textes
persistants distincts, sans 600 KV physiques.

Les invariants d'un éventuel chemin futur seraient :

- Un snapshot de base immuable, créé avant les chiffres d'identité, missions,
  mémoires et sorties privées ; jamais un état final d'agent étiqueté commun.
- Restauration et traitement du suffixe sous le même verrou du slot physique,
  une requête à la fois. Aucun partage mutable de l'état actif.
- Clé liée aux poids vérifiés, tokenizer/gabarit natifs et options, IDs exacts,
  position, hashes des bibliothèques, politique CPU/GPU, dtypes et paramètres
  de contexte/microbatch. Les rendus `low`, `high` ou défaut ne sont pas
  interchangeables.
- Budget global de résidents, copies en vol, caches internes et scratch ;
  gardes RSS, pression et swap inchangées. Pas de dictionnaire de snapshots
  grossissant implicitement avec les 600 agents.
- Invalidation sûre après annulation, erreur, changement d'identité du
  runtime/modèle ou restauration incomplète ; aucune réutilisation après
  redémarrage sans contrat de restauration séparément qualifié.

Cette réutilisation ne raccourcit pas le prompt logique : les tokens communs
occupent toujours leurs positions et comptent toujours dans la limite de
contexte. Elle pourrait éviter leur **recalcul**, pas augmenter la capacité
du contexte ni le débit intrinsèque de décodage.

## Ce qui reste à établir, sans relancer une expérience ici

La piste est plausible structurellement, car le texte commun est déjà placé
en tête et ne demande aucun réordonnancement d'instructions. Son intérêt
économique reste inconnu. Le coût de création du snapshot, de restauration,
des effets de mémoire et des changements de batch doit rester inférieur au
préfill effectivement évité sur plusieurs requêtes. Aucun de ces temps n'est
mesuré ici et la durée du probe clos ne permet pas de les déduire.

Avant une activation, il faudrait une preuve dédiée full recompute versus
restore sur les mêmes prompts complets : logits, routes ordonnées, routeurs,
KDA/états d'attention-indexer, tokens et isolement entre deux agents ; puis
annulation et historique/mémoire modifiés. Les seuls IDs générés ou
l'existence d'une API de sauvegarde ne suffisent pas. Les performances de
préfill et de décodage resteraient rapportées séparément, sans généraliser
deux agents à 600 inférences validées.

## Empreintes des éléments relus

| Élément | SHA-256 |
|---|---|
| Préfixe système commun, 869 octets | `a41e3d7ecdda02807929953b9ac6f04fd2e5b6f5d192d254ae3b22c223204ce3` |
| `agent_context.py`, identique au manifeste gelé | `8babb6f477f154de547b99de073961c80be4f3e310083ec1d6dae5e0167385fa` |
| `ronin_runtime.py`, identique au manifeste gelé | `a4bb3ee49aabccdb5500ca1adc5c95fc1d62dffa8d3a42e6e73740f614e17e51` |
| Préflight compagnon existant | `2efb489c6832b9de805eea89ac6e3289d7b2e6d8040122eea157bd6ccc5214af` |
| Calibration native de chat court | `f440e19bc975eb46ba3383946adffe572ae1c11fadb8e36aa4d9d3f49cdf4ddf` |
| Gabarit compagnon déjà utilisé | `2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4` |

Les messages de décision ont été lus dans la base **isolée du probe** en mode
lecture seule ; aucun historique utilisateur extérieur à ce périmètre n'a
été consulté. Les empreintes de textes/sources ci-dessus n'impliquent aucune
nouvelle tokenisation ni exécution du modèle.
