# Audit des états nécessaires au préfill commun GLM-5.3 Flash

## Décision de cette campagne

La réutilisation exacte est **structurellement plausible**, mais elle n'est
pas qualifiée par la seule présence d'une API de checkpoint. Le cache commun
reste désactivé. Aucun gain de temps, de lectures ou de mémoire n'est déduit
des 869 octets de texte communs ni d'anciens temps de préfill.

Deux propriétés du code interdisent un raccourci d'implémentation :

1. Les checkpoints ordinaires utilisent
   `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`. Pour une attention sans SWA, ce flag
   **omet les données d'attention et d'indexer**. Un tel checkpoint peut
   compléter un préfixe KV encore présent dans le contexte actif, mais ne
   constitue pas un snapshot autonome restaurable après un reset complet.
2. Le cache RAM générique restaure puis consomme son entrée : il libère les
   vecteurs sérialisés, déplace le prompt et supprime l'entrée. Ce n'est pas
   un unique préfixe immuable réutilisé pour 600 identités.

Ces observations n'établissent pas une impossibilité mathématique. Elles
signifient qu'activer `cache_prompt` ou conserver un checkpoint KDA existant
ne suffit pas à réaliser le contrat demandé.

## Sources et portée

Le point de départ est PR #20 `a980f69c40e9170aef411fb4333613d2f7c6415a`,
empilé sur PR #21 `8138298f927dd17f583b5c0caed884ea432c4c01`. La note
[`SHARED_PREFILL.md`](../glm53_ultra/SHARED_PREFILL.md) a été lue avant cet
audit. Le manifeste de construction V4 indique llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c` et les 16 patches conservés dans le
dépôt. Le répertoire temporaire de sources V4 a été supprimé par la
construction. Les dix fichiers relus ont donc été reconstruits séparément
depuis cette révision, en appliquant séquentiellement les 16 patches dont
chaque empreinte a été comparée au manifeste V4. Les dix résultats sont
identiques au checkout V3 lisible inspecté. Les deux patches historiques
`llama-layer-local-hc.patch` et `llama-moe-cache-split.patch`, absents du
checkout PR20, ont été retrouvés dans le checkout local de la campagne
précédente ; leurs empreintes correspondent exactement au manifeste V4.
Cette vérification porte sur les dix sources sélectionnées, pas sur une
nouvelle compilation bit pour bit du binaire V4.

[`native-state-source-audit.json`](evidence/native-state-source-audit.json)
conserve les empreintes des dix fichiers natifs relus, la révision et
l'empreinte du diff du checkout, la reconstruction sélectionnée V4 et les
régions numérotées utilisées ci-dessous.
Ce fichier est une preuve de lecture de sources, pas une capture d'état du
modèle. Les chemins natifs ci-dessous sont relatifs à llama.cpp.

## Inventaire des états

| Élément | Contrat pour une reprise au milieu du premier système | Source inspectée |
|---|---|---|
| IDs du préfixe et positions | Vrai plus long préfixe des prompts complets déjà rendus/tokenisés ; mêmes positions absolues, même `seq_id` physique et nombre de tokens traités. Une tokenisation du texte tronqué n'est pas suffisante. | Métadonnées `server_prompt` ; `common_prompt_checkpoint::update_pos` |
| Attention MLA/K-only | Cellules, clés/latents, positions, association de séquence et layout de streams correspondant exactement au préfixe. Flags `NONE` si le contexte est entièrement vidé avant restauration. | `src/llama-kv-cache.cpp:2068` |
| KDA/DeltaNet | R : historiques de convolution Q/K/V ; S : matrice récurrente par tête/couche, cellules, position, source des lignes et index de rollback actif. L'état doit être capturé **avant** l'identité privée. | `src/models/glm5-next.cpp:830`, `src/llama-memory-recurrent.cpp:783` |
| Indexer DSA | Clés, cellules et layout couplés à l'attention ; groupes KPool à invalider/reconstruire de façon cohérente. Un restore séparé cherchant ses propres cellules n'est pas équivalent. | `src/llama-memory-hybrid-idx.cpp:253` |
| mHC | Les quatre flux résiduels, matrices de mélange, normalisations et sorties sont des activations du graphe courant. Dans cette implémentation ils ne forment pas un cinquième cache temporel à sérialiser à une frontière entre tokens. Ils restent des cibles de comparaison après reprise. | `src/models/glm5-next.cpp:699` |
| Routage MoE et sélection DSA | Les sorties de routeur et sélections sont calculées dans le graphe. `prev_sel` est transmis entre couches DSA du même forward, puis réinitialisé au forward suivant. Il ne doit pas être promu en état inter-agent. Comparer les routes **ordonnées**, pas seulement les experts comme ensemble. | `src/models/glm5-next.cpp:726` |
| Sampler et chat | Un sampler propre à la requête/identité, historique d'acceptation, grammaire éventuelle, limites et parser de sortie frais. Le snapshot de mémoire de séquence ne sauvegarde pas automatiquement ce contrat hôte. | `tools/server/server-context.cpp:325`, initialisation du sampler en fin de prompt |
| Spéculation/MTP | Absente du profil CPU témoin visé. Si elle était ajoutée, état du draft, caches et `data_spec` devraient être inclus et qualifiés. `h_nextn` n'est pas couvert par une simple supposition de KV. | `common/common.h:1300`, `tools/server/server-context.cpp:2778` |
| Poids, templates et runtime | Identité vérifiée des shards, tokenizer, gabarit et options `low`, bibliothèques, paramètres de contexte/batch, types des caches et calcul CPU10/GPU0. Les entêtes de session natifs ne vérifient pas ces identités complètement. | `src/llama-context.cpp:3737` |

Un snapshot autonome emploierait donc la mémoire complète de la séquence
(`LLAMA_STATE_SEQ_FLAGS_NONE`) et des métadonnées hôtes explicites. Le flag
`PARTIAL_ONLY` ne serait admissible qu'avec un protocole différent préservant
le KV commun dans l'état actif et prouvant séparément son reset exact. Ce
dernier protocole ne peut pas effacer puis restaurer seulement R+S.

## Sérialisation, reset et échec

`llama_memory_hybrid_idx::state_write` écrit l'attention, le récurrent puis
l'indexer. La lecture doit adopter pour l'indexer le layout d'attention
restauré. Un échec partiel appelle `state_drop`, qui vide les trois caches
cohérents. Pour toutes les séquences, il passe par `clear(true)` ; pour une
séquence, il efface attention/indexer et supprime entièrement sa cellule
récurrente.

`llama_memory_recurrent::seq_rm` peut refuser le retrait d'un suffixe. Le
rollback est borné aux snapshots de la dernière microbatch et ne reconstruit
pas un état arbitraire ancien. `llama_memory_hybrid_idx::seq_cp` n'accepte que
la copie d'une séquence entière ; une plage de préfixe partielle déclenche
une assertion. Ces fonctions ne permettent pas de transformer l'état final
de l'agent 1 en état commun pour l'agent 600.

`common_prompt_checkpoint::update_tgt` demande la taille, alloue/redimensionne
un vecteur et sérialise ; il vérifie le nombre d'octets écrits.
`load_tgt` vérifie seulement que le nombre d'octets restaurés correspond au
vecteur. Cette vérification est nécessaire, mais elle n'établit ni l'identité
des poids ni l'égalité des logits après reprise. Le sérialiseur global vérifie
l'architecture et contient explicitement un TODO sur les identités de modèle
manquantes ; le sérialiseur de séquence ne résout pas ce manque.

Les checkpoints de serveur sont créés avant le prochain `llama_decode`,
prioritairement au début des messages utilisateur et près de la fin du
prompt. Aucun checkpoint utile à la divergence numérique dans le premier
système n'est attesté. Les checkpoints existants présument aussi qu'une partie
de l'attention demeure dans le contexte actif.

## Frontière de microbatch et exactitude

Une frontière à l'intérieur du vrai préfixe commun peut modifier les formes
des opérations et le scan KDA. La première expérience devrait choisir une
frontière de microbatch déjà employée par le témoin, à l'intérieur du LCP
tokenisé, et conserver les frontières suivantes. Si le LCP ne contient
aucune frontière utilisable, il faut comparer explicitement le préfill
original au préfill découpé avant d'attribuer une différence à la restauration.
Il est interdit de raccourcir ou réordonner les messages pour obtenir une
frontière commode.

Trois comparaisons distinctes seraient nécessaires :

1. Témoin CPU10/GPU0 complet contre même CPU sans cache mais avec le découpage
   candidat : isole les effets des formes et du scan.
2. Même découpage recomputé contre restauration du préfixe : isole le restore.
3. Séquence agent 1 → agent 600 → retour agent 1, avec marqueurs, historique
   modifié et annulation : cherche les fuites et états hôtes non restaurés.

Comparer sur les positions réellement capturées : IDs d'entrée/sortie,
logits complets finis, routes et scores de routeurs, R+S KDA, cellules
attention/indexer/positions, et activations mHC utiles. Des IDs identiques
seuls ne constituent pas une preuve d'égalité des états. Une capture finie
reste une qualification sur son périmètre et n'est pas une preuve universelle.

## Borne mémoire et coût net

Les dimensions déjà inspectées donnent **152 633 344 octets (145,5625 Mio)**
de R+S logiques F32 pour une cellule et 34 couches KDA. Ce n'est pas la taille
sérialisée mesurée d'un snapshot. Attention, indexer, positions, tokens,
métadonnées, capacités des vecteurs et scratch s'y ajoutent. Six cents copies
des seuls R+S feraient environ 85,29 Gio et sont exclues.

Le candidat admissible serait au plus **un snapshot immuable**, un contexte
actif et une restauration en vol sous le même verrou du slot. Le budget
strict doit compter leurs résidents et capacités simultanés, les caches
internes de serveur et le scratch ; il ne peut pas être déduit d'un nombre
de slots égal à un. Une limite de **256 Mio pour le snapshot** est un plafond
de recherche possible, pas la preuve qu'un snapshot complet y tient : refus
avant allocation si la taille native complète dépasse ce plafond, et refus
si le budget global/RSS ne laisse pas la marge requise. Aucune allocation de
ce type n'est activée ici.

Le cache RAM générique borne ses propres entrées en faisant de la place avant
d'allouer, mais il clone aussi les checkpoints du prompt. Cette borne ne couvre
pas les checkpoints encore détenus par le slot ni le graphe actif. Il ne
remplace donc pas un budget global pour le candidat.

Le coût de création comprend calcul du préfixe, sérialisation et copie ; le
restore copie les tenseurs depuis le vecteur. Même sur mémoire unifiée, ces
transferts ont un coût. Aucune valeur de bande passante de la machine ne leur
est substituée. Pour N requêtes, le gain net doit être calculé à partir de
durées réellement mesurées : `temps témoin total - temps candidat total`,
incluant création, restaurations, suffixes, invalidations, I/O et gardes.
Préfill et décodage doivent rester séparés. Mesurer aussi lectures par
processus, RSS maximale, compression et hausse du swap. Si le gain net n'est
pas positif sur les cas mesurés et répétés, garder le candidat désactivé.

## Témoin réellement sans cache

Le backend RONIN actuel envoie `cache_prompt=False`, mais les defaults natifs
comprennent `cache_ram_mib=8192`, `n_ctx_checkpoints=32`, cache des slots inactifs
activé et cache de prompt global activé. `get_available_slot` peut sauvegarder
un ancien prompt avant le traitement du drapeau de cache de la requête.
Ces limites ne sont pas des allocations mesurées.

Un **profil de recherche** strictement sans cache doit donc fixer
explicitement les options natives suivantes, en plus de la requête sans
cache et du slot physique unique :

```text
--cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots
--cache-reuse 0 --no-cache-prompt
```

Le lancement doit vérifier que ce runtime les accepte et enregistrer l'argv
effectif ; il ne faut pas réessayer silencieusement en les supprimant.
Les gardes RSS 8,5 Gio, hausse swap 32 Mio, pression mémoire, intégrité,
alimentation secteur, surveillance PID et arrêt/cancel restent inchangées.
Ce profil de mesure n'est pas une activation de cache en production.

La réutilisation du préfill reste refusée tant que sa sérialisation autonome,
son isolation, sa borne globale et son coût net ne sont pas mesurés. Une
campagne bloquée par les ressources doit produire un résultat **non mesuré**,
jamais un gain estimé présenté comme obtenu.
