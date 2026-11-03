# Vérification HTTP16 contre les captures natives v1 — protocole non exécuté

Statut : lecture des sources V4 et protocole uniquement. Aucun serveur, modèle,
compilateur, appel CLI natif ou GPU n'a été lancé pour ce document. Le chemin
HTTP16 n'est pas numériquement qualifié. Quatre échantillons diagnostiques ne
qualifient ni une réponse utile jusqu'à EOS, ni la campagne C.

## Références et admissions

Le contrôle CPU8 original utilise le helper `62a7d257…`, source `558aa441…`.
Le candidat natif v1 est `cpu8-tail-compatible-v1`, source `dac2113041…`,
binaire `b2404d7d…43b8`, reçu de build SHA `e67695c6…d2073`.
Ces identités doivent être vérifiées intégralement par le comparateur 079398…
et les vrais reçus de build ; ces abréviations ne sont pas des pins utilisables.

Le mode HTTP16 sans préfixe commun exige le pont brut **complet**
CPU8 → full16 v1 : trois requêtes 1 → 600 → 1, gardes closes, code 0, sources/binaries
intègres, captures complètes, continuations et replays exacts. Deux premières
requêtes exactes d'un bras encore actif ne suffisent pas. Le mode avec cache
exige en plus full16 → restore16 strictement positif. Aucun état partiel,
arrêt swap, ancien CPU16 divergent, mock ou JSON déclaratif ne remplace ces gates.

L'HTTP a son identité propre : serveur et runtime réellement chargés,
adaptateur/runner/garde, argv observé, environnement nettoyé, presets absents,
template/tokenizer/shards, paramètres effectifs CPU10/GPU0/ctx2048/batch16,
slot0 unique, options K/V et sampler. Il ne reçoit jamais une clé native CPU
par substitution. Le futur v2 à fences208 est un autre protocole/build/clé.

Utiliser les prompts complets réellement gelés et leurs IDs natifs, pas une
reconstruction textuelle approximative. La troisième entrée du diagnostic
natif répète le prompt initial ; une troisième requête RONIN contenant son
nouvel historique exige une référence native contemporaine de ces **nouveaux**
IDs. Le replay natif ne prouve pas le retour d'historique fonctionnel.

## Ce que l'API V4 expose réellement

| Objet | Possibilité actuelle | Limite de preuve |
|---|---|---|
| IDs du prompt | `/apply-template` puis `/tokenize`, options spéciales explicites | Comparer tous les IDs et positions au bras natif ; aucun texte seul ne suffit. |
| IDs générés/stop/timings | `/completion`, `return_tokens`, événements réels et terminal | Ne pas inventer un ID omis ; vérifier ordre, compteurs et fin naturelle/budget. |
| Probabilités | `n_probs`, `post_sampling_probs` | Ce sont des probabilités/logprobas, pas le vecteur F32 absolu des logits. |
| État complet d'un slot idle | `/slots/0?action=save`, fichier privé validé | Peut permettre un SHA exact du payload natif canonique à une frontière observée ; parseur diagnostique à généraliser au prompt complet. |
| Logits F32 complets | Pas de dump générique dans `/completion` | Bloquant pour la fidélité complète HTTP avec le serveur actuel. |
| Routes MoE, sorties routeur/KDA, écritures KV/KDA | Non exportées par l'API de complétion | Bloquant pour les 118 noms et les écritures intermédiaires exigés. |

Justification source V4, sous le checkout reconstruit :

- `tools/server/server-common.cpp:1664–1715` lit les logits, trie les candidats
  puis applique une softmax. `server-context.cpp:2266–2319` conserve des champs
  `p`, et `server-task.cpp:256–290` sérialise prob/logprob. Même demander 154 880
  probabilités ne restitue pas les bits des logits originaux : l'offset commun
  est perdu et les transformations/arrondis peuvent perdre de l'information.
- `server-context.cpp:2527` lit des logits pour des labels du mode décision,
  mais son handler à 6607 refuse les modèles sans configuration de décision
  et prépare des tâches spécifiques. Ce n'est pas un endpoint de dump GLM
  interchangeable avec la complétion.
- `server-context.cpp:3243–3272` sauvegarde le slot complet et ses compteurs ;
  `:3295–3341` recharge les tokens et le payload. `:656–687` laisse un slot de
  complétion normal idle ; `:343–344` de `server-task.h` réserve le statut
  stateless aux embeddings/rerank. Cette possibilité doit encore être observée.
- `server-context.cpp:4865–4915` échantillonne, puis arrête ou poursuit ;
  `:602–653` n'ajoute le dernier token échantillonné à l'état que lors du décodage
  suivant. Cela motive les frontières N et N+3 ci-dessous, sans les supposer
  acquises à partir du seul paramètre `n_predict`.

## Diagnostic réalisable sans modifier le serveur actif

Préparer ultérieurement un **harness diagnostique séparé**, gardé et journalisé,
qui sauvegarde l'état avant l'effacement final du slot. Le runner fonctionnel
actuel efface le slot après la réponse ; un appel externe tardif ne peut pas
récupérer cet état. Ne pas contourner ses gardes ou réutiliser son serveur
personnellement avec des appels hors journal. Aucun harness de ce type n'est
exécuté ni déclaré disponible par ce document.

Pour chaque prompt fixe, deux passages indépendants sont possibles à tester :

1. Après erase/reset vérifié, appliquer le plan HTTP16 correspondant au natif
   v1. Pour une queue impaire> 8, la préparation s'arrête au préfixe multiple de8
   du dernier lot (521/523→520), puis seul le reste 1/3 est traité. Le mode
   commun restaure 208 avant cette préparation ; le mode sans commun part de 0.
   Chaque préparation `n_predict=0` doit produire exactement un échantillon
   **observé, compté et écarté**, jamais ajouté au prompt ni à l'historique.
   Exiger aussi `tokens_predicted=1`, `tokens_evaluated=end`, les timings
   exacts et le GET du slot à cette fin observée.
2. Passage A : terminer le prompt avec une requête diagnostique donnant un
   seul échantillon final. Observer que le slot contient exactement N IDs et
   positions N, puis sauvegarder. Comparer le payload canonique à
   `final-state-r.bin` et son SHA ; le premier ID échantillonné doit être le
   premier ID de la continuation native. Si le serveur a déjà réinjecté ce
   token, si les compteurs ne concordent pas ou si le slot a été effacé,
   refuser la comparaison à N.
3. Reset/erase indépendant, même plan et mêmes IDs. Passage B : demander les
   quatre échantillons diagnostiques, ou exactement le même EOS naturel à la
   même étape si la référence s’arrête avant quatre. Un EOS HTTP plus précoce
   ou plus tardif que la référence refuse la comparaison. Comparer chaque
   ID/ordre et la classification terminale.
   Observer N+3 positions/IDs effectivement évalués pour quatre échantillons sans
   EOS ; sauvegarder le slot idle et comparer le payload canonique à
   `continuation-state-r.bin`. Le quatrième token échantillonné n'est pas un
   quatrième token décodé. Toute différence de compteurs refuse la comparaison.
4. Refaire1 → 600 → 1 sur ces prompts fixes ; contrôler les replays, l'effacement
   entre identités, les marqueurs privés et les IDs complets. Enregistrer les
   deux préfill passes et tous les samples préparatoires : aucun coût ne doit
   être attribué à une seule passe ou dissimulé comme cache gratuit.

Un parseur externe futur devra conserver le header/format V4 exact et les
tokens sérialisés `server_tokens`, sans média/checkpoint/draft/trailing bytes,
valider les N IDs réellement évalués, puis reconstituer uniquement le header
canonique prévu par l'API native avant SHA du payload. Le parseur 208 de
l'adaptateur ne doit pas être utilisé tel quel pour N ou N+3. L'ensemble du
fichier, métadonnées comprises, reste sous 256 Mio ; calculer cette taille avant
save, lire/hacher en streaming et garder les réserves natives/IO et contrôles
PID/exe/FD/secteur inchangés. Un fichier plus gros ou un état non comparable
est refusé, jamais tronqué pour le faire correspondre.

Ces tests peuvent établir des IDs et des états terminaux exacts **observés**.
Ils n'exposent ni les quatre vecteurs 154 880 F32, ni les routes/KDA par token,
ni les écritures intermédiaires. Leur verdict doit donc rester
`HTTP_numerically_qualified=false`, même si tous les SHA comparables concordent.

## Instrumentation nécessaire pour lever les blocages

Préparer, seulement après revue et coordination, une autre copie du serveur
V4 exact avec un callback diagnostique non mutateur. Il devra capturer les
logits complets aux quatre points d'échantillonnage, les 118 noms originaux avec
types/formes/strides/positions et leur vraie couverture, les écritures d'état
observées et les états complets aux frontières communes. Réutiliser les
lectures brutes et critères CPU bit à bit existants, routes ordonnées exactes,
tokens exacts et toutes valeurs finies. Aucun seuil Metal ne s'applique à ce
chemin CPU16 ; aucune tolérance n'est élargie.

Ce serveur instrumenté aura un nouveau binaire/source/profil/clé, un reçu de
build réel avec en-têtes/libs avant/après et de nouveaux contrôles de mémoire,
annulation et continuité. Il ne remplace ni ne modifie le serveur actif, les
captures gelées ou les comparateurs0793. Demander `n_probs` ou activer une
variable du helper dans le serveur existant ne crée pas ce callback.

Le rapport futur doit séparer : admission native v1 ; transport/IDs HTTP ;
SHA des états HTTP observés ; captures numériques encore absentes ;
diagnostic quatre échantillons ; réponses utiles jusqu'à EOS de C ; performance
complète sous gardes. Aucune étape source-only n'est marquée exécutée.
