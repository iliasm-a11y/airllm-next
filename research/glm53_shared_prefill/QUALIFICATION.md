# Campagne bornée de deux décisions RONIN

`qualify_agents.py` exécute les agents **1 et 600**, chacun sur une mission et
une mémoire distinctes, avec leurs véritables identité, rôle, spécialisation et
instructions RONIN. Les deux tâches demandent un calcul exploitable et sa
justification : consommation et marge d'un stock pour le premier, comparaison
de deux devis pour le second. Le programme ne remplace aucune réponse par du
texte hôte et ne raccourcit aucun contexte pour obtenir un EOS.

Le contexte configuré doit être explicitement compris entre 1 024 et 2 048
tokens. Le contexte historique de 512 tokens ne laisse pas la place aux
prompts complets et au budget de sortie utile. La voie native tokenise chaque
prompt complet et refuse `entrée + budget > contexte`, sans retrait de tour,
résumé automatique ou glissement de contexte. Les 600 identités restent des
contextes logiques ; un seul modèle et un seul slot physique sont acceptés.

## Admission et arrêt

La campagne ne construit pas de runtime d'inférence si l'admission ne confirme
pas le secteur et le couvercle ouvert. Le contrôle est répété avant le
chargement, à la construction de la commande native et pendant l'exécution.
Le contrôle des horloges et l'assertion de veille possédée utilisent les outils
existants. Il s'agit d'observations échantillonnées, pas d'une preuve de chaque
événement électrique matériel.

Les plafonds sont **8,5 Gio de RSS** et **32 Mio de croissance de swap**.
L'ensemble des réserves mémoire, seuils de pression, délais de grâce et
intervalles du watchdog est conservé à sa valeur existante ou rendu plus
strict. Le chargement reste limité à 900 secondes. La requête reste à
1 800 secondes par défaut ; une option explicite `--request-seconds 2700`
permet le plafond de recherche de 45 minutes, en conservant une configuration
plus stricte déjà fournie. Cette option refuse un plafond global supérieur
à 7 200 secondes. La limite globale de 7 200 secondes est conservée ; elle
ne transforme jamais une annulation ou une sortie sur limite en réussite.
Aucun rejeu automatique n'est effectué. L'arrêt conserve les événements
d'annulation et les fonctions de fermeture du backend existant, avec tentative
de sortie native normale et contrôle du processus possédé.

## Profil CPU de comparaison sans cache

Une sous-classe de recherche, utilisée uniquement dans le runtime isolé de la
campagne, ajoute les options vérifiées dans `--help` :

```text
--cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots
--cache-reuse 0 --no-cache-prompt --spec-type none
```

Chaque requête conserve aussi `cache_prompt=false`, `id_slot=0`, CPU10/GPU0,
batch et microbatch de huit. Aucun cache partagé, checkpoint de préfixe ou
restauration d'état d'agent n'est activé. Ce profil explicite diffère du profil
historique PR20, dont les caches natifs gardaient leurs valeurs par défaut :
les résultats historiques ne constituent donc pas son témoin contemporain.
Le module de production est inchangé.

## Preuves conservées

Le rapport JSON et les journaux par agent conservent :

- la tâche, la mission, l'identité, les messages de décision et les marqueurs
  privés distincts ;
- le rendu exact du gabarit et **les IDs renvoyés par l'appel natif existant à
  `/tokenize`**, sans nouvelle tokenisation ou nouvelle inférence ;
- tous les événements SSE, les IDs et le texte natifs, le terminal EOS/limite,
  les timings natifs séparant préfill et décodage et le temps hôte ;
- les compteurs du PID possédé avant/après chaque requête : lectures comptées
  par macOS, pageins, RSS et empreinte physique, avec leurs limites explicites
  d'attribution ; ces compteurs ne mesurent pas directement les lectures NAND ;
- les états du moteur, les pics RSS échantillonnés, les mesures du watchdog,
  les empreintes des sources et du runtime, le reçu de poids et le journal
  `engine-state/native-server.log` ;
- les refus d'admission, annulations, erreurs et traces partielles, ainsi que
  les observations de fermeture du processus.

La version livrée conserve aussi les détails structurés des erreurs et la
commande construite avant un éventuel échec de chargement. Une observation
opportuniste de l'argv du PID possédé est distinguée de cette construction.
Ces ajouts d'instrumentation sont testés hors modèle après la campagne ; les
deux tentatives matérielles utilisent la version antérieure gelée, avec les
limites de preuve décrites dans `FAILURE_AUDIT.md`.

L'inspection initiale des poids utilise les métadonnées et un reçu SHA associé
à l'identité inchangée des fichiers. Elle ne relit pas le shard de 110 Go sur
batterie. Le chargement natif conserve ses propres vérifications d'intégrité
avant toute génération.

## Critères et limites

`functional_two_agents_qualified` ne devient vrai que si les deux réponses ont
un EOS naturel, un transcript d'IDs complet, un texte final après `</think>`,
les résultats attendus avec une conclusion et justification, leurs propres
marqueurs sans celui de l'autre agent, et si l'intégrité, le secteur, les
gardes et la fermeture sont acceptés. Les prompts natifs et hôtes doivent tous
deux présenter cette isolation. Le texte brut reste disponible à la revue
humaine ; les contrôles de ces deux calculs ne mesurent pas une qualité de
raisonnement générale.

`exact_state_qualified` et `full_qualification` restent **faux** dans ce runner :
les événements du serveur ne capturent pas les logits complets, les routes,
KV/KDA, indexer ou tous les états intermédiaires. Un redémarrage sans cache
recalcule le contexte ; les marqueurs ne prouvent pas l'identité numérique de
ce reset. Une éventuelle restauration de préfixe devra faire l'objet d'une
campagne instrumentée comparée au recalcul complet, incluant annulation et
erreur de restauration. Deux décisions réussies ne qualifient pas 600
inférences et n'établissent pas un gain de débit.

Exécution depuis la racine du dépôt avec un environnement disposant de
`psutil` et une configuration locale épinglée :

```sh
python -m research.glm53_shared_prefill.qualify_agents \
  --config /chemin/vers/cpu10-nocache-context1024.json \
  --output /chemin/vers/nouveau-dossier \
  --tokens 256 --max-seconds 7200
```

Le dossier de sortie doit être nouveau. Le code de retour est non nul lorsque
les deux décisions fonctionnelles ne sont pas qualifiées. Les résultats
matériels effectivement obtenus sont fournis dans le rapport de livraison et
ses preuves, séparément de ce protocole.

L’option `--agent-order 600,1` inverse uniquement l’ordre des deux tâches
complètes. Le défaut reste `1,600`, et les contrôles exigent toujours les deux
identités exactement une fois. La reprise mémoire et ses résultats réels sont
détaillés dans [MEMORY_RECOVERY.md](MEMORY_RECOVERY.md).
