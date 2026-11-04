# GLM-5.3 Flash : première qualification n-gram sur M5

Branche `research/glm53-ngram-speculation-m5`, issue de la PR #25 au commit
`195f7745afc3d0ca3fc8aca776287264e16f3a2b`. Production et `main` inchangés.

## Ce qui est implémenté

Le proposeur n-gram simple original de llama.cpp est compilé sans modification.
Il n'utilise que les IDs de l'historique de la requête courante. Aucun second
modèle n'est chargé et aucun poids GLM n'est converti. Le modèle physique et
le contexte restent uniques. Les bibliothèques CPU V4 existantes sont conservées,
avec vérification de leurs empreintes avant et après chaque essai.

Le premier candidat anticipe deux tokens et vérifie trois entrées en un passage
(le token déjà connu, puis les deux propositions). Les formes intermédiaires
ne sont pas activées. Après un rejet, un snapshot complet est restauré et le
préfixe accepté est rejoué séquentiellement. Le temps de ces opérations est
inclus dans le chronométrage. La voie reste réservée à la recherche, désactivée
dans le moteur RONIN installé.

## Fidélité réellement exécutée

Le binaire final de cette campagne est `ngram-build4/ngram-probe`, lié aux mêmes
bibliothèques natives que la référence B4 antérieure. Le nouveau binaire a sa
propre qualification ; il n'hérite pas des seuls résultats historiques.

Sur le checkpoint original complet et le prompt brut court « La capitale de la
France est », trois fixtures ont été exécutées avec les mêmes entrées en
séquentiel puis en lot :

| Fixture | Propositions acceptées | Rejeux après restauration | Résultat |
|---|---:|---:|---|
| Toutes correctes | 2/2 | 0 | Identique à la référence |
| Rejet immédiat | 0/2 | 1 appel | État et token suivant identiques |
| Rejet partiel | 1/2 | 2 appels | État et token suivant identiques |

Pour chaque fixture : les trois vecteurs complets de154880 logits sont
identiques bit à bit, ainsi que354 comparaisons de tenseurs (118 noms GLM par
position : routes ordonnées, routeurs et sorties KDA). Les90 banques canoniques
d'état attendues sont couvertes et leurs écritures observées sont finies. Le
hash de la sérialisation complète de séquence après le lot correspond à celui
obtenu sur les mêmes entrées séquentielles. Après les rejets, le hash d'état du
préfixe accepté et son prochain argmax sont exacts.

Ces propositions sont volontairement correctes ou erronées pour contrôler le
mécanisme. Elles sont des fixtures oracle, **pas une mesure de réussite du
proposeur n-gram**. Cette couverture ciblée sur un préfixe court ne qualifie
pas tous les contextes, le sampling non glouton, les600 agents ou un EOS naturel.

## Mémoire, verrou et tentatives conservées

Le verrou UID/modèle était libre au début du développement. Un autre essai de
préfill s'est ensuite lancé dans le chat « Qualifier le préfill partagé GLM-5.3 ».
Sur autorisation explicite de l'utilisateur, ce chat a été prévenu et son
superviseur identifié a reçu SIGINT. Il a terminé son auxiliaire par SIGTERM,
écrit un reçu `cancelled` et conservé ses diagnostics. Aucun fichier de verrou
n'a été supprimé et cet arrêt n'est pas un échec numérique du préfill.

Les outils Node observés inactifs ont reçu SIGTERM. Ceux montrant une activité
CPU, réseau, I/O ou des enfants ont été conservés. Claude a été quitté sur
autorisation explicite de l'utilisateur. Les empreintes de processus et les
variations de mémoire disponible sont rapportées séparément ; elles ne sont
pas assimilées à une quantité causale de RAM physique libérée.

Un conseil `MADV_DONTNEED` sur des mappings de fichiers en lecture seule peut
aider Darwin à rendre d'anciennes pages du checkpoint récupérables. Le helper
ne parcourt pas le payload, vérifie l'intégrité et détient le verrou UID. Ce
n'est ni une garantie de cache froid ni une admission de modèle. Une fenêtre
mesurée de2 secondes au-dessus du seuil5,5 Gio est désormais exigée avant le
lancement ; l'attente est plafonnée à60 secondes, sans relancer un modèle.

Les seuils restent : RSS8,5 Gio, croissance du swap global32 Mio, réserve
disque32 Gio, mémoire disponible5,5 Gio, pression normale, secteur, capot ouvert,
horloges et identité des processus surveillées. Chaque bras de la campagne
utilise la même préparation de cache et la même origine de swap.

Les tentatives suivantes restent distinctes des résultats qualifiés :

- Premier build : dépendances de journalisation non liées ; corrigées par un
  pont silencieux, sans réécriture du proposeur.
- Premier audit du modèle : valeurs numériques et sérialisation exactes,
  mais refus du contrôleur qui comptait169 noms de vues au lieu des90 banques
  canoniques. Le comptage a été corrigé, sans assouplir les critères numériques,
  et les trois fixtures ont été entièrement réexécutées sur le nouveau binaire.
- Premières préparations de campagne : aucun modèle chronométré lancé. Une
  erreur de lecture du manifeste à trois champs, puis un refus mémoire, ont
  été conservés. La lecture du manifeste et l'attente d'admission sont testées.

## Opportunités historiques et coût attendu

Le rejeu de la trajectoire CPU32 du4 novembre ne charge aucun modèle. Pour un
n-gram de taille2, il trouve une proposition de deux tokens dont aucun ne
correspond au futur observé. Avec une taille1, cinq propositions totalisent
dix tokens dont un seul correspond. Ce ne sont ni des acceptations réellement
vérifiées en lot ni des mesures de vitesse.

Ce témoin court est donc défavorable au n-gram. Il reste utile pour mesurer
le coût d'un rejet réel et contrôler la trajectoire. Une éventuelle réussite
sur un motif répétitif ne pourrait pas être transférée à un raisonnement
RONIN sans une qualification séparée sur ses sorties utiles.

## Campagne chronométrée

Les quatre bras A/B/B/A de32 sorties sont terminés, sur le même binaire et les
mêmes bibliothèques, paramètres et checkpoint. Les128 IDs observés correspondent
à quatre répétitions de la même trajectoire de32 IDs, également identique au
CPU32 antérieur. Aucune trajectoire n'atteint EOS.

| Bras | Mode | Temps des31 intervalles | Intervalles/s | Proposés / acceptés | Appels / positions du modèle |
|---|---|---:|---:|---:|---:|
| A1 | CPU classique | 178,656467s | 0,173517 | 0 / 0 | 31 / 31 |
| B1 | n-gram2, deux anticipés | 196,903692s | 0,157437 | 2 / 0 | 32 / 34 |
| B2 | n-gram2, deux anticipés | 199,942866s | 0,155044 | 2 / 0 | 32 / 34 |
| A2 | CPU classique | 203,327797s | 0,152463 | 0 / 0 | 31 / 31 |

Temps moyen :190,992132s pour CPU classique contre198,423279s pour n-gram,
soit **+3,89 % de temps observé**. Le débit agrégé est0,162310 contre0,156232
intervalle/s. Cette moyenne ne constitue pas une conclusion statistique : les
deux témoins varient de13,81 %, et les deux rapports de vitesse sont
contradictoires (0,907329 puis1,016929). **Aucun gain reproductible n'est établi.**

Dans chacun des deux bras n-gram, une seule proposition de deux tokens est
faite ; les deux tokens sont rejetés. Les compteurs confirment un rejeu, un
appel et trois positions supplémentaires du modèle. Il n'y a donc **aucune
réutilisation productive d'une proposition acceptée** sur ce prompt.

Les quatre sorties natives sont0. Pic RSS surveillé maximal :7474610176 octets,
soit6,96 Gio ; l'audit atteint7,08 Gio. La croissance observée du swap global
reste0, avec pression normale, contrôles secteur/capot/horloges passés et
identités des bibliothèques et du checkpoint inchangées. Le swap préexistant
n'est pas nul. Le verrou UID est vérifié libre après la campagne.

Les mesures portent sur31 intervalles au maximum, après préfill et premier
token. Le préfill reste enregistré séparément. Ce périmètre n'est pas une
mesure d'une réponse RONIN utile de bout en bout.

| Bras | Préfill | CPU, secondes-cœur pendant la génération | Lectures OS du PID pendant la génération |
|---|---:|---:|---:|
| A1 | 13,510266s | 1382,646 | 379852394496 octets |
| B1 | 12,979278s | 1459,605 | 395488776192 octets |
| B2 | 13,139064s | 1490,063 | 395492188160 octets |
| A2 | 13,403107s | 1582,264 | 379853271040 octets |

Les secondes-cœur additionnent les temps des threads et peuvent dépasser le
temps mural. Les octets OS ne sont pas des octets NAND physiques. Le temps
d'attente I/O pur n'est pas isolé ; les snapshots et rejeux sont inclus dans
le temps de génération, sans leur attribuer une durée déduite par soustraction.
Le GPU n'est pas demandé par cette configuration CPU.

## Décision

Conserver **CPU10/GPU0 classique, spéculation désactivée** dans RONIN. Le
prototype n-gram est implémenté, ses contrôles ciblés d'acceptation/rejet passent,
et la comparaison complète32 est exécutée. Il n'a pas satisfait le critère de
gain reproductible. La variation entre témoins reste un facteur non isolé ;
elle ne doit pas être attribuée sans mesure au SSD, à la température ou à macOS.

Quatre tokens anticipés ne sont pas chronométrés : le rejeu historique ne montre
aucun token correct pour cette variante sur cette trajectoire, et le premier
candidat n'accepte déjà rien. Le code exige son propre audit avant de permettre
une telle mesure.64/128 sorties, EOS, prompts Max complets, sampling non glouton
et charge des600 agents ne sont pas qualifiés par cette campagne.

Le résultat ne démontre pas que le n-gram est inutile pour tout GLM. Il montre
que ce témoin court ne justifie pas son activation. Une nouvelle campagne
devrait partir de continuations RONIN présentant des répétitions mesurées,
avec une hypothèse de rentabilité, avant de multiplier les chargements.

## Reproduction et limites

Voir `README.md`, `build.py`, `run.py`, `prepare.py`, `campaign.py` et `compare.py`.
Un seul échec ou refus arrête une campagne ; les bras réussis ne sont pas
sélectionnés après coup. Les comparaisons n'acceptent ni une trajectoire
différente, ni un EOS différent, ni des bibliothèques ou un checkpoint différents,
ni un essai incomplet ou arrêté par un garde-fou.

Les37 tests locaux de contrat passent, dont les fixtures compilées du proposeur
original, les refus de qualification, la comparaison des bras et le maintien
des fichiers en lecture seule. Ces tests ne remplacent pas les essais GLM.
