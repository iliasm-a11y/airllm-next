# Récupération mémoire M5 et campagne GLM bornée

3 novembre 2026. **La pression mémoire a été réduite sans fermer les
applications ni supprimer de fichiers. Aucun des agents 1 et 600 n'est
cependant qualifié jusqu'à EOS dans cette campagne.** Les trois essais réels
conservent leurs échecs : délai, swap, puis perte du secteur. Aucun cache de
préfixe ni chemin GPU nouveau n'est activé.

## Diagnostic et réduction effectuée

Le simple classement par RSS masquait une partie de l'occupation. La mesure
`proc_pid_rusage` puis `footprint` a retrouvé **103 processus Node**, dont
13 instances Firebase MCP et 13 Playwright MCP, chacune avec son lanceur npm.
Les quatre groupes représentaient environ 4,01 milliards d'octets d'empreinte
au premier relevé. Ce sont des outils, pas des modèles GLM supplémentaires.

Les processus ont été examinés par identité, propriétaire, ascendance,
commande exacte, ancienneté, variation des compteurs CPU, enfants et
connexions réseau. Seuls les travailleurs feuilles observés depuis plus d'une
heure, inactifs sur l'intervalle, sans connexion Internet ni navigateur enfant,
avec leur lanceur connu, ont reçu SIGTERM. Aucun SIGKILL n'a été utilisé pour
ce nettoyage. Aucun réglage permanent, compte cloud, modèle, document ou
application utilisateur n'a été modifié ou fermé. Ces outils temporaires
pourront devoir être relancés lors d'un prochain usage.

| Réduction | Travailleurs / lanceurs arrêtés | Empreinte antérieure observée |
|---|---:|---:|
| Première phase | 18 / 18 | 2 533 720 192 octets, soit 2,36 Gio |
| Seconde phase, après nouvelle vérification | 8 / 8 | 1 146 997 464 octets, soit 1,07 Gio |
| Total | **26 / 26** | **3 680 717 656 octets, soit 3,43 Gio** |

Cette somme est une **empreinte comptabilisée des processus**, pas une mesure
de 3,43 Gio de RAM physique libre gagnée. La mémoire compressée et les pages
mappées rendent ces compteurs différents. Apple définit l'empreinte comme
l'ensemble mémoire sale/compressée ; la pression dépend de plusieurs facteurs,
dont la mémoire câblée, les caches et le swap. [Apple Developer](https://developer.apple.com/videos/play/wwdc2021/10180/),
[Apple Support](https://support.apple.com/en-euro/guide/activity-monitor/actmntr1004/10.14/mac/27).

Après la première réduction, la pression passe de 2 à **1**, puis reste à 1
pendant les sept relevés d'admission sur trente secondes, swap constant et
secteur confirmé. Après la seconde, la pression est de nouveau à 1 et la
mémoire disponible estimée atteint environ **5,9 Go** avant lancement.
L'accumulation d'outils inactifs est donc un contributeur établi à la pression
initiale. Elle ne permet pas d'attribuer chaque octet de swap ultérieur à un
processus précis ni d'exclure tout autre facteur système.

## Backend conservé et démarrage mieux observé

Le backend CPU est déjà à demande mmap, `MAP_NOCACHE`, sans repack ni warmup,
CPU10/GPU0, sans cache MoE, avec un slot. La lecture des sources ne démontre
aucune copie complète des 110 Go de poids. Le RSS des pages mappées ne doit
pas être confondu avec toute l'empreinte privée du processus.

Malgré `--no-warmup`, le serveur effectue avant son état prêt un contrôle de
capacité : `common_context_can_seq_rm` décode deux tokens fictifs `[0,0]`, teste
le retrait de suffixe, puis vide la mémoire. Ce contrôle peut provoquer des
lectures de poids avant la première requête d'agent. Les anciennes traces
s'arrêtaient après le threadpool ; ce contrôle est un emplacement plausible
de l'ancien échec, pas une pile d'exécution capturée prouvant son instruction
exacte. Il n'a pas été supprimé ni contourné.

Le nouvel observateur externe mesure les ressources globales et les compteurs
du seul descendant natif possédé dès son apparition, avant l'état prêt. Il
conserve PID, identité de démarrage, argv, RSS, empreinte et lectures comptées
par macOS. Il ne mesure pas les lectures NAND physiques. Les gardes du backend
continuent seules de décider les arrêts mémoire ; l'observateur ne les modifie
pas. Les trois chargements atteignent l'état prêt en environ douze secondes.

## Protocole et critères

Les vrais prompts, rôles, mémoires privées et tâches restent complets. Le
contexte est 1 024, le budget de sortie 256 par requête, batch/microbatch 8,
CPU10/GPU0 et un modèle/slot à la fois. Les 600 identités logiques sont
enregistrées dans chaque base isolée. Les flags natifs de cache de prompt restent tous
désactivés. Aucun résumé, réordonnancement du prompt ou texte hôte ne remplace
une sortie du modèle.

RSS **8,5 Gio**, croissance globale de swap **32 Mio**, réserves, seuils et
grâces de pression, intervalle du watchdog, intégrité des shards/bibliothèques,
contrôles PID, annulation et arrêt sont identiques dans les trois configurations
effectivement enregistrées. Une comparaison structurée les vérifie.

Après l'échec sur délai, le plafond de requête passe explicitement de 1 800 à
**2 700 secondes**, avec décision déléguée par l'utilisateur. C'est le seul
plafond temporel de requête changé ; il est déclaré, pas présenté comme
inchangé. Le défaut du pilote reste 1 800 et le backend de production est
inchangé. La fenêtre globale reste **deux heures depuis le premier lancement**,
avec le temps déjà passé déduit des relances et une réserve de nettoyage.
L'option `--request-seconds 2700` est explicite, refuse une campagne supérieure
à 7 200 secondes et respecte une configuration plus stricte déjà fournie.

L'ordre final 600 puis 1 donne la priorité à l'identité qui n'avait encore
atteint aucun calcul natif. Il change l'ordre d'exécution, pas les tâches ni
les messages. Les contrôles exigent toujours exactement les deux identités,
deux EOS naturels et deux réponses finales utiles ; l'ordre ne permet pas une
qualification partielle présentée comme complète.

Deux faux positifs des contrôles de texte ont aussi été corrigés : une réponse
affirmant que le stock ne suffit pas, ou choisissant B tout en affichant les
bons nombres, pouvait passer l'ancien motif trop large. Les tests les
rejettent désormais. Les heuristiques restent conservatrices et le texte
natif exige toujours une revue humaine.

## Résultats réels et échecs

| Essai | Agent(s) réellement démarré(s) | Entrée / sortie natives | Résultat |
|---|---|---:|---|
| 1, après première réduction | 1 | 524 IDs d'entrée / 73 émis | `REQUEST_TIMEOUT`, 1 805,223 s de requête ; pas de terminal EOS |
| 2, plafond explicite de 45 min | 1 | 522 / 74 | `SWAP_LIMIT` après 1 820,561 s de requête ; pas d'EOS |
| 3, après seconde réduction, ordre 600/1 | 600 | 542 / 0 | Annulation secteur vers batterie pendant le préfill ; pas d'EOS |

Le premier essai dure 1 817,553 s au total, chargement/nettoyage compris. Le
premier ID arrive à **1 186,746 s** de requête. La sortie contient les calculs
corrects 408 et 12, mais s'arrête pendant la vérification du stock. Le marqueur
privé et une fin utile complète ne sont pas établis. Les gardes mémoire ne
signalent aucun échec sur ce passage. Il n'est pas présenté comme une réponse
complète ni comme un débit qualifié jusqu'à EOS.

Le deuxième dure 1 832,313 s, avec premier ID à **1 197,104 s**. La garde
échantillonne une hausse globale de **44 761 088 octets, soit 42,6875 Mio**,
au-dessus de 32 Mio. Le swap continue ensuite de monter pendant le nettoyage ;
ce maximum global ultérieur n'est pas attribué au seul moteur. Le relevé des
processus après échec ne montre pas de forte croissance durable d'une
application particulière. Il ne reconstitue pas tous les événements transitoires :
**la cause exclusive de cette hausse tardive n'est pas isolée**.

Le troisième dure 688,906 s. Pendant sa fenêtre mesurée, la pression reste
**1**, le swap global reste constant à **1 381 564 416 octets**, et le préfill
atteint au moins 304 tokens au dernier relevé de progression. La surveillance
observe la bascule **AC vers batterie le 3 novembre à 11:07:25,964 UTC, soit
12:07:25,964 à Paris**. Elle annule la requête et arrête le processus possédé.
Le couvercle est ouvert et les horloges sont cohérentes ; la batterie est le
motif du refus de continuité. La pression normale sur ces onze minutes ne
qualifie pas une génération soutenue jusqu'à EOS.

Les pics RSS natifs échantillonnés sont respectivement **5 141 217 280**,
**5 013 946 368** et **5 784 846 336 octets**, tous sous 8,5 Gio. Les sorties
natives observées à **−9** relèvent du nettoyage après interruption ; aucune
sortie normale à zéro n'est revendiquée. Les processus natifs possédés et les
assertions de veille sont arrêtés. Les empreintes des sources gelées sont
inchangées à la clôture de chaque essai.

Les compteurs natifs `requests/generated_tokens` restent nuls pour ces requêtes
interrompues car ils sont mis à jour après un terminal complet. Ils ne doivent
pas effacer les **73 et 74 IDs réellement observés** dans les journaux SSE.
Les journaux complets, messages, rendus, IDs d'entrée, traces partielles,
timings, erreurs et sources mesurées sont conservés séparément.

## Conclusion et livraison

La réduction a permis de dépasser l'ancien échec de chargement et, après
la seconde phase, de garder la pression normale pendant le préfill observé
de l'agent 600. Elle ne démontre pas une stabilité universelle du modèle sur
16 Gio. **Zéro agent de cette campagne est qualifié jusqu'à EOS**, et aucun
nouveau comparatif numérique logits/routes/KDA/mHC ou gain de débit n'est
inventé. Le cache exact partagé reste désactivé ; CPU10/GPU0 reste la production.

La suite portable du pilote passe **23 tests**, incluant faux positifs de
conclusion, budget temporel explicite, borne globale, ordre 600/1, gardes
inchangées, PID, annulation et batterie. La sélection publique de preuves
conserve les rapports natifs, journaux, séries de ressources et sources gelées
compressés sans perte. La livraison locale contient en plus les relevés
complets de propriété des processus. Ces relevés locaux ne sont pas publiés
comme inventaire des applications de l'utilisateur.
