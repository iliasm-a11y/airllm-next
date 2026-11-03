# Audit des deux tentatives matérielles, 3 novembre 2026

La tentative `two-agents-cpu10-1` est un **échec de chargement sous garde de
swap**, avant toute requête d'agent. Sa durée murale totale est **8,006217 s** ;
elle inclut les préparatifs et la fermeture. Ce n'est pas une mesure isolée du
chargement, du préfill ou du décodage.

Le rapport commence à 09:21:31 UTC et la dernière observation d'alimentation
date de 09:21:39 UTC, soit 10:21:31–10:21:39 à Paris. Les 18 observations
d'alimentation/horloge et l'assertion de veille possédée sont acceptées :
secteur connu, couvercle ouvert, aucune discontinuité détectée aux instants
échantillonnés. L'échec ne doit donc pas être attribué à la batterie.

## Déclenchement et intégrité

Les six échantillons natifs montrent une pression mémoire passant de 1 à 2.
Le swap global passe de 1 302 003 712 à 1 341 849 600 octets, soit une hausse de
**39 845 888 octets = 38 Mio**, au-delà de la limite conservée de 32 Mio.
Le code d'arrêt est `SWAP_LIMIT`. Les 4,022 secondes de pression d'avertissement
ne sont pas assimilées à un franchissement du délai de pression de 20 secondes.
Le RSS maximal échantillonné est **3 829 415 936 octets**, environ **3,567 Gio**,
donc inférieur à son plafond de 8,5 Gio. Aucune limite n'a été supprimée ou
augmentée pour poursuivre le chargement.

Les compteurs de swap sont globaux : cette observation n'attribue pas chaque
octet de swap au seul serveur et ne prouve pas que le contexte de 1 024 tokens
échouerait sur tout état mémoire du Mac. Elle suffit à refuser cette tentative
dans les ressources réellement disponibles.

Les deux shards sont reconnus et leur reçu SHA lié à l'identité inchangée des
fichiers est accepté. Le runtime et ses 29 artefacts empaquetés, totalisant
25 671 824 octets, sont également vérifiés. Ces faits sont dans
`engine_after_cleanup.native` et le préflight. Les valeurs `weights_integrity`
et `runtime_integrity` de `functional_checks` sont fausses car cette section
exige un état **après génération**, qui n'existe pas ici ; elles ne constituent
pas un constat de corruption des poids ou bibliothèques. Le statut d'échec et
les trois drapeaux de qualification faux sont corrects.

## Arrêt et absence de résultat d'agent

Le processus natif possédé **6046** est arrêté par le watchdog avec SIGTERM.
Son code **−15** est observé via le handle possédé et conservé avec
`source=resource_guard`, `guard_failure_code=SWAP_LIMIT` et
`exit_observed=true`. Il n'est pas présenté comme une sortie normale à zéro.
L'assertion possédée **6037**, rattachée au lanceur **6023**, est elle aussi
nettoyée. Une vérification ultérieure limitée à ces trois PID ne trouve aucun
processus ; la commande, l'instant et le résultat sont conservés dans
`failure-audit.json` à côté des preuves brutes de la tentative.

Le registre contient bien 600 identités. Les compteurs indiquent une tentative
hôte de chargement, **zéro chargement natif achevé, zéro tentative de
génération, zéro requête et zéro token produit**. Aucun message de mission n'a
atteint le modèle. Il n'existe donc pour cette tentative ni IDs de prompt
natif, ni timings de préfill/décodage, ni réponse finale/EOS, ni comparaison
nouvelle de logits/routes/états. Les prompts tokenisés hors inférence dans
l'audit de préfixe sont une preuve séparée et ne doivent pas être attribués à
ce serveur arrêté.

## Limites de l'instrumentation relevées

Le fichier de configuration effectif indique CPU10/GPU0, contexte 1 024,
batch 8. Le journal natif confirme `n_threads=10`. Les options explicites de
désactivation des caches sont acceptées par l'aide du binaire épinglé et sont
ajoutées par la sous-classe de recherche dont l'empreinte est enregistrée :

```text
--parallel 1 --gpu-layers 0 --threads 10
--cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots
--cache-reuse 0 --no-cache-prompt --spec-type none
```

**La ligne d'arguments effectivement transmise au processus n'est toutefois
pas enregistrée dans cette tentative.** Le runner n'écrivait le champ
`native_command` qu'après un chargement réussi ; ce point est identifié comme
une lacune de preuve, pas remplacé par une affirmation d'observation d'argv.
De même, l'observateur I/O par PID est créé après un chargement réussi : aucun
comparatif de lectures par processus n'est disponible ici. Les six
échantillons de ressources et la trace d'arrêt restent bruts et exploitables.

Une éventuelle seconde tentative doit avoir un dossier et une admission
distincts. Le premier échec reste conservé sans réécriture et sans diminution
des gardes ni raccourcissement des contextes.

## Deuxième tentative : refus avant création du processus natif

La tentative distincte `two-agents-cpu10-2` commence à **09:23:42 UTC**
(10:23:42 à Paris). Elle est refusée avec `MEMORY_ADMISSION`, message
`Insufficient safe mmap workspace`, après **1,903052 s de temps mural total**.
Le reçu de poids et les artefacts restent vérifiés ; l'alimentation secteur
et le couvercle sont acceptés par les cinq observations de la campagne.

Les sept observations de la fenêtre externe `resource-window.jsonl`, espacées
d'environ cinq secondes entre 09:22:20 et 09:22:50 UTC, montrent toutes une
pression mémoire **2** avec swap global inchangé à 1 341 849 600 octets et
environ 4,90 à 4,99 milliards d'octets de mémoire disponible estimée. Cette
fenêtre ne remplace pas l'instant d'admission exact du deuxième lancement.
La source gelée montre cependant qu'un **chargement à froid refuse dès la
pression 2**, même si l'estimation de mémoire disponible est supérieure à la
réserve et au workspace calculés. Ce comportement a été conservé.

La première version du runner ne sauvegardait pas les `details` de l'exception
d'admission. Le rapport brut ne donne donc pas l'échantillon exact qui a causé
ce deuxième refus. Il faut conserver cette limite au lieu d'attribuer à cet
instant l'une des sept valeurs de la fenêtre antérieure.

Aucun PID natif n'est enregistré ou observé pour cette tentative ; les
échantillons du watchdog sont vides et les compteurs de chargement natif et de
génération restent nuls. L'absence de processus est cohérente avec un refus
avant `Popen`. Le lanceur **6280** et son assertion possédée **6303** sont
nettoyés et une lecture ultérieure de ces deux PID ne trouve aucun processus.
Cette tentative ne fournit donc, elle non plus, aucun préfill, token ou EOS.
La campagne matérielle est close après **un processus natif démarré puis arrêté
par garde, et un refus d'admission** ; aucun troisième lancement n'est effectué.

## Version mesurée et correction ultérieure de l'instrumentation

Les deux rapports matériels correspondent au runner gelé de SHA-256
`47b94abd560526967df210ca1404ae5f8f63b3a7e9f43ec10df2b31bd7899b08`,
conservé dans la livraison `sources-frozen` avec le manifeste des autres
sources et le commit de départ `a980f69c40e9170aef411fb4333613d2f7c6415a`.
Les preuves brutes de ces deux tentatives ne sont pas réécrites.

Après clôture, une correction d'instrumentation uniquement ajoute :

- les `error_details` globaux et par requête, ainsi que les détails d'une erreur
  native avant son passage par les exceptions hôtes ;
- une copie de la commande **construite**, y compris si le chargement échoue,
  explicitement distinguée d'une preuve de création du processus ;
- une lecture opportuniste de l'argv du seul processus `Popen` possédé, avec
  contrôles de son identité, de son exécutable et de sa durée de vie avant et
  après la lecture, sans énumération de processus ni signal supplémentaire.

Cette version livrée du runner a pour SHA-256
`2bbb89a061c1dd246bac4b87e45b51bdadf3239dbc6ccefcb924c794adcc2112`.
**Elle n'a pas été exécutée avec le modèle.** Dix tests portables passent,
notamment conservation des détails d'erreur, échec en cours de requête avec
terminal nul, restauration des méthodes et de l'environnement, et refus d'une
observation d'argv dont le PID ou l'exécutable change. Ces tests ne créent aucun
modèle ou processus natif et ne sont pas une nouvelle preuve matérielle.
