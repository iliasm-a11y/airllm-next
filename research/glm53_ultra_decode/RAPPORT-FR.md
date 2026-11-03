# Décodage GLM-5.3 Flash sur M5 — reprise du 3 novembre 2026

**La campagne a repris ; aucune nouvelle accélération complète n'est qualifiée.**
Deux captures contemporaines CPU10 de32 sorties passent la comparaison des
logits, routes, KDA et écritures de cache observées. Le témoin sans captures
termine32 sorties :31 appels en188,084548 s, soit **0,164819 intervalle/s**.
Les deux essais suivants de64 sorties sont arrêtés par la pression mémoire,
avant toute sortie puis après17 IDs malgré une admission au-dessus de6 Gio.
128 reste non exécuté. Les protections restent actives.

## Périmètre conservé

Machine M5 GPU8/CPU10,16 Gio, checkpoint original Q2_K_S d'environ110 GB,
deux shards inchangés. La branche séparée part de PR23
`6bdeec357154a7740833b4ad694d08e02aef2f96`. Seuls ce nouveau dossier de recherche
et un workflow indépendant changent. Aucun fichier du préfill, de RONIN ou du
runtime installé n'est modifié ; aucune fusion dans `main` n'est effectuée.
CPU10/GPU0, effort Max,600 identités logiques et un seul slot physique restent
conservés. Le travail préfill de l'autre chat a été attendu sans interruption.

Les profils utilisent le prompt brut déterministe « La capitale de la France
est », contexte512, batch/ubatch8, une séquence, CPU10/GPU0. Ils ne remplacent
pas les prompts complets Max ni les sessions RONIN1024.32 sorties désignent32
IDs, dont le premier provient du préfill :31 transitions sont chronométrées.
Aucun EOS naturel n'est atteint.

## Séquence réelle, échecs conservés

| Essai | Résultat | Portée |
|---|---|---|
| Quatre composants B4, deux graines/forme | Fidélité et provenance conformes ; Metal plus lent | Poids synthétiques résidents |
| Premier diagnostic CPU32 B3 | Réserve disque franchie après16 IDs | Arrêt, aucun débit accepté |
| Référence CPU32 B3 |32 IDs,31 appels, gardes conformes | Diagnostic sérialisé |
| Capture B4 suivante | Admission RAM refusée | Aucun modèle lancé |
| Capture CPU32 B4 reprise |32 IDs, comparaison numérique conforme | CPU ciblé uniquement |
| Témoin CPU32 B4 sans captures |31 appels terminés en188,084548 s | Temps du backend, pas réponse RONIN |
| Témoin CPU64 B4 | Pression2 après4,076 s, zéro ID | Échec, aucune mesure de débit |
| Reprise CPU64 après restitution du cache | Pression2 après133,074 s,17 IDs | Échec, préfixe17 identique uniquement |
| CPU128 de cette séquence | Non exécuté | Arrêt au premier échec |

Une régénération extérieure de cache d'outils avait réduit la réserve disque
pendant le premier diagnostic. Cet arrêt n'est pas effacé par les reprises.
Les poids et bibliothèques passent les contrôles après les arrêts et les
essais terminés.

<!-- RESOURCE_RETRY_START -->
Une application a ensuite été fermée avec autorisation explicite et macOS a
restitué du cache :6,64 Gio disponibles ont été observés. La reprise exigeait
6 Gio disponibles et33 Gio libres en plus des gardes d'origine. Le CPU64 a
été admis avec6 538 641 408 octets disponibles, mais la pression est repassée
à2 après17 sorties. Les17 IDs concordent avec le préfixe CPU32 validé ;
aucun débit de cet essai interrompu n'est accepté.128 n'est pas lancé.
La restitution du cache change son état initial ; aucun résultat comparatif
de performance n'est tiré de cette reprise.
<!-- RESOURCE_RETRY_END -->

## Fidélité CPU32 : seuils d'origine inchangés

B3 et B4 partagent poids, bibliothèques, prompt et paramètres. Critères CPU :
bit à bit, atol=rtol=0, routes ordonnées exactes et valeurs finies.

| Donnée | Couverture | Différences |
|---|---:|---:|
| Logits complets |32 vecteurs,4 956 160 flottants |0, y compris les bits |
| Routes |1 344 tenseurs,12 432 entiers |0 |
| Routeur |1 344 tenseurs,447 552 flottants |0 |
| KDA |2 176 tenseurs,20 611 072 flottants |0 |
| result_output |32 tenseurs,4 956 160 flottants |0 |
| Écritures CPY/SET_ROWS observées |7 584 événements ordonnés |0 hash différent |

Les32 IDs sont identiques. Les écritures représentent5 253 365 760 octets
logiques et5 345 460 224 octets de plages de stockage. Cette preuve concerne
les écritures observées, pas tout état non capturé. Elle ne qualifie ni les
continuations longues, ni les réinitialisations, ni les changements d'agent.
Le témoin sans captures suit les mêmes32 IDs ; ses logits et états non capturés
ne sont pas déclarés comparés. Une continuation longue ne reçoit pas
automatiquement la qualification numérique de cette trajectoire courte.

## Temps du témoin : dénominateur et limites

Les31 intervalles incluent `llama_synchronize` avant la fin de chaque appel.
Ils excluent chargement, préfill/première sortie, sélection gloutonne et gestion
extérieure à l'appel. Les203,662 s du superviseur incluent d'autres phases et
ne sont pas le dénominateur. Une fixture C++ lance un travail réellement
asynchrone et vérifie sa terminaison avant l'arrêt du chronomètre.

Les compteurs de ces31 appels donnent379 852 898 304 octets lus par le processus,
zéro octet écrit,23 330 040 pageins et1 445,090 CPU core-seconds. Ce sont des
compteurs OS, **pas des lectures physiques NAND**. L'attente I/O pure et les
allocations non mesurées restent `null`.

Le témoin est unique, sans A/B/B/A contemporain d'un moteur accéléré.
0,164819 ne démontre aucun gain face à une ancienne campagne. Aucun débit
de réponse utile RONIN ni stabilité multiagent n'est dérivé de ces appels.

## Goulot contemporain instrumenté

La capture B4 sérialise les callbacks. Préfill exclu, les229 803 opérations
de ses31 transitions donnent :

| Famille | Temps d'opérations | Part |
|---|---:|---:|
| MUL_MAT dense |141,755186 s |66,394 % |
| MUL_MAT_ID MoE |40,105353 s |18,784 % |
| Autres |31,645074 s |14,822 % |
| Total |213,505614 s |100 % |

8,144678 s d'inspection sont mesurées séparément et exclues de ce tableau.
Le B3 n'en mesurait pas la durée : ses champs à zéro signifient « non mesuré ».
Le temps total des appels de diagnostic B4 est225,642736 s ; il ne constitue
pas un débit sans instrumentation.

Les instantanés `mincore` préalables couvrent23 170 986 pages cumulées, dont
34 106 résidentes. Ce sont des observations répétées sur les opérandes
inspectées, pas des pages uniques ni une mesure SSD. Le diagnostic étaye le
poids des produits denses et des accès aux poids. Il ne sépare pas causalement
stockage physique, cache OS et calcul. Les CPU core-seconds cumulent les
threads : leur soustraction au temps mural ne donne pas l'attente I/O.

## Kernels Metal : fidélité ciblée, coût total défavorable

Les kernels un token Q8_0/Q2_K gardent les formats et l'ordre du dot ARM de
référence. Les activations sont quantifiées par la bibliothèque CPU native ;
aucun format FP16 intermédiaire de poids n'est ajouté. Tuiles1/2/4 et deux
banques originales peuvent partager un lancement. Le graphe CPU GGML
persistant utilise un seul pool10 partagé pour une paire.

Mesures B4 :3 chauffes,21 répétitions/variante, deux graines par forme.
Buffers, pipelines et petits poids synthétiques sont résidents.

| Composant | CPU10 médian | Meilleur Metal total médian | GPU seul correspondant |
|---|---:|---:|---:|
| Paire Q2 M2048/K4096, graine5303101 |0,090125 ms |0,315000 ms, tuile4 |0,123875 ms |
| Paire Q2, graine5303102 |0,088708 ms |0,306625 ms, tuile4 |0,120750 ms |
| Q8 M1536/K4096, graine5303101 |0,032375 ms |0,500291 ms, tuile1 |0,239375 ms |
| Q8, graine5303102 |0,034709 ms |0,545167 ms, tuile2 |0,269875 ms |

Le total Metal comprend quantification CPU, encodage, lancement et attente.
Préparation et imports initiaux sont exclus des deux régimes chauds. Les
50 176 composantes finales, activations quantifiées et contrôles CPU1/CPU10
passent. Aucune sortie GLM n'est comparée par ce banc.
Metal reste3,46–3,50 fois plus lent pour Q2 et15,45–15,71 fois pour Q8.
Aucun gain complet n'est extrapolé. Le premier banc à deux pools CPU10
concurrents et l'erreur initiale du shader restent rejetés et conservés ;
les anciennes mesures B3 gardent leur propre version.

## Gardes, provenance et correction du rapport d'arrêt

Admission fixe :5,5 Gio disponibles,32 Gio libres, pression normale, secteur,
capot ouvert et horloges cohérentes. En cours d'essai : RSS échantillonnée de
l'arbre possédé≤8,5 Gio, réserve système3 Gio, hausse du swap global≤32 Mio,
durée≤2 400 s, propriété des processus et verrou UID transmis au modèle.
Les gardes échantillonnées peuvent manquer une crête entre deux observations.
Le swap global inclut les autres applications.

Le point de départ commun du swap,2 711 814 144 octets, est conservé entre les
reprises. Son contrôle peut seulement réduire le budget32 Mio par essai ;
il n'est pas réinitialisé pour masquer une hausse cumulée.

| Essai | Pic RSS de l'arbre | RAM disponible minimale | Hausse maximale du swap | Pression maximale |
|---|---:|---:|---:|---:|
| CPU32 référence B3 |7 050 477 568 octets |4 254 334 976 |0 |1 |
| CPU32 capture B4 |7 649 329 152 |4 181 147 648 |196 608 octets |1 |
| CPU32 temps seul |6 670 516 224 |4 227 137 536 |1 048 576 octets |1 |
| CPU64 arrêté |4 756 701 184 |3 425 501 184 |0 |2, arrêt |
| CPU64 repris, arrêté |6 835 044 352 |3 423 731 712 |0 |2, arrêt |

Les trois essais CPU32 passent aussi les contrôles d'alimentation et d'intégrité.
Le CPU64 n'a pas de rapport final du worker : ses marqueurs natifs prouvent
pourtant son lancement et le début du préfill. Son ancien champ
`native_process_launched:false` est donc erroné. Le brut est conservé avec une
correction séparée. Le correctif écrit maintenant la preuve de lancement de
façon atomique et rapporte `null` si l'état reste inconnu après un arrêt.
Un worker non finalisé ne peut pas valider un essai.
Le second arrêt CPU64 confirme ce correctif en situation réelle : lancement
enregistré àtrue, worker non finalisé, échec conservé, aucun débit accepté.

Le reçu SHA256 courant est réutilisé après concordance des identités des deux
shards ; aucun nouveau hash complet de110 GB n'est revendiqué. Les dix
bibliothèques V4 et les binaires mesurés restent inchangés. Le manifeste B4 lie
sources, includes, compilateur et bibliothèques. SHA256 du probe B4 :
`fd4bea9c1bbbe226457f8d1fcdb36936a8b4824492d27a0bbc459e8287431e6b`.

48 tests locaux passent, dont budget cumulatif du swap, preuves après arrêt et
terminaison asynchrone réelle. La [CI portable](https://github.com/iliasm-a11y/airllm-next/actions/runs/41721027098)
passe sur le correctif de supervision ; elle ne mesure ni Metal ni le modèle M5.

## Configuration et douze verdicts

L'ordonnanceur expérimental reste désactivé, sans branchement en production.
Il exige fidélité selon les seuils existants, résidence/provenance valables et
deux cycles A/B/B/A avec gain total≥10 % à chaque cycle. Aucun candidat n'y est
admis. Aucun nouveau cache permanent, chemin TensorOps ou décodage spéculatif
de modèle n'est qualifié.

1. **Goulot réel :** produits denses et accès aux poids,66,4 % du temps
   d'opérations sérialisées contemporain ; partage NAND/cache/calcul non isolé.
2. **Gain complet nouveau :** aucun. Le témoin CPU32 donne0,164819 intervalle/s
   sur un périmètre précis, sans accélération relative démontrée.
3. **GPU complet :** non qualifié ; aucune trajectoire accélérée validée.
4. **Q8/Q2 utiles :** fidèles dans les composants testés, mais coût total
   supérieur à CPU10 ; aucun placement nouveau retenu.
5. **Poids :** aucune économie de génération complète démontrée par un nouveau
   cache ou ordonnanceur.
6. **Spéculation exacte :** aucun gain net mesuré ni nouvelle implémentation
   de modèle qualifiée.
7. **Stabilité :** CPU32 ciblé passe ; les deux CPU64 s'arrêtent sur pression.128, EOS,
   longues continuations et charge multiagent restent non qualifiés.
8. **Configuration :** CPU10/GPU0, Max inchangé,600 identités logiques,
   un modèle/slot physique ; aucun backend nouveau activé.
9. **0,5 token/s :** non atteint ni démontré.
10. **1 token/s :** non atteint. Une impossibilité générale n'est pas prouvée ;
    les compteurs OS ne sont pas une borne NAND physique mesurée.
11. **600 agents rapides :** préfill, décodage, file séquentielle et stabilité
    demeurent les obstacles ; deux réponses utiles Max ne sont pas qualifiées.
12. **Pilote :** le nouveau moteur accéléré n'est pas prêt. Le helper CPU a une
    qualification capturée32 ciblée, pas une qualification de production RONIN.

Les [preuves contemporaines](evidence/resumed/selection.json), y compris les
échecs, complètent les preuves antérieures. Les tableaux par opération,
couche et étape sont conservés dans les rapports sélectionnés. Les tableaux
volumineux et tenseurs bruts restent locaux. Aucun poids ni donnée personnelle
d'application n'est publié.
