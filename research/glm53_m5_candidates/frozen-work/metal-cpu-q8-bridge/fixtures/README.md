# Fixtures natives du provider — préparées, non compilées, non exécutées

Ces sources complètent le gel du provider relu ; elles ne changent aucun de ses
sept fichiers, son runner, ses bibliothèques, ni un modèle actif.

Chaque paire lance d'abord le graphe CPU10 ordinaire avec bridgeOFF, puis le
même binaire et les mêmes bibliothèques avec bridgeON, dans un autre processus.
`use_ref` est explicitement false ; une référence scalaire n'est pas substituée
au dispatcher natif. Les poids sont quantifiés ligne par ligne par le pointeur
natif Q8_0 `from_float`, puis exposés par un vrai buffer `CPU_Mapped` marqué
WEIGHTS. Une seule ligne FP32 de16Kio est nécessaire, sans matrice FP32 entière.
Les SHA des vrais octets quantifiés et des entrées doivent être identiques entre
les deux processus. Les sorties float32 brutes sont finies et vérifiées par
lecture bornée, SHA et comparaison de chaque bit, y compris le zéro signé.

Les tolérances Metal restent exactement atol1e-3/rtol1e-4, séparées du verdict
bit à bit CPU. Un écart de bits dans une sortie GPU, même inférieur à la
Tolérance, demeure un écart du contrat CPU bit à bit. Les replis CPU doivent
être bit à bit identiques au CPU ordinaire du même package.

## Couverture prévue et limites

Les23 paires comprennent :

- M8192, K2048 puis4096, chacun àN8/16/32/64/128 (six tuiles de lignes).
- M1536/K4096 àN8/32/128 et M2048/K2048 àN32 (une et deux tuiles).
- Queues N1/6/14, M1536, pour les deux K : zéro commande GPU.
- Échelles FP16 subnormales sur les activations puis sur les poids : refus
  explicite avant soumission et résultat CPU exact.
- Échelle subnormale au premier rang de la seconde tuile M2048/K4096/N8 :
  une première commande réellement achevée puis erreur fatale contrôlée,
  aucune publication dans dst et aucun fichier de sortie candidat.

Le dernier cas utilise uniquement le callback fatal synchrone déjà public
`ggml_set_abort_callback`, dans le processus de fixture. Le callback vérifie
tous les65536 octets de destination encore à0xa5 et termine ce processus avec
le code86. Il ne réécrit pas le provider et n'introduit aucun fallback. Sa trace
doit prouver l'identité canonique/noyau, les allocations et la première tuile
soumise/achevée/appariée avant d'accepter l'erreur attendue. Il ne simule pas
une panne matérielle, un timeout pendant commande ou une erreur Metal terminale ;
ces chemins restent non mesurés. Le helper GLM n'utilise pas ce callback.

Les maxima planifiés sont35651584 octets de poids,4194304 octets de sortie par
processus,55788224 octets de payload logique total (<64Mio), dont le workspace
de graphe, le metadata1Mio et les8028160 octets Metal fixes. Le payload logique
ne mesure ni RSS ni résidence pilote. `currentAllocatedSize` du provider garde
son plafond séparé64Mio. Les plafonds RSS observés du contrôleur et d'un enfant
sont chacun128Mio ; le watchdog original conserve aussi RSS8,5Gio/swap32Mio,
réserve768Mio et sa politique contextuelle inchangée. Le contrôleur impose en
plus le domaine mémoire conservateur du provider et64Mio pour lui-même.

Au plus46 processus sont lancés séquentiellement et35872768 octets de sorties
brutes sont écrits. Le plafond global est240s, y compris la prévalidation, et
celui d'un enfant20s ; la compilation de la fixture a un plafond120s, puis son
smoke metadata15s. Ce sont des limites, pas des durées estimées ni mesurées.
Toute durée ou résidence encore inconnue reste `null` dans les plans.

## Ownership, garde et refus

Le contrôleur prend BENCHMARK_LOCK en non bloquant puis le verrou UID original
du modèle, pour toute la progression, sans charger un modèle. Un détenteur
existant provoque un refus ENGINE_BUSY avant essais, sans attente ni signal.
Le FD UID est hérité par chaque enfant. Un tuyau de départ empêche son calcul
avant vérification du PID/exécutable, observation OS du FD correct et assertion
anti-veille appartenant au parent. Le parent ne signale que son Popen retenu,
après vérification de son identité. SIGINT/SIGTERM arrêtent ce seul enfant ;
les échecs mémoire, secteur, identité et délai gardent leur classe.

Le départ de chaque enfant demande pression1, secteur et couvercle ouvert.
En cours, le domaine suffisant pression1/2 s'applique, avec la même baseline
swap globale que le watchdog. La progression s'arrête au premier refus, sans
modifier une garde ni retenter pour contourner ce refus. Les pics RSS/device
sont des maxima observés échantillonnés ; ils ne prouvent pas les pics continus
ou la mémoire invisible du pilote.

## Commandes futures, uniquement après signal zéro modèle

Le package `../build-run1` doit auparavant être réellement compilé/reçu, avec
ses smokes de feature et de liaison. Ces commandes ne sont pas exécutées :

```sh
PY=/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/testenv/bin/python
BRIDGE=/Users/macbook/Documents/Codex/2026-11-03/files-pasted-by-the-user-mission-2/work/metal-cpu-q8-bridge
"$PY" "$BRIDGE/fixtures/build_fixture.py" --package-build "$BRIDGE/build-run1" --receipt "$BRIDGE/fixtures/fixture-build-run1.json" --execute
"$PY" "$BRIDGE/fixtures/provider_fixture.py" --execute --package-build "$BRIDGE/build-run1" --binary "$BRIDGE/build-run1/provider-fixture" --fixture-build "$BRIDGE/fixtures/fixture-build-run1.json" --output "$BRIDGE/fixtures/run1"
```

Sept tests portables de métadonnées/raw sont passés sans compilation, sous
100ms. Ils ne valident ni compilation C++, ni les workers, ni Metal. Aucun
résultat de ces fixtures ne qualifie routes, logits, KDA, états GLM, EOS ou
préfill utile complet ; aucun objectif A global n'est annoncé atteint.
