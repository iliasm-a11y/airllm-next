# Contrôle ABBA borné — préparé, non exécuté

Deux formes seulement : M8192/K4096/N8, puis N16. Chacune suit quatre
processus séquentiels **CPU → GPU → GPU → CPU**, chacun exécutant cinq fois
le même graphe avec ses poids synthétiques résidents. Aucune inférence ou
lecture de poids du modèle n'est effectuée par ce composant.

Le contrôleur neuf réutilise exactement le binaire existant
`build-run1/provider-fixture-warm-v2`, SHA
`889a246510320e08d50c463105850f2f82797d184d74795d7aafc4c19f269da2`,
et son reçu réel `fixtures-warm-v2/build-receipt.json`, SHA
`98ade6cab761e4c42f9632ce09ed51930b2707c79fc73bf87e0c26aeff21491c`.
Il ne compile rien, ne réécrit aucun ancien reçu et n'altère aucune fonction
du contrôleur/validateur d'origine. `static-preflight.json` constate uniquement
la vérification des fichiers/reçus existants, sans exécuter de CLI native.

Les 95 entrées source/binaire/lib/reçu sont épinglées avant import/activation,
surveillées par identité pendant l'exécution et rehashées après. Le reçu
sépare source du contrôleur neuf, ancien contrôleur, build de la fixture,
provider, driver, profil et vrais fichiers utilisés.

Les gardes d'origine restent en place : verrou benchmark, UID avant spawn/FD
hérité observé, PID/ctime/exe, secteur/couvercle/continuité, réserve système,
swap32Mio, payload/allocations Metal64Mio, RSS enfant et contrôleur128Mio,
polling0,1s, délai enfant20s et global240s. Le helper existant impose en plus
son pic RSS natif de durée de vie128Mio ; ses cinq SHA d'itérations doivent
être identiques au dernier fichier F32 brut. Aucune limite n'est élargie.

Chaque forme garde quatre comparaisons explicites au **premier CPU** :

1. Premier CPU contre lui-même : taille/SHA et finitude de l'oracle.
2. Premier GPU contre cet oracle : critères Metal originaux, bits CPU séparés.
3. Deuxième GPU contre le même oracle : mêmes critères.
4. Dernier CPU contre l'oracle : égalité F32 bit à bit obligatoire, y compris
   le signe des zéros ; une tolérance Metal ne peut masquer une dérive CPU.

Les huit processus conservent leurs sorties `output.f32`, JSON natifs,
stdout/stderr, échantillons et compteurs de commandes GPU terminales. Tous
les quatre processus d'une forme doivent présenter les mêmes SHA réels des
poids quantifiés et des entrées. Chaque GPU doit publier cinq opérations
complètes ; les CPU ne doivent présenter aucune commande du provider.

La première itération comprend l'initialisation du provider ; les quatre
suivantes sont des timings résidents séparés. Même un ABBA positif ne prouve
ni un gain GLM, ni un gain SSD, ni un fonctionnement de bout en bout.

Préparation déjà effectuée : six tests synthétiques passent ; source AST,
plan et provenance existante sont vérifiés. Aucun processus natif, compilateur,
GPU ou modèle n'a été lancé par cette préparation. Le coordinateur doit
relire ce contrôleur avant de lancer la commande suivante :

```sh
PY=/Users/macbook/Documents/Codex/2026-11-02/files-pasted-by-the-user-mission/work/venv/bin/python
"$PY" work/metal-tensorops-q8-nocopy/abba-two-shapes/abba.py \
  --execute --output work/metal-tensorops-q8-nocopy/abba-two-shapes/run1
```

Le mode par défaut écrit seulement un plan neuf. Une sortie existante est
refusée. Aucun résultat ABBA n'existe tant que le coordinateur n'a pas lancé
et clos cette exécution sous les gardes.
