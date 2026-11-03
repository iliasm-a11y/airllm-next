# Déduplication des preuves natives fermées

Script préparé : `dedup_closed_captures.py`. Aucune preuve réelle n'a été dédupliquée lors de sa préparation ; les tests emploient de petits fichiers synthétiques temporaires.

Le script accepte uniquement un répertoire `native-*` sous ce répertoire `work`, un `guard.json` fermé avec code de sortie natif connu, et un PID natif qui ne correspond plus à un processus vivant. Il prend le verrou de campagne non bloquant pour éviter que les lectures et calculs de SHA perturbent une autre inférence mesurée.

Les seuls candidats sont les fichiers binaires nommés dans les manifestes natifs de tenseurs, de logits et d'états. Les poids, sources, métadonnées et fichiers non référencés restent exclus. Les liens symboliques, tailles anormales, autres propriétaires et incohérences de SHA provoquent un refus.

Les groupes sont constitués par taille et SHA256 réellement recalculée. Les octets sont ensuite comparés intégralement. En mode d'application, une liaison physique temporaire vers la cible d'origine permet un retour arrière ; le remplacement du nom par la liaison vers le contenu commun est atomique. La taille et la SHA de chaque nom d'origine sont vérifiées à la fin. Le journal de transaction est conservé à la racine du run.

Les noms et les octets restent identiques. Les alias partagent désormais un inode et ses métadonnées : conserver ces captures immuables, car une écriture ultérieure sur un alias modifierait tous ses liens. Les estimations rapportées concernent les octets logiquement redondants, sans affirmer un nombre d'octets APFS ou SSD physiquement récupérés.

Préparer le plan, seulement une fois la campagne fermée :

```sh
PYTHON dedup_closed_captures.py --run native-full8 --report native-full8-dedup-plan.json
```

Appliquer après examen du plan, avec un autre reçu :

```sh
PYTHON dedup_closed_captures.py --run native-full8 --report native-full8-dedup-applied.json --apply
```

Remplacer `PYTHON` par le Python de la campagne disposant de `psutil`. Le programme refuse d'écraser un reçu existant. Un run encore actif ou un autre benchmark en cours est refusé avant tout traitement des payloads.

Tests légers exécutés : `test_dedup_closed_captures.py`, **7 tests réussis**, sur fixtures uniquement. Ils couvrent le plan sans remplacement, l'application conservant noms/SHA, le refus d'un processus actif, le verrou de campagne, les symlinks, les incohérences de SHA et le retour arrière atomique.
