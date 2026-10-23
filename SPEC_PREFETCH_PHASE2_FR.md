# SpecPrefetch : deuxième phase expérimentale

SpecPrefetch reste désactivé par défaut. Les corrections ci-dessous réduisent
des coûts identifiés et sécurisent le retrait des lectures inutiles. Elles ne
constituent pas une validation de débit sur SSD externe ni sur modèle géant.
Les résultats natifs finaux sont dans `evidence-io-phase2/specprefetch-paired-stable.json`
et le rapport général de cette phase.

## Corrections

- **Index des transitions.** La prédiction consultait l'ensemble des arêtes de
  toutes les couches. Elle consulte maintenant les seuls successeurs des
  experts précédents de sa couche. L'index secondaire est supprimé avec les
  arêtes et les couches évincées ; les plafonds globaux restent identiques.
- **Retrait asynchrone après une erreur de prédiction.** `observe` annule les
  lectures encore en file et retire les lectures actives à leur achèvement.
  Elle peut rendre immédiatement la main au routeur. Les réservations restent
  détenues dans les deux budgets tant que la lecture n'est pas terminée.
  `discard`, `close` et le drain du lecteur attendent toujours la fin réelle.
- **Libération effective des tableaux.** Les rappels de fin utilisent des
  références faibles. Un rappel conservé dans un `Future` ne forme donc pas de
  cycle retenant le tableau après libération de ses réservations.
- **Admission selon la fiabilité observée.** Les quatre premières prédictions
  sont évaluées sans soumettre de lecture. Ensuite, une
  précision inférieure à 75 % sur les seize dernières prédictions suspend les
  nouvelles lectures. Les prédictions sans lecture continuent à être évaluées,
  ce qui autorise la reprise lorsque les routes redeviennent prévisibles.
- **Admission selon les coûts mesurés.** Lorsque le coût de chargement de la
  matrice a déjà été observé, le lecteur fournit cette mesure au prédicteur.
  Le gain conservateur `(2 × précision − 1) × coût_chargement` doit dépasser
  le coût d'ordonnancement et le coût moyen de prédiction/observation.
  Le lecteur estime l'ordonnancement à partir de l'aller-retour de ses demandes
  exactes, moins le coût de chargement observé de la même matrice. Cette mesure
  existe avant toute lecture spéculative. Sans cette mesure, le prédicteur peut
  effectuer une calibration bornée et mesure son propre délai de lancement.
  Cette règle est une heuristique d'admission ; elle ne prédit pas le débit du
  disque. Les chronomètres supplémentaires du lecteur sont actifs uniquement
  lorsque SpecPrefetch est activé.
- **Chemin des demandes ordinaires.** Un `take` portant sur une matrice non
  prédite évite de réconcilier toutes les lectures en attente. Le contrôle reste
  sous le verrou d'admission, afin de préserver le retry des échecs facultatifs.

Le vrai routeur et ses experts sélectionnés restent seuls responsables du
calcul. Aucune prédiction ne remplace un expert demandé. Les échecs spéculatifs
sont supprimés avant une nouvelle lecture exacte. Une lecture active erronée
peut encore occuper un worker et une partie du budget : le retrait asynchrone
ne permet pas d'annuler un appel système actif.

## Tests et contrôles reproductibles

`tests_next/test_specprefetch.py` contient 55 cas : exactitude des demandes,
erreurs et retries, calibrage, limites des métadonnées, plafonds mémoire,
annulation en file, demande exacte pendant une lecture erronée, drain partagé
concurrent, admission concurrente et collecte des tableaux avec le ramasse-miettes
cyclique désactivé. Les tests avec événements synchronisent les threads ; leurs
délais ne servent jamais de résultat de vitesse.

Les 55 cas et les 20 tests d'intégration natifs ont réussi : **75/75**,
consignés dans `evidence-io-phase2/specprefetch-tests.log`.

Le contrôle `benchmarks/benchmark_specprefetch_control.py` compare la version
publiée de phase 1 à la version courante. Il conserve les empreintes des sources,
du scheduler commun et d'un fichier déterministe local de 8 Mio. Il mesure :

1. 8 192 appels de prédiction, pour 128, 1 024 et 4 096 arêtes conservées ;
2. cinq répétitions de relecture de routes, avec de vrais `os.pread`, des routes
   identiques et un contrôle sans le filtre de précision ;
3. l'identité SHA-256 de tous les tableaux effectivement demandés.

Ce contrôle n'importe pas MLX, ne télécharge aucun modèle et n'injecte aucun
délai. Le cache OS n'est pas purgé. Ses temps incluent le calcul du SHA-256 des
tableaux demandés. Ils mesurent ce contrôle local, pas un gain d'inférence ou de
débit sur SSD. Les données et profils sont dans
`evidence-io-phase2/specprefetch-control.json` ; la sortie est dans le fichier
`.log` associé.

Lors de l'exécution finale, les 8 192 prédictions avec 4 096 arêtes prennent
**951,43 ms en phase 1 contre 11,57 ms en phase 2**. Il s'agit uniquement du
coût du prédicteur, avec un essai de chronométrage par taille d'index. Le replay
sur fichier local donne 14,96 contre 15,07 ms pour une route répétée, et 14,77
contre 13,79 ms pour des routes aléatoires. Ces temps très courts et ces données
échauffées ne permettent aucune extrapolation à un modèle entraîné ou au SSD.

Le diagnostic natif `benchmarks/benchmark_specprefetch_native_probe.py` a
isolé le coût associé à l'admission : une seule lecture spéculative peut
démarrer le second worker MLX, et le ralentissement persiste ensuite. Une
alternance ou une synchronisation accrue entre les threads est une explication
plausible ; le benchmark compte les threads démarrés et ne trace pas chaque
changement de contexte. Sur trois processus frais par variante,
l'ancienne admission donne 75,54 ms avec deux workers, contre 63,62 ms sans
SpecPrefetch et 62,93 ms avec prédiction/observation seules, sans aucune lecture
spéculative. Un préchauffage de quatre prédictions seul ne résout pas ce problème.

Avec le filtre de coût issu des demandes exactes **et** les quatre prédictions
sans lecture, le contrôle suivant donne 61,09 ms et un seul worker dans les
trois processus, contre 58,24 ms sans SpecPrefetch. Avec le filtre de coût seul,
les quelques admissions précoces relancent le second worker : 73,18 ms. Les
logits et tokens de tous ces contrôles sont identiques ; les empreintes des
sources sont cohérentes à l'intérieur de chaque suite. Ce diagnostic justifie
les deux règles d'admission retenues ; il ne démontre toujours pas un gain par
rapport à la désactivation. Les trois suites historiques sont
`specprefetch-shadow-probe.json`, `specprefetch-warmup-probe.json` et
`specprefetch-demand-gate-probe.json`. La comparaison finale après promotion des
règles reste `specprefetch-paired-stable.json`.

Cette suite finale utilise cinq ordres de variantes mélangés, trois répétitions
échauffées, deux prompts et seize nouveaux tokens, soit trente générations
échauffées par variante. Le SpecPrefetch de phase 1 prend **75,445 ms**, le
SpecPrefetch corrigé **60,950 ms**, et la désactivation **58,014 ms**. La correction
réduit donc le temps de 19,2 % par rapport à l'ancienne admission, tout en restant
5,1 % plus lente que la désactivation. Les logits sont identiques bit à bit,
les tokens sont identiques et les empreintes des sources restent cohérentes.
Sur ce petit modèle aléatoire, les règles corrigées n'admettent aucune lecture
spéculative : le résultat mesure une désactivation conditionnelle appropriée et
son coût de contrôle, pas un recouvrement réussi d'E/S physiques.

Exemple, après extraction de `specprefetch.py` du commit `3e4138a` :

```sh
python benchmarks/benchmark_specprefetch_control.py \
  --baseline-source /chemin/phase1/specprefetch.py \
  --output evidence-io-phase2/specprefetch-control.json
python -m pytest -q tests_next/test_specprefetch.py tests_next/test_io_wall_integration.py
```

Les observations exploratoires antérieures au filtre de coût ont aussi été
conservées dans les fichiers `specprefetch-native-*` et
`specprefetch-dispatch-*`. Les empreintes des sources de chaque exécution sont
dans leurs rapports. Les profils instrumentés servent au diagnostic des coûts ;
les comparaisons natives finales sans cette instrumentation font autorité.
