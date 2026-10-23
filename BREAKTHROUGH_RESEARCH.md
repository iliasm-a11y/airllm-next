# Recherche V5 — résultats, hypothèses et échecs

Cette livraison développe des mécanismes fonctionnels dans le moteur. Elle ne
revendique ni nouvelle loi de compression ni résolution démontrée de l'inférence
interactive d'un trillion de paramètres sur 16 Go.

## H1 : conserver davantage de poids sous forme compacte

Problème : le cache décodé dilate FP8/BF16 et peut relire une même matrice
qui aurait tenu dans un cache de représentation source. Pour une matrice
décodée de D octets et une représentation source de C octets, la résidence
compacte est utile si le travail de décodage supplémentaire coûte moins que
les lectures évitées. Le rapport C/D ne prédit pas seul le temps total.

Prototype intégré : `compact_cache.py` + `NativeWeights`, réserve partitionnée,
comptage des allocations réelles, options CLI. Vérification : conservation des
bits source, comparaison des sorties, FP8/échelles BF16, experts, tuiles,
corruption/lecture interrompue, éviction sous faible localité. Expériences
entraînées : `benchmarks/benchmark_weight_residency_v5.py`, mêmes poids,
budgets et séquences. Les mesures négatives sont conservées.

Ce mécanisme économise des relectures lorsqu'il existe une résidence utile.
Il ne peut pas faire résider tous les experts de GLM dans quelques Gio. Il
reste optionnel et n'est pas présenté comme un kernel calculant directement
sur FP8 encodé : le décodage existe encore.

## H2 : conserver les tuiles vocabulaire sans supprimer le tuilage

Problème : V4.1 limitait correctement la taille d'une tête mais la relisait
à chaque passe dès qu'elle dépassait le plafond d'un tenseur. V5 peut admettre
chaque tuile décodée dans le cache déjà budgété. Cela permet une réutilisation
sans allocation intégrale. L'admission peut déplacer des poids plus utiles ;
elle est comparée séparément au cache compact. Les mesures du petit modèle
ne prédisent pas celles d'une tête de plusieurs Gio.

## H3 : optimiser la spéculation par coût constaté

Pour une profondeur k, le coût observé est `(temps_brouillon + temps_cible) /
tokens_émis`. Le coût de transport est compté séparément. Une profondeur très
acceptée peut rester lente ; un brouillon erroné gaspille des passes. La
politique V5 apprend des coûts par profondeur et recalibre avec des passes
sans brouillon. Les contrôles utilisent des horloges déterministes pour les
décisions adverses, puis une véritable génération entraînée pour les temps.
Les horloges synthétiques ne sont jamais présentées comme un benchmark SSD.

Le prompt lookup préserve les règles de correspondance exactes. Une recherche
linéaire Z réduit son pire cas ; une voie rapide bornée évite de dégrader les
correspondances répétées immédiates. L'ancien pire cas et la régression initiale
du prototype sont documentés dans les preuves du sous-programme.

## H4 : beaucoup d'agents logiques, une seule réserve native

L'isolation par rejeu vérifié rend le partage exploitable sans dupliquer les
poids ni multiplier les états résidents. Ce choix sacrifie du temps de reprise
et ne garantit pas un gain de missions/heure. Les journaux sont bornés et les
écritures engagées par quantum ; un mode en mémoire évite la persistance.
Les essais de file et reprise mesurent un travail d'inférence local, pas une
mission autonome externe terminée.

## Contre-exemple découvert à la clôture : égalités DSA

Le fixture GLM inchangé de seed 521, préfixe `[1,8,12]`, a divergé de
Transformers au cinquième token : le décodage natif héritait de choix de clés
différents lors d'égalités exactes à la frontière du top-k. Un token PAD dont
l'embedding est nul produit notamment une requête d'indexeur nulle et des scores
égaux. Le commit 4.1 reproduit le défaut ; un masque Transformers explicitement
à un donne la même référence que le masque initial. Ce n'était pas un problème
du scheduler ni un motif pour changer le seed ou les critères.

Le repli borné sur le top-k CPU de la référence installée corrige ce cas : les
huit tokens redeviennent identiques et les écarts des logits sur les préfixes
imposés restent au plus à 1,49e-7, contre environ 0,440 avant correction.
Les égalités non finies du masque causal ne nécessitent pas ce départage.
Les tests incluent aussi des requêtes entièrement nulles et un workspace
insuffisant refusé avant copie des scores vers l'hôte.

Ce fix de fidélité introduit un coût de synchronisation et, sur égalité, une
copie et un top-k CPU. Il ne constitue pas une accélération. La convention
exacte reste celle du build CPU PyTorch installé ; aucune stabilité de départage
entre versions ou GPU n'est affirmée. Le coût de l'import tardif de Torch et le
RSS restent distincts du scratch géré. Les traces avant/après sont conservées
dans `evidence-v5/glm-dsa-tie-diagnostic.json`.

## Travaux antérieurs et limites de nouveauté

Les sources primaires consultées et leurs instantanés sont dans
`evidence-v5/sources/`. SpecMoEOff étudie déjà l'augmentation du travail par
expert avec la spéculation ; Cascade analyse son coût dans les MoE ; SP-MoE
étudie le préchargement. Leurs accélérations publiées ne sont pas reproduites
ici. Les méthodes de cache et de prompt lookup sont des principes existants,
intégrés et contrôlés dans ce moteur, sans prétendre les avoir inventés.

Les recherches héritées `research/exact_io/` et leurs résultats négatifs
restent intactes. La compression réversible mesurée plus lente en cache chaud
n'est pas réétiquetée comme accélération V5. Les facteurs 2×, 5× ou 10× restent
des objectifs tant que la même charge complète ne les démontre pas.
