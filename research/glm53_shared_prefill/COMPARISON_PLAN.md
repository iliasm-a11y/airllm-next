# Protocole de comparaison du préfill partagé exact

Ce protocole décrit une expérience ultérieure. Il n'est ni un résultat de
performance ni une autorisation d'activer un cache en production. L'audit
des états préalable se trouve dans [`STATE_AUDIT.md`](STATE_AUDIT.md).

## Entrées gelées et admission

Conserver les deux prompts RONIN complets, leurs identités 1/600, leurs
missions et marqueurs distincts, le gabarit `low` et leurs listes d'IDs
natives. Conserver également un retour à l'agent 1 avec son historique et une
variante de mémoire. Ne déplacer aucune instruction et ne retirer aucun
token. Le contexte doit contenir tous les tokens du prompt et le budget
entier de sortie. Une demande trop longue est rejetée, jamais tronquée.

Le préfixe compagnon inspecté contient 211 IDs communs, avec une frontière
candidate à 208 pour les microbatches de 8. Ces nombres servent à préparer
le plan ; l'expérience de préfill doit utiliser le LCP **confirmé par la voie
native** des prompts complets effectivement soumis. Les 869 octets du système
commun ne sont pas une longueur en tokens.

Exiger CPU10/GPU0, batch et microbatch 8, un seul modèle et un slot physique.
Les 600 identités et historiques demeurent des données hôtes isolées.
Enregistrer contexte, types de caches, dtypes, toutes les options effectives,
empreintes des sources/bibliothèques, gabarit/tokenizer et reçus d'intégrité
des deux shards. L'état hôte du sampler/parser est frais pour chaque requête.

Admission et surveillance inchangées : secteur connu et couvercle ouvert,
continuité horloge/alimentation aux instants échantillonnés, assertion de
veille possédée, RSS maximale 8,5 Gio, hausse swap maximale 32 Mio, gardes de
pression/headroom et identité PID. Arrêter à la première violation, garder
les preuves partielles, ne pas rejouer automatiquement. La durée et le volume
de sortie de chaque campagne sont fixés avant son début.

## Trois bras distincts

| Bras | Calcul et réinitialisation | Objet de la comparaison |
|---|---|---|
| A : CPU sans cache | Prompt complet et découpage serveur de référence ; reset complet entre requêtes. `--cache-ram 0 --ctx-checkpoints 0 --no-cache-idle-slots --cache-reuse 0 --no-cache-prompt`, spéculation absente. | Témoin réel sans cache |
| B : même CPU découpé, sans réutilisation | Même prompt complet, frontière choisie dans le LCP natif ; recalcul du préfixe pour chaque requête puis suffixe. Même état remis à zéro entre requêtes. | A/B isole l'effet de la frontière sur les formes et le scan KDA |
| C : même découpage, un snapshot commun | Capturer exactement à la frontière avant toute identité privée. Restaurer l'unique base immuable puis traiter le suffixe complet sous le verrou du slot. | B/C isole le restore et sa réutilisation |

Ni le cache générique de prompt qui consomme ses entrées, ni la copie
partielle de séquence ne remplace le bras C. Après reset complet, le snapshot
doit être autonome : `LLAMA_STATE_SEQ_FLAGS_NONE`, avec attention, KDA,
indexer et métadonnées cohérentes. **Interdire `PARTIAL_ONLY` comme snapshot
autonome** : ce flag peut omettre attention/indexer sans SWA.

Une stratégie préservant physiquement le KV commun mais remplaçant seulement
le récurrent serait un quatrième protocole distinct. Elle devrait prouver
que le KV actif ne porte plus de suffixe privé et que tous les états
positions/indexer/KPool sont cohérents. Elle ne doit pas être substituée
silencieusement à C.

## Exactitude avant chronométrage

Geler un périmètre testable avant le lancement : états à la frontière, les
premières microbatches du suffixe, les premiers pas de décodage, puis des
positions contenant les marqueurs et après la transition entre agents. Les
positions, formes et options d'instrumentation sont identiques entre bras.
Une capture finie ne qualifie que ce périmètre.

À ces points, comparer bit pour bit les IDs, tous les logits finis capturés,
les routes MoE ordonnées et scores de routeur, les sélections DSA, les données
R+S KDA, les cellules d'attention/indexer actives et leurs positions, et les
activations mHC capturées. Comparer les états logiques réellement utilisés,
pas des adresses ni des zones non initialisées. Conserver tailles, formes,
types, offsets, empreintes et fichiers bruts suffisants pour refaire chaque
comparaison ; un hash sans correspondance de périmètre n'est pas suffisant.

Les mHC/routes graph-locales n'ajoutent pas un état temporel à copier, mais
leurs sorties sont des témoins utiles des effets numériques du découpage.
Les sorties identiques ou la seule égalité d'un argmax ne suffisent pas.

Exiger A=B avant de comparer B=C. Si A/B divergent, conserver la preuve et
rejeter cette frontière pour l'exigence exacte. Si B/C divergent, rejeter le
restore. Ne pas ajuster les poids, les tolérances ou la quantification pour
transformer le résultat en réussite.

## Isolation, invalidation et échecs

Exécuter agent 1 → agent 600 → retour agent 1, puis cas d'annulation et mémoire
modifiée. Les messages, IDs natifs et sorties propres doivent contenir leur
marqueur et exclure celui de l'autre agent. L'identité numérique, la mission,
la mémoire et tout historique restent après la frontière commune.

Une seule clé de base doit lier les IDs/positions exacts, poids, gabarit,
tokenizer, options de rendu, runtime, politique CPU et dtypes/contexte/batch.
Toute divergence invalide le snapshot. Aucune restauration après redémarrage
n'est admise sans qualification séparée.

Après annulation, erreur, court restore, taille inattendue, corruption ou
changement de clé : invalider la base, effacer de manière cohérente attention,
KDA et indexer, remettre sampler/parser à zéro et terminer la requête en
échec ou annulation. Ne pas poursuivre sur un état partiellement restauré,
ne pas réessayer automatiquement le prompt. Le cleanup ne signale que le PID
possédé et conserve son vrai code de sortie.

## Budget borné avant allocation

Politique candidate, jamais activée implicitement : un snapshot immuable,
**plafond 256 Mio** pour son vecteur sérialisé complet, aucune entrée par
identité, aucune seconde base ni clone temporaire du snapshot. Interroger la
taille native avant allocation ; rejeter si elle dépasse le plafond. La seule
taille R+S logique de 145,5625 Mio ne permet pas de prédire cette taille.

Le budget global doit inclure l'état actif, les capacités réellement allouées
du snapshot, le scratch de sérialisation/restauration, les caches serveur
internes et le graphe. Mesurer les allocations et le pic RSS pendant création
et restauration. Une limite portant seulement sur les entrées du cache RAM
n'est pas un budget global. Si ces coûts ne peuvent pas être observés ou
bornés avant réutilisation, garder C désactivé. Les gardes existantes ne sont
jamais augmentées pour faire tenir le candidat.

## Mesures et décision de coût net

Séparer les captures numériques instrumentées et les mesures de performance :
le dump des logits et états peut modifier fortement le temps et l'I/O. Faire
passer les contrôles de fidélité d'abord, puis geler exactement les binaires
et paramètres pour les mesures non instrumentées. Rapporter leurs empreintes
et la différence d'instrumentation.

Pour chaque requête et chaque bras, conserver : temps de rendu/tokenisation,
préfill natif et hôte, temps de création et sérialisation du snapshot, temps
de restore, suffixe, TTFT, décodage et temps total ; IDs traités/cachés réels,
arrêt EOS/limite/erreur, sortie native ; compteurs OS de lectures/écritures du
processus et pageins avec identité PID ; RSS/footprint maximaux échantillonnés,
compression disponible, swap initial et hausse maximale, journaux des gardes.
Les lectures OS ne sont pas des octets NAND ni une mesure des seuls poids.

Employer des ordres A/B et B/C équilibrés et répétés, avec la même politique
de chargement et des états de cache de fichiers observés. Distinguer premier
passage et passages suivants. Ne pas qualifier de « froid » un cache dont la
résidence n'est pas contrôlée ; ne pas déduire les performances des 600 agents
de deux requêtes.

Le critère principal est le **temps total réellement mesuré**, y compris
création du snapshot, copies, restore, suffixes et invalidations. Comparer
`T_A(N)` à `T_C(N)` pour le même N, puis rapporter séparément l'effet A/B et
B/C. Pour deux agents, la création doit déjà être amortie si l'on revendique
un gain pour deux agents. Un calcul de seuil d'amortissement reste une
projection clairement étiquetée lorsqu'aucun N supérieur n'est mesuré.

Autoriser seulement une conclusion expérimentale bornée si toutes les
comparaisons exactes prévues passent, l'isolation passe, les gardes restent
claires et le coût net mesuré est inférieur au témoin sur les répétitions.
Sinon : candidat rejeté ou non qualifié, cache désactivé et CPU10/GPU0 de
production conservé. Aucune promotion automatique en production ne découle
de ce protocole.
