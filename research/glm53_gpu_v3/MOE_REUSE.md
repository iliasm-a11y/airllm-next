# Réutilisation réelle des routes et cache précédent par couche

`moe_reuse.py` lit uniquement les métadonnées de la capture B16 complète, ses
petits tenseurs de routes sauvegardés et l'inventaire GGUF. Il vérifie leur
identité contre la preuve d'union des sources. Aucun poids, payload d'expert,
modèle ou GPU n'est lu, alloué ou exécuté par cette analyse. Les résultats sont
dans [moe-reuse16.json](evidence/moe-reuse16.json).

La trajectoire contient quinze étapes de décodage après le préfill. Chacune
sélectionne huit experts dans chacune des 42 couches MoE. Les trois projections
originales d'un expert occupent ensemble 8 257 536 octets. Le replay conserve
les routes et leur ordre ; aucune décision du modèle n'est modifiée.

## Trois périmètres séparés

| Simulation | Demandes d'experts | Hits | Fraction de hits | Misses logiques moyens par token |
|---|---:|---:|---:|---:|
| Quatorze transitions entre les quinze tokens décodés | 4 704 | 2 218 | 47,1514 % | 1,466302 Go |
| Quinze tokens, cache par couche vide avant le premier | 5 040 | 2 218 | 44,0079 % | 1,553518 Go |
| Quinze tokens, dernier token du prompt retenu au départ | 5 040 | 2 346 | 46,5476 % | 1,483053 Go |
| LRU global de 128 Mio, préfill puis quinze tokens | 5 040 au décodage | 0 au décodage | 0 % | 2,774532 Go |

La première transition depuis le dernier token du prompt partage 128 experts
sur 336. Elle ne fait pas partie des quatorze transitions entre tokens décodés.
L'initialisation depuis le prompt est une hypothèse de simulation distincte,
pas une observation de résidence mémoire existante.

Le LRU global traverse 42 couches et ne conserve qu'environ seize experts
complets à cette capacité. Un cache séparé par couche peut conserver les huit
experts précédents de chaque couche, mais demande **2 774 532 096 octets**
persistants, soit 2,584 Gio. La différence de hit rate ne compare donc pas des
capacités équivalentes. Les premiers rangs ci-dessous sont ceux de la route
précédente, sans choix guidé par le token suivant.

| Nombre de rangs précédents gardés par couche | Payload persistant | Hits sur les 5 040 demandes, initialisation depuis le prompt | Misses logiques moyens |
|---|---:|---:|---:|
| 0 | 0 | 0 | 2,774532 Go/token |
| 1 | 346 816 512 octets | 438 | 2,533412 Go/token |
| 2 | 693 633 024 octets | 838 | 2,313211 Go/token |
| 4 | 1 387 266 048 octets | 1 543 | 1,925107 Go/token |
| 8 | 2 774 532 096 octets | 2 346 | 1,483053 Go/token |

Ces capacités ne comptent pas les buffers de staging, les activations, états,
allocations GPU et allocations système. Un vrai cache doit aussi conserver
les réservations pendant les lectures et les leases GPU, puis libérer les
entrées avant de réutiliser leur mémoire.

## Deux tokens et coût mémoire

Sur les quatorze paires adjacentes, l'union des experts de deux tokens occupe
entre **3,682861 et 4,632478 Go**, moyenne **4,240835 Go**. Avec une copie des
poids denses fixes, le total de cette union vaut en moyenne **13,685293 Go**.
Ce serait le coût brut si les ensembles précédents et courants de toutes les
couches étaient retenus simultanément. Ce n'est pas obligatoire pour une
politique qui évince les anciens experts non sélectionnés avant de remplir
leurs remplaçants. Les IDs futurs ne sont pas disponibles avant le routeur.

Conserver tous les poids denses fixes plus les huit experts précédents par
couche représente **12 218 990 592 octets**, soit **11,379822 Gio**. Cela
dépasse à lui seul le plafond RSS actuel de 8,5 Gio pour un cache CPU explicite
entièrement résident, avant tout autre buffer. Sur 16 Gio physiques, ce payload
laisserait seulement **4 960 878 592 octets** pour le système, le compresseur,
le runtime, les états et les buffers temporaires. Cette soustraction ne prouve
pas que le payload soit admissible ; les mesures de la charge normale restent
à confronter à des catégories mémoire sans double comptage.

## Ce que permet — et ne prouve pas — un scénario à 1 Go/s

Avec les poids denses entièrement résidents et le cache précédent par couche,
les quatorze transitions demanderaient en moyenne **1,466302 Go/token** de
payload MoE manquant. Si tous ces misses traversaient une interface délivrant
1 Go/s, avec coût CPU/GPU et latences supplémentaires ignorés, le scénario I/O
seul donnerait environ **0,682 tok/s**. Avec l'initialisation depuis le prompt
sur les quinze tokens, le chiffre serait **0,674 tok/s**. Ce sont des modèles
conditionnels, aucunement des performances mesurées ou des lectures NAND.

Sans rétention des poids denses, le même cache de huit experts réduit les
demandes ciblées de 12,218991 à **10,910761 Go/token** sur les quatorze
transitions. L'économie logique de 1,308230 Go/token ne représente que
**10,7065 %** du volume ciblé, soit environ **11,99 %** de gain de débit dans
un modèle purement linéaire I/O. Elle ne démontre pas un gain de génération.
La priorité aux poids denses reste justifiée.

La mémoire de 16 Gio ne suffit donc pas, à elle seule, à prouver une
impossibilité universelle de 0,5 tok/s : une rétention bien plus forte pourrait
changer le trafic. En revanche, ce cache brut complet ne respecte pas le
plafond RSS actuel du processus, sa résidence dans le cache de fichiers n'est pas établie
et la charge système observée doit être prise en compte. Les arrêts mémoire
réels et l'absence de gain complet reproductible restent des faits distincts.

La résidence de fichiers et les données compressées peuvent couvrir une partie
des misses logiques simulés ; inversement, transferts, relectures et staging
peuvent ajouter du trafic. Aucun nombre de ce document n'est présenté comme
un compteur de lectures physiques. Ce replay n'installe aucun cache dans RONIN.

Reproduction, sans modèle ni lecture des poids :

```sh
python -m research.glm53_gpu_v3.moe_reuse \
  --capture /absolute/path/to/closed-capture \
  --inventory research/glm53_m5/sources/gguf-headers.json \
  --source-union /absolute/path/to/closed-source-union.json \
  --output /absolute/path/to/new-moe-reuse.json

python -m pytest -q research/glm53_gpu_v3/test_moe_reuse.py
```

Les sept tests portables et trois sous-tests passent. Ils vérifient notamment
le départ froid, l'initialisation passée, les rangs sans connaissance future,
le thrashing d'un cache global, l'union des deux ensembles et l'absence d'accès
au fichier GGUF déclaré.
