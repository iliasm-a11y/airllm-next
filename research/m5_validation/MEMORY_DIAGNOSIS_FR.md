# Diagnostic mémoire M5 — 27 octobre 2026

Relevé principal : `evidence/memory-verified.json`, le 27 octobre 2026 à
13:15:45, heure de Paris. Il s'agit de mesures séquentielles en lecture seule.
Les relevés `memory-initial.json` et `memory-final.json` conservent deux autres
instants de la session ; ils ne constituent pas une fenêtre de stabilité.
Les versions publiables omettent les lignes de commande privées et les
répertoires des exécutables. Les originaux locaux sont conservés hors du dépôt,
avec leur SHA-256 dans chaque dérivé public ; les mesures restent identiques.

| Observation | Mesure |
| --- | ---: |
| Mémoire physique | 16 Gio |
| Mémoire disponible selon psutil/macOS | 3,084 Gio |
| Pression macOS `kern.memorystatus_vm_pressure_level` | 2, avertissement |
| Swap utilisé | 3,142 Gio |
| Compresseur : mémoire physique occupée | 5,919 Gio |
| Compresseur : contenu logique avant compression | 10,077 Gio |
| Mémoire câblée | 3,452 Gio |
| Pages résidentes adossées à des fichiers | 2,290 Gio |
| Pages purgeables | 0,006 Gio |

Les pages adossées à des fichiers recoupent les catégories actives/inactives ;
elles ne sont pas toutes un cache immédiatement récupérable. Les deux tailles
du compresseur représentent le même contenu et ne s'additionnent pas. Le
pourcentage de `memory_pressure -Q` est distinct de la mémoire disponible et
ne rend pas un niveau de pression 2 admissible.

## Processus et attribution

L'inventaire voit 168 processus Node/npm, avec une somme RSS de
3 469 819 904 octets, et 104 processus classés Codex/ChatGPT, avec une somme
RSS de 3 363 143 680 octets. Ces sommes incluent des pages partagées et
n'attribuent pas la mémoire compressée ; elles **ne partitionnent pas la
mémoire physique** et ne quantifient pas ce qu'une fermeture libérerait.
Node/npm est classé avant l'application hôte : certains sont des connecteurs
lancés par Codex. Les catégories décrivent l'inventaire, pas un lien causal
exclusif avec la pression.

`footprint`, qui utilise une comptabilité macOS différente du RSS, a relevé :

| Processus | PID | `phys_footprint`, octets |
| --- | ---: | ---: |
| Codex Renderer principal | 46246 | 1 375 196 008 |
| Serveur Codex | 46251 | 1 238 894 920 |
| ChatGPT | 46217 | 363 809 840 |
| Autre Codex Renderer | 47446 | 276 088 848 |

L'identité de création de chaque cible a été revérifiée après le relevé.
La comptabilité d'empreinte physique peut inclure de la mémoire compressée ;
elle n'est ni du RSS résident, ni une promesse de récupération équivalente.
L'analyse détaillée `vmmap` d'un Renderer a signalé une limite de résolution
de sa zone PartitionAlloc ; elle n'est pas utilisée comme inventaire exhaustif.

Aucun worker d'inférence connu n'a été identifié par l'inventaire. Aucun nom
de processus, parent égal à 1 ou ancienneté n'a été traité comme une preuve
d'abandon ou d'appartenance à la mission. Aucun signal n'a été envoyé.

Le diagnostic n'importe pas MLX et ne crée pas de contexte GPU. Les compteurs
MLX étant locaux au processus, il ne peut attribuer les allocations GPU ou les
caches d'une autre application. Aucune allocation actuelle du moteur n'est
revendiquée et aucun cache tiers n'a été effacé.

## Admission et procédure autonome

Les budgets sont extraits des sources actuelles, accompagnés de leurs SHA-256.
La garde globale reste : réserve système de 3 Gio, payload modèle de 768 Mio,
plafond RSS de 2 Gio, croissance du swap plafonnée à 128 Mio. Le moteur du
benchmark réserve séparément 64 Mio de cache poids, 64 Mio d'I/O, 256 Mio de
KV et 128 Mio d'espace de travail ; ces budgets ne sont pas des allocations
résidentes observées au moment du diagnostic.

L'admission du modèle demande 3,75 Gio disponibles, soit un déficit mesuré de
715 505 664 octets (environ 682,4 Mio), **avant** les 256 Mio supplémentaires
de la fenêtre de reprise. Satisfaire seulement cette taille ne suffit pas :
la pression doit également revenir à 1 et rester normale selon le protocole.

La procédure réaliste est de préparer le banc et ses sources figées, puis de
laisser l'utilisateur enregistrer son travail et fermer volontairement les
applications qu'il choisit, y compris Codex/ChatGPT si nécessaire. Le banc se
lance ensuite depuis Terminal, sans dépendance à une conversation active.
Il doit refuser immédiatement si le système reste inadmissible, conserver les
preuves du refus, puis reprendre les travaux non terminés à une invocation
ultérieure. Il n'est pas nécessaire de vider le swap : son niveau initial est
enregistré et sa croissance reste surveillée. Il ne faut ni purger les caches
système, ni relever les gardes, ni exécuter plusieurs modèles lourds ensemble.

Le disque physique observé est uniquement le SSD interne de 500,3 Go.
`/Volumes/SSD` est un lien vers `/` ; ce n'est pas un SSD externe connecté.
Aucune performance de SSD externe n'a été mesurée.

## Reproduction du diagnostic

```sh
python benchmarks/diagnose_m5_memory.py \
  --output /chemin/nouveau-diagnostic.json --footprint-count 8
```

La sortie doit être nouvelle ; l'outil refuse d'écraser un fichier existant.
Il collecte l'inventaire et les commandes de lecture uniquement. Les cinq
tests du parseur, des budgets et de la confidentialité passent ; ils ne lancent
aucun kernel et ne constituent pas un benchmark matériel.
