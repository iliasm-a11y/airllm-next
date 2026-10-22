# Moteur et choix d'implémentation

`AutoModel.from_pretrained(..., ultra=UltraConfig(...))` active `AirLLMUltraModel`.
Les autres appels empruntent le chemin stable conservé. Le modèle Transformers
réel gère l'attention, RoPE, le routage et son cache KV ; le nouveau moteur gère
la résidence des poids. Aucun appel à une API d'inférence distante n'est effectué.

## Stockage

`DirectCheckpoint` lit l'index et les en-têtes safetensors pour connaître les
fichiers, formes, types et offsets. Le manifeste persistant dépend du chemin
résolu, de l'index, des tailles et des dates de modification des fichiers. Une
modification invalide ce manifeste. Cette validation n'est pas un hachage intégral
des centaines de gigaoctets de poids et ne protège pas contre un remplacement
conservant artificiellement taille et date.

Un LRU limite les handles de fichiers, avec verrou pour partager les mappings
entre le calcul et le worker. Les valeurs mises en cache sont matérialisées,
pas seulement mappées : un préchargement doit réellement lire les pages.
Les compteurs `logical_bytes` comptent le payload demandé, pas les lectures
physiques du SSD, qui peuvent différer à cause des pages et caches système.

## Résidence

Le modèle est construit dans un contexte de périphérique `meta`, avant même
la création des `Parameter`. Les petits buffers RoPE calculés sont reconstruits
hors de ce contexte. Les buffers persistants, dont le biais de routage GLM,
chargent leurs valeurs réelles depuis le checkpoint.

Des hooks chargent les embeddings, les couches hors experts, la norme finale et
la tête. Ils ne gardent pas l'embedding entier résident même si ses poids sont
liés à la tête. Les banques MoE 3D sont remplacées par un exécuteur qui reçoit les
indices et probabilités déjà calculés par le vrai routeur. Il applique les mêmes
projections, activation et accumulation que l'implémentation eager de référence.

Les checkpoints par expert (GLM/ancien Mixtral) et les tenseurs 3D sont reconnus
explicitement ; les formats inconnus échouent sans annoncer un support générique.
Le premier axe des tenseurs 3D est découpé avec `safe_open.get_slice`, sans lecture
de toute la banque. Les poids FP8 2D accompagnés de `weight_scale_inv` utilisent
une déquantification par rangées de blocs. Les banques 3D FP8 et les formats
GPTQ/AWQ/MXFP4 ne sont pas implémentés dans le nouveau moteur.

## Cache et préchargement

Le cache hôte sépare les entrées nouvelles et réutilisées. Les scans remplacent
d'abord les entrées nouvelles pour protéger des experts fréquemment rappelés.
Un cache distinct conserve éventuellement des poids MPS/CUDA déjà transférés.
Sur Apple Silicon, ces deux caches partagent la RAM physique avec tout le système.

Un seul worker favorise les lectures séquentielles du SSD. Les réservations
bornent le payload conservé dans les futures ; les prédictions terminées peuvent
être écartées. Une erreur du worker remonte au demandeur. Les prédictions ne
changent jamais la sélection des experts, uniquement leur disponibilité anticipée.
Les fréquences persistent après fermeture ; les poids mis en cache en RAM et le
cache KV ne sont pas sérialisés entre processus. Le cache disque du système peut
survivre à une session, sans garantie de résidence.

## MLX

Le backend `mlx-ultra` exécute un décodeur Llama simple directement en MLX : RMSNorm,
RoPE, attention groupée et projections sur Metal. Il lit les régions de poids
à partir de leurs offsets et utilise un stream CPU pour matérialiser les lectures
anticipées. Les sorties et le cache KV sont évalués avant de libérer les poids
d'une couche pour ne pas retenir tout le modèle dans un graphe paresseux.

Il est volontairement limité aux configurations implémentées. GLM reste sur
le chemin Transformers, qui possède déjà son attention DSA et ses règles de cache.
La présence du code MLX n'est pas une validation sur un GPU Apple.

## Limites opérationnelles

Ce moteur traite une requête à la fois par instance et vise l'inférence seulement.
Le cache KV et les activations restent en RAM ; les contextes très longs sont
inadaptés aux 16 Go cibles. Une taille maximale par groupe de poids empêche les
allocations individuelles trop grandes, mais ne borne pas la RAM totale. Le
backend de référence demeure nécessaire pour les modèles et formats non pris en charge.
