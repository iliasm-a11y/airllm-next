# Architecture d’exécution AirLLM Next

## Intégration dans AirLLM

`AutoModel.from_pretrained(..., backend="mlx-next")` sélectionne le backend
`airllm.nextgen.runtime.AirLLMNextModel`. Les interfaces Ultra et historiques
sont conservées. Les modules AirLLM publics se chargent à la demande ; un accès
aux métadonnées ne démarre plus Torch, Transformers et toutes les familles.

L’accès aux poids réutilise l’index persistant `ultra.DirectCheckpoint`.
`nextgen.OffsetCheckpoint` ajoute des lectures explicites par offset, les tuiles
et les tranches d’experts. Le code ne construit aucun nouveau modèle à 744B
paramètres en RAM. La configuration et les formes attendues suffisent.

## Exécution réelle d’un token GLM

1. Lecture des seules lignes d’embedding des tokens présents.
2. RMSNorm, projections de requêtes et de KV compressés propres à GLM.
3. Indexeur DSA aux couches `full`, réutilisation de sa sélection aux couches
   `shared`, conformément à la configuration officielle.
4. Attention sur les positions sélectionnées, avec cache latent KV.
5. Routeur MoE réel en FP32 : sigmoid, biais de correction, sélection de groupes,
   top-k et pondération normalisée multipliée par 2,5.
6. Regroupement des tokens d’un chunk de prefill par expert réellement choisi.
   Lecture de cet expert uniquement, calcul SiLU/gate/up/down et accumulation.
7. Expert partagé, résidus et couche suivante.
8. Normalisation finale et projection de sortie, en tuiles si nécessaire.

Le prefill est découpé pour borner les activations. `generate` ne calcule que
les derniers logits nécessaires. Les états KV et les poids peuvent survivre
aux appels ; un nouveau `generate` démarre une nouvelle séquence, tandis qu’un
`past_key_values` explicite prolonge la séquence correspondante.

## Attention absorbée et KV latent

Pour chaque tête, les clés non positionnelles sont `C @ Wk.T` et les valeurs
`C @ Wv.T`, où `C` est le latent de rang 512. Les produits sont réassociés :

- score non positionnel : `(q @ Wk) @ C.T` ;
- sortie valeur : `(softmax(score) @ C) @ Wv.T`.

Le terme RoPE est ajouté séparément. Ce sont les mêmes opérations algébriques,
avec un ordre d’arrondi différent. Les positions sont choisies par le vrai
indexeur DSA avant le calcul de l’attention. Les têtes KV ne sont jamais
matérialisées pour tout l’historique. Les tableaux latents croissent par blocs
et restent soumis au budget KV, à la limite du modèle et à la limite locale.

Le RoPE GLM tourne des paires adjacentes et produit la concaténation des parties
paires et impaires tournées. Le backend conserve cette convention exacte, y
compris pour l’indexeur ; une substitution naïve au RoPE Llama serait erronée.

## FP8, entiers et mémoire

Le mode original lit les blocs E4M3FN et leurs multiplicateurs `weight_scale_inv`.
La déquantification s’effectue sur le stream CPU MLX pendant qu’une opération
antérieure peut s’exécuter sur le GPU. Les poids décodés sont évalués puis
mis en cache dans la précision de calcul choisie. L’inférence ne suppose aucune
instruction FP8 native sur le M5. Ce n’est pas un kernel FP8 matmul fusionné.

La conversion facultative crée des matrices affines 4/8 bits packées en uint32,
avec une échelle et un biais par groupe. L’exécution emploie réellement
`mlx.core.quantized_matmul`, sans développer une matrice FP32 complète à chaque
multiplication. Les versions du modèle et du dérivé sont liées par une empreinte.
La conversion écrit et valide une matrice atomiquement ; la reprise ignore les
matrices déjà validées. Un dérivé partiel peut être utilisé, le reste demeure
au format de référence. La précision de référence concerne les poids officiels
déquantifiés, pas une reproduction bit à bit des kernels FP8 avec activations
dynamiquement quantifiées des serveurs CUDA.

Les fichiers quantifiés utilisent échelles et biais FP32 sur disque. Les échelles et biais restent en FP32 en calcul, ainsi que l’accumulation
du matmul quantifié ; un essai BF16 a montré davantage d’erreur de cancellation.
Les activations générales et le KV peuvent rester BF16. Leur coût est compté dans les
budgets ; « 4 bits » ne signifie donc pas exactement 0,5 octet par paramètre.
Les projections sensibles du routeur, les indexeurs et `kv_b_proj` restent dans
le format d’origine. Une conversion mixte 8 bits peut augmenter le volume face
au FP8 ; une conversion mixte 4 bits réduit ce volume mais peut davantage
modifier la qualité. Ces choix sont explicites.

## Streaming et cache

Les offsets safetensors sont persistés sans recopier les poids. L’empreinte
comprend index, tailles et dates nanosecondes des shards. Le lecteur valide
formes, plages, chemins et chevauchements ; une coupure du disque, un fichier
manquant ou une lecture tronquée produit une erreur explicite.

Les descripteurs ouverts suivent un LRU borné. Un descripteur emprunté est
dupliqué sous verrou, ce qui empêche une éviction concurrente de le fermer.
`preadv`, lorsqu’il existe, lit directement dans un buffer ; sinon `pread`
travaille par segments. Les interruptions et lectures courtes sont gérées.
`mmap` est un mode de mesure distinct qui force explicitement la copie des pages.

Un cache unique de poids MLX évite les copies de caches CPU/GPU superposés. Les
politiques LRU, SLRU et LFU sont comparées. Le segment protégé SLRU est lui-même
borné ; les balayages ne bloquent pas les changements d’ensemble d’experts.
Le scheduler réserve les octets des résultats, déduplique les futures et
conserve les réservations jusqu’à consommation ou suppression. La prédiction
ne remplace jamais le routeur : le préchargement vise les composants ou experts
déjà sélectionnés pour cet appel.

L’évaluation paresseuse est interrompue aux frontières nécessaires des couches
et experts pour éviter qu’un graphe conserve toutes les matrices lues. Cela
impose des synchronisations ; leur nombre et leur coût doivent être profilés
sur Metal avant de les réduire davantage. Il n’existe pas ici de kernel Metal
personnalisé non mesuré annoncé comme supérieur à MLX.

## Autotuning et sûreté mémoire

L’autoconfiguration tient compte de la mémoire installée **et disponible**.
Caches de poids, lectures en vol, KV, workspace et réserve système ont des
budgets distincts. Le cache réduit sa taille sous pression ; la génération
s’arrête explicitement avant de dépasser la limite KV ou la réserve système.
L’allocateur MLX garde un cache borné. Les limites de mémoire câblée macOS ne
sont pas augmentées pour forcer un modèle à tenir.

`tune` compare des lectures réelles avec 1, 2 et 4 workers. `calibrate` mesure
une séquence ABBA avec et sans préchargement sur le vrai backend. Le préchargement
n’est retenu que si le gain médian de temps total dépasse 5 %. Le résultat,
lié au matériel, au modèle, au dérivé et au profil, est réutilisé. Le replay
des demandes de poids récentes compare les politiques de cache en octets de
misses ; cela n’est pas présenté comme un benchmark du SSD physique.

Les budgets sont des garde-fous d’objets gérés, pas une preuve formelle de borne
RSS totale. Allocations temporaires MLX, buffers de lecture, processus Python,
cache OS et coexistence des buffers lors d’une croissance KV requièrent le
workspace et la réserve. Les mesures soutenues M5 restent indispensables.
