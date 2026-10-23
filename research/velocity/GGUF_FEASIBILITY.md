# Modèle local GGUF : compatibilité et intégration

## Constat vérifié

Le manifeste local de `qwen3.6:latest` a pour SHA256
`07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522`.
Son blob de poids, nommé par le digest
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`,
occupe exactement **23 938 321 664 octets** (22,294 Gio), au-delà des 16 Gio
physiques. Son SHA256 complet a ensuite été vérifié avec un tampon fixe de
4 Mio et `F_NOCACHE` : 10,596 s, RSS maximal 25,51 Mo et aucune croissance du
swap observée. Cela certifie l'intégrité du blob, pas sa génération sûre.

L'inspecteur `inspect_installed_gguf.py` a lu le header et l'index sans mapper
ni charger les poids : GGUF v3, 57 entrées de métadonnées, 1 194 tenseurs,
début des données à l'octet 12 009 280. L'index contient :

| Format | Nombre de tenseurs |
|---|---:|
| Q4_K | 349 |
| Q6_K | 94 |
| F32 | 544 |
| F16 | 207 |

Les experts gate/up sont Q4_K et les experts down Q6_K. L'embedding est Q4_K;
la projection de sortie est Q6_K. `Q4_K_M` désigne donc une quantification
mixte, pas un seul format affine MLX. Les métadonnées correspondent aux
dimensions officielles de Qwen3.6-35B-A3B : 40 couches, 256 experts, top-8,
un expert partagé et une alternance de trois couches récurrentes pour une
couche d'attention complète.

## Pourquoi le lecteur MLX affine ne suffit pas

Le backend `qwen_runtime.py` réutilise déjà les équations Qwen3.5 texte, mais
attend un checkpoint safetensors flottant avec les conventions Transformers.
L'index GGUF réel et le [convertisseur llama.cpp primaire](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)
font apparaître plusieurs transformations qu'il faut inverser ou intégrer :

- Codec par blocs Q4_K et Q6_K, différents des matrices U32/scales/biases MLX.
- Alias et formes des experts, des projections QKV/Z et des convolutions.
- Normes converties de la convention `1 + weight` vers des poids RMS ordinaires.
- `A_log` remplacé par `-exp(A_log)`.
- Têtes V de l'attention linéaire réordonnées du groupement Transformers vers
  le broadcast tuilé ggml; le flag local `ssm.v_head_reordered` est vrai.
- Réordonnancement éventuel des projections et normes M-RoPE.
- Tokenizer et template conversationnel embarqués dans GGUF.

Un simple adaptateur de noms et quelques lignes de déquantification ne
constitueraient donc pas une implémentation correcte. Une intégration native
requiert des contrôles par tenseur et par couche, puis des logits et des pas
autorégressifs comparés à un décodeur indépendant du même GGUF. La conversion
complète en BF16, proche de 72 Go, n'entre pas dans le budget disque vérifié et
n'est pas nécessaire pour mesurer le système actuel.

## Chemin d'exécution immédiatement accessible

Le binaire déjà installé
`/Applications/Ollama.app/Contents/Resources/llama-server` expose réellement,
d'après son `--help` local :

- `--load-mode mmap` pour mapper le fichier;
- `--no-repack` pour éviter le repacking des poids CPU;
- `--no-warmup` pour séparer l'initialisation d'un calcul initial;
- `--lazy-mode on` pour lire certaines lignes à la demande;
- `--cpu-moe`, `--moe-cache-mib` et `--gpu-layers` pour les placements disponibles;
- `--parallel`, `--ctx-size`, `--batch-size` et `--ubatch-size` pour borner les
  contextes et lots.

La présence de ces options ne garantit pas qu'une configuration donnée reste
dans la réserve mémoire ni qu'elle soit rapide : il faut le vérifier dans les
logs d'allocation et les mesures système. En particulier, un mapping de
23,94 Go ne prouve pas un working set résident de cette taille, et une réponse
générée avec une forte croissance du swap ne satisfait pas la cible sûre.

Un futur adaptateur AirLLM externe peut déléguer à ce serveur existant, en
annonçant explicitement le moteur tiers. Il doit créer son propre processus,
retenir son PID et ses descendants, borner le contexte, surveiller RAM/swap dès
le chargement, inclure le coût des requêtes et de leurs synchronisations, puis
arrêter uniquement les processus qu'il possède. Il ne faut pas identifier le
processus par une sous-chaîne fragile du nom de commande. Une telle délégation
valide un backend externe du produit; elle ne valide pas les kernels, caches
ou lecteurs du backend MLX natif.

Le premier essai Ollama a émis un token mais a déclenché un repacking massif
et une forte croissance du swap. Il doit rester un résultat négatif de sûreté,
sans certification du niveau A. Les essais directs du serveur, leurs sorties
et leurs limites relèvent du rapport d'expérience séparé.
