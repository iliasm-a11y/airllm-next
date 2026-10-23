# Modèle MoE entraîné intermédiaire : protocole Granite

La phase 2 utilise le véritable checkpoint public
[IBM Granite 3.0 1B-A400M Instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/tree/ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445),
révision `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`.
IBM décrit un MoE de 1,3 milliard de paramètres totaux, environ 400 millions
actifs, 24 couches, 32 experts et top-k 8. Le fichier original BF16 contient
2 669 283 096 octets ; son SHA256 LFS officiel est
`f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc`.
Le checkpoint n'est pas inclus dans Git.

L'inventaire initial du dossier de modèles de cette mission ne contenait que
SmolLM2-135M, un modèle dense. Les petits fixtures MoE du dépôt sont aléatoires.
Les candidats officiels [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/main)
(environ 13,8 Go BF16) et [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/tree/main)
(environ 28,6 Go BF16) dépassent la limite de téléchargement choisie de 4 Gio et
n'ont pas de backend natif Next correspondant. Granite permet une expérience
entraînée complète avec un téléchargement borné et une architecture séparée.

Le nouveau backend est une classe Python **explicitement expérimentale** :

```python
from airllm.nextgen import AirLLMNextGraniteMoeModel, NextConfig

options = NextConfig(
    device="cpu", dtype="float32", cache_bytes=256*1024**2,
    io_bytes=32*1024**2, kv_bytes=32*1024**2,
    workspace_bytes=128*1024**2, reserve_bytes=1536*1024**2,
    max_tensor_bytes=32*1024**2, max_context=128,
    load_tokenizer=False, use_saved_tuning=False,
)
with AirLLMNextGraniteMoeModel("/path/to/checkpoint", next_config=options) as model:
    logits = model([[1, 7, 8]], logits_to_keep=1).logits
```

L'inventaire de production et le dispatch `AutoModel` ne certifient pas ce
backend. Il accepte `granitemoe` flottant, des attentions causales complètes,
RoPE par défaut, les scalaires Granite explicites et les banques d'experts
officielles, anciennes ou actuelles. Il rejette quantification, familles
Granite hybrides/partagées et égalités au seuil du top-k, dont PyTorch ne
garantit pas la même sélection entre implémentations.

## Référence indépendante et exactitude

`benchmarks/benchmark_granite_moe.py` construit les modules officiels
Transformers sur `meta`. Des hooks indépendants chargent chaque groupe depuis
les safetensors, convertissent les poids BF16 en FP32, exécutent les opérations
officielles inchangées, puis libèrent les paramètres sur `meta`. La référence
n'importe aucun lecteur ni opérateur mathématique AirLLM. Un test vérifie que
ses logits de préremplissage/décodage sont **bit à bit identiques** au modèle
officiel entièrement chargé sur un petit checkpoint indépendant.

Le modèle entraîné complet est comparé avec les seuils fixés **avant**
l'exécution : `atol=1e-4`, `rtol=5e-4`, mêmes tokens gloutons et mêmes ensembles
top-k d'experts à chaque couche/étape. Les variantes natives doivent aussi
produire les mêmes tableaux de logits et routages **bit à bit** entre elles.
Les traces sont sauvegardées avec leurs hashes ; un écart fait échouer le
programme en conservant les résultats.

## Reproduction

Télécharger les fichiers `.json`, `.txt`, `README.md` et `model.safetensors`
depuis la révision immuable ci-dessus avec le téléchargement borné livré :

```sh
python benchmarks/download_granite_checkpoint.py --destination /path/to/checkpoint
```

Le téléchargement local produit
`download-provenance.json`, avec modèle, révision, taille et hash de chaque
fichier. Ce manifeste est aussi inclus dans le rapport final du benchmark.
Le fichier de poids doit correspondre au SHA256 officiel avant toute mesure.

```sh
MLX_ENABLE_TF32=0 python benchmarks/benchmark_granite_moe.py \
  --model-dir /path/to/checkpoint \
  --output evidence-io-phase2/granite-cpu.json \
  --tokens 4 --repeats 1 --prompts 2 --cache-mib 256 \
  --variants reference native_off native native_spec \
  --max-rss-gib 5 --reserve-gib 1.5
```

Les workers tournent séquentiellement. Un contrôleur échantillonne leur RSS et
la mémoire disponible toutes les 50 ms ; il arrête le worker si le seuil est
dépassé. Il journalise le code de sortie, le pic échantillonné, le RSS maximal
du processus et la mémoire système minimale. Ce mécanisme est une protection
réactive, pas une limite matérielle instantanée. Les budgets du moteur restent
256 Mio cache poids, 32 Mio E/S, 32 Mio KV et 128 Mio espace de travail.

Les temps incluent le décodage glouton avec cache KV, les copies des traces de
routage et les retours de logits. La référence charge des couches complètes,
le natif lit les experts sélectionnés ; leur rapport de temps décrit ces deux
stratégies complètes, pas une accélération isolée des noyaux.

Le hash et le téléchargement peuvent réchauffer le cache OS. Ces mesures ne
sont donc pas des débits de disque à froid. Un checkpoint de 2,67 Go tient
dans les 16 Go de mémoire physique ; il ne valide pas le régime au-delà de la
RAM. Le SSD externe Fanxiang n'est pas impliqué. Aucune validation GLM/Qwen
géant, longue durée thermique ou qualité générale n'en découle.
