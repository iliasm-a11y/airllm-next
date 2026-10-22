# AirLLM Next — reprise scientifique du 22 octobre 2026

Le [rapport scientifique](RAPPORT_SCIENTIFIQUE_FR.md) distingue mesures M5 CPU,
rejeu de traces héritées, simulations et hypothèses. Le code part du commit
`d43032f38a0325aafd2973847d7ed84b21280c61`, vérifié identique à la branche
distante `research/ooc-engine-2026-10-22` avant les changements.

Cette reprise ne démontre pas l'inférence interactive des checkpoints complets
GLM/Qwen sur 16 Go. Elle fournit des expériences exécutées, un transport de poids
réversible optionnel et des contre-exemples aux impossibilités trop générales.

## Résultats et sources

- `inherited/manifest.json` : provenance et empreintes des 19 fichiers retrouvés.
- `results/` : JSON bruts, répétitions individuelles, sorties de vrais experts,
  résultats négatifs et rapports de tests JUnit.
- `palette_decode.c` : décodeur exact scalaire/NEON, compilé localement.
- `numerical_contract.py` : comparaison CPU PyTorch singleton/bloc sur les quatre
  experts Ling conservés ; 20 des 24 comparaisons FP32 diffèrent légèrement,
  malgré des poids identiques. Ce résultat ne mesure pas des tokens du modèle.
- `airllm.nextgen.lossless` : format indexé, lectures sélectives, contrôles de
  taille, SHA256 et transport intégré optionnel aux deux runtimes.
- `airllm.nextgen.prompt_draft.PromptSpanDraft` : propositions issues du contexte,
  toujours soumises au vérificateur cible existant.

Les fichiers `samples/` et `build/` sont reproductibles et ignorés par Git.
Les poids entraînés du checkpoint Ling original ne sont pas redistribués.

## Reproduire

Pour l'ensemble des expériences sur macOS arm64 : Python 3.14 avec NumPy et
pytest, clang/SDK macOS, un checkpoint **local existant** Ling-mini-2.0.
`compression.zstd` est inclus dans Python 3.14. Aucun téléchargement n'est fait
par ces programmes. Les expériences de poids lisent quelques dizaines de Mio
de ce checkpoint ; elles ne chargent pas son modèle complet.

```bash
python3 research/exact_io/run_all.py --model-dir /chemin/local/Ling-mini-2.0
```

Si un autre interpréteur **déjà installé** dispose de PyTorch CPU et NumPy,
ajouter `--torch-python /chemin/vers/python`. Le programme ne l'installe pas.
Après création du pack par `trained_parity.py`, l'expérience numérique se lance
aussi séparément, sans relire le checkpoint entier :

```bash
/chemin/vers/python research/exact_io/numerical_contract.py \
  --expert-pack research/exact_io/build/trained-expert-pack
python3 research/exact_io/backend_probes.py --python /chemin/vers/python
```

Le pack est un **fixture de recherche existant** pour cette expérience. Cela
ne transforme pas l'overlay de production en checkpoint autonome : son contrôle
d'identité de la source reste obligatoire. Les poids du fixture ne sont pas
redistribués. Les résultats bruts suffisent à identifier la mesure, et la commande
complète permet de recréer le fixture depuis son checkpoint local.

Les audits de métadonnées, traces et tests de transport fonctionnent sans Torch,
Transformers, MLX ni safetensors. Les contrôles de l'ancien vérificateur sont
sélectionnés explicitement, car le conftest global exige les dépendances MLX/Torch :

```bash
python3 research/exact_io/audit_metadata.py
python3 research/exact_io/routing.py
python3 -m pytest research/exact_io/tests -q
PYTHONPATH=air_llm python3 -m pytest --noconftest tests_next/test_speculative.py -k 'not native' -q
```

Les tests d'un vrai backend MLX restent nécessaires ; la sélection ci-dessus
n'est pas une exécution de toute la suite historique.

## Utiliser le transport optionnel

Créer d'abord un pack de **tenseurs explicitement sélectionnés**. Le plafond
par défaut de 256 Mio empêche une conversion de plusieurs To involontaire.

```bash
PYTHONPATH=air_llm python3 -m airllm.nextgen.lossless /checkpoint/local /nouveau/pack \
  --key model.layers.3.mlp.experts.0.gate_proj.weight \
  --key model.layers.3.mlp.experts.0.up_proj.weight \
  --key model.layers.3.mlp.experts.0.down_proj.weight \
  --codec auto --block-bytes 1048576
```

Avec un environnement MLX déjà opérationnel :

```python
from airllm.nextgen import AirLLMNextModel, NextConfig

config = NextConfig(lossless_dir='/nouveau/pack', device='gpu')
# Bibliothèque C optionnelle, fournie explicitement après compilation locale :
# config = NextConfig(lossless_dir='/nouveau/pack',
#     lossless_native_library='/chemin/absolu/palette.dylib', device='gpu')
with AirLLMNextModel('/checkpoint/local', next_config=config) as model:
    # Les autres poids restent lus par le chemin original.
    print(model.metrics()['lossless_transport'])
```

La source originale reste requise pour son identité, sa disposition et les
tenseurs non empaquetés. Cette première implémentation **ne permet donc pas
de supprimer le checkpoint original pour économiser le disque**. Un format
autonome est une étape ultérieure, pas une fonctionnalité livrée.

Le pack conserve les bits, y compris les zéros signés et les payloads NaN.
Cela ne certifie pas les kernels FP8 dynamiques du serveur d'origine, ni
l'identité numérique entre vérification par bloc et décodage séquentiel MLX.
Le checksum est activé par défaut. Le mode sans checksum est une ablation
explicitement nommée ; il suppose le stockage intact, comme le lecteur original.
Un sidecar de quantification supplémentaire est refusé dans ce mode.

RAM : un résultat demandé plus un nombre constant de buffers de taille bloc
par lecteur actif ; `workspace_bytes` réserve au moins six blocs par worker.
L'index binaire mesure 56 octets par bloc et n'est pas chargé en dictionnaire
Python. L'index global par tenseur demeure en mémoire. Les fichiers doivent
rester intacts et le pack est lié à l'identité du checkpoint.

`lossless_transport.payload_bytes` et `index_bytes` comptent les octets de
fichiers demandés. `decoded_block_bytes` inclut l'amplification des lectures
partielles. Les métriques `encoded_file_*` incluent les replis sur le lecteur
original. Aucun de ces compteurs ne prétend mesurer la NAND physique.

## Brouillon sans second modèle

```python
from airllm.nextgen.prompt_draft import PromptSpanDraft

# Seulement avec un backend qui implémente generate_speculative, actuellement GLM.
# Le modèle cible valide chaque token ; un mauvais span est rejeté.
draft = PromptSpanDraft(max_lookback=8192, max_match=64, min_match=3)
# result = model.generate_speculative(ids, draft, max_new_tokens=64, max_depth=8)
```

La méthode reprend le principe connu de prompt lookup. Son intérêt dépend du
contenu répété et de l'acceptation réelle. Aucun gain linguistique sur GLM/Qwen
n'a été mesuré ici. L'ancien mode spéculatif conserve ses limites : glouton,
batch 1, modèle compatible, retour arrière validé, chunk assez large. Le MTP
et l'échantillonnage spéculatif du modèle géant ne sont pas implémentés.
