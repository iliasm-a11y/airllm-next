#!/usr/bin/env python3
"""Regenerate the human-readable report from saved measurements, no new claims."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
cpu = json.loads((root / 'benchmarks/results-cpu.json').read_text())
expert = json.loads((root / 'benchmarks/results-experts.json').read_text())
glm = json.loads((root / 'evidence/glm-5.3-layout-audit.json').read_text())


def count(path):
    tree = ET.parse(path)
    cases = tree.findall('.//testcase')
    skipped = sum(c.find('skipped') is not None for c in cases)
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    return len(cases)-skipped-failed, skipped, failed


new = count(root / 'evidence/tests-ultra.xml')
old = count(root / 'evidence/tests-upstream.xml')
summaries = cpu['summary_medians']
labels = {'upstream_first': 'AirLLM stable, premier découpage',
          'upstream_reopen': 'AirLLM stable, fichiers découpés réutilisés',
          'ultra_first_cache0': 'Ultra, premier manifeste, cache hôte désactivé',
          'ultra_reopen_cache0': 'Ultra, manifeste réutilisé, cache hôte désactivé',
          'ultra_cache64': 'Ultra, cache hôte 64 Mio'}
rows = '\n'.join(f'| {labels[label]} | {s["construction_seconds"]:.3f} | '
                 f'{s["ttft_excluding_import_seconds"]:.3f} | '
                 f'{s["ttft_from_worker_start_seconds"]:.3f} | '
                 f'{s["decode_tokens_per_second"]:.2f} |'
                 for label, s in summaries.items())
expert_rows = '\n'.join(f'| {r["mode"]} | {r["median_seconds"]*1000:.3f} | '
                        f'{r["checkpoint_counters"]["logical_bytes"]/1024**2:.0f} |'
                        for r in expert['runs'])
trained_path = root / 'benchmarks/results-pretrained.json'
if trained_path.exists():
    trained = json.loads(trained_path.read_text())
    trained_text = (f'Modèle entraîné **{trained["model_id"]}**, révision '
                    f'`{trained["revision"]}`, testé sur CPU en float32. '
                    f'Écart absolu maximal des logits : **{trained["max_absolute_logit_difference"]:.8g}**. '
                    f'Les **{trained["new_tokens"]} tokens générés** correspondent à la référence.\n\n'
                    f'Sortie observée : « {trained["output"].strip()} ».\n\n'
                    'Reproduction locale :\n\n```bash\n'
                    'python benchmarks/validate_pretrained.py --model-dir /chemin/SmolLM2-135M-Instruct\n```\n\n'
                    'Téléchargement explicite de la révision utilisée (~270 Mo de poids) :\n\n```bash\n'
                    'python benchmarks/validate_pretrained.py --download-dir /chemin/test-smollm2\n```\n')
else:
    trained_text = 'Aucun résultat de modèle pré-entraîné n’est enregistré. Ne pas considérer ce test comme validé.\n'

report = f'''# Validation réelle et reprise sur Mac M5

Rapport du 22 octobre 2026. Base stable : AirLLM v4.0.0,
`ff35db207a0c559af9aa95d686057c3fe84f1d40`.

## Périmètre exécuté

Linux x86-64, Python 3.12.14, PyTorch 2.14.1+cpu, Transformers 5.20.2,
safetensors 0.8.0, un thread CPU pour les benchmarks. L'environnement dispose
d'environ 9,73 Gio de RAM totale. Ni GPU CUDA, ni MPS, ni MLX matériel disponible.
Le fichier `evidence/environment.json` contient la détection réelle.

| Suite | Réussis | Ignorés | Échecs |
| --- | ---: | ---: | ---: |
| Nouveau moteur (`tests_ultra`) | {new[0]} | {new[1]} | {new[2]} |
| Tests existants AirLLM sélectionnés | {old[0]} | {old[1]} | {old[2]} |

Les quatre tests matériels ignorés couvrent MLX natif et MPS pour Llama, Mixtral,
GLM-MoE-DSA. Les autres tests vérifient notamment les logits, la génération et le
cache KV ; les poids liés ; les formats MoE individuels et 3D ; le checkpoint FP8 ;
les limites de cache et de préchargement ; la réutilisation/invalidation du manifeste ;
les erreurs de calcul/E/S ; la fermeture ; les requêtes concurrentes sérialisées ;
la construction des banques sur meta et les biais de routage persistants.

Les 44 tests d'origine concernent découpage Kimi K3, Qwen3.8 et Flash-Next,
données LoRA, composants et gradients LoRA. Les tests CUDA/bitsandbytes et les
anciens tests qui téléchargent des modèles distants ne font pas partie de cette
validation. Aucun résultat GPU d'origine n'est revendiqué ici.

## Démarrage et génération mesurés

**Petit Llama synthétique de {cpu['model']['parameters']:,} paramètres**, poids float32
de {cpu['model']['weight_bytes']:,} octets, quatre couches, huit tokens générés.
Trois essais par configuration dans des processus Python neufs. Même version de
dépendances et même squelette de modèle. Toutes les séquences produites sont identiques.

| Chemin | Construction (s) | Premier token hors import Python (s) | Premier token depuis début du worker (s) | Décodage (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
{rows}

Valeurs médianes. Le temps depuis le début du worker inclut les imports Python
mesurés dans le worker, pas le coût initial de lancement de l'interpréteur par le
processus parent. Le benchmark ne purge pas le cache de pages du système :
« premier » signifie sans fichiers de préparation propres au moteur, pas SSD
physiquement froid. Le chemin stable est le `AirLLMBaseModel` d'origine, non modifié.

Le gain important sur ce petit modèle vient notamment de la suppression des
collectes globales et vidages d'allocateur par couche. Quand le modèle tient dans
64 Mio de cache, le test favorise aussi la réutilisation de tous ses poids.
**Ces tokens/s ne décrivent ni un modèle géant ni le Mac M5.**

Les mesures brutes comprennent les latences par token, les séquences, compteurs
logiques et OS, RSS après exécution et RSS maximale. La RAM maximale est légèrement
plus élevée pour Ultra sur ce test : environ 450 Mio contre 431 Mio pour le chemin
stable ; ce n'est pas un gain universel de mémoire. Le nouveau moteur vise à éviter
la résidence du modèle complet, pas à minimiser le coût fixe de Python/PyTorch.

Reproduire : `python benchmarks/benchmark.py --repeats 3 --tokens 8`.

## Lectures sélectives des experts

Vraie banque locale de 128 experts, 48 Mio de poids float32, huit experts consultés
à chaque passage. Huit passages, comparaison des sommes numériques pour détecter
une lecture incorrecte. Le jeu de routage est fixe et volontairement favorable au cache.

| Mode | Temps médian par passage (ms) | Payload logique cumulé sur huit passages (Mio) |
| --- | ---: | ---: |
{expert_rows}

La sélection seule divise ici le payload par 16 ; le cache réutilise ensuite les
3 Mio nécessaires à ces huit experts. Les temps sont ceux de lectures et copies
CPU sur un système déjà chaud, pas d'inférence complète ou de SSD externe.

Reproduire : `python benchmarks/benchmark_experts.py`.

## Vérification d'un petit modèle entraîné

{trained_text}

Les poids du modèle de contrôle ne sont pas inclus dans l'archive. Le résultat
établit le fonctionnement sur ce petit modèle, pas une qualité de réponse générale.

## GLM 5.3 : vérification structurelle réelle

Configuration et index officiels téléchargés, empreintes conservées dans
`evidence/glm-5.3-layout-audit.json`. Architecture `{glm['architecture'][0]}`,
78 couches ; 256 experts par couche sparse, 8 experts choisis par token ; FP8
par blocs 128 × 128. L'index contient {glm['checkpoint_tensor_entries']:,} entrées,
141 fichiers et {glm['checkpoint_bytes']:,} octets.

Le graphe principal a réellement été construit sur meta :
{glm['native_meta_model_parameters']:,} paramètres, sans allouer les banques sur CPU.
Tous les paramètres et buffers persistants du graphe ont trouvé leurs clés dans
l'index ; les compagnons d'échelle attendus des experts sont présents. Le graphe
principal n'utilise pas le prédicteur MTP supplémentaire contenu dans le checkpoint.

Le petit GLM-MoE-DSA et son checkpoint FP8 vérifient l'exécution. **Aucune inférence
du GLM 5.3 complet n'a été effectuée.** L'ensemble des poids n'a pas été téléchargé.
Les kernels et capacités mémoire MPS doivent être vérifiés sur le Mac. MLX natif
ne prend pas en charge cette architecture dans cette livraison.

Reproduire l'audit sans poids : `python scripts/audit_glm_metadata.py`.

## SSD externe et protocole M5

1. Extraire l'archive sur le Mac et installer avec `scripts/setup_macos.sh`.
   L'installation télécharge les frameworks binaires ; Python 3.12 ou 3.13 est requis.
2. Exécuter `scripts/validate_macos.sh`. Les quatre tests matériels doivent
   s'exécuter et réussir. S'ils échouent, conserver les rapports avant de poursuivre.
3. Mesurer le vrai volume, sans toucher aux fichiers existants :

   ```bash
   .venv-macos/bin/python -m airllm.ultra.cli doctor --directory '/Volumes/VOTRE_SSD'
   .venv-macos/bin/python -m airllm.ultra.cli io-bench \\
     --directory '/Volumes/VOTRE_SSD' --size-mib 256 --repeats 3
   ```

   Le benchmark crée puis supprime uniquement son propre fichier temporaire.
   Le rapport indique le mode de cache utilisé ; même `F_NOCACHE` ne garantit pas
   un contrôleur SSD froid. Le test `results-host-io.json` de cette livraison mesure
   le volume éphémère Linux, dont le matériel n'est pas identifié, pas le futur SSD.
4. Exécuter un petit modèle entraîné sur MPS, comparer sa sortie au CPU, et garder
   les métriques. Tester séparément `mlx-ultra` sur un Llama admissible, si souhaité.
5. Placer le checkpoint géant complet sur le SSD, conserver ses fichiers d'origine,
   vérifier le répertoire avec `python -m airllm.ultra.cli audit --model /chemin`.
   Aucun découpage supplémentaire n'est nécessaire pour Ultra.
6. Commencer GLM avec un contexte court (128 à 256 tokens) et 1 à 8 nouveaux tokens,
   observer la pression mémoire macOS, puis mesurer les performances soutenues.
   Le MacBook Air est sans ventilateur : une mesure courte ne couvre pas la tenue
   thermique d'une longue génération.

Exemple de premier essai GLM, **après** les validations précédentes :

```bash
./scripts/launch_macos.sh \\
  --model '/Volumes/VOTRE_SSD/modeles/GLM-5.3' \\
  --device mps --dtype bfloat16 --max-context 256 --max-new-tokens 8 \\
  --cache-mib 512 --device-cache-mib 256 --prefetch-mib 256 \\
  --metrics validation-local/glm-premier-essai.json
```

Sans réutilisation d'experts, les seules matrices FP8 sélectionnées représentent
3 × 6144 × 2048 × 8 × 75 = **22 649 241 600 octets par token**. Les échelles,
l'attention et les couches denses s'ajoutent ; le cache peut retrancher une partie
des lectures. C'est une borne de volume sous ces hypothèses, pas une prévision de
temps. Le logiciel ne peut supprimer le coût des octets réellement nécessaires.

## Reprise du développement

Le source complet, patch depuis le tag stable, dépendances, wheel et sdist sont
inclus. Les scripts de test recréent leurs petits checkpoints. Le chemin d'origine
est conservé pour les autres modèles et l'entraînement ; le nouveau moteur refuse
les formats non implémentés plutôt que de les charger approximativement.

Priorités après les mesures Mac : corriger d'éventuels écarts MPS/MLX révélés par
les tests, mesurer séparément lecture/déquantification/calcul/transferts sur le
vrai SSD, ajuster les budgets à la RAM disponible, puis envisager des kernels
quantifiés natifs pour le format réellement utilisé. Le MTP de GLM, la quantification
4 bits et l'offload du cache KV ne sont pas implémentés dans ce moteur.

## Sources primaires

- [Sources AirLLM et tag stable](https://github.com/lyogavin/airllm/releases/tag/v4.0.0)
- [Configuration officielle GLM 5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)
- [Index officiel GLM 5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/model.safetensors.index.json)
- [Implémentation native GLM-MoE-DSA](https://github.com/huggingface/transformers/tree/v5.20.2/src/transformers/models/glm_moe_dsa)
- [MLX : attention native](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
'''
(root / 'docs/VALIDATION.md').write_text(report)
print('Wrote docs/VALIDATION.md from saved evidence.')
