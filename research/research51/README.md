# Reproduire la recherche 5.1

Voir [le rapport](../../RESEARCH_5_1_REPORT_FR.md) et [le protocole](PROTOCOL_FR.md). Le checkpoint doit exister localement. Aucun outil de cette campagne ne télécharge ni ne publie des poids.

## Vérifier les preuves sans lancer de modèle

Depuis la racine du dépôt, choisir des dossiers de sortie nouveaux :

```sh
python3 benchmarks/restore_research51_evidence.py --output work/restored-51
python3 benchmarks/analyze_research51.py work/restored-51/primary --output work/reanalysis-51
python3 benchmarks/independent_research51.py --campaign work/restored-51/primary \
  --summary work/reanalysis-51/summary.json --output work/independent-51.json
```

La restauration vérifie archive, taille, liste et SHA-256 de chaque fichier. Elle refuse liens, traversées de chemins et dossiers existants. Les JSON et journaux conservent les chemins et commandes de mesure d'origine ; l'analyseur n'a besoin ni de ces anciens dossiers, ni du modèle. Les chemins des fichiers d'entrée du nouveau résumé changent après restauration ; les agrégats numériques doivent rester identiques.

## Suite de tests

```sh
AIRLLM_PYTHON=/chemin/python3.12 scripts/validate_throughput.sh
```

Le script historique exécute aussi tous les nouveaux tests 5.1. La dernière validation inclut `gguf` 0.19.0 dans le PYTHONPATH, `AIRLLM_TEST_GGUF` vers le checkpoint existant et `AIRLLM_TEST_ROUTER_CAPTURE` vers la capture privée existante du routeur. Cette dernière contient des poids et est exclue de la publication. Sans ces ressources, leurs tests sont ignorés, jamais comptés comme réussites. Le test CUDA bitsandbytes n'est pas disponible sur ce Mac. Le fichier XML et le superviseur exacts sont dans l'archive.

## Refaire la campagne primaire

Créer un checkout détaché de référence depuis l'historique conservé :

```sh
git worktree add --detach work/reference-51 ab59677
```

`primary-plan.json` est le plan original publié avant mesure, avec chemins absolus de cette machine. Pour produire sa version portable, garder IDs, ordre et arguments et ne remplacer que les chemins :

```sh
python3 - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
plan = json.loads((root/'research/research51/primary-plan.json').read_text())
for job in plan:
    task = job['id'].split('.r', 1)[0]
    job['tasks'] = str(root/'research/research51'/f'{task}-task.json')
    job['engine_root'] = str(root/'work/reference-51' if job['id'].endswith('.baseline') else root)
with (root/'work/reproduction-plan-51.json').open('x') as f:
    json.dump(plan, f, indent=2)
PY
python3 benchmarks/campaign_research51.py --model /chemin/modele.gguf \
  --tasks research/throughput/primary-tasks.json --plan work/reproduction-plan-51.json \
  --output work/primary-51-new
```

La campagne refuse l'écrasement, une source/protocole modifié, un worker non supervisé, une garde échouée ou une sortie sans EOS. Elle n'essaie pas automatiquement après refus. Les budgets du protocole restent exigés. La référence et le candidat utilisent les mêmes paramètres ; les options expérimentales du candidat sont explicites dans le plan.

## Utiliser le preset mesuré

```sh
PYTHONPATH=air_llm MLX_ENABLE_TF32=0 python -m airllm.nextgen.gguf_cli gguf-run \
  /chemin/modele.gguf --kernel metal --device gpu --cache-mib 64 \
  --cache-policy lru --row-chunk 8192 --expert-batch --concat-tiles \
  --packed-direct-io --zero-copy-weights --bulk-metadata \
  --prompt 'Cite les trois couleurs du drapeau français.' \
  --max-new-tokens 64 --max-context 512 --report /nouveau/dossier/essai.json
```

La CLI conserve son budget historique KV de 192 Mio ; la campagne primaire fixe 256 Mio. Cette commande d'usage n'est donc pas une reproduction chronométrée de la campagne. Les options restent désactivées par défaut, le chemin historique et les garde-fous sont conservés.

## Autres expériences livrées

```sh
python benchmarks/research51_microbench.py cpu --repeats 51 --output /nouveau/micro.json
python benchmarks/research51_microbench.py metal-lifetime --imports 100 --output /nouveau/lifetime.json
python benchmarks/analyze_expert_reuse51.py --help
python benchmarks/profile_research51.py --help
```

Le microbenchmark porte sur des payloads générés et chauds, sans checkpoint ni débit modèle. Le profil réutilise les arguments et le superviseur du benchmark principal mais ajoute une instrumentation ; ses durées restent distinctes du résultat primaire. Le rejeu d'experts emploie des traces historiques dédupliquées, sans nouveau forward. Les commandes de chaque expérience réelle, versions, sources et valeurs brutes sont conservées dans l'archive ; les notes marquent explicitement les pilotes interrompus et les sorties tronquées.
