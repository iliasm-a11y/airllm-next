# Reproduire les mesures Throughput

Le checkpoint doit déjà exister localement. Aucun outil ne télécharge de modèle ni n'utilise un service d'inférence. Les valeurs de vitesse publiées dans le rapport principal proviennent de `primary-balanced`, séparé des essais exploratoires.

## Tests

```sh
AIRLLM_PYTHON=/chemin/vers/python scripts/validate_throughput.sh
```

Pour reproduire tous les contrôles locaux (1 195 réussites et un test NVIDIA ignoré), ajouter `gguf` 0.19.0 au PYTHONPATH et renseigner `AIRLLM_TEST_GGUF` avec le checkpoint existant ainsi que `AIRLLM_TEST_ROUTER_CAPTURE` avec la capture locale historique du routeur. Celle-ci contient des poids et n'est pas publiée. Sans ces ressources facultatives, les tests correspondants sont explicitement ignorés ; ils ne sont pas comptés comme réussis.

## Campagne réelle

```sh
python benchmarks/campaign_throughput.py \
  --model /chemin/vers/le/modele.gguf \
  --tasks research/throughput/primary-tasks.json \
  --output /nouveau/dossier/de/preuves \
  --repeats 2 --simd-groups 1
```

Le dossier doit être nouveau. La graine, les sources et l'ordre contrebalancé sont enregistrés avant lancement. Aucune relance automatique n'a lieu après un refus d'admission, un échec ou une réponse incomplète.

```sh
python benchmarks/analyze_throughput.py /dossier/de/preuves --output /nouveau/dossier/analyse
python benchmarks/study_gguf_draft.py /dossier/de/preuves --output /nouveau/dossier/draft.json
```

L'étude du draft utilise les propositions causales du composant existant et les compare après coup aux tokens enregistrés. Aucun débit spéculatif de génération n'en est déduit.

## Utiliser la configuration retenue

```sh
PYTHONPATH=air_llm MLX_ENABLE_TF32=0 python -m airllm.nextgen.gguf_cli gguf-run \
  /chemin/vers/le/modele.gguf --kernel metal --device gpu \
  --expert-batch --concat-tiles --cache-policy lru --row-chunk 8192 \
  --cache-mib 64 --max-context 512 --max-new-tokens 256 \
  --prompt 'Explique brièvement le rôle de la mémoire vive.' \
  --report /nouveau/dossier/essai.json
```

Le programme conserve ses protections par défaut. La CLI utilise son budget KV historique de 192 Mio ; la campagne de mesure emploie explicitement 256 Mio. Une différence entre ces deux configurations ne doit pas être masquée dans une comparaison chronométrée.

## Lire les preuves archivées

```sh
python benchmarks/restore_throughput_evidence.py --output /nouveau/dossier/preuves-restaurees
python benchmarks/analyze_throughput.py /nouveau/dossier/preuves-restaurees/primary-balanced --output /nouveau/dossier/analyse
```

`evidence-throughput/raw-manifest.json` liste les empreintes des fichiers d'origine et des captures binaires locales. L'archive contient les JSON, journaux et commandes ; elle exclut les poids et les vecteurs binaires volumineux. Les restaurations et analyses n'exigent aucune inférence.
