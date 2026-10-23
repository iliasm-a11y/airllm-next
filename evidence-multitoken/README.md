# Preuves de la campagne multi-tokens / MoE

Le [rapport scientifique](../MULTITOKEN_REPORT_FR.md) fournit le tableau A–G,
les modèles, budgets, décisions et limites. Il distingue génération réellement
émise, vérification de positions connues, simulation de cache et scénarios géants.

## Restaurer les mesures

Les **68 rapports JSON** sont publiés dans `published/`, compressés sans perte.
`MANIFEST.json` contient chemins, tailles et SHA256 des fichiers originaux et
compressés. Depuis la racine du dépôt :

```sh
python3 benchmarks/restore_multitoken_evidence.py
```

Cette restauration a été exécutée et les 68 identités contrôlées. Le script
refuse d'écraser un fichier différent. Les JSON restaurés restent ignorés par Git ;
les logs texte et les archives gzip sont suivis.

Les **13 NPZ volumineux restent locaux**. Le manifeste contient leur audit et
leurs hashes : neuf hashes attendus concordent et 234 comparaisons de blocs ont
été recalculées sans divergence de logits sous tolérance, routes ou argmax.
Ils ne sont pas présentés comme téléchargeables dans cette PR. Aucun poids,
sidecar de poids ni texte du corpus n'est publié.

## Séries et contrôles

| Fichiers restaurés | Portée |
|---|---|
| `batch-cpu-128`, `batch-metal-256`, `batch-metal-physical` | B/C : 1/2/4/8/16/32 positions ; CPU, Metal et politique F_NOCACHE séparés |
| `generation-prompt-cpu`, `generation-prompt-metal` | A/E1 : recherche dans le préfixe, 2/4/8/16/32 et adaptatif ; EOS natif |
| `generation-smollm-cpu`, `generation-smollm-metal` | A/E1 : vrai modèle draft, enveloppe totale de cache 256 MiB |
| `generation-smollm-resident-metal` | Contrôle draft avec cache total 832 MiB ; deux prompts, une répétition |
| `cache-metal-128`, `cache-metal-512` | D : quatre politiques alternatives contre adaptatif ; mêmes budgets |
| `combination-metal` | G : préchargement exact et SpecPrefetch, quatre prompts, deux répétitions |
| `adaptive-retry-metal` | E2 : AR/fixe8/E1/E2, préchargement activé, quatre prompts, trois répétitions |
| `quantized-conversion`, `quantized-confirmation-metal`, `quantized-cpu` | F : conversion intégrale, générations, gardes mémoire et audit final des sidecars |
| `corpus-prepared`, `corpus-quality-metal` | F : 512 labels humains pré-déclarés ; protocole court, aucune qualité générale certifiée |
| `reference-metal` et variantes | Comparaison courte avec Transformers officiel, mêmes poids BF16 décodés FP32 |
| `storage-replay-cpu128`, `array-audit` | Replays analytiques des routes et contrôles indépendants des tableaux |
| `synthesis-final`, `repeated-and-f-summary` | Synthèses appariées ; protocoles différents séparés |
| `SOURCE_VERSIONS` | Manifeste historique par série et différences avec les fichiers plus récents |

Les noms de ce tableau portent l'extension `.json`. Chaque essai gardé possède
également un fichier `.guard.json` et un log. `quantized-metal.json` conserve les
premiers passages de profilage séparés ; les confirmations sans profilage font
foi pour le débit principal. `exploratory/` et `smoke-*` sont conservés pour
l'audit, exclus des gains principaux. Les anciens horizons forcés au-delà d'EOS
ne sont pas des preuves de débit de génération normal.

## Reproduire

Un environnement MLX compatible et les deux checkpoints officiels épinglés dans
le rapport sont nécessaires. Les benchmarks ne téléchargent pas implicitement
de poids. Les chemins locaux de la machine d'origine restent documentaires.

```sh
PYTHONPATH=air_llm MLX_ENABLE_TF32=0 python benchmarks/synthesize_multitoken.py --output work/synthesis-reproduced.json
PYTHONPATH=air_llm MLX_ENABLE_TF32=0 python benchmarks/summarize_multitoken_followups.py --output work/followups-reproduced.json
```

Les scripts `benchmark_multitoken_moe.py`, `benchmark_adaptive_retry_moe.py`,
`benchmark_quantized_granite.py` et `benchmark_quantized_corpus_quality.py`
exposent leurs paramètres avec `--help`. `run_multitoken_guarded.py --script ...`
surveille le RSS du processus et la réserve système. Rejouer les timings
séquentiellement, avec un fichier de sortie neuf et les mêmes budgets ;
les compteurs physiques du processus ne mesurent pas le trafic NAND du modèle.

Validation finale : **916 tests réussis, un ignoré, 11 sous-tests** ; compilation,
construction du paquet et imports isolés réussis. `validation-final.json` donne
le commit du moteur validé, les identités et les contrôles. Le Fanxiang est absent,
les poids Granite tiennent dans la RAM physique et aucun modèle 300–700B n'est
validé.
