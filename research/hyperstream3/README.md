# HyperStream 3 — protocole de reprise

Base H2 : `5e8234181e736c0319a64df25e6813e92dd3e6da`.
Base H1 : `c8f81e57463ed3d7fd3d7dba226a65500439a9ce`.
Ne pas fusionner ni modifier les branches de référence. Utiliser le même interpréteur,
le checkpoint historique Qwen3.6 GGUF original et le prompt `research/hyperstream2/context.json`.
Aucun résultat réel H3 n'est fourni dans cette livraison.

Les comparaisons historiques, blocs fixes et kernels/transport isolés conservent
ce prompt historique de 364 tokens. Pour les plans spéculatifs et combinés,
**les deux membres de chaque paire** utilisent explicitement le même fichier figé
`research/hyperstream3/speculation-context.json` du candidat : copie de huit lignes
répétées, avec `max_new_tokens=128`. C'est une charge synthétique favorable au
prompt lookup, pas une preuve de qualité générale. Un éventuel gain sur cette copie
ne se généralise pas aux conversations ou à d'autres tâches. Ne pas agréger les
deux charges ; le harnais vérifie encore leur tokenisation et le budget de contexte.

## Contrôles sans inférence

```sh
PYTHONPATH=air_llm python -m pytest --noconftest -q tests_hyperstream3 \
  tests_next/test_gguf_speculative_portable.py tests_next/test_verify_gguf_speculation.py
```

Les 64 cas des nouveaux kernels nécessitent `AIRLLM_M5_GPU_TEST=1` **et** une
admission normale avant l'import MLX. Ne pas les activer sous pression mémoire.
Les tests portent sur de petites matrices et ne remplacent pas les essais réels.

## Préparer les plans

Créer des copies propres aux deux commits de référence, et figer les sources du candidat.

```sh
python benchmarks/plan_hyperstream3.py --h1-root /chemin/h1 --h2-root /chemin/h2 \
  --candidate /chemin/h3 --pairs 10 --output-dir work/h3-plans
```

Les 16 plans séparent H1/H2, blocs fixes 16/32/64/128, TensorOps, réutilisation
2/4/8/16, transport groupé et spéculation 2/4/8/16. Le plan combiné est marqué
non validé ; ses prérequis sont des conditions d'utilisation, pas une garde du runner.
Les blocs au-delà de 128 restent hors de cette campagne tant que les budgets
et les variantes précédentes ne sont pas validés.

Avant tout worker, constater cinq secondes continues de pression normale et
au moins la réserve de 3 Gio + payload 768 Mio + marge d'attente 256 Mio.
Ne pas fermer d'autres applications, purger les caches ni modifier les gardes.

```sh
python benchmarks/hash_checkpoint_guarded.py --model /chemin/original.gguf \
  --output work/h3-integrity-before.json

MLX_ENABLE_TF32=0 python benchmarks/campaign_research51.py \
  --model /chemin/original.gguf --tasks research/hyperstream2/context.json \
  --plan work/h3-plans/00_h1_vs_h2.json --output work/h3-h1-h2 \
  --resource-wait-seconds 600
```

Répéter le hash complet après les essais, avec un autre chemin de sortie exclusif.
Ne pas considérer un nom de blob, un stat ou un ancien hash comme preuve actuelle.
Le refus d'admission est une campagne différée, pas un benchmark réussi.
Le runner utilise la garde locale inchangée, même quand un worker importe H1/H2.

## Contrôler les preuves

Pour les comparaisons non spéculatives : conserver le protocole d'intégrité H2
(`research/hyperstream2/comparison-policy.json`), déclarer les options réellement
comparées et les hashes complets, puis lancer `benchmarks/verify_hyperstream2.py`.
Ce vérificateur exige une campagne complète et compare tous les logits FP32,
routes, tokens, positions, EOS et ressources. Ne pas relâcher ses tolérances.

Pour chaque paire spéculative, conserver les fichiers `.f32` et les banques de
routes par forward du nouveau harnais :

```sh
python benchmarks/verify_gguf_speculation.py serial.json candidat-spec.json \
  --output verification-spec.json
```

Ce contrôle indépendant couvre les tokens/logits et routes sur le préfixe accepté,
pas les états récurrents cachés ni les ressources. Compléter par les preuves
`campaign.json`, chaque garde, sources/protocole inchangés et les hashes complets
avant/après. Le runner vérifie la réussite et les contrôles de chaque worker ;
les comparaisons temporelles ne deviennent recevables qu'après la fidélité.
Les résultats spéculatifs ne passent pas directement dans le schéma H2 à un forward
par token. Ne pas transformer une réussite du contrôleur de logits en validation
complète des autres domaines.

Mesurer les prompts longs et des sorties suffisamment longues : le prompt historique
« 42 » s'arrête vite et ne permet pas d'évaluer seul les profondeurs spéculatives ;
les plans correspondants emploient donc la copie synthétique appariée décrite plus haut.
Conserver template, prompt et budgets identiques dans chaque paire, sans mélanger
les résultats de prompts différents. Une seconde campagne sans observateur corrobore
les temps, après la validation numérique. Inclure compilation/calibration, draft,
rejeux, repli avec nouveau préfill, lectures, attente d'inférence et sorties complètes.
L'attente préalable d'admission est séparée.

Les temps CPU utilisateur/système du worker sont enregistrés. Les intervalles hôte
inclusifs de l'observateur se recouvrent : ne pas les additionner ni les présenter
comme un temps GPU/SSD pur. Aucun nouveau chevauchement n'est prouvé sans chronologie
et mesure appariée. Les compteurs macOS restent distincts des lectures NAND.

## Activation explicite

API : `NativeGGUFQwenModel(..., kernel='metal', expert_batch=True,
packed_direct_io=True, metal_batch_backend='tensorops', metal_token_tile=4,
grouped_direct_io=True)` ; choisir les options séparément
pour les ablations. Le transport groupé requiert `expert_batch=True` et
`packed_direct_io=True`, et contourne le cache des tuiles individuelles.

La génération normale reste `generate`. L'entrée expérimentale séparée est
`generate_speculative(input_ids, max_depth=8, candidate_depths=(8,))`.
Le CLI supervisé `gguf-run` expose `--metal-batch-backend`, `--metal-token-tile`,
`--grouped-direct-io` et `--speculative-depth`; leurs valeurs par défaut désactivent
les nouveaux chemins. Aucune option ne relève les budgets mémoire.

## Preuves présentes et incident

Les fichiers de `evidence/` conservent les tests, les 121 relevés de pression,
les sept refus d'inférence et le refus de hash. Ils ne contiennent aucun token
H3 issu du vrai checkpoint. `validation-incident.json` décrit les 16 anciens tests
sur matrices GPU exécutés accidentellement sans admission ; leur journal est isolé.
Ils sont exclus de toute conclusion de vitesse, fidélité réelle ou stabilité.
Les deux journaux `h3-final.log` et `inherited-final.log` décrivent la sélection finale.
