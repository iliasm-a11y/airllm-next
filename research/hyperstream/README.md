# HyperStream — protocole reproductible sur Apple Silicon

Ce protocole compare le moteur de référence `ca3a42935be876bbdbbc1e2adbd16c8547746175`
au candidat publié `83daeefde83d1714a2cc2ec1579c3c1c66e98a49`. Ce dernier correspond
au commit local de validation `6960ef97ac4aa71d191087320362c1d636562aed` : leurs
arbres Git sont identiques, `1c1c7a3ef6befc5ee0ee209e9ce14844dc065337`.

Le code du moteur vient du commit local `644c3474626ba24d52f7f84cff6666adb7cd1867`,
publié sous `1508c1538ab0e015343ab62a1d36b2f7756eb81d`. Ses sources sont inchangées
dans les instantanés `6960ef9` et `83daeef`. Les outils de campagne, d'analyse et
la documentation constituent des ajouts distincts ; leurs empreintes sont
enregistrées dans les preuves. Les SHA locaux et publiés doivent rester
distingués dans les commandes et les manifestes.

**La campagne principale terminée comporte huit workers et quatre paires exactes.**
Le décodage mesuré passe de 1,474 à 1,584 token/s (+7,47 %), et l'initialisation
médiane de 1,343 à 0,708 s. L'ablation gate/up isolée donne +2,36 % sur deux
paires. Les résultats et limites figurent dans `HYPERSTREAM_REPORT_FR.md`.
La première tentative, arrêtée après une référence et un refus d'admission,
reste archivée séparément et ne contribue pas à ces agrégats.

## Conditions et invariants

Matériel observé : MacBook Air M5, 16 Gio de mémoire unifiée, Metal 4, SSD interne.
Le modèle Qwen3.6 GGUF local fait 23 938 321 664 octets. Le présent protocole ne
télécharge ni ne convertit les poids. Leur format Q4_K/Q6_K, les routes réelles,
les calculs FP32 et l'ordre de réduction des experts sont conservés.

| Paramètre | Deux variantes |
|---|---|
| Cache logiciel / E/S / KV et états | 64 / 64 / 256 Mio |
| Travail / réserve système | 128 Mio / 3 Gio |
| Lecture individuelle maximale | 16 Mio |
| Tuiles / préfill / contexte maximal | 8 192 lignes / 16 tokens / 1 280 tokens |
| Précision / TF32 / cache | FP32 / désactivé / LRU |
| Experts / assemblage / transport | groupés après routage / concaténation / lecture directe |
| Import / métadonnées numériques | DLPack avec fallback exact / lecture groupée |
| Lecture disque | `pread`, cache de fichiers macOS non contrôlé |

Le candidat ajoute uniquement `--expert-pipeline --string-buffer-bytes 65536`
aux options d'exécution de la référence. Le correctif de croissance des
métadonnées du cache appartient aussi au code candidat. La campagne compare
cette combinaison ; elle n'isole pas l'effet causal de chaque changement.

Le pipeline ne traite que des projections gate/up déjà routées, contenues dans
une tuile et admissibles dans le budget simultané. Il soumet gate, prépare les
poids up, attend gate et exécute up. Une seule soumission reste en attente ; les
propriétaires des buffers sont conservés et les erreurs entraînent un drainage.
Les cas trop grands ou sous pression suivent le chemin synchrone et sont comptés.

Les gardes restent : plafond RSS du groupe 2 Gio, réserve système 3 Gio,
croissance maximale du swap 128 Mio, charge admise 768 Mio, pression normale,
32 processus maximum et échantillonnage toutes les 50 ms. Chaque essai est limité
à 600 secondes. `--resource-wait-seconds 180` attend **avant** chaque lancement
une fenêtre stable de cinq secondes, avec 256 Mio de marge au-dessus de
l'admission. Cette attente ne diminue aucun seuil et ne relance pas une inférence
refusée. Si la fenêtre manque ou qu'un worker échoue, la campagne s'arrête.

## Préparer les chemins et le plan

Depuis la racine du checkout candidat, avec un environnement Python contenant
les dépendances validées du projet :

```sh
export MODEL="/chemin/absolu/vers/le-modele.gguf"
export BASELINE="/chemin/absolu/vers/le-checkout-reference"
export AIRLLM_PYTHON="/chemin/absolu/vers/venv/bin/python"
git -C "$BASELINE" rev-parse HEAD
git diff --exit-code 83daeefde83d1714a2cc2ec1579c3c1c66e98a49 -- air_llm/airllm/nextgen
mkdir -p work/hyperstream-reproduction
```

Le premier résultat Git doit être `ca3a42935be876bbdbbc1e2adbd16c8547746175` ; le
second contrôle ne doit montrer aucune différence des sources moteur. Utiliser
des chemins de sortie neufs pour chaque tentative. Ne modifier ni sources,
HEAD, tâches, plan ni outils pendant une campagne.

Le code suivant reconstruit les huit jobs dans le même ordre que
`plan-local.json`, avec les chemins propres à la machine :

```sh
"$AIRLLM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd().resolve()
baseline = Path(os.environ["BASELINE"]).expanduser().resolve()
common = ["--expert-batch", "--concat-tiles", "--packed-direct-io",
          "--bulk-metadata", "--zero-copy-weights"]
order = [("short", 0, "baseline"), ("short", 0, "candidate"),
         ("explanation", 0, "candidate"), ("explanation", 0, "baseline"),
         ("short", 1, "candidate"), ("short", 1, "baseline"),
         ("explanation", 1, "baseline"), ("explanation", 1, "candidate")]
plan = []
for task, repeat, variant in order:
    candidate = variant == "candidate"
    plan.append({"id": f"{task}.r{repeat}.{variant}",
                 "engine_root": str(root if candidate else baseline),
                 "tasks": str(root / "research/hyperstream" / (task + ".json")),
                 "args": common + (["--expert-pipeline", "--string-buffer-bytes", "65536"]
                                   if candidate else [])})
path = root / "work/hyperstream-reproduction/plan.json"
with path.open("x") as stream:
    json.dump(plan, stream, indent=2)
PY
```

Chaque worker frais traite un seul prompt avec le vrai tokenizer, le template,
greedy et EOS. Les limites de sortie des deux tâches sont conservées dans
`short.json` et `explanation.json`. L'ordre opposé des répétitions réduit un
biais d'ordre sans contrôler le cache du système ni l'activité des autres apps.

## Mesures de composants, campagne et analyse

```sh
PYTHONPATH=air_llm "$AIRLLM_PYTHON" benchmarks/benchmark_hyperstream_metadata.py \
  --model "$MODEL" --output work/hyperstream-reproduction/metadata.json \
  --repeats 7 --string-buffer-bytes 65536 --tokenizer

"$AIRLLM_PYTHON" benchmarks/benchmark_hyperstream_cache.py \
  --baseline-ref ca3a42935be876bbdbbc1e2adbd16c8547746175 \
  --requests 100000 200000 --output work/hyperstream-reproduction/cache.json

"$AIRLLM_PYTHON" benchmarks/campaign_research51.py \
  --model "$MODEL" --tasks research/hyperstream/short.json \
  --plan work/hyperstream-reproduction/plan.json \
  --output work/hyperstream-reproduction/primary --resource-wait-seconds 180

"$AIRLLM_PYTHON" benchmarks/analyze_hyperstream.py \
  work/hyperstream-reproduction/primary \
  --output work/hyperstream-reproduction/primary-analysis
```

Le microbenchmark de métadonnées alterne les variantes dans un processus déjà
chargé, sans purge du cache macOS. Il compare métadonnées et index par empreinte,
avec et sans conservation du tokenizer, et vérifie qu'aucun payload de tenseur
n'est lu. `--tokenizer` mesure aussi sa construction séparément. Cela mesure un
parsing chaud au sens du processus ; ce n'est ni une génération ni une lecture
physique froide du SSD. Il ne faut pas extrapoler son facteur au TTFT complet.

Le microbenchmark de cache utilise des charges factices d'un octet et des clés
uniques. `tracemalloc` mesure les allocations Python des historiques et des
structures du cache, pas la RSS ni le GPU. Le temps instrumenté reste un
diagnostic ; il ne devient pas un débit d'inférence. Les sources Git de référence
et les empreintes des deux sources mesurées sont enregistrées.

L'analyse finale exige huit workers réussis et quatre paires complètes, avec
sortie EOS. Elle contrôle les gardes, options et manifestes avant/après, puis les
tokens, textes intégraux, positions et empreintes des logits complets et routes
à chaque pas. Elle produit `summary.json`, `benchmarks.csv` et `responses.md`,
avec empreintes des entrées et des sources d'analyse. Les sorties du modèle ne
sont pas réécrites. Une campagne incomplète ou une dérive sont refusées.

Les débits de décodage restitué et des pas seuls restent séparés. Le débit de
bout en bout inclut le processus supervisé ; le TTFT moteur exclut imports et
initialisation. `initialization_seconds` inclut `tokenizer_seconds` : ne pas les
additionner. Les pics RSS et MLX se recouvrent. Les lectures physiques rapportées
sont des octets comptabilisés par macOS pour le processus, sans certification de
bande passante NAND. L'identité stat du modèle est figée ; la campagne ne
recalcule pas son SHA-256 complet.

Aucun SSD externe physique n'était connecté lors de cette mission :
`/Volumes/SSD` était un lien vers `/`. Le démarrage après reboot, le cache disque
physique froid, le débit du SSD externe, les modèles GLM géants, les contextes
longs et la stabilité thermique prolongée restent non mesurés par ce protocole.

## Ablation séparée du pipeline gate/up

Cette expérience complémentaire utilise le **même checkout** pour les quatre
workers, garde le buffer de chaînes à 65 536 octets et ne change que
`expert_pipeline`. Deux paires sur le seul prompt `explanation` constituent un
petit échantillon descriptif. Depuis le checkout actuel contenant les outils
d'analyse, construire un plan neuf avec les chemins déjà définis :

```sh
"$AIRLLM_PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path.cwd().resolve()
common = ["--expert-batch", "--concat-tiles", "--packed-direct-io",
          "--bulk-metadata", "--zero-copy-weights", "--string-buffer-bytes", "65536"]
plan = [{"id": f"explanation.r{repeat}.{variant}", "engine_root": str(root),
         "tasks": str(root / "research/hyperstream/explanation.json"),
         "args": common + (["--expert-pipeline"] if variant == "on" else [])}
        for repeat, variant in [(0, "off"), (0, "on"), (1, "on"), (1, "off")]]
with (root / "work/hyperstream-reproduction/ablation-plan.json").open("x") as stream:
    json.dump(plan, stream, indent=2)
PY

"$AIRLLM_PYTHON" benchmarks/campaign_research51.py \
  --model "$MODEL" --tasks research/hyperstream/explanation.json \
  --plan work/hyperstream-reproduction/ablation-plan.json \
  --output work/hyperstream-reproduction/ablation --resource-wait-seconds 180

"$AIRLLM_PYTHON" benchmarks/analyze_hyperstream_ablation.py \
  work/hyperstream-reproduction/ablation \
  --expected-head "$(git rev-parse HEAD)" \
  --output work/hyperstream-reproduction/ablation-analysis
```

`--expected-head` exige exactement la révision enregistrée par les workers.
Il vaut `6960ef97ac4aa71d191087320362c1d636562aed` par défaut pour les preuves
locales originales. Une nouvelle campagne emploie le SHA du checkout réellement
utilisé ; cette option ne réétiquette pas les preuves historiques. Les ajouts
de rapports et d'analyse peuvent modifier le HEAD sans modifier les sources du
moteur : les manifestes permettent de vérifier ces deux faits séparément.
L'analyse exige les mêmes options, compteurs du cache et octets logiques ainsi
que la conformité des tokens, textes, logits, routes et positions. Elle vérifie
aussi que le pipeline a effectivement été utilisé dans les workers `on`.
