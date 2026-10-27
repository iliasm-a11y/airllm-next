# HyperStream 2 — protocole reproductible

Branche de recherche issue exactement de la tête de la PR #10 :
`c8f81e57463ed3d7fd3d7dba226a65500439a9ce`. Ne pas fusionner #10 pour
reproduire cette recherche. Les poids restent dans leur GGUF d'origine.

## Activation et limites

`NativeGGUFQwenModel(..., hyper_prefill=True, hyper_prefill_max_tokens=128)`
active uniquement le choix de blocs d'entrée. La valeur par défaut reste
`False`. Le CLI `gguf-run` propose les mêmes options. Les options de pipeline
et de transport d'HyperStream 1 sont indépendantes.

Le planificateur considère les activations, les résultats MoE, l'attention
causale et les états KV/récurrents/convolutionnels. Il garde la moitié du budget
de travail pour les tuiles et opérations transitoires, et sélectionne de nouveau
la taille à chaque bloc. Le décodage reste autorégressif à un token. Aucun
budget n'augmente. Les contrôles réels du lecteur, du contexte, de MLX et du
superviseur restent prioritaires : une estimation admise peut encore être
refusée par ces contrôles, avec restauration du contexte.

Un bloc plus grand change la forme de certaines opérations flottantes. Les
poids, les experts et les kernels sont conservés, mais cela ne démontre pas
l'égalité bit à bit. Le protocole impose les mêmes tokens, toutes les routes
par token, les positions et les conditions d'arrêt. Chaque logit FP32 brut doit
respecter `abs(candidat-référence) <= 1e-4 + 1e-4*abs(référence)` ; le contrôleur
signale séparément l'égalité bit à bit. Un échec interdit de déclarer la
variante numériquement validée.

## Tests portables

Sans dépendance ni GPU :

```sh
python3 -m unittest discover -s tests_next -p test_request_trace.py
python3 -m unittest discover -s tests_next -p test_hyperstream2_observer.py
python3 -m unittest discover -s tests_next -p test_verify_hyperstream2.py
python3 -m unittest discover -s tests_next -p test_hyperprefill_stdlib.py
python3 -m unittest discover -s tests_next -p test_hyperstream2_harness.py
```

La CI Ubuntu exécute ces contrôles. La suite complète `tests_next` requiert
MLX et PyTorch. Sur un vrai Mac avec les dépendances du dépôt :

```sh
MLX_ENABLE_TF32=0 python3 -m pytest -q tests_next/test_hyperprefill.py
MLX_ENABLE_TF32=0 python3 -m pytest -q tests_next tests_ultra
```

Le premier fichier couvre aussi des logits indépendants sur modèle miniature,
le routage, le cache, EOS et la restauration après erreur. Cela ne remplace pas
la campagne du checkpoint réel.

## Campagne M5

Préparer deux copies indépendantes : la référence au commit indiqué et le
candidat au commit publié. Utiliser le même interpréteur et checkpoint local,
sans téléchargement, purge système ou autre génération en parallèle.

```sh
python3 benchmarks/plan_hyperstream2.py \
  --baseline /chemin/reference-pr10 --candidate /chemin/candidat \
  --pairs 10 --max-chunk 128 --output work/hs2-plan.json

MLX_ENABLE_TF32=0 python3 benchmarks/campaign_research51.py \
  --model /chemin/checkpoint.gguf \
  --tasks research/hyperstream2/context.json \
  --plan work/hs2-plan.json --output work/hs2-primary \
  --resource-wait-seconds 600
```

L'ordre est AB, BA, AB, BA ; chaque essai utilise un processus frais. L'absence
de fenêtre mémoire normale diffère l'essai. Ne pas réduire la réserve de 3 Gio,
le plafond RSS de 2 Gio, la limite de croissance du swap de 128 Mio, la garde
de pression normale ou les budgets du modèle. Les 768 Mio d'admission sont
inchangés. Enregistrer séparément le swap préexistant.

Calculer le SHA-256 intégral du checkpoint **avant et après** avec identité de
fichier stable. Le calcul d'intégrité doit lui aussi respecter les ressources
disponibles. Copier `comparison-policy.json` dans le dossier de résultats sous
le nom `campaign-integrity.json` et y ajouter, avec les valeurs réellement
mesurées :

```json
{"model_integrity":{"before_sha256":"SHA256_AVANT","after_sha256":"SHA256_APRES","bytes":0,"stable_file":true}}
```

Ce fragment est un schéma, pas une preuve. Les valeurs fictives seront
refusées. Conserver les JSON, gardes, journaux et fichiers `.f32` produits.

```sh
python3 benchmarks/verify_hyperstream2.py work/hs2-primary \
  --check-current-inputs --output work/hs2-verification.json
```

Le vérificateur n'importe ni moteur ni NumPy. Il recalcule les différences
numériques avec `struct`, les empreintes et les statistiques appariées. Il
rejette les campagnes incomplètes ou les contrôles relâchés. Il vérifie les
preuves enregistrées ; il ne peut pas authentifier rétroactivement une mesure
ni prouver le trafic physique du SSD.

Faire aussi une campagne avec `plan_hyperstream2.py --no-observer`, dans un
autre dossier, pour corroborer les temps sans copie des routes ni
instrumentation des méthodes. Cette campagne de temps seule ne remplace pas
la validation numérique.

## Profilage séparé

`benchmark_research51.py --request-trace --numerical-evidence` produit la trace
monotonique détaillée et les logits d'une unique tâche. Les traces distinguent
attention, MoE, projections, préparation des poids, appels de lecture et attentes
MLX. Les phases `prefill` et `decode` sont séparées. Le premier token est choisi
dans les logits du préfill ; il ne nécessite pas un second passage de décodage.

Les compteurs de tokenizer et de construction du modèle sont distincts.
`initialization_seconds` contient le tokenizer ; ne pas l'ajouter une seconde
fois. Le TTFT moteur commence après préparation du prompt dans l'adaptateur.
Le temps supervisé inclut les imports et la préparation du processus ; l'attente
d'admission est publiée séparément. Les traces ne créent aucun appel
supplémentaire à `mx.eval` ou `mx.synchronize`.

Les durées de méthodes imbriquées se recouvrent. Les unions d'intervalles
donnent la couverture temporelle ; leurs catégories ne s'additionnent pas.
`mx.eval`, le parsing Python et la copie des IDs sont des temps muraux côté
hôte. Ils ne mesurent pas un temps Metal pur. L'instrumentation possède un
coût : ne pas l'utiliser comme mesure principale de vitesse.

Le cache de pages macOS n'est pas contrôlé. Un worker frais ou `F_NOCACHE` ne
crée pas une condition froide prouvée. Les octets logiques et les octets de
stockage imputés au processus sont distincts. Aucune bande passante NAND,
utilisation GPU ou qualification thermique ne découle de ces compteurs.

## Portée

La nouvelle sélection est intégrée au backend GGUF Qwen35 MoE expérimental.
Elle ne certifie ni Qwen3.8 BF16, ni d'autres familles, ni SSD externe, ni modèle
de 100 Go à 1 To. Augmenter les blocs amortit les lectures répétées ; cela ne
supprime pas les poids nécessaires, les experts réellement choisis ou les
limites physiques de bande passante et de contexte.
