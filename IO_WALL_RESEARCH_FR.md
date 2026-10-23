# AirLLM Next 5 — réduction des relectures, validation du 23 octobre 2026

Le moteur a réellement évolué. Sur le MacBook Air M5 de 16 Go, avec le même
checkpoint entraîné SmolLM2-135M-Instruct, les mêmes prompts et **256 Mio de
cache logiciel dans les deux cas**, l'admission adaptative et le pipeline de
sortie ramènent la génération médiane de 16 tokens de **0,893 à 0,642 s** :
**1,39×**, avec **46,4 % d'octets logiques en moins**. Les logits enregistrés et
les séquences gloutonnes sont identiques dans toutes les variantes comparées.

Ce résultat porte sur des fichiers servis avec un cache OS non contrôlé, sur
un petit checkpoint complet. **Il ne mesure ni le SSD Fanxiang, ni une
accélération de GLM-5.3/Qwen géants.** Le disque externe n'est pas connecté :
`diskutil list external physical` ne retourne aucun disque et `/Volumes/SSD`
est un lien vers `/`, le disque interne. Le mur des E/S physiques des modèles
géants n'est donc pas déclaré résolu.

## Provenance et audit préalable

La nouvelle branche `research/io-wall-2026-10-23` a été créée depuis
`origin/research/ooc-engine-2026-10-22` (`d43032f`), puis avancée sans divergence
jusqu'aux travaux déjà publiés : `9629c27` et `b720462` sur
`research/airllm-next-5-generation`. `b720462` est la référence avant/après.
Le checkout antérieur et les preuves historiques sont conservés.
Le commit source du moteur est `1d5d768`. La publication via le connecteur
GitHub peut produire des métadonnées et SHA de commits différents ; l'égalité
des arbres Git est vérifiée et le reçu local de publication donne les deux
références.

L'audit a localisé les mécanismes réels suivants ; ils ne sont pas présentés
comme des nouveautés de cette mission :

| Chemin existant | Constat vérifiable et limite |
|---|---|
| `weights.py`, `cache.py` | Lectures exactes attendues par `SingleFlight.take`; réservations conservées jusqu'à consommation des futures. LRU/SLRU/LFU/coût existent déjà. Les scans plus grands que le cache peuvent relire les mêmes matrices à chaque token. |
| `moe.py`, `qwen_moe.py` | Routage exact, copie des IDs vers l'hôte pour connaître les offsets, chargement des seuls experts choisis ; prélecture exacte du prochain composant/expert déjà présente. |
| `storage.py` | `pread` explicite et `mmap` avec copie, retries des lectures courtes, descripteurs empruntés, regroupement borné de plages déjà implémentés. `coalesce_reads` concerne surtout payload/scales FP8 et lignes d'embedding contiguës. |
| `weights.py::output` | Projection découpée en tuiles ; chargement, calcul et réalisation auparavant séquentiels. Le cache des tuiles et le cache des données encodées existent depuis V5. |
| `attention.py`, runtimes | KV borné, réalisation par expert/couche pour ne pas retenir toute la banque dans des graphes paresseux. Ces barrières ne peuvent pas être supprimées sans revoir les durées de vie mémoire. |
| Quantification et spéculation | FP8 E4M3FN/scales, sidecars affines MLX et décodage spéculatif glouton vérifié sont déjà présents. Aucun nouveau quantificateur, kernel Metal, MTP ou décodage spéculatif stochastique n'est revendiqué. |

Les coûts de conversion BF16/FP8, les copies NumPy→MLX et les synchronisations
route/évaluation restent réels. MLX permet des calculs sur CPU/GPU partageant
la mémoire unifiée ; cela ne rend pas le SSD équivalent à la RAM.
[Documentation MLX officielle](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst).

## Code ajouté et intégré

**Cache adaptatif optionnel.** `adaptive_cache.py` classe les poids selon leur
réutilisation observée, décroissante en nombre de demandes, et leur coût de
rechargement réellement mesuré par EWMA. Une admission compare le bénéfice
estimé du candidat à la somme des victimes nécessaires. Les premiers accès
d'un scan ne chassent pas des poids déjà utiles lorsque le cache est plein.
Toutes les couches partagent le même budget, sans quota fixe par couche.

Le cache garde les objets de poids originaux. Les propriétaires complets des
vues NumPy sont comptés ; le payload MLX utilise son `nbytes`. Les plafonds par
défaut sont 4 096 résidents, 4 096 observations non résidentes et des traces
de 4 096 événements. Les métadonnées Python, l'allocateur et les références
actives appartiennent aux autres postes mémoire ; le budget du cache ne
prétend pas être le RSS total. L'admission peut trier les résidents et ralentir
un changement de phase : les comparaisons synthétiques contre LRU/LFU/coût
conservent aussi les cas défavorables.

**SpecPrefetch expérimental.** `specprefetch.py` apprend les transitions et la
réutilisation des activations de la même couche. Avant l'attention, il peut
lancer la première matrice d'un expert prédit. Après le routage réel, les
prédictions inutiles sont annulées ou drainées. Un échec prédictif retire sa
future et la demande exacte recommence sa lecture. Aucun expert prédit ne
remplace un expert sélectionné. Les plafonds couvrent 256 couches, 4 096 arêtes,
64 positions observées et 32 futures, y compris des futures de taille nulle.

Son budget est une **partition de `io_bytes`**, pas une allocation additionnelle
(`min(32 Mio, io_bytes/4)` par défaut). Une lecture active ne peut pas être
interrompue proprement par `Future.cancel` : son drain peut augmenter la
latence. Les octets utiles/gaspillés rapportés sont des payloads décodés ; le
profiler de stockage mesure séparément les octets de fichiers.

**Pipeline exact de sortie.** `weights.py::_output_pipeline` lance la lecture et
la conversion de la tuile suivante pendant le calcul courant. Chaque tuile
tient dans la moitié de l'enveloppe d'E/S ; le payload courant plus la future
reste dans l'enveloppe initiale. Les erreurs libèrent les réservations et
préservent l'erreur initiale si le nettoyage échoue aussi. Le chemin séquentiel
reste disponible. Les poids affines quantifiés sont aussi testés ; la
quantification n'est pas modifiée par ce pipeline.

**Profilage et sélection des E/S.** `io_profile.py` et `storage.py` mesurent les
octets demandés, copiés, uniques et relus, les erreurs partielles, les durées de
lecture, les modes et les couches/experts. Les unions d'intervalles sont exactes
jusqu'au plafond de 8 192 intervalles ; après saturation, les totaux exacts
deviennent `null` et seules des bornes inférieures explicites restent. Il y a
au plus 512 groupes d'attribution, incluant un groupe de débordement.

`io_mode` rend `pread` et `mmap` activables dans le lecteur natif. Le profiler
des offsets couvre les shards originaux. Pour un overlay compressé, son rapport
annonce explicitement `original_shard_fallback_only` et expose séparément les
compteurs du transport compressé. Les sidecars ont également leurs compteurs.
Le compteur physique du processus est indisponible sur ce macOS : **`null`**.

Les options nouvelles restent explicites : `policy='adaptive'`,
`spec_prefetch=True`, `output_prefetch=True`, `io_mode='mmap'`, `profile_io=True`.
Les valeurs par défaut conservent les chemins stables précédents. Les clés de
calibration incluent les nouvelles options et n'écrasent pas une politique
adaptative explicitement choisie.

Fichiers du moteur modifiés : `config.py`, `cli.py`, `weights.py`, `runtime.py`,
`moe.py`, `qwen_moe.py`, `qwen_runtime.py`, `storage.py`, `lossless.py`, `tuning.py`,
plus les trois nouveaux modules décrits ci-dessus, sous
`air_llm/airllm/nextgen/`. Les nouveaux tests sont
`test_adaptive_cache.py`, `test_specprefetch.py`, `test_io_profile.py`,
`test_io_wall_integration.py`, `test_io_wall_benchmark.py` ; le test DSA existant
est corrigé séparément. Le benchmark, les deux scripts de validation/archivage,
le README et ce rapport complètent les sources modifiées. Les fichiers de
preuves sont inventoriés dans le manifeste.

## Protocole et mesures

`benchmarks/benchmark_io_wall.py` exécute chaque variante dans un processus
frais, dans un ordre mélangé déterministe, avec les mêmes poids, entrées,
précision FP32, nombre de tokens et budgets. La première requête a un cache
logiciel vide ; les répétitions suivantes réutilisent le cache. **Aucune purge
du cache OS n'est réalisée : « processus froid » ne signifie pas « SSD froid ».**

Les temps de génération comprennent les forwards natifs, argmax/readback,
échantillonnage mémoire et drain final ; ils excluent tokenizer/construction du
modèle. Les TTFT ci-dessous sont les temps du premier forward. Les logits de
chaque étape sont conservés et comparés, indépendamment des tokens. Les essais
de temps désactivent le profiler d'intervalles dans les deux versions. Un essai
distinct l'active. Les hashes des sources sont cohérents entre les processus
des essais entraînés.

### SmolLM2 entraîné, Metal M5, cache de 256 Mio

Checkpoint complet `HuggingFaceTB/SmolLM2-135M-Instruct`, local et déjà disponible.
SHA-256 du Safetensors :
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Deux prompts de 7/9 tokens, génération fixe de 16 tokens, trois ordres de
processus, deux répétitions chaudes par prompt : 12 requêtes chaudes par variante.
E/S 32 Mio, KV 32 Mio, workspace 128 Mio, réserve système 1 Gio ; même budget
pour les quatre variantes. Les poids décodés dépassent le cache logiciel, mais
les fichiers du modèle tiennent dans le cache OS du Mac.

| Variante | Temps médian chaud | Octets logiques/requête | Appels de lecture | Débit de décodage médian | Pic RSS du processus |
|---|---:|---:|---:|---:|---:|
| Référence `b720462`, LRU | 0,893 s | 4 304 506 752 | 4 743 | 18,00 tok/s | 682,7 Mo |
| Code actuel, options nouvelles désactivées | 0,898 s | 4 304 506 752 | 4 743 | 17,90 tok/s | 681,6 Mo |
| Adaptatif | 0,683 s | 2 367 578 880 | 1 708 | 23,60 tok/s | 695,4 Mo |
| Adaptatif + pipeline | **0,642 s** | **2 305 981 440** | **1 649,5** | **25,04 tok/s** | **701,5 Mo** |

Le dernier cas réduit la latence de **28,1 %**, les octets de **46,4 %** et les
appels de **65,2 %** par rapport à la référence : **1,39×** sur la durée totale.
La TTFT médiane passe de 59,0 à 43,2 ms. Le pic RSS augmente de 18,8 Mo. La
croissance du swap système relevée pendant ces essais est nulle ; du swap
préexistait déjà, ce qui ne prouve pas l'absence de pages swappées.

[Résultats et comparaisons](evidence-io-wall/trained-cache256.json),
[processus et mesures par requête](evidence-io-wall/trained-cache256-runs/).

### Résultats négatifs et domaine de validité

| Essai | Résultat observé |
|---|---|
| SmolLM2, cache 64 Mio, 16 tokens | Adaptatif : environ −12,4 % d'octets, temps pratiquement inchangé (0,872 contre 0,872 s). Pipeline seul : 0,838 s, environ 1,04×. [Données](evidence-io-wall/trained-metal-fixed.json). |
| SmolLM2, préfixe étendu à 1 024 tokens, cache 64 Mio | Adaptatif : 13,31 contre 15,53 Go logiques/requête, mais **4,122 contre 4,058 s**. Plus lent d'environ 1,6 %. Préfixe synthétique étiqueté, pas évaluation de qualité en long contexte. [Données](evidence-io-wall/trained-context1024.json). |
| GLM miniature aléatoire, cache 32 Kio, prompts 5/48 tokens | Adaptatif : −12,38 % d'octets et −45,17 % d'appels, 30,42 contre 31,35 ms. Écart temporel trop petit pour un gain robuste. Toutes les options : 38,34 ms ; prélecture exacte seule : 46,28 ms ; SpecPrefetch seul : 40,88 ms. [Données](evidence-io-wall/glm-before-after.json). |
| Regroupement d'E/S synthétiques chaudes | 32→4 appels ; environ 1,35× sur le seul service de petits fichiers. Aucun débit physique SSD déduit. Voir le champ `storage` du rapport GLM. |

Toutes les comparaisons enregistrées ont des logits **identiques bit à bit
entre variantes du même moteur**, et les mêmes séquences gloutonnes. Cela ne
promet pas une identité bit à bit universelle entre MLX, Transformers et toutes
les machines. Les tests de référence inter-framework conservent leurs
tolérances numériques prédéfinies.

Le [diagnostic séparé](evidence-io-wall/glm-profile-diagnostic.json) observe,
pour sa référence, 9 242 880 octets lus dont 8 843 872 relus, et pour sa variante
combinée 8 373 184 dont 7 974 176 relus. Les intervalles ne saturent pas.
**Des octets relus ne sont pas automatiquement évitables** sous une enveloppe
mémoire donnée. Le diagnostic ne constitue pas une comparaison de temps.

## Tests, erreurs corrigées et artefacts

La commande reproductible est `AIRLLM_PYTHON=/chemin/python bash scripts/validate_io_wall.sh`.
Les résultats finaux sont dans [le journal CPU](evidence-io-wall/tests-final.log)
et [le journal avec options GPU](evidence-io-wall/tests-metal.log).
**720 tests réussis, 1 ignoré, 11 sous-tests réussis** dans chaque exécution :
14,14 s avec les options CPU, 15,49 s avec les options GPU. Les 720 cas répétés
ne sont pas additionnés comme des tests distincts. L'ignoré concerne CUDA.
Les XML originaux restent locaux et leurs copies `.xml.gz` sont versionnées.
La wheel de recherche a aussi été construite et installée hors checkout :
ses **65 fichiers Python** correspondent aux sources, les trois nouveaux
modules sont présents et six tokens générés par le paquet installé coïncident
avec la référence du benchmark. [Vérification](evidence-io-wall/package-verification.json).
Le numéro `5.0.0a1` reste celui de la branche alpha, sans publication PyPI.

Les nouveaux tests couvrent admission multi-victimes, scans, changement de
phase, limites de métadonnées, propriétaires NumPy, concurrence, prélecture
erronée/annulée, reprises sur demande réelle, erreurs de tuiles, erreurs
concurrentes de nettoyage, formats affines, routes/logits/tokens, budgets et
attribution du transport compressé.

L'échec de référence hérité est conservé dans `baseline-tests.log` : sa séquence
Transformers écrite en dur n'était plus celle de la référence installée. Le test
vérifie désormais sa répétabilité, puis l'égalité des tokens et les logits avec
**les mêmes tolérances**. Le [diagnostic](evidence-io-wall/inherited-dsa-diagnostic.json)
relève une erreur maximale de 5,96×10⁻⁸. La cause historique précise du changement
de séquence n'est pas établie.

Deux erreurs d'intégration ont aussi été corrigées avant livraison : le hook
GLM accédait à un champ absent du runtime Llama ; le hook Qwen accédait trop
tôt à `expert_keys` lors des tests d'interruption. Les appels prédictifs sont
maintenant conditionnels et les journaux initiaux restent présents. Aucun
résultat négatif historique n'a été supprimé.

Les rapports, compteurs, tokens et hashes sont versionnés. Les grands tableaux
de logits et les checkpoints synthétiques restent disponibles dans la livraison
locale ; [leur manifeste](evidence-io-wall/logits-manifest.json) donne les SHA-256
et les fichiers canoniques sous `../raw-evidence/`. Le script d'archivage conserve
les chemins originaux. Les benchmarks peuvent régénérer les artefacts.

## Reproduction et limites restantes

Exemple explicitement activé sur un checkpoint Llama local :

```bash
MLX_ENABLE_TF32=0 PYTHONPATH=air_llm python -m airllm.nextgen.cli run /chemin/modele \
  --backend llama --device gpu --float32 --cache-mib 256 --io-mib 32 \
  --kv-mib 32 --workspace-mib 128 --reserve-mib 1024 --max-tensor-mib 32 \
  --row-chunk 2048 --cache-policy adaptive --output-prefetch --no-prefetch
```

Pour comparer le checkpoint local avec un checkout de référence :

```bash
PYTHONPATH=air_llm python benchmarks/benchmark_io_wall.py \
  --model-dir /chemin/SmolLM2-135M-Instruct --backend llama --device gpu \
  --baseline-source /chemin/checkout-b720462 --variants reference,adaptive,adaptive_pipeline \
  --cache-bytes 268435456 --io-bytes 33554432 --max-tensor-bytes 33554432 \
  --kv-bytes 33554432 --workspace-bytes 134217728 --reserve-bytes 1073741824 \
  --max-context 512 --row-chunk 2048 --tokens 16 --repeats 3 --warm-repeats 2 \
  --output evidence-io-wall/nouvelle-comparaison.json
```

Un essai ciblé du profiler utilise `--profile-io` ; SpecPrefetch utilise
`--spec-prefetch`, avec `--spec-prefetch-mib` et `--spec-prefetch-experts`.
Le préchargement exact reste contrôlé séparément par `--no-prefetch`.

Restent non validés : SSD Fanxiang physique à froid, grand modèle MoE entraîné
complet, qualité/quantification complète GLM-5.3/Qwen, contextes de plusieurs
dizaines de milliers de tokens, thermique longue durée, énergie et utilisation
GPU. Les temps GPU purs et la durée d'inactivité GPU restent `null` : le temps
CPU du processus et les attentes de futures ne les remplacent pas.
[La synchronisation MLX porte sur un stream donné](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html).
Le pic allocateur MLX et le RSS sont exposés séparément.

La réduction des lectures et l'accélération de 1,39× sont établies pour le
protocole documenté. La prédiction n'apporte pas encore de gain démontré ici et
reste expérimentale ; aucune extrapolation vers un facteur 10 ou un modèle
géant interactif n'est faite.
