# GLM-5.3 Flash : expérience GPU fidèle et bornée sur M5

Branche expérimentale issue du commit `0ab4ad0` de la PR #17. Aucun fichier du moteur de production, des poids, du modèle RONIN ou de son ordonnanceur n'est modifié. Les 600 identités logiques, le slot physique unique et son admission mémoire restent ceux de PR #17. Aucun gain nouveau de décodage complet n'est qualifié.

## Résultats mesurés le 2 novembre 2026

MacBook Air M5, 8 cœurs GPU, 16 Gio. Le modèle GGUF Q2_K_S de 110 027 352 864 octets est sur le disque interne APFS. Aucun SSD Fanxiang n'est monté : ce travail ne mesure pas les performances de ce périphérique. RONIN a été arrêté avec autorisation avant les nouvelles inférences ; les processus de diagnostic sont arrêtés à leur fin. Les caches macOS ne sont pas contrôlés.

| Expérience | Fidélité mesurée | Décision |
|---|---|---|
| Runtime patché, options expérimentales désactivées, CPU10 | 619 520 logits et toutes les familles capturées identiques en bits à la référence CPU | Référence conservée |
| Tête de sortie seule sur Metal, reste CPU | Routes/routeurs/KDA identiques en bits ; logits max 0,05558395 | Rejetée |
| CPU avec HC post décomposé | Reproduit exactement les 4 522 248 éléments capturés avant la couche 40 de l'ancien hybride | Cause du préfixe divergent isolée sur cette trajectoire |
| Hybride GPU6 avec placement HC local et frontière de cache corrigée, 384 Mio | Termine ; logits max 0,12101674 ; huit positions de routes diffèrent | Rejeté pour fidélité |
| Kernel Metal Q2_K × activations Q8_K, un expert gate réel 2048 × 4096 | 2048 sorties et activations temporaires identiques en bits au produit natif ARM | Fidélité de composant seulement |
| Rejeu de huit experts gate sélectionnés, deux passages, cache asynchrone | 32 768 sorties identiques en bits ; 21 Mio de données réservées au maximum | Intégration bornée d'un composant seulement |

Les quatre étapes natives utilisent les mêmes tokens d'entrée imposés, le même contexte 512 et batch/ubatch 8. Elles ne constituent pas un test soutenu, jusqu'à EOS ou de tous les contextes. Les captures synchronisent des opérations ; leurs temps muraux ne sont pas des débits de décodage.

Le kernel fidèle reste **plus lent** sur l'expert chaud testé : médianes de cinq répétitions CPU1 0,454 ms, CPU10 0,553 ms, GPU 1,309 ms. Le comparatif CPU utilise un appel natif pour toute la matrice et un groupe de threads persistant ; il évite le coût de 2048 appels Python. Ce composant en mémoire ne permet aucune extrapolation en tokens/s. La référence historique CPU10 GPU0 reste 0,1085–0,1664 tok/s sur les scénarios de PR #17.

Le cache natif de 256 Mio historique affichait zéro hit. Le pilote corrigé de 384 Mio obtient 42 hits / 78 misses (35 %), 614,25 Mio chargés, sans croissance du swap. La comparaison des deux budgets avec les mêmes correctifs conserve tous les logits et tenseurs capturés identiques en bits ; le budget de 256 Mio reste à zéro hit et 945 Mio chargés. Un taux de hits supérieur n'est pas un gain de latence démontré.

Un pilote de génération distinct, **sans capture de tenseurs**, compare ensuite CPU10/GPU0 à CPU10/GPU6/cache384 avec les correctifs, même prompt de six tokens et sortie de huit tokens :

| Pilote court, une répétition par configuration | Débit observé hors premier token | TTFT hors chargement |
|---|---:|---:|
| CPU10 GPU0 | 0,173368 tok/s | 20,965 s |
| CPU10 GPU6 corrigé, cache384 | 0,182671 tok/s | 20,987 s |

L'écart observé de 5,37 % n'est **pas un gain qualifié** : candidat rejeté pour fidélité, une seule paire courte, CPU puis GPU, cache de fichiers non contrôlé et pas de mesure Fanxiang. Les huit IDs générés sont identiques, ce qui ne remplace pas le contrôle des logits/routes. Aucune croissance du swap dans ces deux pilotes ; aucun objectif d'accélération considérable n'est atteint. Les temps du diagnostic instrumenté ne sont pas utilisés dans ce tableau.

## Contenu et limites d'intégration

- [AUDIT.md](AUDIT.md) et `audit_*.py/json` expliquent les divergences, le changement global de fusion mHC et les compteurs I/O.
- `patches/llama-layer-local-hc.patch` conserve les sorties HC sur le dispositif de leur couche avec `AIRLLM_GLM_LAYER_LOCAL_HC=1`. `AIRLLM_GLM_HC_POST=0` sert uniquement à l'ablation.
- `patches/llama-moe-cache-split.patch`, activé par `AIRLLM_GLM_CACHE_SPLIT=1`, sépare la production des IDs de la consultation du cache. L'assertion qui interdit de lire des IDs non calculés reste active. Le premier pilote sans ce correctif a avorté ; son échec est conservé.
- `quantized.py` et `kernels/` lisent directement les blocs Q2_K GGUF d'origine. Ils fusionnent décodage des blocs et produit scalaire, sans grande matrice F32 intermédiaire. Le mode `cpu_q8k` reproduit la quantification temporaire des activations et l'ordre FMA du CPU ARM ; les poids ne sont pas requantifiés. Le mode `float` est un témoin numérique, pas un remplacement fidèle au CPU.
- [CACHE.md](CACHE.md), `expert_cache.py` et `stream_selected.py` relient lecture asynchrone, cache borné, ordre des routes capturées et kernel Metal. Seuls les experts déjà sélectionnés sont lus. Les activations du rejeu sont synthétiques ; ce code ne remplace pas le graphe GLM complet ni le cache natif de llama.cpp.

Les nouveaux kernels MLX ne sont pas branchés dans le serveur RONIN. Le pilote natif utilise les kernels GGML existants, encore divergents. Une intégration générale nécessiterait aussi Q8_0, BF16, KDA/mHC et leurs réductions fidèles. Aucun usage effectif des Neural Accelerators/TensorOps n'est revendiqué pour les nouveaux kernels SIMD.

## Reproduction

Les dépendances de recherche sont Python, NumPy, pytest et MLX 0.32.4 sur Apple Silicon. `psutil` permet l'admission mémoire du rejeu. Aucun téléchargement des poids ni conversion n'est nécessaire.

Depuis la racine AirLLM Next, préparer une **copie isolée** de llama.cpp au commit `3f8af15a76ac8675812e856da45454bafc90f12c`. Appliquer dans l'ordre les quatre patches de `research/glm53_m5/patches/` : mmap-demand, expert-prefetch, dense-prefetch, dense-prefetch-columns. Appliquer ensuite les deux patches de ce dossier. Les expériences restent désactivées sans les drapeaux ci-dessus.

```sh
cmake -S /chemin/llama-isole -B /chemin/build-isole -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build /chemin/build-isole --target llama-server -j 2
xcrun clang++ -std=c++17 -O3 \
  -I /chemin/llama-isole/include -I /chemin/llama-isole/ggml/include \
  research/glm53_gpu/native_probe.cpp -L /chemin/build-isole/bin \
  -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,/chemin/build-isole/bin -o /chemin/native-probe
```

Créer une capture CPU, puis forcer ses entrées dans les candidats :

```sh
python -m research.glm53_gpu.run_native --model /chemin/premier-shard.gguf \
  --probe /chemin/native-probe --output /chemin/cpu
python -m research.glm53_gpu.run_native --model /chemin/premier-shard.gguf \
  --probe /chemin/native-probe --output /chemin/hybride --reference /chemin/cpu/capture \
  --policy weights-only --gpu-layers 6 --cache-mib 384 \
  --layer-local-hc --cache-lookup-split
```

Le superviseur fixe RSS à 8,5 Gio, croissance du swap à 32 Mio et durée à 400 s ; il ne tue que le processus qu'il lance. Le verrou est commun au benchmark natif existant. Il faut aussi libérer le slot RONIN avant de tester. Sortie 2 : processus arrêté/échoué ; sortie 3 : capture terminée mais fidélité rejetée. Les tolérances de PR #17 restent inchangées : routes/IDs exacts, `atol=1e-3`, `rtol=1e-4` pour Metal, comparaisons CPU/I/O en bits. Aucune capture incomplète ne peut être acceptée.

Pour le rejeu des experts et le microbenchmark avec oracle natif, voir [CACHE.md](CACHE.md) et [quantized-notes.md](quantized-notes.md). Leurs budgets de payload ne couvrent pas tout le processus ; conserver le superviseur RSS/swap. Lorsqu'un Python de venv est un lien symbolique, lancer le superviseur avec `/usr/bin/env /chemin/venv/bin/python ...` pour garder la résolution du venv.

Les preuves sélectionnées, les résumés de captures et les empreintes sont dans `evidence/`. L'archive locale de livraison contient les captures binaires et leurs manifests nécessaires au recalcul. Les fichiers de poids ne sont jamais inclus. Les tailles/mtime ont été vérifiées ; aucun nouveau SHA-256 complet des 110 Go n'a été calculé.
