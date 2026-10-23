# Protocole natif GGUF — 23 octobre 2026

Cette campagne prolonge la PR #3, tête `5992ec7778364162dff416a27715f966bbcb3b31`.
Elle exécute le modèle texte dans AirLLM Next, sans serveur Ollama, conversion
complète, chargement des banques de 256 experts ou modification de `main`.
La branche est expérimentale ; le numéro de version du produit n'est pas
transformé en version stable 8.0 par ce travail.

## Checkpoint et matériel

GGUF local `qwen3.6:latest`, blob attendu
`f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` :
23 938 321 664 octets, contre 17 179 869 184 octets de RAM physique.
MacBook Air Apple M5, SSD interne, MLX 0.32.4. Aucun SSD externe n'est testé.
Le lecteur vérifie les offsets, les tailles, les blocs et les troncatures avant
les accès aux poids. `inspect.json` décrit les 1 194 tenseurs du fichier réel.

Le modèle texte comporte 40 couches, 256 experts, top-8 et un expert partagé ;
trois couches DeltaNet alternent avec une attention complète. Le flag GGUF de
réordonnancement des têtes V est pris en compte. Vision et MTP restent exclus.
Q4_K_M est la recette : les tenseurs présents sont F32, F16, Q4_K et Q6_K.

## Budgets et sûreté

Campagne appariée : cache 256 Mio, E/S 64 Mio, KV/états 192 Mio, travail 128 Mio,
réserve système 3 Gio, plafond RSS de l'arbre possédé 2 Gio, arrêt si croissance
du swap système >128 Mio ou pression macOS supérieure à normale. Intervalle
d'échantillonnage 50 ms, limite de durée 900 s par expérience.

Les contrôles d'allocation du lecteur, des tuiles, des états et de MLX agissent
avant les allocations prévues. Le plafond MLX du kernel packed exclut le budget
du cache NumPy pour ne pas l'accorder deux fois. Le superviseur complète ces
contrôles : il ne constitue pas une limite de RSS instantanée du noyau. Il ne
peut observer un pic entre deux échantillons. Il conserve et arrête les identités
des descendants observés, y compris détachés ; un descendant créé puis disparu
entre deux échantillons ne peut pas être identifié. Ces expériences ne créent
aucun worker enfant dans le processus d'inférence.

Le swap préexistant est rapporté séparément. Sa croissance est un compteur
système : d'autres applications peuvent déclencher un arrêt conservateur.
Aucun service personnel n'est arrêté. Les chemins de rapports, logs, gardes et
logits sont contrôlés contre les inputs, y compris liens symboliques et physiques.

## Exactitude

1. Codecs : blocs synthétiques et 28 échantillons du fichier réel, contre
   `gguf` Python officiel et les fonctions C GGML indépendantes compilées sans
   contraction flottante. Aucune approximation de codec n'est introduite.
2. Kernels : produits packed Metal comparés à un décodeur scalaire et aux
   matrices réelles déquantifiées en FP32 ; la réduction flottante diffère.
3. Architecture : modèle hybride miniature converti en conventions GGUF,
   logits et génération contre Transformers, CPU et GPU, y compris permutation
   V non involutive et rollback des états récurrents.
4. Checkpoint réel : modèle officiel Transformers construit sur `meta`, avec
   transports des poids remplacés par des lectures bornées. Les équations
   officielles d'attention, DeltaNet, routeur, experts, résidus et caches restent
   inchangées. Seul le codec GGUF est partagé ; les adaptateurs de configuration,
   noms et permutations sont réimplémentés indépendamment du backend MLX.

Les logits complets sont conservés localement et hashés. Tolérance de comparaison
prévue : `atol=2e-4, rtol=5e-5`, identités gloutonnes vérifiées séparément. La
référence Torch de quatre pas ne s'arrête volontairement pas sur EOS pour
contrôler des positions forcées ; la génération native s'y arrête. Pour le cas
math, seuls les pas avant EOS appartiennent à une génération utilisateur.
Une parité d'implémentation ne constitue pas un score de qualité du modèle.

## Campagne appariée

Trois prompts fixés dans `CASES.json`, deux répétitions, ordre aléatoire de
graine 20261023. Quatre variantes : référence déquantifiée/tuiles 2048,
Metal/tuiles 512, Metal/tuiles 2048, et Metal/2048 avec demi-vie du cache de
32768 requêtes au lieu de 1024. Tous les budgets sont identiques.
Les sorties s'arrêtent à EOS ou huit nouveaux tokens ; ce sont des générations
autorégressives réelles, pas une vérification de tokens déjà connus.

Le débit regroupé est somme des tokens / somme des durées. La génération inclut
le préremplissage, les lectures, imports, calculs, synchronisations et rapports
intermédiaires. Initialisation et tokenisation sont mesurées séparément ; le CLI
mesure aussi la durée avec tokenisation. Le décodage exclut le premier token.
Les phases 1, 4, 16 et 32 ont précédé les ablations ; les JSON portent les SHA256
des sources effectivement mesurées. Les premières explorations peuvent donc
différer du commit final de durcissement sans être effacées.

Les phases 64/128 et le contrôle `pread_nocache` sont séparés de l'ablation
primaire : ils ne peuvent pas être incorporés opportunément à son résultat.

## E/S et limites physiques

Les compteurs applicatifs, le stockage du processus macOS et le trafic NAND sont
des grandeurs différentes. Le cache OS n'est pas purgé ; `F_NOCACHE` est une
politique du descripteur et ne certifie pas un SSD froid. Les temps de workers,
de lecture et de matmul peuvent se chevaucher avec des coûts non instrumentés.
Le temps matmul mesuré encadre import éventuel et synchronisation ; ce n'est pas
un timestamp GPU isolé. Énergie, température et maintien thermique prolongé ne
sont pas mesurés.

Le parcours froid textuel est de 1,965 Go de poids par position avant résidence.
À titre d'équation, pour un débit effectif b Go/s et c Go de poids évités, le
plancher E/S est `(1.965-c)/b` secondes/token. Ni b ni ce plancher ne prouvent
un débit de génération. Les modèles 300–700B demanderaient leurs propres
architectures, poids actifs, bandes passantes et validations.

## Reproduction

```sh
export PYTHONPATH="$PWD/air_llm"
export MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m airllm.nextgen.cli gguf-inspect /chemin/du/modele.gguf
python benchmarks/benchmark_native_gguf.py --model /chemin/du/modele.gguf \
  --output evidence-native-gguf/reproduction-init.json --phase init --kernel metal
python benchmarks/campaign_native_gguf.py --model /chemin/du/modele.gguf \
  --output-dir evidence-native-gguf/reproduction-paired --tokens 8 --repeats 2
python -m airllm.nextgen.cli gguf-run /chemin/du/modele.gguf \
  --prompt 'Explique en deux phrases pourquoi le ciel est bleu.' \
  --kernel metal --row-chunk 2048 --max-new-tokens 32 \
  --report evidence-native-gguf/reproduction-cli.json
```

La validation codec optionnelle nécessite `gguf` officiel. Les tests principaux
ne téléchargent aucun poids. Voir `scripts/validate_native_gguf.sh` pour la suite
complète et les contrôles facultatifs contre le fichier local.

Le contrôle d’intégrité final se reproduit avec `research/native_gguf/hash_checkpoint.py`,
qui utilise un tampon de 4 Mio et son propre processus supervisé. Il ne duplique
pas le fichier et ne publie aucun poids.
