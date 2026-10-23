# Mesures GGUF natives — analyse appariée

**État : CAMPAGNE ET VALIDATIONS TERMINÉES** — 24/24 essais terminés et gardés.

Calcul régénéré depuis `evidence-native-gguf/paired/campaign.json`. Aucun modèle ni poids n’est chargé par ce script.

Protocole : 3 prompts, 2 répétitions, ordre randomisé (seed 20261023), huit tokens maximum avec arrêt EOS.

| Variante | Essais | Tokens | Génération tok/s | Décode tok/s | Mur worker tok/s | Processus supervisé tok/s | TTFT médian s | RSS pic Mio | Swap croissance Mio | Hits cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 6 | 36 | 0.157 | 0.198 | 0.155 | 0.154 | 8.722 | 495.4 | 0.000 | 0.27% |
| metal512 | 6 | 36 | 0.333 | 0.470 | 0.326 | 0.324 | 4.446 | 538.8 | 0.000 | 1.16% |
| metal2048 | 6 | 36 | 0.621 | 0.942 | 0.592 | 0.586 | 3.119 | 678.2 | 0.000 | 1.09% |
| metal-decay | 6 | 36 | 0.634 | 0.947 | 0.605 | 0.598 | 2.278 | 713.5 | 0.000 | 12.18% |

Les débits sont des rapports de sommes. La colonne décode exclut le premier token de chaque essai; génération inclut préfill et boucle modèle. Le mur worker ajoute son initialisation. Le **processus supervisé**, `somme(tokens) / somme(guard.elapsed_seconds)`, inclut imports Python, initialisation et nettoyage. Les IDs d’entrée sont pré-fournis : tokenisation et rendu du chat restent un essai séparé. Aucun débit génération n’est appelé ici débit complet tous coûts. La variante `reference` est le chemin AirLLM déquantification CPU + calcul MLX; Torch est un comparateur numérique indépendant, pas cette base de vitesse.

| Variante | Poids logiques Go/token | Stockage processus Go/token | CPU s/token | Cache pic Mio |
|---|---:|---:|---:|---:|
| reference | 2.426 | 0.577 | 5.102 | 256.0 |
| metal512 | 2.405 | 0.751 | 1.344 | 256.0 |
| metal2048 | 2.409 | 0.727 | 0.721 | 256.0 |
| metal-decay | 2.260 | 0.686 | 0.747 | 256.0 |

Go = 10⁹ octets, Mio = 2²⁰ octets. Le stockage processus est un compteur macOS, pas une mesure NAND. Le temps CPU ne mesure pas les commandes GPU. Les métriques par prompt et répétition restent disponibles dans le JSON.

## Comparaisons appariées

| Variante / référence | Paires valides | Gain génération groupé | Gain décode groupé | Gain processus supervisé | Gain génération géométrique |
|---|---:|---:|---:|---:|---:|
| metal512 / reference | 6 | 2.129× | 2.380× | 2.095× | 2.165× |
| metal2048 / reference | 6 | 3.968× | 4.765× | 3.797× | 3.870× |
| metal-decay / reference | 6 | 4.050× | 4.791× | 3.872× | 4.156× |

Les paires associent le même prompt et la même répétition, avec budgets, sources et IDs identiques. Des résultats partiels ne permettent pas de conclure au meilleur réglage. Le gain contre `reference` regroupe kernel, transport et représentation du cache.

| Ablation Metal | Paires valides | Gain génération groupé | Gain décode groupé | Gain processus supervisé |
|---|---:|---:|---:|---:|
| tiling_512_to_2048 | 6 | 1.864× | 2.002× | 1.812× |
| cache_decay_1024_to_32768 | 6 | 1.021× | 1.005× | 1.020× |

La première ablation change la taille des tuiles compactées; la seconde change seulement la demi-vie du cache adaptatif de 1 024 à 32 768 demandes, à tuiles et budget égaux. Un ratio inférieur à 1 indique un ralentissement.

## Exactitude et portée

| Prompt | Séries natives présentes / attendues | IDs identiques entre séries | Logits finaux ≤4 positions contre Torch | Écart absolu maximal |
|---|---:|---|---|---:|
| hello | 8/8 | oui | passed | 0.00000381 |
| math | 8/8 | oui | passed | 0.00002098 |
| chat | 8/8 | oui | passed | 0.00001073 |

24/24 préfixes disponibles correspondent à la référence Torch indépendante jusqu’à quatre positions, arrêt EOS inclus. Les NPY des séries de huit pas ne contiennent que le dernier vecteur : ils ne sont comparés à Torch que si sa position est couverte. Les positions cinq à huit ne reçoivent aucune validation indépendante ici.

Empreintes sources cohérentes entre essais présents : oui. Les empreintes exactes et comparaisons du premier token historique figurent dans le JSON.

## Générations prolongées — stabilité séparée

| Rapport | État | Tokens obtenus / demandés | Génération tok/s | Décode tok/s | Mur worker tok/s | Processus supervisé tok/s | RSS pic Mio | Swap croissance Mio | Même source que campagne |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| long128-baseline.json | completed | 128/128 | 0.988 | 1.003 | 0.984 | 0.983 | 607.9 | 0.000 | oui |
| long128-decay.json | completed | 128/128 | 0.955 | 0.969 | 0.951 | 0.951 | 628.0 | 0.000 | oui |
| long64-baseline.json | completed | 64/64 | 0.902 | 0.913 | 0.896 | 0.895 | 615.6 | 0.000 | oui |
| stage32-metal.json | completed | 32/32 | 0.964 | 0.989 | 0.951 | 0.948 | 589.3 | 0.000 | non, historique |

Ces essais mesurent durée, débit et sûreté, sans validation indépendante de toutes les positions générées. Les historiques de source différente ne contribuent pas aux gains appariés; un préfixe validé ne certifie pas les 64 ou 128 tokens suivants.

Les 128 IDs baseline/decay sont identiques; le préfixe de 64 IDs est identique. À 128 tokens, decay réduit les lectures logiques de 12.66 %, mais son débit génération varie de -3.36 % et son temps CPU de 7.88 %. Résultat négatif conservé : aucun changement du défaut n’est justifié. Une seule répétition par réglage ne démontre pas une cause universelle.

## Contrôle F_NOCACHE séparé

8 tokens, 0.544 tok/s dans la boucle, 0.523 tok/s processus supervisé. 15.531 Go logiques, 5.199 Go de stockage comptabilisés par macOS. Pic RSS 583.5 Mio, croissance maximale du swap 0.000 Mio.

F_NOCACHE est une politique du descripteur. Cet essai court, non apparié à un cache purgé, ne certifie ni trafic NAND ni débit SSD physique; le décalage entre octets logiques et compteur OS doit rester visible.

## Limites et sources

- Checkpoint Qwen3.6 GGUF quantifié testé, sans certification de la qualité du checkpoint BF16 original.
- Trois prompts, deux répétitions, greedy batch un et huit tokens maximum; EOS peut raccourcir la génération.
- La référence indépendante exécute les blocs officiels Transformers/Torch mais partage le lecteur et les codecs GGUF.
- La référence Torch indépendante couvre quatre positions maximum par prompt; les tokens suivants sont comparés entre variantes natives seulement.
- Trois niveaux de coût: génération (préfill et boucle modèle), mur worker (inclut son initialisation), processus supervisé (imports Python, initialisation, génération et nettoyage, depuis guard.elapsed_seconds). Le dernier est le débit tous coûts processus de cette campagne.
- Les IDs d’entrée sont pré-calculés dans la campagne: la tokenisation et le rendu du chat sont validés séparément et restent exclus même du temps processus supervisé de ces essais.
- TTFT mesuré après initialisation; la somme initialisation+TTFT reste distincte du démarrage complet du processus.
- Cache OS non purgé et état non contrôlé, malgré ordre de campagne randomisé. Les compteurs stockage sont ceux du processus selon macOS, avec réadahead; ils ne certifient ni octets NAND ni bande passante physique du SSD.
- Même budget cache mais représentations différentes: référence déquantifiée, Metal compacté. Les gains mesurent le chemin complet et son cache, pas le kernel GPU isolé.
- cpu_seconds mesure du temps CPU processus. matmul_seconds inclut imports, lancements et synchronisations; aucun temps GPU isolé ou bilan énergétique ne peut en être déduit.
- RSS échantillonné et swap système complètent les réservations avant allocation. La croissance du swap est système, avec un swap initial déjà présent; absence de croissance ne signifie pas swap nul.
- Identité des tokens avec la référence ne prouve pas une réponse sémantiquement juste: le prompt brut 2+2= produit 5 puis EOS sur ce checkpoint.
- Chemin texte uniquement: vision et MTP non exécutés. Aucun résultat SSD externe, long contexte, thermique prolongé ou modèle 300–700B n’est établi.
- Ces mécanismes ont des antériorités documentées; les mesures locales ne prouvent aucune nouveauté algorithmique.

Sources de format et de calcul : [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [GGML C](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c), [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py), [Metal MLX](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). Les versions installées et SHA256 font foi pour les expériences. Antériorités détaillées dans `PRIMARY_SOURCES.md`; ces liens ne remplacent pas les mesures locales.
