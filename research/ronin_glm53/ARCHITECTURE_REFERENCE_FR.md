# Audit architectural et validation numérique GLM5-Next

Le modèle a une architecture distincte des anciens GLM et de Qwen Gated DeltaNet. Le backend expérimental ajouté est un véritable décodeur textuel à lectures GGUF bornées. Sa conformité a été testée sur des miniatures indépendantes, y compris un réseau de 45 couches et 288 experts. Aucune réponse du checkpoint entraîné de 110 Go, ni aucune exécution Metal/M5, n'est établie par ces tests.

## Références inspectées

- [Transformers modeling_glm5_next.py](https://github.com/huggingface/transformers/blob/1b754e9a4636191852a1618fc588ad5c863ab4d5/src/transformers/models/glm5_next/modeling_glm5_next.py), révision `1b754e9a4636191852a1618fc588ad5c863ab4d5` : calculs KDA, indexeur KPool, mHC, routeur et MLP. Huit définitions numériques officielles, conservées sous licence Apache 2.0 et reformattées par AST sans leurs décorateurs externes, constituent l'oracle Torch isolé `tests_next/reference_glm53_torch.py`.
- [Transformers configuration_glm5_next.py](https://github.com/huggingface/transformers/blob/1b754e9a4636191852a1618fc588ad5c863ab4d5/src/transformers/models/glm5_next/configuration_glm5_next.py), même révision : contraintes NoPE, dimensions et traduction de `linear_attn_config`.
- [Configuration Z.ai BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json), inspectée pour l'architecture uniquement ; aucun poids BF16 complet téléchargé. Le GGUF cible est épinglé séparément dans `checkpoint-manifest.json` et fait autorité pour l'exécution.
- [llama.cpp glm5-next.cpp](https://github.com/ggml-org/llama.cpp/blob/74a4fb0037243f1b6196d0f990ff0aff031356cf/src/models/glm5-next.cpp) et [llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/74a4fb0037243f1b6196d0f990ff0aff031356cf/src/llama-arch.cpp), révision `74a4fb0037243f1b6196d0f990ff0aff031356cf` : représentation GGUF, séparation des projections MLA, `ssm_a` transformé et caches hybrides.
- Les tests complets appellent le paquet officiel **Transformers 5.21.0 installé**, et non l'oracle extrait ci-dessus. Le SHA256 exact de son fichier de calcul et les versions Torch/MLX/gguf sont consignés dans `numerical-validation.json`. La comparaison AST exacte des sept définitions principales (mHC, KDA, attention, routeur, indexeur, décodeur et récurrence) est positive entre le paquet installé et la révision épinglée ; résultat dans `transformers-source-comparison.json`. Cette vérification ne présume aucune équivalence entre toutes les versions.

## Structure confirmée

| Élément | Valeur du checkpoint réel |
|---|---|
| Décodeur | 45 couches, largeur 4 096, vocabulaire 154 880 |
| KDA | 34 couches ; 64 têtes × 128 canaux ; convolution causale de largeur 4 |
| DSA/MLA NoPE | Couches 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 et 43, indices à partir de zéro |
| MLA | Q LoRA 1 536 ; KV LoRA 512 ; Q/K 256 et V 256 par tête ; aucune dimension RoPE |
| Indexeur DSA | 32 têtes × 128 canaux ; KPool 4 ; budget 2 048 clés plus jusqu'à 3 clés de queue |
| mHC | 4 flux ; projection vers 24 coefficients ; 20 itérations Sinkhorn ; epsilon 1e-6 |
| FFN dense | Les trois premières couches ; largeur intermédiaire 12 288 |
| MoE | Les 42 couches suivantes ; 288 experts routés, top-8, un expert partagé ; largeur expert 2 048 |
| Routage | Sigmoid FP32 ; biais correctif pour le choix seulement ; probabilités sélectionnées renormalisées puis multipliées par 2,5 |
| SwiGLU | Gate borné supérieurement à 10 ; branche up bornée entre -10 et +10, avant SiLU et produit |
| Contexte déclaré | 1 048 576 ; ce maximum architectural ne constitue pas un budget réalisable sur 16 Gio |

Le texte ne requiert ni encodeur de vision ni couche MTP. Ces deux fonctions sont explicitement indisponibles dans le backend livré. La génération initiale est gloutonne, batch 1, sans padding. Le préfill passe séquentiellement par les mêmes calculs causaux que le décodage ; aucun HyperPrefill approximatif n'est activé.

## Différences de calcul à préserver

**KDA.** Q et K sont normalisés en FP32 par `sqrt(sum(x*x) + 1e-6)`, puis Q est divisé par `sqrt(head_dim)`. Le log-déclin a un coefficient pour chaque canal de clé : il ne peut pas réutiliser la diffusion scalaire par tête de Qwen GDN. Pour chaque token, `S = exp(g)[..., None] * S`, `delta = beta * (v - sum(S*k, axe clé))`, `S = S + k[..., None]*delta[..., None, :]`, puis `o = sum(S*q, axe clé)`. L'état récurrent reste FP32. La sortie est normalisée par tête, multipliée par un gate **sigmoid**, puis projetée.

Le GGUF contient déjà `ssm_a = -exp(A_log)`. Exponentier une deuxième fois ce tenseur changerait le modèle. Avec la borne sûre -5, le gate vaut `-5 * sigmoid((-ssm_a) * (f_b(f_a(x)) + dt_bias))`. La convolution travaille sur les projections Q/K/V brutes précédentes, puis applique SiLU.

**mHC.** Les quatre flux sont aplatis et normalisés sans poids avant la projection. Pre vaut `sigmoid(pre_w*scale+base)+epsilon`, post vaut `2*sigmoid(post_w*scale+base)`. Comb subit d'abord softmax sur les destinations, addition d'epsilon, normalisation des colonnes, puis 19 alternances lignes/colonnes avec epsilon au dénominateur. Le retour de bloc est `post * bloc + transpose(comb) @ résidu`. La tête finale prend la **moyenne non pondérée** des quatre flux, puis RMSNorm et projection vocabulaire.

**DSA/KPool.** Les clés d'indexation utilisent LayerNorm avec biais, et non RMSNorm. Les clés de chaque groupe complet de quatre tokens sont moyennées avec un softmax appris par canal sur `gate + APE`. Le score d'un pool somme, sur les têtes, `weights_proj(x)/sqrt(head_count) * relu(q@pooled_key/sqrt(index_dim))`. Seuls les pools entièrement visibles sont candidats. Les pools sélectionnés sont réétendus vers leurs vrais indices de cache ; la queue incomplète est ajoutée. L'absence de RoPE est intentionnelle. Le backend conserve les latents KV compressés et absorbe les projections K/V pendant le calcul, sans matérialiser toutes les clés/valeurs par tête.

**MoE.** Le biais correctif ne doit pas entrer dans les poids du mélange. Tous les huit experts choisis sont exécutés et leurs contributions pondérées sont additionnées, avec l'expert partagé. L'ordre `topk(sorted=False)` de Torch peut différer du tri stable des identifiants du backend ; les jeux d'identifiants sont comparés exactement et les logits avec une tolérance numérique. Une égalité FP32 exacte à la frontière du top-k MoE ou KPool est explicitement refusée par le runtime : le choix de sous-ensemble n'a pas été qualifié entre backends. Ce refus évite une substitution silencieuse d'expert ou de pool ; il constitue une limitation réelle, sans prétention à une identité bit à bit des choix aux égalités.

## Correspondance des tenseurs

Les dimensions ci-dessous suivent les tableaux Python : dimensions GGUF physiques inversées, matrice `[sortie, entrée]`.

| Groupe HF | Tenseurs GGUF / disposition Python |
|---|---|
| Embedding / head / norm | `token_embd.weight`, `output.weight` : `[154880,4096]` ; `output_norm.weight` : `[4096]` |
| mHC de chaque sous-bloc | `blk.i.hc_attn_fn.weight`, `hc_ffn_fn.weight` : `[24,16384]` ; bases `[24]` et scales `[3]` |
| Normes de couche | `blk.i.attn_norm.weight`, `ffn_norm.weight` : `[4096]` ; aucun décalage +1 de Qwen |
| Q/K/V KDA | `attn_q.weight`, `attn_k.weight`, `attn_v.weight` : `[8192,4096]` |
| Convolution KDA | `ssm_conv1d_q/k/v.weight` : `[8192,1,4]`, avec dimensions singleton supplémentaires admises |
| Forget KDA | `ssm_f_a.weight` : `[128,4096]` ; `ssm_f_b.weight` : `[8192,128]` ; `ssm_dt.bias` : `[8192]` ; `ssm_a` : `[64]`, déjà négatif exponentié |
| Beta / output gate KDA | `ssm_beta.weight` : `[64,4096]` ; `ssm_g_a.weight` : `[128,4096]` ; `ssm_g_b.weight` : `[8192,128]` ; `ssm_norm.weight` : `[128]` |
| Sortie KDA | `attn_output.weight` : `[4096,8192]` |
| Q MLA | `attn_q_a.weight` : `[1536,4096]` ; `attn_q_a_norm.weight` : `[1536]` ; `attn_q_b.weight` : `[16384,1536]` |
| KV MLA | `attn_kv_a_mqa.weight` : `[512,4096]` ; norme `[512]` ; `attn_k_b.weight` : `[64,512,256]` et `attn_v_b.weight` : `[64,256,512]`, séparés du `kv_b_proj` HF |
| Indexeur | `indexer.attn_q_b.weight` : `[4096,1536]` ; `indexer.attn_k.weight` : `[128,4096]` ; `indexer.proj.weight` : `[32,4096]` ; `indexer.k_norm.weight/bias` : `[128]` |
| Compresseur KPool | `indexer_compressor_gate.weight` : `[128,4096]` ; `indexer_compressor_ape.weight` : `[4,128]` |
| Routeur | `ffn_gate_inp.weight` : `[288,4096]` ; `exp_probs_b.bias` : `[288]` |
| Experts routés | Gate/up : `[288,2048,4096]` ; down : `[288,4096,2048]` ; l'axe expert donne les offsets SSD exacts |
| Expert partagé | Gate/up : `[2048,4096]` ; down : `[4096,2048]` |

Le vrai GGUF contient **Q2_K, Q8_0, BF16 et F32**. Les petits tenseurs KDA BF16 sont indispensables. L'inventaire complet des 1 383 tenseurs, types et offsets est fourni séparément ; l'annonce simplifiée « non-experts Q8_0 » de la model card ne suffit pas à définir le lecteur.

## Réutilisation et mémoire

Le runtime réutilise `GGUFWeights`, le lecteur GGUF à lectures positionnelles, les tuiles de lignes, les caches adaptatifs bornés, les contrôles d'entrée/génération, et les gardes mémoire. Il remplace la traduction de noms et les transformations propres à Qwen. Les anciens kernels H1/H2/H3 Q4_K/Q6_K et le récurrent scalaire Qwen ne sont pas activés sur Q2_K/KDA sans validation. Les produits MLX FP32 s'exécutent sur CPU ici ; le chemin Metal devra être mesuré sur le Mac.

Pour le modèle réel, les 34 états KDA FP32 occupent **136 Mio** fixes. Le minimum logique des historiques Q/K/V à trois tokens serait 9,5625 Mio ; le backend conserve et comptabilise les fenêtres complètes de quatre tokens, soit **12,75 Mio**. La base KDA comptabilisée est donc **148,75 Mio**. À 512 tokens, les latents DSA et clés/gates d'indexation ajoutent **16,5 Mio**. La garde réserve aussi la coexistence d'une ancienne et d'une nouvelle banque ; les temporaires de calcul et tuiles de poids ont leurs budgets séparés. Ces valeurs sont des calculs de forme, pas des mesures RSS/macOS.

Les gardes prévoient l'ensemble de la requête et la banque de remplacement avant les lectures de poids. Un échec de calcul/lecture invalide la session : elle n'est jamais réutilisée avec des états partiellement avancés. Le contrôle de contexte préalable laisse intacte une session existante. Le contexte architectural million de tokens est refusé si les budgets ne peuvent pas le contenir.

## Résultats reproductibles

Commande : `python research/ronin_glm53/numerical_validation.py`.

| Miniature | Écart absolu maximal, préfill | Écart absolu maximal, décodage avec cache | Jeux d'experts |
|---|---:|---:|---|
| 5 couches F32, largeur 8, 4 experts/top-2 | 1,78814e-7 | 8,94070e-8 | Identiques |
| 5 couches mixtes Q2_K/Q8_0/F32, largeur 256, 4 experts/top-2 | 2,59746e-6 | 2,68221e-6 | Identiques |
| 45 couches F32, largeur 8, 288 experts/top-8, 34 KDA/11 DSA | 2,08616e-7 | 1,19209e-7 | Identiques |
| 5 couches F32, KPool réellement sélectif : 3 puis 4 pools complets, 2 retenus | 1,78814e-7 | 8,94070e-8 | Identiques |

Les blocs synthétiques Q2_K/Q8_0 sont indépendamment déquantifiés par le paquet officiel `gguf` pour fournir **les mêmes valeurs** au modèle HF. Les tests ne comparent pas un checkpoint quantifié à des poids BF16 non quantifiés. Dans le cas 45 couches, 1 680 évaluations d'experts routés sont comptées pour quatre tokens de préfill et un token de décodage : 42 × 8 × 5. Le détail des erreurs absolues, relatives et L2, tolérances, identifiants routés et budgets observés est conservé dans le JSON, sans résultat de débit du vrai modèle.

Les tests `test_glm5_next_ops.py` valident également les mécanismes isolés ; `test_glm53_runtime.py` vérifie préfill, continuation, génération gloutonne, formats mixtes, sélection effective KPool, architecture de 45 couches, refus de contexte et invalidation après lecture interrompue. Résultat final ciblé : **17 réussis, 1 ignoré** en 21,96 s. Le test GPU est ignoré lorsque Metal est absent. Les fixtures de décodeur général utilisent un budget KPool assez large pour éviter des égalités artificielles de ReLU sur des petites dimensions ; la fixture séparée de sélection utilise deux projections Q opposées et les mêmes poids de tête, de sorte que le calcul officiel produit un score proportionnel à la valeur absolue du produit scalaire. Elle teste bien le retrait de pools réels sans changer le mécanisme KPool. La validation complète entraînée, la fidélité sur longue génération réelle et les mesures SSD/GPU/M5 restent à effectuer après un précontrôle de stockage réussi sur la machine cible.
