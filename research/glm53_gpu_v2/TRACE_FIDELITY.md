# Attribution des premières divergences à la couche 40

Les captures de la nouvelle campagne comparent les mêmes entrées dans le modèle complet, avec CPU10/GPU0 et la politique hybride CPU10/GPU6, cache384, placement HC local et frontière de cache réparée. Les traces initiale et détaillée conservent les sorties nommées des opérations de la couche 40, sans capturer tous les nœuds internes du graphe. L'ordre présenté par `compare_trace.py` est l'ordre d'évaluation de la capture CPU ; ce n'est pas un tri lexical des noms.

Le [premier comparatif](evidence/trace40-comparison.json) observe au préfill :

- Première sortie nommée différente en bits : `hc_mixes-40`, MUL_MAT, maximum absolu `7.62939453125e-6`, encore dans la tolérance Metal annoncée.
- Puis `hc_attn_pre-40`, maximum `1.7881393432617188e-7`, `norm-40` (RMS_NORM), maximum `7.152557373046875e-7`, et `attn_norm-40` (MUL), maximum `2.384185791015625e-7`.
- Première sortie nommée hors tolérance : `kda_q_conv-40`, maximum `0.004960060119628906`, 16 valeurs hors tolérance parmi 49 152.

`kda_q_conv-40` est une **vue RESHAPE après la convolution et SiLU**, pas le nœud qui effectue la projection Q ni la convolution elle-même. `glm5_conv1d` exécute auparavant MUL_MAT, reshape, transpose, concaténation avec l'état, copie de l'état, SSM_CONV et SiLU. Qualifier ce RESHAPE de première opération arithmétique fautive serait incorrect. `hc_mixes` reçoit aussi un RMSNorm de l'aplatissement des quatre flux HC qui n'était pas nommé dans la capture initiale.

## Sonde supplémentaire désactivée par défaut

[patches/llama-hc-kda-trace.patch](patches/llama-hc-kda-trace.patch) ajoute uniquement des callbacks de nommage quand `AIRLLM_TRACE_HC_KDA=1` :

| Nom | Opération réellement nommée | Position |
|---|---|---|
| `hc_flat` | RESHAPE | Quatre flux résiduels avant leur normalisation HC |
| `hc_flat_norm` | RMS_NORM | Entrée de la projection HC F32 |
| `kda_q_proj` | MUL_MAT | Projection Q avant convolution et SiLU |
| `kda_k_proj` | MUL_MAT | Projection K avant convolution et SiLU |
| `kda_v_proj` | MUL_MAT | Projection V avant convolution et SiLU |

Les références du callback existant, du microbatch et du numéro de couche sont passées au helper statique de convolution. Aucun tenseur de calcul, copie, normalisation ou multiplication n'est ajouté, déplacé ou dupliqué. La séquence ordonnée des 229 appels GGML du fichier source reste identique. Le patch s'applique au commit natif épinglé `3f8af15a76ac8675812e856da45454bafc90f12c` et ne chevauche pas le patch de kernels Q8. Après vérification d'application, il a été compilé dans le runtime isolé puis utilisé pour les deux nouvelles captures détaillées CPU/hybride.

Le callback existant peut modifier le nom d'un nœud et la capture peut introduire une synchronisation. Il faut activer la même instrumentation dans les deux bras de l'expérience. Ces traces servent à attribuer la dérive, jamais à mesurer le débit. L'instrumentation de performance doit être désactivée lors des comparaisons de vitesse acceptées.

## Attribution confirmée par la trace détaillée

Le [comparatif détaillé](evidence/trace40-detailed-comparison.json) précise l'ordre au préfill de six tokens :

| Sortie capturée | Opération | Écart absolu maximum | Valeurs différentes en bits | Valeurs hors tolérance |
|---|---|---:|---:|---:|
| `hc_flat-40` | RESHAPE | 0 | 0 / 98 304 | 0 |
| `hc_flat_norm-40` | RMS_NORM | 9,5367431640625e-7 | 30 391 / 98 304 | 0 |
| `kda_q_proj-40` | MUL_MAT | 0,01267658919095993 | 49 152 / 49 152 | 31 614 |

La première différence **parmi ces sorties nommées** apparaît donc à la normalisation HC, sur une entrée `hc_flat` identique. La première sortie hors tolérance est la projection Q, avant convolution et SiLU. La trace ne prouve pas l'absence d'un écart dans tout nœud interne non capturé, ni dans toute autre couche.

Le poids `blk.40.hc_attn_fn.weight` est F32, de forme `[16384,24]`. Les poids Q/K/V de cette couche sont Q8_0, de forme `[4096,8192]`. Le premier écart HC ne peut donc pas être automatiquement attribué à une déquantification Q8 de ce poids HC.

Dans le code CPU `ggml/src/ggml-cpu/ops.cpp:4082-4145`, RMSNorm accumule les carrés FP32 dans `ggml_float`, défini comme `double`, puis arrondit la moyenne en FP32 et applique `sqrtf`. Le kernel Metal `kernels/norm.metal:111-175` accumule des produits et une réduction parallèle en FP32 avant la racine. Les méthodes d'accumulation diffèrent dans le code, et la nouvelle capture localise effectivement un premier écart à cette opération sur une entrée identique. Le produit matriciel HC F32 dispose ensuite lui aussi de chemins de réduction CPU/Metal différents.

Pour les projections Q8, le CPU quantifie temporairement les activations en Q8_0. Metal standard consomme les activations F32 ; pour les six tokens du préfill il peut sélectionner un chemin matriciel qui diffère aussi du produit vectoriel CPU. Le CPU peut alors appeler sa variante `nrc=2`, avec une accumulation MMLA distincte de `nrc=1`. Les entrées de la projection Q ont déjà divergé : ces captures et l'inspection du dispatch n'isolent donc pas la fraction d'erreur attribuable à chaque différence arithmétique.

La voie compatible décrite dans [Q8_FIDELITY.md](Q8_FIDELITY.md) reproduit le cas vectoriel CPU `nrc=1`. Sa politique sélective finale laisse les produits multi-colonnes du préfill, RMSNorm, HC, états/récurrences KDA et routage sur CPU; seuls les produits vectoriels Q8 éligibles dont les poids sont placés Metal passent sur GPU, y compris certaines projections Q/K/V au décodage. Elle conserve aussi le placement CPU des couches pour préserver les décisions de fusion HC. Le contrôle complet à 16 étapes de cette politique est bit à bit identique pour les logits, routes ordonnées, routeurs et sorties KDA capturées. C'est une correction par contrat arithmétique et placement limité; elle ne rend pas fidèle le chemin hybride GPU6 général. Aucun seuil n'est relâché et aucun ordre d'experts n'est corrigé artificiellement.
