# Compatibilité GLM-5.3

Sources officielles consultées le 22 octobre 2026 :

- [AirLLM](https://github.com/lyogavin/airllm), tag v4.0.0 et corrections de main.
- [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), configuration et index à la
  révision `ba32f8c19c8738e2c57c71798c601d477a7bf844`.
- [Implémentation Transformers GLM-MoE-DSA](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa).
- [MLX](https://ml-explore.github.io/mlx/build/html/index.html) 0.32.4,
  [MLX-LM](https://github.com/ml-explore/mlx-lm).
- [MLX et M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- [Spécifications M5 Air](https://support.apple.com/en-us/126320).

Les snapshots, empreintes et résultats structurels figurent dans
`evidence-next/sources/` et `evidence-next/glm-native-audit.json`.

| Élément | Valeur vérifiée | Implémentation Next |
|---|---|---|
| Architecture | GlmMoeDsaForCausalLM / glm_moe_dsa | Chemin MLX natif |
| Paramètres modèle causal méta | 743 377 000 704 | Aucun stockage complet en RAM |
| Couches principales | 78 | Boucle de couches native |
| Dimension cachée | 6144 | Vérifiée par le plan de poids |
| Premières couches denses | 3 | MLP SiLU natif |
| Experts routés | 256 par couche concernée | Poids lus après décision réelle |
| Experts sélectionnés | 8 par token | Routage sigmoid/noaux_tc + correction |
| Expert partagé | 1 | Exécuté en plus des experts routés |
| Taille intermédiaire expert | 2048 | Gate/up/down et banques 3D supportés |
| DSA | top-k 2048 ; indexeurs full/shared | Sélection et partage respectés |
| Rang latent KV | 512 | Cache comprimé et attention absorbée |
| Rang requête | 2048 | q_a, norme et q_b |
| Têtes principales | 64 | Développement des requêtes uniquement |
| Dimensions | nope 192, RoPE 64, valeur 256 | Éviter la confusion avec head_dim sérialisé |
| Poids officiels | FP8 E4M3 par blocs 128×128 | Déquantification en poids calculables + cache |
| Contexte déclaré | 1 048 576 | Limite locale sûre par défaut : 4096 |
| MTP | 1 couche annoncée | Non exécuté ; aucune accélération revendiquée |

La configuration sérialisée contient un `head_dim` qui ne doit pas servir à
dimensionner naïvement le RoPE. Transformers le normalise à la partie RoPE ;
Next emploie explicitement `qk_rope_head_dim=64`.

L’audit structurel confronte 58 794 clés nécessaires du modèle causal à l’index
officiel et aux formes de l’architecture Transformers instanciée sur `meta`.
Il ne télécharge pas les poids. **Il ne valide pas les en-têtes ni les valeurs
des 141 shards complets.** Le chargement local effectuera les vérifications de
formes, offsets, échelles et fichiers avant leur calcul. Les poids MTP éventuels
hors couches principales ne sont pas intégrés au chemin de génération.

## Ce qui est réellement testé

Des checkpoints GLM miniatures reproduisent le routage avec groupes, le biais
de correction, les indexeurs partagés, le KV latent, les formats par expert et
les banques fusionnées FP8 3D. Les sorties sont confrontées à Transformers,
y compris lors du prefill découpé et du décodage incrémental.

Le format d’exécution original conserve les poids issus du checkpoint officiel,
mais utilise des multiplications BF16/FP32 après déquantification. Il ne
reproduit pas bit à bit la quantification dynamique des activations des moteurs
serveurs FP8. L’attention absorbée est algébriquement exacte ; les arrondis
flottants peuvent différer. Un top-k dont plusieurs scores sont égaux à la
frontière possède plusieurs sélections légales : PyTorch et MLX peuvent choisir
des indices différents. Aucun contrat de génération identique dans ces cas
n’est revendiqué. Les tests numériques sans ambiguïté de frontière et les tests
de déquantification sont séparés des évaluations de qualité.

## GLM-5.3 Flash, référence secondaire

La configuration officielle de [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
est distincte : `Glm5NextForConditionalGeneration`, modèle texte
`glm5_next_text`, 45 couches, dimension 4096, 288 experts et 8 experts par token,
avec un mélange d’attention linéaire et DSA et une composante visuelle.

Le backend Next livré **ne l’exécute pas**. L’assimiler à GLM-MoE-DSA aurait
ignoré ses états d’attention linéaire et donné un résultat incorrect. Sa
configuration est archivée pour l’étude suivante ; GLM-5.3 principal reste la
cible de cette livraison.
