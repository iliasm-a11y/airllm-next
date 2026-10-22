> Historical delivery document. Current candidate status, installation and evidence: [README](README.md) and [FINAL_VALIDATION](FINAL_VALIDATION.md). The original report below is preserved.

# AirLLM Next — livraison du 22 octobre 2026

Évolution réelle d’AirLLM v4.0.0 et de la livraison Ultra fournie. Le nouveau
backend MLX exécute GLM-MoE-DSA directement : routeur, experts, attention DSA,
cache KV latent et projection de sortie. L’entrée AirLLM reste `AutoModel`.

**État : exécutable et testé sur MLX CPU Linux. La validation Metal/M5, le
checkpoint complet de GLM-5.3 et le SSD Fanxiang restent à réaliser.**
Les fichiers `evidence/` et `benchmarks/results-*` sont les preuves historiques
Ultra. Les nouveaux résultats sont uniquement dans `evidence-next/`.

## Installation sur le Mac

Décompresser cette archive, ouvrir un terminal dans son dossier `airllm-next`,
puis exécuter :

```bash
bash scripts/setup_next_macos.sh
bash scripts/validate_next_macos.sh
```

Python natif Apple Silicon 3.12 ou 3.13 requis. Le script vérifie la plateforme,
macOS et l’espace libre avant l’installation des dépendances. Aucun poids de
modèle n’est téléchargé. MLX nécessite macOS 14 au minimum ; les Neural
Accelerators M5 utilisés par MLX demandent macOS 26.2 ou ultérieur selon Apple.

Le test Mac fabrique de véritables checkpoints GLM miniatures et confronte le
backend GPU aux calculs de référence CPU. Il ne télécharge pas GLM-5.3.

## Démarrer des poids déjà présents

Le dossier doit contenir `config.json`, le tokenizer, les fichiers safetensors
et leur index d’origine. Ne placer pas les centaines de Go de GLM-5.3 sur le
SSD interne de 512 Go. Utiliser le disque externe une fois reçu.

```bash
bash scripts/launch_next_macos.sh /Volumes/SSD/GLM-5.3 --prompt "Bonjour" --max-new-tokens 16
```

Cette commande utilise exclusivement les fichiers locaux. Le modèle principal
reste GLM-5.3 ; Flash n’est pas substitué. Le profil `balanced` est choisi par
défaut. `--profile performance` augmente les budgets tout en gardant une réserve.

Pour préparer seulement les métadonnées, sans charger le modèle :

```bash
.venv-next/bin/airllm-next prepare /Volumes/SSD/GLM-5.3
.venv-next/bin/airllm-next tune /Volumes/SSD/GLM-5.3
```

`prepare` ne signifie pas « première réponse prête ». Il ne crée aucune copie
des poids. `tune` mesure des lectures partielles sur les fichiers existants ;
il distingue débit logique et compteurs physiques disponibles, et enregistre
le nombre de workers retenu. Les lectures répétées peuvent toucher le cache OS.
Les réglages persistants sont réutilisés par défaut ; `use_saved_tuning=False`
permet à l’API d’imposer tous les paramètres explicitement. `calibrate` mesure
aussi le coût réel du préchargement sur une série de générations, puis le
désactive automatiquement si son gain médian n’atteint pas 5 %.

## Réduire les transferts : conversion facultative

Le mode original décode les poids FP8 par blocs et les réutilise dans le cache
MLX. Il ne nécessite pas d’API d’inférence ni de format GPU FP8 natif.

Une conversion locale facultative des seuls experts permet d’exploiter les
multiplications MLX en poids entiers compressés :

```bash
.venv-next/bin/airllm-next convert /Volumes/SSD/GLM-5.3 \
  --output /Volumes/SSD/GLM53-experts-int4 --bits 4 --group-size 64
bash scripts/launch_next_macos.sh /Volumes/SSD/GLM-5.3 \
  --quantized-dir /Volumes/SSD/GLM53-experts-int4 --prompt "Bonjour"
```

Cette opération écrit un dérivé partiel volumineux. Les originaux restent
nécessaires. Elle ne s’exécute jamais implicitement et reprend sur les matrices
déjà terminées. `--max-matrices 10` permet une première conversion bornée.

`--scope mixed --linear-bits 8` convertit aussi certaines projections, tout en
conservant routeur, indexeurs et projection KV absorbée à leur précision
originale. `--linear-bits 4` réduit davantage les lectures mais requiert une
évaluation de qualité spécifique. Le format entier 8 bits avec échelles et biais
peut être plus gros que le FP8 d’origine : son intérêt éventuel est le kernel,
son coût E/S doit être mesuré. Aucune quantification n’est déclarée équivalente
au modèle original.

La conversion initiale travaille par tuiles et ne nécessite pas de deuxième
copie intégrale de GLM-5.3. Le dérivé et les originaux doivent tenir ensemble
sur le disque ; leur taille réelle doit être vérifiée avant une conversion
complète.

## API conservée

```python
from airllm import AutoModel
from airllm.nextgen import NextConfig

with AutoModel.from_pretrained(
    "/Volumes/SSD/GLM-5.3", backend="mlx-next",
    # Omettre next_config pour l'autoconfiguration et les réglages persistants.
) as model:
    ids = [model.tokenizer.encode("Bonjour")]
    tokens = model.generate(ids, max_new_tokens=16)
    print(model.tokenizer.decode(tokens.tolist()[0]))
```

Le modèle et ses poids en cache peuvent être réutilisés pour plusieurs appels.
Pour prolonger une même séquence, passer `past_key_values` retourné par
`model.forward(...)` à l’appel suivant. Les sessions ne peuvent pas être
transférées entre modèles. Une instance sérialise ses requêtes.

Les chemins `backend="ultra"`, `backend="mlx-ultra"` et les interfaces
historiques restent disponibles. Le backend Next accepte uniquement GLM-MoE-DSA,
une séquence sans padding et un batch de taille 1. Il rejette les variantes
incompatibles au lieu d’employer des calculs approximatifs.

## Limites à lire avant GLM complet

- Le décodage d’un 744B sur USB reste contraint par des dizaines de Go de poids
  nécessaires par token. Les chiffres miniatures ne prédisent pas ce débit.
- Le million de tokens annoncé par le modèle ne tient pas dans la mémoire de
  cette machine avec le cache KV actuel ; la limite locale par défaut est 4096.
- MTP et speculative decoding ne sont pas implémentés ni annoncés comme actifs.
- L’absorption de l’attention préserve l’algèbre ; les arrondis de faible précision
  et les égalités au top-k peuvent modifier une génération entre moteurs.
- Les journaux sont complets pour les tests exécutés ici. Les métriques GPU,
  thermiques et E/S physiques indisponibles sont laissées absentes ou à `null`.

Détails : [architecture](docs/NEXT_ARCHITECTURE.md),
[compatibilité GLM](docs/GLM53_COMPATIBILITY.md),
[validation](docs/NEXT_VALIDATION.md),
[protocole M5](docs/M5_VALIDATION.md).
