# Validation M5 V5 — état et protocole exécutable

La session V5 courante utilise Linux x86_64, MLX CPU. Elle ne dispose ni de
Metal ni du SSD USB-C externe de l'utilisateur. Les mesures M5 V4.1 dans
`evidence-final/` demeurent des preuves historiques distinctes. **Aucune mesure
V5 de thermique, de débit du SSD externe ou de génération GLM complète sur M5
n'est certifiée par ce document.**

## Contrôles déjà exécutables hors ligne

```bash
AIRLLM_PYTHON=python3.12 bash scripts/setup_next_macos.sh
export MLX_ENABLE_TF32=0
AIRLLM_PYTHON=.venv-next/bin/python bash scripts/validate_v5.sh
AIRLLM_PYTHON=.venv-next/bin/python bash scripts/validate_next_macos.sh
.venv-next/bin/airllm-next doctor
.venv-next/bin/airllm-next preflight /Volumes/SSD/GLM-5.3 --tokens 4096
```

Les scripts ne téléchargent pas de poids. `validate_next_macos.sh` exige Metal
et exécute les tests actuels et un benchmark réduit ; ce benchmark ne valide
pas le checkpoint géant. Le rapport `preflight` sépare les ressources observées
d'un scénario fourni explicitement. Laisser la réserve système et commencer
avec des contextes courts ; une admission refusée est un résultat à conserver.

## Poids entraînés et avant/après

Le script CPU indépendant suivant rejoue le protocole entraîné sans télécharger :

```bash
.venv-next/bin/python research/v5/benchmark_trained.py \
  --baseline-source /chemin/source-v4.1 --candidate-source "$PWD" \
  --model-dir /chemin/SmolLM2-135M-Instruct \
  --output evidence-v5/m5-cpu-trained.json
```

Ce script sélectionne explicitement CPU. Pour la validation Metal, utiliser
les tests avec `AIRLLM_TEST_DEVICE=gpu` et les commandes `run --device gpu`
actuelles, comparer mêmes prompts/tokens/budgets avec les deux checkouts et
conserver leurs JSON `--json-report`. Ne pas convertir un résultat CPU en
résultat Metal. La collecte doit distinguer construction, préremplissage,
TTFT, décodage et rejeu d'agents.

## Mesure du stockage et session prolongée

`airllm-next tune /checkpoint/local` lit un échantillon existant et ne crée
pas de gros fichier. Son état de cache OS est non contrôlé : il ne certifie
pas le débit NAND à froid. Conserver le modèle/volume/câble effectivement
utilisés, la taille des données lues, le temps, les erreurs et les compteurs
physiques lorsqu'ils sont disponibles. Le moteur ne purge pas le cache OS et
ne génère pas d'écritures de benchmark massives.

La validation finale géante exige un payload plus grand que la mémoire de
la machine, des générations complètes, un processus frais, des mesures RSS,
MLX, swap et lectures physiques. Une courte réponse miniature ou un budget
de cache artificiellement réduit ne suffit pas à établir ce résultat.

Pour une session prolongée, conserver les relevés pendant au moins trente
minutes de travail réellement utile : contextes courts/longs, réponses longues,
modèle froid/chaud, erreurs USB, annulation, redémarrage. Enregistrer température
et pression mémoire seulement si un compteur réel les fournit ; sinon null.
Une température non disponible ne signifie pas absence de throttling.

## Critères non satisfaits par les seuls tests logiciels

Génération fidèle de GLM-5.3 officiel ; Qwen3.8 FP8/NVFP4 ; performances M5 V5 ;
SSD externe mesuré ; trente minutes thermiques ; autonomie RONIN réelle. Les
commandes ci-dessus rendent la reprise concrète sans les annoncer réussis.
