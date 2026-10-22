# AirLLM Ultra — moteur de streaming direct

Livraison du 22 octobre 2026. Développement local basé sur **AirLLM v4.0.0**, commit
`ff35db207a0c559af9aa95d686057c3fe84f1d40`. Aucun changement publié sur un dépôt distant.
Le README d'origine est conservé dans `README.md` ; commencez par ce document pour les ajouts.

Le nouveau moteur fonctionne et a été testé sur **Linux x86-64 / CPU**. Il charge
les poids depuis les safetensors d'origine, sans découpage préalable du modèle.
Les résultats CPU réels sont dans `benchmarks/`. Apple Silicon, Metal et le GLM 5.3
complet restent à valider sur le matériel cible : aucune vitesse M5 n'est annoncée.

## Ce qui est implémenté

- Accès direct aux tenseurs et aux tranches d'experts dans les fichiers d'origine ;
  aucune seconde copie intégrale du checkpoint.
- Construction des banques d'experts directement sur `meta`, y compris leurs
  tenseurs 3D : évite une allocation CPU temporaire de plusieurs gigaoctets.
- Experts MoE chargés après le routage réel, individuellement ; prise en charge
  des banques 3D et des checkpoints à fichiers/tenseurs par expert.
- Cache à deux segments, borné en octets et résistant aux lectures séquentielles ;
  cache supplémentaire sur l'accélérateur pour éviter des transferts répétés.
- Préchargement sur un worker d'E/S, futures dédupliquées, réservations mémoire
  bornées ; lecture effective des pages plutôt que simple ouverture d'un mmap.
- Préchargement du prochain expert réellement sélectionné et prédictions
  facultatives fondées sur les fréquences de routage des sessions précédentes.
- Déquantification FP8 par blocs sur CPU, poids sélectionnés seulement, sans
  nécessiter un noyau de multiplication FP8 natif sur Metal.
- Réutilisation du squelette, des mappings et des métadonnées ; aucun
  `gc.collect()`, `malloc_trim()` ou `empty_cache()` à chaque couche dans ce moteur.
- Backend natif MLX/Metal pour **Llama simple, sans quantification ni RoPE modifié**,
  livré avec un test matériel dédié. Ce backend ne prétend pas prendre en charge GLM.

Les budgets concernent les poids retenus par le moteur. Ils ne plafonnent pas la
RAM du processus entier : activations, cache KV, pages mmap, allocations temporaires,
allocateur Metal et macOS s'y ajoutent. Le contrôle de pression réduit les caches ;
ce n'est pas une garantie contre un manque de mémoire.

## Démarrage sur le Mac

Il faut Python **3.12 ou 3.13**, macOS Apple Silicon, et une connexion pour installer
les dépendances. L'archive contient leurs versions et un wheel du logiciel ; elle
ne contient ni les frameworks binaires multi-plateformes ni les poids des modèles.

Depuis le dossier extrait :

```bash
./scripts/setup_macos.sh
./scripts/validate_macos.sh
```

La validation génère de petits modèles localement ; elle ne télécharge pas GLM.
Les tests MLX et MPS ignorés sur Linux doivent réellement s'exécuter sur le Mac.
Le SSD n'est pas nécessaire pour cette première étape.

Quand le SSD sera disponible et qu'un petit checkpoint sera téléchargé :

```bash
./scripts/launch_macos.sh \
  --model '/Volumes/VOTRE_SSD/modeles/petit-modele' \
  --prompt 'Bonjour, présente-toi.' \
  --device mps --max-new-tokens 8 \
  --metrics validation-local/petit-modele-mps.json
```

Le launcher installe les dépendances au premier lancement si nécessaire. Il ne
télécharge jamais implicitement un checkpoint. Les fichiers originaux doivent
rester présents pendant toute l'utilisation ; il ne faut pas débrancher le SSD.

Pour Llama simple sur le backend natif MLX, après réussite du test dédié :

```bash
./scripts/launch_macos.sh \
  --model '/Volumes/VOTRE_SSD/modeles/llama-compatible' \
  --backend mlx-ultra --max-new-tokens 8
```

## GLM 5.3 : portée exacte

Les véritables `config.json` et index officiel sont inclus dans `evidence/`, pas
les poids. La configuration est `GlmMoeDsaForCausalLM` / `glm_moe_dsa`, 78 couches,
256 experts routés par couche sparse et 8 sélectionnés par token, FP8 par blocs
128 × 128. L'index annonce **755 617 140 416 octets**, répartis sur 141 fichiers.

L'audit crée le véritable graphe principal sur `meta` avec Transformers 5.20.2,
compte 743 377 000 704 paramètres dans ce graphe et vérifie ses noms de tenseurs
contre l'index. Les poids supplémentaires du prédicteur MTP, notamment la couche
78, ne sont pas utilisés : la génération est autorégressive standard, sans MTP.

Un petit GLM-MoE-DSA, puis un petit checkpoint FP8 de cette architecture, ont été
comparés à une référence Transformers. Les logits et séquences correspondent.
**Le chargement et l'inférence du GLM 5.3 complet n'ont pas été testés.**
Le backend envisagé pour ce modèle est `ultra` sur PyTorch/MPS ; son bon
fonctionnement sur M5 reste une condition à vérifier, pas une capacité certifiée.

Le SSD de 2 To peut contenir le checkpoint d'origine en capacité nominale ; son
espace réellement libre, sa connexion et son débit restent à mesurer. Il ne
remplace pas la RAM. Sans cache, les seules matrices des experts actifs du graphe
principal représentent environ **22,65 Go de FP8 par token**, avant les échelles,
l'attention et les couches denses. C'est un calcul de volume, pas un benchmark.
Les mesures des petits modèles ne se transposent pas à ce modèle géant.

## Utilisation Python

```python
from airllm import AutoModel
from airllm.ultra import UltraConfig

config = UltraConfig(
    cache_bytes=512 * 1024**2,
    device_cache_bytes=256 * 1024**2,
    prefetch_bytes=256 * 1024**2,
    max_group_bytes=2 * 1024**3,
    cache_dir='/Users/votre_nom/Library/Caches/airllm-ultra',
)
with AutoModel.from_pretrained(
    '/Volumes/VOTRE_SSD/modeles/modele', device='mps', ultra=config
) as model:
    ids = model.tokenizer('Bonjour', return_tensors='pt')['input_ids'].to(model.device)
    tokens = model.generate(ids, max_new_tokens=8, do_sample=False)
    print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))
    print(model.metrics())
```

`close()` / le contexte `with` attend le worker, ferme les mappings et enregistre
les fréquences de routage. Une instance sérialise ses requêtes ; ce moteur n'est
pas un serveur d'inférence multi-requêtes. L'entraînement reste sur le chemin
AirLLM d'origine. L'appel historique sans `ultra` conserve son comportement.

## Reproduire les validations CPU

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements-linux-tested.txt
.venv/bin/python -m pip install --no-deps -e air_llm
.venv/bin/python -m pytest tests_ultra -q
.venv/bin/python benchmarks/benchmark.py --repeats 3 --tokens 8
.venv/bin/python benchmarks/benchmark_experts.py
.venv/bin/python scripts/audit_glm_metadata.py
```

`requirements-linux-tested.txt` est le gel exact de l'environnement exécuté ;
utiliser `requirements-macos.txt` sur Mac. Le benchmark principal prend plusieurs
minutes car il mesure aussi le chemin stable d'origine, avec ses nettoyages mémoire.

Une validation facultative d'un modèle entraîné est reproductible par
`benchmarks/validate_pretrained.py`. Son téléchargement est explicite et plafonné
à 350 Mo ; les instructions et résultats sont dans le rapport de validation.

## Contenu et reprise

- `air_llm/airllm/ultra/` : moteur, stockage, cache, experts, FP8, MLX et CLI.
- `tests_ultra/` : tests CPU, erreurs, concurrence, budgets et tests matériels Mac.
- `benchmarks/` : scripts, résultats JSON bruts, latences individuelles et journaux.
- `evidence/` : environnement, rapports de tests, index/config GLM et audit du graphe.
- `docs/VALIDATION.md` : résultats, limites, protocole M5/SSD et prochaines étapes.
- `scripts/*macos.sh` : installation, lancement et validation.
- `changes.patch` : modifications par rapport au tag stable, sans les résultats binaires.
- `dist/` : wheel installable et archive source du paquet.

Le code source officiel complet et sa licence Apache-2.0 sont conservés. Cette
version locale `4.0.0+ultra.1` est un travail dérivé, pas une version publiée par
les mainteneurs d'AirLLM. Les modèles ont leurs propres licences.
