# Validation finale — AirLLM Next 4.1.0rc1

**22 octobre 2026. Branche : `release/airllm-next-final`.**

Le candidat est construit, installé et utilisé avec un checkpoint entraîné
complet sur CPU et Metal du Mac M5. Le stockage autonome réversible, les lecteurs
bornés, la CLI et l'API locale sont intégrés. **La génération de GLM-5.3 ou
Qwen3.8 complets sur 16 Go reste non validée.** Cette livraison ne les déclare
ni compatibles intégralement, ni interactifs.

## 1. Provenance et restauration

Base distante vérifiée : `d43032f38a0325aafd2973847d7ed84b21280c61`, branche
`research/ooc-engine-2026-10-22`. Les travaux exacts ont été retrouvés localement :
`b8b6c19928ed932a5fc90ea9d6949cd86aeecb26`, puis
`af54e4048bcb404a1a6883eda2f149e21878e05b`. Le candidat part de ce dernier état,
dans une copie séparée ; le checkout de recherche original reste préservé.

L'archive historique contient 260 entrées : CRC intégral vérifié sans erreur.
SHA-256 : `7433255c37830b6d1603e276452154a81e67c7bdbd6eae826216230b1bb93e2d`.
Les résultats historiques, y compris négatifs, restent présents. Voir
[provenance](evidence-final/provenance.json), [commits](evidence-final/provenance-commits.txt)
et [empreinte d'archive](evidence-final/archive.sha256).

L'authentification Git du terminal n'était pas configurée. La publication
autorisée utilise le connecteur GitHub : son commit doit avoir **le même arbre
Git** que le commit source local, même si son parent/métadonnées diffèrent.
L'ascendance locale exacte est conservée sur une référence dédiée et dans le
bundle de livraison. Le reçu de publication accompagne la livraison, avec les
SHA local/distant, arbre commun et têtes protégées avant/après. Aucun écrasement
de branche distante, merge dans `main`, PR ou publication stable n'est requis.

Pour identifier le commit effectif du checkout livré : `git rev-parse HEAD`.
Le bundle se restaure avec `git clone CHEMIN_DU_BUNDLE restored-airllm-next`.

## 2. Environnement réellement exécuté

MacBook Air Apple M5, 16 Go, macOS 27.2 arm64. Python de mesure 3.12.13,
MLX 0.32.4, Torch 2.14.1, Transformers 5.20.2, NumPy 2.5.3. Les installations
propres ont aussi validé NumPy 2.5.4 ; les 40 dépendances de l'environnement
utilisable sont figées dans `requirements-release-macos.lock.txt`.

Metal est actuellement visible et les générations sont réellement exécutées
sur son stream GPU. Les anciens rapports « aucun device visible » décrivent
leurs contextes antérieurs, pas ce nouvel essai. La précision de validation est
`MLX_ENABLE_TF32=0`, fixée avant Python. Sans cela, le test M5 initial échouait
avec un écart FP32 d'environ 6,74×10⁻⁴ ; aucune tolérance n'a été élargie pour
le masquer. Voir [environnement](evidence-final/environment.json) et les rapports
de tests initiaux conservés.

Aucun disque physique externe n'a été listé. Le SSD externe 2 To n'est donc
pas mesuré. Les lectures physiques macOS, énergie, températures et utilisation
GPU restent indisponibles, et non remplacées par des octets logiques.

## 3. Code effectivement intégré

| Domaine | Modules et résultat |
|---|---|
| Llama natif | `llama_runtime.py`, hooks communs de `runtime.py`, adaptateurs et dispatch ; GQA/RoPE/SiLU/KV, embeddings liés ou distincts |
| Projection de sortie | `weights.py` : tuiles plafonnées par I/O/tenseur, drain du préchargement, libération avant lecture suivante ; défaut 2 048 lignes mesuré |
| Stockage réversible | `lossless.py` : overlays et packs complets autonomes, assets hachés, gardes de tailles/chemins, commit atomique |
| Fiabilité E/S | `storage.py`, `sidecar.py` : fermeture concurrente, requêtes génératrices, validation des sélecteurs et reprise de manifeste |
| Numérique | `generation.py`, `attention.py`, GLM/Qwen : IDs vérifiés avant narrowing, suffixes de logits multi-chunks, bornes de scratch |
| Qwen | Porte DeltaNet `swish` interprétée comme SiLU, attention complète sigmoid, snapshots conservés ; backend toujours expérimental |
| Spéculation | Prompt lookup et brouillons GLM/Llama en CLI, rollback et adaptation ; séparation des octets logiques/encodés ; budget cumulé des paires CLI |
| API et installation | Validation JSON/contextes/tokens, vrai tokenizer structuré, erreur/récupération ; wheel/sdist et scripts macOS/CPU |

Les quatre politiques de cache restent explicites. Aucune route future inconnue
ne remplace les experts réellement choisis. Aucun kernel maison n'est ajouté
sans bénéfice complet mesuré. Les expériences utiles restent sous `research/`.

## 4. Tests finaux et installations

| Suite | Résultat réel | Preuve |
|---|---:|---|
| Next, options générales CPU | 351 réussis | [journal](evidence-final/tests-final-next-cpu.log), [JUnit](evidence-final/tests-final-next-cpu.xml) |
| Même suite, options générales Metal | 351 réussis | [journal](evidence-final/tests-final-next-metal.log), [JUnit](evidence-final/tests-final-next-metal.xml) |
| Ultra | 38 réussis | [journal](evidence-final/tests-ultra.log) |
| Héritage AirLLM | 69 réussis, 1 ignoré, 5 sous-tests réussis | [journal](evidence-final/tests-upstream.log) |
| Recherche exacte | 56 réussis | [journal](evidence-final/tests-exact.log) |

Cela représente **514 tests distincts réussis**, plus la répétition des 351 Next
avec les options GPU. Les cinq sous-tests ne sont pas ajoutés une seconde fois.
L'ignoré concerne bitsandbytes/CUDA, absent du matériel Apple. Certains contrôles
spécialisés sélectionnent directement le GPU même dans la suite générale CPU.

Les tests couvrent corruption, troncature, flux trop grands, budgets, close
concurrent, interruptions/reprise, layouts, experts, routes, logits, générations,
KV et états récurrents. Les échecs initiaux sont conservés : hypothèses de device
dans les anciens tests, TF32, mauvais masque de référence, import du paquet non
installé et retour structuré du tokenizer HTTP.

Wheel et sdist finaux : 58 fichiers Python identiques aux sources. Deux
installations séparées, plus `.venv-next`, ont un contrôle de dépendances propre.
Les deux distributions importent 57 modules et génèrent huit tokens GLM/Llama
miniatures conformes à Transformers, hors checkout source. Le sdist se construit
en isolation ; licence, README et notices sont inclus, tests/exemples/poids exclus.
[Rapport et hashes finaux](evidence-final/packaging.json). Syntaxe des scripts,
compilation Python et `git diff --check` vérifiés.

## 5. Génération avec de vrais poids entraînés

Modèle : **HuggingFaceTB/SmolLM2-135M-Instruct**, révision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, 272 tenseurs. SHA-256 du Safetensors :
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Seul ce petit checkpoint a été téléchargé explicitement ; aucun shard géant.

Trois prompts anglais/français/code, trois répétitions, 16 tokens par génération,
deux ordres de backends dans des processus frais. Les séquences Next CPU et
Metal coïncident toutes avec la référence Transformers. Écart maximal des
derniers logits : 5,53×10⁻⁵ CPU et 6,39×10⁻⁵ Metal ; tolérance préfixée
`atol=1e-4, rtol=5e-4`. C'est une proximité numérique et une équivalence des
séquences observées, pas une identité bit à bit universelle entre frameworks.

La première comparaison, conservée sous `trained-initial-inferred-mask*`,
échouait : avec EOS désactivé et PAD=EOS, Transformers supprimait de vrais
marqueurs de chat dans son masque automatique. Le protocole corrigé fournit
un masque explicite entièrement à un, identique au contexte non paddé de Next.

L'API installée, lancée hors sources, répond **HTTP 200** à une vraie demande
chat : « The capital of France is Paris. » Un mauvais modèle retourne 400 et
le serveur reste sain. [Preuve HTTP](evidence-final/trained-api.json).
Le lancement fourni à côté du dépôt a aussi généré depuis la wheel installée.
Un petit modèle peut néanmoins produire des erreurs factuelles ; aucune qualité
applicative générale n'est certifiée par ces prompts.

## 6. Performances observées

Tous les temps ci-dessous sont des durées complètes du chemin indiqué, non
des durées pures de kernels. Les TTFT excluent construction/tokenizer. Les pages
OS ne sont pas purgées ; les poids de ce petit modèle tiennent dans leur cache.

| Comparaison | Avant | Après | Portée |
|---|---:|---:|---|
| Sortie 128 → 2 048 lignes, médiane 16 tokens Metal | 1,992 s | 1,030 s | 1,93× ; même moteur/poids/prompts/budgets |
| Même essai, décodage médian | 8,46 tok/s | 17,27 tok/s | 2,04× ; séquences identiques, écart de logits entre tuilages 0 dans ces cas |
| Même essai, TTFT médian | 0,221 s | 0,162 s | hors chargement initial |
| Même essai, RSS de processus frais | 530,3 Mo | 535,9 Mo | coût mémoire supplémentaire mesuré |
| Ultra MPS → Next Metal, ordre initial, 16 tokens | 3,898 s | 1,102 s | 3,54×, backends GPU différents du même Mac |
| Même comparaison, ordre inversé | 3,383 s | 1,034 s | 3,27× ; pas un gain universel GLM/Qwen |
| Ultra CPU → Next CPU, ordre initial | 0,492 s | 0,960 s | **Next CPU plus lent**, environ 1,95× |
| Même comparaison CPU, RSS | 1 088 Mo | 546 Mo | compromis mémoire/temps, pas accélération CPU |
| 16 Mio synthétiques, appels pread | 256 | 32 | mêmes payloads, regroupement explicite |
| Même lecture chaude médiane | 1,719 ms | 1,406 ms | 1,22× local ; aucun débit SSD déduit |

Sources : [ablation tuiles](evidence-final/output-tiling-ablation.json),
[comparaison entraînée](evidence-final/trained-comparison.json),
[ordre inversé](evidence-final/trained-reverse-comparison.json),
[E/S](evidence-final/io-warm.json). Les données par prompt/répétition et logits
complets sont dans les JSON voisins, les plus volumineux archivés sans perte en
`.json.gz`. [Index et hashes originaux](evidence-final/compressed-evidence.json) ;
`gzip -dk FICHIER.json.gz` restitue les octets originaux. Ils sont aussi préservés
dans le commit local précédant cette compression réversible des preuves.

Pour cinq GLM aléatoires réduits, 16 tokens et cinq répétitions avant/après,
tous les logits sont identiques et les séquences coïncident. Ratios de débit
0,997–1,017 : aucun gain général robuste sur ce chemin.
[Résultat GLM](evidence-final/glm-before-after.json). Le benchmark Metal miniature
est distinct : [données](evidence-final/glm-mini-metal.json).

Charge finale : **40 générations, 1 280 tokens en 61,09 s**, décodage médian
21,43 tok/s pour un court préfixe et 32 tokens/requête, pic RSS échantillonné
519 815 168 octets. Croissance maximale du swap système observée pendant ce run :
0 octet. Du swap était déjà utilisé ; cela ne prouve pas l'absence de pages
swappées. [Session](evidence-final/sustained-final-trained-metal.json).
Ce run court ne certifie pas trente minutes ni la thermique.

Une sonde séparée du paquet installé relève un pic allocateur MLX de
140 115 204 octets, 134 498 560 actifs et 6 360 444 de cache au dernier relevé.
Ce compteur est global au processus MLX et différent du RSS.
[Mesure MLX](evidence-final/mlx-memory-trained-metal.json).

## 7. Compression : gain de capacité, résultat négatif en lecture chaude

269 030 016 octets de tenseurs deviennent 194 089 363 octets de payload,
ratio **0,72144** ; le pack complet avec index/tokenizer/config mesure
197 530 673 octets. Conversion : 8,67 s, blocs 1 Mio, budget explicite 512 Mio.
Les logits et huit tokens du checkpoint complet sont identiques au lecteur
original sur CPU et Metal, avec construction du lecteur original activement
interdite pendant le test du pack.
[CPU](evidence-final/standalone-smollm2-cpu.json),
[Metal](evidence-final/standalone-smollm2-metal.json).

Sept passes chaudes mélangées, contrôles d'intégrité activés : lecture originale
15,04 ms médiane, pack 587,47 ms. Le pack est environ **39× plus lent** dans
cet essai de transport. Les limites de tuiles entraînent aussi des relectures :
204 006 435 octets encodés lus par passe, pas seulement la taille du payload.
Pic RSS du processus de benchmark/conversion : 60 702 720 octets.
[Mesures brutes](evidence-final/lossless-trained-transport.json).

Le format est intégré comme option de capacité/transport exact, sans promesse
de vitesse. Aucune extrapolation de ce ratio vers le BF16 Qwen complet.

## 8. Installation, utilisation et limites restantes

Les instructions exactes sont dans le [guide](docs/USER_GUIDE.md). La livraison
locale comprend un environnement installé, un modèle entraîné distinct du dépôt
et un lanceur à côté de celui-ci. Pour une installation neuve, utiliser la wheel
`dist-final/airllm-4.1.0rc1-py3-none-any.whl[next]` avec les contraintes macOS,
ou le script arm64. Génération locale, API, conversion, inspection et benchmarks
ne déclenchent pas de téléchargement implicite.

[Architecture](docs/RELEASE_ARCHITECTURE.md),
[compatibilité](docs/RELEASE_COMPATIBILITY.md),
[limites détaillées](docs/TECHNICAL_LIMITS.md).
Restent non démontrés : checkpoints géants, qualité/quantification complètes,
SSD externe, long contexte, plateau thermique et énergie. NVFP4, MTP et
spéculation stochastique ne sont pas implémentés. Les modèles MLX construits
indépendamment en bibliothèque nécessitent une coordination explicite de
l'allocateur partagé ; le garde cumulé ajouté vise les paires créées par la CLI.

## 9. Résultat scientifique et statut

Résultats nouveaux **dans ce dépôt** : checkpoint autonome exact entraîné sans
lecture du modèle source, gain mesuré du tuilage de sortie, génération Llama
native CPU/Metal comparée indépendamment, et détection/correction des deux
incompatibilités réelles du protocole/tokenizer. Ce sont des intégrations et
expériences reproductibles, pas une nouvelle famille de compression ou un
théorème de vitesse pour les modèles géants.

Les pistes de résidence encodée, calcul compressé, prédiction, MTP et grands
facteurs OTA restent des hypothèses ou des travaux antérieurs avec leurs limites
explicitement conservées. La borne de transport d'une implémentation n'est pas
une impossibilité fondamentale. Le candidat livre un moteur réel pour son
périmètre validé ; il ne transforme pas l'objectif ultime en résultat établi.
