# Validation AirLLM Next 5.0.0a1

**23 octobre 2026 — branche `research/airllm-next-5-generation`.**

Cette version est une **alpha de recherche**, pas une RC ni une version stable
certifiant GLM-5.3 ou Qwen3.8 sur un M5 de 16 Go. Elle intègre réellement des
caches de poids supplémentaires dans le budget existant, un contrôleur de
spéculation mesuré, une file d'inférence agentique persistante et des gardes
mémoire renforcées. Un checkpoint entraîné complet de 13,477 Go a généré en
dépassant la mémoire de cet environnement, avec comparaison numérique
indépendante. **La vitesse obtenue ne rend pas ce cas CPU interactif.**

**Clôture logicielle : 626 tests distincts réussis, 11 ignorés et 11 sous-tests
réussis ; wheel et sdist installés et vérifiés hors des sources.** Les preuves
chiffrées restent limitées aux protocoles et au matériel décrits ci-dessous.

## 1. Base, conservation et environnement

La branche part exactement de `9629c27b9d8bdf697ae866498ad468e11eb183d4`,
arbre `b5be3af163ffb81b8fe7be184222c94f18d84195`, version 4.1.0rc1.
La référence est conservée dans un worktree distinct pour les comparaisons.
`main`, `release/airllm-next-final`, les preuves `evidence-final/`, les branches
antérieures et `research/exact_io/` ne sont pas remplacés. Les têtes de départ
et l'inventaire de provenance sont dans `evidence-v5/provenance.json` ; les
rapports de workers identifient aussi leurs sources par SHA256.

Les expériences entraînées ont précédé le dernier correctif de départage DSA.
Leurs empreintes désignent les sources effectivement exécutées, pas un arbre
final reconstruit a posteriori. Elles utilisent Llama, dont ce correctif GLM
ne change pas le calcul. `evidence-v5/final-source-coverage.json` vérifie les
empreintes du chemin Llama et l'absence de changement de son RMS partagé.
La régression et les deux installations finales incluent le correctif GLM,
les dernières gardes mémoire et l'identité de reprise incluant la version Torch.

Exécution V5 : Linux x86_64, Python 3.12.14, MLX 0.32.4 **CPU**, Torch
2.14.1+cpu, Transformers 5.20.2, NumPy 2.5.4 et Safetensors 0.8.0.
`MLX_ENABLE_TF32=0` est fixé avant l'initialisation des expériences.
L'hôte expose 10 451 439 616 octets de mémoire physique ; le cgroup commun
est configuré à **8 589 934 592 octets, soit 8 Gio**. Ces quantités diffèrent
du matériel cible M5 16 Go. Source : `evidence-v5/environment.json`.

Les compteurs cgroup incluent le cache de fichiers et d'autres processus ;
ce ne sont pas des mesures du RSS d'un modèle isolé. Les lectures physiques
Linux sont distinctes des octets demandés au lecteur. Metal, le SSD externe
USB-C de l'utilisateur, les températures et la pression mémoire macOS ne
sont pas mesurés ici. Les mesures M5 de V4.1 restent historiques.

## 2. Régressions et contrôles adverses

La suite complète du commit de départ a été réexécutée sans modification :
**504 tests distincts réussis, 11 ignorés et 5 sous-tests réussis**, en
52,33 secondes. Les ignorés concernent notamment Metal indisponible et le
chemin CUDA/bitsandbytes absent. Le JUnit contient des entrées supplémentaires
de sous-tests ; elles ne sont pas additionnées aux tests distincts.
Preuves : `evidence-v5/baseline/tests.log` et `tests.xml`.

La régression complète du dernier arbre réussit : **626 tests distincts,
11 ignorés et 11 sous-tests réussis en 55,99 s**. Elle couvre Next, Ultra,
l'héritage AirLLM, la recherche exacte et la référence HF streamée.
Les gardes finales DSA, d'allocateur et de remplacement Qwen y sont incluses.
La compilation Python, `doctor` et `git diff --check` réussissent également.
Preuves : `evidence-v5/final/tests.log`, `tests.xml` et `doctor.json`.

| Contrôle ciblé déjà exécuté | Résultat | Preuve |
|---|---:|---|
| Poids compacts, tuiles, interruption, budget et parité | 19 réussis | `evidence-v5/weight-tests.xml` |
| Spéculation, coûts adverses, propositions et lookup exact | 65 réussis | `evidence-v5/speculation-tests.xml` |
| File, journaux, reprise, API et sessions natives GLM/Llama/Qwen | 27 réussis | `evidence-v5/agent-tests.xml` |
| Llama RoPE linéaire et régressions Llama/config/adaptateurs | 71 réussis | `evidence-v5/linear-rope-tests.xml` |
| Référence Transformers streamée contre Transformers résident | 1 réussi | `evidence-v5/streamed-reference-test.xml` |
| Correction DSA, natif, FP8, fiabilité et banques Qwen | 43 réussis, 3 ignorés | `evidence-v5/dsa-fix-tests.xml` |

Ces contrôles se recouvrent et ne forment **pas** un total distinct à sommer.
Le total de clôture est celui de l'exécution complète ci-dessus, sans ajouter
une seconde fois les sous-suites ou leurs sous-tests.

Les tests couvrent notamment lectures interrompues, sélecteurs invalides même
sur cache hit, capacité presque saturée, admission/éviction, anciens et nouveaux
tampons coexistants, brouillons faux/vides/coûteux, rollback, erreurs de journaux,
corruption, mauvaise identité de reprise, priorités, annulation et isolation des
états récurrents. Ils ne remplacent pas une panne réelle de câble ou de SSD.

Les échecs initiaux sont conservés : `integration-stage1.*` pour l'accord du
précontrôle avec les fichiers Safetensors sans index, et
`streamed-reference-initial-failure.*` pour une signature de constructeur
Transformers devenue incompatible. La référence a été corrigée puis contrôlée
contre le modèle résident ; aucune tolérance numérique n'a été élargie.

La première vérification des distributions a ensuite révélé un **défaut GLM
hérité de 4.1**, absent des préfixes des anciens tests : le fixture seed 521
générait un PAD à embedding nul, puis des scores DSA égaux à la frontière du
top-k. MLX et Torch choisissaient des clés différentes ; les tokens divergeaient
au cinquième nouveau token. Un masque explicite à un ne changeait pas la
référence. Le seed, le prompt et les critères ont été conservés.

V5 garde la sélection native non ambiguë et résout les seules égalités finies
traversant la frontière avec le top-k CPU PyTorch installé, après réservation
du scratch supplémentaire. Les huit tokens et les logits cachés correspondent
désormais à la référence, écart maximal observé 1,49×10⁻⁷ contre 0,440 avant fix.
Ce chemin ajoute une synchronisation après T>k et un coût de copie/CPU en cas
d'égalité ; l'import tardif de Torch peut ajouter du RSS hors des seuls buffers
gérés. Les temps et compteurs sont instrumentés. La convention de départage
n'est pas promise entre builds CPU/CUDA/serveur, et aucun gain Metal n'est
annoncé. Traces : `evidence-v5/glm-dsa-tie-diagnostic.json` ; échecs de package
conservés sous `evidence-v5/initial-packaging/`.

## 3. Génération entraînée dépassant réellement la RAM

Checkpoint : **`codellama/CodeLlama-7b-hf`**, révision
`6c284d1468fe6c413cf56183e69b194dcfa27fe6`. Les deux shards officiels
contiennent 291 tenseurs BF16 et **13 477 093 376 octets de payload** ; les
fichiers Safetensors totalisent 13 477 127 208 octets. Ils dépassent à la fois
la mémoire physique exposée et le plafond du cgroup. Les fichiers complets,
leurs hashes LFS et les assets sont vérifiés ; config et poids restent inchangés.
Le téléchargement explicite était plafonné à 15 Go. Les poids ne sont pas
redistribués dans les sources.

Protocole : trois processus frais séparés, même checkpoint, même préfixe de
dix tokens, quatre nouveaux tokens, calcul FP32 et budgets explicites identiques
pour V4.1/V5. L'inférence lit les poids à la demande et n'alloue pas le modèle
entier. Le cache OS n'est pas purgé et ne peut contenir tous les shards.

La référence indépendante emploie les vrais blocs Transformers, RoPE, RMSNorm
et caches, en matérialisant puis libérant une couche à la fois. Ses embeddings
sont sélectifs et sa tête est tuilée indépendamment. Un test miniature préalable
la compare à l'exécution Transformers entièrement résidente.

| Mesure observée | V4.1 | V5 | Référence Transformers streamée |
|---|---:|---:|---:|
| Durée de génération de 4 tokens | 165,164 s | 164,401 s | 44,709 s |
| Débit de décodage, hors premier token | 0,03616 tok/s | 0,03648 tok/s | 0,09499 tok/s |
| Pic RSS du processus | 1 405 104 128 B | 1 404 461 056 B | 1 370 660 864 B |
| Lectures physiques Linux du processus | 50 575 691 776 B | 50 383 945 728 B | 41 399 136 256 B |

V5 produit les quatre tokens de **`return a + b`** à partir de
`def add(a, b):` suivi de l'indentation. Ses logits sont identiques bit à bit à
V4.1 dans cet essai. Les deux séquences correspondent à la référence indépendante ;
l'écart maximal des logits est **5,5313×10⁻⁵**, sous le critère fixé avant
mesure `atol=1e-4, rtol=5e-4`.

Pour V5, le premier token prend 82,161 s ; le lecteur compte
52 859 379 712 octets logiques. Le swap système observé au début et à la fin
vaut zéro. Les événements OOM/oom_kill du cgroup restent à zéro, tandis que les
événements de plafond et le recyclage du cache de fichiers sont enregistrés.
Le pic cgroup est commun à l'expérience et ne doit pas être attribué au seul
modèle. Ce résultat établit **cette génération hors mémoire**, pas un long
contexte, une session thermique ou une vitesse utilisable par RONIN.

Il n'y a pas d'accélération majeure V5 démontrée dans cette comparaison unique.
La référence Torch streamée est nettement plus rapide que les deux moteurs
Next CPU sur cette charge. Les mesures négatives sont conservées.
Preuves : `evidence-v5/trained-large.json`, rapports `*-baseline.json`,
`*-candidate.json`, `*-reference.json` et leurs logits `.npz` ; acquisition et
inspection dans `trained-large-download.json` et `trained-large-preflight.json`.

## 4. SmolLM2 entraîné et ablations de poids

Checkpoint complet : `HuggingFaceTB/SmolLM2-135M-Instruct`, révision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, Safetensors de 269 060 552 octets,
SHA256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Il tient en RAM : ses résultats ne servent pas de preuve hors mémoire.

Le contrôle indépendant initial emploie trois prompts, trois répétitions et
seize nouveaux tokens : les neuf générations V4.1 correspondent à Transformers,
écart maximal des derniers logits 7,25×10⁻⁵. Médiane V4.1 5,919 s contre
1,007 s pour Torch résident. Ce contrôle est séparé des ablations suivantes.

Quarante-huit générations des quatre variantes de résidence produisent les
mêmes séquences, soit 768 tokens. Deux ordres inversés, mêmes poids/budgets
au sein de chaque paire, seize tokens et répétitions chaudes sont conservés.

| Paire | Observation |
|---|---|
| 300 Mio décodés → 300 Mio source compacte | Lectures chaudes 4 729 352 256 B → 0 B ; médiane 8,320 s → 8,631 s, **plus lent d'environ 3,75 %** |
| 600 Mio sans → avec admission des tuiles | Lectures 906 027 840 B → 58 176 B ; médiane 6,889 s → 6,502 s, facteur observé 1,059× |

L'effet temporel des tuiles change avec l'ordre : environ +10,4 % de vitesse,
puis −2,3 %. Un téléchargement réseau séparé du grand checkpoint chevauchait
ces expériences ; le premier passage exclu a aussi pu chevaucher de petits
tests. Les processus étaient séparés, le système n'était pas entièrement
isolé. Ces chiffres ne prouvent pas un gain robuste ni un débit de SSD à froid.
Les deux options restent désactivées par défaut. Les preuves brutes et
limites sont dans `weight-residency-trained.json` et
`weight-residency-analysis.json`, sous `evidence-v5/`.

## 5. Spéculation et partage RONIN

Le contrôleur observe les coûts par profondeur, les octets cible/brouillon
connus, les passes de recalibration et le temps cumulé de décodage, y compris
ses frais de gestion. Il désactive les tentatives défavorables observées ;
cela n'est pas une garantie de gain avant l'observation. Les transports de
brouillons non instrumentés restent inconnus, pas artificiellement nuls.

Treize générations entraînées contrôlent le lookup et un faux brouillon.
Les séquences correspondent au décodage ordinaire observé. Sur un motif répété,
les médianes sont 6,927 s en ordinaire, 5,614 s avec le lookup V4 et 5,835 s
avec V5 : V5 bénéficie du motif mais est **plus lent que V4** dans ce cas.
Sur la phrase répétée, V5 reste plus lent que le décodage ordinaire.
Le brouillon volontairement faux n'accepte aucun token proposé et est désactivé
après quatre tentatives ; sa sortie reste celle de la cible.

Le lookup seul améliore ses cas défavorables observés d'environ 26,5×/33,4×
contre l'ancienne recherche, sans gain notable sur les répétitions immédiates.
**Ce sont des microbenchmarks de recherche, pas des accélérations de génération.**
Le premier prototype Z plus lent sur les répétitions reste dans les preuves.
Sources : `evidence-v5/speculation-trained.json`, `speculation-lookup*.json`
et `research/v5/SPECULATION.md`.

Le moteur RONIN partage un modèle et **un seul contexte natif résident**.
Sur deux jobs entraînés de huit tokens, l'exécution ordinaire prend 12,341 s,
la file avec quantum 8 prend 12,899 s, et le quantum 1 prend 88,671 s, dont
75,696 s de rejeu. Les sorties sont identiques à l'ordinaire. La reprise après
redémarrage vérifie le préfixe et termine la réponse. Le pic RSS du processus
de cette expérience est 517 914 624 octets.

La file fournit isolation, priorité, annulation, reprise et budgets de
transcripts/journaux ; des changements de contexte fréquents coûtent cher.
Les écritures sont bornées séparément des fichiers retenus et comptées comme
octets offerts aux écritures, pas comme usure NAND mesurée. Ces essais sont
des jobs d'inférence locaux, **pas des missions autonomes RONIN terminées**.
Preuve : `evidence-v5/agent-queue-trained.json` ; contrat complet dans
`RONIN_INFERENCE_GUIDE.md`.

## 6. Composants réels GLM-5.3 et chemin Qwen

La mission a acquis **1 154 264 octets de corps HTTP**, en six requêtes
strictement HTTP 206, depuis le checkpoint GLM-5.3 officiel épinglé à
`ba32f8c19c8738e2c57c71798c601d477a7bf844`. Un HTTP 200 est refusé avant
lecture du corps ; le protocole est plafonné à 16 Mio. Aucun shard complet
GLM/Qwen n'est téléchargé.

Deux sous-matrices entraînées de l'expert 0, couche 3, sont validées :
`gate_proj` 128×6144, lignes originales 128:256, et `down_proj` 128×2048,
lignes 256:384, avec leurs vraies échelles de bloc. La reconstruction garde
les octets exacts ; le décodage FP32 de `NativeWeights` est identique bit à bit
au cast FP8 PyTorch indépendant multiplié par les échelles officielles.
Les produits linéaires sur trois positions fixes respectent les tolérances
préfixées `atol=rtol=2e-5` : écarts maximaux 1,15484×10⁻⁷ et 3,72529×10⁻⁸.

Il s'agit de tuiles réelles réindexées dans un fixture partiel, **pas** d'une
matrice entière, d'un expert complet, de l'attention DSA complète, d'une
génération GLM ou d'une reproduction de la quantification dynamique des
activations du serveur d'origine. Aucun débit de ce test n'est annoncé.
Preuve : `evidence-v5/glm53-real-sample.json` et script
`research/v5/validate_glm53_sample.py`.

Les configs et index GLM/Qwen BF16/FP8/NVFP4 sont épinglés et inspectés.
Qwen reste expérimental en flottant ; son FP8 hybride n'est pas certifié et
ModelOpt/NVFP4 reste explicitement refusé. À 4 096 tokens BF16, son état,
snapshot et remplacement conservateur d'une banque réservent 1 968 766 976 B,
au-delà du KV par défaut de 512 Mio. Le RoPE linéaire Llama ajouté est testé
séparément et ne modifie pas les gardes RoPE GLM/Qwen.
Le tableau précis et les sources sont dans `MODEL_COMPATIBILITY_V5.md`.

## 7. Package, API et restauration

Sources, CLI, API synchrone et API jobs sont intégrés ; la version du package
est `5.0.0a1`. La wheel et le sdist de `dist-v5/` sont construits, installés
dans deux préfixes séparés puis vérifiés hors du checkout. Leurs **62 fichiers
Python correspondent octet pour octet aux sources finales**. Chaque installation
importe 61 sous-modules et le module racine ; les huit tokens GLM/Llama des
fixtures inchangés correspondent à Transformers et à la CLI. Le rejeu durable
des agents reproduit cette même génération ordinaire.

Les dépendances proviennent du pool préinstallé réutilisé : il ne s'agit pas
d'une nouvelle résolution isolée de toutes les dépendances. Cette limite est
explicite dans `evidence-v5/packaging.json`, avec les tailles, hashes SHA256 et
preuves de correspondance des distributions. Les premiers échecs d'installation
et le diagnostic DSA restent conservés ; le package final corrigé repasse les
deux vérifications. Une wheel héritée de `dist-final/` est une livraison 4.1,
pas cette alpha V5.

Pour restaurer un bundle Git livré, dans un nouveau répertoire :

```bash
git clone /chemin/du/bundle airllm-next-v5
git -C airllm-next-v5 switch research/airllm-next-5-generation
git -C airllm-next-v5 merge-base --is-ancestor 9629c27b9d8bdf697ae866498ad468e11eb183d4 HEAD
git -C airllm-next-v5 rev-parse HEAD
```

Depuis les sources restaurées, avec les dépendances du profil choisi :

```bash
python3.12 -m venv .venv-next
.venv-next/bin/python -m pip install './air_llm[next]'
MLX_ENABLE_TF32=0 .venv-next/bin/airllm-next doctor
.venv-next/bin/airllm-next preflight /checkpoint/local --tokens 4096
AIRLLM_PYTHON=.venv-next/bin/python bash scripts/validate_v5.sh
```

Les téléchargements de poids sont explicites et séparés des constructeurs,
tests ordinaires et API. Les noms/hashes des distributions, du bundle et le
reçu de publication doivent être pris dans les pièces finales de livraison,
et non inférés d'une archive historique. Aucun merge dans `main` ni
publication stable n'est nécessaire pour restaurer la branche de recherche.

## 8. Résultat et limites restantes

Résultats établis : mécanismes V5 intégrés, génération entraînée complète
au-delà de la mémoire de cet hôte, comparaison indépendante conservant les
tolérances, validation de vraies tuiles GLM FP8, et partage agentique récupérable
avec coûts négatifs visibles. Les baisses de lectures ne sont pas confondues
avec des accélérations de bout en bout.

Restent non établis : génération complète GLM-5.3/Qwen3.8, vitesse pratique
des modèles géants sur M5 16 Go, SSD externe mesuré, long contexte, réponses
longues, trente minutes thermiques, absence de fuite sur cette durée,
NVFP4/MTP, vérification spéculative récurrente Qwen et missions autonomes
externes. Aucun facteur V5 de 2×/5×/10× sur une génération géante n'est certifié.
Le protocole concret de reprise M5 reste dans `M5_VALIDATION.md`.
