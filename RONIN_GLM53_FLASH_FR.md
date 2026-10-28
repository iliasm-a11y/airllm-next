# RONIN — port natif expérimental GLM-5.3 Flash

Mission du 28 octobre 2026 (Europe/Paris). Branche `research/ronin-glm53-flash`,
créée directement sur la source de la PR #14 :
`0fb55ba32121c437fb1d80319e1fc40f26546310`.
Aucune fusion de PR et aucune modification de `main`.

## Résultat et périmètre de preuve

Un décodeur textuel natif `glm5-next` est implémenté dans AirLLM Next. Il utilise
les lectures GGUF sélectives, les produits matriciels MLX par tuiles et le cache
adaptatif existants. Il exécute KDA, DSA/KPool, mHC et les experts réellement
sélectionnés. Les primitives et des décodeurs miniatures sont comparés à une
référence indépendante Transformers. Ce résultat constitue un port expérimental
testé sur CPU, pas une qualification du checkpoint entraîné de 110 Go.

**Aucune inférence du checkpoint réel, conversation RONIN réelle, exécution
Metal/M5 ou mesure de débit M5 n'a été effectuée.** Le matériel accessible est
Linux x86_64 avec environ 27–28 Gio de stockage libre au début de cette mission.
Le Mac cible et ses volumes ne sont pas accessibles ; aucun accès distant
configuré n'est documenté dans les éléments disponibles.

| Phase | Réalisation | Limite |
|---|---|---|
| A — audit | Source PR14, références officielles, métadonnées et index réels | Aucun poids complet inspecté |
| B — moteur | Décodeur natif, shards, codecs, budgets, conversation locale | Expérimental, batch 1, FP32 de calcul |
| C — validation | Primitives, préfill, cache, génération miniature, codecs et tokenizer | Poids synthétiques ; GPU ignoré ici |
| D — checkpoint réel | Manifeste et premier shard de métadonnées vérifiés | Second shard de 110 Go non téléchargé |
| E — performances | Instrumentation et essai supervisé préparés | TTFT, tok/s, SSD, GPU, swap et stabilité réelle non mesurés |
| RONIN | Adaptateur local avec historique et événements de tokens | Aucune mission autonome ni conversation réelle qualifiée |

## Sources et architecture

Sources primaires consultées et épinglées :

- [Transformers `glm5_next`](https://github.com/huggingface/transformers/tree/1b754e9a4636191852a1618fc588ad5c863ab4d5/src/transformers/models/glm5_next).
- [Configuration et tokenizer Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/f9fed3bbc82291073373d67452e9f25317ef7577).
- [Modèle GGUF llama.cpp](https://github.com/ggml-org/llama.cpp/blob/74a4fb0037243f1b6196d0f990ff0aff031356cf/src/models/glm5-next.cpp).
- [Checkpoint GGUF choisi](https://huggingface.co/ggml-org/GLM-5.3-Flash-GGUF/tree/119034bc3d9d76eb41f91508e6f9fb19fde42672).

La configuration réelle contient 45 couches : 34 KDA et 11 DSA aux indices
3, 7, …, 43. Les trois premières FFN sont denses ; les 42 autres utilisent
288 experts routés, top-8, et un expert partagé. mHC possède quatre flux et
20 itérations Sinkhorn. Le backend antérieur `glm_moe_dsa` ne représente pas
cette architecture.

Différences traitées explicitement :

- KDA applique un oubli par dimension de clé et conserve état matriciel et
  convolutions causales Q/K/V. Sa normalisation L2 utilise la somme des carrés.
  Le GGUF contient déjà `ssm_a = -exp(A_log)` : le moteur ne réapplique pas cette
  transformation.
- DSA utilise une attention latente sans RoPE. L'indexeur KPool compresse les
  groupes complets, sélectionne causalement leurs positions et inclut la queue
  incomplète selon la configuration. Les matrices latentes K/V suivent les
  transpositions de la conversion GGUF.
- mHC calcule séparément les connexions avant/après et la matrice de mélange
  Sinkhorn. La tête réduit les quatre flux par moyenne non pondérée.
- Le biais de correction sert au choix des experts ; les coefficients du
  mélange proviennent des scores sigmoid originaux. Les projections SwiGLU
  utilisent les bornes officielles, sans suppression d'expert ni de couche.
- Le tokenizer GLM4 conserve les formes Unicode distinctes, regroupe les
  chiffres par trois et utilise `ignore_merges`. Le template embarqué ouvre
  `<think>` ; il n'expose pas le mode Qwen `enable_thinking=False`.

## Réemploi et streaming

| Composant existant | Usage dans ce port |
|---|---|
| `GGUFReader`, lectures `pread`, sélection d'expert | Réutilisés ; nouveaux codecs Q2_K/Q8_0/BF16 |
| `GGUFWeights`, tuilage, cache adaptatif | Réutilisés ; résolution des noms et normalisations GLM dédiées |
| `NextConfig`, réserves système et allocations | Réutilisés ; états KDA/KV et croissance vérifiés avant allocation |
| `GGUFChatSession` | Réutilisé avec options du template propres au tokenizer |
| Superviseur PR14 et compteurs IO | Réutilisés pour le protocole d'essai, sans revendiquer de qualification locale |
| Ancien GLM DSA, Qwen GDN/RoPE | Non assimilés aux calculs GLM5 Next |
| H1/H2/H3, Metal Q4_K/Q6_K | Conservés pour leurs modèles existants ; non activés sur GLM5 |

Le lecteur multi-shard maintient un propriétaire par tenseur et un budget IO
commun, incluant les tableaux encore vivants. Il vérifie numéros de shards,
nombre global de tenseurs, doublons et métadonnées. Aucun mapping ni chargement
global des 110 Go n'est utilisé. Les experts inactifs restent sur le stockage.
Chaque expert sélectionné reçoit ses trois matrices ; les tests vérifient les
experts effectivement lus contre le routage de la référence miniature.

Le chemin actuel déquantifie seulement les tuiles nécessaires vers FP32 puis
les multiplie avec MLX, sur CPU ou sur GPU Metal si disponible. Il ne modifie
pas les fichiers GGUF et n'effectue aucune nouvelle quantification. Le cache
FP32 peut être coûteux ; cette première implémentation favorise la correction.
Le préfill avance token par token. Le préchargement asynchrone et les kernels
Metal Q2_K empaquetés restent à développer et qualifier. Aucune supériorité de
MLX sur llama.cpp ni transposition des gains Qwen n'est déduite.

## Checkpoint exact et stockage

| Fichier indispensable | Octets |
|---|---:|
| `GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf` | 9 430 112 |
| `GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf` | 110 017 922 752 |
| Total | **110 027 352 864** |

Le premier fichier contient exclusivement les métadonnées et le tokenizer.
Le second contient 1 383 tenseurs : 126 Q2_K, 306 Q8_0, **225 BF16** et 726 F32.
Le nom commercial Q2_K_S ne signifie donc pas que tous les tenseurs sont Q2_K.
Les identités SHA256, tailles, offsets et dimensions sont conservés dans
[`research/ronin_glm53/checkpoint-manifest.json`](research/ronin_glm53/checkpoint-manifest.json)
et les inventaires adjacents.

Seuls le premier fichier de 9,43 Mo et une plage de 1 Mio du second ont été
récupérés pour l'audit. Cette plage inclut environ 0,92 Mio du premier tenseur ;
elle ne constitue pas un checkpoint utilisable. Aucun BF16/FP8 complet, mmproj
vision ou sidecar MTP n'a été téléchargé.

Le précontrôle refuse un téléchargement qui ne préserverait pas 32 Gio libres
sur le volume de destination, plus 64 Mio auxiliaires. Sur macOS, il contrôle
aussi le volume système, même avec une destination externe. La réserve est une
politique conservatrice, pas une garantie d'absence de swap.

L'ancien diagnostic de 125,45 Go libres laisserait seulement environ 14,30 Gio
après installation et marge auxiliaire : **ce scénario est refusé**. Il ne
constitue pas une mesure actuelle du Mac. Le précontrôle réellement exécuté sur
Linux refuse également le téléchargement. Un futur SSD externe suffisamment
libre peut servir de destination définitive, sans cache HF ni deuxième copie.

Le protocole écrit par plages de 4 Mio dans un `.partial` identifié, vérifie
sa reprise et son SHA256, puis le renomme sur le même volume. Aucun fichier
personnel, checkpoint existant ou partiel inconnu n'est supprimé. Voir le
[protocole détaillé](research/ronin_glm53/CHECKPOINT_STORAGE_FR.md).

## Validation numérique et régressions

Les tests des codecs comparent les blocs aux décodeurs GGUF indépendants :
Q2_K, Q8_0 et les anciens Q4_K/Q6_K concordent bit à bit en FP32. Les 65 536
représentations BF16 sont vérifiées, y compris zéros signés et valeurs spéciales.
Ce résultat concerne le décodage des valeurs, pas l'égalité d'une génération
entière avec une exécution BF16 ou FP8.

Les primitives NumPy sont confrontées à des définitions Torch officielles
conservées séparément. Les tests de décodeur exportent un vrai modèle miniature
Transformers vers GGUF, puis comparent les logits de préfill et de continuation
et les tokens gloutons. Une variante utilise des blocs synthétiques Q2_K/Q8_0
déquantifiés indépendamment dans la référence : elle vérifie la même
quantification pour les deux calculs, sans toucher au checkpoint entraîné.
Les tolérances et erreurs observées figurent dans les preuves numériques.
Les comparaisons de produits matriciels sont sous tolérance, pas bit à bit.

Le tokenizer réel est confronté au tokenizer Z.ai épinglé : huit textes
Unicode/code/multilingues et douze rendus conversationnels concordent. Le GGUF
ajoute 24 entrées de padding au vocabulaire source. Les attributs de suppression
de tokens spéciaux lors du décodage sont ceux du GGUF, parfois plus larges que
ceux du tokenizer HF ; les IDs et le texte brut sont conservés.

La première suite élargie `tests_next tests_hyperstream3` produit 1 869 tests
réussis, 137 ignorés, 13 échecs et 20 erreurs, plus 126 sous-tests réussis.
La relance des quatre modules concernés sur la source PR14 inchangée reproduit
exactement les 33 mêmes cas d'échec/erreur, sans cas nouveau. Les causes incluent
`sandbox-exec` macOS indisponible, l'espace de PID observé par psutil, une API de
référence Granite modifiée dans Transformers 5.21, `accelerate` absent et trois
tests hérités d'injection d'erreur `fstat`. Les dépendances installées ici ne
reproduisent pas intégralement les anciens locks PR14. Aucune protection n'a été
retirée pour transformer ces refus en succès.

Les logs et la comparaison de base sont conservés dans
`research/ronin_glm53/regression-*.json` et `.log`. Le résultat final des tests
ciblés est de **199 réussis et 2 ignorés** en 28,81 secondes, avec les vrais
assets de tokenizer fournis séparément. Les deux tests ignorés nécessitent
respectivement Metal et un checkpoint réel complet. Le journal et les résultats
JUnit sont dans `focused-tests.log`, `focused-tests.xml` et
`focused-tests-summary.json`.

Sur les quatre cas numériques de décodeur, l'écart absolu maximal observé est
de **2,68221 × 10⁻⁶** et les ensembles d'experts routés concordent. La miniature
à 45 couches respecte 34 KDA/11 DSA, 288 experts/top-8 et un expert partagé ;
1 680 évaluations d'experts sont vérifiées sur quatre tokens de préfill et un
token de décodage. Une autre miniature exerce une sélection KPool effective.
Ces mesures ne sont ni des réponses entraînées ni des benchmarks M5. Voir
l'[audit architectural](research/ronin_glm53/ARCHITECTURE_REFERENCE_FR.md).

## Reproduction et essai futur sur Mac

Depuis cette branche, dans un environnement Python 3.12 dédié :

```bash
python -m pip install -r requirements-ronin-glm53-runtime.txt
export PYTHONPATH="$PWD/air_llm"
export MLX_ENABLE_TF32=0
python scripts/download_glm53_flash.py --destination /Volumes/Models/glm53-flash
```

La dernière commande vérifie seulement le stockage. Après les validations du
moteur retenues par l'opérateur et si le stockage le permet :

```bash
python scripts/download_glm53_flash.py --destination /Volumes/Models/glm53-flash --download
python -m airllm.nextgen.glm53_cli inspect /Volumes/Models/glm53-flash/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf
python -m airllm.nextgen.glm53_cli run /Volumes/Models/glm53-flash/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf --device gpu --prompt "Bonjour" --max-new-tokens 8 --report outputs/glm53-first-run.json
```

Ces commandes sont préparées ; elles n'ont pas été exécutées sur le M5.
L'inspection vérifie le layout sans inférence. Le lancement utilise le
superviseur existant : plafond RSS de 3 Gio, réserve RAM de 4 Gio, croissance
maximale de swap de 128 Mio et délai maximal de 900 secondes. Il garde les logs,
les événements de tokens et l'erreur éventuelle ; un nom de rapport déjà utilisé
est refusé. Ces seuils peuvent refuser une exécution pourtant possible avec
d'autres budgets ; aucune hausse automatique n'est effectuée.

Les compteurs distinguent lectures GGUF logiques et compteur disque observé
du processus. Ils ne certifient ni les écritures/lectures NAND, ni la bande
passante physique du SSD. L'utilisation GPU, la puissance, la température et
une longue génération doivent encore être mesurées sur le matériel cible.

Pour utiliser l'adaptateur conversationnel dans RONIN après qualification :

```python
from airllm.nextgen.ronin_glm53 import load_conversation

model, session = load_conversation("/Volumes/Models/glm53-flash/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf")
try:
    response = session.reply("Analyse ce code…", max_new_tokens=32,
                             on_token=lambda event: print(event, flush=True))
    print(response["text"])
finally:
    model.close()
```

L'historique est conservé en mémoire et le cache n'est réutilisé que pour un
préfixe de tokens exactement identique. Un changement de template ou un échec
reconstruit l'état. Aucune commande générée n'est exécutée automatiquement.
L'intégration aux outils de programmation et les permissions de périmètre
restent du ressort du projet RONIN ; son dépôt/chemin exact n'est pas fourni.

## Limites restantes

- Vision, MTP, outils multimodaux, batch supérieur à 1, échantillonnage et
  partage inter-couche d'indexeur non pris en charge dans cette route.
- Contexte initial limité à 512 tokens par défaut malgré la capacité théorique
  du modèle. Le coût mémoire des états et la longueur maximale sûre restent à
  mesurer sur 16 Gio.
- Aucun kernel Q2_K Metal fusionné, préfill H2, préchargement asynchrone ou
  comparaison de performance MLX/llama.cpp qualifié pour ce modèle.
- Fidélité du checkpoint Q2_K_S entraîné face à llama.cpp, routage réel,
  cohérence sur génération longue et stabilité thermique encore à établir.
- Les égalités FP32 exactes à la frontière du top-k MoE ou KPool sont refusées
  avec un diagnostic : les références CPU/Metal n'ont pas de convention unique
  de sélection dans ces cas. L'ordre interne du mélange conserve une tolérance
  numérique ; aucune identité bit à bit des logits n'est revendiquée.
- Les diagnostics conservent `ronin_ready=False`,
  `real_checkpoint_generation_validated=False` et `m5_execution_validated=False`.
  Une génération seule ne suffit pas à lever ces qualifications.
