# Rapport d’ingénierie — 22 octobre 2026

Cette livraison contient du code exécuté, des tests, des mesures brutes et une
branche Git restaurable. Elle ne démontre pas encore une génération interactive
avec les checkpoints complets GLM-5.3 ou Qwen3.8 sur un Mac de 16 Go.

## Provenance et méthode

L’archive demandée a été matérialisée, vérifiée intégralement par CRC ZIP, puis
comparée au hash Git du fichier public. Son SHA-256 est conservé dans
`evidence-ooc/source-integrity.json`. Une première révision importe exactement
les sources extraites, sans leur attribuer les changements ultérieurs. Le
parent est le commit public `9453681973ddfc93d66c6144c8eca2d5a39a8f1c`.
Les licences, notices et preuves historiques sont conservées.

L’environnement exécuté est Linux x86-64, Python 3.12.14, MLX CPU 0.32.4,
Torch 2.14.1+cpu, Transformers 5.20.2. Les versions et empreintes se trouvent
sous `evidence-ooc/baseline/` et `evidence-ooc/candidate/`. Aucun poids des
checkpoints géants n’a été téléchargé. Aucun résultat CPU n’est présenté comme
un résultat Metal. Énergie, utilisation GPU et thermique ne sont pas mesurées
sur cette machine ; leurs champs restent indisponibles lorsque nécessaire.

## Ce qui change réellement

Le lecteur sait regrouper des plages adjacentes sans copies par tenseur, avec
plafond d’allocation et comptage des espaces éventuellement relus. Les échelles
FP8 sont lues uniquement sur les lignes utiles à la tuile. Le regroupement des
embeddings est optionnel. Une politique de cache optionnelle utilise les coûts
observés de lecture et de conversion, et conserve les traces nécessaires à une
évaluation. Deux courses de réservation/fermeture du préchargement sont corrigées.
Le regroupement d’experts déjà présents dans un bloc existait dans la source ;
la nouveauté spéculative exploite cette réutilisation, elle ne réinvente pas ce
calcul MoE préexistant.

La conversion INT4/INT8 écrit directement les tuiles dans un safetensors final,
sans tableau compressé complet en RAM. Chaque matrice est publiée atomiquement,
avec empreinte SHA-256 et métriques d’erreur. La reprise vérifie les nouvelles
empreintes et refuse la corruption. Les en-têtes, dimensions et allocations
sont contrôlés avant les lectures. Les anciens manifests restent lisibles,
mais sans hash ils ne peuvent pas être certifiés par un audit strict.

La vérification spéculative gloutonne est exécutée par le vrai modèle cible.
Le brouillon propose, le cible vérifie chaque token, le cache est ramené au
préfixe accepté et le cible fournit correction ou bonus. La profondeur dépend
de l’acceptation observée et de coûts mesurés ; le système revient au décodage
ordinaire quand le bénéfice disparaît. Le chemin stochastique et MTP ne sont pas
implémentés. Aucune substitution d’un expert prédit à un expert choisi n’a lieu.

Un backend Qwen expérimental distinct partage le lecteur et le cache génériques.
Il calcule les projections, convolution causale, récurrence Gated DeltaNet en
FP32, attention complète avec porte, RoPE et MoE sélectionné exactement. Ses
états récurrents ont de vrais snapshots avec budget et restauration, au lieu
d’une simple modification de position. Il reste explicitement expérimental.
L’API locale propose le sous-ensemble texte non streaming d’OpenAI, sur loopback,
sans client d’inférence distant. Les appels d’outils ne sont pas implémentés.

## Tests automatisés

Les suites finales passent : 157 tests Next, 34 tests Ultra et 66 tests hérités,
soit **257 tests réussis**, avec 9 tests ignorés pour leurs conditions matérielles
ou optionnelles. Les sous-tests Pytest ne sont pas additionnés une seconde fois.
Les XML et journaux complets sont dans `evidence-ooc/tests-*` et leur synthèse dans
`validation-summary.json`. La wheel `4.0.0+next.2` est construite et ses nouveaux
modules sont importés depuis une installation séparée. Une revue indépendante a
aussi conduit à rejeter les alias de RoPE non pris en charge, au lieu de les
ignorer silencieusement.

## Résultats avant/après

| Expérience | Avant | Après | Interprétation |
|---|---:|---:|---|
| Conversion d’une matrice 32768 × 4096, pic RSS médian | 126,14 Mo | 44,72 Mo | −64,5 % de mémoire du processus |
| Même conversion, durée médiane | 3,178 s | 3,756 s | +18,2 % de temps ; compromis mémoire assumé |
| Lecture de 16 Mio, nombre d’appels | 256 | 32 | Payloads identiques par SHA-256 |
| Lecture pread chaude, médiane | 4,455 ms | 4,468 ms | Aucun gain ; option désactivée par défaut |
| Lecture pread avec éviction conseillée | 6,642 ms | 5,705 ms | 1,164× sur ce chemin Linux seulement |
| GLM miniature, vérification spéculative profondeur 8 | 0,343 s | 0,196 s | 1,75× dans ce cas de recherche précis |
| Même cas, octets logiques du cible | 12 991 232 | 3 138 528 | ÷4,14 ; octets totaux avec brouillon ÷3,39 |

La conversion compare trois processus frais par version, dans un ordre alterné,
avec le même flux de tuiles déterministe. Son pic RSS inclut interpréteur et
préparation des tuiles. La légère différence dans la statistique d’erreur L2
vient du calcul des statistiques en float64 ; les poids quantifiés utilisent
le même algorithme. Voir `conversion-comparison.json` et les mesures individuelles.

Les lectures utilisent neuf répétitions. Les mesures chaudes ont zéro octet
physique imputé au processus ; elles mesurent principalement le chemin mémoire.
Le mode avec conseil d’éviction enregistre environ 16,78 Mo de lectures physiques,
mais une éviction conseillée ne garantit pas un cache froid. Il ne certifie ni
le SSD externe ni sa NAND. Voir `io-warm.json` et `io-cold-hint.json`.

L’expérience spéculative compte 30 générations, ordre mélangé et trois répétitions
par variante. Les caches logiciels du cible commencent vides avec 4 Kio de budget.
Le brouillon précis est une seconde instance indépendante du **même petit modèle
aléatoire**, avec 8 Mio de cache : c’est un cas favorable de validation et de
réutilisation, pas un petit modèle entraîné disponible pour GLM-5.3. Son calcul,
ses lectures et son temps de proposition sont inclus. Le temps de construction
des modèles est séparé du temps de génération. Un brouillon volontairement faux
active le retour au décodage ordinaire dans 3 essais sur 3. Les anciennes mesures
avec concurrence ou politique antérieure sont conservées sous des noms distincts.
Voir `speculative.json`, les journaux et `docs/SPECULATIVE_STREAMING.md`.

La comparaison gloutonne ordinaire avant/après utilise exactement cinq mêmes
checkpoints miniatures, cinq répétitions et deux ordres de version. Les logits
sont strictement identiques (écart maximal 0), et toutes les séquences de 32 tokens
coïncident. Les débits varient selon l’ordre sans gain général robuste : les
optimisations ne justifient pas une annonce globale « 2× plus rapide ».
Voir `candidate/revision-pair.json` et sa répétition en ordre inversé.

Le Qwen hybride miniature produit 24 tokens identiques à Transformers sur trois
prompts. Son erreur maximale de logits FP32 est 5,96 × 10⁻⁸. Une validation BF16
séparée utilise des tolérances explicites propres à ce cas. Ce sont des modèles
aléatoires réduits, pas une mesure de qualité linguistique sur les poids entraînés.
Voir `qwen-runtime-parity.json` et les tests du backend.

## Limites physiques et statut des modèles

L’estimation issue de la configuration GLM donne environ 22,65 Go d’experts
routés actifs par token en FP8, ou 14,16 Go avec le format affine INT4 groupe 64
et ses échelles/biais, avant cache. Ce sont des estimations de trafic théorique
sans les autres poids, le calcul, les surcoûts système et la qualité. Pour un
trafic effectif D et un débit de stockage mesuré B, le temps de transfert seul
est au moins D/B. Les scénarios de `theoretical-glm.json` utilisent des débits
explicitement hypothétiques, jamais le débit présumé d’un câble USB-C.

La réutilisation d’experts sur plusieurs tokens n’est pas automatique : leurs
routes peuvent différer. À titre de modèle mathématique seulement, avec E=256,
K=8 et 8 routes indépendantes uniformes, l’union attendue vaut
E × (1 − (1 − K/E)^8) ≈ 57,42 experts, au lieu de 64 lectures séparées :
seulement environ 1,115× de gain sur les experts. Les vraies routes doivent être
mesurées. Le gain du petit modèle ne doit donc pas être extrapolé au géant.

Le checkpoint Qwen BF16 déclare 4 892 365 451 008 octets de tenseurs : il dépasse
2 To. La variante NVIDIA inspectée mélange NVFP4 et FP8 ; ce n’est pas le format
affine INT4 de MLX et son encombrement exact n’a pas été certifié ici. Le backend
refuse ce format, ainsi que la déclaration de porte `swish` du Qwen3.8 officiel,
car la référence inspectée ne valide pas explicitement cette sémantique.

Le GLM FP8 est décodé vers la précision de calcul choisie. Les kernels de
quantification dynamique des activations FP8 de serveurs de production ne sont
pas reproduits ; la parité avec une référence de poids déquantifiés ne certifie
pas ces kernels ni la qualité du checkpoint complet. Toute quantification plus
agressive reste optionnelle et nécessite une évaluation sur le modèle réel.

| Domaine | Validé dans cette livraison | Reste à vérifier ou développer |
|---|---|---|
| GLM DSA | Calculs et génération miniatures ; parité avant/après | Poids complets, qualité, M5/Metal |
| Qwen hybride | Backend flottant miniature, MoE, snapshots, génération | Porte Qwen3.8, NVFP4, MTP, modèle complet |
| Spéculation | Glouton GLM, rollback, compteurs et arrêt adaptatif | Brouillon entraîné utile, stochastique/arbre/MTP, intégration Qwen |
| E/S et mémoire | Mesures Linux et budgets logiciels, conversion bornée | SSD réel, pression macOS/swap, budget MLX commun à deux modèles |
| Kernels | Opérations natives MLX et matmul affine testé | Profilage M5 ; fusion Metal/C++ seulement si mesurée utile |
| API locale | Texte/chat non streaming et génération HTTP miniature | Streaming, outils et compatibilité complète des agents |

## Installation, validation et reprise Git

Le guide `OOC_M5_GUIDE.md` fournit installation arm64, mesure du SSD réel,
validation Metal explicite et session prolongée. Les 153 Go/s de mémoire unifiée
annoncés par Apple ne sont pas le débit du SSD externe. L’accélération M5 via MLX
nécessite l’environnement macOS pris en charge ; aucun kernel TensorOps maison
ni mesure d’énergie/thermique M5 n’est revendiqué.

Pour auditer les poids quantifiés avant leur utilisation :

```bash
python -m airllm.nextgen.cli verify /local/base --quantized-dir /local/sidecar
```

Ce scan explicite vérifie les hashes des sidecars. L’inférence ne relit pas
implicitement toutes les matrices pour les hasher à chaque chargement. Le
fingerprint du checkpoint original porte sur les métadonnées de fichiers et
l’index, ce n’est pas une preuve cryptographique de tous ses poids. Les sidecars
restent actuellement liés à ce fingerprint, donc leur déplacement nécessite de
préserver cette identité ou de les reconvertir.

Les sources, tests et résultats bruts sont publiés sur la branche dédiée
[`research/ooc-engine-2026-10-22`](https://github.com/iliasm-a11y/airllm-next/tree/research/ooc-engine-2026-10-22).
Le refus d’accès GitHub initial a été résolu par l’autorisation de l’utilisateur.
Ni `main` ni le dépôt officiel d’AirLLM n’ont été modifiés.

Un ancien jeton Hugging Face intégré au notebook `test_sealllm.ipynb` a été
retiré avant publication ; cet exemple lit désormais la variable `HF_TOKEN`.
Les références de commits locales du rapport et le bundle initial conservent
l’état de l’audit. Les commits GitHub ont été recréés avec le notebook corrigé
et ce statut de publication ; les mesures d’inférence n’ont pas été modifiées.

Le critère final « grand modèle interactif, qualité vérifiée et mémoire maîtrisée
sur M5 16 Go » n’est pas encore atteint. Cette livraison avance le moteur et
fournit des preuves réexécutables ; elle ne remplace pas la validation matérielle
et les implémentations encore manquantes par une annonce de compatibilité.
