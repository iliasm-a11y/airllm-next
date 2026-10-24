# AirLLM Next — Throughput GGUF natif sur MacBook Air M5

24 octobre 2026. Recherche issue de la [PR #5](https://github.com/iliasm-a11y/airllm-next/pull/5), sans fusion dans `main` ni publication d'une version stable 9.0.

**Le décodage réel passe de 1.044 à 1.453 token/s, soit +39.1 %. Le débit de bout en bout passe de 0.612 à 0.801 token/s (+30.9 %). Les six paires sont favorables ; les douze générations atteignent EOS avec les mêmes tokens et vecteurs complets de logits, sans croissance de swap ni dépassement détecté. Le seuil soutenu de 1,5 token/s n'est pas atteint.**

## Expérience principale

Le véritable Qwen3.6-35B-A3B GGUF Q4_K_M de 23 938 321 664 octets est lu sélectivement par AirLLM Next, MLX/Metal FP32, sur le M5 à 16 Gio et SSD interne. Aucun service Ollama ne produit les réponses. Le modèle reste dans son magasin existant, sans duplication. Son SHA-256 intégral a été recalculé en lecture seule après la campagne principale : `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` ; taille, inode et date de modification sont inchangés. Les contrôles courts ultérieurs n'écrivent aucun poids.

La référence utilise le chemin numérique de #5, les nouvelles options désactivées, des tuiles de 2 048 lignes et le cache adaptatif. Le candidat associe regroupement exact des experts déjà sélectionnés, concaténation des tuiles évaluées, tuiles de 8 192 lignes et cache LRU. Aucun gain n'est obtenu en changeant les poids, la précision, le routage ou la consigne.

Budgets identiques : cache 64 Mio, E/S 64 Mio, états/KV 256 Mio, travail 128 Mio, lecture individuelle 16 Mio. Préremplissage par blocs de 16, contexte 1 280, vrai tokenizer/template GGUF, greedy, thinking désactivé et EOS actifs. L'import sans copie, le pipeline et les groupes SIMD élargis restent désactivés dans le candidat retenu.

Trois prompts et deux répétitions donnent six paires / douze processus frais, contrebalancés par prompt et mélangés avec la graine publiée. Le préremplissage, les écritures de preuves par token et les empreintes sont compris dans le coût. Aucun cache système n'est purgé ; la machine reste utilisable et non isolée. Le [protocole](research/throughput/PROTOCOL_FR.md) donne les définitions exactes et la correction du premier tirage non équilibré, interrompu avant tout résultat apparié.

| Prompt | Tokens d'entrée | Tokens générés, EOS inclus | Décodage référence | Décodage candidat | TTFT moteur médian réf. / cand., s |
|---|---:|---:|---:|---:|---:|
| short | 45 | 8 | 1.055 | 1.439 | 16.36 / 12.93 |
| explanation | 72 | 152 | 1.048 | 1.465 | 23.14 / 20.47 |
| long_context | 233 | 26 | 1.015 | 1.386 | 83.44 / 67.19 |

Les débits sont des ratios de sommes. Chaque variante produit 372 tokens. Les processus de référence prennent 607.76 s cumulées, ceux du candidat 464.26 s. Le facteur médian apparié de décodage vaut 1.367× ; il diffère du facteur agrégé car les longueurs de réponse pondèrent ce dernier. Les six rapports et leur dispersion sont dans [summary.json](research/throughput/results/summary.json), les métriques par exécution dans [benchmarks.csv](research/throughput/results/benchmarks.csv).

## Coûts dominants et mécanismes

Le profil séparé de la courte génération identifie les attentes de graphes MLX, les lectures pread, puis la préparation des buffers/cache. Temps muraux hôte **exclusifs par token décodé**, sur sept pas après le premier token :

| Étape | Référence, s/token | Candidat, s/token |
|---|---:|---:|
| Évaluations / synchronisations MLX | 0.5287 | 0.3381 |
| Lectures pread | 0.1857 | 0.1840 |
| Buffers compactés et cache, hors lectures | 0.0699 | 0.0697 |
| Admission / éviction du cache | 0.0562 | 0.0034 |
| Appels d’import hôte vers MLX | 0.0376 | 0.0415 |

Les appels `mx.eval` du décodage passent de 38 752 à 19 390 sur ce contrôle. Le regroupement des huit experts remplace 24 appels indépendants de projection par trois appels groupés par couche mono-token. Le préremplissage regroupe les experts ayant le même nombre de positions, au plus huit à la fois. Les sommes SIMD par matrice et l'addition des experts dans l'ordre d'origine restent inchangées. Les grands blocs réduisent aussi les appels des projections et de la tête de sortie ; le cache LRU réduit le coût de gestion sur cette charge de travail. Le cache adaptatif global reste le défaut existant.

Ces profils instrumentés perturbent la mesure ; ils expliquent les mécanismes, pas le gain chronométré final. Les temps CPU du processus et les temps muraux sont distincts. Les groupes inclusifs sont imbriqués et ne doivent pas être additionnés. L'attente d'un graphe inclut soumission, calcul en attente et synchronisation ; ce n'est pas du temps GPU isolé. La déquantification Q4/Q6 est fusionnée au kernel, sans durée GPU séparée inventée. L'API MLX installée n'expose pas les timestamps GPU par commande et `xctrace` n'est pas disponible. Les groupes attention, DeltaNet, routage, experts partagés/sélectionnés, tête de sortie, imports et cache sont conservés dans [profiles.json](research/throughput/results/profiles.json).

## Contrôles par retrait et résultats négatifs

Deux blocs courts en ordre miroir retirent un réglage à la fois ; toutes les sorties, empreintes de logits et routes restent identiques jusqu'à EOS. Le facteur compare la configuration complète au réglage retiré. Ce sont des effets locaux conditionnels, sans addition causale des pourcentages.

| Configuration | Bloc 1, tok/s | Bloc 2, tok/s | Complète / retrait, médiane |
|---|---:|---:|---:|
| Configuration complète | 1.452 | 1.441 | 1.000× |
| Sans regroupement des experts | 1.264 | 1.244 | 1.154× |
| Sans concaténation | 1.448 | 1.441 | 1.001× |
| Cache adaptatif au lieu de LRU | 1.328 | 1.329 | 1.089× |
| Tuiles 2 048 au lieu de 8 192 | 1.395 | 1.392 | 1.038× |

L'effet propre de la concaténation est faible sur ce court cas ; aucun gain indépendant robuste ne lui est attribué. Les pilotes suivants restent exploratoires, sans répétitions suffisantes pour en faire des gains confirmés :

| Pilote | TTFT, s | Décodage, tok/s |
|---|---:|---:|
| screen-baseline | 18.82 | 1.018 |
| screen-combined-lru | 15.99 | 1.378 |
| screen-combined-wide | 13.74 | 1.158 |
| screen-combined-zero | 14.10 | 1.221 |
| screen-concat-lru-zero | 21.88 | 1.129 |
| screen-pipeline-zero | 13.91 | 1.133 |
| screen-simd4 | 16.76 | 1.377 |
| screen-simd8 | 14.06 | 1.166 |
| screen2-combined-lru-zero | 13.55 | 1.341 |
| screen2-pipeline-lru-zero | 14.48 | 1.237 |
| screen2-pipeline-lru | 13.53 | 1.334 |

Le pipeline à deux tuiles, l'import sans copie et les groupes SIMD élargis n'apportent pas de bénéfice supplémentaire convaincant dans ces essais. Ils restent facultatifs, couverts par des tests et désactivés dans la configuration retenue. Aucun réencodage avec pertes, sidecar ou placement modifié des poids n'est utilisé.

## Fidélité et draft peu coûteux

Les 372 positions de génération sont comparées entre variantes par empreinte du vecteur complet de logits ; IDs d'entrée/sortie, ensembles/comptes d'experts par couche et positions de l'état concordent. Les probabilités brutes du routeur ne sont pas toutes capturées. Huit positions séparées comparent directement les tableaux binaires de logits et les empreintes complètes des états convolution, récurrence et KV ; elles sont identiques. Ce contrôle limité d'états ne devient pas une assertion sur chaque position longue. Les tolérances des tests existants ne sont pas élargies.

Le composant causal existant `PromptSpanDraft` a été évalué sur les vrais préfixes enregistrés : à profondeur 4, six tokens proposés concordent dans l'explication de 152 tokens, onze dans l'extraction de 26 tokens et aucun dans la réponse courte de huit tokens. Le coût CPU du draft est de l'ordre de la milliseconde pour une réponse, mais cette étude ne mesure ni vérification cible ni rollback récurrent. **Aucun gain de génération spéculative n'est revendiqué**, et les gains historiques de vérification à positions connues ne sont pas réutilisés comme débit réel. Voir [draft-study.json](research/throughput/results/draft-study.json).

Les [réponses intégrales](research/throughput/RESPONSES_FR.md) sont celles du modèle. L'explication initiale dépasse légèrement les 100 mots selon un comptage par espaces et parle du swap, alors que le moteur n'en ajoute pas. La fidélité au moteur de référence préserve aussi ses erreurs ; ce travail ne prétend pas améliorer la qualité sémantique générale du modèle.

## Mémoire, swap et stockage

| Mesure primaire | Référence | Candidat |
|---|---:|---:|
| Pic RSS de l'arbre supervisé | 540.16 Mio | 485.08 Mio |
| Mémoire système disponible minimale | 4.262 Gio | 4.270 Gio |
| Swap initial/final, toutes les fenêtres | 0 / 0 Mio | 0 / 0 Mio |
| Croissance maximale observée | 0 Mio | 0 Mio |
| Lectures logiques de décodage par token | 1.964 Go | 1.965 Go |
| Lectures stockage OS du processus par token | 165.9 Mo | 173.9 Mo |

Les protections restent : plafond RSS 2 Gio, réserve système 3 Gio, admission 768 Mio, croissance swap 128 Mio, pression normale exigée, surveillance 50 ms et délai maximal 900 s. Aucune application personnelle fermée ni protection macOS désactivée. Aucun dépassement détecté dans les générations principales ou contrôles natifs ; le tirage initial est un arrêt de protocole, pas un dépassement mémoire. L'échantillonnage ne garantit pas l'absence d'un pic entre deux observations.

Le RSS ne représente pas tout le cache de fichiers ni toute la mémoire unifiée. Les compteurs mémoire se recouvrent et ne doivent pas être additionnés. `ri_diskio_bytesread` est un compteur macOS au niveau du processus, pas une mesure NAND ou un débit SSD froid. Les lectures logiques ne baissent pratiquement pas : le gain mesuré vient surtout de l'orchestration et des calculs, sans preuve de réduction du trafic physique. Aucun taux de cache OS par lecture n'est disponible.

## Objectifs et limites physiques

À volume logique constant d'environ 1.965 Go par token, les besoins conditionnels sont :

| Objectif, tok/s | Service logique nécessaire, Go/s | Réduction supplémentaire du coût/token depuis le candidat |
|---|---:|---:|
| 1.5 | 2.95 | 3.2 % |
| 2 | 3.93 | 27.4 % |
| 4 | 7.86 | 63.7 % |
| 8 | 15.72 | 81.8 % |

Ce calcul ne démontre pas une limite physique interdisant 2, 4 ou 8 tok/s. Les caches, le chevauchement, le kernel et les latences peuvent changer ; les compteurs disponibles ne certifient pas la bande passante NAND ni un temps GPU pur. Atteindre les seuils supérieurs demande de nouvelles mesures et méthodes. Le regroupement est proportionnel aux experts actifs et reste borné par les mêmes budgets, mais il n'est pas validé sur plusieurs centaines de milliards de paramètres. Aucun SSD externe, tenue thermique prolongée, énergie ou maintien quotidien n'est certifié.

## Validation, garde-fous et reproduction

**1 195 tests réussis, un test NVIDIA ignoré, onze sous-tests réussis**, avec les contrôles GGUF réels et la capture historique du routeur activés. Les 1 166 tests de #5 sont conservés et 29 contrôles ajoutés : kernels groupés, préremplissage irrégulier, tuiles, cache, durée de vie des buffers, refus avant allocation, rollback et garde-fous. Une première suite de 1 193 tests avait terminé, puis son nettoyage avait échoué sur `killpg` après sortie du groupe. Le correctif exige une identité possédée encore vivante et persiste les erreurs de nettoyage ; les plafonds ne sont pas réduits. Les journaux originaux et le contrôle final sont conservés.

Le code est livré en recherche, avec les nouvelles options désactivées par défaut. La [revue locale](research/throughput/CODE_REVIEW_FR.md), le [protocole et les commandes de reproduction](research/throughput/README.md) décrivent les périmètres. Les JSON/journaux/commandes d'origine sont compressés sans perte dans `evidence-throughput/raw-json-logs.tar.gz`, avec empreintes dans `raw-manifest.json`. La restauration vérifie chaque fichier dans un nouveau répertoire ; les poids, assets tokenizer et captures binaires restent locaux. Les métriques et analyses lisibles sont dans `research/throughput/results/`.

## Livraison des preuves

L'archive publique restaure **159 fichiers** dont toutes les empreintes ont été vérifiées. Une analyse des preuves fraîchement restaurées reproduit le fichier `summary.json` octet pour octet. Les sources mesurées restent identiques aux fichiers du moteur livrés, avec leur manifeste séparé. Le contrôle réel de la CLI optimisée termine avec « Bleu, blanc, rouge », sans violation ; son budget KV historique de 192 Mio reste distingué des 256 Mio de la campagne.

Commande de campagne effectivement exécutée depuis le dépôt de travail :

```sh
/Users/macbook/Documents/Codex/2026-10-22/files-pasted-by-the-user-mission/work/validation-venv/bin/python benchmarks/campaign_throughput.py --model /Users/macbook/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d --tasks research/throughput/primary-tasks.json --output evidence-throughput/primary-balanced --repeats 2 --simd-groups 1
```

Pour une nouvelle exécution, choisir un autre dossier de sortie ; le programme refuse d'écraser les preuves existantes. Les commandes exactes de chaque enfant sont dans l'archive.
