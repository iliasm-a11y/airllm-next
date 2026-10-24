# Réemploi d'experts — réanalyse hors ligne 5.1

## Résultat et portée

Cette étape exécute une **nouvelle analyse de traces historiques**, sans aucun forward modèle, sans ouvrir de poids et sans inférence simultanée. Elle produit des comptes de demandes, des fréquences et des unions vérifiables; elle ne mesure ni tokens/s, ni latence de chargement d'expert, ni trafic SSD physique, ni nouvelle qualité de génération.

L'archive publique contient six contrôles de parité du **même** court tour Qwen3.6-35B-A3B, pas six conversations indépendantes. Ils ont les mêmes routes, entrées, sorties et logits capturés : le script les déduplique. Les huit captures comprennent un préfill agrégé et sept vrais pas de décodage. La position de cache passe de 45 à 52; les histogrammes du préfill ne permettent pas de reconstruire chaque position et sont exclus. Les campagnes longues possèdent surtout des empreintes de routes, insuffisantes pour cette analyse.

- Archive : `evidence-throughput/raw-json-logs.tar.gz`, SHA-256 `2d796c8489d38eef38e6ae692b5332f170e62cec3f1ba2dac18f2472230296b8`.
- Sources Qwen : six membres `parity-*.json` / `pilot-wide-combined.json`, commits historiques `76fd683c37534807e21b5fda07a3d717b163215d` et `97dd05b7af693a6387c699dcbd090c77da01ae2e`; ces commits et empreintes de sources restent enregistrés comme provenance historique, pas comme validation du moteur courant.
- Routes OLMoE héritées : `research/exact_io/inherited/routing_traces_olmoe.npz`, SHA-256 `fa055bc19a2c0f8a654a0a4a5d919097a43c893d1581fc14fb9e06a657444b91`, forme `[16 couches, 8 catégories, 4096 positions, top-8]`. Ce sont des positions **teacher-forced**, pas huit générations libres exécutées ensemble.

Le résultat détaillé se trouve dans [results/expert-reuse.json](results/expert-reuse.json). Les coûts par expert absents restent explicitement `null`.

## A. Fréquences, cache et prévision causale sur Qwen

Les sept pas décodés demandent **2 240 couples couche/expert**, dont **1 017 distincts**. Le script conserve les fréquences par couche. Un expert numéro 30 dans deux couches différentes correspond à deux objets distincts.

Les prédicteurs sont évalués sur les six transitions disponibles : ils proposent avant d'apprendre la sélection cible. La transition combine uniquement les cooccurrences déjà observées depuis les experts du pas précédent, avec un repli sur les fréquences antérieures.

| Prévision top-8 | Bons experts / 1 920 | Mauvaises prévisions | Rappel |
|---|---:|---:|---:|
| Même sélection qu'au token précédent | 918 | 1 002 | 47,81 % |
| Fréquence causale | 908 | 1 012 | 47,29 % |
| Transition causale | 973 | 947 | 50,68 % |

La légère différence ne suffit pas pour activer une prélecture : ces sept positions concernent un seul court prompt, et les prédictions fausses représentent encore presque la moitié des propositions. Les experts déjà résidents, la bande passante perdue, les prédictions tardives et les évictions causées ne sont pas enregistrés ici. Une amélioration du rappel ne démontre donc pas un avantage de génération.

Un rejeu de cache global froid compare les mêmes ensembles demandés. La capacité est exprimée en **unités d'experts de même taille supposée**, sans attention, échelles, poids décodés ou coût de buffers. Les histogrammes perdent l'ordre top-k : l'admission emploie un ordre d'identifiant stable; une couche ne peut pas évincer ses demandes courantes. Ces hypothèses excluent toute revendication de RAM ou de débit réel.

| Capacité en unités | Misses LRU | Misses LFU | Misses LFU avec admission |
|---:|---:|---:|---:|
| 64 | 2 240 | 2 234 | 2 217 |
| 128 | 2 240 | 1 999 | 1 961 |
| 256 | 2 240 | 1 630 | 1 630 |
| 512 | 1 091 | 1 188 | 1 188 |

La préférence entre récence et fréquence change avec le budget. Le balayage de 40 couches explique pourquoi LRU ne réutilise rien sous 320 unités dans ce petit rejeu. Cela ne valide pas un cache LFU moteur : la vraie taille et le coût des experts et des autres tenseurs sont indispensables. L'ablation **historique** `research/throughput/results/ablations.json` mesurait d'ailleurs un taux médian 1,329 token/s avec cache adaptatif contre 1,447 pour le bras complet de cette ancienne campagne; elle n'a pas été réexécutée ici.

L'expérience suivante pertinente est une admission au coût évité par octet résident, avec taille réelle et latence observée, comparée à LRU et LFU dans le même budget. Compter les erreurs de prévision en octets chargés inutiles, évictions et attente de premier plan. Une politique exacte change le placement, jamais la sélection du routeur.

## C. Partage multi-séquence par rejeu

Le rejeu OLMoE aligne uniquement les positions enregistrées 512 à 1023 de catégories différentes. Il énumère toutes les combinaisons de 2 et 4 catégories, puis les huit catégories. C'est un ordonnancement contrefactuel de demandes connues; aucune indépendance statistique ni simultanéité des générations n'est supposée.

| Flux catégoriels regroupés | Combinaisons | Réduction idéale des demandes par union |
|---:|---:|---:|
| 2 | 28 | 1,077× |
| 4 | 70 | 1,240× |
| 8 | 1 | 1,592× |

Ces facteurs supposent une lecture unique de chaque expert de l'union par couche/pas. Ils excluent les autres poids, KV, mémoire temporaire, déquantification, files d'attente, échéances et état des caches. Les demandes répétées dans différentes combinaisons servent à calculer une moyenne sur les groupes; ce ne sont pas des observations supplémentaires indépendantes.

Sur la courte séquence Qwen, regrouper les **positions temporelles d'une seule séquence** donne des facteurs de demandes 1,314× à profondeur 2 et 1,792× à profondeur 4. Ces nombres concernent de vrais préfixes; les routes peuvent changer après un brouillon incorrect. Ils ne sont ni des résultats multi-conversation Qwen, ni des gains spéculatifs.

### Expérience exacte suivante, sans générations simultanées

1. Capturer deux ou quatre prompts différents lors de **générations séquentielles supervisées**, avec identités ordonnées des experts, tailles des projections, logits et empreintes de chaque banque KV/récurrente. Garder une borne stricte de positions et de volume de capture; les vieux duplicatas ne sont pas des flux distincts.
2. Rejouer leurs événements déjà enregistrés avec trois ordres : conversation entière après conversation entière, un pas de chaque conversation, puis union par couche et par expert. Le simulateur doit vérifier l'égalité du multiensemble des demandes et préserver les dépendances intra-conversation. Aucun modèle n'est nécessaire pour établir ces comptes.
3. Pour tester numériquement un futur kernel partagé, capturer aussi les lignes d'activation aux frontières d'un bloc limité. Calculer l'expert une fois par groupe de lignes, puis disperser les résultats vers chaque séquence en conservant son ordre canonique d'addition. Comparer chaque ligne à son exécution isolée; ceci reste une preuve de composant. Ne pas intervertir les états récurrents ni valider le cache KV par une seule empreinte de routes.
4. Seule une campagne complète ultérieure pourra mesurer temps, mémoire et avantage de génération. Aucune mise en concurrence de modèles ni capacité «128 agents» n'est introduite par ce rejeu.

Le script livré réalise déjà l'étape de comptage d'unions sur les traces OLMoE disponibles. La capture nécessaire à une campagne Qwen indépendante manque encore.

## D. Quantification et compression : maintenir les contrats

Les expert-id ne contiennent aucune taille ni précision : on ne peut pas transformer leur compte en octets sans métadonnées vérifiées. Une quantification plus petite change les poids; elle exige une comparaison de fidélité contre une référence définie. Une compression sans perte doit au contraire reconstruire exactement les octets du checkpoint courant.

Le fichier **historique** `research/exact_io/results/compression.json` porte sur 21,23 MB d'échantillons BF16 entraînés, pas sur le GGUF actuel : ratio zstd1 0,770, débit décodé+hash+allocations 1,018 GB/s; palette C ratio 0,750, 34,21 GB/s pour le kernel seul mais 3,015 GB/s avec allocation et hash. Ces mesures anciennes ne sont pas réexécutées ici et le kernel seul n'est pas le transport complet. Elles ne prouvent ni le ratio d'un poids Q4_K, ni un débit modèle amélioré.

Pour une décompression séquentielle de ratio `r`, débit disque `B` et débit décodé `D`, la condition favorable simplifiée est `r/B + 1/D < 1/B`. En recouvrement idéal, le plafond change, mais les bulles et buffers doivent être mesurés. Le ratio seul ne suffit jamais.

## E. Décodage spéculatif : le manque de preuve reste explicite

`research/throughput/results/draft-study.json` contient une ancienne étude causale de propositions n-gram sur tokens enregistrés. Elle marque déjà `generation_speedup_measured=false` et `recurrent_rollback_cost_measured=false`. Le court tour n'obtient aucune proposition; les autres tours obtiennent quelques correspondances sur tokens connus. Aucun bloc cible n'a été exécuté pour ces propositions, donc aucune acceptation réelle ni économie d'experts n'est validée.

La nouveauté marginale des vrais préfixes peut aider au budget de coût, mais ne prédit pas les routes des branches rejetées. Un prochain test doit comptabiliser les unions réellement vérifiées, le draft, les tokens acceptés et le rollback KV/récurrent. Pour l'égalité gloutonne, garder réductions et top-k déterministes; pour la même loi à température positive, une correction probabiliste correcte est nécessaire. L'hypothèse «le modèle complet vérifie» ne suffit pas à elle seule.

## PRISME : réutiliser le modèle de coût, corriger les surinterprétations

Branche inspectée sans merge : `origin/research/prisme-shared-execution-2026-10-23`, commit `b7204624bc4ca99d61615920f16033da076eca39`. Les objets utiles sont les ensembles d'experts `U`, leur différence avec le cache `C`, les octets par token accepté, les coûts non routés et le rollback récurrent conditionnel à l'invariance des kernels. `prisme_sim.py` est un modèle de coût avec hypothèses de débit, pas un moteur d'ordonnancement publié ni un benchmark géant.

Les propositions héritées demandent les restrictions suivantes avant réemploi :

- **T1/T8, entropie :** des entropies marginales empiriques ou une faible information mutuelle par paire ne prouvent ni l'indépendance jointe, ni une borne absolue contre tout codage sans perte d'un checkpoint fixe. Les ratios réellement vérifiés ont une portée d'échantillon, pas d'impossibilité universelle.
- **T3, union :** la formule uniforme est une borne supérieure sous indépendance entre positions. Elle ne fournit pas une borne inférieure imposant la lecture de tout le modèle lors d'un préfill. La conclusion «aucun algorithme exact ne fait mieux» est donc trop forte pour cette étape du raisonnement.
- **T4, balayage :** une borne sous monotonie, saturation et `n_max` faisable ne prouve pas qu'un arbre de brouillon vient gratuitement avec le préfill. Calcul, mémoire, routes, vérification et acceptation doivent être établis.
- **T5, zéro expert résident :** le tri par probabilité concerne un objectif d'octets avec tailles/coûts homogènes. Expansion de format, coût réel de déquantification, temps de lecture et latence des routeurs peuvent changer l'admission. La politique optimale du moteur ne découle pas d'un chiffre de RAM supposé.
- **T9/T10, exactitude :** le rejeu récurrent peut retrouver les mêmes états si les projections et l'ordre des opérations sont identiques. Qualifier son coût de «négligeable» sans mesure est injustifié. L'invariance des kernels est une condition à vérifier, pas une garantie apportée par la mise en batch.

Aucun ancien historique ni moteur PRISME n'a été fusionné. Les valeurs théoriques de performances ne sont pas reprises comme résultats nouveaux.

## Reproduction et contrôles exécutés

```sh
python3 benchmarks/analyze_expert_reuse51.py --self-test
python3 benchmarks/analyze_expert_reuse51.py \
  --olmoe-trace research/exact_io/inherited/routing_traces_olmoe.npz \
  --output research/research51/results/expert-reuse.json
```

Sept auto-tests réussissent : conservation du budget/cache courant, absence d'apprentissage du futur, refus des histogrammes préfill comme decode, comptage d'union, restauration du tri numérique des clés de hash, déduplication des variantes et refus d'une empreinte altérée. Le journal est [expert-reuse-self-test.log](results/expert-reuse-self-test.log). Les empreintes historiques avant sérialisation utilisent des clés d'expert entières : le script les restaure avant hachage pour éviter une fausse divergence entre `2` et `10`.

Un second calcul indépendant avec des ensembles Python concorde pour les unions Qwen de profondeur1/2/4 et pour les huit catégories OLMoE; voir [expert-reuse-validation.json](results/expert-reuse-validation.json). Le seul rapport historique contenant des `step_checks` dans ces sources apporte sept hash de routes vérifiés. Les autres variantes sont comparées par leurs données capturées. Les vecteurs binaires et états complets de l'ancienne campagne ne sont pas revalidés par cette étape.
