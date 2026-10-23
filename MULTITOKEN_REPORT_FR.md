# AirLLM Next — campagne multi-tokens / MoE

**23 octobre 2026 — campagne expérimentale sur Apple M5 / 16 Go.** Les résultats des sept hypothèses, les contrôles indépendants, l'audit des tableaux et les tests sont renseignés.

## Résultats principaux

Le regroupement des positions et des experts réduit fortement les lectures applicatives et le temps d'une **vérification causale dont les entrées sont déjà connues**. Sur Granite entraîné, à 32 positions, l'union est environ 5,47× plus rapide sur CPU et 5,97× sur Metal que le contrôle multi-position sans union ; la combinaison avec le traitement par couche atteint environ 10,01× et 11,08× face au contrôle position par position. Ces nombres ne sont pas des accélérations de génération autoregressive.

En génération réelle arrêtée sur EOS, le gain est beaucoup plus limité et dépend du texte. Le brouillon par recherche dans le préfixe gagne surtout sur une sortie répétitive. Sur six prompts Metal, la profondeur fixe 2 a la meilleure médiane appariée observée, 1,0558×, tandis que la profondeur 8 a le meilleur débit regroupé, 5,517 tok/s contre 4,906 pour la référence de cette série. Une seule répétition et des états d'E/S différents empêchent d'en faire un gain général établi.

Le vrai modèle brouillon SmolLM2-135M, avec conversion de tokenizers et coût complet, ralentit la génération dans les séries CPU et Metal à budget total de cache 256 MiB. L'adaptatif E1 réduit une partie du surcoût mais abandonne trop tôt les brouillons initialement vides. E2 récupère une partie du cas répétitif connu, mais n'améliore pas E1 sur trois des quatre prompts de confirmation : il doit rester facultatif.

Le préchargement des experts déjà déterminés accélère les huit paires de l'ablation G, avec une médiane de 1,1850×. Ajouter SpecPrefetch à ce même préchargement produit une médiane de 0,9801× et davantage d'octets logiques. Ces résultats soutiennent le préchargement simple et le maintien de SpecPrefetch désactivé par défaut dans cette configuration.

Les quatre nouvelles politiques de résidence perdent contre le cache adaptatif existant à 128 et 512 MiB. Même la priorité aux poids partagés, qui réduit de 11,91 % les octets à 512 MiB, ralentit la génération dans cette série. Un cache suffisamment grand pour le petit brouillon améliore quelques cas courts de SmolLM2, mais le cas long reste plus lent : on ne transforme ni le résultat négatif à petit cache ni la victoire ponctuelle à grand cache en loi générale.

La quantification mixte réduit nettement les lectures et augmente le débit de sortie Metal dans sa confirmation, au prix d'un changement de modèle. Sur 512 labels humains pré-déclarés, sa perplexité augmente de 5,25 % ; ce compromis n'est ni exact ni une qualité préservée certifiée. La quantification reste une troisième piste conditionnelle, distincte des optimisations exactes de routage et de stockage.

## 1. Origine, sources et portée

- Point de départ demandé : [PR #1](https://github.com/iliasm-a11y/airllm-next/pull/1), tête vérifiée `000ade1e04b3e460f7b00ac8ed97aadd189dc35f`.
- Branche de recherche : `research/multitoken-moe-2026-10-23`. Implémentation publiée au commit `2c1adc17d60eed32f89857b3a06cbe43ef244f54`, arbre `0802cbd848cc69221f3165a0aa99972ba9c47f48`, identique au candidat local validé `34d665d`. La première vague était publiée à `918bf660a3de551b53129bbadfcad933ed152671`. `main` reste à `945368` et la tête de PR #1 reste à `000ade1e04b3e460f7b00ac8ed97aadd189dc35f`.
- Livraison : [PR brouillon #2](https://github.com/iliasm-a11y/airllm-next/pull/2), dépendante de #1. Le commit de code validé est celui cité ci-dessus ; les preuves et ce rapport sont ajoutés par un commit distinct. Le reçu local de publication donne le SHA final et la correspondance des arbres, sans créer une auto-référence dans ce rapport. Aucune fusion effectuée.
- Les sept rapports initiaux B/C et A/E1 (hors D/G) déclarent `primary_campaign=true`, `source_unchanged=true`, `failures=[]`, et partagent le même manifeste de sources. Digest SHA256 de sa sérialisation JSON canonique : `3ae5a420bdfe97f12e385a626285c5ae1e35d1df0bd3431457e0934adc332ef2`.
- `combination-metal.json` utilise ce même manifeste et passe les mêmes contrôles. La confirmation `adaptive-retry-metal.json` utilise un manifeste distinct `b2d83c19a24feb9c4aa19d498e83acefdc4c37d640934f2e3396a731cfe2a168` ; elle déclare `source_unchanged=true`, `original_files_unchanged=true`, aucune divergence de sortie et aucun échec. Ne pas fusionner les manifestes ou les protocoles.
- Les deux séries de cache D partagent le manifeste initial. Le contrôle SmolLM2 avec cache total 832 MiB possède son propre enregistrement de sources et déclare `source_unchanged=true`, `failures=[]` ; il reste distinct de la série à cache 256 MiB.
- Ce manifeste identifie les fichiers du moteur et le benchmark pendant ces séries. Il ne doit pas être remplacé rétroactivement par le SHA d'une version ultérieure contenant E2.
- Les contrôles logits/routes ci-dessous proviennent des campagnes et des fichiers bruts conservés. L'audit indépendant de 13 NPZ et de 234 comparaisons de blocs passe ; les neuf hashes attendus déclarés correspondent. Les preuves primaires et exploratoires restent identifiées séparément.

Sources de mesures :

| Sujet | Rapport primaire |
|---|---|
| B/C CPU, cache explicite 128 MiB | `evidence-multitoken/batch-cpu-128.json` |
| B/C Metal, cache explicite 256 MiB | `evidence-multitoken/batch-metal-256.json` |
| B/C Metal, politique F_NOCACHE, cache 128 MiB | `evidence-multitoken/batch-metal-physical.json` |
| A/E1, recherche de préfixe, CPU | `evidence-multitoken/generation-prompt-cpu.json` |
| A/E1, recherche de préfixe, Metal | `evidence-multitoken/generation-prompt-metal.json` |
| A/E1, brouillon SmolLM2, CPU | `evidence-multitoken/generation-smollm-cpu.json` |
| A/E1, brouillon SmolLM2, Metal | `evidence-multitoken/generation-smollm-metal.json` |
| A/E1, brouillon SmolLM2 avec cache total 832 MiB | `evidence-multitoken/generation-smollm-resident-metal.json` |
| D, politiques de résidence, cache 128 et 512 MiB | `evidence-multitoken/cache-metal-128.json`, `evidence-multitoken/cache-metal-512.json` |
| G, préchargement simple et SpecPrefetch | `evidence-multitoken/combination-metal.json` |
| E1/E2, confirmation ciblée avec préchargement activé | `evidence-multitoken/adaptive-retry-metal.json` |
| F, conversion des sidecars | `evidence-multitoken/quantized-conversion.json` |
| F, confirmation Metal sans instrumentation des projections | `evidence-multitoken/quantized-confirmation-metal.json` |
| F, contrôle CPU | `evidence-multitoken/quantized-cpu.json` |
| F, qualité sur labels humains | `evidence-multitoken/corpus-quality-metal.json` |
| Contrôle natif contre Transformers | `evidence-multitoken/reference-metal.json` |
| Tests et synthèse des séries compatibles | `evidence-multitoken/tests-final-e2-f.log`, `evidence-multitoken/synthesis-final.json` |

Les fichiers de démarrage `smoke-*` et les anciennes séries déplacées dans `evidence-multitoken/exploratory/` ne fondent pas les gains primaires de ce document. Les anciennes séries à horizon forcé pouvaient générer au-delà d'EOS ; elles sont conservées comme résultats exploratoires et traces du correctif de protocole.

## 2. Matériel, modèles, précisions et budgets

Environnement déclaré dans `evidence-multitoken/environment.json` : Apple M5, 10 cœurs logiques, 17 179 869 184 octets de mémoire physique, macOS 27.2 arm64, Python 3.12.13. Versions de campagne : MLX/MLX Metal 0.32.4, PyTorch 2.14.1, Transformers 5.20.2, NumPy 2.5.3, Safetensors 0.8.0.

Le système utilisait déjà environ 5,14 Go de swap lors du relevé initial. Cela ne mesure pas la croissance imputable à une variante. Les rapports conservent les compteurs de croissance de swap disponibles, les pics RSS échantillonnés et la réserve système ; la machine n’est pas présentée comme vide ou sans swap.

| Rôle | Artefact effectivement utilisé | Révision |
|---|---|---|
| Cible MoE entraînée | `ibm-granite/granite-3.0-1b-a400m-instruct` | `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445` |
| Petit modèle brouillon entraîné | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Brouillon sans modèle | `PromptSpanDraft`, recherche bornée dans les tokens du préfixe disponible | code du manifeste |

Les poids officiels Granite BF16 sont lus et décodés pour un calcul FP32 dans cette campagne. Les essais A/B/C ne sont pas des essais Q4. Le brouillon SmolLM2 utilise son artefact officiel ; les fichiers et leur identité sont consignés dans `draft-provenance.json` et dans chaque rapport de série concerné. Sa compatibilité sémantique avec Granite n'est pas supposée : le texte est réencodé en tokens cible, avec refus des raccords instables. Chaque proposition reste vérifiée par Granite. Cette conversion et le recalcul du préfixe du brouillon sont compris dans le temps mesuré.

L’inventaire local n’a pas révélé de checkpoint MoE plus grand complet et exploitable : le cache Gemma plus volumineux consulté déclare `enable_moe_block=False`, et le cache Qwen 8B ne contient pas de snapshot complet. Aucun de ces caches n’est présenté comme une validation MoE supplémentaire. Les configurations géantes n’alimentent que les scénarios analytiques et microcontrôles.

Granite compte 32 experts par couche et en sélectionne huit par position. Le checkpoint d'environ 2,67 Go tient dans la RAM physique ; seuls les caches explicites sont volontairement contraints. Ces essais ne reproduisent donc pas les lectures physiques d'un checkpoint de plusieurs centaines de gigaoctets.

| Série | Cache explicite | E/S | KV | Espace de travail | Somme des quatre budgets |
|---|---:|---:|---:|---:|---:|
| CPU B/C et Metal F_NOCACHE | 128 MiB | 32 MiB | 32 MiB | 256 MiB | 448 MiB |
| Metal B/C et génération référence | 256 MiB | 32 MiB | 32 MiB | 256 MiB | 576 MiB |
| D, cache étendu | 512 MiB | 32 MiB | 32 MiB | 256 MiB | 832 MiB |
| Génération avec SmolLM2 : cible | 192 MiB | 24 MiB | 24 MiB | 192 MiB | 432 MiB |
| Génération avec SmolLM2 : brouillon | 64 MiB | 8 MiB | 8 MiB | 64 MiB | 144 MiB |
| Contrôle SmolLM2 résident : référence sans brouillon | 832 MiB | 32 MiB | 32 MiB | 256 MiB | 1 152 MiB |
| Contrôle résident : cible | 256 MiB | 24 MiB | 24 MiB | 192 MiB | 496 MiB |
| Contrôle résident : brouillon | 576 MiB | 8 MiB | 8 MiB | 64 MiB | 656 MiB |

La somme cible+brouillon reste égale à la référence sans brouillon : 576 MiB dans la série initiale, 1 152 MiB dans le contrôle résident. Ces budgets réservés ne sont pas la mémoire totale du processus : interpréteur, métadonnées, copies hôte et cache OS sont distincts. Les pics RSS enregistrés sont ceux de l'ensemble de la série, pas ceux d'une variante isolée.

## 3. Méthode et définitions des résultats

Pour B/C, un passage naturel déclaré dans le rapport est injecté en teacher forcing sur 1, 2, 4, 8, 16 et 32 positions. Les positions sont causales mais connues à l'avance. Comparaisons :

1. `tokenwise` : un forward par position ;
2. `serial_positions` : un forward de bloc, experts exécutés séparément par position ;
3. `union` : même forward de bloc, positions regroupées par expert natif.

`union / serial_positions` isole le regroupement des experts B. `union / tokenwise` combine B avec l'amortissement par couche C. Chaque variante conserve les mêmes budgets et la même précision pour une série donnée. Les identifiants d'experts sélectionnés sont comparés comme ensembles par position, les logits avec `atol=1e-4, rtol=5e-4`, et les argmax séparément.

Pour A et E, la cible génère réellement jusqu'à EOS ou au plafond déclaré. La référence autoregressive est répétée selon la grille et mélangée avec les variantes ; les lignes initiales `sequence_seed` servent au contrôle des sorties et sont exclues des gains. Les propositions restent des propositions : le moteur vérifie un bloc causal puis ne conserve que le préfixe accepté, la correction ou le bonus cible. Le cache revient au préfixe retenu, avec son dernier token en attente.

Ces premières séries A/B/C ont `prefetch=False` et `spec_prefetch=False`. Les confirmations avec préchargement activé sont rapportées séparément en sections 9–10 ; leurs mesures ne sont pas amalgamées à celles-ci.

Le débit principal est `nouveaux tokens effectivement émis / temps mural complet de génération`, préremplissage, brouillon, conversions et vérification compris. Les gains sont les médianes de rapports de débit appariés par même cas et répétition. Le débit regroupé utilise `somme(tokens) / somme(temps)` ; il n'est pas interchangeable avec la médiane des rapports. Les octets par token sont `somme(octets) / somme(tokens réels)`. L'acceptation est pondérée par le nombre de propositions, jamais assimilée au débit.

Les temps excluent la construction du modèle. Les coûts de celle-ci, de démarrage et de premier token restent à publier séparément lorsque nécessaires pour l'expérience utilisateur.

## 4. Tableau comparatif des sept hypothèses

| Hypothèse | Implémentation observée | Tests/mesures achevés | Gain réel observé | E/S par unité réelle | Mémoire | Décision |
|---|---|---|---|---|---|---|
| A — MTOCV | Vérification gloutonne causale, correction/bonus, rollback ; profondeurs 2/4/8/16/32 ; recherche de préfixe et vrai brouillon | Générations conformes aux références ; contrôle court Transformers passé ; tests globaux passés | Recherche Metal : médiane 1,0558× à profondeur 2 ; CPU : meilleure profondeur fixe médiane 0,9822×. SmolLM2 négatif à cache total 256 MiB ; résultats mixtes à 832 MiB | Metal recherche fixe 2 : 742,628 Mo logiques/token contre 778,127 ; brouillon inclus | Budgets totaux 576 ou 1 152 MiB, comparaisons à enveloppe égale | Conserver le chemin expérimental ; aucune activation générale justifiée |
| B — Expert Union Scheduling | Regroupement après routeur natif ; routes conservées ; chargement par groupe | 36 comparaisons CPU, 36 Metal et 12 F_NOCACHE consignées, logits proches, ensembles de routes et argmax identiques | À 32 positions : 5,47× CPU et 5,97× Metal face à `serial_positions` | Réduction logique à 32 positions : 82,69 % CPU, 77,92 % Metal | Même budget par variante ; espace de travail borné | Mécanisme favorable pour les blocs connus ; confirmer répétitions et généralisation |
| C — Layerwise Token Batching | Forward causal de plusieurs positions ; réutilisation des poids partagés et union | Mêmes essais B/C, comparaison `union` contre `tokenwise` | À 32 positions : 10,01× CPU et 11,08× Metal pour la vérification contrôlée | 80,270 Mo logiques/position en union ; réductions de 90,11 % CPU et 89,41 % Metal | Activations et KV supplémentaires comptés dans les budgets ; RSS individuel non isolé | Conserver ; ne jamais appeler ce gain une accélération autoregressive de 11× |
| D — Shared Weight Residency | Priorités partagé/expert/hybride/sensible au lot comparées au cache adaptatif existant | Quatre prompts Metal, cache 128 et 512 MiB, une répétition ; sorties conformes | Toutes les variantes perdent : médianes 0,8211–0,9497× à 128 MiB et 0,7372–0,8899× à 512 MiB | Priorité partagé à 512 MiB : 671,396 vs 762,163 Mo/token, mais 0,8899× en débit médian | Même capacité et autres budgets par paire | Conserver le cache adaptatif ; politiques alternatives uniquement expérimentales |
| E — Adaptive Draft Length | E1 ajuste les profondeurs par coût observé ; E2 reprend après vide bon marché, sur option explicite | Confirmation Metal : quatre prompts ciblés, trois répétitions, 48 exécutions, tous tokens et gardes conformes | E2 vs E1 par cas : 1,0235× / 0,9849× / 0,9817× / 0,9889× ; pas de victoire générale | E2 rouge/bleu : 748,490 Mo/token contre E1 781,832 ; pas de gain constant ailleurs | Historiques bornés ; 21 probes et trois réactivations observées | Garder E2 facultatif ; ne pas le promouvoir au défaut |
| F — Quantification et stockage | Experts Q4 ; mixte experts Q4 + projections attention Q8 ; routeur/embedding/normes conservés | Confirmation Metal quatre prompts × deux répétitions, contrôle CPU, audits de hashes, 512 labels humains | Mixte Metal : ratio médian de débit de sortie 1,1104× ; CPU 0,8154×. Contenus modifiés, pas accélération exacte | Mixte Metal : 274,585 vs 888,619 Mo/token ; payload effectif 942,252 vs 2 669,257 Mo | Même budget ; calcul/déquantification fusionnés inclus dans le temps global | Option conditionnelle : PPL +5,25 %, aucune garantie de qualité préservée |
| G — Ordonnancement/préchargement | Préchargement des experts connus et prédictions SpecPrefetch séparés | Quatre prompts × deux répétitions, profondeur fixe 4, mêmes tokens et budgets | Préchargement simple vs désactivé : 1,1850× ; SpecPrefetch vs simple : 0,9801× | 732,865 Mo/token simple contre 715,682 sans ; 734,516 avec SpecPrefetch | Pic file E/S : 6 MiB simple, 10 MiB avec prédictions, budget 32 MiB | Conserver le préchargement simple ; SpecPrefetch reste désactivé par défaut |

## 5. B/C : résultats primaires détaillés

Deux contextes distincts et une répétition pour les séries CPU et Metal ; un contexte et une répétition pour la série F_NOCACHE. Les tâches sont mélangées mais les répétitions sont insuffisantes pour quantifier une distribution de variabilité.

| Positions | CPU union / série d'experts par position | CPU union / tokenwise | Metal union / série d'experts par position | Metal union / tokenwise | Union, Mo logiques/position |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,990× | 1,060× | 1,004× | 1,105× | 857,319 |
| 2 | 1,232× | 1,349× | 1,157× | 1,446× | 604,035 |
| 4 | 1,362× | 1,929× | 1,582× | 2,132× | 445,936 |
| 8 | 1,886× | 2,821× | 2,341× | 3,203× | 276,643 |
| 16 | 3,562× | 5,458× | 2,685× | 4,118× | 153,559 |
| 32 | 5,471× | 10,011× | 5,967× | 11,078× | 80,270 |

Les traces de routage enregistrent les mêmes ensembles d'experts et les mêmes argmax pour toutes les comparaisons consignées. Écart maximal absolu de logits déclaré : `1,1539459e-4` CPU, `6,8664551e-5` Metal et `4,7683716e-5` Metal F_NOCACHE. Le maximum CPU peut dépasser l'atol seul ; c'est le critère combiné atol+rtol pré-déclaré qui passe. Il n'y a pas d'égalité bit à bit revendiquée.

Contrôle Metal F_NOCACHE à 32 positions : 5,868× face à `serial_positions`, 9,988× face à `tokenwise`, 79,975 Mo logiques/position. Le compteur OS de lectures de ce cas relève cependant 122 322 944 octets pour union contre 58 490 880 pour tokenwise : **le résultat ne démontre pas une réduction des lectures physiques**. La mesure murale et la réduction des requêtes applicatives sont réelles dans ce contrôle ; le trafic NAND et le débit d'un périphérique ne sont pas établis.

## 6. D : résidence des poids à budget identique

Les séries `cache-metal-128.json` et `cache-metal-512.json` comparent quatre prompts en génération autoregressive arrêtée sur EOS ou au plafond de 32 tokens. Les sorties réelles comptent 9, 32, 32 et 32 tokens. Chaque série comprend une répétition par politique, avec ordre mélangé et mêmes capacités par paire. La référence pour les ratios est le cache `adaptive` existant du même cas et de la même répétition ; les capacités 128 et 512 MiB ne sont pas appariées entre elles.

| Politique | 128 MiB : médiane vs adaptatif | 128 MiB : Mo logiques/token | 512 MiB : médiane vs adaptatif | 512 MiB : Mo logiques/token |
|---|---:|---:|---:|---:|
| Adaptatif existant | 1× | 866,983 | 1× | 762,163 |
| Priorité aux poids partagés | 0,8354× | 912,395 | 0,8899× | **671,396** |
| Priorité aux experts | 0,9497× | 910,519 | 0,7902× | 849,202 |
| Hybride | 0,8381× | 868,110 | 0,7372× | 763,406 |
| Sensible au lot | 0,8211× | 900,674 | 0,8245× | 763,530 |

Toutes les politiques alternatives sont plus lentes que la référence sur chacun des quatre cas, aux deux capacités. Les pics de cache consignés restent dans leur capacité : les pertes ne sont pas masquées par un budget supplémentaire. La priorité aux poids partagés à 512 MiB réduit le trafic logique de 11,91 %, mais son débit regroupé atteint 4,189 contre 4,715 tok/s pour l'adaptatif ; sa médiane de rapports est 0,8899×. C'est un résultat négatif explicite pour l'objectif prioritaire de tokens corrects par seconde, malgré l'amélioration de volume.

Le contrôle n'établit pas que la résidence partagée est inutile pour les géants ou en lecture physique froide. Il concerne ces politiques implémentées, les dimensions Granite et ce système où le checkpoint tient dans le cache OS. La variante sensible au lot est ici évaluée en génération avec préremplissage par blocs et décodage unitaire ; une interaction avec de longues vérifications spéculatives reste à mesurer. L'écart de débit entre les deux séries de capacité ne doit pas être attribué au cache seul, leurs états système n'étant pas appariés.

Décision : garder le cache adaptatif existant. Préserver les autres politiques comme essais de recherche et résultats négatifs, sans les promouvoir au chemin courant. Toute nouvelle complexité ou modification de priorité doit gagner le test de débit complet, en plus de la réduction d'octets.

## 7. A : génération réelle avec recherche de préfixe

CPU : quatre prompts, plafond 48 tokens ; Metal : six prompts, plafond 64 tokens ; une répétition par cas. Les nombres de tokens Metal réellement émis sont respectivement 9, 54, 53, 64, 63 et 64. L'arrêt sur EOS explique les sorties plus courtes ; les ratios ne divisent pas par le plafond demandé.

| Profondeur | CPU, médiane vs AR | Metal, médiane vs AR | Metal, débit regroupé | Metal, Mo logiques/token |
|---|---:|---:|---:|---:|
| Référence AR | 1× | 1× | 4,906 tok/s | 778,127 |
| 2 | 0,9822× | **1,0558×** | 5,388 tok/s | 742,628 |
| 4 | 0,9685× | 1,0241× | 5,441 tok/s | 734,787 |
| 8 | 0,9497× | 1,0224× | **5,517 tok/s** | **724,793** |
| 16 | 0,9052× | 1,0088× | 5,452 tok/s | 729,399 |
| 32 | 0,9564× | 1,0116× | 5,461 tok/s | 729,547 |
| E1 adaptatif | 1,0024× | 1,0103× | 5,126 tok/s | 771,298 |

La profondeur gagnante dépend de la métrique : 2 sur la médiane appariée, 8 sur le débit regroupé et les octets par token. Il serait incorrect de sélectionner l'une après coup puis la présenter comme gagnante universelle.

Cas répétitif Metal, prompt `Repeat exactly this pattern several times: red blue green red blue green` : profondeur 8 = 1,7479×, 456,518 Mo/token contre 772,092, soit environ 40,9 % de lectures logiques en moins. Les profondeurs 16/32 n'améliorent pas de manière convaincante ce résultat. Le cas court factuel, la suite numérique et l'explication en français ralentissent aux profondeurs ≥4.

### Résultat négatif E1

E1 s'arrête sur `empty_draft_proposals` dans cinq cas Metal sur six. Sur le cas répétitif, quatre tentatives vides initiales le désactivent pour le reste de la sortie : zéro proposition acceptée, alors que fixe 8 en accepte 37. Son taux global d'acceptation de 64,3 % porte sur seulement 14 propositions et neuf acceptations ; ce taux élevé ne signifie pas un bon débit ni une bonne couverture.

E2 étudie une correction de ce cas précis : après un arrêt exclusivement causé par des propositions vides, réessayer avec un intervalle borné, uniquement si le coût de préparation passé est faible et les compteurs de transport connus nuls. Les propositions produites sont toujours soumises au vérificateur ordinaire. La confirmation avec interrupteur explicite E1/E2 figure en section 10 ; elle ne justifie pas une activation par défaut.

### Variabilité et E/S observées

Dans la série Metal, la référence du cas code relève 1 738 104 832 octets dans le compteur de stockage du processus, contre zéro pour fixe 2, fixe 8 et E1 de ce même cas. C'est un facteur de confusion plausible ; un compteur process-wide ne permet pas d'attribuer précisément ce trafic aux poids. La série n'établit donc pas un avantage thermique ou SSD général. On conserve ce cas et sa limite ; on ne l'efface pas pour améliorer le bilan.

## 8. A : vrai brouillon SmolLM2, petit cache et contrôle résident

Les limites de prompt et de génération diffèrent entre les séries CPU/Metal : il faut comparer chaque variante à sa propre référence, et non les débits CPU/Metal entre eux. Deux prompts CPU, quatre Metal ; une répétition par cas.

| Profondeur | CPU, médiane vs AR | CPU, Mo totaux/token | Metal, médiane vs AR | Metal, Mo totaux/token |
|---|---:|---:|---:|---:|
| Référence AR | 1× | 805,716 | 1× | 779,398 |
| 2 | 0,6410× | 1 076,591 | 0,7798× | 903,231 |
| 4 | 0,6177× | 1 294,755 | 0,8059× | 926,457 |
| 8 | 0,6192× | 1 546,267 | 0,7720× | 1 037,140 |
| 16 | 0,4453× | 1 983,121 | 0,6212× | 1 311,533 |
| 32 | 0,2907× | 2 765,536 | 0,4189× | 1 833,836 |
| E1 adaptatif | 0,8105× | 867,233 | 0,9188× | 828,163 |

Les octets de cette table incluent la cible et le brouillon ; le temps inclut son recalcul et le réencodage. À profondeur 2 sur Metal, 99 propositions sont acceptées sur 139, mais la médiane reste 0,7798× : l'acceptation seule ne suffit pas à payer les coûts. E1 désactive le chemin devenu trop cher dans plusieurs cas et limite les pertes, sans gagner globalement.

Sur les quatre cas Metal, les 20 exécutions à profondeur fixe et les quatre exécutions E1 sont toutes plus lentes que leur référence appariée. À profondeur 2, la préparation du brouillon représente 33,0 à 47,5 % du temps mural ; à profondeur 32, 73,4 à 85,3 %. Ce coût inclut le recalcul conservateur du préfixe et les conversions de tokenizers. La profondeur 4 a la meilleure médiane fixe, 0,8059×, mais la profondeur 2 a un meilleur débit regroupé, 3,490 contre 3,266 tok/s ; les deux restent sous les 5,046 tok/s de la référence de cette série.

E1 réduit le coût avec un débit regroupé de 4,680 tok/s et une médiane de 0,9188×. Ses quatre rapports par cas sont 0,7731×, 0,9133×, 0,9852× et 0,9243×. Le résultat négatif ne dépend donc pas d'une seule profondeur mal choisie. Il reste limité à ce couple de modèles, ce mode de brouillon qui recalcule le préfixe et cette répartition 192+64 MiB du cache ; une expérience avec brouillon résident ou incrémental doit être rapportée séparément.

### Contrôle avec brouillon résident et cache total de 832 MiB

`generation-smollm-resident-metal.json` porte sur deux prompts, un plafond de 48 tokens et une répétition. La référence sans brouillon reçoit 832 MiB de cache ; la variante reçoit 256 MiB pour Granite et 576 MiB pour SmolLM2. Les partitions E/S/KV/espace de travail restent égales en somme : 1 152 MiB de budgets dans les deux branches. Les sorties comptent neuf et 48 tokens et restent identiques à la référence. L'augmentation de mémoire n'est donc pas offerte gratuitement à la variante spéculative.

| Variante | Cas court, rapport vs AR | Cas long, rapport vs AR | Médiane des deux rapports | Débit regroupé | Mo totaux/token |
|---|---:|---:|---:|---:|---:|
| AR, cache 832 MiB | 1× | 1× | 1× | **4,930 tok/s** | **642,313** |
| Fixe 2, brouillon résident | 0,9401× | 0,8397× | 0,8899× | 4,225 tok/s | 767,494 |
| Fixe 4, brouillon résident | **1,0639×** | 0,7269× | 0,8954× | 3,812 tok/s | 807,714 |
| Fixe 8, brouillon résident | **1,3539×** | 0,6691× | **1,0115×** | 3,647 tok/s | 816,557 |
| E1, brouillon résident | 0,9532× | 0,9450× | 0,9491× | 4,666 tok/s | 809,821 |

Le cache étendu change le résultat de certains cas : le court gagne à profondeur 4 et 8, ce qui interdit une conclusion universelle « SmolLM2 ne peut jamais accélérer Granite ». Mais la médiane de deux rapports 1,0115× à profondeur 8 masque un fort ralentissement sur les 48 tokens du cas long ; le débit regroupé reste 3,647 contre 4,930 tok/s et les octets sont plus élevés. Une moyenne de deux rapports ne doit pas devenir une annonce de gain général.

Le trafic total du brouillon enregistré dans les exemples de cette série est voisin d'une lecture de son checkpoint, avec nombreux hits de tuiles de sortie : la résidence a réduit les relectures, pas le coût du recalcul du préfixe ni celui des tokens que le vérificateur rejette. Deux prompts et une répétition ne certifient pas le gain court ni un choix optimal de capacité.

Décision : conserver les résultats des deux capacités et le chemin de vrai brouillon comme facultatif. Les données à 256 MiB sont négatives ; le contrôle résident apporte des victoires ponctuelles et une sensibilité à la capacité, sans accélération générale. Un brouillon mieux aligné, un cache incrémental correctement géré ou un autre couple de modèles nécessitent leurs propres mesures.

## 9. G : préchargement simple et SpecPrefetch séparés

`combination-metal.json` comprend quatre prompts, deux répétitions et des tâches mélangées. L'ablation G compare les variantes à profondeur fixe 4, sur la même sortie cible et à budgets identiques. Les huit paires sont appariées par cas et répétition. Le brouillon est la recherche de préfixe, dont les compteurs de transport sont nuls connus ; les octets totaux ci-dessous correspondent donc au trafic applicatif de la cible.

| Contrôle à profondeur fixe 4 | Médiane appariée vs contrôle précédent | Débit regroupé | Mo logiques/token émis | Pic de file E/S |
|---|---:|---:|---:|---:|
| Sans préchargement | référence | 5,524 tok/s | 715,682 | budget commun 32 MiB |
| Préchargement des experts déjà déterminés | **1,1850×** vs sans préchargement | **6,545 tok/s** | 732,865 | 6 MiB |
| Préchargement simple + SpecPrefetch | **0,9801×** vs préchargement simple | 6,417 tok/s | 734,516 | 10 MiB |

Le préchargement simple gagne dans huit paires sur huit ; ses médianes par cas sont 1,2250×, 1,1752×, 1,2002× et 1,1693×. Il réduit le temps malgré une hausse des octets logiques regroupés d'environ 2,40 %. Ce résultat soutient le masquage des lectures et l'ordonnancement des accès ; il ne démontre pas une réduction de volume.

SpecPrefetch perd dans sept paires sur huit et toutes les médianes par cas sont inférieures à un : 0,9850×, 0,9801×, 0,9802× et 0,9798×. Ses octets logiques regroupés augmentent encore d'environ 0,225 % par rapport au préchargement simple. Son pic E/S de 10 MiB reste dans l'enveloppe commune de 32 MiB ; il n'y a pas de dépassement de budget servant à expliquer ou masquer ce résultat.

L'ordonnanceur enregistre 159 301 demandes soumises, 158 942 consommées, 359 jetées et 921 dédupliquées avec SpecPrefetch, contre 159 150 soumises/consommées et aucun rejet/dédoublonnage pour le contrôle simple. Ces compteurs de file ne sont pas des nombres d'experts distincts ni un taux de prédiction correcte. Le détail du snapshot SpecPrefetch n'est pas exploitable dans ce rapport : la clé finale `spec_prefetch` porte l'indicateur de configuration et a remplacé le snapshot homonyme lors de l'enregistrement. On ne reconstitue donc pas artificiellement une précision du prédicteur ou des octets de « fausses prédictions ».

Décision : conserver le préchargement simple comme référence forte pour les confirmations ; garder SpecPrefetch désactivé par défaut. Deux répétitions sur quatre prompts ne certifient pas son comportement sur un SSD physique saturé, d'autres familles MoE ou des contextes longs. Une nouvelle tentative demanderait un mécanisme précis, des métriques de prédiction préservées sous une clé distincte et une ablation favorable.

## 10. E2 : confirmation ciblée contre une référence déjà préchargée

`adaptive-retry-metal.json` compare AR, fixe 8, E1 et E2 sur quatre prompts et trois répétitions par variante, soit 48 exécutions. La référence et toutes les variantes ont le préchargement simple activé, SpecPrefetch désactivé, le même budget total de 576 MiB et l'arrêt EOS de la cible. Les sorties sont identiques pour les 48 exécutions et toutes les gardes consignées sont vraies. Les sources originales sont déclarées inchangées.

La sélection est délibérément ciblée : un contrôle positif rouge/bleu déjà connu, un nouveau cycle cuivre/sable/olive, une nouvelle tâche de copie et une explication générale conçue comme contrôle négatif. Elle ne représente pas une distribution générale des demandes. L'explication générale produit elle aussi des tokens réutilisables ; son rôle prévu de contrôle négatif ne doit pas remplacer son résultat observé.

| Cas | Tokens émis par exécution | Fixe 8 / AR, médiane | E1 / AR, médiane | E2 / AR, médiane | E2 / E1, médiane directe |
|---|---:|---:|---:|---:|---:|
| Rouge/bleu, positif connu | 64 | 1,7179× | 1,0039× | 1,0300× | **1,0235×** |
| Cuivre/sable/olive, nouveau | 23, EOS | 0,8182× | 1,0111× | 0,9958× | 0,9849× |
| Copie, nouveau | 12, EOS | 1,8261× | 1,4340× | 1,4252× | 0,9817× |
| Explication générale, nouveau | 64 | 1,1802× | 1,1477× | 1,1351× | 0,9889× |

Les rapports E2/E1 sont calculés directement sur chaque même répétition, puis médiannés ; ils ne sont pas le quotient des deux colonnes de médianes. Les lignes à EOS n'émettent aucun token après leur premier EOS. La conformité aux sorties du modèle de référence ne prouve pas qu'il a respecté l'instruction de quatre copies : la tâche de copie s'arrête ici après 12 tokens. Aucune nouvelle validation de capacité générale ne découle de ce tableau.

La règle E2 a été pré-déclarée : attendre huit tokens émis après une suspension pour propositions vides ; coûts de préparation récents inférieurs à 1 % d'un pas de référence ; compteurs logiques et encodés connus nuls. Les arrêts pour mauvaise acceptation ou coût mesuré ne sont pas réouverts par ce mécanisme. L'interrupteur permet E1/E2 dans la même source expérimentale stable.

E2 effectue 21 probes au total : trois sur le contrôle rouge/bleu, six sur cuivre et douze sur l'explication ; aucun sur la copie. Il se réactive trois fois, uniquement sur rouge/bleu, et y accepte 21 tokens au total contre zéro pour E1. La préparation des probes totalise environ 0,376 ms, mais le coût complet de la génération et des nouvelles vérifications reste inclus. Les trois exécutions rouge/bleu finissent ensuite arrêtées par le garde de coût : le bas coût de recherche n'est pas une garantie de rentabilité du bloc vérifié.

Sur rouge/bleu, E2 réduit les octets logiques de 781,832 à 748,490 Mo/token face à E1. Le nouveau cycle reste un résultat négatif : aucune proposition acceptée et une profondeur fixe 8 ralentissant d'environ 18 %. E2 ne gagne contre E1 que sur le cas connu ; ses médianes directes sont inférieures à un dans les trois nouveaux cas. Le débit regroupé de ce petit ensemble peut varier selon son mélange de longueurs ; il n'est pas utilisé pour revendiquer un gain général.

Décision : E2 corrige un défaut de couverture d'E1, mais sa valeur globale n'est pas démontrée. Le conserver comme option expérimentale avec ses preuves et résultats négatifs, sans remplacer E1 ou l'AR par défaut. Des essais plus larges ou un autre critère de reprise doivent être pré-déclarés plutôt que réglés sur ces nouveaux prompts déjà mesurés.

## 11. F : quantification, stockage et qualité mesurée

Deux sidecars affines à groupes de 64 sont construits sans écraser les poids officiels : `expert4` quantifie les 1 536 matrices d'experts routés en codes Q4 ; `mixed` ajoute 96 matrices d'attention/linéaires Q8. Le routeur, l'embedding et les normalisations restent originaux. Les paquets incluent échelles, biais et index : « Q4 » ne signifie pas exactement 0,5 octet effectif par poids. Les hashes de chaque entrée et des manifests sont vérifiés ; les originaux et les tokenizers sont déclarés inchangés après les mesures.

| Représentation | Matrices quantifiées | Payload effectif du modèle | Fichier sidecar supplémentaire |
|---|---:|---:|---:|
| Original | 0 | 2 669,257 Mo | 0 |
| Experts Q4 | 1 536 | 1 008,312 Mo | 755,325 Mo |
| Mixte Q4/Q8 | 1 632 | 942,252 Mo | 840,281 Mo |

Le payload effectif exprime les poids quantifiés et les poids originaux encore nécessaires à l'exécution. Les originaux sont préservés sur disque : le sidecar s'ajoute aux fichiers source ; ces nombres ne sont pas une économie totale de disque après suppression du checkpoint. Les erreurs de poids et de logits ne sont pas nulles.

### Débit complet, hors instrumentation des projections

La première série `quantized-metal.json` contient des générations principales non instrumentées et des passes supplémentaires de profilage des projections, séparées de leur chronométrage. Elle est conservée comme diagnostic. Les chiffres principaux proviennent de `quantized-confirmation-metal.json`, sans cette instrumentation, avec quatre prompts, plafond de 16 tokens, deux répétitions et gardes de budget satisfaites. Le contrôle CPU porte sur deux prompts et une répétition. Préremplissage, lecture des paquets et calcul quantifié sont inclus ; construction, conversion et teacher forcing sont exclus du débit de génération.

| Variante | Metal : ratio médian du débit de sortie vs original | Metal : débit regroupé | Metal : Mo logiques/token | CPU : ratio médian du débit de sortie | CPU : Mo logiques/token |
|---|---:|---:|---:|---:|---:|
| Original | 1× | 4,228 tok/s | 888,619 | 1× | 907,572 |
| Experts Q4 | 0,9815× | 4,728 tok/s | 382,232 | 0,8256× | 390,990 |
| Mixte Q4/Q8 | **1,1104×** | **4,983 tok/s** | **274,585** | 0,8154× | 278,800 |

La réduction logique regroupée vaut environ 57,0 % pour experts Q4 et 69,1 % pour mixte sur Metal. La médiane d'experts Q4 est légèrement inférieure à un malgré un débit regroupé supérieur : les deux agrégats ne portent pas les mêmes pondérations. Le contrôle CPU est négatif pour les deux variantes. La déquantification affine/échelles est intégrée au calcul quantifié et à son temps mural ; le compteur `dequant_seconds` du lecteur suit le décodage FP8 explicite et ne doit pas être interprété comme un coût affine nul.

Les variantes quantifiées ne produisent la même séquence que l'original que pour deux des huit exécutions Metal, et une des deux CPU. Ce tableau décrit des débits de sortie d'un **modèle modifié**, pas une accélération produisant les mêmes réponses. Les générations effectivement émises servent aux dénominateurs ; aucune économie de temps n'est obtenue en divisant par un plafond de tokens non émis.

### Corpus humain tenu à l'écart de cette expérience

`corpus-quality-metal.json` évalue 512 labels de prochain token humain : quatre fenêtres de 128 labels, précédées de 32 tokens de contexte, choisies avant l'inférence. La source est WikiText-2 test dans sa version prétraitée par mots distribuée par le dépôt officiel PyTorch examples, révision `d5678bc8ac0cdd79dbd5e44d4130271018bcec4e`. Il ne s'agit pas du format WikiText raw ni de la perplexité standard sur le test complet. Les fenêtres sont indépendantes et l'éventuel recouvrement avec le préentraînement Granite est inconnu.

| Variante | NLL moyenne | Perplexité des 512 labels | Écart de perplexité | Accord top-1 avec l'original | KL moyenne original→variante |
|---|---:|---:|---:|---:|---:|
| Original | 2,59409 | 13,38435 | référence | 100 % | 0 |
| Experts Q4 | 2,64953 | 14,14743 | **+5,701 %** | 87,109 % | 0,06989 |
| Mixte Q4/Q8 | 2,64526 | 14,08709 | **+5,250 %** | 86,719 % | 0,06974 |

L'accuracy du prochain token humain passe de 265/512 pour l'original à 267/512 pour experts Q4 et 268/512 pour mixte. Deux ou trois labels de plus ne justifient pas une revendication d'amélioration de qualité, alors que NLL et perplexité se dégradent et que l'accord avec l'original baisse. L'échantillon mesure une dégradation limitée à ce protocole ; il ne certifie ni la qualité générale, ni les tâches spécialisées, ni une fidélité acceptable pour tous les utilisateurs.

Les audits finaux des sidecars déclarent les mêmes manifests et entrées qu'avant l'évaluation ; sources, tokenizers et originaux restent inchangés. Décision : conserver le mode mixte comme piste facultative de réduction du trafic et d’amélioration du débit Metal **sous acceptation explicite du compromis de qualité**, sans en faire un mode exact ou un défaut. L'expert Q4 seul n'est pas un vainqueur de débit médian ; le CPU conserve ses résultats négatifs. Une compression sans perte ou une quantification mieux calibrée demanderait une nouvelle expérience.

## 12. Référence indépendante et validation du logiciel

`reference-metal.json` compare, sur deux prompts et un horizon court de huit tokens, Transformers officiel en FP32 et trois exécutions natives : préchargement désactivé, préchargement simple et SpecPrefetch. Les tokens, ensembles d'experts top-k et critères de logits passent dans les trois variantes natives. L'écart maximal absolu est `1,41143798828125e-4` ; il satisfait le critère pré-déclaré combiné `atol=1e-4, rtol=5e-4`, sans modification après observation. Les trois variantes natives donnent des logits bit à bit identiques entre elles dans ce contrôle.

Cette comparaison indépendante renforce l'exactitude courte du backend Granite et l'absence de modification du routage par les préchargements testés. Elle ne valide pas tous les contextes, toutes les sorties ou un modèle géant. Les chargements et représentations mémoire de Transformers et du moteur natif diffèrent : les temps de ces deux implémentations ne sont pas utilisés ici comme mesure isolée d'accélération de kernel.

Validation finale consignée dans `tests-final-e2-f.log` et `tests-final-e2-f.xml` : **916 tests réussis, un ignoré, 11 sous-tests ; 15,93 s**. La compilation Python se termine avec code 0. La synthèse finale accepte 11 rapports, n'en exclut aucun et construit 88 groupes ; les preuves E2, quantification et corpus utilisent leurs propres schémas et analyses complémentaires. Les mesures ne deviennent pas des essais de modèles géants par l'effet de ces tests.

`array-audit.json` recalcule les hashes des 13 NPZ locaux, confirme les neuf hashes attendus déclarés et recontrôle 234 comparaisons de blocs, primaires et exploratoires : logits sous tolérance, ensembles de routes et argmax passent sans erreur. Le replay de stockage comporte 588 lignes : 12 unions et 576 replays aux budgets 128/256/512 MiB. Ce sont des contrôles de traces et d'ordonnancement, pas de nouvelles inférences géantes. La construction de la wheel et son contrôle d'import passent. Les résultats négatifs, hypothèses abandonnées et limites de mesure restent conservés.

## 13. Ce qui est démontré, ce qui ne l'est pas

**Observé dans les rapports primaires :** exécution de Granite entraîné sur MLX CPU et Metal ; routes/argmax conservés dans les blocs contrôlés ; sorties originales générées identiques aux références des mêmes séries ; réduction des lectures applicatives par unité de vérification ; cas de gain et de ralentissement en génération ; surcoût et sensibilité au cache d'un modèle brouillon ; préchargement simple favorable ; SpecPrefetch et politiques de cache alternatives négatifs ; débit et dégradation de qualité des deux cibles quantifiées.

**Non démontré :** modèles de 300–700B ; GLM/Qwen géants de bout en bout ; Fanxiang externe ; débit NAND ou SSD certifié ; maintien d'un débit interactif quand le checkpoint ne tient plus dans le cache OS ; échantillonnage spéculatif correct implémenté et testé ; MTP ; arbres spéculatifs ; qualité générale préservée après quantification ; tenue thermique prolongée ; supériorité générale des politiques D/F ou d'E2 ; bénéfice de SpecPrefetch sur cette configuration.

Les compteurs `ri_diskio_bytesread` mesurent des lectures de stockage imputées au processus, incluant réadahead et d'autres lectures. Un zéro ne prouve pas l'absence de toute lecture sur le périphérique. F_NOCACHE est une politique de descripteur, pas une purge privilégiée du cache ni une certification de stockage froid. Le hachage préalable du checkpoint réchauffe des pages, mais ne prouve pas leur résidence pendant chaque variante.

## 14. Théorie, antériorités et cible géante

Le dossier `MULTITOKEN_THEORY_FR.md` rassemble les sources primaires et les équations de coût. Les mécanismes spéculation/offload, cache d'experts, organisation contiguë et longueur adaptative ont des antériorités directes ; cette campagne revendique une adaptation et une évaluation dans AirLLM Next, pas l'invention de ces principes. Référence particulièrement proche : [SpecMoEOff](https://arxiv.org/html/2508.21706v1).

Le modèle de débit utilise les tokens réellement conservés divisés par le coût complet du brouillon et du vérificateur. L'union réelle des experts, leur résidence et la précision déterminent les octets du bloc. Granite (`32` experts, top-`8`) sature son union beaucoup plus tôt qu'un grand MoE à `160` ou `256` experts ; son gain ne se multiplie pas par la taille totale d'un modèle géant.

À titre de scénario, un débit physique **supposé** de 3 GB/s exige au plus 1,5 / 0,75 / 0,375 / 0,2 GB transféré par token pour viser 2 / 4 / 8 / 15 tok/s, avant les autres coûts. Ce ne sont pas des caractéristiques du Fanxiang. Dans le scénario Q4 idéal sans cache d'experts, routes uniformes indépendantes, poids partagés résidents et acceptation parfaite, les blocs de 32 positions restent plafonnés par les E/S à environ 0,709 tok/s pour GLM-4.5 et 0,460 pour DeepSeek-V3. Une forte localité réelle ou un cache adapté pourrait changer ces bornes conditionnelles ; aucun modèle géant n'est exécuté pour les confirmer.

## 15. Classement et livraison

1. **Union d'experts native + traitement causal par blocs, B/C combinés.** C'est le mécanisme exact le mieux soutenu pour réduire les lectures par position vérifiée. Son intégration au décodage dépend de la présence de propositions assez souvent acceptées ; les gains contrôlés ne s'appliquent pas automatiquement à chaque token généré.
2. **Préchargement des experts déjà déterminés.** Il améliore les huit paires de son ablation et constitue la référence forte pour les autres méthodes. SpecPrefetch n'apporte pas de bénéfice démontré supplémentaire.
3. **Quantification mixte, conditionnelle.** Elle réduit le trafic et accélère le débit de sortie Metal de cette confirmation, avec une dégradation mesurée de perplexité de 5,25 % sur les 512 labels. Elle est une piste de recherche prometteuse si ce compromis est acceptable, et reste entièrement facultative. Si l'exactitude vis-à-vis des poids originaux est impérative, la troisième piste est le brouillon léger par recherche de préfixe, également facultatif et dépendant du texte ; aucun troisième gain universel exact n'est démontré.

Le cache adaptatif existant reste la référence ; les nouvelles politiques de résidence, SpecPrefetch, E2 et le vrai brouillon restent des options de recherche lorsqu'ils n'ont pas produit de bénéfice général. Aucun de ces principes n'est revendiqué comme une invention ; la contribution est leur implémentation, leur combinaison contrôlée et leur confrontation aux mesures M5.

La livraison publie les preuves JSON sous forme compressée, avec un script permettant de restaurer les fichiers et de vérifier leur identité. Les tableaux NPZ de logits, volumineux, sont conservés localement et identifiés par leurs hashes ; ils ne sont pas présentés comme joints au dépôt. Les poids originaux, sidecars de poids et octets du corpus restent hors publication ; les rapports fournissent leurs identités et leur provenance.

Publication : [PR brouillon #2](https://github.com/iliasm-a11y/airllm-next/pull/2). Le manifeste `evidence-multitoken/MANIFEST.json` identifie 68 rapports JSON compressés et les tableaux locaux. La restauration vérifiée se fait avec `python3 benchmarks/restore_multitoken_evidence.py`. La branche dépend de PR #1 ; `main` et cette PR restent inchangés, sans fusion automatique. Le passage des résultats Granite à une inférence géante interactive demeure un problème ouvert.
