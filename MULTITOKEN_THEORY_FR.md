# AirLLM Next — antériorités et modèle quantitatif

Recherche Web vérifiée le 23 octobre 2026. Cette note fournit un cadre de décision et des calculs déterministes, **aucune mesure du Fanxiang, aucun benchmark de modèle géant**. Les paramètres de débit SSD ci-dessous sont des scénarios, pas les caractéristiques constatées d'un appareil. Les résultats d'autres équipes ne sont pas des performances d'AirLLM Next.

## 1. Antériorités : la contribution possible est expérimentale et d'intégration

| Sujet | Source primaire | Ce qu'elle établit ; conséquence pour la campagne |
|---|---|---|
| Décodage spéculatif exact | [Leviathan, Kalman, Matias, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | Petit brouillon, vérification parallèle et règle de rejet préservant la distribution cible existent déjà. A ne constitue pas une invention de ce principe. |
| Offloading et ordonnanceur de tenseurs | [FlexGen, ICML 2023](https://proceedings.mlr.press/v202/sheng23a.html) | Placement GPU/CPU/disque, réordonnancement et compression sont antérieurs. Le débit publié à grand lot ne signifie pas latence interactive d'une conversation. C et D adaptent des principes connus. |
| Lectures flash contiguës et réutilisation | [LLM in a Flash, Apple / ACL 2024](https://machinelearning.apple.com/research/efficient-large-language) | Réutilisation d'activations récentes et regroupement physique des données pour limiter les transferts. Pertinent pour F et G, mais son cadre de parcimonie neuronale diffère du routage MoE exact. |
| Cache et préchargement d'experts | [MoE-Infinity, arXiv 2401.14361](https://arxiv.org/abs/2401.14361) | Traces d'activation pour cache et préchargement. G prolonge un domaine déjà étudié ; conserver le routage natif reste indispensable. |
| Ordonnancement par charge des experts | [Fiddler, article ICLR 2025](https://openreview.net/pdf?id=N5fVv6PZGz) | Coûts dépendant du nombre de positions par expert et choix CPU/GPU. Les transferts PCIe et la DRAM hôte de ces machines diffèrent d'un Mac à mémoire unifiée avec SSD. |
| Spéculation spécifiquement pour MoE offload | [SpecMoEOff, arXiv 2508.21706v1, 29 août 2025](https://arxiv.org/html/2508.21706v1) | Antériorité directe A+C+E : accroître le travail par expert via spéculation, attention par blocs, gestion mémoire du brouillon et réglage des hyperparamètres. Expériences Mixtral sur GPU NVIDIA et 190/250 Go de DRAM hôte ; elles ne valident pas SSD+16 Go. |
| Longueur de brouillon adaptative | [DISCO, arXiv 2405.04304v5](https://arxiv.org/abs/2405.04304) et [implémentation Intel/Hugging Face](https://huggingface.co/blog/dynamic_speculation_lookahead) | Le choix dynamique de longueur et l'arrêt selon confiance sont antérieurs. E doit être jugée sur sa prise en compte des vrais octets, du cache et du coût complet, pas sur son simple caractère adaptatif. |
| Quantification | [GPTQ, ICLR 2023](https://arxiv.org/abs/2210.17323), [AWQ, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) | Quantification post-entraînement 3/4 bits et protection des poids importants sont établies. Leur qualité doit être revalidée pour le modèle et le format effectivement produits. |
| Union des experts et coût de la vérification | [AcceptMoE, arXiv 2608.02989v1, 4 août 2026](https://arxiv.org/html/2608.02989v1) | Distingue nombre de positions, union d'experts et trafic dépendant de la résidence. Sa restriction d'experts modifie la distribution : mécanisme approximatif, incompatible avec B exacte si copié tel quel. |
| Spéculation et réutilisation sur appareil contraint | [S²-MoE, arXiv 2608.15018v2, 19 août 2026](https://arxiv.org/html/2608.15018v2) | Expansion sensible au coût et réutilisation sur appareils contraints existent déjà. Son biais ajouté aux logits du routeur change les sélections d'experts : ce volet ne préserve pas le routeur cible original. On peut étudier ses coûts sans reprendre cette approximation. |

L'apport défendable est une adaptation **mesurée et reproductible** au moteur AirLLM Next, au chargement SSD et à Metal sous 16 Go, avec conservation du routage. Aucune recherche ciblée ne prouve une absence exhaustive d'antériorité. Il faut parler de combinaison, d'adaptation et de résultat empirique ; aucune revendication d'invention n'est justifiée ici.

## 2. Exactitude de A, B et C

Pour un préfixe validé `x`, le brouillon produit `y1...yγ`. Le vérificateur calcule causalement chaque distribution cible `p_i(. | x,y_<i)`. En mode glouton, accepter le plus long préfixe tel que `y_i = argmax p_i`, puis émettre la correction cible à la première divergence ; si tout est accepté, émettre éventuellement le token bonus cible. Le KV cache est ramené au préfixe effectivement retenu. Un brouillon dont les tokens ne sont pas compatibles avec le vocabulaire cible ne peut pas être traité par simple égalité des identifiants.

En échantillonnage, la simple comparaison de tokens ne suffit pas. Avec distribution brouillon `q_i`, accepter `y_i` avec probabilité `min(1,p_i(y_i)/q_i(y_i))` ; au rejet, échantillonner dans la distribution normalisée `max(p_i-q_i,0)`. Les distributions doivent déjà inclure les mêmes règles de température et de troncature effectivement utilisées par le générateur. Cette correction et le traitement du bonus sont le mécanisme de conservation de distribution de [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html).

B peut réordonner uniquement les calculs **après** le routage natif de la couche. Pour chaque expert, regrouper les positions déjà sélectionnées, effectuer les projections, puis accumuler avec les poids de routage natifs et reconstruire l'ordre. Il ne faut ni imposer un expert déjà en cache ni remplacer une route pour réduire l'union.

C est possible pour une séquence proposée connue : toutes les positions d'une couche sont calculables ensemble avec le masque causal ; on ne peut pas inventer les futurs tokens autoregressifs sans proposition. Le KV historique est visible pour chaque position, les positions futures ne le sont pas. Les calculs sur une suffixe rejetée restent des coûts réels. Les derniers états récurrents, si une architecture en possède, exigent eux aussi un rollback.

Le regroupement GEMV/GEMM peut changer les arrondis. Les tests doivent séparer équivalence mathématique du masque/routage, écarts numériques des logits et identité effective des tokens gloutons. Une tolérance de logits seule ne garantit pas le même argmax lorsque deux logits sont presque égaux. Quantifier change le modèle : « exact » signifie alors exact par rapport à **la même cible quantifiée**, pas au checkpoint BF16.

## 3. Débit complet et convention de comptage

Notations par cycle `j` :

- `γ_j` : tokens proposés ; `n_j` : positions réellement exécutées par le vérificateur. Une implantation courante exécute `n=γ+1` pour inclure la dernière position en attente et produire le bonus. Instrumenter `n` au lieu de le déduire d'une option.
- `K_j` : propositions consécutives acceptées ; `A_j` : nouveaux tokens effectivement émis, correction/bonus compris. Hors EOS et troncature, `A=K+1`.
- `B_j` : octets de poids réellement lus par l'application ; `B_phys,j` : octets physiques observés à part ; `F_j` : travail du vérificateur ; `D_j` : temps complet du brouillon.
- `V_j` : temps de vérification, comprenant lecture, calcul, déquantification, synchronisation et rollback, avec recouvrements éventuels.

Le débit observé correct est

`R = (Σ_j A_j) / (Σ_j (D_j + V_j + autres coûts non recouverts))`.

Il ne faut ni rapporter seulement les propositions acceptées, ni exclure le brouillon, ni prendre la moyenne non pondérée des débits par cycle. Le temps de préremplissage est publié séparément, puis inclus dans un second débit complet si l'on annonce une vitesse utilisateur.

Si `s_i = Pr(K≥i)` est la probabilité de survie du préfixe de longueur `i`, alors exactement

`E[A] = Σ_(i=0)^γ s_i`, avec `s_0=1`.

L'approximation iid de probabilité conditionnelle constante `α` donne

`a(γ,α) = (1-α^(γ+1))/(1-α)` ; si `α=1`, `a=γ+1`.

Cette approximation doit être remplacée par les préfixes observés. Le taux agrégé de tokens acceptés ne décrit pas à lui seul la survie des longs préfixes.

Modèle sériel de coûts :

`T = D + B/β_eff + F/φ_eff + T_déquant + T_sync + T_alloc - T_overlap`.

`β_eff` et `φ_eff` sont mesurés sur les tailles et les kernels concernés. Si seuls lecture et calcul cible peuvent se recouvrir parfaitement, une borne optimiste est

`T ≥ D + max(B/β_eff, F/φ_eff)`.

Sans preuve de recouvrement, conserver la somme. Un débit de copie RAM/page-cache ne peut remplacer `β_eff` SSD pour un modèle beaucoup plus grand que la RAM. Un nombre d'octets demandé par l'application ne prouve pas une lecture physique.

## 4. Union des experts : le gain dépend des routes, pas seulement du lot

À la couche `l`, `S_(l,t)` est l'ensemble natif de `k_l` experts sélectionnés à la position `t`. `U_l = union_t S_(l,t)` et `C_l` l'ensemble résident. Si chaque expert est lu au plus une fois dans le bloc :

`B_block = B_shared_miss + Σ_l Σ_(e in U_l \ C_l) W_(l,e) + B_prefetch_inutile + B_relectures`.

Pour des experts égaux et un top-k constant :

`k ≤ |U_l| ≤ min(E,nk)`.

Dans un **modèle synthétique**, chaque position choisit un sous-ensemble top-k uniforme indépendant :

`E[|U_l|] = E · [1-(1-k/E)^n]`.

Sous un modèle plus général de probabilités marginales indépendantes par position, `E[|U_l|] = Σ_e [1-(1-p_e)^n]`. La corrélation temporelle exige des traces réelles ; les équations ne prédisent pas ses valeurs.

Sans cache d'experts et à poids égaux, le gain E/S par token émis sur une référence à `k` experts par token est approximativement `G_io = k·E[A]/E[|U|]`. Il faut `E[A] > E[|U|]/k` pour seulement réduire les octets par token, avant de payer le brouillon. Des propositions rejetées peuvent accroître l'union sans augmenter la sortie.

### Pourquoi Granite n'est pas une preuve de passage à 700B

Le [config officiel Granite-3.0-1b-a400m-instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/blob/main/config.json) indique `E=32,k=8,L=24,d=1024,m=512`. Les [configs GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json) et [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json) indiquent respectivement `E=160,k=8,L_MoE=89,d=5120,m=1536` et `E=256,k=8,L_MoE=58,d=7168,m=2048`.

| Positions vérifiées `n` | Union Granite attendue | Union GLM-4.5 attendue | Union DeepSeek-V3 attendue |
|---:|---:|---:|---:|
| 1 | 8,000 | 8,000 | 8,000 |
| 2 | 14,000 | 15,600 | 15,750 |
| 4 | 21,875 | 29,679 | 30,531 |
| 8 | 28,796 | 53,853 | 57,421 |
| 16 | 31,679 | 89,580 | 101,962 |
| 32 | 31,997 | 129,006 | 163,314 |

Table calculée par la formule uniforme indépendante ; **pas une mesure de ces modèles**. Granite réutilise proportionnellement davantage de poids dès les petits blocs : son union sature à 32. Une amélioration démontrée sur Granite doit être accompagnée de microbenchmarks `E=160/256` et de traces réelles avant généralisation.

Exemple synthétique Granite avec `n=γ+1, α=0,8`, aucun expert résident :

| `γ` | `E[A]` | Union attendue | Gain E/S théorique `8·E[A]/U` |
|---:|---:|---:|---:|
| 2 | 2,440 | 18,500 | 1,055× |
| 4 | 3,362 | 24,406 | 1,102× |
| 8 | 4,329 | 29,597 | 1,170× |
| 16 | 4,887 | 31,759 | 1,231× |
| 32 | 4,997 | 31,998 | 1,249× |

Les gains E/S modestes de ce scénario peuvent disparaître une fois le brouillon et les calculs inutiles inclus. Inversement, des routes réelles très corrélées peuvent produire un gain nettement supérieur. Seule l'expérience tranche.

## 5. Conditions nécessaires pour 2, 4, 8 et 15 tok/s

Pour un objectif `r`, il faut `T_cycle ≤ E[A]/r`. Il faut simultanément, même avec recouvrement idéal,

`B_phys/E[A] ≤ β_phys/r`, `F/E[A] ≤ φ_eff/r`, et `D/E[A] < 1/r`.

| Objectif | Temps total maximal / token | Octets physiques maximaux / token à **1 GB/s hypothétique** | À **3 GB/s hypothétiques** | À **6 GB/s hypothétiques** |
|---:|---:|---:|---:|---:|
| 2 tok/s | 500 ms | 0,500 GB | 1,500 GB | 3,000 GB |
| 4 tok/s | 250 ms | 0,250 GB | 0,750 GB | 1,500 GB |
| 8 tok/s | 125 ms | 0,125 GB | 0,375 GB | 0,750 GB |
| 15 tok/s | 66,667 ms | 0,0667 GB | 0,200 GB | 0,400 GB |

Ce sont des **conditions nécessaires optimistes**, pas des prévisions. Brouillon, attention, déquantification et synchronisation consomment une partie du budget ; sans recouvrement ils diminuent d'autant le volume admissible. Les nombres sont en GB décimaux (`10^9` octets), pas en GiB.

Une borne simple du calcul des seules multiplications linéaires est `F ≈ 2·n·P_active` opérations, à augmenter de l'attention et des autres opérations. Les cartes officielles annoncent [GLM-4.5 : 355B totaux / 32B actifs](https://huggingface.co/zai-org/GLM-4.5) et [DeepSeek-V3 : 671B / 37B](https://huggingface.co/deepseek-ai/DeepSeek-V3). Sans calcul spéculatif rejeté, il faudrait approximativement :

| Objectif | GLM-4.5, calcul linéaire minimal | DeepSeek-V3, calcul linéaire minimal |
|---:|---:|---:|
| 2 tok/s | 128 GFLOP/s | 148 GFLOP/s |
| 4 tok/s | 256 GFLOP/s | 296 GFLOP/s |
| 8 tok/s | 512 GFLOP/s | 592 GFLOP/s |
| 15 tok/s | 960 GFLOP/s | 1 110 GFLOP/s |

En spéculation, multiplier par `n/E[A]`, puis ajouter le brouillon. Exemple `γ=8,α=0,8` : `n/E[A]≈2,079`. Ces exigences ne disent rien du débit Metal soutenu réellement disponible, qui doit être mesuré sur les kernels concernés.

### Scénario géant explicitement réfutable

Un expert SwiGLU de largeur `d,m` contient environ `3dm` paramètres. Aux dimensions officielles ci-dessus, les seuls experts routés représentent : GLM-4.5 `335,964B`, DeepSeek-V3 `653,909B`. On suppose ici : Q4 idéal à exactement 0,5 octet/paramètre, zéro métadonnée, tous les poids partagés résidents, zéro expert routé résident, routes uniformes indépendantes, acceptation parfaite de toutes les `n` positions, calcul et brouillon gratuits, débit physique **supposé 3 GB/s**.

| Modèle | Positions `n` | Experts routés lus / bloc, Q4 idéal | Plafond E/S `3n/B` |
|---|---:|---:|---:|
| GLM-4.5 | 1 | 8,399 GB | 0,357 tok/s |
| GLM-4.5 | 8 | 56,539 GB | 0,424 tok/s |
| GLM-4.5 | 16 | 94,049 GB | 0,510 tok/s |
| GLM-4.5 | 32 | 135,442 GB | 0,709 tok/s |
| DeepSeek-V3 | 1 | 10,217 GB | 0,294 tok/s |
| DeepSeek-V3 | 8 | 73,336 GB | 0,327 tok/s |
| DeepSeek-V3 | 16 | 130,223 GB | 0,369 tok/s |
| DeepSeek-V3 | 32 | 208,579 GB | 0,460 tok/s |

**Dans ce scénario précis**, même 2 tok/s est incompatible avec le trafic minimal modélisé : les blocs ne suffisent pas. Cela n'est pas une impossibilité universelle des modèles de 300–700B. Une forte localité réelle, une population d'experts chauds tenant en RAM, une meilleure compression ou un débit physique supérieur peuvent changer la conclusion. À `n=32`, atteindre 2/4/8/15 tok/s demanderait, avant tout autre coût, de réduire le trafic GLM ci-dessus d'environ 64,6/82,3/91,1/95,3 %, et DeepSeek de 77,0/88,5/94,2/96,9 %. Une faible acceptation rend encore ces exigences plus sévères.

## 6. Mémoire : le budget 16 Go n'est pas un cache de 16 Go

Comptabiliser ensemble : poids partagés résidents, experts résidents, modèle brouillon, deux KV caches s'ils ne sont pas réellement partagés, activations, tenseurs de routage, scratch de déquantification, copies et requêtes de préchargement en vol, runtime et marge du système. Les deux variantes d'un ablation doivent respecter le même plafond **total** ; attribuer le cache libéré par l'absence de brouillon à la référence est nécessaire.

Avec un KV GQA conventionnel à `b_kv` octets/élément :

`M_KV = 2·L·S·h_kv·d_head·b_kv`.

Granite BF16 donne `49 152 octets/token`, soit 192 MiB pour `S=4096`. GLM-4.5 BF16 donne `376 832 octets/token`, soit 1 472 MiB pour `S=4096`, avant brouillon et buffers. Les formules MLA ne sont pas celles du GQA : ne pas appliquer mécaniquement ce calcul à DeepSeek-V3, qui déclare un état latent `kv_lora_rank=512` et une composante RoPE distincte dans son [config](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json).

En exécution couche par couche, conserver deux activations `n×d` peut suffire pour le flux principal ; des intermédiaires de projections, routage et attention s'ajoutent. Il n'est pas nécessaire de conserver toutes les activations des `L` couches comme à l'entraînement. Le scratch d'expert peut être borné indépendamment de l'union : charger un expert, traiter ses positions, accumuler, libérer, poursuivre. Éviter une matérialisation simultanée de tous les experts de l'union.

En soustrayant les seuls experts routés du nombre total **arrondi** annoncé, le reste vaut environ 19,0B paramètres GLM et 17,1B DeepSeek, soit 9,5 et 8,5 GB au Q4 idéal. C'est un ordre de grandeur, pas un inventaire exact des tenseurs partagés. À BF16 ces restes dépassent 16 Go ; à Q4 ils prennent déjà la majeure partie du budget disponible après système, brouillon et KV. Une politique « tout partagé résident » doit donc prouver qu'elle tient ; sinon résidence partielle et streaming restent nécessaires.

La [carte officielle DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) distingue 671B de cible principale et 14B de module MTP supplémentaire dans les fichiers distribués. Les paramètres de prédiction et leur état ne sont pas gratuits : les intégrer exige un chemin d'exécution effectivement implémenté, son propre budget, sa compatibilité de cache et une validation. Les tableaux précédents n'incluent pas un module MTP ni un modèle brouillon.

## 7. Critères de D, E, F et G

**D — Résidence.** L'intérêt d'un objet en cache est son coût de relecture évité par unité mémoire, pas son étiquette « partagé ». Pour un objet de taille `W_i` utilisé avec fréquence `f_i`, un score de départ est `f_i·T_load,i/W_i`, ajusté au prochain usage, à la contention et au coût d'éviction. Le batching lit un poids partagé une seule fois par cycle : sa fréquence relative baisse face à l'union d'experts qui s'élargit. Comparer cache adaptatif, partagé, experts et hybride sur les mêmes traces, au même budget ; aucun classement universel n'est établi.

**E — Longueur adaptative.** Choisir `γ` dans `{0,2,4,8,16,32}` pour maximiser `a_hat(γ)/T_hat(γ)` sous contrainte mémoire ; `γ=0` est la référence sans spéculation. Accepter une extension de lot seulement si son gain marginal estimé satisfait `Δa/ΔT > a/T`. Les estimations d'acceptation, union, octets, durée et mémoire proviennent exclusivement du passé. Explorer de manière bornée, réévaluer quand le contexte/cache change, et désactiver la spéculation lorsqu'elle perd. Comparer à chaque longueur fixe, pas seulement à une longueur arbitrairement mauvaise. La dynamique seule est déjà étudiée dans [DISCO](https://arxiv.org/abs/2405.04304).

**F — Quantification/stockage.** La taille réelle vaut `P·bits/8 + échelles + zéros + index + padding`. Par exemple, un groupe de 64 poids à 4 bits avec échelle FP16 coûte 4,25 bits/poids ; ajouter un zéro FP16 donne 4,5 bits/poids. Ces exemples ne décrivent pas tous les formats. Mesurer le fichier produit. Quantifier doit réduire le temps lecture+déquantification+calcul et la mémoire globale, puis passer une évaluation de qualité ; une proximité de logits sur une poignée d'exemples n'est pas une mesure de perplexité ou de capacité. Comparer le regroupement exact à une référence de même quantification.

Pour une compression sans perte de ratio `ρ` et un décompresseur de débit `δ` exprimé en octets décompressés/s, le modèle sériel donne un gain seulement si `ρB/β+B/δ < B/β`, donc `δ > β/(1-ρ)`, hors copies et allocations. Une réduction de fichier peut ralentir la génération à cache OS chaud. Le format expert contigu peut améliorer `β_eff` sans modifier les valeurs ; mesurer taille des requêtes, nombre d'appels et copies, avec déquantification incluse. Un kernel fusionné peut éviter d'étendre tout l'expert, mais il doit être réellement disponible et validé sur le backend utilisé.

**G — Préchargement.** Une lecture prédictive est admissible si son gain attendu est positif : `p_hit·T_masqué > coût_des_lectures_inutiles + évictions + synchronisation + contention`. Avec SSD déjà saturé, une fausse prédiction consomme une bande passante nécessaire aux demandes et peut ralentir le moteur. Dédupliquer demande/prédiction, plafonner octets en vol et réservations, prioriser les demandes ; mesurer taux d'utilisation, octets inutiles et gain bout en bout. MoE-Infinity établit l'antériorité du préchargement guidé par activation. Rien ne justifie de réactiver SpecPrefetch par défaut sans ablation positive dans la configuration cible.

## 8. Expériences minimales capables de réfuter les hypothèses

1. **A+B+C** : cible Granite officielle, mêmes prompts/longueurs/précision/budget ; glouton référence et blocs 2/4/8/16/32. Tester propositions parfaites, complètement mauvaises, courtes, EOS et divergences tardives en contrôle ; inclure ensuite un vrai brouillon entraîné. Une proposition copiée de la référence est un oracle, pas une accélération utilisable.
2. **B seule** : mêmes activations et routes capturées, exécution position par position puis union ; comparer routes, poids de routage, sorties et logits. Pour `n=1,2,4,8,16,32`, mesurer octets et nombre de chargements uniques. Un gain de volume avec augmentation du temps doit rester un résultat négatif de performance.
3. **Généralisation** : microbenchmarks `E=32/160/256`, `k=8`, experts réutilisés, indépendants et disjoints, fichiers et buffers plus grands que le cache explicite. Aucun de ces microbenchmarks ne vaut validation GLM ou DeepSeek entraîné.
4. **D/E/G** : ablations sur ordre de prompts aléatoire fixé, mêmes seeds/budgets, mesures répétées, coût du brouillon et préchargement inclus ; ne pas utiliser les futurs taux d'acceptation ou futures routes pour la politique. Garder les ralentissements et les scénarios sans gain.
5. **F** : mesurer octets empaquetés, temps de lecture/déquantification/calcul, mémoire de pointe et qualité sur un corpus déclaré ; préserver le checkpoint de référence et documenter exactement le format et sa calibration.
6. **État du stockage** : publier séparément les résultats cache OS chaud et ceux où des lectures physiques sont observées. Un redémarrage de l'objet cache Python ne vide pas le cache OS. Aucun résultat sur SSD interne ne doit être étiqueté Fanxiang. Publier les limites de l'instrument physique disponible.

La sélection finale doit utiliser `tokens émis corrects / seconde`, `octets physiques et applicatifs / token émis`, mémoire totale de pointe, exactitude, variance et complexité. Les trois mécanismes les plus prometteurs **à vérifier**, pas encore les trois gagnants mesurés, sont : union native + blocs causaux ; stockage quantifié contigu avec kernel adapté ; politique de lot/cache fondée sur acceptation et octets. Le préchargement n'est qu'un complément si la fenêtre de calcul peut réellement masquer les lectures.

## 9. Synthèse automatique des preuves

Après les campagnes, exécuter `python benchmarks/synthesize_multitoken.py --primary-only --output evidence-multitoken/synthesis.json`. Le script ne lance aucune inférence. Il apparie les mêmes cas et répétitions, conserve les campagnes dans des groupes séparés et exclut les lignes `sequence_seed`. Il rapporte les médianes des rapports de débit appariés, les sommes de temps/octets divisées par les tokens réellement émis, l'acceptation pondérée par les propositions et les budgets cible+brouillon.

Une série dont la grille est incomplète, les sources ont changé, les sorties divergent, les budgets diffèrent ou les tableaux bruts ne correspondent pas à leur hash ne produit aucun gain accepté. Les données exploratoires sans `primary_campaign=true` sont ignorées avec `--primary-only`. Un code de sortie 2 signale une série rejetée ou l'absence de série acceptée ; le fichier de synthèse conserve les raisons. Les positions contrôlées de B sont toujours étiquetées comme vérifications causales, et ne deviennent jamais des tokens générés par cette synthèse.
