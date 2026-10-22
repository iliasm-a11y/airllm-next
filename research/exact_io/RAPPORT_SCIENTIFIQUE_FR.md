# AirLLM Next — recherche exacte hors cœur

**Reprise du 22 octobre 2026. Branche : `research/exact-io-2026-10-22`.**

## 1. Résultat principal et portée

Nous avons retrouvé les travaux de Claude, reproduit leurs décomptes depuis les
en-têtes, réanalysé les traces, exécuté des expériences sur de vrais poids et
ajouté un transport réversible au moteur existant. Aucun résultat ne démontre
encore l'inférence interactive de GLM-5.3 ou Qwen3.8 complets sur 16 Go.

La stratégie la mieux défendable pour une réduction **importante** du trafic
reste le partage d'une lecture entre plusieurs positions **toutes vérifiées par
le modèle cible**, associé à une résidence réellement rentable par octet de RAM.
Elle peut fournir de grands facteurs en prefill, en débit multi-flux et dans des
sorties suffisamment prévisibles. Son efficacité en génération libre dépend du
routage et de l'acceptation effectivement mesurés. Nous ne possédons pas ces
deux mesures sur les modèles géants.

La compression sans perte est complémentaire. Notre codec palette réduit les
octets des échantillons BF16 d'environ 25 %, mais l'intégration complète coûte
du temps. Elle reste optionnelle. Ni ce codec, ni les scénarios de spéculation
ne constituent une découverte démontrée d'un facteur 10× en inférence complète.

Trois distinctions guident l'ensemble du dossier :

- **Exactitude des poids** : la décompression restitue tous les bits.
- **Exactitude du calcul** : le même ordre d'opérations, la même précision et les
  mêmes kernels doivent produire le résultat de référence.
- **Exactitude de la génération** : même séquence gloutonne, ou même loi pour un
  échantillonnage correctement corrigé. Même loi ne signifie pas mêmes tokens
  pour une même graine et un ordonnancement de tirages différent.

## 2. État réellement retrouvé

La branche distante de référence a été comparée par l'API GitHub au commit local
`d43032f38a0325aafd2973847d7ed84b21280c61` : statut `identical`, zéro commit
d'écart. Le travail a été isolé dans une nouvelle branche, sans toucher `main`.

Le dossier `inherited/` conserve 19 fichiers, leurs tailles, empreintes et
provenances dans `manifest.json`. Il comprend les 141 en-têtes GLM, 213 en-têtes
Qwen BF16, un inventaire NVFP4, les décomptes de Claude, la théorie PRISME et son
simulateur, les programmes de collecte et les traces OLMoE.

Les répertoires de passation retrouvés étaient vides. Nous n'avons retrouvé ni
décodeur C de Claude ni résultats bruts de sa piste entropie. Le décodeur C de
cette reprise a donc été écrit ici. Aucune mesure manquante n'a été reconstituée
par supposition.

Les traces OLMoE contiennent 16 couches, 8 textes de 4 096 tokens et 256 textes
de 64 tokens : 49 152 positions au total. Les requêtes de notre analyse longue
portent sur les 32 768 premières positions longues, avec leurs experts à chaque
couche. Il s'agit de **teacher forcing** : les tokens sont imposés. La validation
héritée de deux couches, avec faibles écarts FP32 et accord des ensembles de
top-k, n'a pas été réexécutée. Le modèle OLMoE et son environnement ne sont pas
disponibles ici sous une forme directement exécutable.

Un checkpoint Ling-mini-2.0 déjà présent a permis de lire des fragments et des
experts réels. Aucun checkpoint GLM/Qwen, aucun nouveau grand modèle, n'a été
téléchargé.

## 3. Matériel et conditions de mesure

L'inventaire local indique **MacBook Air, Apple M5, Mac17,3, 16 GB**, CPU 10 cœurs.
Python 3.14, NumPy et pytest sont disponibles. Torch, MLX et safetensors ne sont
pas installés dans cet interpréteur. Lors de la reprise, un environnement
existant Python 3.12 avec **PyTorch 2.14.0**, NumPy 2.5.3, MLX et safetensors a
été retrouvé. Aucun paquet n'a été installé. Le produit matriciel CPU PyTorch
fonctionne ; l'import de MLX y échoue avec « No Metal device available », même
en demandant ensuite le device CPU. Les sorties des deux sondes sont conservées
dans `backend_probes.json`. Ce nouvel environnement étend la validation CPU,
sans lever le blocage du runtime MLX.

Le lecteur par offsets n'a besoin
d'aucun de ces trois modules : l'import de safetensors a été rendu paresseux.

Une sonde Objective-C appelle réellement `MTLCreateSystemDefaultDevice` : aucun
device n'est visible dans ce contexte d'exécution. Cela ne signifie pas que le
Mac n'a pas de GPU. Nos mesures sont donc des **mesures CPU sur M5**, pas Metal.

Les lectures de comparaison sont chaudes, sur le volume local. Le SSD externe
n'a pas été mesuré. Les compteurs de lectures physiques restent `null` ; les
octets demandés aux fichiers ne sont pas présentés comme des octets NAND. La
bande passante de la RAM unifiée n'est utilisée nulle part comme débit de SSD.

`environment.json` conserve ces conditions sans numéro de série ni nom d'hôte.
Les deux inventaires `environment-resumed*.json` distinguent les interpréteurs
retrouvés lors de la reprise. Le script de capture interroge le device au lieu
de supposer le résultat, sépare commit de départ et commit courant, et ne
réemploie plus un échec DNS historique comme une nouvelle mesure réseau.
Les répétitions des benchmarks sont toutes conservées. Une première comparaison
de transport exécutée avec d'autres expériences est conservée séparément sous
`transport-initial-contended.json` et exclue du classement principal.

## 4. Audit des modèles et des 13,8 / 8,1 secondes

Les sommes ci-dessous sont recalculées indépendamment sur les en-têtes hérités.
Les architectures ont aussi été confrontées aux configurations officielles [1–3].
Les nombres « hors échelles » incluent quelques buffers, notamment des biais de
routage ; ils ne doivent pas être confondus avec un comptage `nn.Parameters`.
Go = 10⁹ octets, To = 10¹² octets.

| Quantité | GLM-5.3 officiel FP8 | Qwen3.8 officiel BF16 |
|---|---:|---:|
| Shards inspectés | 141 | 213 |
| Éléments hors échelles, MTP compris | 753 329 940 480 | 2 446 182 725 504 |
| Éléments hors échelles du modèle principal | 743 377 019 904 | 2 419 804 697 984 |
| Octets des tenseurs, MTP compris | 755 617 140 416 | 4 892 365 451 008 |
| Poids toujours actifs par passe, hors embedding d'entrée | 18,729 Go | 93,897 Go |
| Experts routés actifs par position, échelles comprises | 22,655 Go | 92,610 Go |
| Une ligne d'embedding d'entrée | 12 288 octets | 16 384 octets |
| Total naïf par position, avant cache | **41,383 Go** | **186,507 Go** |

Le MTP est compté pour le stockage, pas comme un passage obligatoire du
décodage ordinaire. Même sans le MTP, le Qwen BF16 représente environ 4,840 To.

À un débit **supposé** de 3 Go/s, 41,383447104 / 3 = 13,794482368 secondes de
transfert GLM par token. C'est une reproduction arithmétique des **13,8 s**,
pas une mesure du matériel, et elle ignore calcul, routage et synchronisations.

Les **8,1 s** ne sont pas indépendamment reproduites. Le jeu de paramètres exact,
les ratios de compression GLM et les acceptations MTP correspondantes n'ont pas
été retrouvés. Ajuster des paramètres pour retrouver ce chiffre serait une
calibration circulaire. `scenarios.json` conserve cette conclusion.

Pour que l'intégralité du Qwen BF16 tienne dans 2 To, il faudrait un ratio sans
perte d'au plus **0,4088**, avant système de fichiers et fichiers auxiliaires.
Nous ne disposons d'aucun encodage mesuré qui l'atteigne sur ce checkpoint. Il
serait néanmoins incorrect d'en faire une impossibilité mathématique générale
à partir des seules entropies marginales. NVFP4/INT4 et FP8 dérivés ne sont pas
des encodages sans perte du checkpoint BF16 original.

## 5. Modèle OTA corrigé

Une passe p vérifie un ensemble de positions Sₚ. Pour chaque bloc de poids b,
sa représentation sérialisée mesure c_b octets. Soit Dₚ l'ensemble des blocs
effectivement chargés pendant cette passe, en tenant compte du cache, de ses
évictions, des reprises et de l'ordonnancement. On mesure :

\[
B_p=\sum_{b\in D_p}c_b+B_{index,p}+B_{surlecture,p}+B_{draft,p}+B_{KV,p}.
\]

La somme est un multiensemble si un bloc est rechargé. Des experts distincts
ne sont pas nécessairement des lectures distinctes si un encodage partage
réellement de l'information ; ce partage doit alors être compté explicitement.

Pour une exécution finie, avec τₚ tokens réellement émis par passe :

\[
OTA=\frac{\sum_p B_p}{\sum_p\tau_p}.
\]

On inclut corrections et bonus, pas uniquement les brouillons acceptés. Un EOS
accepté peut supprimer le bonus. Il faut éviter la moyenne non pondérée des
ratios de chaque passe. Sous des conditions ergodiques ou régénératives
appropriées, la limite est E[B]/E[τ] ; la stationnarité seule n'établit pas une
structure de renouvellement.

En régime pipeline idéal, une borne de temps est

\[
T_p\ge\max(B_p/\beta_{SSD},\ D_p^{decode}/\delta,\ F_p/\Phi).
\]

Ce maximum est une **borne optimiste**, pas la preuve qu'on réalise le
recouvrement. Un chemin sériel paie approximativement une somme, avec des
surcoûts. La mesure finale demeure la durée totale divisée par les tokens émis.

## 6. Démonstrations, corrections et contre-exemples à PRISME

### 6.1 La sensibilité globale des bits ne suffit pas à la borne annoncée

L'hypothèse « pour chaque bit, il existe une entrée qui le rend pertinent » ne
dit pas « pour chaque entrée, chaque bit est pertinent ». Une application
linéaire W x peut retourner zéro pour x = 0 sans lire W, tout en utilisant W
sur les autres entrées. Un algorithme reste exact sur toutes ses entrées en
choisissant son chemin selon x. Cela réfute la déduction par passe de T1 dans
sa généralité, pas l'utilité pratique de compter les blocs d'une implémentation.

Une borne défendable doit préciser la famille de poids, les entrées, les
prétraitements et la représentation autorisés. Dans le contrat plus étroit
« reconstruire ces blocs originaux et exécuter ces kernels », un bloc requis
absent de la RAM doit être acquis ou reconstruit à partir de sa représentation.
Cette contrainte de transport n'exclut pas une autre identité algébrique exacte.

### 6.2 Une entropie marginale n'est pas une borne inférieure du fichier

Pour des variables Wᵢ, H(W₁,…,Wₙ) ≤ Σ H(Wᵢ). L'indépendance ne découle ni d'un
histogramme ni d'une faible corrélation linéaire. Une information mutuelle
empirique presque nulle n'est pas une égalité d'indépendance prouvée. La borne
utile avec une information résidente R serait conditionnelle, H(W | R), pour
une distribution et un contrat d'encodage définis.

Nous retirons donc l'affirmation « aucun codage exact ne peut faire tenir Qwen
dans 2 To » en tant que théorème démontré. Les tailles brutes trop grandes et
l'absence actuelle d'une solution mesurée sont, elles, des faits établis.

### 6.3 Ce que prouve réellement l'amortissement spéculatif

Avec K brouillons, πⱼ est la probabilité d'accepter les j premiers, π₀ = 1.
Sans EOS, l'espérance de progression est τ = Σⱼ₌₀ᴷ πⱼ. Pour une vérification
linéaire complète, soit k νⱼ le nombre moyen de nouveaux experts introduits par
la position j. Hors cache, le trafic routé moyen est R Σνⱼ. Ainsi

\[
G_r=\frac{\sum_j\pi_j}{\sum_j\nu_j}.
\]

Si tous les νⱼ sont strictement positifs, Gᵣ est une moyenne pondérée des
πⱼ/νⱼ et se situe entre leur minimum et maximum. Si νⱼ = 0 alors que πⱼ > 0,
le plafond fini correspondant disparaît. L'identité est utile, mais ni π ni ν
ne sont mesurés ici sur les brouillons des cibles géantes.

Les routes teacher-forced ne représentent pas nécessairement les routes après
un brouillon erroné. Les branches d'un arbre peuvent parfois introduire très
peu de nouveaux experts ; les chaînes ne dominent donc pas tous les arbres par
un théorème général. Notre prototype reste linéaire.

### 6.4 Le modèle uniforme indépendant n'est pas le pire cas absolu

Sous indépendance des positions, E|U(n)| = Σₑ[1−(1−pₑ)ⁿ], et les probabilités
uniformes maximisent cette somme pour Σpₑ=k. En présence de corrélations
arbitraires, cette conclusion ne tient plus : deux choix uniformes parmi quatre,
toujours différents, ont une union 2, contre 1,75 sous indépendance.

Le prefill n'exige donc pas nécessairement un balayage intégral pour tous les
prompts. Un prompt avec un routage presque constant est un contre-exemple.
Greffer une vérification au prefill peut réutiliser des lectures, mais son
calcul, ses activations, ses états et ses éventuelles routes nouvelles ne sont
pas « gratuits ».

### 6.5 La résidence dépend des octets réellement conservés

Pour une décision statique simplifiée, la densité d'économie d'un bloc vaut
q_b × octets_stockés / octets_résidents, éventuellement corrigée du coût de
décodage et des surlectures. Le moteur existant conserve des poids **décodés**.
Un octet FP8 devient généralement deux octets en BF16. Un poids toujours actif
avec cette expansion a une densité 1/2 ; un expert BF16 utilisé avec probabilité
0,75 a une densité 0,75. Les poids toujours actifs ne dominent donc pas
automatiquement tous les experts, même dans ce modèle statique.

L'optimum dynamique dépend aussi des échéances et de l'accès couche par couche.
Belady donne un oracle pour une trace connue, pas une politique réalisable qui
connaîtrait le routage futur. Les scans peuvent rendre LRU et SLRU inefficaces.

### 6.6 Un facteur 10× n'est pas interdit sans hypothèses supplémentaires

Considérons 16 positions proposées correctement, avec les mêmes experts à
chaque couche. Le cible exécute tous les experts requis pour les 16 positions,
mais lit leurs poids une seule fois : le trafic par token est divisé par 16.
Cela ne requiert aucune suppression d'expert. Il faut encore compter calcul et
latence du bloc, et disposer de ces propositions correctes.

Ce contre-exemple réfute une impossibilité inconditionnelle de 10× fondée
uniquement sur la taille du modèle. Il ne démontre pas l'existence de ce régime
sur GLM/Qwen en conversation libre. Une spéculation parfaite de profondeur 512
donne de grands gains **simulés**, avec des pauses de vérification très longues.
Nous les conservons comme cas limites, sans les présenter comme résultats utiles.

### 6.7 Exactitude numérique et calcul probabiliste

Décompresser les mêmes poids et réordonner un GEMM ne sont pas la même opération.
Pour la parité bit à bit par bloc, chaque réduction, normalisation, attention,
combinaison d'experts et mise à jour récurrente doit respecter le contrat de
référence, indépendamment du batch. Un faible écart de logits peut changer un
top-k ou un argmax presque ex æquo. Nos tests de transport ne certifient pas
cette invariance dans MLX, ni les activations FP8 dynamiques de serveurs GPU.

Pour un échantillonnage spéculatif exact [4], on accepte une proposition q(x)
avec min(1,p(x)/q(x)), puis on utilise le résidu normalisé (p−q)⁺ au rejet.
La masse acceptée de x est min(p(x),q(x)). La masse résiduelle est
(p(x)−q(x))⁺. Leur somme vaut p(x). Les tests vérifient cette identité avec des
fractions rationnelles exactes, y compris des supports disjoints. Le moteur
livré reste glouton ; ce test n'est pas une implémentation de sampling GLM/Qwen.

## 7. Expériences exécutées

### 7.1 Réutilisation et caches sur traces OLMoE

Les experts d'une fenêtre sont regroupés par couche. On compte exactement son
union. La réduction est comparée à une lecture par expert et par position, pas
au prefill AirLLM actuel qui regroupe déjà les experts.

| Positions connues par fenêtre | Réduction du nombre de lectures routées |
|---:|---:|
| 2 | 1,251× |
| 4 | 1,606× |
| 8 | 2,187× |
| 16 | 3,248× |
| 64 | 9,421× |
| 128 | 17,525× |

Ce sont des réductions de requêtes sur des **tokens connus**. Elles ne sont ni
des accélérations de décodage ni des acceptations mesurées. Pour les profondeurs
1 à 65, `routing.json` conserve des courbes par catégorie sur les mêmes fenêtres
échantillonnées. Les catégories ne sont pas des réplicats indépendants d'un
échantillonnage de conversation libre ; aucune précision statistique artificielle
n'est revendiquée.

Sur 524 288 requêtes d'experts, avec 64 unités de cache globales, le vrai
`WeightCache(policy='slru')` produit 524 288 défauts. Un ensemble statique choisi
sur 512 tokens **antérieurs et disjoints** produit 395 816 défauts, coût initial
des pins inclus : environ 24,5 % de lectures en moins. L'oracle avec bypass en
produit 315 527, mais connaît l'avenir. À 128 unités, SLRU donne 456 945 défauts,
contre 372 857 pour LRU dans le simulateur. Ces résultats justifient le choix
mesuré d'une politique, pas une modification globale du défaut sur GLM/Qwen.

Les unités ont une taille égale et excluent attention, tête et expansion des
poids. Ce sont des défauts de cache simulés/rejoués, pas des volumes SSD mesurés.
Le calcul de Belady a été confronté à une recherche exhaustive sur de petites
traces.

### 7.2 Compression de poids entraînés

81 fragments de 256 Kio ont été lus : 3 couches, 3 experts, 3 projections,
début/milieu/fin. Total : **21 233 664 octets** BF16 de Ling-mini. Les offsets,
formes et SHA256 sont dans `sample_provenance.json`. L'échantillon n'est pas
représentatif garanti d'un autre checkpoint.

| Codec | Ratio encodé/original | Débit de sortie, avec hash sauf mention |
|---|---:|---:|
| Brut | 1,0000 | 3,428 Go/s, hash seul |
| zlib niveau 1 | 0,7916 | 0,461 Go/s |
| zstd niveau 1 | 0,7698 | 1,018 Go/s |
| Octets séparés + zstd 1 | **0,6739** | 0,648 Go/s |
| Palette 16, NumPy | 0,7504 | 0,724 Go/s |
| Palette 16, C/NEON + allocation + hash | 0,7504 | **3,015 Go/s** |
| Palette 16, noyau C seul | 0,7504 | 34,213 Go/s, sans hash ni allocation |

Les 15 répétitions individuelles sont conservées. Ces débits sont ceux du
décodeur CPU à partir de mémoire ; le dernier ne représente pas le chemin
complet et ne doit pas être retenu comme débit de stockage.

Le codec conserve les octets bas tels quels. Les octets hauts sont représentés
par 15 valeurs fréquentes codées sur quatre bits, avec une seizième valeur
d'échappement et un flux exact des exceptions. La table et les exceptions sont
comptées. L'inverse NEON reconstruit les octets, sans calcul flottant. Les
65 536 mots de 16 bits ont été testés, ainsi que les longueurs impaires en
nombre de mots et les flux riches en exceptions. Ce codage dictionnaire est
une technique connue ; nous n'en revendiquons pas l'invention.

### 7.3 Le coût d'intégration ne disparaît pas

Le vrai lecteur AirLLM par offsets et son overlay lisent les mêmes 81 fragments,
dans un ordre de variantes mélangé, avec sept répétitions par variante. Les
contrôles de bits sont hors chronométrage ; les checksums du mode normal sont
dans le chronométrage. `transport.json` donne les temps, octets et classements
exacts de la dernière exécution isolée.

Le lecteur non compressé est le plus rapide avec les pages chaudes. Le mode C
sans hash reste plus lent ; le mode avec SHA256 est encore plus lent. Le trafic
du pack, index binaire compris, est d'environ **0,75046 octet par octet demandé**
pour ces lectures entières. Le résultat négatif d'intégration est conservé.

| Chemin complet, pages chaudes | Durée médiane pour 21,23 Mo | Rapport de vitesse au lecteur original |
|---|---:|---:|
| OffsetCheckpoint original | 1,355 ms | 1,000× |
| Palette NumPy + hash | 32,537 ms | 0,0416× |
| Palette C + hash | 10,488 ms | 0,1292× |
| Palette C sans hash, ablation | 4,216 ms | 0,3214× |

Un gain futur sur SSD doit satisfaire les coûts réels de lecture, copies,
décompression et éventuel hash. Pour un ratio ρ et un décodeur à δ octets
reconstruits/s, le recouvrement idéal plafonne à min(β/ρ,δ). Un décodeur mesuré
à environ 3 Go/s avec hash ne permet pas de promettre 4 Go/s utiles sur un SSD
de 3 Go/s. Le pipeline à plusieurs workers et l'absence de buffers intermédiaires
doivent être mesurés, pas présumés.

### 7.4 Parité sur de vrais experts

Quatre experts complets sont testés, dans les couches 1, 10 et 19 de Ling-mini.
Pour chacun, gate, up et down passent par le lecteur original puis par le pack.
Quatre vecteurs FP32 de dimension 2 048 alimentent exactement les mêmes
réductions par ligne et le même SwiGLU.

**Poids identiques bit à bit, sorties identiques bit à bit, écart maximal 0**
pour les quatre experts. Les appels singleton concordent aussi avec le
regroupement de l'interface testée. Les tableaux de sorties sont conservés
dans `trained_expert_outputs.npz`, avec leurs empreintes dans `trained_parity.json`.
Il s'agit de blocs réels entraînés, pas de logits du modèle entier ni d'une
évaluation de qualité linguistique.

### 7.5 Factorisation exacte et information partagée

Neuf projections réelles ont été testées modulo le nombre premier 65 521.
Les BF16 finis sont des rationnels dyadiques, convertis exactement modulo ce
premier, sans arrondi flottant. Un rang r modulo le premier certifie un rang
d'au moins r sur les rationnels : un mineur non nul modulo le premier ne peut
être nul sur Q. Les dénominateurs sont des puissances de deux, donc inversibles.

Six projections ont rang **512**, maximal. Trois projections de l'expert 0 de
la couche 1 ont rang **452** et exactement 60 lignes ou colonnes nulles. Leur
rang est certifié par la borne inférieure modulaire et la borne supérieure
donnée par ces zéros. Les résultats empêchent une factorisation de rang inférieur
à ces valeurs ; ils ne prouvent pas l'absence de toute structure exploitable.

La structure nulle constitue une piste de codage réversible réelle sur ces
tenseurs. Elle ne justifie pas de retirer des neurones des calculs de référence :
les ordres de somme, zéros signés et cas non finis devraient aussi être traités.
Aucune telle suppression n'a été implémentée.

Un second essai code trois experts alignés par ancre + XOR des deux autres,
puis zstd, avec le coût de l'ancre. Il produit **3,33 % d'octets supplémentaires**
par rapport à trois encodages indépendants. Ce résultat réfute l'intérêt de ce
transform précis sur l'échantillon ; il ne mesure pas l'information mutuelle
générale entre experts.

### 7.6 Regrouper les positions peut changer les arrondis

`numerical_contract.py` lit les **quatre experts Ling entraînés déjà conservés**,
en vérifiant chaque hash de bloc. Il reproduit d'abord les quatre empreintes
de sorties NumPy précédentes. Les mots BF16 importés dans PyTorch sont également
comparés bit à bit avant chaque expérience : le transport des poids est exact.

Chaque expert est ensuite exécuté sur 1, 2, 4, 8, 16 et 32 positions, avec un
seul thread CPU. Nous comparons ses appels PyTorch singleton à un appel par
bloc, pour des poids/calculs FP32, puis BF16. Les vecteurs sont déterministes ;
les quatre premiers sont ceux de la précédente expérience. Ce sont **des
sorties d'experts**, pas des logits ni des tokens d'un modèle complet.

| Contrat CPU | Comparaisons | Comparaisons avec au moins un bit différent | Écart absolu maximal |
|---|---:|---:|---:|
| PyTorch FP32, singleton contre bloc | 24 | **20** | **0,0000152587890625** |
| PyTorch BF16, singleton contre bloc | 24 | 0 | 0 |
| Même ordre d'appels singleton, deux exécutions | 48 | 0 | 0 |

En FP32, toutes les tailles strictement supérieures à 1 produisent au moins un
écart pour chacun des quatre experts. La correspondance BF16 sur cet échantillon
ne constitue pas un théorème d'invariance pour d'autres entrées ou kernels.
Les 48 comparaisons détaillées, versions et empreintes sont conservées dans
`numerical_contract.json`. Aucune accélération n'est déduite de cette expérience.

Ce résultat transforme une réserve théorique en **contre-exemple mesuré** à
l'implication « poids identiques ⇒ sorties bit à bit identiques après batching ».
Il ne prouve ni dégradation des capacités, ni changement de token dans un modèle
complet. Il suffit toutefois à interdire une garantie bit à bit fondée sur le
seul transport réversible. Les prédictions d'un bloc spéculatif doivent encore
être confrontées au contrat singleton du moteur cible.

Une seconde précaution concerne l'ordre d'accumulation des experts. En FP32,
avec a = 2²⁴, ((a + 1) − a) = 0 alors que ((a − a) + 1) = 1. Cet exemple
**synthétique**, conservé séparément, interdit de supposer l'associativité
flottante. Partager un chargement reste compatible avec un calcul ligne par
ligne et le même ordre de somme ; le prix de ce contrat strict doit être mesuré.
Le contrôle CPU identique ligne par ligne établit seulement son propre contrat,
sans certifier les attentions, états récurrents ou kernels MLX du modèle entier.

## 8. Méthodes : fidélité, ressources, gains et réfutation

| Méthode | Propriété préservée et hypothèse | RAM et calcul supplémentaires | OTA / gain observé ou potentiel |
|---|---|---|---|
| Union d'experts par bloc | Tous les experts routés sont exécutés ; invariance numérique à valider | Activations/KV du bloc et un expert à la fois ; chaque position est calculée | 2,187× moins de requêtes à 8 positions connues OLMoE ; déjà présent au prefill de référence |
| Brouillon de spans vérifié | Le cible décide chaque token ; proposition parfois juste | Contexte déjà détenu, recherche bornée ; calcul cible des propositions rejetées | Gain de génération non mesuré ; remplace le besoin d'un checkpoint brouillon, pas le modèle cible |
| Cache avec admission/résidence | Aucun changement des opérations, stockage intact | Budget explicite de poids ; choix selon coût décodé et réutilisation | −24,5 % de défauts contre SLRU au cas 64 unités ; résultat de trace |
| Palette sans perte C | Tous les bits restitués, flux intact | Résultat + O(bloc) par worker ; décompression et checksum | −24,95 % d'octets sur échantillon BF16 ; perte de vitesse en lecture chaude |
| Séparation d'octets + zstd | Tous les bits restitués | Décodage et réentrelacement plus coûteux | −32,61 % de taille, seulement 0,648 Go/s de sortie mesurés avec hash |
| XOR inter-experts + ancre | Transformation réversible, ancre nécessaire | Ancre retenue/décodée + XOR | +3,33 % de taille : rejeté pour cet échantillon |
| Factorisation exacte | Identité sur Q ; ordre flottant ensuite à établir | Facteurs potentiellement aussi grands ; calcul intermédiaire | Pas de rang minuscule trouvé : 512 ou 452 sur neuf projections |
| Prefetch et disposition contiguë | Ne change pas les experts exécutés | Buffers réservés, copies et synchronisations | Réduit l'attente ou les petits accès ; ne supprime pas automatiquement des octets |
| Cache de poids encodés | Identité des bits avant calcul | Décompression à chaque usage ; budget partagé à construire | Peut retenir plus de FP8 que le cache décodé ; pas de gain cible mesuré ici |

Pour GLM, les trois premières stratégies sont pertinentes sous réserve de ses
kernels DSA/FP8. Pour Qwen, elles doivent aussi respecter les états DeltaNet,
convolutions, masques et retours arrière. La porte `swish` officielle, NVFP4,
le MTP et la spéculation Qwen complète restent non validés dans ce moteur.
L'overlay conserve les formats originaux mais ne lève aucun de ces gardes.

Chaque proposition se réfute expérimentalement par un échec de parité, un
dépassement du budget mémoire, une OTA qui augmente ou une durée par token
qui ne baisse pas. L'acceptation faible, les routes nouvelles, les sorties non
répétitives et le cache froid sont des cas défavorables nécessaires.

## 9. Modifications fonctionnelles livrées

Le module `airllm.nextgen.lossless` fournit un format par blocs, un index binaire
de 56 octets par bloc, une création bornée en mémoire, le fallback brut, les
lectures de lignes et d'experts, les bornes avant décodage et le SHA256 par bloc.
Le lecteur original reste utilisé pour les tenseurs non empaquetés. Le pack
n'exécute jamais de bibliothèque mentionnée par son manifeste ; le décodeur
natif est fourni explicitement dans la configuration.

Les deux runtimes acceptent `lossless_dir`, sans activation par défaut. Le budget
de travail réserve six blocs par worker. Les requêtes reconstruisent un tableau
borné et ne chargent pas une banque d'experts entière. L'index global reste
par tenseur et les enregistrements de blocs sont lus à la demande, ce qui évite
des millions de dictionnaires Python.

Les compteurs séparent octets sérialisés, index, octets reconstruits et octets
demandés. Les métriques de génération exposent les octets encodés ; avec l'overlay,
les compteurs du vérificateur spéculatif portent sur ces octets de fichiers,
avec les replis et le prefetch, pas sur la NAND. Le mode sans hash est une
ablation explicite et n'est jamais choisi automatiquement.

Le pack est un **overlay**, pas un checkpoint autonome : la source originale
reste nécessaire. Il ne résout donc pas à ce stade la capacité disque du Qwen
BF16. Le mode refuse l'association à un sidecar de quantification supplémentaire.

`PromptSpanDraft` propose uniquement des tokens déjà disponibles dans le
contexte, avec une recherche bornée. Il s'insère dans le vérificateur existant,
sans déduire leur validité du fait qu'ils viennent du prompt. Ce principe de
prompt lookup est déjà connu [5].

Les nouveaux tests ont révélé que, lorsqu'un EOS proposé était accepté, le
cache conservait exceptionnellement le dernier token, contrairement à l'invariant
documenté des autres arrêts. Le rollback a été corrigé. Le rapport des deux
échecs initiaux est conservé dans `tests-before-eos-fix.xml`.

Les contrôles comprennent tous les bits 16 bits, sorties d'experts réels,
lectures partielles 3D, concurrence, corruption, troncature, bombes de
décompression, budgets, compteurs, propositions erronées, EOS, longueurs limites
et identités probabilistes rationnelles. La suite MLX historique complète
ne fait pas partie des **97 contrôles réussis** : 56 nouveaux tests et 41 tests
existants du vérificateur exécutés sans leur conftest de framework. Cette suite
ne peut pas être collectée avec les dépendances actuelles ; son blocage est
enregistré et n'est pas présenté comme une suite passée.

Lors de la reprise, les **mêmes 97 tests ont été réexécutés avec succès** dans
deux rapports JUnit séparés. Ils restent 97 tests uniques, pas 194. Les 48
comparaisons de l'expérience PyTorch sont rapportées comme observations
expérimentales, sans les ajouter artificiellement à ce décompte.

## 10. Lecture des simulations

`scenarios.json` garde toutes les profondeurs, y compris défavorables. Les
paramètres alpha, débit 3 Go/s, compression à 0,75 et résidence compacte de
10 Go sont **des hypothèses**. La résidence compacte n'est pas le cache décodé
actuel. Le coût du brouillon, du décodage, du calcul et les allocations est
volontairement exclu : les valeurs sont des coûts de transfert, pas des temps
d'inférence prévus.

Dans le modèle uniforme indépendant, avec alpha = 0,8, sans compression et
10 Go de résidence compacte supposée, la meilleure profondeur explorée est
K=1 pour GLM et K=2 pour Qwen : réductions OTA d'environ 1,40× et 1,28× contre
le cas naïf sans cache. À alpha = 0,95, elles deviennent environ 1,63× et 1,61×.
Une part de ces gains vient déjà du cache. Ces valeurs ne justifient pas 10×.

Avec alpha = 1, K=512 donne de grands facteurs théoriques, mais présuppose
513 positions correctes/émises et des passes de plusieurs minutes. Le moteur
actuel découpe en `prefill_chunk` : une union de 513 positions ne serait même
pas réalisée avec le chunk par défaut de 16. Les grands facteurs du scénario
ne sont donc ni une mesure, ni une configuration actuellement validée.

## 11. Position par rapport aux travaux existants

La vérification spéculative exacte est établie [4]. Les travaux sur prefetch,
cache et ordonnanceurs MoE traitent déjà la dépendance au routage [6–7]. Ils
ne permettent pas de transférer des résultats OLMoE vers GLM sans mesurer.

SSD-LLaMA regroupe les experts, utilise une hiérarchie et une décompression
sans perte [8]. Ses expériences utilisent notamment CUDA, une RTX 5090 et des
formats quantifiés ; conserver les bits d'un Q4 ne signifie pas conserver ceux
du BF16 d'origine. Ces résultats ne se transposent pas au M5/SSD externe.

ZipServ et ECF8 articulent codecs et kernels [9–10]. Ils confortent l'intérêt
de supprimer les copies et de mesurer le décodage à proximité du calcul.
Le travail ANS de 2026 étudie également la décompression par tuiles [11]. Ses
chiffres de compression et de débit restent des résultats d'autres formats,
modèles et matériels. Aucun de ces papiers ne prouve notre cible sur 16 Go.

Notre contribution vérifiée est le protocole de reprise, la correction des
affirmations trop fortes, les replays reproductibles, le prototype de transport
exact intégré et ses résultats positifs **et négatifs** sur M5 CPU. Nous ne
revendiquons pas une nouvelle famille d'algorithmes qui dépasserait la littérature.

## 12. Validation finale sur M5 et ordre des recherches suivantes

1. **Rétablir un environnement MLX visible.** La référence PyTorch CPU fonctionne
   désormais dans un environnement existant. Il reste à obtenir un import MLX
   et une sonde Metal positifs dans l'environnement
   autorisé. Exécuter les suites natives historiques, sans les remplacer par
   les seuls tests de transport. Garder configurations et versions exactes.

2. **Fixer le contrat de référence.** GLM : choisir explicitement la sémantique
   FP8 du serveur de référence ou celle des poids déquantifiés. Qwen : reproduire
   la porte officielle, les récurrences et convolutions avant d'étudier le MTP.
   Comparer logits, routes et états pour des batchs/longueurs différents ; tester
   les quasi-ex æquo et zéros signés. Tout écart doit être rapporté, pas absorbé
   silencieusement dans une tolérance présentée comme bit-exacte. L'expérience
   PyTorch a déjà produit un contre-exemple FP32 sur de vrais experts ; un mode
   strict devra préserver l'ordre effectif ou fournir un autre certificat.

3. **Mesurer le vrai SSD externe.** Lectures seules, tailles de blocs pertinentes,
   accès aléatoires d'experts et accès séquentiels, profondeur 1/2/4, cache chaud
   et scénario froid documenté. Compter les surlectures, métadonnées et pages.
   Ne pas écrire des centaines de Go pour fabriquer un benchmark. Enregistrer
   débit soutenu, p50/p95, durée, température si accessible et limitations des
   compteurs physiques.

4. **Échantillonner les poids des cibles avec un plafond explicite.** De petites
   plages FP8 et BF16, plusieurs couches/expert/projections, révisions épinglées.
   Tester codecs et décodeurs sur ces données avant toute extrapolation. Un
   format autonome compact devra être développé séparément pour le stockage.

5. **Capturer des générations libres et des brouillons.** Même tokenizer,
   prompts français/anglais, code, maths, JSON, copie et conversations. Conserver
   pour chaque passe les routes **des tokens proposés**, les acceptations,
   corrections, KV/états et durées. Les données teacher-forced actuelles servent
   à préparer le protocole, pas à remplacer cette expérience.

6. **Optimiser par mesures, à budget commun.** Comparer cache décodé/encodé,
   politiques anti-scan, blocs natifs, pipeline à deux workers et spans de prompt.
   Inclure le brouillon dans RAM, calcul et OTA. Explorer des vérifications
   moyennes/longues seulement lorsque l'acceptation et la latence du bloc le
   justifient. Un cache froid et un brouillon inutile doivent être dans le jeu.

7. **Valider le checkpoint complet seulement ensuite.** Aucun téléchargement
   massif n'est lancé par ce dossier. La capacité BF16 de Qwen doit d'abord avoir
   une solution fidèle vérifiée. Sur le matériel final, mesurer temps au premier
   token, temps total, débit p50/p95, pics RSS/MLX, swap, énergie si accessible,
   30 minutes de charge et parité des sorties. C'est seulement cette expérience
   qui pourra soutenir une annonce « utilisable » ou « 10× ».

La barrière actuellement établie est concrète : le checkpoint BF16 brut dépasse
le disque, la partie active dépasse la RAM, la lecture par token reste énorme,
et les kernels/acceptations de la cible ne sont pas validés. Nous n'avons pas
démontré une impossibilité générale de contourner ces contraintes, et nous
n'avons pas démontré qu'une méthode disponible les contourne toutes.

## Sources primaires consultées

1. [Configuration GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json).
2. [Configuration Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/resolve/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json).
3. [Index Qwen BF16](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/model.safetensors.index.json).
4. [Leviathan et al., Fast Inference from Transformers via Speculative Decoding, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html).
5. [Hugging Face, Assisted decoding / prompt lookup](https://huggingface.co/docs/transformers/assisted_decoding).
6. [Apple, SpecMD, 2026](https://machinelearning.apple.com/research/specmd-expert-prefetching).
7. [Wang et al., SpecMoEOff](https://arxiv.org/abs/2508.21706).
8. [Liang et al., SSD-LLaMA, 2026](https://arxiv.org/html/2609.18110v1).
9. [Fan et al., ZipServ, ASPLOS 2026](https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf).
10. [Yang et al., ECF8, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/57c3fef21638eb9d1d67d32ba0a6d1af-Abstract-Conference.html).
11. [Tan et al., Approaching Shannon Bound with Lossless LLM Weight Compression, 2026](https://arxiv.org/abs/2606.15789).
12. [Apple, LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language).

Les faits numériques de nos expériences proviennent des JSON/NPZ/JUnit du
dossier `results/`. Les résultats publiés par les auteurs restent distincts de
nos mesures. Les fichiers originaux de Claude restent distincts des résultats
recalculés pendant cette reprise.
