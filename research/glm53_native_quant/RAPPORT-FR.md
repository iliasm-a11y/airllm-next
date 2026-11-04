# RONIN — audit et banc de dispatch quantifié natif sur Apple M5

Date : 4 novembre 2026, Europe/Paris. Branche expérimentale distincte :
`research/glm53-native-dispatch-m5`. Base vérifiée : PR #24 à
`6cfa76930626520c15246286b3baa2a58c0cf952` ; aucune fusion dans `main`.

**Résultat : aucun nouveau moteur accéléré qualifié.** L'audit des sources
est terminé, un adaptateur minimal vers GGML CPU/Metal est compilé et les
contrôles portables sont testés. L'exécution native reste bloquée par le
verrou d'un autre travail GLM ; le modèle complet est également refusé par
la réserve RAM. Les phases cache et pipeline sont différées conformément à
l'ordre demandé. Ce rapport ne transforme pas ces refus en résultats négatifs
de performance des kernels.

## 1. Périmètre vérifié et référence

Les PR #17 à #24 ont été consultées avec le connecteur GitHub ; leurs versions
et descriptions sont figées dans [pr-audit.json](evidence/pr-audit.json).
PR #23 a progressé jusqu'à `4d07956` après la base `6bdeec3` utilisée par
PR #24. Cette branche conserve exactement la base #24 demandée : elle ne
fusionne pas silencieusement les changements ultérieurs de #23.

Le Mac accessible expose Apple M5, 10 CPU et 17 179 869 184 octets de RAM.
La mission spécifie GPU 8 cœurs ; l'inventaire enregistré confirme le GPU
Apple M5 sans extraire un nouveau décompte de ses cœurs. Aucun disque
externe physique n'a été retourné par `diskutil list external physical`.
Aucun essai Fanxiang ni résultat de stockage externe n'est annoncé.

Un premier lancement CPU10/GPU0, 32 sorties demandées, s'arrête **à
l'admission**, sans processus natif. Pression 2, 4 073 013 248 octets
disponibles, intégrité non reconnue par le reçu ancien. Le reçu différait
du montage actuel et n'a pas été réécrit pour forcer son acceptation.

Un reçu actuel existant a ensuite été retrouvé et vérifié avec le code natif
inchangé : architecture, hashes attendus et cinq champs d'identité sont
acceptés pour les deux shards. Il s'agit d'une réutilisation d'un reçu SHA
existant, **pas d'un nouveau scan intégral** du checkpoint. La deuxième
admission CPU32 échoue uniquement sur la RAM : 3 997 515 776 octets
disponibles, contre 5 905 580 032 requis. Pression 1, 36 208 267 264 octets
libres sur disque et 372 572 160 octets de swap déjà utilisés. Un swap déjà
présent n'est pas assimilé à une croissance causée par cette mission.

Le chiffre **0,164819 intervalle de décodage/s** reste celui de PR #24 :
31 appels terminés en 188,084548 s pour 32 IDs émis. Chargement, préfill,
première sortie et gestion extérieure sont exclus. Il ne devient ni une
référence contemporaine de cette branche, ni un débit de tokens utiles RONIN.

## 2. Pourquoi le Metal natif n'est pas utilisé efficacement

L'[audit reproductible](evidence/audit/audit.json) lit avec `git show` la
révision llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`. Il vérifie
séparément les 13 fichiers V4 restaurés : tous correspondent à leur manifeste
historique. Les [extraits](evidence/audit/source-excerpts.txt) distinguent
précisément source amont, politique V4 et intégration AirLLM.

1. **Désactivation explicite.** La commande CPU de production ajoute
   `--device none --no-op-offload` à GPU0. Ce choix conserve la référence
   fidèle ; il ne provient pas d'une bibliothèque Metal absente.
2. **Placement et buffers.** Le backend ne traite pas un buffer CPU comme un
   buffer Metal parce que la mémoire physique est unifiée. Le scheduler doit
   disposer d'un buffer compatible, respecter les dépendances et effectuer
   les copies ou alias requis. L'offload opportuniste utilise par défaut un
   seuil de 32 colonnes ; le décodage mono-colonne et microbatch 8 sont en
   dessous. Abaisser ce seuil seul ne prouve ni fidélité ni rentabilité.
3. **Support des formats établi.** Les kernels
   `kernel_mul_mv_q2_K_f32`, `kernel_mul_mv_q8_0_f32`, leurs variantes
   `mul_mv_id` et des chemins matriciels sont présents. Les branches MMA,
   petit batch et matvec dépendent de N, K, strides, alignements et capacités.
   Le graphe mono-colonne du banc vise la branche matvec, pas un kernel Q4/Q6
   employé à tort sur des octets Q2.
4. **Contrat arithmétique différent.** Le kernel Metal standard reçoit un
   vecteur FP32. Le CPU Q2_K quantifie temporairement ce vecteur en Q8_K,
   puis applique `ggml_vec_dot_q2_K_q8_K`. Le CPU Q8_0 utilise Q8_0 temporaire.
   Quantification d'activation et ordre de réduction diffèrent : compatibilité
   du format de poids n'implique pas conformité aux critères du modèle.
5. **Opérations GLM et synchronisations.** mHC, KDA, DSA/indexer et états
   introduisent des dépendances entre projections, routage et écritures.
   L'amont possède certains kernels mHC/GATED_DELTA_NET sous contraintes ;
   leur présence ne qualifie pas leur arithmétique face au CPU. Les anciens
   essais avaient notamment changé la fusion mHC par le placement global.
   V4 fournit déjà un placement sélectif gardant ces calculs CPU et n'admettant
   que certains matvecs Q8 mono-colonne sur des poids Metal. Le reproduire
   inutilement ou activer tout le graphe n'est pas une correction justifiée.

L'inventaire historique distingue les codecs réels :

| Poids denses | Matrices | Octets logiques par étape |
|---|---:|---:|
| Q8_0 | 305 | 8 278 966 272 |
| BF16 | 225 | 796 917 760 |
| F32 | 154 | 368 574 464 |
| Total dense | 684 | 9 444 458 496 |

Les 126 banques Q2_K sont toutes des experts routés. Le nom global Q2_K_S
n'implique pas que les grandes matrices denses soient Q2_K. Les
12 218 990 592 octets d'opérandes logiques par token et les pourcentages de
temps PR #24 proviennent de périmètres différents. Ni ces volumes ni les
compteurs OS ne sont des lectures physiques NAND.

## 3. Implémentation minimale et coût à vérifier

`native.cpp` utilise les backends GGML existants. Les poids originaux restent
quantifiés ; seule une activation de K éléments peut être quantifiée puis
déquantifiée pour l'adaptateur. L'objectif est de tester si cette petite
adaptation suffit à rendre le kernel natif admissible, sans écrire un autre
kernel. **Elle ne reproduit pas par construction l'ordre ARM**, et n'est
branchée ni dans GLM ni dans RONIN tant que sa fidélité et son intérêt ne sont
pas mesurés.

Le banc compare sur les mêmes données CPU1, CPU10, Metal standard, Metal avec
adaptateur et `tile1` historique PR #24. Les formes prévues sont un expert
gate Q2 de 2 048 × 4 096 (2 752 512 octets) et une projection Q8 de
1 536 × 4 096 (6 684 672 octets). Les activations par défaut sont synthétiques ;
la validation avec des activations GLM représentatives reste à effectuer.

La compilation réelle de l'adaptateur réussit sous garde en 1,217 s environ,
avec un pic RSS échantillonné de 135 200 768 octets et zéro croissance de swap.
Les bibliothèques V4 ne sont pas reconstruites. Le reçu de compilation garde
les hashes des sources, headers, bibliothèques et du nouvel exécutable
dynamique ; cette compilation n'est aucune qualification GPU ou GLM.

L'essai Q2 commence sa supervision, puis le verrou UID occupé le refuse avant
tout calcul natif ou lecture de tranche de poids. L'observation limitée au
fichier de verrou montre Python et `native-prefill-final-controls`, appartenant
à un autre travail. Aucun signal ne leur est envoyé. Le banc est ensuite
corrigé pour préserver ce refus avant les imports de tableaux et pour traiter
également la course entre préobservation et acquisition. L'admission suivante
refuse proprement `ENGINE_BUSY`, sans worker. **Aucun des nouveaux kernels
visés n'a donc été exécuté par cette mission.**

Les critères existants sont conservés : composantes Metal `atol=1e-3`,
`rtol=1e-4`, comparaison bit à bit rapportée séparément ; la future comparaison
de modèle devra garder routes ordonnées exactes, toutes les familles de logits,
routeurs, KDA et écritures observées. Les temps du banc sont synchronisés,
préparation/lancement/attente inclus ; upload et initialisation sont séparés.
Aucun de ces temps n'a encore été mesuré ici.

## 4. Cache MoE et pipeline : état des preuves

La phase 1 n'ayant pas de conclusion expérimentale complète, aucune nouvelle
campagne de cache inter-agents n'est engagée. Aucun hit entre agents 1 et 600
n'est mesuré ; cette valeur est **inconnue**, pas zéro. Aucun état privé n'est
partagé et aucun cache n'est installé.

L'audit conserve les résultats historiques défavorables de #19. Dans le
rejeu de métadonnées 128 Mio, LRU obtient 704 hits au total, tous au préfill,
et 0/1 008 en décodage ; SLRU obtient 15/1 008 en décodage. Il ne s'agit ni
d'un test de 600 agents ni d'un gain net. Sous coût uniforme par octet logique,
un cache MoE parfait donnerait au plus 1,293774× sur la composante I/O de cet
inventaire. Cette hypothèse ne borne pas universellement le temps complet :
une résidence OS différente entre denses et experts change les coûts.

Les préchargements bornés et lectures d'experts déjà sélectionnés de #17 à
#21 existent déjà. Aucun pipeline complet SSD/CPU/GPU avec gain soutenu n'est
qualifié par cette mission. Les buffers doubles, nouvelles lectures
spéculatives et batching inter-agents ne sont pas engagés avant la stabilité
mémoire et les phases précédentes. Aucun recouvrement I/O/calcul nouveau
n'est mesuré.

## 5. Comparaison et protections

| Configuration demandée | État contemporain |
|---|---|
| A — CPU10/GPU0 | Admission CPU32 refusée sur RAM ; aucun débit nouveau |
| B — natif quantifié | Adaptateur compilé ; essai de composant refusé sur verrou |
| C — natif + cache MoE | Différé derrière B ; aucun benchmark |
| D — natif + cache + pipeline | Différé derrière B/C et mémoire ; aucun benchmark |

Il n'y a pas d'essai A/B/B/A accéléré, de 64/128 sorties, d'EOS naturel ou
de mesure agrégée de tokens utiles de cette branche. Les anciennes mesures
CPU32 et les anciens arrêts CPU64 sont préservés dans la base #24 sans être
réattribués au nouvel adaptateur.

Les plafonds restent RSS 8,5 Gio et croissance globale du swap 32 Mio ; le
modèle exige 5,5 Gio disponibles et 32 Gio libres. Le composant utilise le
profil antérieur 3 Gio de réserve système + 256 Mio de payload. Ses poids
sélectionnés sont limités à 16 Mio, les buffers GGML à 32 Mio après comptage,
et le sampler englobe les processus détenus. Secteur, capot, horloges,
assertion temporaire anti-veille, identités de PID, verrou unique et nettoyage
sûr sont conservés. Un baseline commun maintient le budget cumulatif de swap.
Aucun seuil, reçu ou réglage de veille n'a été altéré pour forcer une mesure.

Les 600 identités logiques et un seul moteur/slot sont conservés. Aucun poids,
secret ou fichier personnel n'est publié ; aucune application personnelle,
VM, backend installé ou configuration de production n'est modifié.

## 6. Verdicts obligatoires

1. **Pourquoi le Metal natif était-il peu utilisé ?** Désactivation explicite
   de l'offload, contraintes de placement/buffers et seuil mono-colonne, puis
   différences d'arithmétique et dépendances GLM ; les kernels sont présents.
2. **Accélération réelle par réutilisation ?** Non démontrée. L'exécution du
   nouveau banc est refusée par le verrou ; aucune comparaison complète.
3. **Format/kernel le plus performant ?** Inconnu pour le nouveau chemin.
   Les denses sont surtout Q8_0, les experts Q2_K. Aucun gagnant GPU nouveau
   n'est déclaré à partir de la présence d'un kernel ou de mesures anciennes.
4. **Réutilisation inter-agents mesurée ?** Aucune nouvelle mesure ; agents
   1/600 non exécutés ici. Le rejeu historique intra-trajectoire reste distinct.
5. **Gain net du cache MoE ?** Non établi. Activation différée.
6. **Lectures masquées par le pipeline ?** Non mesurées. Pipeline différé.
7. **Meilleure combinaison ?** CPU10/GPU0 reste la référence et le secours ;
   aucune nouvelle combinaison GPU/cache/streaming n'est qualifiée.
8. **Débit soutenu obtenu ?** Aucun nouveau débit. L'historique #24 est
   0,164819 intervalle/s sur 31 appels, sans réponse utile RONIN mesurée ici.
9. **Fidélité préservée ?** Production inchangée ; fidélité du nouvel
   adaptateur et modèle intégré encore inconnue. Les 64 tests portables ne
   remplacent pas une comparaison réelle de logits/routes/états.
10. **16 Gio et protections respectés ?** Les gardes ont refusé les essais
    inadmissibles et la compilation est restée sous garde. Aucune stabilité
    d'inférence accélérée sur 16 Gio n'est prouvée.
11. **Prochain goulot après optimisation ?** Non déterminable : aucune
    optimisation n'est encore appliquée. Le profil historique #24 attribue
    66,4 % du temps d'opérations aux produits denses et 18,8 % au MoE,
    avec calcul/pagination/I/O mêlés ; il ne donne pas le goulot futur.
12. **Pilote RONIN ?** Ce nouveau chemin n'est pas prêt pour un pilote.
    Il faut d'abord un créneau exclusif et une admission RAM valide, exécuter
    les composants, puis la fidélité et les débits complets avant les autres
    phases. Aucune activation automatique n'est effectuée.

## 7. Livraison et limites restantes

Les sources, l'audit, les reçus d'admission, empreintes de compilation,
résultats de tests et résultats négatifs sélectionnés sont livrés avec une
nouvelle PR en brouillon empilée sur #24. Le workflow portable ne prétend pas
qualifier du matériel M5. Les refus d'admission sont des preuves de protection,
pas des preuves d'accélération ni des contre-exemples de fidélité.

Optimisation conservée : aucune nouvelle activation, référence CPU10/GPU0.
Pistes en attente : Metal natif/adaptateur, cache partagé de poids, pipeline.
Voies exclues de toute promotion immédiate : offload global sans fidélité,
assimilation d'un microbenchmark chaud à un gain sur 110 Go, relâchement de
garde ou transfert automatique de qualification à un binaire reconstruit.
