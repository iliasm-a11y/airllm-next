# RONIN — reprise mémoire, CPU32 et kernels quantifiés natifs

4 novembre 2026, Europe/Paris. Branche `research/glm53-native-dispatch-m5`,
PR #25 en brouillon, base #24 `6cfa76930626520c15246286b3baa2a58c0cf952`.

**Le blocage d’exécution est levé. CPU32 termine sous garde et les kernels
Metal natifs Q2_K/Q8_0 sont réellement exécutés. Aucun gain GPU complet n’est
qualifié : les chemins natifs testés restent plus lents que CPU10, et le Metal
standard échoue aux tolérances de composant.** Production et `main` inchangés.

## Résultats contemporains

| Mesure | Résultat |
|---|---|
| CPU10/GPU0, 32 sorties | Terminé, sortie native 0 |
| Décodage | 31 appels en 191,954455206 s ; **0,161496642 intervalle/s** |
| Dénominateur | Préfill/première sortie et gestion extérieure exclus |
| CPU dans ces appels | 1 477,241770 secondes-cœur |
| RSS maximal surveillé | 7 350 779 904 octets, soit 6,85 Gio |
| Swap global | Départ 372 572 160 octets ; croissance observée **0** |
| Pression mémoire maximale | **1**, normale |
| IDs face à la capture B4 de PR #24 | Les 32 IDs sont identiques |
| Logits/routes/états dans ce nouveau témoin | Non capturés ; aucune nouvelle fidélité complète déclarée |
| EOS naturel / réponse utile RONIN | Non testés par ce prompt brut court de 32 sorties |

L’essai supervisé complet dure 205,894986 s. Le chargement des métadonnées
mappées est enregistré séparément ; ce temps court n’est pas la lecture
intégrale ni la mise en résidence des 110 Go. Les lectures OS comptées dans
les 31 appels atteignent 379 851 124 736 octets : ce ne sont pas des lectures
physiques NAND. Les états et compteurs détaillés restent dans les
[preuves de reprise](evidence/recovery-20261104/selection.json).

Le 0,164819 intervalle/s de PR #24 reste historique. La nouvelle valeur est
légèrement inférieure ; l’ordre, la résidence de fichiers et l’activité du
système ne sont pas contrôlés en paire. Aucune accélération n’est attribuée
au nettoyage par comparaison de ces deux débits.

## Verrou et nettoyage mémoire réellement effectués

L’ancien détenteur observé avant cette reprise était un travail
`native-prefill-final-controls` avec son superviseur Python. Lors du contrôle
actuel, il avait terminé : les deux verrous sont acquis sans contention par
nos sondes. Aucun ancien PID n’est arrêté sur la seule base d’un relevé passé,
et aucun fichier de verrou n’est supprimé.

Les processus Node Firebase/Playwright ont été sélectionnés par commande,
UID, exécutable, naissance, parent et ancienneté supérieure à 30 minutes.
Trois observations sur 30 s confirment compteurs CPU et I/O inchangés, aucune
connexion réseau et aucun enfant navigateur. Les identités et l’inactivité
sont revérifiées immédiatement avant les signaux.

- **8 travailleurs** arrêtés par SIGTERM ; leurs **8 lanceurs** se ferment
  normalement. Aucun survivant et aucun SIGKILL.
- Empreinte comptabilisée des paires : **1 164 662 776 octets**. Ce chiffre
  n’est pas autant de RAM physique libérée.
- Mémoire disponible juste avant/après les signaux : 6 471 909 376 →
  7 057 227 776 octets, soit un delta global de **585 318 400 octets**.
- Une partie de la marge était déjà revenue avant le nettoyage ; la hausse
  depuis les anciens relevés autour de 4,4 Gio n’est pas attribuée entièrement
  à ces signaux.

Les applications, VM, fichiers et outils récents/connectés sont préservés.
Aucun purge, changement de swap ou réglage permanent de veille. Les
inventaires détaillés de processus restent privés ; la sélection publique
contient les agrégats utiles à la recherche.

Une fenêtre de 30 s, sept observations, confirme les réserves et l’alimentation
sur secteur avant la relance. Une erreur de chemin Python dans le premier
lanceur est conservée : elle s’arrête avant le modèle. Le lanceur corrigé
emploie un chemin absolu vers les sources. Le worker inchangé acquiert ensuite
son propre verrou UID ; une prise concurrente aurait provoqué un refus.
Notre CPU32 est le PID 80732, naissances et exécutable suivis, puis fermé
normalement. Le budget de swap conserve le même départ que la campagne
initiale, sans réinitialiser artificiellement le compteur.

## Dispatch et arithmétique : conclusions vérifiées

L’[audit](evidence/audit/audit.json) lit la révision llama.cpp
`3f8af15a76ac8675812e856da45454bafc90f12c` avec `git show` et vérifie les
13 sources V4 restaurées contre leur manifeste. Les [extraits avec lignes](evidence/audit/source-excerpts.txt)
distinguent amont, politique V4 et intégration AirLLM.

Le GPU0 de production impose `--device none --no-op-offload`. Les buffers CPU
ne deviennent pas Metal par la seule présence de mémoire unifiée. Le seuil
opportuniste natif est 32 colonnes ; le décodage à une colonne est inférieur.
Les branches de dispatch ont des contraintes de forme, strides, alignements
et capacités. Le placement sélectif V4 existe déjà ; il conserve sur CPU les
normalisations, mHC, KDA, DSA/indexer et états hors projections admises.

Les kernels Q2_K et Q8_0, leurs variantes à IDs et des chemins matriciels
existent. Dans les essais mono-colonne, les journaux de compilation et les
sorties confirment la réutilisation de **`kernel_mul_mv_q2_K_f32`** et
**`kernel_mul_mv_q8_0_f32`**. Aucun nouveau kernel quantifié n’est écrit.

Le Metal standard reçoit l’activation FP32. Le CPU utilise temporairement
Q8_K pour Q2_K, Q8_0 pour Q8_0, puis son accumulation ARM. Le banc mesure
maintenant l’effet concret de cette différence. L’adaptateur ne déquantifie
qu’un petit vecteur d’activation après quantification CPU ; les poids
originaux restent intacts, sans matrice FP32 de poids intermédiaire.

L’inventaire dense historique compte 305 matrices Q8_0 (8 278 966 272 octets),
225 BF16 (796 917 760) et 154 F32 (368 574 464), soit 9 444 458 496 octets
logiques par étape. Les 126 banques Q2_K sont des experts routés. Les
12,219 Go logiques/token et les fractions de temps du profil #24 ont des
périmètres différents et ne sont pas des mesures NAND.

## Comparaison des composants, coûts complets

Poids GGUF originaux, activations FP32 synthétiques explicitement identifiées,
deux graines par forme, 21 répétitions et trois chauffes. CPU1, CPU10 et le
dot natif indépendant sont conformes. Un seul pool CPU persistant est actif
à la fois ; les handles CPU sont fermés avant Metal. Les modes Metal sont
alternés aléatoirement, dans une phase distincte des mesures CPU et du timer
historique : ce n’est pas un A/B/B/A de modèle complet.

| Forme / graine | CPU10 ms | Metal natif ms | Avec adaptateur ms | Historique tile1 ms |
|---|---:|---:|---:|---:|
| Q2_K, 2048×4096, 5304101 | 0,047666 | 0,306125 | 0,295292 | 0,694375 |
| Q2_K, 2048×4096, 5304102 | 0,044792 | 0,311167 | 0,297458 | 0,750666 |
| Q8_0, 1536×4096, 5304101 | 0,033333 | 0,306542 | 0,303625 | 0,558708 |
| Q8_0, 1536×4096, 5304102 | 0,032667 | 0,293875 | 0,304416 | 0,570583 |

Ces temps muraux incluent préparation/import de l’activation, lancement,
attente terminale réelle et lecture de sortie pour les backends natifs.
L’initialisation et l’upload initial sont séparés. Le timer historique inclut
sa quantification, lancement et attente ; sa lecture de vérification de
sortie est extérieure au temps annoncé, ce qui ne lui crée pas un avantage
retenu puisqu’il reste le plus lent. Tous ces poids sont chauds et résidents.
Aucun temps de composant n’est converti en débit sur le checkpoint complet.

| Chemin | Fidélité de composant observée |
|---|---|
| CPU10 / CPU1 / dot indépendant | Conforme bit à bit sur les références vérifiées |
| Metal natif standard | Échec des tolérances dans les quatre cas ; max absolu 0,017114 à 0,032813 |
| Adaptateur d’activation | Tolérances respectées dans toutes les répétitions ; max absolu ≤1,1921e-6 ; pas bit à bit |
| Kernel personnalisé historique | Bit à bit sur les 7 168 sorties finales des quatre cas |

Les critères restent `atol=1e-3, rtol=1e-4` ; aucune tolérance n’est élargie.
L’adaptateur ne constitue pas une preuve de routes ordonnées ou états GLM
fidèles. Son temps total est **6,20 à 9,32 fois** le temps CPU10 sur ces cas ;
le standard est 6,42 à 9,20 fois plus lent et échoue aussi numériquement.
Aucun de ces deux candidats ne justifie son activation dans le modèle.

Une première série sans pool CPU explicitement attaché recréait les threads
à chaque appel : ses temps CPU sont exclus de cette table. Les données et
empreintes initiales restent conservées. Le correctif compile un nouvel
adaptateur, puis les quatre cas repassent leur propre comparaison sous son
nouveau hash. Aucune qualification n’est transférée depuis l’ancien binaire.
Le binaire B4 du CPU32 complet et les bibliothèques V4 restent inchangés.

Chaque essai final conserve 96 buffers de commandes Metal terminés (statut 4),
pour 48 appels des deux modes, chauffes comprises. Les buffers incluent des
commandes vides : ils ne sont pas comptés comme 96 matvecs. Le code natif
emploie un encodeur supplémentaire ; les journaux montrent un buffer de
calcul et un buffer presque vide par appel. Les temps GPU bruts sont conservés
sans fabriquer de médiane par mode. Le coût total et les attentes hôte sont
nettement supérieurs au seul calcul sur ces petits composants ; une politique
à un encodeur reste une autre hypothèse, non testée ni qualifiée ici.

## Cache, pipeline et configuration retenue

La référence reste CPU10/GPU0, cache GPU/expert 0, un moteur/slot, mmap à la
demande et les préchargements bornés denses/experts déjà présents. Aucune
nouvelle optimisation GPU/cache/streaming n’est activée. Les 600 identités
logiques et l’isolation des contextes sont préservées.

Aucune nouvelle mesure de réutilisation entre agents 1 et 600 n’est produite.
Le rejeu historique 128 Mio de #19 garde 0/1008 hits de décodage LRU et
15/1008 SLRU ; les 704 hits LRU agrégés appartiennent au préfill. Ces résultats
ne démontrent ni gain net ni réutilisation inter-agents. La borne 1,293774×
issue des volumes suppose un même coût par octet logique ; elle ne borne pas
universellement les lectures physiques ou le temps complet.

Le pipeline complet SSD/CPU/GPU reste différé. Les préchargements précédents
ne sont pas une piste entièrement nouvelle. Aucun overlap réel nouveau,
double buffering, batching de deux séquences ou débit de 600 agents n’est
qualifié. CPU32 est un essai complet borné ; 64/128 et EOS restent à qualifier.
Aucun disque externe physique n’était disponible ; aucun résultat Fanxiang.

## Douze verdicts

1. **Dispatch :** kernels disponibles, mais offload explicitement désactivé,
   buffers/placement contraints et arithmétique d’activation différente.
2. **Accélération native réelle :** les composants sont maintenant mesurés,
   mais plus lents que CPU10 ; aucun gain complet retenu.
3. **Meilleur format/kernel :** CPU natif ARM gagne sur les deux formes
   testées. Les formats Q2_K/Q8_0 ne sont pas classés globalement entre eux,
   leurs formes et rôles étant différents ; aucune requantification.
4. **Réutilisation inter-agents :** aucune nouvelle mesure, valeur inconnue.
5. **Gain cache MoE :** non établi, activation différée.
6. **Pipeline masquant des lectures :** non mesuré, non activé.
7. **Combinaison :** CPU10/GPU0 reste la référence, préchargements existants
   inchangés ; aucune nouvelle combinaison qualifiée.
8. **Débit contemporain :** 0,161496642 intervalle/s sur 31 appels pour
   32 IDs ; préfill exclu, aucun EOS ni débit de tokens utiles RONIN annoncé.
9. **Fidélité :** standard Metal rejeté sur composants ; adaptateur toléré
   mais non exact et non intégré. Les 32 IDs CPU correspondent à B4 ;
   logits/routes/états non recapturés dans cette mesure de temps.
10. **Mémoire/protections :** CPU32 et quatre composants terminent sous les
    gardes inchangées, swap sans croissance. Cela ne prouve pas la stabilité
    prolongée des générations ni une qualification générale sur 16 Gio.
11. **Goulot suivant :** attente/encodage dominants sur les composants Metal
    testés ; goulot du modèle futur inconnu. Le profil historique conserve
    66,4 % dense et 18,8 % MoE, calcul/pagination/I/O mêlés.
12. **Pilote RONIN :** aucun nouveau chemin GPU prêt. Le témoin CPU32 fonctionne
    mais ne qualifie pas les réponses utiles/EOS ni la charge des 600 agents.

Les garde-fous restent RSS 8,5 Gio, croissance globale du swap 32 Mio,
admission modèle 5,5 Gio, réserve disque 32 Gio, pression normale, checkpoint
et artefacts vérifiés, secteur/capot/horloges, PID détenus et verrou unique.
Aucun seuil n’est abaissé. Poids et quantification inchangés ; aucun secret,
poids ou inventaire personnel publié ; aucune fusion/main ni activation de
production. L’[état initial](RAPPORT-INITIAL-FR.md), ses refus et les résultats
négatifs restent disponibles séparément.
