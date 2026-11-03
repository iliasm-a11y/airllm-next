# RONIN M5 — résultats mesurés et qualification en cours

État du 2026-11-03T16:43:14.321334+00:00. Les objectifs A (préfill GPU complet plus rapide), B (cache exact avec gain net) et C (deux agents utiles jusqu’à EOS) restent **non qualifiés**. Le CPU10/GPU0 et le slot unique de production sont conservés.

## Résultats clos sur le Mac

| Essai | Mesure et verdict |
|---|---|
| Agent1 RONIN CPU8, Max | Prompt complet522 tokens, préfill1238,015911s,171 IDs de sortie puis délai2700s. Pas de réponse finale utile ni EOS. Agent600 et retour1 non lancés. |
| Témoin natif CPU8 | Trois prompts gelés521/542/521 tokens, préfills1285,763678 /1322,539045 /1314,183277s. Un modèle/slot, quatre vrais échantillons greedy diagnostiques par requête, code0. Ce diagnostic ne qualifie pas les tâches fonctionnelles. |
| Rejeu CPU8 | Les154880 logits finis, états sérialisés, routes/routeurs/KDA, écritures observées, continuations et post-état sont exacts pour request0→2. Il s’agit d’un rejeu dans le témoin. |
| CPU16 original | Premier préfill820,050836167s, mais154880 logits différents, maxabs1,61790836 et état final différent. Première différence capturée token512/KDA0. Annulation contrôlée, native−15. Aucun gain retenu. |
| CPU16 avec queue corrigée | Deux requêtes terminées en823,460549791 /876,08467325s, logits/états/continuations bruts exacts. Arrêt pendant retour1 à2231,936556s : croissance globale du swap216727552B (206,6875Mio)>32Mio. Bras incomplet non qualifié, native−15. |
| CPU32 avec frontières absolues | Nouveau helper réellement compilé et lié au témoin. La fenêtre complète128 tokens diverge dans les118 séries capturées. KDA0/token0 :8192 valeurs différentes, maxabs6,379559636e−6. Annulation contrôlée à≈278,4s ; aucun prompt terminé, swap observé0. Ni batch32 ni cache de ce profil qualifiés. |
| Préfixe commun | 600 identités avec template natif et Max ;211 IDs communs, frontière208. Trois sérialisations complètes à208 identiques :156739588B (149,478519Mio), SHA256 `4c4d6f24b2cf3c4f30a7894b8609035960a3ec5687521c942aa6a578e2e5a3cd`. |
| Restore8 | Préfixe208 calculé une fois en449,387450125s. Première requête restaurée exacte sur logits/état/continuations. Arrêt à1955,907108334s sur41025536B de croissance globale de swap (39,125Mio)>32Mio, avant la fin du deuxième prompt. Bras incomplet, cache non qualifié. |

Le CPU32 atteint les frontières128/208/256/384/512 et préserve les petites queues CPU8. Cela corrige la géométrie du protocole32+, mais pas sa fidélité numérique. L’écart capturé dans KDA ne prouve pas que KDA est sa cause : une projection ou réduction en amont reste une hypothèse à instrumenter. Les captures et les reçus d’arrêt sont conservés.

## GPU réellement exécuté

Le provider expérimental Q8 a été intégré au vrai dispatcher CPU et compilé dans un runtime isolé. Les poids restent dans leur buffer CPU original ; quatre buffers Metal fixes totalisent8028160B. Q2_K/MoE, KDA, mHC et les autres opérations restent CPU. La présence de `n_gpu_layers=0` n’est pas utilisée pour nier les commandes GPU observées.

La première suite23 cas a été refusée avant tout enfant sous pression2. Après retour réel à1, huit paires CPU/GPU ont terminé :2490368 sorties finies bit à bit et48 commandes GPU terminales. Le neuvième cas M8192/K4096/N64 a dépassé le plafond de composant128Mio (138133504B observés) et a été arrêté. Ce bras reste un échec ; ses formes64/128 les plus grandes ne sont pas qualifiées.

Une suite distincte de21 cas exclut explicitement ces deux grandes formes. Elle exécute cinq fois chaque graphe avec les mêmes poids synthétiques résidents, distingue la première opération des quatre suivantes et exige le même SHA pour chaque répétition. Elle a terminé en11,519106209s :20 comparaisons finales,2903040 sorties finies bit à bit,60 opérations GPU publiées et266 commandes terminales, dont une tuile avant un refus fatal délibéré. Ce refus ne publie aucune sortie partielle et ne repart pas sur CPU après soumission. Pics observés : enfant133070848B, contrôleur42909696B, allocations Metal8486912B ; croissance globale de swap0, secteur et intégrité des sources validés.

Sur les12 paires normales GPU, la médiane des quatre répétitions résidentes est **3,7675 à38,9248 fois plus lente que le CPU10**. Exemple M8192/K4096/N32 : CPU0,0031958755s, GPU0,05952525s. Ces temps incluent les coûts du graphe et du provider, pas seulement le timestamp GPU. Aucun débit SSD ou gain GLM n’en est déduit. Le provider scalaire reste OFF et n’est pas retenu pour un essai GLM coûteux.

Une variante séparée Metal4/TensorOps int8×int8→int32 a également été réellement exécutée. Après deux refus du compilateur (arguments temporaires puis type `char` distinct de `int8_t`), la version corrigée compile son shader et son pipeline. Elle garde un dot entier par bloc32 et une FMA FP32 dans l’ordre CPU. La suite21 cas termine en10,632500917s avec2903040 sorties finies bit à bit,60 publications et266 commandes terminales. Pic enfant133103616B, allocations Metal8486912B, swap0 ; secteur/intégrité validés. Les graphes résidents normaux restent **2,3524 à25,3694 fois plus lents que le CPU**. Exemple M8192/K4096/N32 : CPU0,0027420415s, TensorOps0,0251155625s. Le temps GPU cumulé des265 commandes normales passe à0,139173248s, mais ce compteur isolé ne mesure pas le coût total du graphe. La même restriction des deux plus grandes formes reste explicite. Aucun préfill GLM complet ni bénéfice de bout en bout n’est qualifié.

Les métadonnées de backend peuvent soumettre des commandes Metal vides d’initialisation. Les champs historiques «0 commandes» signifient0 dispatch de matrice du provider, sans garantir l’absence de toute activité du périphérique.

## Réponses aux douze questions

1. **Le GPU accélère-t-il GLM complet ?** Non démontré. Les providers scalaire et TensorOps sont fidèles sur les composants admissibles, mais plus lents que le CPU pour le coût total mesuré. Aucun des deux n’est retenu pour un long essai GLM en l’état.
2. **Meilleur batch fidèle ?** Inconnu. CPU8 est le témoin complet ; CPU16 corrigé reste incomplet et CPU32 est rejeté numériquement.64/128 ne sont pas qualifiés sur GLM.
3. **Préfill économisé ?** Aucun gain qualifié. Les durées partielles CPU16 ne constituent pas une sélection stable ni un gain de décodage.
4. **208 tokens réutilisables exactement ?** Snapshot complet autonome construit et première restauration exacte ; bras complet et gain net non qualifiés. Cache désactivé.
5. **Coût sauvegarde/restauration ?** Capture totale0,061756958s ; sauvegarde native0,009878666s ; première lecture native0,004909917s ; reset/restauration/vérification0,067238s. Taille149,478519Mio, sous plafond256Mio. Points de contrôle du témoin :0,121350125 /0,114084291 /0,195911333s. Ces opérations ne sont pas toutes le même chronométrage.
6. **Gain GPU+cache ?** Non mesuré, aucune addition de gains supposée.
7. **Deux EOS utiles ?** Non obtenus. Agent1 a produit171 IDs partiels ; aucun succès agent600 revendiqué.
8. **Isolation correcte ?** Identités/prompts et rejeu natif gelé contrôlés. Les historiques fonctionnels et la séquence complète après restauration restent non qualifiés.
9. **Protections respectées ?** RSS8,5Gio, swap32Mio, pression/réserves, secteur/continuité/intégrité et contrôle des processus restent appliqués. Les deux bras swap sont des échecs arrêtés, pas des succès sans dépassement. La suite GPU initiale a également déclenché son plafond additionnel128Mio. Tous les processus possédés clos sont absents. Le verrou UID du témoin CPU8 fut acquis tardivement ; cette limite historique reste explicite.
10. **Configuration RONIN retenue ?** CPU10/GPU0, un modèle/slot ; aucune nouvelle optimisation en production.
11. **Prêt pour le pilote fonctionnel ?** Non, deux réponses utiles jusqu’à EOS et la validation complète manquent.
12. **Limites pour600 agents ?** Modèle≈110Go/16Gio, préfill CPU8≈21–22min sur les prompts gelés, décodage lent, un slot physique, gros lots divergents ou instables et cache complet non qualifié. Aucune performance600 n’est extrapolée.

Les hausses de swap sont des compteurs système globaux ; elles ne sont pas attribuées exclusivement au modèle. Les lectures physiques du SSD et les décompositions de temps non instrumentées restent inconnues. Les annulations gardent leur classification native et les réponses partielles ne sont jamais reclassées EOS.

## Livraison et intégration

[PR23 en brouillon](https://github.com/iliasm-a11y/airllm-next/pull/23), sans merge de main ni basculement de production. Les dernières preuves de reprise sont dans [Preuves-reprise-M5.json](../glm53_m5_candidates/resumed-summary.json) et les [métriques](../glm53_m5_candidates/metrics-resumed.json). Les archives CPU8 et les anciens candidats restent des instantanés historiques explicitement gelés ; leur CI ne qualifie pas les nouveaux shaders.

Le cache commun reste unique/immuable, avec sérialisation complète `NONE`. `PARTIAL_ONLY` n’est pas un snapshot autonome. Le sampler et les logits ne sont pas sauvegardés ; un suffixe privé non vide et un sampler neuf restent nécessaires. Les adaptateurs HTTP/RONIN préparés restent désactivés tant que leurs preuves natives et leur transport propre ne sont pas qualifiés.
