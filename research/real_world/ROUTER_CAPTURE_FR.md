# Reproduction ciblée de l’arrêt D2

La campagne initiale s’est arrêtée après 77 tokens de D2, lorsque le routeur a refusé une égalité à la frontière des huit experts retenus. Une reproduction isolée a utilisé exactement le même prompt utilisateur, le même système, le plafond de 256 tokens et la configuration enregistrée : cache de poids 64 MiB, KV 256 MiB, contexte 1 280, calcul FP32 et noyau Metal.

L’enfant a importé les sources originales, dont les empreintes concordent avec la campagne échouée. Un wrapper temporaire dans ce seul processus appelle le routeur original sans changer ses décisions. À l’exception, il lit les variables locales conservées dans la traceback du routeur et sauvegarde leurs tableaux ; il relance ensuite l’exception. Il ne poursuit pas la génération après la défaillance et ne remplace aucune réponse.

## Observation reproduite

| Mesure | Résultat |
| --- | --- |
| Préfixe généré | 77 IDs sur 77 exactement identiques au fragment de la campagne initiale |
| Erreur | `Qwen router top-k boundary tie: reference tie-breaking is not validated` |
| Couche | 5, préfixe de poids `blk.5.` |
| Position du cache avant le morceau en cours | 561 |
| Token d’entrée de la passe échouée | 16246 |
| Nombre d’experts / nombre retenu | 256 / 8 |
| Experts au bord de la sélection | 49 retenu, 206 exclu par l’ordre courant |
| Logit FP32 des deux experts | −4,415645599365234, exactement identique dans les tableaux capturés |
| Probabilité des deux experts | 0,010838142596185207, exactement identique |
| Valeurs non finies | Aucune dans x, poids, logits, probabilités ou scores capturés |

Il s’agit donc bien d’une égalité des logits dans ce calcul FP32, et pas uniquement de deux probabilités devenues égales après arrondi du softmax. Ce constat ne détermine pas, à lui seul, la politique de départage qu’un autre moteur ou une autre précision appliquerait.

## Fichiers et protection

Les preuves sont `evidence-real-world/router-failure-replay.json`, `.npz`, `.guard.json` et `.log`. L’archive NPZ contient `x`, `weight`, `logits`, `probabilities`, `order`, `selected`, `scores` et `next_score`, soit 2 108 484 octets non compressés au total. Son SHA-256 est `0b343cc940dfa5099fb87832acc1117998f3c95833658bf8b4a0b8095125cacd`.

Le processus supervisé s’est terminé normalement après avoir enregistré l’échec recherché : code 0, aucune violation du garde, pic RSS échantillonné de 499 040 256 octets, disponible système minimal 4 012 916 736 octets, aucune croissance de swap observée, durée totale supervisée 263,10 secondes.

L’instrumentation exacte est archivée dans `evidence-real-world/sources_router_capture/replay_real_world_router.py`, avec son empreinte dans le rapport. Les sources originales du moteur sont archivées dans `evidence-real-world/sources_before_router_fix/`.

Les fichiers source sur disque ont été corrigés pendant que le processus conservait ses objets Python originaux déjà importés. Le champ `source_files_unchanged_during_replay=false` rend ce changement visible. Les empreintes d’import du moteur, les tableaux originaux et l’identité des 77 tokens constituent les preuves de la version effectivement reproduite ; une simple lecture des fichiers actuels ne la représenterait pas.

Une première tentative de contrôle des empreintes a refusé la différence du helper de sauvegarde avant toute inférence. Son journal et son garde sont conservés sous `router-failure-source-preflight.*`. La reproduction suivante autorise explicitement cette différence du helper non utilisé pour la génération, consignée séparément, tout en exigeant l’identité des modules `air_llm/` à l’import.

Cette capture valide la reproduction de l’arrêt du moteur. Elle ne constitue ni une réponse D2 achevée, ni une validation d’un correctif de départage. Ces étapes doivent être évaluées séparément.
