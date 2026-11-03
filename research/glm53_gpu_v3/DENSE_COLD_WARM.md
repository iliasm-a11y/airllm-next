# Diagnostic d'une matrice Q8 complète, première utilisation et répétitions

`dense_cold_warm.cpp` exécute uniquement le vrai produit GGML CPU10 d'une matrice originale `blk.0.attn_q.weight`, Q8_0 `[4096,8192]`, soit **35 651 584 octets**. La matrice reste mappée en lecture seule ; aucune déquantification persistante, copie des poids, inférence GLM ou initialisation GPU n'est effectuée. Un pool CPU10 persiste entre les phases.

L'activation est le vecteur FP32 réel de 4 096 valeurs capturé à `attn_norm-40`, étape 1 de `trace2-cpu`. Ce choix correspond à la largeur du produit et fournit une entrée représentative du modèle, mais il **ne reproduit pas l'opération originale de couche 0**. Les 8 192 sorties sont conservées et hachées après chaque phase, avec égalité bit à bit exigée face à la première phase.

La première utilisation n'est pas déclarée « cache froid ». Chaque phase mesure le temps mural du produit natif, les deltas de lectures OS et pageins de **son propre PID**, et des snapshots de bloc système pris avant/après la phase via une synchronisation avec le processus parent. Les écritures de résultats, autres processus et readahead peuvent contribuer aux compteurs système. Ils ne mesurent pas les cellules NAND. Les compteurs processus peuvent aussi inclure les premières fautes de pages des bibliothèques. Les répétitions ne sont pas supposées sans I/O : leurs compteurs le vérifient.

Une différence de durée première/répétition inclut effets de pages, caches, fréquences, allocation initiale du workspace et ordonnancement. Elle n'est pas automatiquement une mesure exacte du temps d'attente SSD. Le test est une attribution de composant, pas un débit par token ni une qualification du modèle.

Le payload déclaré reste sous 128 MiB. Le garde externe propre au composant limite le RSS à 256 MiB, la croissance globale du swap à 32 MiB, refuse une pression critique/inconnue et arrête le processus possédé sur dépassement ou timeout. Le verrou physique de benchmark est commun aux autres expériences.

```sh
# Prépare métadonnées et activation seulement, sans lire les poids du checkpoint.
python -m research.glm53_gpu_v3.dense_cold_warm \
  --model-directory /absolute/path/GLM-5.3-Flash-Q2_K_S \
  --directory /absolute/path/new-component-directory \
  --output /absolute/path/new-component-report.json

# Exécution explicite uniquement dans un créneau coordonné sans autre expérience.
python -m research.glm53_gpu_v3.dense_cold_warm --execute \
  --model-directory /absolute/path/GLM-5.3-Flash-Q2_K_S \
  --directory /absolute/path/another-new-directory \
  --output /absolute/path/another-new-report.json
```

Les répertoires de source, bibliothèques et capture se règlent avec `--source`, `--libraries`, `--capture`. La compilation du petit exécutable est indépendante du runtime ; elle ne reconstruit pas les bibliothèques et ne change pas la preuve de source native gelée.

## Exécution réelle unique sur M5

Le [rapport brut](evidence/dense-cold-warm-01.json) conserve les quatre phases, compteurs avant/après des contrôleurs de bloc, hashes de toutes les sorties, activation capturée et hashes des bibliothèques. Le payload connu est 36 978 688 octets ; le pic RSS échantillonné est **42 680 320 octets**, sans croissance globale du swap ni violation de garde. Les statistiques du checkpoint sont inchangées.

| Phase | Produit GGML mural | Lectures OS du PID | Pageins du PID | Lectures de bloc système |
|---|---:|---:|---:|---:|
| Première utilisation observée | 15,783 ms | 35 672 064 o | 2 177 | 35 815 424 o |
| Répétition 1 | 0,322 ms | 0 | 0 | 81 920 o |
| Répétition 2 | 0,349 ms | 0 | 0 | 462 848 o |
| Répétition 3 | 0,314 ms | 0 | 0 | 425 984 o |

Les **8 192 valeurs** sont identiques en bits dans chaque phase, SHA-256 `4d43f62056d934601a565fda2726017c92f573474cd8cccf9783e318d2dea30d`. Les fichiers F32 bruts et stderr sont conservés dans `work/v3/dense-cold-warm-01` du workspace de campagne. L'activation provient de la couche 40 et le poids de la couche 0 : cette paire sert au composant compatible en forme, pas à rejouer une vraie opération de couche 0.

Le temps de chaque phase commence juste avant `ggml_backend_graph_compute` et se termine à son retour. Il comprend, pour la première phase, la première activation du calcul et l'allocation initiale éventuelle du workspace. La création du backend, des objets du pool CPU10, le mapping et la copie d'activation ont lieu **avant** les phases et sont exclus de ces durées. La soustraction première/répétition n'isole donc pas exactement l'I/O ; les effets d'initialisation du premier calcul et de caches matériels restent présents.

Cette unique exécution établit un fort écart entre première utilisation avec lectures OS mesurées et répétitions sans lecture OS attribuée au PID. Elle ne démontre pas qu'on peut maintenir chauds les 9,444 Go denses du modèle entier, ni un gain de débit de décodage ou une performance sur le SSD externe.
