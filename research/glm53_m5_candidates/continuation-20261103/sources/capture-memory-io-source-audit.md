# Captures natives : mémoire et écritures — audit statique borné

## Conclusion

**Pas de conteneur natif conservant les gigaoctets de captures trouvé dans les sources lues.** Un levier concret existe cependant dans le contrôleur : toutes les mesures sont conservées en mémoire puis le rapport complet est réécrit toutes les dix mesures, alors qu’un journal append-only existe déjà. Le volume de sérialisation/réécriture croît donc quadratiquement avec le nombre de mesures. Ce travail est évitable sans supprimer les preuves finales, sous un nouveau contrat de provenance.

Les sorties binaires utilisent les écritures ordinaires de `std::ofstream`, sans demande explicite `F_NOCACHE` ou `fsync`. Une contribution des écritures au cache de fichiers et à la pression mémoire est une **hypothèse mesurable**, pas une cause établie du swap. Le premier essai pertinent est un composant d’écriture borné, après clôture de C16 ; ces faits ne justifient pas à eux seuls une nouvelle campagne modèle.

## Ce qui est observé, et ce qui ne l’est pas

Le contexte doit être corrigé sur un point : **v3 n’a pas échoué au swap**. `native-noblas-full32-summary.json` indique annulation numérique, natif −15, 2 488 mesures, pic RSS échantillonné 5 868 306 432 B et croissance globale du swap observée 0. V4 échoue à `SWAP_LIMIT`, natif −15, 2 177 mesures, pic RSS 5 967 937 536 B, swap global +38 010 880 B. L’inventaire des captures donne respectivement 3 272 127 642 et 2 621 226 423 octets logiques, soit **5 893 354 065 B combinés**, et non 6 Go dans chacun des deux runs. Leur taille sur disque n’est pas une quantité de pages sales simultanément résidentes.

Le coordinateur rapporte que C16 sans captures brutes reste stable en swap depuis son admission à environ 35 minutes. Cette observation n’a pas été relue dans une base ou un flux vivant pour cet audit. Prompts, lots, runtime/BLAS, géométrie, historique mémoire du système, allocations et durée d’instrumentation ne sont pas tous équivalents. La différence ne constitue donc pas une expérience causale isolant les captures.

Les reçus ne donnent pas la quantité de cache sale attribuable à chaque fichier, le heap retenu par l’allocateur, la mémoire compressée du PID natif externe, les écritures physiques SSD ni le coût pur des captures. Ces valeurs restent inconnues. Le contrôleur observe la RSS du **natif** et les ressources globales ; la RSS du wrapper n’est pas incluse dans ce champ (`glm_native.py:225–252`).

## Durée de vie des buffers natifs

Référence : `work/native-noblas208-v4-candidate/research/glm53_batched_prefill/native_prefill.cpp` ; les copies mesurées de snapshot et state capture sont inchangées dans ce candidat.

| Source et lignes | Mémoire effectivement prévue par le code |
| --- | --- |
| `native_prefill.cpp:107–119` | Un `vector<unsigned char>` de `ggml_nbytes(node)` par capture sélectionnée, plafond 64 Mio ; fichier et vecteur détruits au retour du callback. Aucun tableau de tous les payloads. |
| `state_capture.inc:23–68` | Lecture/hash des écritures observées par morceaux d’au plus 1 Mio ; seuls métadonnées, finitude et SHA sont écrits dans JSONL. Le contenu complet des écritures d’état n’est pas stocké dans une collection. |
| `native_prefill.cpp:256–266` | Un buffer de sérialisation complète, plafonné à 256 Mio, local au checkpoint ; détruit avant le chunk suivant. |
| `native_prefill.cpp:297–310,339–349` | Buffers distincts pour état final puis post-continuation, dans deux portées successives ; pas de rétention par agent ni jusqu’au prochain prompt. |
| `snapshot.h:69–91,131–169` | Une seule allocation persistante `data_`, bornée à 256 Mio, uniquement après capture en mode restore. Taille observée à 208 : 156 739 588 B. Elle n’est pas créée dans les bras full v3/v4. |
| `native_prefill.cpp:177–183` | Le JSON des prompts reste en mémoire pendant le run, fichier admis jusqu’à 16 Mio ; les fichiers observés font environ 5,4 Mo. Trois listes d’IDs sont copiées pour 1→600→1. Ce sont des prompts, pas 600 états natifs. |

Le plafond 64 Mio n’est pas une allocation observée de 64 Mio. Les **118 premières lignes** du JSONL de captures de chaque bras, toutes request0/offset0, décrivent un maximum de **1 048 576 B par tenseur** et une somme de **58 946 560 B exportés pour ce chunk**. Les 90 premières lignes d’écritures d’état V4 décrivent **75 137 024 B logiques relus pour hash**, avec scratch maximal 1 048 576 B. Ces observations viennent seulement des petits en-têtes JSONL, sans lecture des binaires. Elles démontrent du trafic de capture important, pas une allocation persistante de cette somme.

Une libération C++ ne prouve pas le retour immédiat de toutes les pages à l’OS ; l’allocateur et le runtime peuvent garder des réservations. Aucun profil d’allocations ne permet ici de confirmer ou d’exclure une rétention effective. Le budget simultané de sérialisation native de 16 Mio est une enveloppe conservatrice déclarée, pas une trace d’allocation.

## Deux coûts évitables ou testables

**Réécritures du garde — établi par source.** `native_run.py:245–265` ajoute chaque mesure à `report["samples"]`, écrit aussi la même mesure dans `guard-samples.jsonl` avec flush, puis réécrit le JSON complet toutes les dix mesures. `write`, lignes 45–49, construit une chaîne indentée complète et remplace atomiquement le fichier. Les fichiers guards finaux observés pèsent 8 991 931 B (v3) et 7 872 339 B (v4), en plus des journaux append-only de 6 634 731 et 5 800 932 B. Le volume cumulé de ces réécritures et leur RSS temporaire ne sont pas mesurés ; le code établit leur croissance quadratique, pas leur trafic physique APFS/SSD.

Correction future possible : conserver la fréquence, les décisions et seuils exacts du watchdog, journaliser chaque mesure une fois, publier seulement un petit état de progression, puis produire **le même garde final exhaustif** après arrêt du natif. Une variante peut d’abord conserver la liste en RAM pour ne changer que les réécritures. Une seconde peut reconstituer les mesures depuis le journal après cleanup pour éviter leur rétention pendant le modèle. Il faut prouver ordre/comptage, absence de trou, refus de journal incomplet, mêmes verdicts et traitement des erreurs d’écriture. Ne pas supprimer l’historique que les comparateurs exigent. Changer le wrapper change ses empreintes et demande un contrat de comparaison explicite ; aucune preuve ancienne n’est transférée automatiquement.

**Politique des fichiers binaires — hypothèse testable.** Le helper ferme ses fichiers et appelle `flush()` sur plusieurs JSONL, mais ne demande pas explicitement d’évitement de cache, d’éviction ou de synchronisation au stockage (`native_prefill.cpp:113–119,250–265,288–309,326–347`). Le SDK local expose `F_NOCACHE` comme option par descripteur, et `F_NOCACHE_EXT` avec relaxation des contraintes de taille/alignement (`MacOSX.sdk/usr/include/sys/fcntl.h:257,340`). `F_FULLFSYNC`, ligne 260, demande en plus une vidange vers le support ; c’est une autre opération. Le libc++ livré ferme le `filebuf` via `sync()` et `fclose`, sans `fsync` explicite dans ce chemin (`.../include/c++/v1/fstream:752–760`).

Un writer expérimental dédié pourrait appliquer un flag de non-cache au **seul FD de sortie**, enregistrer son résultat et refuser s’il échoue. Il doit conserver exactement les bytes, offsets, strides, SHA et erreurs de close. Aucun `F_GLOBAL_NOCACHE`, purge globale, éviction des poids ou changement de protection n’est justifié. `fsync` n’est ni une preuve d’éviction, ni gratuit ; le faire à chaque petit fichier peut aggraver le débit et les synchronisations. Aucun effet garanti sur la mémoire ou le swap n’est déduit des seuls headers.

Réutiliser un petit buffer de lecture pour les captures pourrait limiter le churn, mais le maximum observé sur le premier chunk32 est déjà 1 Mio. Le levier de pic évident est plutôt le buffer temporaire de sérialisation complète (~150–156 Mio). L’API publique actuelle demande un buffer contigu complet : la rendre réellement streaming exige une autre interface. La capture du checkpoint final puis celle du fichier d’état final répètent aussi une sérialisation très proche (`256–266` puis `290–309`) ; une réutilisation peut économiser ce travail si un composant prouve l’identité de l’état et des SHA au même point. Elle ne réduit pas la taille du snapshot commun requis pour B et ne doit pas transformer une capture absente en donnée supposée.

## Protocole minimal proposé, sans exécution maintenant

1. **Contrôleur sans modèle :** rejouer un petit journal synthétique de mesures avec le schéma existant ; comparer réécriture historique et état compact + journal. Compter bytes applicatifs demandés, pic RSS propre, décisions/ordre/SHA final identiques et refus de journal tronqué. Gardes inchangées, sortie et durée strictement plafonnées. Ce test mesure le levier structurel avant de toucher au natif.
2. **Writer sans modèle/GPU :** même payload synthétique borné et même séquence de fichiers dans deux processus, writer ordinaire puis writer avec option par-FD acceptée. Petit budget, par exemple 64 Mio de sorties et 128 Mio RSS ; mêmes contrôles secteur/identité/UID, aucune purge et aucun dépassement autorisé. Relecture/SHA après la mesure ; observer temps complet, RSS native, mémoire disponible et compteurs VM/swap globaux étiquetés comme tels. Une taille faible peut être inconclusive sur le cache : ne pas extrapoler alors à 6 Go ou à GLM.
3. Seulement si une réduction reproductible utile et sans perte de preuve est mesurée, préparer une nouvelle version source/pins/comparateur, puis décider séparément si un nouveau témoin et un B sont matériellement justifiés. La fidélité doit toujours garder tous les logits, routes/KDA, états, écritures observées, checkpoints et continuations requis. Aucun succès de writer ne promet la fin d’un B.

**Limite de cet audit :** sources, headers officiels locaux, résumés clos et en-têtes JSONL seulement. Aucun test/import/build/modèle/GPU, benchmark disque, archive, lecture de payload, base ou SSE vivante ; aucune source mesurée, sortie livrée ou publication modifiée.
