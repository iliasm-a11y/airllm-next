# Revue indépendante — transport packed et protocole 5.1

## Verdict au gel

**Apte à poursuivre une campagne expérimentale supervisée**, sous les limites ci-dessous. Aucun résultat de cette revue ne démontre une supériorité sur Colibrì. Le screening initial reste exploratoire : il a été arrêté sur le transport direct + DLPack et précède le durcissement du protocole.

Base relue : `f82a064c821dcc6d79f4331844928eb1e8d22093`, puis les correctifs gelés de contre-pression, du superviseur et du harness. Ces correctifs n'appartiennent pas à `f82a064` tant qu'un nouveau commit ne les contient pas. Chaque essai doit être attribué à son HEAD et à ses hashes de sources effectifs, ainsi qu'aux hashes du protocole.

Aucun modèle et aucune suite complète lancés par cette revue. Les résultats de campagne réelle et de suite complète sont produits par le parent. Vérifications indépendantes effectuées : lecture des sources/tests et microtest DLPack CPU sur un payload synthétique de 144 octets. Les tests ciblés du protocole sont exécutés sans conftest ni import de moteur.

## Transport direct et propriété des buffers

Le lecteur direct réserve le payload complet avant `np.empty` et avant IO ; le contrôle de sélection, `max_read_bytes`, le budget du lecteur et le budget propriétaire restent actifs. Le verrou de descripteur est maintenu jusqu'à la fin de la lecture, empêchant un `close` concurrent. Les short reads poursuivent uniquement la sélection ; EOF et dépassement échouent ; EINTR/EAGAIN ont un nombre de retries borné. Le suivi du propriétaire est transféré avant les appels susceptibles de retenir l'allocation dans un traceback.

L'allocation directe appartient à un ndarray NumPy ; les vues et les exports DLPack conservent ce propriétaire. Le microtest indépendant avec un consommateur `np.from_dlpack` confirme : après suppression de la référence hôte, **144 octets encore chargés**, propriétaire vivant et données exactes ; après suppression du consommateur, **0 octet**, propriétaire détruit. Cette preuve CPU ne remplace pas un essai Metal réel.

L'import DLPack est opportuniste. `--zero-copy-weights` exprime une demande ; les compteurs `packed_zero_copy_imports` et `packed_zero_copy_fallbacks` indiquent son exécution. Le fallback conserve la copie exacte. Le compteur de copies hôte évitées décrit une différence de chemins de copie, pas une mesure de bande passante mémoire ou de NAND.

Le screening réel du parent a révélé une lease DLPack encore vivante après calcul : environ 9,4 Mo hors cache rendaient impossible le bloc suivant de 13 762 560 octets sous le plafond packed de 83 886 080 octets. Le refus était correct. Le correctif gelé effectue `mx.synchronize()` seulement lorsqu'une nouvelle sélection ne tient pas, puis vérifie à nouveau la place sous verrou. Il n'augmente aucun budget, n'efface pas le cache et n'appelle pas le GC. Une référence utilisateur encore vivante doit continuer à provoquer un refus. La réussite réelle de ce correctif doit être établie par les essais suivants ; elle n'est pas déduite des tests de mock.

## Supervision des processus

Le superviseur ajoute un seuil de 32 processus possédés, surveille aussi les descendants observés devenus détachés et conserve leur identité de création. Les mesures RSS/réserve/swap/pression/durée restent présentes. Aucun signal n'est envoyé à un processus étranger à l'expérience.

Il s'agit d'un **seuil d'arrêt échantillonné**, pas d'une limite noyau garantissant qu'aucun instant ne dépasse 32 processus. Les créations et pics entre deux échantillons peuvent échapper à la mesure. Employer `sampled_peak_processes` et préciser la portée de l'échantillonnage ; ne pas annoncer une borne instantanée absolue.

## Défauts de protocole corrigés après le screening

La première version de campagne ne gelait que `nextgen/*.py`, permettait des arguments bruts réservés, ne validait pas de preuve indépendante du superviseur et changeait de cwd sans normaliser les chemins. Le screening initial utilisait un plan sûr connu, mais son statut doit rester exploratoire.

Le harness et la campagne gelés ajoutent :

- une allowlist des seuls contrôles de comparaison ; `--worker`, `--model`, `--tasks`, `--output`, `--label`, `--engine-root`, les options internes et les abréviations sont refusés dans `job.args` ;
- un champ `job.engine_root` explicite pour comparer un checkout historique, avec chemins absolus avant tout changement de cwd ;
- des tâches non vides, identifiants uniques, prompts réels non vides et limites de tokens strictement entières et positives ; la capture des tâches et leur hash proviennent de la même lecture ;
- le gel du script benchmark, du script campagne, du plan et de tous les fichiers de tâches, vérifié avant/après les essais ;
- un manifeste HEAD + contenu par moteur, y compris le moteur du superviseur, au début et à la fin ; le worker vérifie les valeurs attendues avant import du moteur ;
- une garde indépendante obligatoire : commande exacte supervisée, limites identiques, échantillons présents, sortie 0, absence de violation, d'erreur d'admission et d'erreur de nettoyage ;
- des sorties une pour une avec les tâches demandées, statut terminé, tokens réellement enregistrés et dernier token membre des EOS déclarés. Un EOS n'est pas une évaluation de la justesse du contenu.

Tests CPU ciblés : **90 passed**, sortie 0, via `python -m pytest --noconftest tests_next/test_research51_campaign.py -q`, puis **1 passed** pour la régression supplémentaire de finalisation. Les appels susceptibles de lancer une inférence sont mockés. Aucune nouvelle dépendance ni aucun modèle utilisés. La suite complète communiquée par le parent avant ce dernier test compte **1 386 passed, 2 skipped, 11 subtests passed** ; elle n’a pas été relancée par cette revue.

### Finalisation après dérive tardive

Le défaut résiduel initial permettait une sortie shell à 0 lorsqu’un fichier changeait exactement après le dernier contrôle d’un run mais avant le `finally` final. Le parent l’a corrigé après la suite complète : le statut `stopped` persiste et la campagne lève désormais une erreur après finalisation, sans masquer une exception antérieure. Un test CPU isolé simule des preuves et une garde valides, puis une dérive seulement au dernier contrôle : **1 passed**, statut arrêté enregistré et erreur de sortie vérifiée. Le consommateur doit toujours exiger **`campaign.status == completed`, `protocol_unchanged == true` et `engines_unchanged == true`**, en plus des codes de sortie et gardes individuelles.

## Sens exact des mesures

| Champ | Portée | Limite d'interprétation |
| --- | --- | --- |
| `forward_seconds` | Appel moteur, logits évalués avant retour | Exclut le hash des logits/routes effectué ensuite par le wrapper |
| `ttft_seconds` | Début de la boucle de génération au premier token synchronisé | Exclut template, tokenisation du prompt et création de session ; inclut le wrapper de vérification |
| `decode_tokens_per_second` | Tokens après le premier jusqu'au dernier token enregistré, EOS compris | Inclut callbacks, décodage partiel et réécriture JSON à chaque token |
| `total_seconds` | Boucle de génération instrumentée | N'inclut pas toute la préparation du prompt ni toute la finalisation textuelle |
| `wall_seconds` du tour | Appel chat et instrumentation du tour | Inclut la préparation propre à `chat.reply` et les écritures intermédiaires |
| `worker_wall_seconds` | Initialisation tokenizer/modèle et tâches après les imports/setup initiaux | Ce n'est pas la durée lancement-processus → réponse |
| `guard.elapsed_seconds` | Admission, processus enfant et nettoyage supervisés | Inclut l'instrumentation et le superviseur ; ce n'est pas une durée moteur pure |

Ces mesures sont exploitables pour comparer des variantes sous **le même harness figé**, avec mêmes tâches et paramètres. Elles ne sont pas un débit de bout en bout non instrumenté. Un classement externe demande un protocole commun ou la publication explicite des différences.

Les workers sont frais, mais le cache de pages OS est incontrôlé et aucun purge n'est effectué. Un ordre fixe legacy → readonly → direct → DLPack ne suffit pas à isoler un gain de transport. Les répétitions alternées/contrebalancées doivent être privilégiées pour la campagne primaire.

`process_storage_read_bytes` provient d'un compteur OS à portée processus, avec lectures annexes et readahead possibles ; ce n'est pas une mesure par fichier ni une preuve de débit SSD physique. `physical_ssd_bandwidth_certified` reste faux. Le contrôle du modèle par taille/inode/mtime n'est pas un hash intégral de ses 23,9 Go ; conserver la distinction entre identité stat vérifiée et intégrité cryptographique complète.

## Gates avant conclusion de recherche

Conserver le screening arrêté et ses sorties partielles. Valider la nouvelle génération complète après contre-pression, puis les paires de comparaison contre le checkout historique avec même GGUF, mêmes paramètres et budgets, sorties/EOS, hashes de logits/routes et gardes vérifiés. Distinguer préfill, TTFT, décodage, total instrumenté, sortie complète et état des ressources. Ne promouvoir aucun gain observé sur une tâche courte en validation universelle, en débit SSD certifié ou en résultat supérieur à Colibrì.
