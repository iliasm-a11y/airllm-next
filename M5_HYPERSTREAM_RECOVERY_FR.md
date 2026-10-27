# AirLLM Next — récupération HyperStream sur M5

27 octobre 2026. **H2 avec l’option `router_reference_chunk=16` réduit le premier token de 65,41 à 66,71 % sur le prompt historique mesuré, avec quatre paires AB/BA acceptées et scellées dans deux préparations successives.** Chaque paire présente 14 640 routes ordonnées égales et 744 960 logits émis identiques bit à bit. Le H2 original conserve son rejet. Aucun gain de décodage soutenu au-dessus de 1,584 token/s n’est encore qualifié ; la campagne complète reste inachevée.

Ce rapport décrit les preuves déjà acquises et l’état de reprise lu le **27 octobre à 19:43:10, heure de Paris**. Les éventuelles tentatives ultérieures doivent être lues dans [l’état courant](research/m5_validation/evidence/recovery-20261027-ultra/delivery-snapshot.json), sans extrapoler leur résultat.

Le travail part de la PR #13, commit `965afa1402335767e700e4ce81072bd380579c66`, sur `research/hyperstream-recovery-m5`. Le résultat H2 accepté appartient à `campaign-recovery-003`, candidat figé au commit **`1a9c7274c2e5ea4346f8fb6b79e0c3bbef839c3a`**. Les réparations de supervision ultérieures portent le candidat courant à **`a5d289704f436e59fc9b4aab5d62a1dba80f1347`**, sans changer le calcul Qwen gagnant. Au contrôle de publication communiqué, le commit distant **`bc29d7739d6832619b9c0ada5e2bc6c09307d4e5`** possède le même arbre de sources, **`f278b90afc24c257ba86e451464bc500c42aa666`**. La [PR #14 est créée en brouillon](https://github.com/iliasm-a11y/airllm-next/pull/14), empilée sur #13. `main` et les PR existantes ne sont pas fusionnés. Les preuves GPU et les contrôles numériques sélectionnés sont publiés dans cette PR ; les archives locales conservent les preuves complètes des blocs sélectionnés.

**Résultat matériel accepté : H1 contre H2 reference16.** Le checkpoint original de **23 938 321 664 octets** et le prompt historique de **364 tokens** sont figés. Les quatre workers donnent exactement `[19, 17, 248046]`, puis EOS. Le SHA-256 complet du checkpoint avant et après le bloc de quatre workers est `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`.

| Ordre | H1 : premier token | H2 reference16 : premier token | Réduction observée | Qualification |
|---|---:|---:|---:|---|
| AB | 162,982350 s | 54,255501 s | 66,710812 % | acceptée et scellée |
| BA | 157,010395 s | 54,317349 s | 65,405253 % | acceptée et scellée |

Une nouvelle répétition AB/BA sur les sources finales `a5d2897` est aussi acceptée et scellée dans la préparation 005 : **154,691489 → 53,122647 s (−65,658973 %)** et **155,238712 → 53,200001 s (−65,730197 %)**. La [vérification indépendante finale](research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-final-independent.json) revalide 416 fichiers du manifeste, les quatre workers et leurs bornes de hash. Les mêmes 14 640 routes ordonnées et 744 960 logits par paire sont identiques bit à bit. Cela porte le résultat mesuré à **quatre paires acceptées**, toujours sur le même prompt de 364 tokens et trois tokens émis.

La réduction est calculée séparément par paire : `100 × (1 − TTFT_H2 / TTFT_H1)`. Les lectures logiques de la génération passent de **139 447 912 832 à 40 928 599 936 octets**, dans chaque paire. Ce sont des demandes de lecture logiques ; elles ne certifient pas les octets NAND.

La [vérification indépendante, révision 2](research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-independent.json) relit les quatre workers, leurs gardes, les artefacts de logits, les identités des sources, le manifeste et les bornes du scellement par hashes complets. Pour **chacune** des deux paires : 14 640 routes ordonnées égales, 744 960 logits émis comparés, écart maximal nul et égalité bit à bit. L’expérience `h1_vs_h2_reference16` porte bien le statut `accepted` dans la campagne. Les [analyses AB](research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-forensic-pair-00.json) et [BA](research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-forensic-pair-01.json) conservent les contrôles détaillés.

La révision 2 corrige un champ dérivé hérité d’un modèle de rapport de rejet : elle n’altère ni les observations brutes, ni le scellement, ni les contrôles numériques. Les critères d’acceptation des routes et logits restent identiques. Le schéma reconnaît explicitement le nouveau paramètre 0/16 ; il n’accepte toujours pas une permutation des mêmes experts.

**Portée exacte :** le gain accepté concerne le TTFT instrumenté de ce prompt et ces quatre paires, sur le M5 observé. Trois tokens jusqu’à EOS ne mesurent pas un décodage soutenu. La parité directe de tous les états internes DeltaNet/KV, la conformité de tous les prompts et un avantage durable en génération longue ne sont pas établis par ce résultat.

**Pourquoi H2 original permutait les experts.** Les paires du H2 original restent rejetées, sans réécriture de leurs preuves. La couche 30, position 154, échange les experts **193 et 104 aux rangs 7/8** ; les ensembles sélectionnés restent égaux. La première campagne fraîche a reproduit ce rejet en AB/BA malgré un TTFT de 160,819 → 53,395 s puis 158,962 → 53,659 s. Ces gains du H2 original restent non qualifiés.

Les captures réelles H1/H2 ont haché intégralement le checkpoint avant/après. Leurs poids de routeur sont identiques bit à bit ; les activations d’entrée diffèrent au maximum de **5,72205 × 10⁻⁶**. À la cible :

| Chemin | Logit de l’expert 193 | Logit de l’expert 104 | Ordre observé |
|---|---:|---:|---|
| H1, bloc 16 | −4,5252580642700195 | −4,525258541107178 | 193 puis 104 |
| H2 original, bloc 128 | −4,525256156921387 | −4,525256156921387 | 104 puis 193 |

H1 distingue les deux logits d’**un ULP FP32**. H2 produit une égalité exacte, également présente dans ses probabilités : **0,012111574411392212** pour chacun. `argsort` stable conserve alors leur ordre d’indice, 104 avant 193 ; il s’agit d’un classement cohérent des valeurs FP32 reçues, pas d’une inversion entre l’ID d’un expert et son poids. Le caractère stable est documenté dans [MLX 0.32.4](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html). Les [captures comparées](research/m5_validation/evidence/recovery-20261027-ultra/router/h1-h2-comparison.json) conservent valeurs et empreintes.

Le [contrôle sur les mêmes activations H2 et les mêmes poids](research/m5_validation/evidence/recovery-20261027-ultra/router/controlled-reference.json) reproduit le bloc 128 bit à bit. Le calcul GPU sur un bloc contigu de 16 tokens rétablit **193 avant 104**, avec deux ULP de séparation. Modifier la forme de la réduction FP32 suffit donc à produire ou lever l’égalité, sans changer les activations de ce contrôle. Les petites divergences d’activation amont existent aussi ; l’ordre exact des instructions Metal n’a pas été tracé.

Le repli existant ne détectait que l’égalité à la frontière **8/9**, absente ici. Le calcul CPU Torch FP32 complet sur les 128 lignes conserve lui aussi 104 avant 193 : cette seule extension de repli ne restaurerait pas la convention H1. Les sommes FP64 servent uniquement d’oracle de diagnostic ; elles ne sont jamais injectées dans le moteur.

L’option **`router_reference_chunk=16`** réemploie réellement le routeur GPU FP32 historique sur des blocs contigus de 16 positions, puis concatène les positions de tokens et leurs poids associés. Elle ne trie pas les experts après coup. Les calculs experts et le streaming groupé de H2 restent en place. Le mode 0 conserve le chemin original ; l’option est explicite dans les arguments, les sources et la configuration scellés. Le [contrôle de l’implémentation](research/m5_validation/evidence/recovery-20261027-ultra/router/canonical-smoke.json) rétablit les 128 routes H1 du bloc capturé ; les quatre workers complets ci-dessus apportent ensuite la qualification sur le prompt mesuré.

Les microblocs sont relatifs à chaque chunk fourni. Une autre segmentation adaptative, un autre prompt ou une combinaison H3 doit donc réussir ses propres contrôles. La première capture H1 avait chargé une version antérieure du script : son source exact a été restauré et vérifié ; les fonctions de capture sont identiques. Cette [provenance](research/m5_validation/evidence/recovery-20261027-ultra/router/source-provenance.json) distingue les captures instrumentées des générations complètes et des mesures de performance.

**Supervision : défaut initial et régression réelle réparés.** Le crash historique provenait de `psutil.process_iter(attrs=...)` : l’exception survenait pendant l’avancement de l’itérateur, avant le `try` par processus. macOS remontait un `SystemError` de `proc_cmdline` avec un `PermissionError` de `sysctl(KERN_PROCARGS2)` comme cause.

L’inventaire utilise maintenant les PID puis des lectures unitaires protégées. Il traite les refus d’accès et les disparitions attendus, y compris cette chaîne précise ; un autre `SystemError` ou une erreur de programmation continue de remonter. Il vérifie la naissance et la vie du processus avec un nouvel objet psutil. Une identité incertaine n’autorise aucun signal. Les diagnostics répétitifs sont bornés et comptés ; un processus d’inférence réellement vivant mais non identifiable diffère l’admission.

Un second défaut a été découvert pendant la campagne 003 : son premier worker H1 long avait produit **187 tokens partiels** quand le moniteur a pris un PID déjà disparu pour un Python inconnu. La lecture `exe` avait donné `NoSuchProcess`, puis le code avait poursuivi jusqu’à `cmdline`, refusé, et combiné cela avec une naissance mise en cache. **Cet arrêt provenait de l’inventaire, pas d’une violation RAM constatée.** Il n’a pas invalidé le bloc court déjà scellé. Les 187 tokens restent une sortie inachevée, sans EOS ni benchmark accepté.

Le correctif abandonne immédiatement le PID sur `NoSuchProcess`, zombie ou `ESRCH` dans n’importe quel champ. Il contrôle aussi le statut avant les arguments et à la fermeture de l’observation ; une naissance mise en cache ne suffit plus à faire croire que le processus vit encore. Le cas exact du PID 29572 et les disparitions tardives sont testés, ainsi que le refus d’accès persistant d’un vrai Python vivant. Un essai avec un enfant local quittant naturellement pendant l’observation a également vérifié ce chemin, sans signal envoyé.

Le verrou de campagne empêche deux superviseurs d’attendre ou lancer simultanément la même préparation. Les états JSON sont remplacés atomiquement, avec synchronisation du fichier et du répertoire. Les journaux sont vidés et synchronisés régulièrement. La parenté et l’identité des descendants nouvellement créés évitent de les confondre avec un autre modèle. Un refus vérifié de pression, réserve ou swap renvoie **75** ; les erreurs de code, source ou intégrité restent distinctes.

La dernière suite portable donne **388 tests réussis, 24 sous-tests réussis et 64 tests GPU ignorés**, sans erreur ni échec dans [le XML de validation](research/m5_validation/evidence/recovery-20261027-ultra/tests-recovery-latest.xml). Ces tests couvrent les erreurs psutil, la régression de cycle de vie, les PID réutilisés, les verrous, la reprise, le profil reference16 et le maintien du rejet d’une permutation. Les tests ignorés ne sont pas des exécutions GPU. Le superviseur réel a continué malgré un `AccessDenied` facultatif sur le PID système 0.

**Ressources et nettoyage.** Les **4,71 Gio historiques désignaient la RAM disponible**, `psutil.virtual_memory().available`. Le disque disposait de plus de 50 Gio. Les ressources sont mesurées séparément sur les chemins des preuves, de la campagne et du checkpoint ; la réserve disque de **5 Gio** n’a pas été abaissée.

Cinq lots ont arrêté **154 auxiliaires MCP inutilisés** : **118 SIGTERM**, plus 36 parents sortis naturellement après leur enfant, aucun survivant et **aucun SIGKILL**. Aucun fichier, checkpoint, modèle ni preuve n’a été supprimé. L’audit vérifie UID, exécutable, arguments, dossier, naissance, descendants, absence de navigateur/document et terminal. Le cinquième lot autorise uniquement les connexions TLS du lanceur npm vers les adresses résolues du registre npm ; les enfants ne peuvent avoir de connexion IP. Ces auxiliaires étaient ceux des chats AirLLM inactifs et de la mission courante ; les applications personnelles et processus Codex principaux sont préservés. Après le cinquième lot, le disponible observé est passé de **4 941 922 304 à 8 339 701 760 octets**, swap inchangé. Les journaux `Journal-nettoyage-M5*.json` conservent les identités et signaux ; les inventaires détaillés restent privés dans `work/resource-audit/`. Des auxiliaires se sont ensuite relancés avec de nouveaux PID, ce qui empêche de considérer cette marge comme durable.

| Protection ou mesure | Valeur |
|---|---:|
| RAM physique | 16 Gio |
| Disponible initialement | environ 2,50 Gio, pression 2 |
| Fenêtres observées après libérations | environ 4,0 à 7,77 Gio, pression 1 |
| Stockage disponible | environ 50 Gio |
| Réserve moteur | 3 Gio |
| Payload déclaré / marge superviseur | 768 Mio / 256 Mio |
| Admission du superviseur | 4 Gio disponibles, pression 1 pendant 5 s |
| Plafond RSS / croissance de swap | 2 Gio / 128 Mio |
| Réserve disque | 5 Gio |

Le keeper historique hors dépôt ajoutait un seuil RAM de 5 Gio. Le superviseur publié dans #13 exigeait déjà 4 Gio : cette admission et les gardes du moteur sont conservées. Les 5 Gio de disque sont indépendants. Le swap préexistant est enregistré, sans être présenté comme nul.

Le hash complet ordinaire du checkpoint avait précédé une baisse de RAM disponible gênant l’admission suivante. Le hachage utilise désormais `F_NOCACHE` sur son seul descripteur macOS, sans purge globale ni changement des lectures du moteur. Le premier essai avec cette politique a retrouvé le SHA complet attendu sur les **23 938 321 664 octets** en **12,96 s**, pression 1 et swap inchangé. Les variations de RAM sont des instantanés système, sans attribution exclusive démontrée au hash. Cette politique reste déclarée dans les nouvelles configurations ; elle ne certifie pas des lectures physiques NAND.

**GPU M5.** Une tentative a été refusée avant lancement sous pression 2. Après libération, deux grilles complètes de **160 cas distincts** ont été exécutées : **320 exécutions de cas**. Sur la grille vérifiée indépendamment, **128/128 sorties reuse** sont identiques bit à bit à la référence GPU historique. Les **32 cas TensorOps** compilent et s’exécutent, mais échouent à la qualification numérique globale conservée. La déquantification C GGML correspond aux 32 jeux de poids ; les désaccords de réduction scalaire/FP64 restent des résultats négatifs visibles.

La politique reuse, déjà versionnée dans #13, exige les tenseurs bruts exacts et le diagnostic C lié. Elle ne qualifie ni TensorOps ni le modèle entier. Les ablations reuse, streaming groupé et spéculation du vrai Qwen attendent encore leurs résultats complets acceptés.

**État des campagnes au dernier relevé.** Les préparations historiques restent intactes. Toute modification de sources ou de scripts couverts a entraîné une nouvelle préparation ; les résultats anciens ne sont pas transplantés dans un protocole modifié. Le profil historique garde **24 expériences / 96 jobs** ; le profil reference16 comprend **25 expériences / 100 jobs**, deux paires AB/BA par expérience. Ses ablations H3 partagent le même contrôle H2 corrigé, et les combinaisons nécessitent des bénéfices individuels préalablement acceptés.

| Préparation | Jobs prévus | Workers observés lancés | Terminés avec garde propre | Workers scellés | Paires acceptées / rejetées |
|---|---:|---:|---:|---:|---:|
| 001, H2 original | 96 | 5 | 4 | 4 | 0 / 2 |
| 002, H2 original | 96 | 1 | 1 | 0 | 0 / 0 |
| 003, reference16 | 100 | 5 | 4 | 4 | 2 / 0 |
| 004, reference16 | 100 | 1 | 1 | 0 | 0 / 0 |
| 005, reference16 | 100 | 9 | 7 | 4 | 2 / 0 |

Les workers de hachage et de kernels sont distincts de ces compteurs de modèle. Un worker démarré ou terminé ne vaut pas une paire acceptée. Les workers non scellés restent bruts ; ils doivent être rejoués dans une tentative admissible.

La préparation 005, pilotée par le superviseur 006, s’est arrêtée à **19:27:41, heure de Paris**, pendant son quatrième worker : **pression macOS 2**, aucune croissance de swap, sortie propre du worker par SIGTERM, code de supervision **75**. Trois workers étaient complets mais leur bloc AB/BA n’était pas encore scellé. Ce dernier arrêt est un refus de ressources mesuré, distinct du défaut d’inventaire de la campagne 003 ; il ne justifie aucune nouvelle modification du moteur.

La reprise par le superviseur **007** a ensuite scellé et accepté les quatre workers courts de la préparation 005. Le premier H1 long a été arrêté après **71,96 s**, pression 2, RSS échantillonnée **481 968 128 octets**, aucune croissance de swap dans le worker. Les auxiliaires MCP ont été observés relancés avec de nouveaux PID. Ce constat ne mesure pas une contribution exclusive de chaque service à la pression ; il établit la coexistence du redémarrage et du refus de ressources. La dernière tentative 008 reste à effectuer dans les budgets restants, sans modification du moteur.

L’attente entre workers est maintenant bornée à **300 s**, avec toujours 4 Gio, pression 1 et cinq secondes stables avant admission. Les tentatives partagent un budget global de **8 heures**, **1 800 s d’attente externe cumulée** et **huit tentatives au maximum** ; le budget restant diminue aux reprises. Aucun seuil mémoire, RSS, swap ou disque n’est réduit. L’exécution est séquentielle et le détachement sous PPID 1 a été observé ; la survie après fermeture réelle de Codex n’a pas été testée. **La campagne complète et les comparaisons longues ne sont pas déclarées achevées.**

**Réponses aux trois questions.**

1. **H2 peut-il réduire le premier token d’environ 66 % avec une fidélité conforme ?** **Oui, sur le prompt mesuré avec reference16 : −66,71 % en AB et −65,41 % en BA**, routes ordonnées égales et logits émis bit à bit, hashes et gardes vérifiés. Le H2 original reste rejeté. Ce résultat ne qualifie pas tous les prompts, les états internes ou le débit long.
2. **TensorOps, reuse et spéculation améliorent-ils significativement le vrai débit ?** **Ce n’est pas encore démontré.** TensorOps reste exclu numériquement ; reuse passe sa grille miniature, et la spéculation attend ses paires réelles complètes. Aucun débit soutenu accepté au-dessus de **1,584 token/s**, ni les objectifs 3/5/10 tokens/s, n’est établi. Les 319 tokens H1 historiques à **1,46747 token/s** restent une mesure antérieure, distincte de cette reprise ; les 187 tokens partiels ne constituent pas une nouvelle comparaison longue.
3. **Quelles limites physiques restent ?** Les poids, caches, états et applications partagent les **16 Gio** du CPU/GPU. Le run long historique relu demande **1 964 699 264 octets logiques par forward de décodage**, avec un cache de poids de 64 Mio, et **2 730 appels `eval` existants par pas**. À volume logique inchangé, 3/5/10 tokens/s impliqueraient **5,89/9,82/19,65 Go/s sur ce chemin logique** : estimation de budget, pas débit SSD mesuré. Réutilisation et spéculation pourraient réduire les octets par token final accepté ; un préfill accéléré ne suffit pas à le démontrer en décodage.

Les lectures logiques, compteurs de stockage du processus et octets NAND ne sont pas confondus. Les temps hôte de lecture et de synchronisation incluent des attentes et se recouvrent : ils ne s’additionnent pas pour produire un temps GPU. Les **153 Go/s annoncés pour la mémoire unifiée M5** ne sont pas un débit de stockage. Ni plafond thermique ni limite NAND mesurée ne sont établis ici. Les calculs locaux sont dans [l’analyse physique](research/m5_validation/evidence/recovery-20261027-ultra/physical-limits.json) ; sources primaires : [Apple](https://www.apple.com/macbook-air/specs/), [mémoire unifiée MLX](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst), [évaluation différée MLX](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst).

Les preuves brutes, configurations, états, journaux, logits, tenseurs et résultats négatifs restent conservés dans les dossiers locaux `gpu-attempt-*`, `router-*`, `campaign-recovery-*` et `supervisor-recovery-*`. Les inventaires privés de processus ne sont pas présentés comme des preuves publiques.

Une archive locale sélectionnée de 39,2 Mo (artefact local : `Preuves-M5-selection-verifiee.zip`) et son manifeste (artefact local : `Manifest-selection-preuves-M5.json`) réunissent 1 023 membres vérifiés CRC et SHA-256 : les quatre workers acceptés de la préparation 003, leurs logits et scellements, 288 fichiers GPU, 323 sources/protocoles historiques et les captures du routeur. Cette sélection ne contient pas les essais inachevés ; elle ne constitue pas la campagne de 100 jobs achevée. SHA-256 du ZIP : `d5c29f90b896bae40eb6d80ee2801f8dd15108602332f1f4bf3fbd92fb30611e`. La [preuve numérique compacte](research/m5_validation/evidence/recovery-20261027-ultra/router/public-numeric-proof.json) et son [vérificateur autonome](research/m5_validation/evidence/recovery-20261027-ultra/router/verify-public-proof.py) se rejouent sans modèle ni GPU.
