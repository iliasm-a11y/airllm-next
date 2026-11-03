# GLM-5.3 Flash sur M5 — rapport de reprise clos

Campagnes des 2–3 novembre 2026, arrêt des modèles et contrôle de nettoyage à 08:26:43 UTC le 3 novembre. Checkpoint original Q2_K_S, MacBook Air M5 16 Gio. Travail expérimental sans fusion dans `main`.

**L’accélération considérable recherchée n’est pas obtenue.** Le meilleur bénéfice comparatif répété sous gardes reste celui de la phase initiale V3, **+9,1355 % médian de décodage sur huit sorties** pour Tail5 : **0,151471331 tok/s médian**, meilleur passage **0,153744680 tok/s**. Aucun gain GPU long ni stabilité longue GPU n’est qualifié.

La reprise apporte des preuves fonctionnelles précises : **CPU + nocache fidèle sur 32 pas**, **un chat CPU + nocache terminé à EOS avec 88 IDs et réponse visible `OK`**, et **un reset natif 2×4 répétable en logits/écritures observées**. Tail12 GPU est fidèle sur 32 pas mais échoue ensuite dans son cycle de débit. Tail5 plus petit échoue avant sa FID32 complète. Le probe réel agents 1/600 est rejeté pour changement d’alimentation ; l’agent 1 n’a produit qu’un délimiteur après 15 min 37 de préremplissage, et l’agent 600 est annulé. Il ne qualifie ni réponse utile ni 600 agents.

## Les cinq réponses de la mission

1. **Goulot principal : résidence et déplacement des poids denses, avec calcul et I/O encore entremêlés.** Les matrices denses portent 77,3 % des octets d’opérandes distincts inspectés et environ 69,3 % du temps d’un ancien diagnostic sérialisé. L’expérience chaude d’une vraie matrice et les lectures OS observées appuient ce diagnostic. La fraction exacte d’attente SSD séparée du calcul n’est pas mesurée.
2. **Vitesse réellement établie : 0,151471331 tok/s médian et 0,153744680 tok/s au meilleur passage du cycle court V3 Tail5.** Le bénéfice médian face à son propre CPU est +9,1355 %, sur deux passages candidats de huit sorties. Le record ancien 0,199422 tok/s appartient à un autre runtime et à un cycle ensuite rejeté pour swap ; il n’est ni durable ni celui de cette livraison. Aucun nouveau maximum long qualifié n’est démontré.
3. **Le GPU bénéficie au cas court, mais aucune génération GPU complète ou accélération longue n’est qualifiée.** Les gains courts V3 tête/Tail5 sont respectivement +4,8135 % / +9,1355 %. Tail12 passe la fidélité 32 puis son B1 de débit s’arrête après 13 IDs sur le swap ; Tail5 FID32 s’arrête après cinq sorties. Aucun gain du cache dense n’est établi. L’EOS réussi concerne le CPU + nocache.
4. **RONIN/600 agents ne sont pas qualifiés durablement.** La configuration conserve 600 identités logiques, un slot physique partagé/adaptatif et le secours CPU10/GPU0. Le reset natif 2×4 est validé sur sa paire. Le probe agents ne valide qu’un prompt natif d’agent 1 avec sortie partielle inutile, puis une annulation 600. La stabilité de 600 agents, les sessions serveur et des décisions utiles répétées ne sont pas établies.
5. **Les paliers +20 %, 0,5, 1, 3 ou 5 tok/s ne sont pas démontrés.** Le scénario idéal de 0,682 tok/s à 1 Go/s suppose une forte résidence utile, ignore calcul et latences, et n’est pas mesuré. Les seuls 16 Gio nominaux n’interdisent pas universellement 0,5 tok/s ; les mesures de ce Mac ne permettent aucune promesse à ce niveau. Le Fanxiang n’est pas validé. Pour viser une génération sensiblement plus rapide avec ce checkpoint, il est raisonnable d’envisager davantage de mémoire utile pour les poids et un meilleur service de stockage, puis de requalifier les résultats ; aucun débit de cette future configuration n’est garanti.

**Configuration retenue : production CPU10/GPU0 existante conservée.** Le profil CPU10 + nocache reste expérimental et limité aux cas qualifiés ci-dessous. Aucune voie GPU/cache nouvelle n’est activée en production par ces preuves.

## Environnement, référence et intégrité

Matériel enregistré : Apple M5, GPU huit cœurs, 16 Gio de mémoire unifiée, macOS 27.2. Le runtime llama.cpp est épinglé à `3f8af15a76ac8675812e856da45454bafc90f12c`. La recette V4 complète à seize patches a été construite ; dix bibliothèques régulières et treize sources natives modifiées ou nouvelles, en-têtes inclus, sont identifiées par les manifestes.

La référence **CPU10/GPU0** de ces expériences associe l’arithmétique CPU sur dix threads à un socle I/O hérité commun : `mmap_demand`, `expert_prefetch`, `dense_prefetch` activés. Les captures, les commandes ABBA et les configurations EOS observées concordent sur ces trois hints. Les valeurs OFF par défaut de l’API autonome du benchmark ne s’appliquent pas aux invocations observées, qui les passent explicitement. Ce paramétrage expérimental commun n’est pas une baseline de production universelle. Le gate ne compare pas exhaustivement tous les flags ; la concordance ciblée est établie dans les artefacts réellement exécutés.

Ces trois hints ajoutent ou retirent des conseils de mémoire virtuelle, sans changement arithmétique intentionnel des poids, routes, graphe ou ordre des réductions. Ils ne prouvent ni résidence, succès de préchargement ou asynchronisme. Nocache, lookahead et dense-lock sont des switches distincts.

Le checkpoint est **`ggml-org/GLM-5.3-Flash-GGUF`, révision `119034bc3d9d76eb41f91508e6f9fb19fde42672`, Q2_K_S**, soit 110 027 352 864 octets :

| Shard original | Octets | SHA-256 intégral enregistré |
|---|---:|---|
| `…-00001-of-00002.gguf` | 9 430 112 | `22e3599ff1cac02e854269b610030800e319151e1ed01838d7087f20b187091d` |
| `…-00002-of-00002.gguf` | 110 017 922 752 | `61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4` |

La reprise dispose d’une vérification coordonnée intégrale des deux payloads, transcrite dans `resume-current-checkpoint-integrity.json`, sans second balayage des 110 Go pour ce rapport. L’extrait ne fournit pas d’horodatage propre à cette opération ; aucune heure de hash n’est inventée. Les captures vérifient séparément les tailles/dates ; les reçus et le postflight final lient les identités à cinq champs. Poids et quantification sont conservés ; aucune variante plus petite ou re-quantifiée n’est substituée.

### Stockage et gardes

Le checkpoint et les mesures sont sur le **SSD interne APFS**. Le relevé de métadonnées du 3 novembre à 07:11:59,621155 UTC retourne une liste vide pour les disques physiques externes. Le contrôle final à 08:29:53,444778 UTC retourne également une liste externe vide (`resume-storage-mounts-final.json`). Le Fanxiang n’était pas monté lors de la phase initiale et n’est pas validé par la reprise. Aucun débit logique de lecture ou de cache n’est présenté comme sa vitesse.

Les gardes sont conservés : **8,5 Gio de RSS**, **32 Mio de croissance du swap système**, arrêt sur pression critique, erreur de mesure ou timeout. Les plafonds temporisés adaptés aux sondes restent enregistrés. Les gardes de puissance exigent couvercle ouvert, source connue constante et accord des horloges ; l’assertion temporaire d’inactivité est liée au processus possédé et n’outrepasse pas la fermeture du couvercle. Aucun réglage permanent d’alimentation n’est modifié.

RSS et footprint sont distinctes et non additives. Le swap est global ; sa croissance ne désigne pas des octets exclusivement imputables au modèle. Zéro croissance ne signifie pas zéro swap préexistant. Les pics échantillonnés peuvent manquer des pointes entre relevés.

## Chemins évalués et cause de divergence Metal

Le chemin hybride général antérieur atteignait un écart maximal de logits de **0,5348455906** et **364 éléments de routes divergents**, malgré quatre argmax identiques. Les différences documentées incluent les réductions HC/Sinkhorn CPU accumulées en double face aux réductions FP32 Metal, et les activations temporairement quantifiées en Q8_0 sur CPU face au chemin Metal général en FP32. L’égalité des poids ne suffit donc pas à préserver les routes.

La voie sélective conserve le contrat numérique testé : quantification temporaire d’activation et ordre d’accumulation ARM reproduits pour Q8, poids GGUF packed inchangés, placement limité aux opérandes admissibles et replis CPU. Les prototypes Q2/MLX restent des preuves de composants ; ils ne constituent pas une accélération complète GLM mesurée. Aucun gain de composant n’est transformé en gain du modèle.

Les voies de reprise combinent, selon le cas, nocache, anticipation dense bornée à 32 Mio et placement sélectif. Le hint n’est pas une garantie d’I/O physique. Le cache MoE explicite est nul dans les nouveaux cas présentés. Le dense-lock256 est un autre mécanisme, désactivé hors de ses essais et capable de refuser puis annuler sa rétention. Aucun cache brut de plusieurs gigaoctets n’est déployé sous les gardes pour 600 agents.

| Sélection GPU statique | Matrices | Poids packed sélectionnés | Portée |
|---|---:|---:|---|
| Tail12, blocs 33–44 + tête | 82 | 2 592 538 624 B | FID32 passe ; cycle de débit rejeté |
| Tail5, blocs 40–44 + tête | 35 | 1 482 883 072 B | Capture32 arrêtée après cinq sorties |

Ces volumes décrivent le plan, sous plafond statique de 3 Gio. Ils ne sont ni tout Q8 du modèle, ni une borne de RSS/working set ou de mémoire GPU, ni une preuve de résidence simultanée. Seuls les produits matrice-vecteur admissibles suivent la voie Metal.

## Performances acceptées et campagnes rejetées

### Bénéfice court V3 accepté sur sa portée

Les trois ABBA V3 sont complètes : deux passages par bras, huit sorties par passage, mêmes IDs et arrêt sur limite. Les douze essais restent sous gardes, sans croissance de swap. L’audit civil/compteur trouve 0,043405–0,055323 s d’écart et aucun événement de veille dans leurs intervalles selon le journal sélectionné. Cela ne prouve pas toutes les conditions thermiques ou une stabilité longue.

| Candidat V3 | CPU médian, tok/s | Candidat médian, tok/s | Écart décodage |
|---|---:|---:|---:|
| CPU + nocache + anticipation 32 | 0,137902860 | 0,137456182 | −0,323907 % |
| Tête Q8 Metal + streaming | 0,141110962 | 0,147903386 | +4,813534 % |
| Tail5 Q8 Metal + streaming | 0,138791928 | 0,151471331 |**+9,135548 %** |

Chaque candidat est comparé à son propre CPU contemporain. La dérive A0→A3 est −8,41 %, −1,02 % et −3,59 % respectivement ; deux observations par bras ne donnent pas un intervalle de confiance robuste. La baisse du temps total médian n’est que 2,88 % pour la tête et 6,97 % pour Tail5. La plage CPU historique fournie de 0,1085–0,1664 tok/s n’est pas un bras apparié de ces expériences.

Le **0,199422 tok/s** historique PR #19 est conservé comme résultat isolé d’un autre runtime, dans une campagne ensuite rejetée pour swap. Il n’est pas le maximum durable du runtime livré. La meilleure mesure comparative répétée sous gardes dans les cycles courts V3 reste **0,151471331 tok/s médian**, pic observé **0,153744680**.

### Pourquoi les nouvelles comparaisons ne donnent pas de gain

| Essai / campagne | Résultat clos | Conséquence |
|---|---|---|
| Ancienne E8 V4 avec cache | Deux B traversent 14 entrées en veille/DarkWake |−23,94 % brut invalide : ni gain ni régression |
| Reprise E8 « awake » | Passage batterie→secteur avant toute sortie A0 | Aucun débit ; trois essais restants non lancés |
| Reprise E8 secteur | A0 complet 8 ; B1 produit 8 mais rétention cache refusée | Configuration E non acquise ; B2/A3 non lancés |
| ABBA32 Tail12 | A0 complet 32 ; B1 arrêté après 13 IDs sur swap | Aucune médiane A/B ; B2/A3 non lancés |
| Replay CPU32 indépendant | Arrêt sur swap avant première sortie | Aucune reproductibilité indépendante 32 qualifiée |
| Tail5 FID32 plus petit | Arrêt sur swap après cinq sorties | Pas de FID32 complète ; aucune ABBA5 lancée |

Les B interrompus de l’ancienne E8 ont des durées civiles de 5 911,965176 et 1 640,284975 s contre 163,383954 et 287,488937 s au compteur interne. Les preuves numériques antérieures restent indépendantes de ce cycle physiquement invalide.

La rétention dense E8 secteur demandait 103 noms et 268 255 232 octets de pages. Les 22 premiers appels de verrouillage réussissent et vérifient 66 420 736 octets, puis la politique refuse l’admission **avant le 23e appel**. Les 22 verrous sont annulés sans erreur ; **zéro octet reste retenu**. Il ne s’agit pas d’un errno `mlock`. La pression 2 aux 38 échantillons appuie une cause probable, mais la sous-raison native exacte — pression, mesure ou marge — n’est pas journalisée. Le débit GPU brut 0,146040849 tok/s ne valide donc pas la configuration E demandée.

L’ABBA Tail12 clôt A0 avec 32 IDs, **0,1294932732970116 tok/s côté hôte** et **0,12949057438944442 tok/s côté serveur**. Ces horloges/points de mesure distincts ne sont pas une incohérence. B1 n’a que 13 IDs et aucun terminal : delta swap de 42 598 400 B au garde, pic global échantillonné 44 498 944 B, au-delà des 32 Mio. Son serveur sort 0 mais son runner reste en échec. Aucun débit partiel n’est utilisé pour un gain 32 contre 13.

Le plan utilise un gate **A16 CPU et B32 Tail12**, couvertures explicitement différentes. A0 compare 16 IDs de préfixe ; B1 compare les 13 émis. Cela ne recapture pas 32 logits/états du témoin chronométré. Le replay CPU32 ensuite tenté échoue avant sortie : pic swap échantillonné **211 088 835 B**, RSS **5 988 663 296 B**. Son delta entre snapshots avant/après **379 647 427 B** couvre un autre intervalle et ne remplace pas ce pic.

Une fenêtre ultérieure sans modèle de 30,071 s montre sept relevés en pression 1, swap constant 1 074 003 968 B et disponible 5,596–5,768 Go. Elle ne qualifie pas un chargement. Tail5 est ensuite arrêté avec pic swap **295 111 229 B**, RSS **5 957 844 992 B**, cinq sorties et aucun marqueur complet. C’est un échec mémoire de capture, pas une divergence de logits démontrée. Les essais GPU longs sont clos sans qualification ; aucun essai raccourci ne remplace ces échecs de 32 sorties.

## Preuves numériques et fonctionnelles réellement acquises

| Cas | Couverture close | Résultat et limite |
|---|---|---|
| V4 CPU16 puis Tail5+cache16 | 2 478 080 logits, routes/routeurs/KDA, 3 792 écritures observées | Exact en bits sur les trajectoires capturées |
| CPU32 + nocache vs CPU32 naturel existant | 4 956 160 logits  ; 12 432 routes, 447 552 routeurs, 20 611 072 KDA  ; 7 584 écritures | Exact, candidat rejoué à entrées identiques ; pas répétition CPU indépendante |
| Tail12 GPU32 vs même CPU32 existant |Même couverture 32 | Exact, sans qualification de débit/stabilité longue |
| Reset natif CPU+nocache 2×4 |4 paires de 154 880 logits,948 écritures observées par requête | Répétabilité interne après clear, pas sessions serveur |
| Chat CPU+noca, requête courte |88 IDs, terminal EOS cohérent, réponse visible OK | Fonctionnel sur cette requête ; pas 88 logits/routes/états comparés |

Toutes les valeurs des captures comparées sont finies, avec erreur absolue maximale et RMSE nulles. Les logits et `result_output` se recouvrent et ne sont pas additionnés comme des preuves indépendantes. Pour les captures32, les 7 584 événements `CPY`/`SET_ROWS` couvrent 5 253 365 760 octets logiques ; la portée de stockage 5 345 460 224 octets peut inclure des espacements. Les états non observés ne sont pas certifiés.

### CPU + nocache : un EOS réussi, sans conclusion causale de vitesse ou mémoire

La première tentative sans nocache s’arrête après 78 IDs sur `SWAP_LIMIT` : 60 882 944 B de croissance globale observée, pas de terminal ni EOS. La tentative nocache-1 est volontairement annulée avant sortie après interférence de compilation signalée par la coordination ; elle est conservée comme annulée, pas comme échec du modèle.

La tentative **nocache-2** produit **88 IDs complets et un événement terminal**, `stop_type=eos`, non tronquée. Le texte visible après le bloc de raisonnement est **OK** ; le champ `content` de clôture est vide et aucun ID n’est inventé pour le terminal. Les 78 premiers IDs sont exactement ceux de l’essai interrompu. Le même body et message utilisateur sont utilisés ; les configurations d’entrée ne diffèrent que par nocache, les fichiers isolés ajoutant leurs propres chemins.

Tous les contrôles EOS finaux passent : intégrité checkpoint/runtime, transcript, une charge native, une requête et un slot, gardes et puissance. Après réponse, SIGINT est demandé puis le serveur sort 0. Le pic RSS échantillonné retenu après flux est **6 019 579 904 B**, la dernière RSS **5 278 990 336 B**. Le **delta de swap du dernier échantillon**, **2 228 224 B**, n’est pas un pic global enregistré. La continuité secteur/horloges passe sur 1 352 relevés, dernier écart 0,004406129 s.

Ce succès unique lève le blocage fonctionnel pour ce chat court de 19 tokens de prompt. Il n’isole pas un bénéfice causal du hint, ne qualifie pas tous les prompts ou une performance répétée, et ne valide pas EOS GPU. Aucune durée de sortie partielle ou de diagnostic n’est convertie en record de débit.

### Reset 2×4 : même contexte, pas deux sessions RONIN

Le diagnostic CPU + nocache répète le prompt après **`llama_memory_clear` dans le même contexte natif**, deux requêtes naturelles de quatre sorties. Les quatre paires de 154 880 logits sont exactes : **619 520 positions comparées**, huit vecteurs finis validés au total. Les **948 écritures observées par requête** reproduisent les mêmes 948 paires ; les IDs et entrées suivantes concordent.

La RSS maximale échantillonnée est **6 941 556 736 B**, footprint 414 583 576 B, croissance de swap 0 avec déjà 1 369 115 197 B de swap initial, code 0. Les 142 relevés puissance/horloges passent. Cela ne compare pas à une référence extérieure, n’est pas une génération continue de huit tokens, ne qualifie pas le replay CPU32 ni une remise à zéro de sessions HTTP/RONIN ou 600 agents.

## Probe réel agents 1/600 : sortie partielle, annulation et nettoyage vérifié

Le registre contient 600 identités logiques, mais seules les tentatives des agents 1 et 600 sont engagées. Le scheduler conserve une limite physique de un et le même PID natif. Les deux appels de chargement haut niveau ne signifient pas deux chargements physiques : les compteurs natifs enregistrent **un chargement et une requête terminée**.

Pour l’agent 1, le terminal natif évalue **447 entrées** et enregistre **937 168,513 ms de préremplissage**, soit **15 min 37,168513 s**. Le budget d’une sortie donne uniquement **ID 154842, `</think>`**, avec `stop_type=limit`, non tronqué, décision **partial**, sortie non vérifiée. Le terminal limité existe, mais **aucune réponse utile ni OK** n’est fournie. Ce temps n’est pas transformé en débit de décodage ; le budget d’un token ne qualifie pas une décision autonome.

Le prompt natif réel de l’agent 1 contient **ALPHA et pas OMEGA**. Cette preuve de séparation porte seulement sur ce prompt. L’agent 600 conserve un contexte hôte OMEGA et une décision **cancelled**, sans terminal natif enregistré. Aucun prompt natif 600 ni résultat 600 n’est certifié à partir du contexte hôte ou du compteur scheduler `completed=2`.

Le probe global échoue quand le garde observe **secteur→batterie à 08:23:47,255844 UTC**, couvercle ouvert ; la composante horloge passe. Aucun échec de garde RAM/SWAP n’est enregistré : `native.failure=null`, dernier échantillon pression 1 et delta swap 0. Ces données ne prouvent pas une cause RAM au rejet ou au préremplissage lent. Le pic RSS retenu est 6 800 736 256 B, dernière RSS 5 375 901 696 B.

Le retour natif **−9** est observé pendant le nettoyage forcé **après annulation** ; un crash antérieur n’est pas démontré. Le postflight à **08:26:43,189226 UTC** passe : sept sources figées, configuration/reçu, identités de shards à cinq champs, processus possédés disparus. Les quatre PID contrôlés ont disparu selon la coordination ; le JSON publie ce verdict de nettoyage agrégé. **Ce PASS de postflight ne transforme pas le probe échoué en succès.**

La mission d’intégration600 agents, l’isolation native de l’agent 600, les décisions utiles et leur stabilité durable restent **non qualifiées**. Aucune exécution des 600 agents n’a été réalisée. La configuration 600 logiques/un slot est conservée, sans activation GPU/cache de production.

## I/O, capacité et optimisations écartées

L’union des opérandes du graphe capturé ferme **684 matrices denses fixes : 9 444 458 496 B**, plus **2 774 532 096 B d’experts sélectionnés**, soit **12 218 990 592 B distincts par pas** sur la trace. Ce sont les plages déclarées d’opérandes, sans chevauchement compté deux fois ; ce n’est pas une mesure de lectures physiques ni un minimum universel de toute autre implémentation.

Une vraie matrice dense de 35,65 Mo passe de 15,78 ms à environ 0,32 ms en répétition résidente. Les probes 1 Gio enregistrent environ 1,075–1,087 Go de lectures OS du PID au premier passage, puis 270 336 B lors de la répétition immédiate. La différence inclut cache/résidence et les conditions d’exécution. Les temps de lecture seuls et leurs ratios logiques ne sont pas une certification de vitesse SSD. On ne soustrait pas les secondes cœur du temps mural pour inventer une part I/O.

Un cache des huit experts précédents par couche occuperait 2 774 532 096 B. Sur la trace, il réutilise 47,1514 % des demandes entre tokens. Avec tous les poids denses résidents, il laisserait environ 1,466302 Go d’experts manquants par token ; à une interface hypothétique de1 Go/s, calcul/latences ignorés, le scénario idéal serait **0,682 tok/s**. Dense + cache représentent déjà **12,219 Go, 11,380 Gio**, avant système/états/staging, au-delà de la RSS 8,5 Gio pour un cache brut résident dans le processus. Les caches OS et la compression ont d’autres portées ; la seule RAM nominale ne prouve pas une impossibilité universelle.

Pour `U` octets distincts requis et `C` octets utiles déjà couverts par tous les caches, le service minimal conditionnel est `max(U−C,0)`. `C` est une couverture en octets originaux utiles, pas une RSS interchangeable. Les cibles suivantes, à 1 Go/s, ignorent encore le calcul :

| Cible | Couverture utile nécessaire au minimum |
|---|---:|
| 0,5 tok/s | 10,219 Go |
| 1 tok/s | 11,219 Go |
| 3 tok/s | 11,886 Go |
| 5 tok/s | 12,019 Go |

La compression sans perte pilotée n’est pas intégrée : ratios Q8≈0,918–0,929, BF16≈0,78, Q2≈0,978, avec décodage d’environ 1,4–1,56 Go/s sur de petits échantillons. Aucun gain global du format n’est démontré. La piste dense-lock32 Mio est close sans nouvelle campagne : une sélection théorique restreinte aux 103 noms existants ne retire que 0,060062166 s sur l’ancien diagnostic 6,280404787 s même si tout son temps disparaissait, soit **0,9656 % conditionnel**, sans optimum global ni mesure réelle. Les refus d’admission et les protections sont conservés.

### Unités CPU et limites des compteurs

L’audit Apple+psutil confirme les temps `proc_pid_rusage` en **unités Mach timebase**, pas en nanosecondes déjà converties. Sur cet hôte : `CPU_s = delta(user_ticks+system_ticks) × 125/3 / 1e9`. Le ratio est observé, pas universel. Sur le même intervalle clos, 448,909312083 s converties concordent avec 448,909302768 s de psutil, à 9,315 microsecondes près entre appels non atomiques. [Source Apple épinglé](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/recount.h#L120-L123), [conversion Mach](https://developer.apple.com/library/archive/qa/qa1398/_index.html).

Ces secondes cœur somment les threads et peuvent dépasser le temps mural. Les intervalles de requête et d’échantillonnage ne sont pas interchangeables. Les lectures OS du PID incluent pagination/anticipation ; les compteurs de bloc sont globaux et incluent d’autres processus. Aucun n’est une attribution NAND ou un taux exact de cache. Les buffers Metal terminés observés incluent copies/événements et leur capture n’est pas exhaustive.

## Reproduction

La [recette native et les commandes de diagnostic](https://github.com/iliasm-a11y/airllm-next/blob/research/glm53-m5-dense-stream-final/research/glm53_gpu_v3/README.md) décrivent le checkout épinglé et les patches. Les fichiers `experiment.json`, `plan.json`, configurations et wrappers de chaque essai conservent la commande exacte, les paramètres et les identités. Les captures32 nécessitent la référence naturelle CPU32 avant le rejeu candidat ; le reset2×4 est un contrôle séparé. Le code du probe de chat/agents et ses sources gelées sont conservés avec les preuves et dans la PR20 distincte. Une reconstruction possédant de nouveaux SHA binaires doit produire ses propres preuves ; aucune qualification ne lui est transférée automatiquement.

La livraison locale fournit un index et un manifeste liant chaque fichier brut à son chemin d’origine. Les commandes doivent être adaptées aux chemins de restitution des deux archives, sans réduire les gardes ni substituer le modèle. Les instructions sont une procédure de reproduction, sans garantie que l’état mémoire du Mac permettra de terminer un essai.

## Livraison, publications et traçabilité

Le travail reste expérimental, porté par la [PR #21](https://github.com/iliasm-a11y/airllm-next/pull/21). La [PR #20 préexistante](https://github.com/iliasm-a11y/airllm-next/pull/20) est liée comme repère distinct ; elle n’est pas remplacée par ce rapport. Le relevé clos du 3 novembre 07:40:02 UTC donne [#17](https://github.com/iliasm-a11y/airllm-next/pull/17), [#18](https://github.com/iliasm-a11y/airllm-next/pull/18) et [#19](https://github.com/iliasm-a11y/airllm-next/pull/19) ouvertes, en brouillon et non fusionnées. La reprise conserve la chaîne de recherche et ne documente aucune fusion dans `main`.

L’identité finale de publication est consignée dans **`Publication.json` séparé** ; ce rapport ne fabrique aucun SHA de commit final. Les SHA du checkpoint, des binaires et des preuves sont des identités distinctes de ce commit.

La livraison historique V3 reste inchangée. Ses résultats invalidés sont conservés avec leur audit, et non écrasés par la reprise :

- [Rapport historique V3](https://github.com/iliasm-a11y/airllm-next/blob/c6b0148020ae2ed9a2fbca9ef80cf90436674e62/research/glm53_gpu_v3/README.md)
- Archive historique V3 — 543 444 287 octets — `$TASK_ROOT/outputs/GLM53-M5-v3/Preuves-brutes-M5.tar.gz` (livraison locale)
- Manifeste historique — `$TASK_ROOT/outputs/GLM53-M5-v3/Manifest-preuves.json` (livraison locale)

L’empreinte enregistrée de l’archive historique est `e86cd45f19c175e8dde54ee2b69d1a9e0619a379dc5be2a44abecc033ab82f7e`. L’archive n’a pas été relue ou reconstruite pour ce rapport. La reprise est livrée séparément ; ses preuves détaillées identifient les fichiers bruts, leurs paramètres et leurs limites.

| Preuve textuelle close, sous `work/v3` | SHA-256 |
|---|---|
| `resume-current-checkpoint-integrity.json` | `be3dfa9af2e15bad5009158b1a3fce3744e2434444fe3f3b19f881af7ae3edc2` |
| `resume-storage-mounts.json` | `1fb4fab0dd48692f4a0408bc0c8844a88dbbe5ea798d70677ccf4d80bec29aec` |
| `resume-pr-chain-readback.json` | `868272ccc5f2fd2743f8bae1b260ec53c9dabf660e8322fd73885a82fd7b54ab` |
| `power-continuity-audit.json` | `7415e2865e32cd6e1a068458b908f977d5477e646903386aaa68eac5505f01d0` |
| `performance-results.md` | `eeab5943c70d00947102079c4c0815d3676a8f4bba6096d08ad8ba32f14df006` |
| `fidelity-v4.md` | `ed38dfd8ebe3121997f6c5cdbcb44f84bfd161d5ca49e0fee3ed53198b5b11ca` |
| `resume-short-results.json` | `494363a3938e809fd011fdb40f8ea2039fc651c397719eb4f3f52675b30298c2` |
| `resume-dense-cache-admission-audit.md` | `d85080084204c551558bdb0511a67a93bcd87f5dad403ee89236babdfe9dca47` |
| `resume-eos-cpu-results.json` | `34fa3b75fa0d6b44eaf95d053042c0925ea90ed4e9da7a72f893bc892773fba6` |
| `resume-nocache32-results.json` | `c9c38e6f9c34ebc9338506fc66cc5c1c36617c75a03070e49426d7bf40759b66` |
| `resume-eos-nocache-results.json` | `c1f3ce37f5d90c70b9b7e5cd1a8c1b597e18fff0baef669625eee7be33917b4d` |
| `resume-tail12-fidelity32-results.json` | `49c08adb13accf40ade13a73b7b71ab2c97f6e217325ee0d66f4a223b2449c1c` |
| `resume-tail12-campaign32-results.json` | `782b2f8ce5d17c83b21e7a736849348a1addd68095a2d626d2f07edc018c3f1a` |
| `resume-cpu32-repeat-admission.json` | `5a93a9a8a1581ac234a3e76e52f06c36a73d6dc46f6057b3bd9692c54af223ca` |
| `resume-tail5-fidelity32-results.json` | `75b9aefa363a6d839f06c6312b8c23e09dc13833f3876a686eb469dce178b6a5` |
| `resume-cpu-nocache-reset-results.json` | `d5edd3fe510ee99f7372e752139b5a4bd48ea4805389b89b8d492312f7beecb9` |
| `resume-real-agents-nocache-results.json` | `978320a3756612936f0cf033849ebad26a6275c759881c3fb2eae61b3553372a` |
| `agent-nocache-root-postflight.json` | `bddd4467840e66a05a15f6033ac43dcbf312415265abfd5957b3c84f5ffd162d` |
| `rusage-cpu-units-audit.json` | `35b6fb367693dbfe65069eb135d90cf6df9ab3c3dd11665d47c4071308a86b7c` |
| `legacy-io-hints-fidelity-scope.md` | `ccbece450401d00ac37a989416711fd8a3877cfc6a0149a471a8a67c5d95f7d3` |

Les modèles ont été arrêtés et les identités/nettoyages contrôlés ; aucun nouveau test matériel n’est requis pour interpréter les phases closes de ce rapport. Les opérations non exécutées ou non qualifiées sont explicitement signalées. La rédaction et la transcription finales n’ont lancé aucun modèle, build, test, archive, lecture de poids ou modification Git/sourcecode.

Métadonnées closes publiées : [index et empreintes](evidence/published/resume-publication-index.json). Les chemins privés sont remplacés par des préfixes symboliques ; les SHA des sources brutes originales restent conservés dans le pack JSON.
