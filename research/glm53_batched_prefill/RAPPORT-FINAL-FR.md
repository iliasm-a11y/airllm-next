# RONIN M5 — rapport final des essais du 3 novembre 2026

**Résultats clos, portée de qualification explicitée.** Les résultats ci-dessous proviennent des reçus locaux du 3 novembre 2026. C16 est clos sur délai, sans réponse utile ni EOS ; A, B et C restent non qualifiés. Le nouveau build NoCopy et la sonde réelle NoCopy128 sont clos : les 118 séries sur les positions 0..127, les 136 alias et le checkpoint128 sont exacts. Cette preuve bornée ne qualifie ni un prompt complet, ni EOS, cache ou gain de performance. Ce document ne lance aucune expérience et ne change aucune source mesurée, archive, publication ou configuration de production.

## Verdict des essais clos

| Objectif | Preuve disponible | Verdict actuel |
| --- | --- | --- |
| A — préfill GPU complet, fidèle et reproductiblement plus rapide | Vraies commandes Metal ; fenêtre GLM de 128 tokens exacte avec CPU+GPU, sans préfill complet ni gain qualifié | **Non qualifié** |
| B — préfixe commun exact, isolation et gain net | Snapshot complet borné ; première requête restaurée exacte ; bras interrompu par la garde swap | **Non qualifié, cache désactivé** |
| C — deux agents distincts, réponses utiles et EOS naturels | CPU8 puis CPU16 sans cache : délai dépassé dès le premier agent ; aucun EOS utile | **Non qualifié** |

Ces trois critères sont indépendants. L’échec de B n’empêche pas le pilote fonctionnel C sans cache. Une réussite future de C ne qualifierait pas automatiquement la fidélité numérique du profil HTTP CPU16, le cache ou les performances GPU. L’observation de deux agents utiles jusqu’à EOS devra être distinguée de la réussite complète du retour agent 1 dans la séquence 1 → 600 → 1.

La production conserve **CPU10/GPU0, un moteur et un slot physique**. Le meilleur profil complet, fidèle, stable et plus rapide reste inconnu ; CPU8 est le témoin complet disponible, sans preuve qu’il soit optimal.

## Les douze réponses obligatoires

### 1. Le préfill GPU par lots accélère-t-il réellement GLM-5.3 Flash sur le M5 ?

**Aucun gain de préfill GLM complet n’est démontré.** Des providers Q8 réellement intégrés au dispatcher CPU ont soumis et achevé des commandes GPU. Les variantes scalaire, TensorOps et workers en attente bloquante passent leurs comparaisons de composants admissibles, mais leurs graphes résidents mesurés restent plus lents que les témoins CPU. Le provider NoCopy produit quelques ratios bruts favorables, sans comparaison temporelle répétée admissible : le contrôle alterné ABBA a échoué dès le premier CPU.

Les critères Metal originaux restent `atol=1e-3`, `rtol=1e-4`. Une différence avec le CPU bit à bit ne suffit donc pas à rejeter un composant GPU si elle reste dans ces tolérances. Inversement, une tolérance passée sur une multiplication synthétique ne prouve pas les routes, KDA, états ou logits du modèle complet.

| Composant clos | Résultat numérique et GPU réel | Portée temporelle |
| --- | --- | --- |
| TensorOps Q8, 21 cas dont 20 paires comparées | 2 903 040 sorties finies bit à bit ; 60 publications, 266 commandes dont une tuile avant refus fatal contrôlé | Ratios graphe GPU/CPU de 2,3524 à 25,3694 : plus lent |
| TensorOps avec workers bloqués, 21 cas dont 20 paires comparées | Même nombre de sorties, publications et commandes ; contrat de composant passé | Ratios de 1,8401 à 14,6291 : plus lent |
| NoCopy résident, 25 paires / 50 enfants | 3 002 176 sorties finies ; tolérances Metal passées ; 16 cas GPU × 5 itérations = 80 commandes et publications | Ratios bruts de 0,41795 à 30,6173, **non qualifiés** |
| Contrôle NoCopy ABBA | Premier témoin CPU arrêté à 136 773 632 B > 128 Mio ; natif −15, 0 commande GPU, 0 comparaison achevée | Aucun gain accepté |

Pour chacune des deux suites TensorOps ci-dessus, les 21 cas se décomposent en **12 paires avec GPU réel, 6 fallbacks CPU, 2 refus CPU et 1 refus fatal contrôlé**. Les 20 paires non fatales fournissent les comparaisons numériques ; les 60 publications viennent des 12 cas GPU × 5 itérations. Sur les 266 commandes achevées, 265 appartiennent aux cas normaux et une tuile est achevée avant le refus fatal, sans publication de cette sortie. Les ratios temporels présentés portent sur les 12 paires GPU, pas sur 21 accélérations.

Dans NoCopy, 24 paires sont bit à bit, dont les neuf paires de fallback/refus CPU. La seule paire différente est M8/K2048/N8 : **63 valeurs F32 sur 64**, maximum absolu `2,956390380859375e-5`. Ce n’est pas une « erreur d’un bit ». Les cinq hashes de réutilisation restent stables dans chacun des 50 enfants. Les 50 enfants et leurs auxiliaires d’éveil sont clos sans signal.

NoCopy utilise un mmap synthétique en lecture seule avec décalage de 64 octets. L’alias est observé et la copie explicite des poids vaut zéro ; cela ne mesure ni les déplacements internes du pilote, ni les copies d’activations ou de sorties. Les trois buffers fixes totalisent 4 751 360 B ; le pic d’allocation Metal observé est 40 878 080 B. La résidence du pilote reste inconnue. Les deux grandes formes M8192/K4096/N64 et N128 restent exclues de cette suite. La preuve de refus fatal d’un ancien provider ne se transfère pas au nouveau chemin NoCopy.

**NoCopy128 PROBE : fenêtre réelle de 128 tokens passée, portée incomplète.** Le superviseur dédié v2 a achevé ses huit phases en **8,649163083 s** : compilation, signature, cinq descriptions de paramètres et contrôle backend. Sources, secteur, fins des enfants et fermeture des deux verrous sont validés. Le helper obtenu porte le SHA-256 `273b4eb4706847486e650ddf5dc9badc3e66f2046bdc721e64d4ef71a1767fa7`. Ce build seul ne charge pas de modèle et ne prouve aucune arithmétique du modèle.

La tentative run1 a été bloquée par **ENGINE_BUSY**, puis run2 a refusé une configuration visant un autre paquet runtime ; toutes deux s’arrêtent avant lancement du modèle. Le refus de propriété n’est pas un refus mémoire ou secteur et n’autorise aucun arrêt d’un moteur étranger. Run3 utilise la copie exacte de la configuration NoCopy préparée et termine sous garde en **148,691776458 s**, avec **590 commandes N32 soumises, achevées et publiées**, alias des poids vérifiés et zéro copie explicite des poids. Il s’agit d’un vrai chemin CPU + GPU avec poids mappés hors mémoire, sur une fenêtre du modèle ; pas d’un préfill GLM complet sur GPU.

La comparaison stricte v5 retourne `identical_probe_window`, code 0 : **37 242 880 éléments logiques bit à bit dans les 118 séries**, tous finis, routes ordonnées exactes, plus **136 lignes d’alias KDA vérifiées octet par octet** sur quatre chunks. Le checkpoint opaque à la position 128 est exact : **155 160 708 B**, SHA-256 `956fd9a155e0d5c4ef9ee7eaf7058f85628eba52d9b75f762969908cdcc6b829`. Les deux vrais builds, leurs propres clés et les gardes closes sont revalidés. Les tolérances Metal initiales passent aussi, sans remplacer le critère bit à bit.

L’entrée complète et les 600 identités gelées sont conservées, mais **seules les positions 0..127 de la requête0 de 521 tokens sont évaluées**. Il n’y a ni projection finale des logits à la fin du prompt, ni génération/continuations, ni EOS utile ou restauration de cache. `complete=false` ; tous les flags full/A/B/C/EOS/cache/performance restent faux. Les versions v3/v4 ont refusé l’accès aux preuves avant comparaison numérique : borne de lecture du garde CPU8 historique, puis catalogue d’alias incomplet. La copie v5 admet seulement ce garde exact sous une borne de lecture **32 Mio de métadonnées**, et vérifie strictement les 34 alias réellement émis par chunk avant les 118 comparaisons inchangées. Cette borne de lecture n’est pas la limite de **croissance du swap de 32 Mio**, qui reste inchangée.

### 2. Quelle taille de batch/microbatch donne les meilleures performances fidèles ?

**Inconnue.** Le témoin complet CPU10/GPU0 possède `n_batch=n_ubatch=8`, contexte 2048 et un slot. Ses trois prompts gelés font 521 / 542 / 521 tokens ; le dernier rejoue exactement le premier prompt, sans prétendre représenter un historique fonctionnel ultérieur.

| Profil natif clos | Préfill observé | Cause de non-sélection |
| --- | --- | --- |
| CPU8 témoin | 1 285,763678 / 1 322,539045 / 1 314,183277 s | Complet, code 0, rejeu strict positif ; pas de recherche d’optimum démontrée |
| CPU16 original | Premier prompt : 820,050836 s | 154 880 logits différents, maxabs 1,61790836, état différent ; arrêt contrôlé |
| CPU16 queue corrigée | Deux prompts : 823,460550 / 876,084673 s, sorties brutes finales et continuations exactes | Retour 1 incomplet ; garde swap à 2 231,936556 s, +206,6875 Mio globaux |
| CPU32 avec fences | Aucun prompt complet | 118 séries comparées dans la fenêtre 128 ; divergence observée, candidat rejeté |
| CPU32 noBLAS v3 | 616,986133 / 636,737411 s | Premier prompt exact ; deuxième divergent après 512 ; annulation contrôlée |
| CPU32 noBLAS v4 | Premier prompt : 605,342729 s, sorties finales et continuations exactes | Deuxième arrêté à 448/542 ; garde swap à 1 180,247291 s ; queue corrigée non atteinte |

La sonde GPU N32 n’évalue que 128 tokens et ne concurrence donc pas le témoin CPU8 complet dans ce tableau. Les tailles 64/128 ne sont pas qualifiées sur GLM complet. Une comparaison de composants plus grands ne remplit pas cette lacune.

### 3. Combien de temps de préfill est économisé ?

**Aucune économie reproductible et admissible n’est qualifiée.** Le premier point V4 de 605,343 s est inférieur au point CPU8 gelé de 1 285,764 s, mais il appartient à un bras incomplet terminé en échec de garde, sans alternance contemporaine répétée. Ce n’est ni un gain retenu, ni un gain de décodage, ni un temps total pour deux agents. L’ancien chiffre de 937,169 s pour 447 tokens appartient à une autre campagne et ne sert pas de dénominateur à ces prompts de 521/542 tokens.

Pour la fenêtre128, le temps cumulé decode/synchronisation/captures du NoCopy GPU est **147,352371417 s**, face à **136,689729166 s** dans un ancien CPU32 noBLAS v4. Cette comparaison est **historique, non appariée et non répétée** ; elle ne mesure pas un coût de calcul pur et ne fournit aucun bénéfice GPU retenu. Les graphes résidents précédents sont également plus lents. Ces constats ne justifient pas une nouvelle campagne modèle longue pour clôturer ce point.

Le diagnostic a identifié deux raisons concrètes de ne pas sélectionner simplement le plus grand lot. Le scheduler peut choisir BLAS dès N32 : un composant réel [BLAS, CPU] a sélectionné BLAS à N32 et divergé sur 4 096/4 096 valeurs par graine, tandis que le CPU direct restait exact. Après retrait effectif de BLAS avant le modèle, la queue N30 a encore engendré deux partitions de 15 colonnes et un autre chemin arithmétique.

L’inventaire du vrai graphe recense **13 574 occurrences d’opérations Q8 ordinaires capturées**, et non 13 574 matrices de poids uniques. Le composant CPU-tail teste 42 cas synthétiques, deux graines, M8192/K4096 : les 19 791 872 sorties comparées par la politique V4 sont bit à bit face aux groupes de huit ; le calcul monolithique diffère dans 20 cas, maxabs `3,0994415283203125e-6`. Les comparaisons et SHA sont calculés en mémoire ; **les tableaux F32 bruts de ce composant ne sont pas exportés**. La queue 512→542 du modèle V4 n’a pas été atteinte avant son arrêt, donc cette correction reste non qualifiée sur ce prompt réel.

### 4. Les 208 tokens communs peuvent-ils être réutilisés avec un snapshot exact ?

**Une réutilisation exacte a été observée sur la première requête, pas qualifiée sur tout le protocole.** Le tokenizer et le template natifs Max ont confirmé 211 IDs communs sur 600 identités de recherche fraîches. La frontière candidate 208 possède trois sérialisations complètes identiques dans le témoin : **156 739 588 B, soit 149,478519 Mio**, SHA-256 `4c4d6f24b2cf3c4f30a7894b8609035960a3ec5687521c942aa6a578e2e5a3cd`.

Le mécanisme utilise la sérialisation séquentielle complète `LLAMA_STATE_SEQ_FLAGS_NONE`. Elle couvre la mémoire hybride persistante de la séquence : attention/KV, KDA récurrent R+S, indexer et métadonnées/positions natives. `PARTIAL_ONLY` omet des données supposées déjà présentes et ne constitue pas un snapshot autonome après reset.

Le snapshot expérimental est unique, immuable et limité à 256 Mio. La réservation comprend son payload et 16 Mio de scratch déclaré, sous les limites globales. Sa clé lie modèle, tokenizer, template, 208 IDs, paramètres et provenance du runtime. Les logits et le sampler ne sont pas sérialisés avec cette API : le contrat exige un suffixe privé non vide, un sampler neuf et une comparaison réelle de la continuation. Il ne permet pas une reprise arbitraire à partir des seuls logits du préfixe.

Restore8 a calculé le préfixe une fois, puis terminé la première requête restaurée avec logits/état/continuations exacts. La deuxième restauration API est rapportée, mais sa requête n’est pas achevée. Le run s’arrête sur swap global +39,125 Mio, au-delà des 32 Mio. **Aucun bras B complet positif : cache commun désactivé.**

### 5. Quel est le coût réel de sauvegarde et de restauration ?

Les chronomètres existants sont mesurés, mais leurs périmètres diffèrent et le bras Restore8 a échoué :

| Opération | Mesure et périmètre |
| --- | --- |
| Calcul initial des 208 tokens | 449,387450125 s, une fois dans Restore8 |
| Capture totale du snapshot | 0,061756958 s |
| Écriture par l’API native seule | 0,009878666 s |
| Première lecture par l’API native seule | 0,004909917 s |
| Deux lectures API cumulées | 0,018917125 s ; deux restaurations API, pas deux agents terminés |
| Premier reset + restauration + vérifications | 0,067238 s |
| Premier suffixe avec reset/restauration | 704,354400125 s ; coût initial commun et création exclus |
| Payload / réservation snapshot | 156 739 588 / 173 516 804 B |

Les sérialisations diagnostiques à 208 du témoin prennent 0,121350125 / 0,114084291 / 0,195911333 s. Elles ne sont pas assimilées au seul appel natif de sauvegarde. Le gain net sur deux réponses utiles reste `null`.

### 6. Le GPU et le cache partagé apportent-ils un bénéfice combiné ?

**Non mesuré.** CPU sans cache dispose d’un témoin diagnostique complet ; CPU avec cache possède un bras B interrompu ; le GPU complet et GPU + cache n’ont pas de bras positif. Aucun produit ou somme de gains de composants et de coûts API n’est utilisé pour annoncer un gain combiné.

### 7. Les agents 1 et 600 terminent-ils chacun une réponse utile jusqu’à EOS ?

**Non, sur les deux essais fonctionnels clos.** Le CPU8 a traité le vrai prompt complet de 522 tokens : préfill 1 238,015911 s, 171 IDs de sortie observés, puis délai de requête 2 700 s. Il ne fournit ni réponse finale utile ni EOS naturel. Agent 600 et retour 1 n’ont pas été lancés dans ce bras. Le code natif −9 est observé pendant le nettoyage possédé après délai, sans l’assimiler à un EOS ou à un crash non observé.

Le pilote original CPU16 sans cache a conservé 600 identités enregistrées, un modèle, un slot et les limites Max512 / 2 700 s par requête / 7 200 s globales. Le premier agent reçoit **523 tokens** et produit **245 IDs partiels**, puis `REQUEST_TIMEOUT`. Le préfill observé dans le journal natif est de **818,994569 s** ; aucun chronométrage SSE terminal complet n’existe. La campagne dure **2 719,097061166 s**, nettoyage compris. Aucun `</think>`, EOS naturel ou réponse finale utile n’est observé. **Agent 600 et retour 1 non lancés.** Le natif sort proprement avec le code **0 pendant le nettoyage** : ce code n’est pas un EOS. Le prompt de 523 tokens est distinct du prompt diagnostique gelé de 521 tokens ; aucune comparaison directe de gain n’est revendiquée.

Le résumé v2 est fondé sur les reçus clos. L’audit de parsing v1 cherchait les ressources dans `engine_after_stream.native`, écrit seulement après trois requêtes réussies ; ses flags négatifs ne sont donc pas des violations du watchdog natif. Le correctif `audit_agents_cpu16_v2.py`, SHA-256 `1f16a96c9344098d8f931e88a5bcd53ea9b38abfd7b9504a6ebd57ce1e998ca7`, lit les vrais snapshots load/avant-cleanup/après-cleanup pour les ressources, tout en conservant le snapshot de stream requis pour une réussite fonctionnelle complète. **Le correctif v2 a été exécuté** : ressources, RSS, slot, poids, nettoyage et alimentation passent ; `no_campaign_error=false`, sans EOS utile ni historique de retour. Le verdict reste `closed_evidence_rejected` et `C_global_qualification=false`. L’audit v1 et les sources originales sont conservés ; aucun stream réussi n’est synthétisé.

La validation finale exige séparément deux agents distincts, deux tâches effectivement accomplies et deux EOS naturels. Un budget de tokens, une fermeture `</think>`, quatre tokens diagnostiques identiques ou un texte partiel ne satisfont pas ce critère. Les classifications `partial`, `cancelled`, `error` et les arrêts de garde sont conservés.

### 8. Les contextes des agents sont-ils correctement isolés ?

**Les contrôles de structure et le rejeu natif existent ; la preuve fonctionnelle complète reste ouverte.** Les 600 identités et les prompts privés sont conservés. Le témoin natif vérifie le reset et le rejeu 0→2 avec mêmes IDs, logits, état final, continuations et écritures observées. Ce sont deux exécutions du même prompt gelé, pas une validation des historiques privés de RONIN.

Les deux campagnes fonctionnelles closes s’arrêtent dès l’agent 1. La séquence 1 → 600 → 1 avec l’historique réel accumulé, les marqueurs privés et les décisions stockées chez le bon agent n’est donc pas démontrée. Aucun couple de deux succès distincts n’est observé. Pour une future campagne, deux EOS utiles devront rester une observation séparée du retour 1 complet. Le cache commun ne sera jamais autorisé à contenir le suffixe ou l’historique privé d’un agent.

### 9. Les protections mémoire restent-elles respectées ?

**Les limites sont conservées et appliquées ; plusieurs candidats les ont dépassées et sont des échecs.** La limite native reste RSS 8,5 Gio et croissance de swap 32 Mio, avec pression/réserves, intégrité, secteur, continuité et identité des processus. Les suites de composants TensorOps/parked/NoCopy du tableau ont en plus leur plafond RSS de 128 Mio et leurs budgets Metal distincts. Ce plafond de composant n’est pas celui de la sonde GLM NoCopy128 : son run3 clos relève des gardes natives, avec pic RSS échantillonné **6 731 333 632 B**, croissance de swap observée **0**, secteur et intégrité passés.

Le témoin CPU8 termine sans croissance de swap observée. Restore8, CPU16 corrigé et V4 se terminent avec `SWAP_LIMIT` et natif −15 : respectivement +39,125 / +206,6875 / +36,25 Mio globaux. Ces chiffres ne doivent pas être renommés « gardes respectées sans dépassement ». Ce sont des trajectoires refusées par des protections inchangées. La hausse du swap est **un compteur global**, sans attribution exclusive au modèle.

Les anciens processus possédés des bras clos sont absents selon leurs reçus de cleanup. Pour V4, les PID 79771/79772/79783/79784 et leurs anciens descripteurs sont absents ; le descripteur UID parent est clos. C16 a ensuite utilisé puis libéré ce verrou : son reçu confirme l’absence des PID 88931/88910/88921 et l’acquisition indépendante des deux verrous après clôture. Son auxiliaire d’éveil sort avec −15, `owner_term`, selon la politique admise. Une disponibilité observée à cette clôture ne vaut pas disponibilité permanente : le premier refus NoCopy ultérieur atteste un nouveau détenteur du verrou. Le témoin CPU8 conserve sa limite historique d’acquisition UID tardive ; les superviseurs ultérieurs établissent la propriété avant lancement et l’héritage du descripteur.

C16 conserve **2 685 échantillons**, un pic RSS natif échantillonné de **7 155 351 552 B**, sous 9 126 805 504 B, et un watchdog natif `failure=null`. La pression finale est 1 et le natif rapporte une croissance de swap finale 0 ; **la véritable baseline native et le pic de croissance depuis cette baseline ne sont pas reconstruits indépendamment**, donc ce pic reste `null`. La continuité d’alimentation est validée. L’échec est fonctionnel par délai, sans échec de ressource rapporté. Aucun arrêt d’application personnelle, assouplissement des protections ou réinitialisation opportuniste d’une baseline swap n’est effectué pour obtenir un succès.

### 10. Quelle configuration est la meilleure pour RONIN ?

**Aucun nouveau profil optimisé n’est sélectionné.** La référence de production demeure CPU10/GPU0, un modèle et un slot. CPU8 est le témoin numérique complet de cette étude ; il n’est pas déclaré le meilleur lot possible. Le pilote HTTP CPU16 a testé C indépendamment du cache, sans obtenir de réponse finale : **son profil numérique reste non qualifié** et les résultats des helpers ne lui sont pas transférés.

Les providers GPU restent opt-in et OFF par défaut. Q2_K/MoE, KDA, mHC et les opérations non admises ne sont pas reclassées GPU grâce à la seule présence d’un provider Q8. Les adaptateurs de préfixe restent fermés en l’absence d’un vrai verdict B et de leur preuve de transport propre.

### 11. Le système est-il prêt pour une phase pilote contrôlée ?

**Les deux pilotes fonctionnels CPU sans cache sont clos sans qualification ; un pilote revendiquant A/B/C ou un gain optimisé n’est pas prêt.** Le dispositif a réellement chargé et réutilisé le modèle, protégé ses ressources et conservé ses reçus, mais le premier agent dépasse le délai dans les deux bras. Le code natif 0 de C16 et ses gardes sans échec rapporté n’effacent pas l’absence d’une réponse utile et d’un EOS.

Un candidat serveur V4 distinct a été préparé en source : même règle de chunks et fences dans l’assemblage natif, une seule requête HTTP, aucun échantillon privé jeté. Le splitter interne seul ne peut pas fusionner deux appels logiques ; l’assemblage doit aussi être conforme. Le nouveau profil refuse les cas hors contrat, enregistre ses vues effectives et termine sur erreur sans retry de géométrie différente. Sa revue a corrigé trois défauts de champs API, `swa_full` et sampler effectif. **Il n’est ni compilé ni exécuté**, ne possède pas encore les captures brutes nécessaires et n’hérite aucune qualification du helper V4 interrompu.

### 12. Quelles limites empêchent encore une utilisation rapide des 600 agents ?

Le modèle original d’environ 110 Go dépasse les 16 Gio unifiés disponibles. La fenêtre NoCopy128 démontre un calcul CPU+GPU fidèle sur ses seules positions observées ; elle n’apporte pas d’amortissement temporel qualifié ni la fin d’une réponse utile. Le préfill CPU8 des prompts gelés demande environ 21 à 22 minutes ; le décodage reste lent et les états hybrides réduisent la marge mémoire. L’augmentation des lots peut modifier le backend ou la géométrie arithmétique ; le cache complet n’a pas encore passé son protocole de bout en bout ; les kernels GPU rapides sur un composant ne prouvent pas un amortissement utile des poids du modèle hors mémoire.

Les compteurs OS de lecture et les temps CPU du processus sont disponibles sur certains bras, mais n’isolent pas les transferts physiques SSD, l’attente des poids, le calcul pur ou la synchronisation. Les valeurs non instrumentées restent `null`. Le SSD externe Fanxiang n’était pas monté au relevé matériel ; aucun bénéfice de cet appareil n’est mesuré. La présence de Metal 4 sur ce M5 ne démontre pas une accélération GLM complète ni une utilisation particulière de tout accélérateur matériel.

L’audit statique des captures ne trouve pas de collection native conservant leurs gigaoctets bruts : v3 garde une croissance swap observée de 0 malgré 3,27 Go logiques, tandis que v4 dépasse le swap avec 2,62 Go. Le contrôleur réécrit son garde de façon quadratique, mais son impact mémoire/swap et un éventuel effet de `F_NOCACHE` restent non mesurés ; cette lecture seule ne justifie aucun nouveau run ni reclassement d’un échec.

Un seul slot sert 600 identités logiques. Deux réponses réussies ne constitueraient ni 600 inférences simultanées, ni une mesure de débit ou de stabilité de 600 générations. La priorité reste une preuve fonctionnelle fermée, puis une variante complète fidèle et stable avec comparaison utile ; aucune nouvelle répétition longue identique n’est justifiée par une simple hypothèse de gain. Aucun autre modèle n’est prévu pour cette clôture ; les opérations restantes portent sur la consolidation et la livraison des preuves.

## Livraison et limites de reproduction

Le dépôt expérimental est `research/glm53-m5-batched-prefill`, PR #23 en brouillon, sans fusion de `main` ni basculement de production. Les archives et reçus sont des instantanés de révisions précises ; leur CI ne valide pas un shader, un nouveau helper ou un runtime reconstruit après leur gel. Le commit publié, sa CI et les empreintes de livraison sont consignés dans `LIVRAISON-M5.md` et `Empreintes-livraison-M5.json`. La CI documentaire ne qualifie pas le matériel.

| Artefact | Statut et limite |
| --- | --- |
| `outputs/Preuves-CPU8-M5.tar.gz` | **Créé et vérifié** : 1 585 638 687 B, 31 840 membres vérifiés et 18 340 hardlinks ; SHA-256 `6c99328cc3254369ee18bcd90b3f00b41c378b0947c8b92523720af67b9f1c21` ; témoin CPU8 et timeout fonctionnel, pas A/B/C |
| `outputs/Preuves-CPU8-M5.manifest.json` | Tailles/SHA, liens et portée du même instantané |
| Archive brute noBLAS v3/v4 | **Non créée, optionnelle et non promise** ; le plan source reste disponible, mais la réserve mission de 32 Gio doit être préservée ; taille et SHA finaux `null` |
| Packager noBLAS | Source v2 relue, non exécutée, journal de déduplication admis explicitement ; deux runs clos, allowlist stricte, verrous et vérification exhaustive. Cap interne 4 Gio / plancher interne 8 Gio ne remplacent pas la réserve globale mission de 32 Gio |
| Inventaire noBLAS v3/v4 | 12 476 noms avant consolidation, 5 893 354 065 octets logiques ; pas une taille d’archive ni une attribution APFS. Déduplication exacte ultérieure : 4 631 noms préservés, environ 2,908 Go de redondance logique consolidée |
| Composant Q8-tail | Reçus/hashes de calcul en mémoire ; **aucun tableau F32 brut à archiver ou recompter indépendamment** |
| Build NoCopy supervisé v2 | **Réussi**, huit phases / 8,649163083 s, binaire et binding des deux builds réels ; pas de modèle ni dispatch matrice GPU pendant ce build |
| Sonde NoCopy128 run1 / run2 | Refus avant modèle : **ENGINE_BUSY** puis incohérence du paquet runtime configuré ; aucun résultat numérique dans ces deux tentatives |
| Sonde NoCopy128 run3 / comparateur v5 | **Fenêtre0..127 exacte**, 118 séries, 136 alias et checkpoint128 ; 590 commandes N32 réelles, gardes passées ; aucun prompt complet, EOS, cache ou gain qualifié |
| Archives de sources existantes | À lire avec leurs manifests ; les travaux préparés après leur gel ne sont pas supposés déjà inclus |

L’archive CPU8 est un **livrable local**, et non une archive hébergée sur GitHub. Les métadonnées publiques ne rendent pas ses payloads bruts téléchargeables publiquement. La continuation après `f0acff65ca6091b7ef2aed8bb127b0c0696b1dce` est livrée dans `Sources-et-preuves-continuation-M5.zip` : sources exactes et petits reçus clos, avec index des SHA et tailles. Les grands payloads exclus restent identifiés comme locaux ; les candidats non compilés restent distincts des binaires mesurés.

Les gigaoctets de tenseurs des bras noBLAS v3/v4 et de la sonde NoCopy restent **locaux et absents des payloads publics GitHub**. Leurs métadonnées/chemins/SHA permettent d’identifier les preuves, pas d’en recomposer publiquement les valeurs brutes. La livraison de continuation privilégie les sources et petites preuves closes ; aucune archive complète d’environ 6 Go n’est promise avec seulement environ 32,7 Gio libres pour une réserve de 32 Gio. Le reçu de déduplication observe +1 385 766 912 B libres au niveau système ; cela reste distinct des 2,908 Go de redondance logique et ne constitue pas une attribution physique APFS exclusive.

Le modèle GGUF Q2_K_S original, le tokenizer et le runtime V4 épinglé restent des dépendances externes. Les poids, leur quantification et les bibliothèques mesurées ne sont pas remplacés par une archive de recherche. Le manifest runtime témoin est `7b0ac6dddabf542aae42d7fb8bb8762d6b98e1705dc423b09ad071112d0c8750` ; le helper CPU8 mesuré a pour SHA `62a7d2578c58e96839be2d4c79d8c414c8d535e5a1adac54820043f9e695f4b0`, le helper V4 interrompu `070d786e4f0690641468bc9f38c4525a885d4cf76b58ca830a621b9d5d73d951`.

Le binaire fourni pour ce même Mac conserve sa provenance tant que son hash et ses dépendances restent identiques. Une recompilation, un nouveau serveur ou une autre bibliothèque crée un nouveau profil et exige ses propres témoins. Le ZIP source est un extrait de recherche à appliquer au dépôt/base indiqués, pas une installation autonome. Les comparateurs stricts lient les vrais builds, sources, paramètres, guards et payloads : une preview128 ou un test de headers ne les remplace pas. La sonde v5 a passé son contrôle brut borné après clôture ; ce succès ne remplace toujours pas la comparaison d’un prompt ou d’un bras complet.

Les sources préparées pour l’HTTP, le noBLAS et NoCopy restent des outils de recherche. Le HTTP V4 ordinaire expose probabilités/logprobabilités, pas les logits F32 absolus ni les 118 séries ; une preuve de fidélité cross-runtime exige donc un serveur de recherche instrumenté, un nouveau build et ses propres captures. Les fichiers de slots peuvent aider à diagnostiquer l’état complet, sans combler les captures absentes.

## Récapitulatif des preuves closes

Reçus clos principaux : `work/native-full8-summary.json`, `work/native-full8-self-compare.json`, `work/agents-cpu8-summary.json`, `work/agents-cpu16-summary-v2.json`, `work/agents-cpu16-cleanup-audit.json`, `work/native-restore8-summary.json`, `work/native-tail-full16-summary.json`, `work/native-noblas32-diagnostic-summary.json`, `work/native-noblas-v4-cleanup-audit.json`, `work/native-noblas-v4-full32-first-request-check.json`, `work/gpu-tensorops-fixture-summary.json`, `work/gpu-parked-fixture-summary.json`, `work/gpu-nocopy-fixture-summary.json`, `work/metal-tensorops-q8-nocopy-abba-closed.json`, `work/cpu-q8-tail-dispatch-candidate/mechanism-summary.json`, `work/native-prompts-max/report.json`, `work/native-nocopy-build-supervisor-v2/build-run1/build-guard.json`, son `driver-build.json`, `work/native-noblas-nocopy-probe-compare-v5/actual-build-binding.json`, `work/native-nocopy-probe128-final-summary.json`, `work/native-nocopy-probe128-compare-v5.json`, `work/native-nocopy-probe128-run3/guard.json` et `work/agents-cpu16-closure-audit-v2.json`. Les métriques de livraison et manifests existants servent de références d’instantané ; les bases et captures ne sont lues par les audits qu’après leur clôture.

Les statuts ci-dessous concernent uniquement les preuves effectivement closes :

| Champ | Valeur à cette rédaction |
| --- | --- |
| Verdict C16 | **Échec REQUEST_TIMEOUT**, 245 IDs partiels, sans réponse utile ni EOS ; natif 0 au cleanup |
| Observation de deux agents distincts utiles/EOS | **Non obtenue** ; agent 600 non lancé |
| Retour agent 1, historique réel et isolation complète | **Non exécuté** ; pas de qualification de la séquence |
| C16 : ressources et cleanup | Pic RSS échantillonné 7 155 351 552 B / 2 685 mesures ; failure natif `null`, secteur valide ; anciens PID absents et verrous libérés selon reçu |
| Pic swap C16 depuis la véritable baseline native | `null` — non reconstruit ; valeur finale rapportée 0, sans extrapolation du pic |
| Correctif d’audit C16 v2 | **Exécuté, rejected** ; ressources/cleanup vrais, erreur de campagne conservée ; l’ancien audit v1 et les preuves sont conservés |
| NoCopy128 PROBE : build | **Réussi**, helper SHA `273b4eb4706847486e650ddf5dc9badc3e66f2046bdc721e64d4ef71a1767fa7`, binding des deux builds créé |
| NoCopy128 PROBE : fenêtre numérique | **Exacte sur0..127**, 37 242 880 éléments/118 séries, 136 alias, checkpoint128 exact ; `complete=false` |
| NoCopy128 PROBE : performance | **Aucun gain qualifié** ; 147,352371 s contre CPU32 historique136,689729 s, non apparié/non répété |
| Archive brute noBLAS : création, membres, taille compressée et SHA | `null` — non créée, optionnelle, réserve32Gio prioritaire |
| Publication : head, CI, fichiers inclus et empreintes | Voir `LIVRAISON-M5.md`, les index du ZIP et `Empreintes-livraison-M5.json` |

Ce rapport conserve **A, B et C non qualifiés**. Les clôtures négatives restent visibles ; aucun succès complet de modèle n’est reconstruit à partir d’un progrès SSE, d’une sortie propre, d’un build réussi, d’un composant fidèle ou de la seule fenêtre128 exacte.
