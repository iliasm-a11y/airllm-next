# HyperStream M5 — résultats de récupération et limites

Rapport généré le **2026-10-28T20:07:37+01:00**, à partir d’instantanés locaux lus sans modèle, GPU, hash complet du checkpoint ni signal. Seuls les JSON bornés déjà lus sont hachés pour relier les preuves. État C5 lu : **2026-10-28T19:22:07+01:00** ; agrégat État : **2026-10-28T19:54:22.947127+01:00** ; synthèse : **2026-10-28T18:54:24.804497+00:00**. Ces fichiers peuvent avoir des dates différentes. L’état scellé C5 commande le tableau ; les agrégats ne remplacent pas les preuves d’origine.

**22 expériences acceptées et scellées, 88 jobs scellés sur 100 emplacements prévus ; 1 exclusion(s), 0 rejet(s) de fidélité.** **La clôture vérifiée de toute la campagne n’est pas établie : 88 jobs scellés + 4 emplacements TensorOps exclus + 4 jobs de spéculation libre profondeur 16 manquants + 4 emplacements de combinaison bloqués, sans exclusion encore enregistrée pour cette combinaison.** État enregistré : `deferred_or_interrupted` ; superviseur : `campaign_deferred_resources` ; dernier job actif déclaré : aucun job actif enregistré. Un statut persistant ne prouve pas, à lui seul, la vie actuelle d’un processus.

Les mesures C5 restent attribuées à **`a5d289704f436e59fc9b4aab5d62a1dba80f1347`**, configuration **`b14bafb51bbfd1cb03bb8d8ae76ce077567ede27711501ad652f21e0b2cf6f07`**. Le correctif permanent de manifeste **`5f0e0c5f376dae7da4f9f039b1ef51d7942e198a`**, arbre **`be2400848363406006190f2cdd8514e5c6a8a938`**, est une autre source, validée sur CPU. **Statut de cette révision : quatre workers courts M5 AB/BA scellés et vérifiés indépendamment ; 524 tests CPU réussis ; aucun rejeu complet des 100 jobs sur cette source.** Il n’a pas été substitué au moteur pendant C5. Configuration figée (preuve locale conservée, non incluse dans cette sélection publique) · État et scellements (preuve locale conservée, non incluse dans cette sélection publique) · `Vérification finale de campagne` (absent au relevé).

## 1. H2 réduit-il réellement le premier token d’environ 66 % avec une fidélité conforme ?

**Oui, avec `router_reference_chunk=16`, sur le prompt historique de 364 tokens mesuré.** C3 a donné −66,710812 % et −65,405253 % de TTFT en AB/BA. C5 répète le résultat sur la source A5 figée :

| Paire C5 | TTFT H1 (s) | TTFT H2 reference16 (s) | Réduction |
|---|---:|---:|---:|
| 0 | 154,691489 | 53,122647 | 65,658973 % |
| 1 | 155,238712 | 53,200001 | 65,730197 % |

Chaque paire courte compare **14 640 routes ordonnées et 744 960 logits émis identiques bit à bit**, avec les tokens exacts `[19, 17, 248046]` et EOS. Les deux paires C3 et les deux paires C5 portent sur la même tâche ; trois tokens ne mesurent pas un décodage soutenu. [Vérification C3](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-independent.json) · [Vérification courte C5](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/h2-reference16-final-independent.json).

La **génération libre** H1/H2 reference16 est aussi acceptée : **319 tokens jusqu’à EOS par worker**, deux paires, TTFT médian **37,240 → 15,666 s** (−57,93 %), débit de décodage **0,87652 → 0,87826 token/s**. La requête complète a un ratio C/T de **1,0590**. Chaque paire compare **79 214 080 logits bit à bit et 16 920 routes ordonnées**. Le préfill est plus rapide ; le débit de décodage reste voisin. [Extraction longue acceptée](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Resultat-long-H1-H2-M5.json).

Le H2 original garde son rejet : les experts 193/104 sont permutés aux rangs 7/8, couche 30 position 154. Les poids capturés sont identiques ; les activations diffèrent au maximum de 5,72205 × 10⁻⁶. H1 distingue les logits d’un ULP FP32 ; le bloc H2 de 128 produit une égalité exacte à −4,525256156921387. Le tri stable conserve alors 104 avant 193. Il n’y a pas d’égalité à la frontière 8/9 qui aurait activé l’ancien repli. Le contrôle sur les mêmes entrées et poids, calculé réellement par blocs contigus de 16 sur GPU, restaure 193 avant 104. Torch CPU FP32 sur 128 conserve l’ordre rejeté ; ce repli n’a pas été retenu.

L’option 16 réemploie le routeur GPU FP32 historique puis concatène les **positions de tokens et les poids associés**, sans tri des experts après coup ni injection FP64. Le mode 0 reste inchangé. La qualification ne s’étend pas automatiquement à tous les prompts, segmentations ou états internes DeltaNet/KV. [Captures H1/H2](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/router/h1-h2-comparison.json) · [Contrôle sur entrées identiques](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/router/controlled-reference.json) · [Provenance](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/router/source-provenance.json) · [Tri stable MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html).

## Les 25 expériences du protocole figé

C = contrôle, T = traitement. **C/T > 1 signifie une durée plus courte du candidat.** Les ratios sont les médianes des deux ratios appariés ; ils ne sont pas les rapports de TTFT médians arrondis. La requête inclut brouillon, calibration, vérification et replay éventuels ; le worker inclut aussi son initialisation instrumentée. Le **processus** inclut la durée contrôleur complète, avec les coûts avant le début des timers internes et après leur fin. Ces trois périmètres peuvent donner des conclusions différentes. Aucun chiffre de gain n’est affiché pour une ligne non acceptée et non scellée. Les réponses de trois tokens servent surtout au TTFT. La copie synthétique est distincte de la génération libre ; les longueurs effectivement acceptées sont affichées par ligne.

| Expérience | État / jobs scellés | Tâche / tokens finaux | TTFT médian C → T (s) | Requête C/T × | Worker C/T × | Processus C/T × | Logits émis |
|---|---|---|---:|---:|---:|---:|---|
| `h1_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 154,965 → 53,161 | 2,837 | 2,812 | 2,781 | bit à bit |
| `general_long_h1_vs_h2_reference16` | acceptée / 4/4 | Libre / 319 | 37,240 → 15,666 | 1,059 | 1,059 | 1,057 | bit à bit |
| `prefill16_vs16` | acceptée / 4/4 | Contexte 364 / 3 | 159,062 → 162,676 | 0,979 | 0,979 | 0,979 | bit à bit |
| `prefill32_vs16` | acceptée / 4/4 | Contexte 364 / 3 | 162,122 → 112,232 | 1,433 | 1,430 | 1,426 | bit à bit |
| `prefill64_vs16` | acceptée / 4/4 | Contexte 364 / 3 | 163,843 → 71,383 | 2,250 | 2,232 | 2,214 | bit à bit |
| `prefill128_vs16` | acceptée / 4/4 | Contexte 364 / 3 | 160,702 → 53,507 | 2,928 | 2,898 | 2,865 | bit à bit |
| `tensorops_vs_h2_reference16` | exclue / 0/4 | Contexte 364 / — | — | — | — | — | — |
| `reuse_tile2_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 57,921 → 62,091 | 0,933 | 0,933 | 0,934 | bit à bit |
| `reuse_tile4_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 53,187 → 55,954 | 0,953 | 0,953 | 0,955 | bit à bit |
| `reuse_tile8_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 53,997 → 57,294 | 0,944 | 0,944 | 0,945 | bit à bit |
| `reuse_tile16_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 53,634 → 57,514 | 0,935 | 0,935 | 0,936 | bit à bit |
| `grouped_transport_vs_h2_reference16` | acceptée / 4/4 | Contexte 364 / 3 | 54,264 → 52,563 | 1,032 | 1,031 | 1,029 | bit à bit |
| `speculation_depth2_vs_h2_reference16` | acceptée / 4/4 | Copie synthétique / 96 | 28,011 → 27,416 | 1,303 | 1,301 | 1,175 | tolérances conservées |
| `speculation_depth4_vs_h2_reference16` | acceptée / 4/4 | Copie synthétique / 96 | 27,905 → 27,932 | 1,353 | 1,351 | 1,215 | tolérances conservées |
| `speculation_depth8_vs_h2_reference16` | acceptée / 4/4 | Copie synthétique / 96 | 27,923 → 27,778 | 1,434 | 1,430 | 1,268 | tolérances conservées |
| `speculation_depth16_vs_h2_reference16` | acceptée / 4/4 | Copie synthétique / 96 | 27,412 → 28,334 | 1,108 | 1,107 | 1,017 | tolérances conservées |
| `legacy_vs_direct` | acceptée / 4/4 | Contexte 364 / 3 | 166,191 → 167,928 | 0,990 | 0,991 | 0,991 | bit à bit |
| `direct_vs_zero_copy_import` | acceptée / 4/4 | Contexte 364 / 3 | 166,279 → 166,546 | 0,998 | 0,998 | 0,998 | bit à bit |
| `gate_up_pipeline` | acceptée / 4/4 | Contexte 364 / 3 | 165,414 → 161,359 | 1,026 | 1,026 | 1,026 | bit à bit |
| `general_long_serial_control` | acceptée / 4/4 | Libre / 319 | 36,020 → 33,201 | 1,008 | 1,008 | 1,008 | bit à bit |
| `general_long_speculation2` | acceptée / 4/4 | Libre / 319 | 39,402 → 37,429 | 1,022 | 1,022 | 0,814 | bit à bit |
| `general_long_speculation4` | acceptée / 4/4 | Libre / 319 | 36,953 → 37,026 | 0,946 | 0,947 | 0,799 | bit à bit |
| `general_long_speculation8` | acceptée / 4/4 | Libre / 319 | 37,130 → 37,936 | 0,990 | 0,990 | 0,812 | bit à bit |
| `general_long_speculation16` | en attente / 0/4 | Libre / — | — | — | — | — | — |
| `combined_tensorops_grouped_spec8` | en attente / 0/4 | Copie synthétique / — | — | — | — | — | — |

Motifs consignés :

- `tensorops_vs_h2_reference16` : Some required full-grid cases failed compilation or numerics for tensorops/4.

Expériences restant sans résultat qualifié dans cet instantané : `general_long_speculation16`, `combined_tensorops_grouped_spec8`. Une combinaison dont les prérequis sont refusés ne devient une exclusion enregistrée qu’une fois ce statut écrit ; elle ne devient jamais un succès par extrapolation.

## 2. TensorOps, reuse et spéculation améliorent-ils le vrai débit ?

**Le bénéfice dépend du mécanisme et de la tâche.** TensorOps reste exclu par la qualification numérique : ses 32 cas compilent et s’exécutent mais ne satisfont pas la grille conservée. Les grilles matérielles comprennent 160 cas nominaux distincts ; trois répétitions ont produit 480 exécutions, dont une sur la source finale. Les trois rapports et gardes propres sont consignés dans l’agrégat GPU daté. Les 128/128 cas reuse de chaque grille contrôlée sont exacts bit à bit. Cela ne suffit pas à prédire le gain du modèle entier.

Les quatre tuiles reuse fidèles ont des ratios de requête de **0,933 à 0,953** sur la tâche courte : elles sont plus lentes, malgré leur exactitude. Le transport groupé et `gate_up_pipeline` montrent de petits écarts favorables, voisins de 3 % sur deux paires courtes. Leur précision instrumentale ne constitue pas une preuve statistique ni un gain de décodage long. Les contrôles série et les variantes de transport proches de l’unité donnent le contexte de variabilité.

**Sur copie synthétique**, la profondeur 8 donne **0,86789 → 1,39604 token/s** (1,6085×), 96 tokens finaux ; le ratio de requête est **1,4343×**, soit une durée réduite de **30,28 %**. Les lectures logiques de toute la génération passent de **207,481 à 107,285 Go** (−48,29 %). Les logits passent les tolérances conservées `atol=rtol=1e-4`, sans être tous bit à bit ; tokens et routes ordonnées restent exigés. La profondeur 16 est moins favorable : une profondeur plus grande ne garantit pas un bénéfice.

Le nouveau **contrôle série libre**, aux arguments strictement identiques, est scellé sur 319 tokens EOS : ratio de requête **1,008364×**, worker **1,008395×** ; débit médian **0,89420 → 0,89528 token/s**. Les deux paires ont chacune 79 214 080 logits bit à bit et 16 920 routes ordonnées. Cet écart caractérise le bruit entre répétitions ; il n’est pas une optimisation.

La spéculation **en génération libre** a maintenant 3/4 familles acceptées et scellées : `general_long_speculation2` : 1,0219× requête, 0,8139× processus, 0,92590 → 0,94334 token/s; `general_long_speculation4` : 0,9457× requête, 0,7993× processus, 0,91049 → 0,85762 token/s; `general_long_speculation8` : 0,9897× requête, 0,8118× processus, 0,93977 → 0,93116 token/s. Ce sont leurs résultats propres ; les résultats de copie ne leur sont pas transférés. Une baisse du temps interne ne suffit pas à conclure à une baisse du temps processus. L’annexe conserve propositions, tokens de brouillon acceptés et octets logiques : un repli sans proposition ne justifie aucune attribution à une réduction d’I/O spéculative.

En durée processus complète, les ralentissements observés sont : `general_long_speculation2` : **+22,87 %**; `general_long_speculation4` : **+25,11 %**; `general_long_speculation8` : **+23,18 %**. La conversion utilise 100 × (1/ratio C/T − 1), sans attribuer l’écart à une cause encore non chronométrée.

Aucun débit soutenu médian accepté au-dessus de **1,584 token/s**, ni les objectifs 3/5/10 token/s, n’est établi dans ces lignes. Les deux paires par option ne donnent pas une estimation de dispersion fiable à long terme. Les mesures historiques 1,46747/1,584 token/s ont d’autres conditions et ne servent pas à attribuer un changement entre sessions à un seul correctif.

## 3. Quelles limites physiques restent ?

La mémoire physique est **16 Gio = 17 179 869 184 octets**, partagée entre CPU, GPU, système et applications. Le checkpoint de **23 938 321 664 octets = 23,938 Go** dépasse déjà ce pool ; le budget du moteur est inférieur à la RAM totale. Un modèle MoE ne relit toutefois pas tous ses poids à chaque token : le moteur sollicite les poids nécessaires avec un cache borné, ici 64 Mio. [Mémoire unifiée Apple](https://www.apple.com/macbook-air/specs/) · [Mémoire unifiée MLX](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst).

Sur la génération libre H1/H2 acceptée, **624 774 365 952 octets logiques pour 318 forwards de décodage**, soit **1 964 699 264 octets par pas**, sont identiques des deux côtés. H2 diminue le préfill, pas ce volume par pas. Les lectures logiques totales sont 655 926 703 232 contre 635 842 815 872 octets. Les compteurs OS de stockage du premier couple sont 76 698 894 336 contre 70 661 296 128 octets : un autre périmètre.

**1 Go = 10⁹ octets ; 1 Gio = 2³⁰ octets.** Demandes logiques du moteur, octets de stockage comptés par le processus dans l’OS et trafic physique NAND sont trois mesures distinctes. À volume logique par pas inchangé, 3/5/10 token/s demanderaient **5,894 /9,823 /19,647 Go/s sur ce chemin logique**. Ce calcul conditionnel n’est ni un débit NAND mesuré ni une borne matérielle du SSD. Les caches OS peuvent servir certaines demandes ; la spéculation peut réduire les demandes par token final accepté.

Les **2 730 appels `eval` existants par pas** et les durées hôte de lecture/synchronisation peuvent inclure des attentes et se recouvrir : on ne les additionne pas pour inventer un temps GPU pur. Les 153 Go/s de mémoire unifiée annoncés par Apple ne sont pas un débit SSD. Aucun plafond thermique ni débit NAND maximal n’est certifié. [Analyse physique extraite](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Analyse-limites-physiques-courante.json) (son ancien statut de scellement peut précéder l’état C5 courant) · [Évaluation différée MLX](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst).

## Comptabilité des essais et des preuves

C5 comporte **100 emplacements**, pas 100 workers accomplis. Le relevé des seuls dossiers de jobs prévus et sessions enregistrées trouve **106 tentatives avec lancement observé**, **91 jobs planifiés distincts** concernés, **98 rapports avec garde de sortie propre**, **7 gardes avec violation** et **0 gardes sans lancement observé**. Les fichiers de garde de hashing/kernels sont exclus de ces compteurs. Un enfant actuellement actif sans rapport/garde final peut ne pas encore être compté.

L’état contient **88 enregistrements de jobs**, dont **88 scellés** et **0 non scellés** ; **10 anciens enregistrements non encadrés, dans 4 groupes, sont consignés séparément comme écartés**. Les exclusions enregistrées représentent **4 emplacements**. Ces catégories et les tentatives brutes ne s’additionnent pas : des reprises répètent un même job. Les résultats qualifiés représentent **44 paires acceptées** ; les autres états restent visibles dans le tableau.

| Préparation | Instantané | Lancements bruts / uniques | Gardes propres | Jobs scellés | Paires acceptées / rejetées |
|---|---|---:|---:|---:|---:|
| campaign-recovery-001 | agrégat daté | 5 / 5 | 4 | 4 | 0 / 2 |
| campaign-recovery-002 | agrégat daté | 1 / 1 | 1 | 0 | 0 / 0 |
| campaign-recovery-003 | agrégat daté | 5 / 5 | 4 | 4 | 2 / 0 |
| campaign-recovery-004 | agrégat daté | 1 / 1 | 1 | 0 | 0 / 0 |
| campaign-recovery-005 | état C5 courant | 106 / 91 | 98 | 88 | 44 / 0 |

Un worker terminé, un rapport numérique accepté, un bloc scellé et une campagne entièrement vérifiée sont des étapes distinctes. Le renderer lit ces preuves enregistrées ; il ne rehache pas le checkpoint et ne rejoue pas le vérificateur.

## Incidents de supervision et reprise du manifeste

Le crash initial venait d’une exception pendant l’itération `psutil.process_iter(attrs=...)`, hors du `try` par processus. L’inventaire lit désormais les champs séparément, borne les diagnostics attendus et vérifie la naissance et la vie avec un objet frais. Un processus de modèle réellement vivant mais non identifiable diffère l’admission ; une identité incertaine ne permet aucun signal. Les erreurs de programmation ne sont pas absorbées comme de simples refus de ressources.

Un premier H1 long de C3 avait été interrompu à **187 tokens partiels** : `exe` indiquait `NoSuchProcess`, mais l’ancien code poursuivait vers `cmdline` refusé et fabriquait un Python inconnu à partir d’une naissance en cache. L’arrêt provenait de l’inventaire, sans violation RAM mesurée. Le correctif abandonne immédiatement ce PID sur disparition/zombie/ESRCH dans tout champ. Les 187 tokens ne sont pas un résultat accepté ; les blocs courts scellés restent valides.

Après interruption du contrôleur 009, l’état contenait 72 jobs scellés mais le manifeste global précédait les nouveaux fichiers S3. Le superviseur 010 a refusé ces fichiers non listés. La récupération à l’arrêt a vérifié **3 593 fichiers** et conservé **558 947 156 octets nouvellement indexés** ; 16 nouveaux jobs scellés et quatre expériences ont été revalidés à résultats identiques. État/configuration/sources sont restés identiques ; aucun worker interrompu n’a été promu. Trace de récupération (preuve locale conservée, non incluse dans cette sélection publique).

Le correctif permanent est séparé : préparation en lecture seule, seuls chemins de sessions enregistrés autorisés, anciens immuables conservés ; nettoyage des identités possédées puis **quiescence obligatoire**, revalidation des bornes et résultats, réindexation atomique et vérification stricte avant reprise. Un défaut fatal garde le code 1 ; il ne devient pas un refus mémoire 75. Une phase morte avant enregistrement de propriété reste refusée. Source **`5f0e0c5f376dae7da4f9f039b1ef51d7942e198a`**, arbre **`be2400848363406006190f2cdd8514e5c6a8a938`** : **524 tests CPU réussis, 24 sous-tests, 64 GPU ignorés** ; XML : 612 cas comptés, 0 échec, 0 erreur, 64 ignorés. La provenance de cette révision distingue le calcul Qwen, les deux vérificateurs numériques et les seuils de ressources conservés du module `experiment_guard.py`, dont le préflight de chemins est maintenant indexé. Les autres fonctions de la garde ont été comparées par AST et les fonctions numériques restent identiques par AST ; verify_gguf_speculation.py reste identique octet pour octet, tandis que verify_m5_validation.py change seulement son indexation des métadonnées Finder. L’ensemble `air_llm` n’est donc plus déclaré identique à A5. Provenance de la source de contrôle (preuve locale conservée, non incluse dans cette sélection publique). Le statut des quatre essais M5 ciblés de cette nouvelle révision est détaillé séparément ci-dessous ; C5 reste attribuée à A5. [Validation CPU du manifeste](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/tests-recovery-final-finder-cpu-admitted.xml).

Le lanceur 013 avait précréé ses sorties standard dans le dossier réservé au superviseur : refus **avant campagne ou modèle**, **0 job modèle lancé**. Les logs ont été redirigés vers `work/` et le dossier 013b est neuf. Cette erreur ne modifie ni source, ni configuration, ni échéance et n’ajoute pas de budget de tentative. Incident du lanceur (preuve locale conservée, non incluse dans cette sélection publique).

## Ressources, échéance et nettoyage

Les **4,71 Gio historiques étaient de la RAM disponible**, pas du disque. Les gardes conservées sont : réserve **3 Gio**, payload **768 Mio**, RSS **2 Gio**, croissance de swap **128 Mio**, **600 s par worker**, réserve disque **5 Gio**. L’admission interne exige **4 Gio disponibles, pression 1 pendant 5 s**, avec attente entre workers bornée à 300 s. L’admission externe renforcée de 013 a observé **6 Gio/pression 1 pendant 20 s**, `ready=true` ; elle ne certifie pas le résultat des jobs ultérieurs.

La fenêtre active de l’orchestration commence **2026-10-28T17:21:46+01:00** et finit **2026-10-28T20:21:46+01:00** : **3 heures**, **900 s d’attente externe cumulée**, au plus **4 nouvelles tentatives**, numéros [12, 13, 14, 15]. Les anciennes bornes de huit/dix heures et huit/douze tentatives sont des révisions historiques, pas des budgets remis à zéro au lancement 013b. Période humaine (preuve locale conservée, non incluse dans cette sélection publique) · Admission renforcée (preuve locale conservée, non incluse dans cette sélection publique) · [État du superviseur lu](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/supervisor-015-state.json).

Après l’arrêt mesuré de 013b sous pression mémoire 2, le budget initial de quatre heures fixé par l’agent a reçu une extension d’une heure et une tentative 014. Celle-ci a ensuite été refusée avant modèle pour deux métadonnées Finder modifiées. La récupération s’est achevée après cette période, pendant un intervalle sans observation : une fenêtre technique de trois heures a donc été enregistrée, sans prétendre à une nouvelle demande humaine. Les surveillances ont atteint leur plafond de **200 SIGTERM physiques ou réservés comme incertains**. La révision 03 a ensuite autorisé un dernier lot ponctuel de 20 signaux, portant ce sous-total à 220 ; aucun nouveau nettoyage n’est lancé. Chaque journal antérieur est conservé et lié par SHA-256. Fenêtre technique et compteurs conservés (preuve locale conservée, non incluse dans cette sélection publique).

L’index A5 a été récupéré sous verrou à l’arrêt : **8 360 fichiers vérifiés**, deux `.DS_Store` Apple seuls réindexés, toutes les autres entrées scientifiques et l’état/configuration/sources conservés. Aucun job non scellé n’a été promu. Trois métadonnées Finder seulement ont reçu un drapeau temporaire d’immuabilité ; leur restitution doit être confirmée par le journal final, distinct de cet audit historique. Le nouveau protocole exclut uniquement les bases Finder régulières reconnues et refuse un ancien manifeste qui les déclarait déjà comme preuve : il ne réinterprète pas C5. Audit Finder (preuve locale conservée, non incluse dans cette sélection publique).

Restitution Finder : statut `restored` ; les drapeaux d’origine des trois seuls fichiers concernés ont été remis après la sélection brute. Aucune donnée scientifique n’a été modifiée par cette restitution. [Journal de restitution Finder](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Restitution-drapeaux-Finder-finale.json).

Un watchdog externe couvre aussi le startup de l’ancien superviseur avant son timer interne. Il est lié au PID, à la naissance, aux arguments, à l’exécutable et à l’UID exacts, plus aux SHA de lancement/phase/source. Un seul SIGTERM est possible après l’échéance murale, jamais de groupe ni SIGKILL. La borne monotone et l’observation finale sont finies. État enregistré : `supervisor_gone_before_watchdog_signal` ; SIGTERM enregistré : `False`. Son journal ne constitue pas une preuve de vie instantanée. [Watchdog absolu](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Watchdog-absolu-015.json).

Le rapprochement final vérifie **512 SIGTERM effectivement envoyés**, **271 sorties naturelles** et **783 anciennes identités désormais absentes**. Zéro SIGKILL et zéro fichier supprimé sont déclarés. Il complète les clôtures manquantes des journaux historiques par une nouvelle observation PID/naissance, sans réécrire leurs compteurs. Les 200 signaux des surveillances plus les 20 du lot final épuisent le budget de phase de 220 ; le cumul historique de 512 comprend aussi les lots antérieurs. [Rapprochement final des sorties](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Bilan-nettoyage-M5-final.json).

| Journal | Statut | Arrêts consignés | Signaux consignés | En attente / survivants |
|---|---|---:|---:|---:|
| [Journal-nettoyage-M5-lot10.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot10.json) | `completed_clean_exit` | 20 | 20 | 0 / 0 |
| [Journal-nettoyage-M5-lot2.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot2.json) | `completed_clean_exit` | 8 | 8 | 0 / 0 |
| [Journal-nettoyage-M5-lot3.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot3.json) | `completed_clean_exit` | 16 | 15 | 0 / 0 |
| [Journal-nettoyage-M5-lot4.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot4.json) | `completed_clean_exit` | 4 | 4 | 0 / 0 |
| [Journal-nettoyage-M5-lot5.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot5.json) | `completed_clean_exit` | 74 | 39 | 0 / 0 |
| [Journal-nettoyage-M5-lot6.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot6.json) | `completed_clean_exit` | 52 | 30 | 0 / 0 |
| [Journal-nettoyage-M5-lot7.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot7.json) | `completed_clean_exit` | 64 | 35 | 0 / 0 |
| [Journal-nettoyage-M5-lot8.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot8.json) | `completed_clean_exit` | 20 | 19 | 0 / 0 |
| [Journal-nettoyage-M5-lot9.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-lot9.json) | `completed_clean_exit` | 90 | 90 | 0 / 0 |
| [Journal-nettoyage-M5-surveillance-v2.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-surveillance-v2.json) | `signal_budget_exhausted` | 184 | 92 | 0 / 0 |
| [Journal-nettoyage-M5-surveillance-v3.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-surveillance-v3.json) | `deadline_expired` | 68 | 34 | 0 / 0 |
| [Journal-nettoyage-M5-surveillance-v4.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-surveillance-v4.json) | `actual_signal_budget_exhausted` | 86 | 44 | 1 / 0 |
| [Journal-nettoyage-M5-surveillance.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5-surveillance.json) | `manual_stop` | 44 | 30 | 0 / 0 |
| [Journal-nettoyage-M5.json](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/cleanup/Journal-nettoyage-M5.json) | `completed_clean_exit` | 52 | 52 | 0 / 0 |

Les journaux lus ne sont plus marqués en cours ; leur statut demeure celui de ces instantanés. Les totaux des en-têtes ne sont pas additionnés sans déduplication. Les journaux déclarent zéro fichier supprimé et zéro SIGKILL ; les applications personnelles, les processus Codex principaux, les modèles et preuves restent protégés. La tentative avortée `Incident-nettoyage-8.json` est une intention sans signal, séparée des actions réalisées. Une hausse de RAM ou de disque disponible n’est pas attribuée exclusivement à ces opérations. Les inventaires privés de processus ne sont pas publiés dans ce rapport.

Le dernier essai long de C5 reste interrompu : `MemoryError: Worker deferred by verified resource guard: macos_memory_pressure`. Les sorties partielles de cette session restent brutes, sans promotion en benchmark accepté. Les quatre workers courts de la préparation finale sont un contrôle distinct.

## Annexe — spéculation et périmètres de temps mesurés

Seuls les workers des blocs spéculatifs **acceptés et scellés** figurent ici. Un compteur absent reste « — ». Les propositions et tokens de brouillon acceptés permettent de distinguer une vraie spéculation d’un repli série. Les lectures logiques couvrent toute la génération, préfill inclus.

| Expérience / worker | Tokens / arrêt | Propositions | Tokens brouillon acceptés | Forwards cible | Motif de désactivation | Octets logiques |
|---|---|---:|---:|---:|---|---:|
| speculation_depth2_vs_h2_reference16/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth2_vs_h2_reference16/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 60 | 56 | 40 | — | 132 339 871 872 |
| speculation_depth2_vs_h2_reference16/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 60 | 56 | 40 | — | 132 339 871 872 |
| speculation_depth2_vs_h2_reference16/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth4_vs_h2_reference16/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth4_vs_h2_reference16/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 79 | 66 | 30 | — | 121 407 686 912 |
| speculation_depth4_vs_h2_reference16/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 79 | 66 | 30 | — | 121 407 686 912 |
| speculation_depth4_vs_h2_reference16/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth8_vs_h2_reference16/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth8_vs_h2_reference16/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 91 | 75 | 21 | — | 107 285 246 848 |
| speculation_depth8_vs_h2_reference16/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 91 | 75 | 21 | — | 107 285 246 848 |
| speculation_depth8_vs_h2_reference16/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth16_vs_h2_reference16/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| speculation_depth16_vs_h2_reference16/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 100 | 83 | 13 | — | 112 812 709 376 |
| speculation_depth16_vs_h2_reference16/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | 100 | 83 | 13 | — | 112 812 709 376 |
| speculation_depth16_vs_h2_reference16/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 96 / eos | — | — | — | — | 207 480 783 488 |
| general_long_speculation2/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |
| general_long_speculation2/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation2/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation2/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |
| general_long_speculation4/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |
| general_long_speculation4/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation4/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation4/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |
| general_long_speculation8/baseline-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |
| general_long_speculation8/candidate-p00 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation8/candidate-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | 0 | 0 | 319 | `empty_draft_proposals` | 655 926 703 232 |
| general_long_speculation8/baseline-p01 (preuve locale conservée, non incluse dans cette sélection publique) | 319 / eos | — | — | — | — | 655 926 703 232 |

Le temps **processus** vient de `controller_wall_seconds`, la garde de `elapsed_seconds`, le worker de `worker_wall_seconds` et la requête du tour de `total_seconds`. Leurs points de départ diffèrent. L’écart processus moins garde inclut ce qui se produit avant la mesure de garde et après celle-ci ; sa cause détaillée n’est pas établie. L’initialisation instrumentée est conservée séparément : cet écart ne lui est pas attribué par défaut.

| Expérience / worker | Requête (s) | Worker (s) | Garde (s) | Processus contrôleur (s) | Initialisation instrumentée (s) |
|---|---:|---:|---:|---:|---:|
| `speculation_depth2_vs_h2_reference16/baseline-p00` | 138,627 | 139,445 | 139,759 | 142,221 | 0,736 |
| `speculation_depth2_vs_h2_reference16/candidate-p00` | 107,196 | 108,019 | 108,325 | 122,119 | 0,768 |
| `speculation_depth2_vs_h2_reference16/candidate-p01` | 107,015 | 107,820 | 108,119 | 121,677 | 0,748 |
| `speculation_depth2_vs_h2_reference16/baseline-p01` | 140,580 | 141,429 | 141,807 | 144,250 | 0,770 |
| `speculation_depth4_vs_h2_reference16/baseline-p00` | 139,960 | 140,841 | 141,203 | 143,691 | 0,796 |
| `speculation_depth4_vs_h2_reference16/candidate-p00` | 102,125 | 102,954 | 103,258 | 116,786 | 0,776 |
| `speculation_depth4_vs_h2_reference16/candidate-p01` | 103,718 | 104,546 | 104,896 | 118,511 | 0,774 |
| `speculation_depth4_vs_h2_reference16/baseline-p01` | 138,627 | 139,462 | 139,809 | 142,278 | 0,753 |
| `speculation_depth8_vs_h2_reference16/baseline-p00` | 135,082 | 135,928 | 136,312 | 138,739 | 0,762 |
| `speculation_depth8_vs_h2_reference16/candidate-p00` | 97,015 | 97,915 | 98,238 | 113,354 | 0,845 |
| `speculation_depth8_vs_h2_reference16/candidate-p01` | 94,688 | 95,530 | 95,874 | 109,306 | 0,790 |
| `speculation_depth8_vs_h2_reference16/baseline-p01` | 139,798 | 140,612 | 140,982 | 143,440 | 0,734 |
| `speculation_depth16_vs_h2_reference16/baseline-p00` | 136,279 | 137,127 | 137,454 | 139,903 | 0,765 |
| `speculation_depth16_vs_h2_reference16/candidate-p00` | 127,000 | 127,815 | 128,137 | 141,568 | 0,750 |
| `speculation_depth16_vs_h2_reference16/candidate-p01` | 121,476 | 122,282 | 122,616 | 136,081 | 0,744 |
| `speculation_depth16_vs_h2_reference16/baseline-p01` | 138,754 | 139,625 | 139,957 | 142,374 | 0,789 |
| `general_long_speculation2/baseline-p00` | 376,140 | 377,376 | 377,957 | 393,989 | 1,065 |
| `general_long_speculation2/candidate-p00` | 362,196 | 363,271 | 363,783 | 484,711 | 0,965 |
| `general_long_speculation2/candidate-p01` | 387,756 | 388,830 | 389,346 | 503,574 | 0,954 |
| `general_long_speculation2/baseline-p01` | 389,868 | 391,179 | 391,811 | 410,376 | 1,016 |
| `general_long_speculation4/baseline-p00` | 394,003 | 394,973 | 395,429 | 405,150 | 0,827 |
| `general_long_speculation4/candidate-p00` | 429,902 | 430,935 | 431,398 | 522,458 | 0,938 |
| `general_long_speculation4/candidate-p01` | 388,471 | 389,459 | 390,077 | 491,567 | 0,833 |
| `general_long_speculation4/baseline-p01` | 378,771 | 380,378 | 381,047 | 404,622 | 1,452 |
| `general_long_speculation8/baseline-p00` | 369,919 | 370,940 | 371,399 | 384,100 | 0,871 |
| `general_long_speculation8/candidate-p00` | 375,886 | 376,815 | 377,279 | 478,686 | 0,830 |
| `general_long_speculation8/candidate-p01` | 383,072 | 384,004 | 384,475 | 481,146 | 0,837 |
| `general_long_speculation8/baseline-p01` | 381,308 | 382,262 | 382,754 | 395,140 | 0,809 |

**6 worker(s) candidat(s) n’ont proposé ni accepté aucun token de brouillon.** Un débit apparent plus élevé dans ces lignes ne prouve pas une économie d’I/O due à la spéculation. Comparer leurs octets logiques au contrôle correspondant avant toute attribution causale.

**Statut de l’analyse du coût de lancement :** Mesure isolée sur 375 entrées protégées et 1 537 sorties synthétiques : 77,549 s avant correction, 0,082438 s après. Avec 384 sorties : 11,006 s puis 0,044878 s. Un appel par cas, métadonnées réchauffées par la vérification du jeu ; aucun débit modèle, GPU ou NAND déduit. Le module de garde conserve ses autres fonctions et seuils. C5 reste figée sur A5.

## Vérifications finales de la portion disponible

L’audit indépendant partiel est `sealed_available_evidence_verified` et déclare `full_campaign_complete=False`. Il porte sur les jobs scellés disponibles, leurs sorties brutes, gardes, bornes de hash et comparaisons recalculées ; il ne résout pas les quatre jobs de spéculation libre profondeur 16 manquants ni les quatre emplacements de combinaison bloqués par leurs prérequis. [Audit indépendant partiel final](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Verification-independante-partielle-finale.json).

La révision finale `5f0e0c5f376dae7da4f9f039b1ef51d7942e198a` a fait l’objet d’une nouvelle préparation et de **quatre workers courts AB/BA**, puis d’une vérification indépendante ciblée : statut `accepted`, 4 workers scellés. Configuration distincte `7d769cc5b6c03bd971cd8677f34575bc050161bd0753b711e54a6a3b8a7f2f01` ; état `partial_selected_experiments_completed`. Ce contrôle matériel ciblé ne représente pas les 100 emplacements de cette nouvelle préparation. [Vérification ciblée de la source finale](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Verification-independante-source-finale.json).

Mesures propres à cette validation ciblée : paire 0 : 154,654316 → 52,891130 s (−65,800 %); paire 1 : 154,885478 → 54,070265 s (−65,090 %). Les résultats C5 historiques restent inchangés.

[Diagnostic final des ressources](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Diagnostic-ressources-M5-final.json).

[Bilan final du nettoyage](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Bilan-nettoyage-M5-final.json).

[Restitution des drapeaux Finder](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Restitution-drapeaux-Finder-finale.json).

[Mesure isolée du préflight de chemins](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Mesure-preflight-chemins-M5.json).

Portée et empreintes de la sélection brute finale (preuve locale conservée, non incluse dans cette sélection publique).

[Résumé de l’archive brute et de ses empreintes](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Resume-preuves-finales-M5.json).

La sélection brute finale contient **2018 fichiers de preuves**, plus son manifeste embarqué ; archive de **1,186 Go**, vérifiée par CRC et SHA après écriture. Quatre workers longs H1/H2 de 319 tokens EOS et les quatre courts de la source finale sont conservés avec leurs tableaux FP32 et leurs sources. **Ce ZIP est une sélection, pas l’archive complète de C5** ; il ne contient aucun checkpoint. Archive brute sélectionnée (preuve locale conservée, non incluse dans cette sélection publique).

## Provenance et condition de clôture

Le checkpoint attendu est `f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`, **23 938 321 664 octets**. Ce générateur reprend l’identité de la configuration et les bornes enregistrées ; il n’effectue aucun nouveau hash. Les anciennes préparations, résultats négatifs, sorties brutes, gardes et scellements restent dans leurs dossiers d’origine. [Agrégat État daté](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Etat-recovery-M5.json) · [Synthèse datée](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/final/Synthese-benchmarks-M5.json) · [Preuve compacte du routeur](https://github.com/iliasm-a11y/airllm-next/blob/research/hyperstream-recovery-m5/research/m5_validation/evidence/recovery-20261027-ultra/router/public-numeric-proof.json).

La clôture complète exige simultanément : tous les résultats prévus résolus, tous les jobs non exclus encadrés et scellés, un état de campagne terminal et un superviseur `completed_evidence_verified`. Le SHA-256 du JSON brut de configuration doit égaler celui de l’état et de l’identité du superviseur ; le SHA-256 du JSON brut de vérification doit égaler `supervisor.verification_sha256`, avec statut indépendant terminal accepté. Le vérificateur actuel ne renvoie pas de champ de configuration : son absence est admise par ces liens ; si un tel champ existe, il doit aussi concorder. Ces seuls petits JSON bornés sont hachés par ce renderer. L’absence d’une de ces preuves conserve la mention d’attente. **Aucun seuil numérique n’est relâché pour accepter une permutation d’experts.**

La [PR #14](https://github.com/iliasm-a11y/airllm-next/pull/14) est un brouillon de publication déjà créé ; ce rapport n’en a pas vérifié l’état distant. Sa génération locale ne constitue ni une publication ni une fusion.
