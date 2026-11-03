# Campagnes closes : CPU-stream8, tête Metal-stream8 et tail5-stream8

Transcription descriptive des JSON clos, sans exécution de
`summarize_observed.py`, de test, modèle, GPU ou build pour cette lecture.
Les trois campagnes CPU-stream, tête et tail5 sont closes. Les résultats sont
limités à **huit sorties**, soit sept intervalles de décodage par passage.
Ils concernent le runtime v3 à quinze patches identifié ci-dessous. Une
construction native ultérieure doit conserver ses propres résultats et preuves ;
aucune qualification n'est transférée automatiquement à une version v4.

Les trois cycles A/B/B/A sont complets. Leurs rapports enregistrent
`comparison_valid_on_stated_scope=true`, `memory_stability_qualified=true`
et `optimization_qualified=false`. Le dernier champ est un refus de
qualification automatique, pas un remplacement de l'interprétation des
mesures. La stabilité de longue durée, d'autres prompts et EOS ne sont pas
qualifiés par ces cycles courts.

## Charge et critères communs

Serveur neuf par passage ; CPU10 ; contexte 512 ; batch/microbatch 8 ; prompt
`La capitale de la France est`, six tokens évalués ; température 0, seed 42 ;
pas de réutilisation KV. A est CPU10/GPU0 de contrôle. B ajoute le réglage
nocache et l'anticipation dense de 32 Mio ; dans le second cycle, B ajoute aussi
la tête Q8 Metal compatible CPU. Dans le troisième, B utilise le sous-ensemble
Q8 tail5 compatible CPU, avec accès mappé sûr et le même streaming. L'effet
individuel des deux réglages de streaming n'est pas isolé par cette comparaison.

Tous les passages terminent huit sorties avec événement SSE final et
`stop_type=limit`, **sans EOS observé**. Les huit IDs sont identiques dans
tous les passages : `[12089,13,4929,9621,1788,650,21193,294]`.

Le débit est `(8−1)/(dernier token reçu−premier token reçu)`. TTFT, chargement,
requête entière et durée totale sont séparés. Les gardes restent RSS 8,5 Gio,
croissance globale du swap 32 Mio et pression mémoire. Les douze passages
enregistrent une croissance échantillonnée du swap de zéro, aucune violation,
et les statistiques des fichiers modèle inchangées. Le swap absolu système
préexistant n'est pas nul.

Les preuves distinctes A/B sur seize étapes sont vérifiées avant campagne :
2 478 080 logits et familles capturées identiques en bits, avec mêmes neuf
bibliothèques natives et identités de modèle. La preuve candidat comprend les
3 792 écritures d'état observées. Cette porte porte sur la trajectoire capturée
et son préfixe ; les logits complets et hashes d'états ne sont **pas** capturés
à nouveau dans les requêtes chronométrées. L'égalité des seuls huit IDs n'est
pas présentée comme une nouvelle preuve complète de fidélité.

## Résultat principal, médianes de deux passages par bras

| Cycle | A, CPU tok/s | B tok/s | Variation du décodage B/A | TTFT A / B | Requête A / B | Total A / B |
|---|---:|---:|---:|---:|---:|---:|
| CPU-stream8 | 0,137903 | 0,137456 | **−0,323907 %** | 23,024250 / 22,206919 s | 73,883392 / 73,245171 s | 84,479119 / 83,773478 s |
| Head-stream8 | 0,141111 | 0,147903 | **+4,813534 %** | 22,465818 / 21,784387 s | 72,074351 / 69,113106 s | 82,614719 / 80,239222 s |
| Tail5-stream8 | 0,138792 | 0,151471 | **+9,135548 %** | 22,800936 / 21,720748 s | 73,254142 / 67,945229 s | 84,002315 / 78,146111 s |

CPU-stream n'établit aucun gain de décodage. Son TTFT baisse de 3,549871 %,
mais ce changement de latence ne doit pas être annoncé comme un gain de
décodage. La requête médiane baisse de 0,863822 % et le total de 0,835284 %.

La tête Metal présente un petit gain local dans ce cycle : ses deux passages
sont plus rapides que les deux contrôles. TTFT −3,033189 %, requête entière
−4,108596 %, total −2,875392 %. Son chargement médian est plus long :
10,938066 contre 10,352865 s, soit +5,652555 %. Le gain de décodage ne doit
donc pas être recopié comme gain du temps total.

Tail5 présente le plus grand gain local des trois comparaisons, toujours face
à **son propre** contrôle A : +9,135548 % en décodage. TTFT −4,737472 %,
requête entière −7,247254 %, total −6,971480 %. Le chargement médian est
10,008732 contre 10,560816 s (−5,227665 %). Ses deux B dépassent ses deux A.
Ce n'est pas un essai directement alterné tête contre tail5 : la différence
entre leurs valeurs absolues ne permet pas d'attribuer seule le bénéfice
marginal des matrices GPU supplémentaires.

Aucun objectif de +20 % ou 0,5 tok/s n'est démontré par ces cycles. Deux
passages par bras et huit sorties ne donnent pas une estimation fiable de la
variance générale ou une qualification durable pour RONIN.

## Tous les passages et plages temporelles

| Cycle / passage | tok/s | TTFT s | Requête s | Chargement s | Total s | Pic RSS, octets |
|---|---:|---:|---:|---:|---:|---:|
| CPU-stream / 0-A | 0,143959 | 22,102295 | 70,727996 | 10,199920 | 81,108681 | 6 718 259 200 |
| CPU-stream / 1-B | 0,143878 | 22,402445 | 71,056990 | 10,476934 | 81,733154 | 6 672 695 296 |
| CPU-stream / 2-B | 0,131034 | 22,011393 | 75,433353 | 10,198905 | 85,813803 | 6 707 281 920 |
| CPU-stream / 3-A | 0,131847 | 23,946205 | 77,038788 | 10,610975 | 87,849558 | 6 625 771 520 |
| Head-stream / 0-A | 0,141831 | 22,246704 | 71,601953 | 9,862692 | 81,657511 | 6 701 039 616 |
| Head-stream / 1-B | 0,147752 | 21,539393 | 68,916617 | 10,624829 | 79,731716 | 5 984 468 992 |
| Head-stream / 2-B | 0,148055 | 22,029381 | 69,309596 | 11,251303 | 80,746728 | 6 093 242 368 |
| Head-stream / 3-A | 0,140391 | 22,684932 | 72,546748 | 10,843037 | 83,571928 | 6 630 735 872 |
| Tail5-stream / 0-A | 0,141327 | 22,849364 | 72,381100 | 10,664362 | 83,233010 | 6 702 055 424 |
| Tail5-stream / 1-B | 0,149198 | 21,351060 | 68,269292 | 9,812994 | 78,257574 | 6 145 818 624 |
| Tail5-stream / 2-B | 0,153745 | 22,090436 | 67,621165 | 10,204470 | 78,034648 | 6 205 030 400 |
| Tail5-stream / 3-A | 0,136257 | 22,752507 | 74,127185 | 10,457270 | 84,771621 | 6 705 774 592 |

Plages complètes par bras :

| Cycle / bras | Débit min–max | TTFT min–max s | Requête min–max s | Chargement min–max s | Total min–max s |
|---|---:|---:|---:|---:|---:|
| CPU-stream A | 0,131847–0,143959 | 22,102295–23,946205 | 70,727996–77,038788 | 10,199920–10,610975 | 81,108681–87,849558 |
| CPU-stream B | 0,131034–0,143878 | 22,011393–22,402445 | 71,056990–75,433353 | 10,198905–10,476934 | 81,733154–85,813803 |
| Head-stream A | 0,140391–0,141831 | 22,246704–22,684932 | 71,601953–72,546748 | 9,862692–10,843037 | 81,657511–83,571928 |
| Head-stream B | 0,147752–0,148055 | 21,539393–22,029381 | 68,916617–69,309596 | 10,624829–11,251303 | 79,731716–80,746728 |
| Tail5-stream A | 0,136257–0,141327 | 22,752507–22,849364 | 72,381100–74,127185 | 10,457270–10,664362 | 83,233010–84,771621 |
| Tail5-stream B | 0,149198–0,153745 | 21,351060–22,090436 | 67,621165–68,269292 | 9,812994–10,204470 | 78,034648–78,257574 |

Dans CPU-stream, le contrôle A passe de 0,143959 à 0,131847 tok/s
(−8,413437 %) et B de 0,143878 à 0,131034 (−8,927068 %). Le ralentissement
dans les deux bras domine la minuscule différence médiane ; il n'est pas
attribué à une cause précise.

Dans head-stream, A dérive de −1,015560 % et B de +0,205202 %. Les journaux
ne rapportent aucun niveau d'avertissement thermique ou de performance.
Ils ne mesurent toutefois ni température de puce ni fréquence effective ;
l'absence d'avertissement ne démontre pas une stabilité thermique parfaite.

Dans tail5-stream, le contrôle A dérive de −3,587124 % et B de +3,047427 %.
Le gain court reste positif dans les deux passages candidats, mais les
dérives de sens opposés interdisent de présenter +9,135548 % comme une
constante intrinsèque du moteur. Aucun avertissement thermique n'est enregistré
dans ce cycle non plus, sans mesure de température effective.

## Compteurs sur la requête entière, préfill compris

Les endpoints de compteurs encadrent **préfill + décodage**. Ils sont séquentiels
et non atomiques. Ils ne permettent aucune attribution exacte au seul
décodage ; les diviser par huit ne donne pas des octets physiques par token
décodé. Le chargement initial est hors de ces deltas.

| Cycle / passage | Lectures OS du PID, octets | Pageins du PID | Lectures bloc globales, octets | CPU cumulé du PID, secondes | Commandes GPU valides / union GPU s |
|---|---:|---:|---:|---:|---:|
| CPU-stream / 0-A | 116 761 759 744 | 7 185 550 | 118 189 473 792 | 417,748723 | 0 / 0 |
| CPU-stream / 1-B | 116 758 892 544 | 7 189 177 | 118 132 862 976 | 428,661818 | 0 / 0 |
| CPU-stream / 2-B | 116 758 028 288 | 7 193 838 | 118 236 958 720 | 465,228335 | 0 / 0 |
| CPU-stream / 3-A | 116 759 134 208 | 7 192 094 | 118 195 249 152 | 465,160604 | 0 / 0 |
| Head-stream / 0-A | 116 759 801 856 | 7 187 122 | 118 298 832 896 | 418,797597 | 0 / 0 |
| Head-stream / 1-B | 111 366 791 168 | 6 859 811 | 112 819 601 408 | 404,475635 | 26 / 0,117610168 |
| Head-stream / 2-B | 111 365 058 560 | 6 860 299 | 112 751 218 688 | 407,505043 | 26 / 0,116520917 |
| Head-stream / 3-A | 116 760 363 008 | 7 186 679 | 118 188 101 632 | 426,878711 | 0 / 0 |
| Tail5-stream / 0-A | 116 763 156 480 | 7 185 477 | 118 244 179 968 | 427,236549 | 0 / 0 |
| Tail5-stream / 1-B | 104 080 003 072 | 6 416 829 | 105 522 425 856 | 403,597463 | 430 / 0,317732207 |
| Tail5-stream / 2-B | 104 082 096 128 | 6 417 322 | 105 456 672 768 | 396,583597 | 430 / 0,313971754 |
| Tail5-stream / 3-A | 116 759 465 984 | 7 186 913 | 118 339 141 632 | 436,470603 | 0 / 0 |

La conversion CPU utilise, pour chacun des douze PID possédés, l'unique
timebase **mesurée et cohérente dans ses snapshots**, numer=125, denom=3 :
`(delta_user_ticks + delta_system_ticks) × numer / denom / 10^9`.
L'identité PID/start-abstime reste cohérente dans chaque rapport. Les secondes
CPU additionnent les threads ; elles ne sont ni une durée murale ni un temps
de calcul pur à soustraire pour obtenir l'I/O.

Les deltas de bloc ont été revérifiés entre leurs endpoints : même identifiant
contrôleur `4294969444`, compteurs croissants et égalité avec les deltas
sauvegardés. Il s'agit du système entier, incluant les autres processus,
**pas** de transferts physiques attribués au PID ou de cellules NAND.

Les médianes head-stream sont 116,760082 / 111,365925 Go de lectures OS A/B
(−4,619864 %), 118,243467 / 112,785410 Go de bloc globaux (−4,615948 %),
et 422,838154 / 405,990339 secondes CPU (−3,984459 %). Ces observations
concernent la requête entière. Chargement, readahead, attribution GPU et état
du cache empêchent d'en déduire une économie physique exacte par token ou
une causalité exhaustive.

Pour tail5-stream, les médianes sont 116,761311 / 104,081050 Go de lectures
OS A/B (−10,859986 %), 118,291661 / 105,489549 Go de bloc globaux
(−10,822497 %) et 431,853576 / 400,090530 secondes CPU (−7,355050 %).
Les mêmes restrictions de périmètre requête/préfill, d'attribution et de
cache s'appliquent. Les volumes ne deviennent pas des lectures NAND parce
qu'ils baissent dans le candidat.

Les 26 command buffers valides des deux B tête sont réellement observés dans
la fenêtre requête. Leur union vaut 0,116521–0,117610 s, médiane 0,117066 s.
Ces buffers incluent copies/événements ; `complete_capture_guaranteed=false`.
Ce ne sont pas tous les temps hôte, toutes les synchronisations ni un compteur
de calcul par kernel. Ils prouvent l'exécution GPU sans assimiler son taux
d'utilisation à un gain de génération.

Les deux B tail5 contiennent chacun **430 command buffers valides** dans la
requête, union 0,313972–0,317732 s, médiane 0,315852 s. La capture n'est pas
garantie complète et les copies/événements sont inclus. Ces durées n'isolent
pas les attentes des poids, la synchronisation hôte ou le calcul par kernel.

Les pics de footprint OS sont 611–613 Mo dans les passages CPU et environ
636,7 Mo pour les B tête. Ils ne remplacent pas les pics RSS, ne doivent pas
leur être ajoutés et ne mesurent pas tout le pinning GPU. L'intervalle RSS
observé dans les deux B tête est 5 984 468 992–6 093 242 368 octets, contre
6 630 735 872–6 701 039 616 pour leurs contrôles. Swap observé : zéro
croissance dans les douze passages. Pour tail5, les B ont un pic RSS de
6 145 818 624–6 205 030 400 octets, contre 6 702 055 424–6 705 774 592
pour leurs A. Leurs footprints sont au contraire un peu plus élevées,
639,8–640,3 Mo contre 612,1–612,5 Mo : ces deux comptabilités mémoire
restent distinctes.

## Provenance brute SHA-256

Tous les fichiers de passage sur disque correspondent exactement aux
`report_data` embarqués dans leur campagne. Les neuf hashes de bibliothèques
enregistrés sont identiques sur les douze passages et à leur plan. Le serveur
commun a SHA-256
`4deaf2ff1620610da1e086d9d7e04a17cae27ac0ce58a94c1f8efc317bc67dc1`.
Les chemins du tableau sont relatifs à `work/v3/`.

| Fichier | SHA-256 |
|---|---|
| `campaign-cpu-stream8/campaign.json` | `53a63a5f5ea2d6992cafdfcb51d7dc70b7858590926a17ed6fc343e1375df002` |
| `campaign-cpu-stream8/runs/00-A.json` | `aec3941ab3437cae603fa6bb46324a92b0f2d33973db96a673e1a88e8095df99` |
| `campaign-cpu-stream8/runs/01-B.json` | `af1318768644e86db2f70050aaa346f354a2b674d8a7242d47878f9de9a1077b` |
| `campaign-cpu-stream8/runs/02-B.json` | `0e38734ef9886f65ed804f37d9e82a3f036299a6608a34536a20baf363100f52` |
| `campaign-cpu-stream8/runs/03-A.json` | `88927efda24cf676248fdc5943c531be93c3602b047daf353940fa621ee805bc` |
| `campaign-head-stream8/campaign.json` | `a4131a636d793cac74c50bb7a03b2e9cdc9147586ea24678350d2d603a769c6b` |
| `campaign-head-stream8/runs/00-A.json` | `d76f0bb287a64dd2e996b119f0130d576288077ec7626d20dbefd24f7b665d19` |
| `campaign-head-stream8/runs/01-B.json` | `6e3b012fac8a9e01ceabac15b504bfb16756f548febbb7145de432f4ae7da9e6` |
| `campaign-head-stream8/runs/02-B.json` | `de118ff4683c55ece0cd2ea21b33eeea49f6cc3a567c10f145c469b619ff5e06` |
| `campaign-head-stream8/runs/03-A.json` | `9c88c0fff071b92ac4a0de3c5995ad0dde9c8c8b89b6c41bd317a0115724801a` |
| `campaign-tail5-stream8/campaign.json` | `c36c8dc921f3bb8522648af441f61d1bfa1753ba7729cf461a8967fd65abc6b8` |
| `campaign-tail5-stream8/runs/00-A.json` | `99829907498c4e5c958b8f7732ffa1e90003fd6e2ecdf4a8e6f4902f9d4ae8b0` |
| `campaign-tail5-stream8/runs/01-B.json` | `73f64167ac98cc71e5cdaac641fef9688ded580f7e79f569a2dfcd41e357494b` |
| `campaign-tail5-stream8/runs/02-B.json` | `66f44fe6005bde64315da14e8d4653222f5724bb6470b4eefbfc35f5ed12b845` |
| `campaign-tail5-stream8/runs/03-A.json` | `31f35a2bc284bebe92f24452d5e8329f96472fa09fb408cf5b5e37d601fd4758` |

Manifestes de fidélité cités et hashés par les plans :

| Capture | SHA-256 de `experiment.json` |
|---|---|
| `cpu-control16-isolated` | `e1d55b64d09897cff209a1d1157b83142457e3854e7494cdef78e3914efdc880` |
| `cpu-stream16-isolated` | `7d2723b55086fb2e168477f502a17a3516b46cefee77eb94afcf7e271eda48b6` |
| `head-stream16-isolated` | `e447d8c7a074cd37656c582bef9a9a2ef84cf09e8692c1cdff646d00f80472af` |
| `tail5-stream16-isolated` | `ac785ffa7cd0c2abcbaba61adae430e6a2e7ae46e2108e3d619ba434ca1d8a4b` |

Ce tableau ne qualifie pas automatiquement une optimisation et n'annonce pas
un maximum universel. Les échantillons restent courts et le cache de fichiers
ainsi que la charge de fond ne sont pas contrôlés. Les mesures longues et
les éventuelles nouvelles versions natives doivent être ajoutées séparément,
avec leurs propres hashes et portes de fidélité.

<!-- LONG_RESULTS_PENDING -->
