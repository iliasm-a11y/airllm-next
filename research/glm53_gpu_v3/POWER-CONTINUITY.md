# Audit de continuité physique des campagnes closes

**La campagne V4 E8 (`tail5 + lock256`) est INVALIDÉE POUR LES PERFORMANCES.** Deux de ses essais ont traversé des veilles répétées et des DarkWake. Le résultat brut −23,943670618940473 % ne démontre ni un gain ni une régression. Le statut original `complete` décrit une fin de processus ; il ne corrige pas cette rupture des conditions physiques. Les fichiers originaux restent inchangés.

Les douze essais V3 CPU-stream8, head-stream8 et tail5-stream8 ne présentent pas cette anomalie : leur écart entre temps UTC et compteur interne est compris entre 0,043405 et 0,055323 s, sans événement Sleep dans leurs intervalles selon le texte sauvegardé. Les gains V3 déjà limités à huit sorties conservent leur portée antérieure ; cet audit ne les transforme pas en preuve longue.

## Méthode et horloges

Pour chacun des 16 rapports clos, l’intervalle civil est calculé par `system_after.timestamp_utc − started_at`, puis comparé à `total_seconds`. Le code du runner utilise `time.perf_counter()` pour le compteur interne ; le rapport civil encadre aussi un court travail d’initialisation et de clôture. Le seuil de cohérence de 2 s est un critère de dépistage de cet audit a posteriori, pas un nouveau seuil de qualification du benchmark. Les événements du texte sauvegardé `power/sleep-wake-selected.txt` sont convertis depuis leur fuseau explicite `+0100` vers UTC avant croisement.

| Campagne | Essai | Début UTC | Fin UTC (`system_after`) | Intervalle civil (s) | Compteur interne (s) | Écart (s) | Sleep / DarkWake dans l’intervalle |
|---|---|---|---|---:|---:|---:|---:|
| cpu-stream8 | 0A | 2026-11-02T22:13:20.445481+00:00 | 2026-11-02T22:14:41.597567+00:00 | 81.152086 | 81.108681 | 0.043405 | 0 / 0 |
| cpu-stream8 | 1B | 2026-11-02T22:14:41.720818+00:00 | 2026-11-02T22:16:03.506240+00:00 | 81.785422 | 81.733154 | 0.052268 | 0 / 0 |
| cpu-stream8 | 2B | 2026-11-02T22:16:03.629110+00:00 | 2026-11-02T22:17:29.493884+00:00 | 85.864774 | 85.813803 | 0.050971 | 0 / 0 |
| cpu-stream8 | 3A | 2026-11-02T22:17:29.621865+00:00 | 2026-11-02T22:18:57.526616+00:00 | 87.904751 | 87.849558 | 0.055193 | 0 / 0 |
| head-stream8 | 0A | 2026-11-02T22:30:39.996423+00:00 | 2026-11-02T22:32:01.698474+00:00 | 81.702051 | 81.657511 | 0.044540 | 0 / 0 |
| head-stream8 | 1B | 2026-11-02T22:32:01.814840+00:00 | 2026-11-02T22:33:21.599215+00:00 | 79.784375 | 79.731716 | 0.052659 | 0 / 0 |
| head-stream8 | 2B | 2026-11-02T22:33:21.702402+00:00 | 2026-11-02T22:34:42.498610+00:00 | 80.796208 | 80.746728 | 0.049480 | 0 / 0 |
| head-stream8 | 3A | 2026-11-02T22:34:42.625380+00:00 | 2026-11-02T22:36:06.246388+00:00 | 83.621008 | 83.571928 | 0.049080 | 0 / 0 |
| tail5-stream8 | 0A | 2026-11-02T22:36:09.965549+00:00 | 2026-11-02T22:37:33.244090+00:00 | 83.278541 | 83.233010 | 0.045531 | 0 / 0 |
| tail5-stream8 | 1B | 2026-11-02T22:37:33.370800+00:00 | 2026-11-02T22:38:51.683697+00:00 | 78.312897 | 78.257574 | 0.055323 | 0 / 0 |
| tail5-stream8 | 2B | 2026-11-02T22:38:51.817122+00:00 | 2026-11-02T22:40:09.906779+00:00 | 78.089657 | 78.034648 | 0.055009 | 0 / 0 |
| tail5-stream8 | 3A | 2026-11-02T22:40:10.039871+00:00 | 2026-11-02T22:41:34.859355+00:00 | 84.819484 | 84.771621 | 0.047863 | 0 / 0 |
| v4-tail5-lock256-8 | 0A | 2026-11-02T23:09:40.262540+00:00 | 2026-11-02T23:11:37.515342+00:00 | 117.252802 | 117.201581 | 0.051221 | 0 / 0 |
| v4-tail5-lock256-8 | 1B | 2026-11-02T23:11:37.654228+00:00 | 2026-11-03T00:50:09.619404+00:00 | 5911.965176 | 163.383954 | 5748.581222 | 10 / 10 |
| v4-tail5-lock256-8 | 2B | 2026-11-03T00:50:10.224784+00:00 | 2026-11-03T01:17:30.509759+00:00 | 1640.284975 | 287.488937 | 1352.796038 | 4 / 4 |
| v4-tail5-lock256-8 | 3A | 2026-11-03T01:17:30.758928+00:00 | 2026-11-03T01:22:36.292033+00:00 | 305.533105 | 305.429096 | 0.104009 | 0 / 0 |

La première veille après la campagne tail5 V3 commence le **2 novembre à 23:42:35 heure locale** (22:42:35 UTC), après la fin de son dernier essai à 23:41:34,859355 locale. Un réveil complet est ensuite journalisé à 23:55:27 locale. Les deux contrôles V4 A16/E16 se terminent respectivement à 00:06:21,185091 et 00:09:16,623408 locale le 3 novembre. La veille suivante commence à **00:12:48 locale**, pendant E8 B1. Ces dates séparent les contrôles de fidélité des interruptions de performance ultérieures.

E8 B1 présente **5 748,581222 s** d’écart civil/compteur, et B2 **1 352,796038 s**. Le journal montre des entrées Sleep répétées, des réveils partiels DarkWake et l’utilisation de la batterie pendant ces essais. A0 et A3 ont des horloges proches ; A3 intervient cependant après un DarkWake, sans nouveau réveil complet observé dans le texte. Ils ne permettent pas de réparer rétrospectivement une paire ABBA interrompue. Le journal, à résolution de la seconde, ne permet pas d’attribuer exactement chaque seconde d’écart à un état de veille.

Les 16 fichiers bruts sont identiques aux rapports intégrés aux quatre fichiers `campaign.json`. Les champs historiques `comparison_valid_on_stated_scope` et tout résumé du runner restent archivés tels quels. Pour E8, le verdict indépendant de cet audit prévaut sur leur usage comme preuve de performance.

## Fidélité indépendante

Les contrôles V4 A16 et E16 sont clos avant la veille de 00:12:48 locale. Aucun événement Sleep/DarkWake/Wake du texte sélectionné ne tombe dans leurs intervalles. Leurs 2 478 080 logits finis et 3 792 écritures d’état observées restent une preuve numérique indépendante sur la trajectoire capturée. Ils ne sont pas utilisés pour estimer un débit. Voir `work/v3/fidelity-v4.md`.

## Portée et provenance

La continuité des horloges et l’absence d’événement sélectionné signifient « aucune rupture détectée dans ces preuves ». Elles ne prouvent ni l’exhaustivité du journal, ni une température ou une fréquence constantes, ni toutes les conditions d’alimentation. Une différence d’horloges peut aussi avoir d’autres causes ; dans E8 les événements de veille concordants apportent une preuve supplémentaire directe. Aucun nouvel essai, test, build, accès aux poids ou contrôle de puissance n’a été lancé pour cet audit.

Le texte sauvegardé (180489 octets) a pour SHA-256 `5823ed4800e80309c8a832292935ebe7aa4bd54b020436ee306acfe4c8a5a027`. Le JSON associé `work/v3/power-continuity-audit.json` contient les dates et empreintes de tous les fichiers sources, les 16 calculs, les numéros des lignes d’événements, les empreintes de runner enregistrées dans chaque rapport et les empreintes des contrôles de fidélité indépendants. Il conserve distinctement l’identité du code courant inspecté et les identités historiques.

| Campagne brute | SHA-256 |
|---|---|
| `work/v3/campaign-cpu-stream8/campaign.json` | `53a63a5f5ea2d6992cafdfcb51d7dc70b7858590926a17ed6fc343e1375df002` |
| `work/v3/campaign-head-stream8/campaign.json` | `a4131a636d793cac74c50bb7a03b2e9cdc9147586ea24678350d2d603a769c6b` |
| `work/v3/campaign-tail5-stream8/campaign.json` | `c36c8dc921f3bb8522648af441f61d1bfa1753ba7729cf461a8967fd65abc6b8` |
| `work/v3/campaign-v4-tail5-lock256-8/campaign.json` | `946e069aff60e9cb3be9ad5f4038e1dd8faa77e69192e1e98c244fbe62fe76f6` |

| Rapport brut | SHA-256 |
|---|---|
| `work/v3/campaign-cpu-stream8/runs/00-A.json` | `aec3941ab3437cae603fa6bb46324a92b0f2d33973db96a673e1a88e8095df99` |
| `work/v3/campaign-cpu-stream8/runs/01-B.json` | `af1318768644e86db2f70050aaa346f354a2b674d8a7242d47878f9de9a1077b` |
| `work/v3/campaign-cpu-stream8/runs/02-B.json` | `0e38734ef9886f65ed804f37d9e82a3f036299a6608a34536a20baf363100f52` |
| `work/v3/campaign-cpu-stream8/runs/03-A.json` | `88927efda24cf676248fdc5943c531be93c3602b047daf353940fa621ee805bc` |
| `work/v3/campaign-head-stream8/runs/00-A.json` | `d76f0bb287a64dd2e996b119f0130d576288077ec7626d20dbefd24f7b665d19` |
| `work/v3/campaign-head-stream8/runs/01-B.json` | `6e3b012fac8a9e01ceabac15b504bfb16756f548febbb7145de432f4ae7da9e6` |
| `work/v3/campaign-head-stream8/runs/02-B.json` | `de118ff4683c55ece0cd2ea21b33eeea49f6cc3a567c10f145c469b619ff5e06` |
| `work/v3/campaign-head-stream8/runs/03-A.json` | `9c88c0fff071b92ac4a0de3c5995ad0dde9c8c8b89b6c41bd317a0115724801a` |
| `work/v3/campaign-tail5-stream8/runs/00-A.json` | `99829907498c4e5c958b8f7732ffa1e90003fd6e2ecdf4a8e6f4902f9d4ae8b0` |
| `work/v3/campaign-tail5-stream8/runs/01-B.json` | `73f64167ac98cc71e5cdaac641fef9688ded580f7e79f569a2dfcd41e357494b` |
| `work/v3/campaign-tail5-stream8/runs/02-B.json` | `66f44fe6005bde64315da14e8d4653222f5724bb6470b4eefbfc35f5ed12b845` |
| `work/v3/campaign-tail5-stream8/runs/03-A.json` | `31f35a2bc284bebe92f24452d5e8329f96472fa09fb408cf5b5e37d601fd4758` |
| `work/v3/campaign-v4-tail5-lock256-8/runs/00-A.json` | `6bb927c7c95e6d086aecfbb0a2cec97131db0d842c26c4a54387cdffccf386ba` |
| `work/v3/campaign-v4-tail5-lock256-8/runs/01-B.json` | `6f3d00ea9410e9b3b820156f20a3a6fe771dc7acb1e7e5226d296c65073a9cdf` |
| `work/v3/campaign-v4-tail5-lock256-8/runs/02-B.json` | `b8902f1380f7dee5acc428bf387a3fca490b51c6401a5996e8618f2d77722df0` |
| `work/v3/campaign-v4-tail5-lock256-8/runs/03-A.json` | `4df82be999cf47a2163078f8e7204fc6fedc7b110ff54cef8c0c6b0362abd30b` |
