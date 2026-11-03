# noBLAS v3, seconde requête : divergence confirmée à la position 512

La requête 1 complète, de 542 IDs, échoue au critère CPU bit à bit. Les 118 séries
capturées ont été comparées sur **tout** le prompt, après vérification des SHA
bruts : aucun écart capturé avant la position 512, indexée à partir de zéro.
Le constat ne repose donc pas seulement sur l’égalité du checkpoint à 512.
Les lectures ont porté sur 653 098 240 octets de tenseurs du témoin et
655 450 240 octets du candidat, ainsi que sur leurs sorties brutes complètes.
Aucun modèle, build ou GPU n’a été relancé par cet audit.

| Premier écart observé | Résultat |
|---|---|
| Sortie KDA, token 512, couche 0 | `kda_scan_out-0` : 6 909 / 8 192 valeurs différentes à ce token ; écart absolu maximal 6,9849193 × 10⁻¹⁰ |
| Premier élément de cette sortie | `0x35f614f0` → `0x35f614e8`, soit 1,8334521883 × 10⁻⁶ → 1,8334512788 × 10⁻⁶ |
| Routeur MoE, token 512, couche 3 | Premiers logits de routeur capturés divergents ; tous restent finis |
| Première route ordonnée, token 512, couche 8 | Permutation des rangs 3/4 : `[112,116,199,66,273,192,63,26]` → `[112,116,66,199,273,192,63,26]` ; même ensemble d’experts |
| Premier ensemble d’experts différent, token 512, couche 27 | `[139,51,161,7,224,75,180,197]` → `[139,51,161,7,224,75,180,222]` |

Les checkpoints complets à 128, 208, 256, 384 et 512 correspondent au CPU 8.
Celui à 542 diffère, comme le fichier final de 163 331 412 octets et l’état
post-continuation de 163 390 620 octets. Les 154 880 logits de préfill diffèrent
tous, avec un écart maximal de 1,3769762516. Les quatre vecteurs de continuation
diagnostique, dont le premier est celui du préfill, diffèrent aussi, malgré les quatre mêmes IDs diagnostiques
`[785, 1196, 374, 10156]`. Ces IDs ne remplacent pas la fidélité des vecteurs.

Le segment final réel est `[512,542)`, soit un seul lot de 30. Le témoin traite
8 + 8 + 8 + 6. Le journal d’opérations atteste notamment, à cette frontière,
`blk.0.attn_q.weight` Q8_0 avec M = 8 192, K = 4 096, N = 30. Dans la source V4
exacte reconstruite `ggml/src/ggml-cpu/ggml-cpu.c:1478–1503`, le chunk de 16 donne
`nchunk1 = ceil(30/16) = 2`, puis `dr1 = ceil(30/2) = 15`. Pour cette grande
projection, `nchunk0*nchunk1 = 1 024`, au-dessus de `threads*4 = 40` : la petite
partition par threads n’est pas sélectionnée. Les deux tranches de colonnes
ont donc une longueur impaire. La condition aux lignes 1518–1526 rabat les
kernels à deux lignes éligibles sur `num_rows_per_vec_dot = 1`. Le trait Q8_0
à deux lignes est conditionné à I8MM aux lignes 286–294.

Le mécanisme est fortement étayé par les sources exactes, les formes réellement
capturées et la frontière de divergence. La variable nrc n’a cependant pas été
tracée dynamiquement dans cet essai ; les projections antérieures à KDA 0 ne
sont pas capturées en valeurs. La première **sortie capturée** divergente n’est
pas une preuve de la première instruction arithmétique divergente. Aucun essai
correctif ne fait partie de ce diagnostic.

Le bras reste incomplet : deux requêtes sont closes dans le rapport et une
première microbatch de 32 tokens de la troisième avait déjà été exécutée avant
l’arrêt contrôlé. Le superviseur a reçu SIGTERM à 18:05:34 UTC, puis a demandé
l’arrêt de son wrapper ; le natif a fini à −15, le wrapper à 2. Ce sont des
sorties d’arrêt volontaire après rejet numérique, pas une fin normale qualifiée.

La vérification OS bornée retrouve absents les PID du superviseur 69 187,
du wrapper 69 188, du natif 69 207 et de caffeinate 69 208. Le reçu confirme
le FD UID parent fermé ; `lsof` limité au fichier connu ne retrouve aucun
détenteur. Le helper d’éveil est sorti naturellement à 0. Les reçus de secteur
et d’intégrité sont vrais, la croissance de swap échantillonnée est nulle et
le RSS échantillonné maximal est de 5 868 306 432 octets. Le statut global reste
`cancelled` avec le motif contrôlé `CANCELLED/SIGTERM` ; aucune qualification
complète, cache, EOS utile ou performance n’est ajoutée.

Les preuves durables sont `native-noblas-full32-first-divergence.json` et
`native-noblas32-cleanup-audit.json`. Le diagnostic principal et son supplément
de routes sont reproductibles par les deux scripts `audit_noblas32_first_divergence.py`
et `supplement_noblas32_route_audit.py`, sans appel moteur.
