# Augmenter le batch CPU sans changer le scan KDA

Le runtime épinglé fournit une piste CPU indépendante de Metal. Aucun gain
n'est encore mesuré pour les modifications proposées ici. Aucun runtime
actif n'est modifié, et aucune nouvelle inférence n'est lancée par cet audit.

## Le scan KDA CPU n'a pas besoin d'un nouveau sous-chunk8

Dans les sources mesurées, `src/llama-context.cpp:379-381` initialise
`fused_gdn_ar=true`, `fused_gdn_ch=true`, `auto_fgdn=false`. Sur ce chemin,
`src/models/delta-net-base.cpp:439` sélectionne le même opérateur
`GATED_DELTA_NET` pour8,16 ou32 tokens. Le terme "chunked" de ce flag ne
signifie pas qu'une formule matricielle parallèle remplace la récurrence CPU.

`ggml/src/ggml-cpu/ops.cpp:11211-11263` parcourt les tokens dans l'ordre :
exp des gates, multiplication de l'état, dots avec K, delta, mises à jour
MAD de l'état, dots avec Q, sortie. Chaque tête conserve cet ordre ; la
distribution de travail `:11269-11307` porte sur têtes et séquences. Les
opérations stockent les valeurs dans les buffers FP32. À entrées Q/K/V/g/beta
et état initial identiques, grouper16/32 tokens conserve donc le même ordre
arithmétique de cette récurrence. C'est une déduction du source, pas une
preuve de fidélité du modèle complet.

Les copies des fenêtres de convolution R sont construites dans
`src/models/glm5-next.cpp:224-233`. La convolution CPU
`ggml/src/ggml-cpu/ops.cpp:9952-9970` conserve le même dot sur la fenêtre
`d_conv` pour chaque token. Le tenseur R doit toujours correspondre aux
derniers tokens et le tenseur S à la dernière étape traitée. Un sous-chunk8
interne supplémentaire ajouterait des copies/graphes sans résoudre les
écarts provenant de projections ou d'attention en amont.

Le fallback non fused reste différent :
`src/models/delta-net-base.cpp:61` fixeCS16 pour KDA, puis emploie cumsum,
exponentielles, produits et résolution triangulaire. Changer vers ce fallback
ou vers une récurrence token par token distincte ne préserverait pas
automatiquement l'arithmétique de la référence. Les slots de rollback
supplémentaires n'ont pas non plus le même contrat qu'un unique état final.

`cpu_batch_kda_fixture.cpp` prépare une comparaison native de composant
8/16/32 avec entrées synthétiques identiques,128 valeurs par tête et2 têtes,
sorties complètes et état S. Elle ne charge aucun modèle, ne lit pas de poids,
et ne mesure aucun gain. Sa compilation/exécution est laissée hors campagne
chronométrée ; aucun résultat de cette fixture n'est revendiqué ici.

## Les véritables obstacles à la fidélité

Les projections denses Q8 peuvent changer de noyau ou de réduction selon la
forme : LLAMAFILE, vec_dot nrc1/nrc2 et partitions de travail sont sélectionnés
dans `ggml/src/ggml-cpu/ggml-cpu.c:1349-1530`. Les fixtures Metal présentes
dans ce dossier ont déjà exposé un écart entre nrc2 et le dispatcher CPU10
aux tailles64/128. Un scan KDA inchangé ne corrige pas une projection déjà
divergente. Les sorties de routeur, routes ordonnées, mHC, normalisations,
indexeur DSA/k-pool, caches d'attention et écritures doivent également passer
leur propre comparaison sur prompts complets.

Le code `ggml/src/ggml-cpu/tiled/tiled.cpp` possède bien une voie tuilée
K-quants, mais les lignes597-604 retournent faux sur ARM : cette voie est
conditionnée par AVX/AVX2/AVX512. Elle ne peut pas servir de preuve que le M5
exécute déjà ce tuilage. Le fallback ARM conserve toutefois ses petits blocs
de16 et groupe les activations par expert avant calcul.

## Petite proposition Q2_K MoE, désactivée

`cpu_batch_reuse.patch` ajoute une option
`AIRLLM_CPU_Q2_BATCH_REUSE=1`, uniquement Apple ARM, Q2_K et buffer de poids.
Elle ne change pas les poids, le tokenizer, les routes, la quantification
d'activation, les dots, les additions finales d'experts ni le graphe KDA.

La liste des activations par expert est déjà calculée par le moteur.
Aujourd'hui, les travaux sont découpés en blocs de lignes et de tokens
(`ggml-cpu.c:1898-1907`). Si un même expert reçoit plus de16 activations,
la proposition attribue à un travail un bloc de16 lignes couvrant toutes
ces activations. Le même worker réemploie alors logiquement les mêmes plages
de poids plutôt que de répartir les morceaux de tokens entre plusieurs
travaux. **Chaque sortie appelle toujours le dot ARM Q2_K natif avec nrc1**
(`:1724-1733`). Aucun tampon de poids ou de résultat supplémentaire n'est
créé. La résidence effective dans les caches CPU reste à mesurer.

L'option ne modifie rien lorsque l'expert reçoit16 activations ou moins.
Elle ne peut donc pas, à elle seule, accélérer les lots8/16. Le bénéfice est
seulement plausible sur des lots32+ dont les routes sélectionnent souvent
le même expert. Il faut d'abord compter ces occurrences sur des routes
réelles. Si elles n'existent pas ou sont rares, ne pas lancer de campagne
complète pour cette option. L'admission exige au moins autant de blocs de16 lignes que de threads,
mais la distribution et le coût de synchronisation restent à mesurer.
Aucune sélection automatique n'est ajoutée.

Le patch s'applique au source épinglé déjà patché V3/V4 selon
`git apply --check`. Il n'a pas été appliqué, compilé ou chronométré. Il est
un candidat expérimental, pas une optimisation retenue.

## Étape utile suivante

Comparer d'abord le préfill CPU16 puisCPU32 au témoin CPU8, avec le scan fused
déjà présent, sans ce patch. Utiliser les mêmes entrées natives, logits,
routeurs, routes ordonnées, états complets R/S/KV/indexeur, écritures et
paramètres mémoire. En cas d'écart, capturer les premières projections et les
normalisations avant KDA pour localiser la première opération divergente.
Un échec sur les états ne doit pas être masqué par un texte ou un token égal.

Une fois un batch plus grand fidèle établi, mesurer les comptes par expert.
Tester l'option Q2 seulement si ces comptes justifient son admission. Les
poids logiquement consultés, lectures OS, défauts de page et lectures SSD
physiques doivent rester distincts. Aucun facteur d'accélération ni taille
optimale n'est déduit de cet audit.
