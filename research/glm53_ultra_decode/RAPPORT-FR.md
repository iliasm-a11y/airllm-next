# Décodage GLM-5.3 Flash sur M5 — bilan expérimental du 3 novembre 2026

**Aucune accélération complète nouvelle n'est qualifiée.** Du code natif et
Metal a été développé, compilé et exécuté. Les composants sont fidèles sur les
cas mesurés, mais leur coût total dépasse celui de CPU10 dans ce régime chaud.
La campagne de modèle contemporain reste bloquée avant lancement par ses
réserves. CPU10/GPU0 demeure la configuration conservée pour RONIN.

## État vérifié et isolement

Les PR #17 à #23 ont été consultées sur GitHub. Elles sont ouvertes, en brouillon
et non fusionnées à l'inspection. PR #23 est au commit
`6bdeec357154a7740833b4ad694d08e02aef2f96`. Sa capture CPU16 est rejetée ;
la première différence capturée est au token 512, KDA couche 0. Sa restauration
CPU8 a été arrêtée sur 39,125 Mio de croissance du swap global. Ses composants
Metal et ses rejeux ciblés ne qualifient ni le décodage complet ni deux réponses
utiles d'agents. Les résultats négatifs sont conservés.

La branche `research/glm53-ultra-decode-engine` part de cette révision dans un
arbre de travail séparé. Seul `research/glm53_ultra_decode` et un workflow
indépendant changent. `main`, PR #23, RONIN et le runtime installé restent
inchangés. L'effort Max des agents n'est pas réduit.

## Profil quantitatif hérité, recalculé

Le résumé complet compressé de PR #19 a été décompressé et son SHA256 vérifié.
Le préfill est exclu ;15 intervalles de décodage subsistent :

| Famille | Somme des temps d'opération sérialisés | Fraction |
|---|---:|---:|
| MUL_MAT dense |64,356480 s|70,63 %|
| MUL_MAT_ID MoE |18,452227 s|20,25 %|
| Autres opérations |8,305882 s|9,12 %|
| Total |91,114589 s|100 %|

Les CPU core-seconds totalisent 677,177714 s. Ils cumulent plusieurs threads et
ne permettent pas de calculer un temps d'attente I/O par soustraction.
Le processus comptabilise183 799 644 160 octets lus sur ces15 intervalles,
soit12,253 GB/intervalle. C'est un compteur OS historique, pas une mesure NAND.
Les données détaillées par couche/opération sont dans
`evidence/historical-analysis.json`. Les couches sans suffixe restent
explicitement non attribuées.

Une **autre** capture, PR #21, donne12 218 990 592 octets d'opérandes originaux
distincts/token :9 444 458 496 denses (77,293 %) et2 774 532 096 MoE (22,707 %).
Ces volumes ne sont pas des lectures physiques. Les versions et captures
diffèrent : les deux tableaux ne constituent pas une comparaison A/B.

Le goulot le mieux étayé est la résidence et la circulation des poids,
principalement denses. Le partage exact stockage/mémoire/calcul n'est toujours
pas isolé. L'auxiliaire nouveau observe le chargement, l'initialisation,
le préfill, chaque appel de décodage, les CPU core-seconds, lectures/écritures
OS, défauts de pages, pageins et résidences `mincore`. Il laisse à `null` les
lectures NAND, allocations et temps d'attente purs non mesurés. Aucun de ces
nouveaux profils complets n'a été exécuté après le refus d'admission.

## Nouveaux kernels et mesures réelles de composants

Les formats des poids restent Q8_0 et Q2_K. Le Q8 conserve les FMA et l'arbre
pair/impair du dot ARM nrc1. Le Q2 effectue les réductions entières exactes puis
les FMA dans l'ordre de la référence native. Les activations sont quantifiées
par la bibliothèque CPU native ; aucun déquantificateur FP16 intermédiaire
des poids n'est ajouté. Des tuiles 1/2/4 réutilisent les activations et leurs
métadonnées ; deux banques peuvent partager un lancement sans concaténation
des poids. Les buffers et pipelines sont persistants dans le banc.

Les poids/activations sont synthétiques, dans les formats originaux, avec
deux graines déterministes. Les formes sont celles déjà étudiées : projection
Q8 M1536/K4096 et paire de projections Q2 M2048/K4096. Le CPU utilise le vrai
graphe GGML et un pool 10 partagé ; l'oracle dot ARM indépendant vérifie aussi
un sous-ensemble. Chaque variante comporte3 chauffes et21 répétitions, ordre
randomisé. CPU et GPU sont mesurés dans des phases successives ; il ne s'agit
pas d'un A/B/B/A complet du modèle.

| Composant, un token | CPU10 médian | Meilleur Metal total médian | Temps GPU seul correspondant |
|---|---:|---:|---:|
| Q8 M1536/K4096, graine 1 |0,032833 ms|0,533584 ms (tuile 1)|0,085458 ms|
| Q8 M1536/K4096, graine 2 |0,032291 ms|0,610417 ms (tuile 2)|0,267417 ms|
| Paire Q2 M2048/K4096, graine 1 |0,089792 ms|0,459625 ms (tuile 4)|0,237625 ms|
| Paire Q2 M2048/K4096, graine 2 |0,091417 ms|0,325625 ms (tuile 4)|0,137792 ms|

Le temps Metal total inclut la quantification native CPU, l'encodage,
la soumission et l'attente. Les imports initiaux, compilations et allocations
de préparation sont exclus des deux régimes chauds. CPU10 inclut sa propre
quantification dans le graphe ; sa copie de sortie hors graphe est exclue.
Les poids tiennent dans un petit ensemble chaud : ces nombres ne peuvent pas
être extrapolés à un checkpoint de110 GB.

**50 176 composantes finales de sortie sont comparées bit à bit**, sans NaN,
sur les quatre essais retenus et leurs variantes. Les activations quantifiées
et les contrôles CPU1/CPU10 passent. Cela ne compare ni logits GLM, ni routes,
ni états récurrents du modèle. Les sorties finales des variantes sont
contrôlées ; ce nombre n'est pas un nombre de tokens ni de trajectoires.

Le regroupement Q2/tuile 4 économise0,240416 à 0,309542 ms face à deux
commandes GPU synchronisées séparément sur la même paire synthétique.
Il reste3,56 à 5,12 fois plus lent que CPU10. Aucun temps de génération GLM
économisé n'est démontré. Pour Q8, aucune tuile nouvelle n'améliore de manière
reproductible le coût total de la référence tuile 1.

Un premier banc à deux pools CPU10 distincts produisait une forte contention.
Ses timings sont **rejetés**, conservés séparément, et exclus du tableau.
Un premier shader a aussi échoué à la compilation sur une conversion
`as_type<half>` ; le refus et sa correction sont conservés. Les essais refusés
sur pression mémoire ne sont pas transformés en débits.

## Admission, mémoire et provenance

Le reçu SHA256 récent a été réutilisé uniquement après concordance des cinq
éléments d'identité de chaque shard. Les deux shards et l'architecture sont
vérifiés par ce reçu inchangé ; aucun nouveau hash complet de110 GB n'est
revendiqué. L'ancien reçu embarqué avait une identité périmée et a correctement
été refusé lors de la première tentative.

L'admission CPU32 corrigée observe4 155 604 992 octets de mémoire disponible
(3,870 Gio) et17 140 989 952 octets libres sur le volume de sortie (15,964 Gio).
Elle exige5,5 Gio et32 Gio. **Aucun processus natif de modèle n'est lancé.**
Cette enveloppe conservatrice prolonge la politique de la campagne locale
précédente ; elle n'est pas un pic mémoire mesuré. La pression oscille aussi
entre1 et2 et plusieurs composants sont refusés avant lancement.

Les composants retenus ont une croissance de swap observée nulle. Leurs pics
RSS sont échantillonnés : un essai très bref ne fournit pas nécessairement un
échantillon au pic. Ces essais ne certifient pas la stabilité d'une génération
longue. Les gardes RSS8,5 Gio, swap global32 Mio, pression/réserves, secteur,
capot, horloges et propriété des processus sont conservés. Aucun processus
personnel n'est fermé, aucun fichier utilisateur n'est supprimé. Fanxiang
n'est pas monté ; aucune mesure ne lui est attribuée.

Le build3 compile réellement les trois binaires contre le runtime V4 existant :

| Binaire | SHA256 |
|---|---|
| composant CPU |`b83907a7ca20c4ad46d9b8654c2d7490b2a0f4f5f066514fe1501ad9e0330ae3`|
| timer Metal |`e0b502a8b8e2795561ce1538d0ca505b1cba697a04b31c962e928b5085580c27`|
| profil natif |`66a9b1c9d1138dd1deea39df334bb46292e2cad38cb249384889ec16eeb5d299`|

Les sources, headers, includes, compilateur et bibliothèques liés figurent dans
le manifeste de build. Le nouveau profil n'a qu'un smoke de paramètres ;
aucune qualification numérique de modèle ne lui est transférée. Après les
mesures retenues, l'ajout de contrôles de provenance Python plus stricts n'a
pas modifié les sources C++/Metal ni leurs binaires. Les reprises correspondantes
sont refusées par la pression mémoire ; elles ne créent pas rétroactivement
une observation de provenance supplémentaire pour les essais précédents.

## Poids, ordonnanceur, TensorOps et spéculation

Le cache dense/streaming et les caches MoE des PR #19–21 existent déjà.
Leurs gains complets ne sont pas établis ; le LRU128 avait0 / 1 008 hits en
décodage. Les nouveaux composants ne changent ni leur budget ni les experts.
La priorité reste une mesure contemporaine des poids denses. Aucun nouveau
cache de poids permanent n'est ajouté.

La politique CPU/GPU expérimentale est désactivée, sans branchement à RONIN.
Elle demande résidence prouvée, identité du runtime, seuils existants,
logits/routes/états conformes sur au moins 32 étapes, gardes et deux cycles
A/B/B/A dont les coûts complets passent. Aucune preuve actuelle ne l'admet.

Metal4/TensorOps expose des opérations matricielles adaptées au M5 selon
[Apple](https://developer.apple.com/videos/play/tech-talks/111432/).
Cela ne prouve pas un avantage sur les GEMV quantifiés N=1 de cette mission.
Leur conversion, remplissage et ordre de réduction exigeraient une preuve
indépendante. Aucun nouveau bénéfice Neural Accelerator n'est revendiqué.

La spéculation exacte reste conditionnelle. Son gain exige que le coût de
proposition, sauvegarde/restauration et vérification divisé par les tokens
réellement acceptés soit inférieur au décodage séquentiel. Vérifier
séquentiellement tous les candidats n'élimine aucun passage de poids ;
une vérification groupée fidèle est donc le point critique. Le changement
de microbatch CPU16 déjà rejeté et les états KDA empêchent de présumer cette
fidélité. Aucune nouvelle spéculation GLM n'est intégrée ou chronométrée.

## Bornes conditionnelles

Avec U=12,219 GB d'opérandes distincts, une couverture utile C et une interface
servant B octets/s, le temps de service nécessaire est au moins
`max(U−C,0)/B` **si ces octets doivent effectivement traverser cette interface**.
Une superposition parfaite donne `max(t_IO,t_calcul)`, aucune superposition
donne `t_IO+t_calcul`. U ne mesure pas les lectures NAND ; B et C ne sont pas
mesurés ici. Le JSON expose des scénarios B=1/2/3 GB/s, C=0/4/8 GB et calcul 0/1 s.

| Cible | Couverture utile minimale à 1 GB/s | À 3 GB/s |
|---|---:|---:|
|0,5 token/s|10,219 GB|6,219 GB|
|1 token/s|11,219 GB|9,219 GB|

Ces conditions sont nécessaires dans ce scénario, avant les coûts de calcul,
de synchronisation et les réserves. Elles ne démontrent ni faisabilité
universelle ni impossibilité physique. Une fraction de temps d'un ancien
profil ne permet pas de promettre un débit pour une autre version.

## Validation

- 33 nouveaux tests portables passent.
- La sélection de contrôles existants et nouveaux donne229 tests réussis,
  40 sous-tests réussis et1 fixture native ignorée faute de chemin configuré.
  Cette fixture est ensuite réellement compilée et exécutée avec les sources
  et bibliothèques épinglées :1 test réussi. Total distinct :230 tests réussis.
- Compilation native réelle, smoke paramètres CPU10/batch8/slot1, tests Metal
  matériels et vérification de résidences sans lire les pages protégées.
- Ces résultats locaux ne préjugent pas de la CI ni d'une stabilité prolongée.

## Verdict obligatoire

1. **Goulot principal :** poids denses/résidence/circulation ;70,6 % du temps
   des opérations denses du diagnostic historique de décodage. L'attente I/O
   pure reste inconnue.
2. **Débit maximal reproduit avec fidélité :** aucun nouveau débit complet.
   L'historique PR #21 reste0,151471 token/s médian sur huit sorties avec fidélité
   capturée distincte ;0,129552 token/s concerne un EOS CPU de88 IDs. Ces cas ne
   sont pas directement comparables et ne sont pas remesurés ici.
3. **GPU M5 et décodage complet :** aucun nouveau gain complet démontré.
   Le bénéfice court historique tail5 de9,14 % ne devient pas une qualification
   soutenue32/64/128/EOS.
4. **Avantage net Q2_K/Q8_0 :** les nouveaux kernels sont fidèles sur les
   composants ; CPU10 reste plus rapide dans le régime résident testé.
5. **Temps économisé sur les poids :** aucune économie GLM démontrée.
   Le regroupement GPU économise0,240–0,310 ms par paire synthétique face à
   deux commandes GPU ; il ne démontre pas de réduction des lectures SSD.
6. **Spéculation exacte :** aucun bénéfice mesuré ; vérification groupée et
   rollback KDA fidèles non qualifiés.
7. **Stabilité longue :** non qualifiée ; la campagne complète est refusée.
   Les petits essais retenus n'ont pas de croissance de swap observée.
8. **Meilleure configuration RONIN :** conserver CPU10/GPU0, Max,600 identités
   logiques et un slot physique. Aucun nouveau backend n'est activé.
9. **0,5 token/s :** non atteint.
10. **1 token/s :** non atteint ; plausibilité seulement conditionnelle aux
    transferts/résidence/calcul, pas justifiée par les mesures actuelles.
11. **Obstacles pour600 agents rapides :** débit physique faible, longs prompts
    Max, attente du slot, mémoire/swap, absence de deux décisions utiles EOS
    qualifiées et absence de charge prolongée.600 identités ne signifie pas
    600 raisonnements simultanés.
12. **Pilote contrôlé :** le nouveau moteur accéléré n'est pas prêt.
    Le code constitue un banc et des candidats expérimentaux ; sa promotion
    attend un CPU contemporain, la fidélité complète et des générations
    longues admissibles. La configuration CPU existante reste disponible.
