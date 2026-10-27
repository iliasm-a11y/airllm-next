# Audit des kernels M5, 27 octobre 2026

Les kernels existants `gguf_m5.py` restent inchangés. Après le nettoyage
explicitement autorisé et le retour à une pression mémoire normale, les **160
cas du banc ont réellement compilé et exécuté leurs kernels sur Apple M5 avec
MLX 0.32.4**. La compilation n'a nécessité aucune correction du code GPU.
La qualification numérique reste partielle : les variantes de réutilisation
reproduisent toutes exactement la référence GPU, tandis que TensorOps présente
des désaccords sur certains cas de forte annulation numérique. Les résultats
détaillés de reprise figurent en fin de document. Le chemin historique et le
repli du dispatcher restent actifs.

## Vérifications réalisées

L'interpréteur local est Python 3.12.13 avec MLX **0.32.4**. Les signatures
installées ont été lues sur disque sans importer MLX. `mx.fast.metal_kernel`
accepte `header` et `compile_options={'math_mode': 'safe'}` ; cette option est
également explicitement analysée par le [code officiel du tag
0.32.4](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/python/src/fast.cpp).
Les hachages des signatures installées et des en-têtes SDK sont conservés dans
`evidence/kernel-api-audit.json`.

Le SDK macOS 27.0 installé décrit les opérandes `tensor_inline` et la combinaison
float/float/float. Le descripteur actuel conserve `relaxed_precision=false` et
une destination FP32. La [documentation
Apple](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
décrit l'accumulation par tuiles et les tenseurs coopératifs. La présence des
types dans les en-têtes permet un contrôle d'API ; elle ne démontre pas la
compilation de cette spécialisation sur le GPU. Le test effectif du runner est
nécessaire.

Les outils de ligne de commande `xcrun --find metal` et `xcrun --find xctrace`
retournent 72. L'absence du compilateur hors ligne ne démontre pas l'absence de
la compilation JIT Metal de MLX. Cette dernière demeure à exécuter sous admission.

## Banc léger livré

`benchmarks/validate_m5_kernels.py` lance son propre processus isolé avec la
garde globale existante. La réserve système reste **3 Gio**, la croissance
maximale du swap **128 Mio**, et la pression doit être normale. Le payload
léger est **256 Mio** et le plafond RSS **512 Mio** ; ils sont distincts de
l'admission du modèle complet. Aucun chemin de test ne change la politique de
pression. Même le point d'entrée privé vérifie l'admission avant l'import MLX.

Le protocole complet couvre Q4_K et Q6_K, un ou deux experts, plusieurs tailles
de sortie et largeurs 256/512/1024, blocs de tokens 16/32/64/128 ainsi que des
dimensions partielles. Il compare TensorOps et les variantes de réutilisation
2/4/8/16 à un décodeur scalaire indépendant et à l'ancien kernel GPU. Aucun
poids de checkpoint n'est créé ni modifié. Les entrées et accumulations restent
FP32 ; les échelles demi-précision sont les représentations GGUF existantes.

Chaque cas conserve erreur numérique, succès/échec de compilation, coût de la
première exécution, répétitions AB/BA, temps CPU du processus et temps hôte
synchronisé. Un candidat plus lent ou rejeté reste dans les preuves avec
`selected=false`. Ce choix par tenseur n'active aucune optimisation du modèle
réel sans les contrôles supplémentaires de fidélité et de performance.
La comparaison sur les tenseurs applique `atol=1e-4, rtol=5e-5`, sans relâcher
les contrôles indépendants des logits réels. Un échec de cette tolérance
conserve le chemin de référence ; il ne déclenche aucune réduction de précision.

La limite d'allocateur MLX est une **indication**, comme l'indique son API, pas
une limite physique dure. Le banc la complète avec une charge explicite de
tous ses buffers, des relevés MLX actif/cache/pic et la supervision globale RSS,
réserve, pression et swap. Le cache de ce worker est fixé à zéro ; aucun cache
système n'est purgé.

Les durées GPU pures restent `null` : MLX 0.32.4 n'expose pas les timestamps
des command buffers dans son API Python inspectée. Le temps hôte inclut le
travail GPU et l'attente ; il ne lui est pas assimilé. `--capture chemin.gputrace`
permet de conserver une capture Metal si l'environnement la prend en charge,
mais une capture non analysée ne devient pas une mesure de durée GPU.
Apple décrit les [timestamps GPU et leur conversion en temps
CPU](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).

## Exécution autonome

Depuis la racine du dépôt, après libération volontaire des ressources :

```sh
MLX_ENABLE_TF32=0 /chemin/python-avec-mlx-0.32.4 \
  benchmarks/validate_m5_kernels.py --output /chemin/nouveau/kernels.json
```

`--quick` réduit les formes et les variantes pour un premier contrôle. Les
sorties existantes sont protégées contre l'écrasement. Le code de sortie 75 et
`status=deferred` signifient un refus de ressources, avec aucun worker lancé.
`status=completed` signifie que le protocole est terminé ; lire chaque cas pour
distinguer candidat compilé, rejet numérique, indisponibilité et gain mesuré.

Pendant la phase initiale, le premier contrôle réel a été refusé à pression **2**,
avec **3 308 126 208 octets disponibles** et **3 373 465 600 octets de swap
utilisé**. Les fichiers `kernel-admission.json` et `kernel-admission.guard.json`
conservent ce refus avant lancement. Le contrôle portable ciblé a terminé avec
**74 tests réussis et 64 cas GPU ignorés**. Ces tests portent sur les contrôles
du banc et du dispatcher ; ils ne valident aucun résultat GPU ou checkpoint.

## Compilation et mesures après récupération de ressources

Les preuves sont dans `evidence/recovery-20261027/`. Le premier lancement
`kernels-real-001` a compilé et exécuté 25 cas avant qu'un désaccord de la
référence GPU existante avec le produit matriciel NumPy FP32 ne provoque un
arrêt conservateur. Cette preuve négative est conservée. Le banc a ensuite été
corrigé pour enregistrer ce désaccord, conserver les tenseurs bruts et poursuivre
les autres cas, sans changer les tolérances ni les protections.

`kernels-real-002.json` contient **160 cas compilés et exécutés**, soit les deux
types Q4_K/Q6_K, les groupes d'experts simples et doubles et les huit formes
prévues, avec les cinq variantes. La garde complète a duré **5,144 secondes**,
avec un pic RSS échantillonné de **67 256 320 octets**, une pression de **1** du
début à la fin, une réserve minimale disponible de **5 496 274 944 octets** et
**aucune croissance du swap observée**. Ce temps comprend la totalité du banc,
notamment les comparaisons, les compilations et l'écriture des preuves ; ce
n'est ni une durée GPU pure ni une vitesse du modèle.

Les **128 sorties de réutilisation sont identiques bit pour bit à l'ancien
kernel GPU**. Sur les 32 cas TensorOps, la comparaison au produit NumPy FP32
est identique bit pour bit dans 28 cas ; les désaccords avec la référence GPU
empêchent une qualification globale. Parmi les 80 cas satisfaisant tous les
contrôles initiaux et donc chronométrés, 20 variantes de réutilisation sur 64 et
un cas TensorOps sur 16 franchissent le seuil local de 3 % sur le temps hôte.
Ces micro-mesures ne démontrent aucun gain sur le checkpoint réel et ne
justifient aucune activation générale.

Le diagnostic reproductible `benchmarks/diagnose_m5_numerics.py` a relu tous
les tenseurs bruts, vérifié leurs SHA-256 et compilé le décodeur C GGML
indépendant conservé dans le dépôt. Sur les **32 jeux d'opérandes**, les poids
déquantifiés par l'oracle scalaire et par le décodeur NumPy sont **identiques au
décodeur C**. Le défaut observé ne provient donc pas d'un agencement incorrect
des blocs de quantification dans ces fixtures.

En revanche, les ordres de réduction FP32 divergent sur des sommes fortement
annulées : 16 jeux sur 32 échouent à la comparaison référence GPU/NumPy FP32,
et huit jeux sur 32 échouent aussi à la comparaison référence GPU/accumulation
FP64, avec les mêmes tolérances. Par exemple Q4_K, 64 lignes, largeur 512,
16 tokens, élément `[12,31]` : GPU historique **1,2983703613**, NumPy FP32
**1,2981696129**, somme FP64 confirmée par `math.fsum` **1,2983043568**.
L'écart entre les deux résultats FP32 est **0,0002007484**, pour une tolérance
locale de **0,0001649085**. La somme des valeurs absolues des produits vaut
environ **10 099**, soit un rapport d'annulation d'environ **7 779**. Aucun seuil
n'a été relâché ; le détail des erreurs et des éléments concernés est conservé
dans `kernels-real-002-numeric-diagnosis.json`.

Cette investigation établit la compilation matérielle et caractérise les
écarts numériques miniatures. La fidélité des logits/tokens et les performances
du checkpoint complet nécessitent leur propre campagne.
