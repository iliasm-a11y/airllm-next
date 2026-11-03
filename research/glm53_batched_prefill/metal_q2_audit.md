# Composant Q2_K réellement multi-token : préparé, non compilé, non exécuté

Le nouveau shader `metal_q2_batch.metal` prépare un produit d'un expert entier
M2048/K4096 par N8/16/32 activations. Il n'est pas intégré au moteur GLM et
n'a pas été compilé ou exécuté. Ce statut vaut pour le shader, le probe et
le timer isolé préparés le 3 novembre 2026.

Le contrôle externe du 3 novembre à 13:53:23 UTC a refusé le lancement : le
Mac était sur batterie en décharge, avec pression normale de niveau 1. Le CLI
Q8 n'a pas démarré, aucun oracle natif n'a été appelé et le timer Q2 n'a pas
été compilé. Le reçu nettoyé est conservé dans
[`evidence/metal-components-admission-refusal.json`](evidence/metal-components-admission-refusal.json).
Aucune garde n'a été contournée.

## Sources officielles Apple et portée des API

Sources consultées le 3 novembre 2026 :

- [Tech Talk 111432 — Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/)
  décrit les Neural Accelerators des GPU M5, les API de tenseurs Metal et
  l'importance des formes de matrices. Ces capacités documentées motivent
  l'étude ; les gains d'autres applications n'établissent aucun gain GLM ici.
- [WWDC2026/330 — Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)
  décrit TensorOps/MPP et ses produits matriciels quantifiés. Apple distingue
  le support entier 4/8 bits ajouté à macOS/iOS 26 des formats supplémentaires,
  dont l'entier 2 bits, dans macOS/iOS 27. Le support d'un type 2 bits ne
  démontre pas une prise en charge native du schéma GGUF Q2_K et de ses minima.

Ces références documentent une API et ses versions ; elles ne vérifient pas
la disponibilité de chaque fonction dans le SDK local. Les shaders Q8 et Q2
préparés utilisent des opérations entières, réductions SIMD et FMA explicites.
Ils n'appellent ni TensorOps ni `matmul2d`. Aucun usage de TensorOps ou des
Neural Accelerators n'est mesuré par ces nouveaux composants. Toute adaptation
future devra encore vérifier formes, disposition des échelles et ordre
numérique contre les critères CPU/Metal, puis routes, logits et états GLM.

## Arithmétique reprise de la référence ARM

Le runtime épinglé à `3f8af15a76ac8675812e856da45454bafc90f12c` utilise
`ggml_vec_dot_q2_K_q8_K(..., nrc=1)` sur ARM, `quants.c:1688` et
`:1945-2011`. Pour chaque bloc de 256 valeurs dans l'ordre, il calcule la contribution
entière des minima, fait son ajout FMA, puis le dot entier pondéré et sa FMA.
Il ne faut pas réordonner les contributions FP32 entre blocs.

Le composant conserve ce contrat :

1. La routine CPU native `quantize_row_q8_K` prépare les activations. Sur ce
   source ARM, elle délègue à `quantize_row_q8_K_ref` (`:134-136`).
2. Le GPU consomme directement les blocs Q8_K de 292 octets : échelle FP32
   signée, 256 entiers et 16 sommes. Les buffers sont initialisés à zéro,
   car le fast path natif d'un bloc nul ne remplit pas ses `bsums`.
3. `minimum = somme high4(scale[k/16])*qs[k]` est l'équivalent entier exact
   des 16 `bsums` utilisés par le CPU. Le maximum absolu est 487680. Le dot
   pondéré est borné à 1463040 ; ces entiers sont exacts en int32 puis FP32.
4. Pour chaque bloc croissant : `md=(-dy)*dmin`, FMA minimum, puis `dd=dy*d`,
   FMA dot. `dy` reste FP32 signé. Les échelles FP32 d'activation et FP16 de
   poids non finies ou subnormales non nulles sont refusées, tant que ce
   domaine n'a pas sa propre vérification sur Metal.

Les deux FMA ordonnées suivent les composants mono-token v2 déjà testés.
Cette provenance ne transfère aucune preuve numérique au nouveau shader.
L'oracle du futur essai appelle directement le dot ARM natif pour chacune
des M*N sorties et compare tous leurs bits. Ce n'est pas une mesure CPU10.
Le reçu conserve séparément cette égalité bit à bit et la tolérance Metal
existante `atol=1e-3`, `rtol=1e-4`. Une différence de bits reste une différence
de ce contrat, même si elle satisfait cette tolérance. Aucun des deux résultats
ne qualifie les routes, logits ou états du modèle ; les critères du modèle
ne sont pas modifiés et l'objectif A reste ouvert.

## Réutilisation entre huit tokens

Un threadgroup contient 256 threads, huit SIMD de 32 lanes. Il traite une
ligne de poids et huit activations. Pour chaque bloc de 256 valeurs, ses 84 octets Q2_K
sont chargés une seule fois dans un tableau threadgroup puis lus par les
huit SIMD. Chacun effectue ses sommes entières indépendamment pour son
token. Deux barrières par bloc protègent les chargements et la réutilisation.
Aucune lane ne sort avant ces barrières.

Un seul dispatch calcule le lot entier ; il n'y a pas N lancements mono-token
successifs. Aucune matrice complète n'est déquantifiée en FP32. Le traitement
ne sélectionne pas d'experts et n'effectue pas leur combinaison pondérée :
l'intégration devra préserver le mapping des activations et l'ordre des
routes, qui restent hors de cette preuve de composant.

Le petit patch `metal_q2_timer.patch` exige `threadExecutionWidth==32` avant
le dispatch et consigne cette valeur. Le probe refuse un rapport ne portant
pas cette vérification. Le timer antérieur non patché convient toujours aux
essais Q8, mais ne fournit pas ce nouveau reçu Q2.

## Budget et garde d'exécution

Pour M2048/K4096/N32 : poids 2752512 octets, activations natives Q8_K 149504
octets et sorties 262144 octets. Les données FP32 d'entrée sont 524288 octets.
Le budget connu host/device conservateur vaut 7376896 octets, sous le plafond
inchangé de 64 Mio. Les buffers du timer occupent 3164172 octets. Ces chiffres
n'incluent pas les bibliothèques, Python et le pilote Metal ; leurs pics RSS
seront rapportés séparément. Le plafond du timer reste 64 Mio.

Le CLI prend le même verrou d'expérience que les runs GLM. Il doit être
lancé uniquement après l'arrêt et le nettoyage du modèle, sur secteur et
sous pression normale. Les gardes avant/après chaque cas restent RSS 8,5 Gio
et croissance globale de swap 32 Mio ; elles ne prouvent pas une attribution
exclusive de tout changement système. Chaque appel au timer est limité à 60 s.

Les sorties CPU/GPU sont synthétiques par défaut. Une option explicite
`--gguf` peut lire un expert original borné avec le lecteur antérieur, sans
conversion ni écriture dans le GGUF. Les fichiers de poids `packed.bin`
produits dans le répertoire de travail ne doivent alors pas être publiés.
Conserver seulement métriques, empreintes et sources dans la sélection publique.

## Construction différée du timer

La copie isolée du timer est prête dans `work/metal-q2-prepare` et le patch
y a été appliqué. Aucune compilation n'a été lancée. Le plan exact, avec
arguments absolus, empreintes et budgets, est `work/metal-q2-plan.json`.
Après le signal de fin du modèle, compiler cette copie comme le timer existant :

```sh
xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
  /absolute/work/metal-q2-prepare/q2_metal_timer.mm \
  -o /absolute/work/metal-q2-prepare/metal-q2-timer
cd /absolute/airllm-next
python -m research.glm53_batched_prefill.metal_q2_probe \
  --libraries /absolute/runtime-v4/bin \
  --timer /absolute/work/metal-q2-prepare/metal-q2-timer \
  --work /absolute/work/metal-q2-batch-run1 \
  --output /absolute/work/metal-q2-batch-run1.json \
  --repetitions 3
```

Employer l'interpréteur exact fourni dans le plan JSON. L'estimation préalable
est de 5 à 60 secondes pour le probe, non mesurée. Trois cas bornés appellent
chacun le timer, avec un warmup et trois commandes mesurées. Les travaux CPU
de référence ne sont pas inclus dans le timeout de chaque timer.

La vérification AST Python et l'application du patch à la copie isolée ont
réussi. Une seconde revue source n'a trouvé aucun défaut concret d'indexation,
de barrières ou d'ordre FMA sur le domaine admis ; cette revue n'est pas un
résultat de compilation ou d'exécution.

Le probe consigne séparément le temps de quantification CPU, celui de l'oracle,
les timestamps GPU et la durée de soumission/synchronisation. Les timestamps
GPU seuls ne comprennent pas la quantification des activations. Ni les poids préparés en mémoire ni
une égalité de composant ne qualifient un gain de préfill, un débit GLM, une
restauration d'état ou une réponse d'agent jusqu'à EOS.
