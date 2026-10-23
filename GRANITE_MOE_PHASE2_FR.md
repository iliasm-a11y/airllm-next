# Phase 2 — véritable checkpoint MoE entraîné Granite

**Résultat : checkpoint MoE entraîné complet validé au protocole strict sur CPU,
puis sur Metal après correction d'une course de placement CPU/GPU dans les
lecteurs asynchrones. Les variantes Metal corrigées sont bit à bit identiques.
L'échec initial et les preuves avant/après sont conservés.**

Le moteur est explicitement expérimental et accessible en Python via
`AirLLMNextGraniteMoeModel`. Il n'active aucun dispatch de production automatique.
Il partage les lecteurs bornés, caches, prélecture, sessions et génération Next ;
les équations Granite et les alias des banques d'experts restent séparés des
architectures GLM, Llama et Qwen.

## Identité et protocole

- Modèle officiel : [IBM Granite 3.0 1B-A400M Instruct](https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/tree/ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445),
  MoE entraîné de 1,3 milliard de paramètres totaux, environ 400 millions actifs,
  24 couches, 32 experts, top-k 8. Ce sont les vrais poids publics, pas le fixture aléatoire des tests.
- Révision : `ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`.
- Poids BF16 : **2 669 283 096 octets** ; SHA256 officiel et local identiques :
  `f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc`.
- Les fichiers non-LFS, dont configuration et tokenizer, sont vérifiés contre
  leurs identifiants de blobs Git officiels, en plus de leurs SHA256 locaux.
- Machine : MacBook Air M5, 16 Gio. Poids sur le stockage interne. **Fanxiang
  externe absent de cette validation.** Le checkpoint tient dans la RAM physique.
- Calcul FP32 à partir des mêmes poids BF16. `MLX_ENABLE_TF32=0` avant démarrage.
  PyTorch 2.14.1, Transformers 5.20.2, MLX 0.32.4, Safetensors 0.8.0,
  NumPy 2.5.3, psutil 7.2.2.
- Référence indépendante : modules officiels Transformers construits sur `meta`,
  chargement safetensors par groupe, puis libération des paramètres sur `meta`.
  Aucune opération mathématique ni lecteur de poids AirLLM dans la référence.
- Deux prompts : capitale de la France et complétion d'une séquence. Respectivement
  15 et 18 tokens d'entrée ; quatre nouveaux tokens gloutons par cas, cache KV actif.
- Seuils fixés avant la première mesure : `atol=1e-4`, `rtol=5e-4`, mêmes tokens,
  mêmes ensembles top-k à chaque couche/étape ; égalité bit à bit entre variantes natives.
  Aucun seuil relâché après résultat.

Le checkpoint source avant correction est `5a7ea348746c0e630c5f2f8f34c1e65db1e3dbc1`.
Chaque worker enregistre les hashes du moteur et du script, les vérifie à la
fin et partage la même identité avec les autres workers de sa série. La série
CPU instrumentée ajoute la collecte de KV et versions ; ses tableaux sont bit à bit
identiques à ceux de la première série CPU. Les deux séries sont conservées.
La série Metal après correction enregistre les hashes du lecteur corrigé et
des mêmes modules/scripts, sans mutation pendant les workers.
Le contrôle CPU après correction a ensuite été exécuté sur le commit local
`51be52d72df0b1227f48123a467b75652dede370`, avec le même hash du lecteur corrigé
`40542eed377ddb4f243e7d522054ac6d01a4362666cec0b0821a34a8e8e84824`.

## CPU avant correction de la course : tous les contrôles stricts réussis

Temps médians des deux cas, chacun générant quatre tokens, série CPU instrumentée avant correctif :

| Variante | Temps | Erreur absolue max. contre référence | Tokens/top-k | Bit à bit entre natifs |
|---|---:|---:|---|---|
| Référence officielle streamée | 4,160670 s | 0 | référence | — |
| Natif sans prélecture | 1,030532 s | 7,724762e-5 | identiques | référence native |
| Natif avec prélecture exacte | 0,896345 s | 7,724762e-5 | identiques | oui |
| Natif avec SpecPrefetch | 0,911554 s | 7,724762e-5 | identiques | oui |

Le contrôleur CPU termine avec **code 0**. Les temps de la référence et du natif
incluent des stratégies de chargement différentes ; leurs rapports ne mesurent
pas une accélération isolée des noyaux. Le cache OS varie entre les workers.
La mesure courte ne permet pas d'attribuer causalement tous les écarts à la prélecture.

Sur cette série, SpecPrefetch soumet effectivement six lectures : deux utiles,
quatre gaspillées ; respectivement 8 et 16 Mio de payload décodé. Le mécanisme
ne montre pas de gain CPU sur cet échantillon et reste désactivé par défaut.
Les compteurs de prédiction, admission, attentes et octets sont dans les JSON.

### Contre-vérification CPU sur le lecteur corrigé

La même suite stricte a été relancée après correction, sans modifier script,
prompts, précision, budgets ni seuils : **code 0**. Tous les tableaux de logits
et de routage, y compris ceux de la référence, sont **bit à bit identiques à la
série CPU précédente**. Les trois variantes natives restent identiques entre
elles, avec erreur maximale officielle commune `7,724762e-5`.

| CPU après correction, quatre tokens | Temps médian des deux prompts |
|---|---:|
| Référence officielle streamée | 3,510503 s |
| Natif sans prélecture | 1,018905 s |
| Prélecture exacte | 0,891837 s |
| Prélecture exacte + SpecPrefetch | 0,897065 s |

RSS de tous les workers inférieur à **777 Mo** ; cache poids au plus 256 Mio,
E/S en vol au plus 10 Mio / 32 Mio et KV mesuré 12 Mio / 32 Mio. La mémoire
disponible minimale échantillonnée dépasse 3,91 Go, sans déclenchement des gardes.
Les deltas physiques natifs sont respectivement `[54 149 120, 9 060 352]`,
`[9 076 736, 13 533 184]` et `[1 032 192, 0]` octets pour les deux prompts.
Le cache OS reste mixte et non contrôlé ; aucun gain causal n'est déduit de la
variation des temps. Preuves : `granite-cpu-after-stream-fix*.json/.npz` et
`granite-cpu-stream-fix-diagnostic.json`.

## Metal avant correction : échec bit à bit préservé

Première série complète, mêmes prompts et seuils :

| Variante | Temps médian | Erreur max. contre référence | Tokens/top-k | Bit à bit avec natif sans prélecture |
|---|---:|---:|---|---|
| Référence officielle CPU | 2,773790 s | 0 | référence | — |
| Natif Metal sans prélecture | 1,355502 s | 8,106232e-5 | identiques | référence native |
| Natif Metal avec prélecture exacte | 0,979492 s | 1,025200e-4 | identiques | **non** |
| Natif Metal avec SpecPrefetch | 0,955484 s | 8,869171e-5 | identiques | **non** |

La comparaison utilise `atol + rtol*abs(référence)`, donc une erreur maximale
supérieure à `atol` seul n'implique pas un échec de la tolérance combinée.
Toutes les comparaisons officielles respectent le seuil préétabli. En revanche,
la condition supplémentaire bit à bit entre natifs échoue : **code 1 du
contrôleur original**, journal et tableaux préservés.

Différences maximales entre variantes Metal :

- sans prélecture / prélecture exacte : `5,2452087e-5` ;
- sans prélecture / SpecPrefetch : `5,7697296e-5` ;
- prélecture exacte / SpecPrefetch : `4,4643879e-5`.

Le contrôle sans aucune prélecture a été répété deux fois dans un worker puis
comparé à un autre worker sans prélecture : **bit à bit identique**, différence
maximale zéro. Ce résultat ne justifie donc pas une explication par un
nondéterminisme général observé de Metal.

## Cause démontrée et résultat Metal après correction

Dans MLX 0.32.4, le device par défaut est global au processus. Les contextes
`mx.stream(mx.cpu)` utilisés par les workers de décodage le modifiaient et le
restauraient pendant que le thread principal construisait son graphe GPU.
Certaines opérations principales pouvaient donc être placées sur CPU. Les
streams locaux aux threads ne rendaient pas ce device global local au thread.
Sources primaires : [device.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/device.cpp),
[stream.cpp](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/stream.cpp).

Le correctif enlève les trois contextes globaux F32/BF16/FP8 du lecteur et
place explicitement les opérations de conversion/décodage sur le stream CPU.
Il conserve le parallélisme des E/S. Un test contrôlé avec barrières force le
chevauchement : le vrai lecteur d'origine fait échouer l'assertion de device
principal dans les trois formats ; le lecteur corrigé passe. Preuves :
`evidence-io-phase2/worker-stream-race.json` et `tests_next/test_worker_streams.py`.

Même checkpoint, prompts, budgets, précision et seuils, nouvelle série complète :

| Variante après correction | Temps médian / 4 tokens | Erreur max. officielle | Tokens/top-k | Bit à bit entre natifs |
|---|---:|---:|---|---|
| Référence officielle CPU streamée | 7,112001 s | 0 | référence | — |
| Natif Metal sans prélecture | 1,306190 s | 8,106232e-5 | identiques | référence native |
| Natif Metal avec prélecture exacte | 1,071312 s | 8,106232e-5 | identiques | **oui** |
| Natif Metal avec SpecPrefetch | 1,041403 s | 8,106232e-5 | identiques | **oui** |

Le contrôleur termine avec **code 0**. Tous les tableaux de logits et de
routage de ces trois variantes sont aussi **bit à bit identiques au contrôle
Metal sans prélecture d'avant correction**. Le bug est donc corrigé sans
modifier les décisions du témoin exact ni relâcher un critère. Les résultats
sont dans `granite-metal-after-stream-fix*.json/.npz` et
`granite-stream-fix-diagnostic.json`.

Les différences temporelles avant/après ne sont pas attribuées au correctif :
les compteurs physiques ci-dessous et ceux des JSON montrent que l'état du
cache OS varie. Aucune accélération globale ni gain SpecPrefetch général n'est
déduit de ces deux prompts courts.

## Budgets et E/S réelles observées

Série CPU finale : cache poids au plus **256 Mio / 256 Mio**, E/S en vol au plus
**10 Mio / 32 Mio**, cache KV mesuré **12 Mio / 32 Mio**, budget d'espace de
travail 128 Mio. RSS maximal de tous les workers inférieur à **814 Mo** ;
mémoire système disponible minimale échantillonnée supérieure à **4,29 Go**.
Les gardes échantillonnées à 50 ms (5 Gio RSS, réserve disponible 1,5 Gio)
n'ont déclenché aucun arrêt. Ces gardes sont réactives ; elles ne constituent
pas une limite matérielle instantanée.

Après correction, les workers Metal restent sous **752 Mo** de RSS maximal ;
minimum système disponible observé **3,33 Go**. Cache KV mesuré toujours
**12 Mio / 32 Mio** ; cache poids et E/S restent dans leurs budgets.

Les compteurs physiques sont lus pour chaque génération et conservés séparément
des octets logiques du lecteur. Exemples CPU finaux, en **octets** :

| Variante | Prompt 1 | Prompt 2 |
|---|---:|---:|
| Référence | 1 664 090 112 | 36 749 312 |
| Natif sans prélecture | 47 005 696 | 10 698 752 |
| Prélecture exacte | 20 480 000 | 1 982 464 |
| SpecPrefetch | 884 736 | 933 888 |

Après correction, compteurs physiques de la série Metal, mêmes unités :

| Variante | Prompt 1 | Prompt 2 |
|---|---:|---:|
| Référence | 4 383 576 064 | 727 273 472 |
| Natif sans prélecture | 320 671 744 | 102 481 920 |
| Prélecture exacte | 37 715 968 | 28 311 552 |
| SpecPrefetch | 19 136 512 | 46 661 632 |

Le compteur est celui du **processus entier**, sans attribution par fichier ni
périphérique. Le téléchargement et les SHA peuvent réchauffer le cache OS,
dont l'état n'est pas contrôlé ; ces chiffres révèlent des régimes mixtes, ils
ne prouvent pas un débit physique SSD à froid. Les tables Metal et les données
CPU initiales restent dans les JSON, avec les scopes du compteur. Aucun débit
Fanxiang, modèle dépassant la RAM, GLM/Qwen géant ou régime thermique prolongé
n'a été validé ici. Deux prompts courts ne constituent pas une évaluation
générale de qualité du modèle.

## Reproduction et preuves

```sh
python benchmarks/download_granite_checkpoint.py --destination /path/to/Granite
MLX_ENABLE_TF32=0 python benchmarks/benchmark_granite_moe.py \
  --model-dir /path/to/Granite \
  --output evidence-io-phase2/granite/granite-cpu-final.json \
  --device cpu --tokens 4 --repeats 1 --prompts 2 --cache-mib 256 \
  --variants reference native_off native native_spec \
  --max-rss-gib 5 --reserve-gib 1.5
```

Pour Metal, remplacer `--device cpu` par `--device gpu` et utiliser un autre
nom de sortie. Le gate bit à bit reste actif ; le lecteur corrigé le passe sur
les échantillons mesurés ici.
Pour répéter le contrôle sans prélecture : `--variants reference native_off --repeats 2`.

Le protocole détaillé est dans `research/io_wall/GRANITE_MOE_PROTOCOL.md`.
Les tests Granite (architecture, alias, BF16, routages, sessions, budgets,
référence indépendante, corruption et reprise du téléchargement) : **23 réussis**.

Tout est conservé sous `evidence-io-phase2/granite/` :
`checkpoint-provenance.json`, `run-status.json`, séries CPU initiale/finale,
Metal originale et après correction, contrôle Metal répété, diagnostics comparatifs, gardes mémoire,
journaux et tableaux `.npz` avec SHA256. Les poids du modèle ne sont pas versionnés.
