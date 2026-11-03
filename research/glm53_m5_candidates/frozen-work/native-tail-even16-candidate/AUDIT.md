# Candidat isolé : préserver la queue CPU8

Statut : **sources préparées, non compilées, non mesurées**. Le profil CPU16
brut qui diverge reste rejeté. Ce candidat ne remplace ni le témoin CPU8, ni
le helper, le wrapper, le snapshot, les bibliothèques ou les comparateurs gelés.

## Hypothèse et source exacte

Dans le checkout V4 reconstruit exactement, `ggml/src/ggml-cpu/ggml-cpu.c`
déclare pour Q8_0 `nrows=2` sous `__ARM_FEATURE_MATMUL_INT8`. Dans le chemin
fallback de `ggml_compute_forward_mul_mat`, `ne11` impair ramène toutefois
`num_rows_per_vec_dot` à 1. Les voies tiled et LLAMAFILE peuvent précéder ce
fallback ; des dimensions et frontières internes impaires peuvent aussi forcer
1. Une taille de lot paire maintient donc seulement une **éligibilité liée à
la parité**, sans prouver le kernel effectivement choisi pour toute opération.

`ggml/src/ggml-cpu/arch/arm/quants.c` contient deux chemins Q8_0 distincts,
`nrc=2` et `nrc=1`, avec des organisations de réduction différentes. Les
comparaisons de composants précédentes n'autorisent aucune assimilation
universelle de leurs résultats. L'hypothèse d'une divergence en fin de préfill
reste à vérifier par les nouvelles captures réelles ; aucun calcul modèle n'a
été exécuté pour ce candidat.

## Politique `cpu8-tail-compatible-v1`

Les pleins lots 8/16/32/64/128 sont conservés. Pour le dernier lot de chaque
phase de calcul :

- une taille paire reste entière ;
- une taille impaire de 1/3/5/7 reste entière, comme la queue CPU8 ;
- une taille impaire supérieure à 8 est divisée en son préfixe multiple de 8,
  puis son reste impair de 1/3/5/7.

Exemples CPU16 :

| Longueur | Découpage candidat |
|---:|---|
| 208 | 13 × 16 |
| 515 | 32 × 16 + 3 |
| 521 | 32 × 16 + 8 + 1 |
| 523 | 32 × 16 + 8 + 3 |
| 542 | 33 × 16 + 14 |

`523 → 32×16 + 10 + 1` serait incorrect pour cette politique : CPU8 conserve
ses trois derniers tokens dans une colonne globale impaire. Le test géométrique
couvre chaque longueur de 1 à 8192 pour les cinq microbatches, en comparant les
intervalles d'éligibilité par token au découpage CPU8. Ce test ne compile et
n'exécute pas la fonction C++ ; il vérifie le contrat source et sa spécification.

La politique est inscrite dans `--state-parameters`, le rapport et les reçus
de chaque microbatch (`unadjusted_take`, `odd_tail_split`, `prefill_chunk_policy`).
La clé du wrapper inchangé inclut automatiquement les nouvelles empreintes du
helper et ces paramètres supplémentaires. Les logits ne sont demandés que
pour le dernier token complet, exactement comme dans le helper de référence.
Les budgets, écritures d'état, checkpoints, snapshot et continuation ne changent pas.

## Préparation, puis build uniquement après coordination

Le layout isolé conserve les chemins relatifs nécessaires au wrapper :
`research/glm53_batched_prefill/{native_prefill.cpp,snapshot.h}` et
`research/glm53_gpu_v3/state_capture.inc`. Le header et l'include sont des copies
identiques ; l'original et les bibliothèques natives restent intacts.

Depuis `work`, le plan est produit sans démarrer de processus :

```sh
$PYTHON native-tail-even16-candidate/build_helper.py \
  --native-source metal-integration-v4-source/source \
  --libraries /absolute/runtime-v4/bin \
  --receipt native-tail-even16-candidate/new-build-plan.json
```

Seulement après clôture des inférences et autorisation du coordinateur, ajouter
`--execute` avec un reçu neuf :

```sh
$PYTHON native-tail-even16-candidate/build_helper.py \
  --native-source metal-integration-v4-source/source \
  --libraries /absolute/runtime-v4/bin \
  --receipt native-tail-even16-candidate/build-receipt.json --execute
```

Cela compile uniquement le nouvel exécutable isolé. Les bibliothèques sont
vérifiées avant/après et ne sont pas reconstruites. Les cinq smokes
`--state-parameters` retournent avant tout chargement modèle. Le SHA du nouveau
binaire et celui du reçu réel alimentent alors `build-binding.json` ; avant ce
build, ils restent absents de `binding-template.json`. Le script n'appelle
aucune inférence, même en mode de compilation.

Le futur lancement matériel doit utiliser le wrapper gelé et son superviseur
UID avec `--helper-source` pointant vers cette copie isolée et `--binary` vers
le nouveau helper. Aucune validation du binaire de référence ne se transfère
au nouveau binaire sans les captures correspondantes.

## Comparateur distinct à deux builds

`variant_compare.py` est nouveau et isolé. Il ne réécrit aucun reçu et ne modifie
pas `native_compare.py` ou `batch_compare.py`. `_read` accepte déjà les chunks
contigus variables ≤microbatch ; `_series` et `_token_compare` alignent les
valeurs logiques stridées selon les positions réelles. En revanche, les anciennes
fonctions de dépendances refusent correctement les nouvelles empreintes de
helper, et le comparateur natif direct demande des frontières identiques.

Le nouveau comparateur lie donc explicitement :

1. le build CPU8 d'origine, binaire `62a7d257…`, source `558aa441…` ;
2. le futur build réel du candidat, source `dac21130…`, binaire encore inconnu ;
3. le patch exact entre ces deux sources et les deux reçus de compilation.

Chaque clé est recalculée à partir de son propre manifeste. Seuls les champs
batch/microbatch, la politique déclarée et les identités helper/source réellement
vérifiées peuvent être normalisés. Shards, tokenizer, template, préfixe,
bibliothèques, runtime, wrapper, snapshot, state_capture et autres paramètres
d'état restent strictement identiques. Les smokes du nouveau binaire et les
reçus de chunks doivent correspondre exactement à la politique annoncée.

Le contrôle numérique réutilise les fonctions existantes : gardes complets,
logits CPU finis bit à bit, routes ordonnées, routeurs/KDA stridés, mêmes
couvertures, états sérialisés et checkpoints communs, continuation et rejeu de
l'agent 1. Il exige les trois requêtes complètes. Un passage numérique reste
limité à ces captures et ne qualifie ni EOS utile ni vitesse globale.

```sh
$PYTHON native-tail-even16-candidate/variant_compare.py \
  native-full8/capture /absolute/new-candidate/capture \
  --reference-guard native-full8/guard.json \
  --candidate-guard /absolute/new-candidate/guard.json \
  --reference-build native-prefill-final-controls.build.json \
  --candidate-build native-tail-even16-candidate/build-receipt.json \
  --binding native-tail-even16-candidate/build-binding.json \
  --output /absolute/new-comparison.json
```

Les tests utilisent une spécification géométrique et des métadonnées synthétiques
neuves pour vérifier les refus faux build/politique/binaire/source/clé. Aucun reçu
mesuré n'est retouché, aucune comparaison complète n'est simulée comme réussie,
et aucun binaire candidat n'est inventé. Les smokes de compilation et la
comparaison numérique restent à exécuter après le vrai build.

## Renforcement de provenance avant le premier build

Les trois headers API explicites et `vendor/nlohmann/json.hpp` font désormais
partie des entrées suivies par chemins/SHA, au même titre que le helper,
`snapshot.h`, `state_capture.inc` et le script de build. Les huit entrées sont
recalculées avant et après la compilation, puis après les smokes de métadonnées.
Les trois headers API correspondent au reçu de référence ; le header JSON est
lié au header actuel du checkout source V4 épinglé, sans prétendre ajouter une
mesure rétroactive au reçu de compilation d'origine. Le comparateur recalcule
également les fichiers d'en-tête du candidat au lieu de croire leur seule
valeur déclarée dans le reçu.

Cinq modules délégués sont épinglés : `native_compare`, `batch_compare`,
`compare_fidelity`, `snapshot_key` et la politique `glm_native`. Leur SHA et
l'emplacement d'un éventuel module déjà chargé sont vérifiés avant import,
autour de chaque appel délégué et avant d'accepter le résultat final. Une
dérive tardive remet toujours `passed` à faux. Les fichiers délégués restent
intacts ; seuls les contrôles du nouveau comparateur isolé sont renforcés.

Six tests négatifs supplémentaires couvrent la dérive de chaque header avant
une compilation, leur dérive dans un callback purement synthétique, un header
omis des entrées suivies, chaque module altéré dans un arbre temporaire,
l'absence d'import avant vérification et un module chargé depuis un autre
fichier. Aucun compilateur, binaire natif, modèle ou GPU n'est appelé par ces
tests. Avec les cinq tests structurels et les sept tests de refus antérieurs,
**18 tests légers passent** ; aucune qualification numérique réelle du candidat
n'en est déduite.
