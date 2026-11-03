# Comparaison isolée CPU → GPU, liée à deux builds réels

Ce dossier ne lance aucun modèle, compilateur ou calcul GPU. Il prépare une
comparaison de diagnostics complets 1 → 600 → 1 ; aucune réponse utile/EOS ou
performance de bout en bout n'est qualifiée par ces fichiers.

Le témoin est le CPU8 original, helper `62a7d257…`, source `558aa441…`, reçu de
build `4cf35cfb…`. Le candidat sera le nouveau runtime CPU → Metal Q8 avec le
helper `cpu8-tail-compatible-v1`, ses propres bibliothèques, sa propre clé et
ses reçus réels. Aucun hash binaire GPU n'est inventé avant le build.

`prepare.json` doit épingler le gel relu indépendamment. `build_binding.py`
produit ensuite une liaison uniquement à partir de builds réussis existants,
après vérification de leurs fichiers. Il ne remplit aucun champ de succès ou
de vérification absent dans un reçu de build.

Les reçus de build du provider doivent contenir les sources exactes sous
`inputs_sha256`, avec `inputs_before == inputs_sha256 == inputs_after`, les
rôles explicites du schéma2, le patch, la feature compilée
et les bibliothèques réellement construites. Le reçu du helper doit aussi
suivre source, snapshot, captures et quatre en-têtes avant/après compilation
et après les smokes de métadonnées, conserver les sorties brutes sous
`parameter_smoke[UB]`, les SHA des bibliothèques, la commande réelle et son
code retour. Les reçus mesurés ne sont jamais réécrits pour satisfaire cela.

Depuis `work/`, après les builds coordonnés et uniquement avec leurs chemins
réels :

```sh
PY=/Users/macbook/Documents/Codex/2026-10-27/mission-codex-installation-qwen-3-8/.venv/bin/python
"$PY" native-gpu-two-build-compare/build_binding.py \
  --reference-runtime-manifest "$CPU_RUNTIME_MANIFEST" \
  --candidate-build "$GPU_DRIVER_BUILD_RECEIPT" \
  --provider-build "$GPU_PROVIDER_BUILD_RECEIPT" \
  --candidate-runtime-manifest "$GPU_RUNTIME_MANIFEST" \
  --candidate-libraries "$GPU_LIBRARIES" \
  --gpu-profile "$GPU_PROFILE" --output "$GPU_BINDING"
"$PY" native-gpu-two-build-compare/gpu_compare.py \
  native-full8/capture "$GPU_FULL/capture" \
  --reference-guard native-full8/guard.json \
  --candidate-guard "$GPU_FULL/guard.json" \
  --binding "$GPU_BINDING" --kind cpu-gpu --output "$CPU_GPU_COMPARISON"
```

Chaque clé est recomputée depuis son propre manifeste. Les différences de
helper, de runtime, de wrapper et de politique GPU ne sont normalisées dans
une copie locale qu'après leur validation par les deux builds et le patch
exact. Modèle, tokenizer, template, préfixe, paramètres mathématiques hors
batch et knobs CPU doivent rester identiques. L'égalité stricte
`report.state_parameters == manifest.state_parameters` reste obligatoire.
Les enveloppes de liaison et de garde natives ne deviennent pas des
paramètres mathématiques implicites.

La couverture complète des 118 noms d'origine, des trois requêtes, logits
finis, routes ordonnées, checkpoints, écritures d'état et continuations
diagnostiques reste exigée. Les comparateurs existants sont épinglés avant
import et appel ; aucun n'est modifié ou remplacé par un monkeypatch.
Les traces brutes doivent prouver des commandes GPU réellement soumises,
terminales et publiées, et le reçu doit prouver le verrou UID hérité avant
lancement et la fermeture du helper possédé.

Deux verdicts numériques sont exposés : Metal existant (`atol=1e-3`,
`rtol=1e-4`, routes/tokens exacts, tout fini) et diagnostic CPU bit à bit.
Pour cette première expérience Q8, `passed` exige en plus l'identité exacte
des états sérialisés CPU/GPU et des écritures de continuation. Une réussite
allclose avec état opaque différent conserve `metal_numeric_passed=true`
mais laisse le verdict global négatif et le cache non qualifié.

Pour une restauration GPU, le couple doit utiliser exactement le même profil
GPU, le même helper/runtime et **la même clé GPU**, sans substitution de clé
CPU :

```sh
"$PY" native-gpu-two-build-compare/gpu_compare.py \
  "$GPU_FULL_OR_SPLIT/capture" "$GPU_RESTORE/capture" \
  --reference-guard "$GPU_FULL_OR_SPLIT/guard.json" \
  --candidate-guard "$GPU_RESTORE/guard.json" \
  --binding "$GPU_BINDING" --kind gpu-restore --output "$GPU_RESTORE_COMPARISON"
```

Ce chemin réemploie le comparateur de restauration exact existant. Son
contrat de frontières reste inchangé : full→restore direct exige 208 aligné
sur UB ; pour UB32+ une référence split et les comparaisons préalables
appropriées restent nécessaires. Une comparaison split→restore seule ne
prouve pas l'équivalence au préfill full.

Tests locaux : petites matrices synthétiques et métadonnées négatives,
aucun modèle/GPU/build. Les smokes ne remplacent pas les futures fixtures
natives multiworkers ni les captures réelles du modèle.
