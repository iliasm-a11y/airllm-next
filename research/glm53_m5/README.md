# GLM-5.3-Flash Q2_K_S : diagnostics de fidélité

Ce diagnostic capture les logits complets et certains tenseurs internes du
runtime natif épinglé à `3f8af15a76ac8675812e856da45454bafc90f12c`.
Il ne mesure pas le débit : les callbacks provoquent lectures, copies et
synchronisations supplémentaires.

## Critères fixés avant lecture des résultats GLM

| Comparaison | F32 | Routes I32 | Tokens |
|---|---|---|---|
| Modification d’I/O seulement | Identité bit à bit | Ordre et IDs exacts | Exact |
| Nombre de threads CPU | Identité bit à bit | Ordre et IDs exacts | Exact |
| Metal/hybride | `abs(test-ref) <= 1e-3 + 1e-4*abs(ref)` | Ordre et IDs exacts | Exact |

Toute valeur NaN ou infinie entraîne un refus, même si elle apparaît des deux
côtés. Les seuils Metal ne sont pas ajustables par les options du comparateur.
Les différences de signe de zéro sont détectées par le critère bit à bit.
La somme et le routage MoE ne sont pas réordonnés pour faire passer un résultat.

Les fichiers doivent provenir des mêmes poids vérifiés. La preuve du hash des
poids, du runtime et de sa configuration est à conserver avec le manifeste de
campagne ; le comparateur ne certifie pas cette provenance à lui seul.

## Compilation liée au build externe

Exécuter depuis la racine d’AirLLM Next, en remplaçant les deux chemins par
ceux du checkout et du build **de la même révision** :

```sh
LLAMA_SOURCE='/chemin/llama.cpp'
LLAMA_BUILD='/chemin/llama-metal'
clang++ -std=c++17 -O2 \
  -I "$LLAMA_SOURCE/include" -I "$LLAMA_SOURCE/ggml/include" \
  research/glm53_m5/native_fidelity.cpp \
  -L "$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  -lllama -lggml -lggml-base -lggml-cpu \
  -o "$LLAMA_BUILD/bin/native_fidelity"
```

Le binaire de diagnostic et les bibliothèques doivent rester ensemble. Il
faut vérifier la compilation et les hashes avant toute exécution réelle.
Aucun backend ne doit être substitué si la compilation ou le chargement échoue.

## Captures sous surveillance

Le dossier de capture et le rapport de garde doivent être nouveaux :

```sh
python3 research/glm53_m5/run_guarded_probe.py --output reference.guard.json -- \
  "$LLAMA_BUILD/bin/native_fidelity" \
  '/chemin/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf' \
  reference-capture 4 0 4 'La capitale de la France est'
```

Les arguments du probe sont : premier shard GGUF, dossier de sortie, nombre de
threads, couches GPU, nombre de tokens, puis prompt. Les experts restent sur
CPU dans le chemin hybride du probe et le cache expert Metal vaut 256 Mio.
Le nombre de couches GPU ne doit pas être extrapolé en budget mémoire sûr.
La garde limite par défaut le RSS à 9 Gio, la croissance du swap système à
64 Mio et la durée à 1200 s ; la pression critique macOS (niveau 4) arrête
le seul groupe de processus enfant possédé. Elle utilise uniquement la
bibliothèque standard Python, sans importer Torch ou charger les poids.

Les rapports contiennent des échantillons du RSS, de l’empreinte mémoire, du
CPU et de `rusage_info_v2` libproc : pageins et octets de stockage comptabilisés
par macOS. Ces octets ne sont ni les lectures logiques GGUF ni les octets NAND.
Les valeurs finales échantillonnées peuvent manquer le dernier intervalle.
Le swap et la pression sont des observations du système entier.

## Comparaison et séquence forcée

```sh
python3 research/glm53_m5/compare_fidelity.py reference-capture candidate-capture \
  --mode cpu --output comparaison-cpu.json
```

Utiliser `--mode io` pour une modification d’I/O uniquement, `--mode cpu` pour
les threads CPU, et `--mode metal` pour le chemin GPU/hybride.
Le code de sortie est 0 uniquement si tous les critères passent, sinon 2.
Le rapport inclut max absolu, RMSE, nombre d’éléments divergents et finitude,
par tenseur et par famille. Les fichiers de tenseurs sont appariés par
`(step, name, occurrence)`, et non par le numéro de fichier. Chaque vue NumPy
respecte les `shape` et `strides` en octets GGML ; les octets de padding ne
sont pas comparés comme des valeurs.

Le comparateur exige `run.json`, `prompt_tokens.json`, `tokens.jsonl`,
`tensors.jsonl`, les vecteurs `logits-STEP.f32` et un marqueur de succès
`complete.json`. Un dossier interrompu ne peut pas passer les critères.

En génération libre, les logits du premier token divergent restent
comparables, car ils utilisent encore le même préfixe. Dès que les tokens
réellement nourris au decode diffèrent, les étapes suivantes sont marquées
incomparables et ne peuvent pas être qualifiées fidèles.

Pour examiner les erreurs numériques sans faire diverger les entrées,
rejouer les tokens de référence comme huitième argument du probe :

```sh
python3 research/glm53_m5/run_guarded_probe.py --output forced.guard.json -- \
  "$LLAMA_BUILD/bin/native_fidelity" \
  '/chemin/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf' \
  forced-capture 4 3 4 'La capitale de la France est' reference-capture/tokens.jsonl
python3 research/glm53_m5/compare_fidelity.py reference-capture forced-capture \
  --mode metal --output comparaison-forcee.json
```

Le champ `token` conserve l’argmax observé. Le champ `next_input_token`
documente le token effectivement fourni au decode suivant. Le mode forcé
ne valide ni la génération libre ni les performances ; des argmax divergents
restent un refus du critère tokens exacts.

## Tests du comparateur

```sh
python3 -m unittest discover -s research/glm53_m5 -p 'test_compare_fidelity.py' -v
```

Ces fixtures synthétiques vérifient les strides avec padding, le zéro signé,
les NaN/inf, les seuils prédéfinis, les routes exactes et les séquences forcées.
Elles ne sont pas des résultats du modèle GLM.
