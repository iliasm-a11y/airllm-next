# Options de l’adaptateur : vérification statique du parseur V4

Toutes les options de `EXTRA_FLAGS`, ainsi que `--slot-save-path`, existent
dans le parseur V4 reconstruit et sont admises pour `LLAMA_EXAMPLE_SERVER`.
Cette conclusion provient des sources ; aucun binaire, serveur, modèle ou GPU
n’a été lancé pour la vérifier. Le futur contrôle CLI sans modèle reste à faire
pendant une fenêtre autorisée sans moteur.

Sources lues sous `work/metal-integration-v4-source/source` :

- `common/arg.cpp`, SHA256 `83cd576c95940ac4f84059da3c5b34914c15a94a834a608743389f2d43f25ea2`.
- `common/speculative.cpp`, SHA256 `667f3e71026a4701d4c3e6338ca25ba38b7f83d6a41ae9f9c33763beb12a64ad`.
- `tools/server/server.cpp`, SHA256 `d4d343db7c1b648eb5358ae2b87966a5f78ce5c9b245c7fbfe9827ba6971f08b`.

Le serveur appelle `common_params_parse(..., LLAMA_EXAMPLE_SERVER)` en
`tools/server/server.cpp:791`. Le filtre d’options de `arg.cpp:1386` garde les
options communes et celles déclarées pour SERVER, à moins d’une exclusion.
`common_params_parse_ex` enregistre les noms positifs et négatifs à
`arg.cpp:762`, puis applique `handler_bool(params, is_positive)` à la ligne 835.
Les formes négatives ci-dessous sont donc de vrais alias du parseur.

| Option et valeur ajoutées | Définition dans `common/arg.cpp` | Effet du handler |
|---|---:|---|
| `--threads-batch 10` | 1477 | `cpuparams_batch.n_threads=10` |
| `--swa-full` | 1640 | `swa_full=true` |
| `--no-kv-offload` | 2366 | `no_kv_offload=true` |
| `--no-kv-unified` | 1676 | `kv_unified=false` ; option SERVER explicite |
| `--cache-type-k f16` | 2388 | `cache_type_k=GGML_TYPE_F16` |
| `--cache-type-v f16` | 2401 | `cache_type_v=GGML_TYPE_F16` |
| `--flash-attn auto` | 1705 | `LLAMA_FLASH_ATTN_TYPE_AUTO` |
| `--cache-ram 0` | 1666 | `cache_ram_mib=0`, `cache_ram_explicit=true` |
| `--ctx-checkpoints 0` | 1648 | `n_ctx_checkpoints=0` |
| `--no-cache-idle-slots` | 1684 | `cache_idle_slots=false` |
| `--cache-reuse 0` | 3570 | `n_cache_reuse=0` |
| `--no-cache-prompt` | 3563 | `cache_prompt=false` ; option SERVER explicite |
| `--spec-type none` | 4269 | types spéculatifs `{NONE}` |
| `--slots` | 3594 | `endpoint_slots=true` |
| `--jinja` | 3666 | `use_jinja=true` |
| `--slot-save-path <répertoire privé>` | 3602 | `slot_save_path` ; exige un répertoire existant |

`f16` est admis par la liste `kv_cache_types` et la recherche par nom de type
de `arg.cpp:280–299`. `auto` est reconnu par `is_autoy` à `arg.cpp:1292` et par
le handler Flash Attention à `arg.cpp:1713`. `none` figure dans la table de
`speculative.cpp:75` et provoque un retour `{COMMON_SPECULATIVE_TYPE_NONE}` à
`speculative.cpp:2435`. Les paramètres batch/ctx de l’adaptateur sont explicités
en plus des valeurs communes du serveur ; le slot de sauvegarde existe avant
construction de la commande.

Cette lecture ne prouve ni le démarrage du binaire fourni avec ces options, ni
la fidélité numérique HTTP, ni la borne de mémoire effectivement mesurée. Elle
n’annule aucune admission native ou de quarantaine. Si le futur binaire refuse
une option, le parseur rejette les noms inconnus à `arg.cpp:813`, puis
`common_params_parse` restitue l’échec à `arg.cpp:1262` et le serveur retourne 1.
L’adaptateur ferme son enfant sans réessayer avec des options retirées. Le runner
conserve le code d’erreur natif, le journal serveur et un état d’échec après
début d’activité physique ; il ne transforme pas ce cas en succès ou EOS.

Point complémentaire découvert : `arg.cpp:709–759` applique automatiquement
`/etc/llama.cpp/config.ini` puis le fichier utilisateur avant la CLI ;
`common/common.cpp:1032–1045` résout ce dernier à partir de `XDG_CONFIG_HOME`
ou du répertoire `.config` de l’utilisateur. Les flags reconnus ne suffisent
donc pas à garantir tous les paramètres du helper. Le candidat ferme ce trou
par un `XDG_CONFIG_HOME` privé vide propre à son enfant et par le refus de tout
fichier de configuration système/privé présent, sans lecture de ces fichiers.
Ce contrôle ne modifie pas `HOME`, ne supprime aucune configuration et ne change
pas le serveur fourni. Un échec reste un refus explicite.
