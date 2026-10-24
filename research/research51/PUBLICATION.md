# Publication GitHub et équivalence des sources

Le transport Git HTTPS local n'avait pas d'identifiants utilisables. La publication utilise le connecteur GitHub autorisé. Les métadonnées des commits sont donc différentes, mais **chacun des quatre arbres Git a été vérifié identique** à l'instantané local correspondant, avant création du commit distant. Les deux historiques Throughput et PR #6 sont conservés comme parents du premier instantané publié.

| Commit local de validation | Instantané GitHub équivalent | Arbre Git identique |
|---|---|---|
| `f82a064c821dcc6d79f4331844928eb1e8d22093` | [`9273749`](https://github.com/iliasm-a11y/airllm-next/commit/9273749e21a659537eb0872677900c0385564322) | `9b89251fab34a813466116dff19f6d4b7db0fa29` |
| `6625211a2bb765a7ab89c1bea273c4419cf64bef` | [`5605e25`](https://github.com/iliasm-a11y/airllm-next/commit/5605e251888fca7adfbe176466a3ae9f2c8bbe48) | `f56ae6049194551a126e7c000008fb527890dad4` |
| `cdd426e1670a95c1798e45962c48e8d7ae87daaf` | [`01e9cb5`](https://github.com/iliasm-a11y/airllm-next/commit/01e9cb5cad50a5e3843c5358f553b6cd1f70c850) | `a7ea9abf0089a2aa60191af851e48520f53a1899` |
| `d4698016da369690d27e2575fc3f1ec1a45cc276` | [`02650f6`](https://github.com/iliasm-a11y/airllm-next/commit/02650f63179db0f9412302f2d80a32dabee51fcd) | `0412921c07f94e0de75328a7831322656f582110` |

Le moteur candidat de la campagne primaire est l'instantané local `cdd426e`, équivalent à l'instantané GitHub [`01e9cb5`](https://github.com/iliasm-a11y/airllm-next/commit/01e9cb5cad50a5e3843c5358f553b6cd1f70c850). Les rapports d'origine gardent les SHA réellement enregistrés pendant la mesure. Le manifeste `results/delivered-engine-sha256.json` vérifie aussi l'identité des fichiers du moteur livré.

Le commit suivant ajoute uniquement cette note et sa table de correspondance au rapport. Aucune source de moteur ni mesure brute n'est modifiée. Aucun poids, secret ou capture privée n'est publié ; `main` n'est pas fusionné.
