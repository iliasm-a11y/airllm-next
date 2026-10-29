# RONIN — ajout expérimental AJ-IQ3_XXS

Portage incrémental du 29 octobre 2026, depuis la PR #15 au commit
`eeddbddd0589e92de463a3f904e70d3b8a942e1e`.
Branche : `research/ronin-aj-iq3xxs`. Le moteur GLM5-Next existant est conservé.

## Résultat et limites

Le lecteur prend maintenant en charge IQ2_S, IQ3_XXS et IQ3_S. Le profil de
téléchargement AJ est distinct de Q2_K_S, qui reste le défaut. Les allocations
bornées, les lectures par tuiles et la réserve disque minimale de 32 Gio sont
conservées.

| Qualification | État |
|---|---|
| Codecs IQ contre GGML C indépendant | Validés sur les cas décrits ci-dessous |
| En-têtes réels des trois shards | Inspectés ; structure compatible |
| Génération miniature mixte contre Transformers | Validée ; poids synthétiques |
| Génération du checkpoint AJ complet | **Non exécutée / non validée** |
| Performance et stabilité sur Apple M5 | **Non mesurées / non validées** |
| RONIN utilisable avec ce checkpoint sur 16 Gio | **Non qualifié** |

L'environnement accessible est Linux x86_64. Le Mac, son disque interne et
sa mémoire ne sont pas accessibles ici. Le précontrôle Linux refuse le
téléchargement : environ 26,62 Go libres lors de la mesure pour 112,41 Go de
poids, avant réserve. Aucun checkpoint complet n'a été téléchargé. Un gain de
vitesse ne se déduit pas de cette nouvelle quantification.

## Checkpoint réellement inspecté

Dépôt : https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF
Révision immuable : `07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2`.

| Shard AJ-IQ3_XXS | Taille exacte, octets |
|---|---:|
| 00001-of-00003 | 39 842 483 616 |
| 00002-of-00003 | 39 974 541 184 |
| 00003-of-00003 | 32 594 142 624 |
| Total | **112 411 167 424** |

Les chemins complets et SHA256 LFS vérifiés figurent dans
`research/ronin_aj_iq3xxs/checkpoint-manifest.json`. Les fichiers locaux
conservent leurs basenames ; le sous-dossier Hugging Face est présent dans les
URLs distantes.

L'audit reproductible a lu 12 585 232 octets, dont 2 320 octets d'échantillons
IQ. Deux sondes exploratoires supplémentaires de 256 et 65 536 octets portent
l'investigation GGUF à 12 651 024 octets. Ces petites plages ne constituent pas
un modèle utilisable. Les plages d'en-têtes incluent environ 3,07 Mo de
surlecture de préfixes de tenseurs : ce ne sont pas uniquement des métadonnées.
L'auditeur borne les lectures cumulées à 24 Mio, refuse
HTTP 200 avant lecture du corps et vérifie Range, taille et encodage.

Les trois shards contiennent des tenseurs. Leurs 1 383 noms et dimensions sont
strictement identiques au répertoire Q2_K_S capturé dans la PR #15 :

| Type réel | Tenseurs |
|---|---:|
| F32 | 704 |
| Q8_0 | 214 |
| Q6_K | 339 |
| IQ2_S | 84 |
| IQ3_XXS | 34 |
| IQ3_S | 8 |

Les projections gate/up des 42 couches MoE utilisent IQ2_S. Les projections
down utilisent IQ3_S aux couches 3, 4, 5, 6, 41, 42, 43 et 44 ; les 34 autres
down utilisent IQ3_XXS. Le dispatch lit le type de chaque tenseur.

L'architecture du fichier AJ s'appelle `glm5next` et ses clés portent ce
préfixe, alors que Q2 utilise `glm5-next`. Le backend accepte ces deux
orthographes vérifiées, refuse un mélange des préfixes et contrôle les
dimensions, le vocabulaire, la largeur de l'expert partagé et l'absence de MTP.
Les calculs KDA, DSA/KPool, mHC et MoE restent ceux du port existant.

AJ omet deux métadonnées : sélection de la queue KPool et types d'indexeurs.
Leurs défauts True / tous complets sont confirmés par le loader et les
hparams llama.cpp épinglés, avec provenance dans le dossier de preuves.
Le tokenizer gpt2/glm4 et son template sont lus dans le checkpoint ; le
template AJ réel diffère de celui de Q2 et fait l'objet d'un test de rendu.
Ce test de rendu ne prouve pas la parité des IDs sur le vocabulaire réel.

## Fidélité des codecs et mémoire

Référence GGML : llama.cpp
`7df8b876b91208a86a6d792e1140b3deec14535b`.
Les trois fonctions C amont et leurs tables sont conservées séparément des
décodeurs Python, avec empreintes et licence MIT. Le shim FP16 indépendant
est vérifié sur ses 65 536 représentations.

L'oracle compare **14 104 blocs / 3 610 624 valeurs FP32 bit à bit**, dont
24 blocs réels AJ / 6 144 valeurs. Erreur absolue maximale : zéro.
Les tests couvrent indices de grilles, masques de signes, échelles locales,
frontières FP16, zéros signés et pseudo-aléatoire déterministe. NaN et
infinis sont vérifiés séparément par classification. Il ne s'agit pas d'un
balayage exhaustif de toutes les combinaisons de bits de chaque codec.

Les tables immuables occupent 11 400 octets ; une réserve conservatrice de
16 Kio par lecteur IQ les couvre avant import/décodage. Les temporaires
dépendants des blocs restent bornés par les budgets existants. Les tests
vérifient lecture sélective d'experts, tailles de lignes, buffers empaquetés,
réservation avant payload et durée de vie des sorties.

La miniature AJ répartit de vrais tenseurs synthétiques sur trois shards :
gate/up IQ2_S partout, down IQ3_S aux bords et IQ3_XXS au milieu. Transformers
utilise les mêmes octets déquantifiés par la référence Python amont.
Les deux préfixes passent : erreur maximale de logits 3,6061 × 10⁻⁶ au
préfill et 2,3544 × 10⁻⁶ en continuation, tolérances fixées à rtol 4e-4 /
atol 5e-5. Les argmax, 18 routages, 36 évaluations d'experts et 108 appels
gate/up/down concordent. Ces résultats ne proviennent pas du modèle entraîné.

## Stockage et téléchargement

Le précontrôle est sans téléchargement et sans création de destination.
Il réutilise seulement les fichiers complets ou préfixes dont l'identité est
vérifiée, calcule les octets restant à écrire et garde au moins 64 Mio de
marge auxiliaire plus 32 Gio libres. Il contrôle le volume système et les
volumes APFS partageant un conteneur sans additionner leurs capacités.
La découverte APFS ambiguë reste conservatrice. L'espace purgeable n'est
pas ajouté aux blocs disponibles.

Le téléchargement reste une action explicite. Écriture directement dans la
destination, fichiers partiels identifiés, contrôle de reprise, taille et
SHA256 final, puis renommage sur le même volume : aucun cache HF ni
deuxième copie du checkpoint. Pas de fusion des shards, de suppression de
Q2_K_S ni de nettoyage automatique. L'espace est recontrôlé pendant
l'écriture et avant l'inférence. Une réserve disque ne garantit pas
l'absence de swap.

## Reproduire et préparer l'essai Mac

Les commandes suivantes sont préparées pour le dépôt et un environnement
Python 3.12 équipé des dépendances du projet. Leurs entrées d'aide ont été
vérifiées ici ; les commandes de génération et benchmark n'ont pas été
exécutées sur le Mac.

Précontrôle sur le stockage interne :

```bash
RONIN_MODEL_DIR="$HOME/Models/glm53-aj-iq3xxs"
python scripts/download_glm53_flash.py --profile AJ-IQ3_XXS --destination "$RONIN_MODEL_DIR"
```

Seulement si ce précontrôle réussit et si le téléchargement est souhaité :

```bash
python scripts/download_glm53_flash.py --profile AJ-IQ3_XXS --destination "$RONIN_MODEL_DIR" --download
PYTHONPATH=air_llm python -m airllm.nextgen.glm53_cli inspect "$RONIN_MODEL_DIR/GLM-5.3-Flash-AJ-IQ3_XXS-00001-of-00003.gguf"
PYTHONPATH=air_llm python -m airllm.nextgen.glm53_cli run "$RONIN_MODEL_DIR/GLM-5.3-Flash-AJ-IQ3_XXS-00001-of-00003.gguf" --device gpu --prompt "Bonjour" --max-new-tokens 8 --report outputs/aj-first-run.json
```

Huit tokens constituent un essai initial, pas la preuve d'une réponse
complète. Les sorties gardent les IDs, le texte brut et la cause d'arrêt.
L'inspection seule ne génère aucun token. Le superviseur conserve ses
plafonds : RSS 3 Gio, réserve RAM 4 Gio, croissance de swap 128 Mio,
payload admis 1 536 Mio, délai 900 s ; aucune hausse automatique.

L'audit distant et l'oracle sont reproductibles sans télécharger les poids :

```bash
python scripts/inspect_glm53_aj.py --help
python research/ronin_aj_iq3xxs/validate_iq_codecs.py --output outputs/iq-codecs.json
```

## Protocole de qualification M5 restant

Après vérification complète des trois fichiers, établir une vraie génération,
sa continuation et son annulation sous supervision. Comparer si possible
quelques logits/tokens à une révision llama.cpp compatible et épinglée
utilisant exactement le même checkpoint. Sans cette comparaison, la fidélité
du checkpoint complet reste non validée.

La CLI enregistre séparément chargement/tokenizer, premier token,
préfill forward synchronisé, continuation forward et débit après premier
token. Les appels MLX sont évalués avant les horodatages ; le préfill
forward inclut les lectures de poids et exclut le choix glouton du token.
Les compteurs de mémoire MLX et système sont distincts et se recouvrent :
il ne faut pas les additionner.

Réaliser plusieurs passages avec sorties distinctes, mêmes prompt/contexte/
nombre de tokens et budgets, en distinguant premier passage et passages
répétés. Un nouveau worker ne prouve pas un cache disque froid ; aucune purge
globale n'est effectuée. Rapporter dispersion, erreurs, limites, pression
mémoire, swap, compteurs IO et conditions thermiques accessibles.

Comparer à Q2_K_S seulement si les deux checkpoints sont déjà disponibles
avec assez d'espace. Sans mesure comparable, indiquer « comparaison de
vitesse non réalisée ». Les IO du processus ne mesurent pas directement
les accès physiques NAND. Aucun résultat GPU NVIDIA ni benchmark Qwen
ne qualifie AJ sur ce M5.

## Preuves

La suite ciblée finale donne **276 tests réussis et 3 ignorés** en 87,75 s.
Les trois exclusions concernent Metal, les assets du tokenizer de référence
réel et un autre test opt-in de checkpoint local. Les blocs AJ réels sont,
eux, testés par l'oracle C. La base PR #15 inchangée, exécutée dans le même
environnement, donne 198 réussites et les mêmes 3 exclusions. Aucun échec
dans ces deux suites ciblées. La roue Python construite contient les deux
modules IQ et leur notice MIT ; cela ne valide pas le matériel Apple.

Le dossier `research/ronin_aj_iq3xxs/` contient manifeste, captures de
métadonnées, répertoire de tenseurs, plages et hashes, petits blocs réels,
oracle reproductible, environnement et résultats des tests. Les sources de
référence sont liées dans ces artefacts. Le workflow CPU couvre également
les nouveaux tests ; son résultat distant doit être vérifié après publication.

Q2_K_S est conservé et les tests pertinents sont comparés à la base PR #15
dans le même environnement. Un test ignoré reste non validé. La réussite
des codecs ou de la miniature ne change pas `ronin_ready`, la qualification
du checkpoint entraîné ou celle du M5.

