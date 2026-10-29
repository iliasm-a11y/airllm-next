# Audit réel des trois shards AJ-IQ3_XXS

Source : [aj9o9/GLM-5.3-Flash-GGUF](https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF/tree/07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2), révision immuable `07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2`.

L'audit a exécuté de vraies lectures HTTP partielles sur les trois fichiers,
avec contrôle du statut 206, de Content-Range, de la taille totale épinglée et
de la longueur du corps. Le parseur n'importe pas le runtime d'inférence et ne
normalise pas les métadonnées pendant l'observation.

| Shard | Taille complète, octets | Tenseurs | Fin brute du répertoire, octets |
|---|---:|---:|---:|
| 1/3 | 39 842 483 616 | 523 | 9 462 291 |
| 2/3 | 39 974 541 184 | 469 | 29 670 |
| 3/3 | 32 594 142 624 | 391 | 24 707 |
| Total | 112 411 167 424 | 1 383 | 9 516 668 |

Le premier shard contient des poids, contrairement au checkpoint Q2_K_S de la
base. Les trois répertoires passent les contrôles de split, doublons, alignement,
bloquage des lignes quantifiées, bornes des données et absence de chevauchement.

L'audit principal a lu **12 585 232 octets**, dont 2 320 octets d'échantillons IQ
explicites. Les préfixes sont demandés par tranches bornées de 1 Mio : leur
surlecture après le répertoire inclut environ 3,07 Mo de premières données de
tenseurs. Ce ne sont pas exclusivement des octets de métadonnées. Deux sondes
préliminaires de 256 et 65 536 octets portent le total de l'investigation à
**12 651 024 octets**, sous le budget de 24 Mio. Aucun shard complet n'a été
téléchargé et aucun hash SHA256 de shard complet n'a été vérifié localement.

## Observations utiles au portage

Les 1 383 noms et dimensions de tenseurs sont identiques à l'inventaire Q2_K_S
conservé dans `research/ronin_glm53/`. Les types observés sont : 339 Q6_K, 704
F32, 214 Q8_0, 84 IQ2_S, 34 IQ3_XXS et 8 IQ3_S.

Gate et up des experts utilisent IQ2_S sur les 42 couches MoE. Down utilise
IQ3_S sur les couches 3, 4, 5, 6, 41, 42, 43 et 44, et IQ3_XXS sur les 34
autres couches MoE. Les dimensions de chaque down expert sont
`[2048, 4096, 288]` dans l'ordre GGUF.

L'architecture s'appelle réellement `glm5next`, avec préfixe de métadonnées
`glm5next.*`, tandis que la base utilise `glm5-next`. AJ ajoute
`expert_shared_feed_forward_length` et `nextn_predict_layers=0`, mais omet les
champs `attention.indexer.kpool_select_tail` et `attention.indexer.types`
présents dans l'autre conversion. La compatibilité doit donc être vérifiée dans
le chargement ; le nom commercial ne suffit pas à établir un alias.

Les valeurs implicites des deux champs d'indexeur absents sont confirmées dans
llama.cpp à la révision `7df8b876b91208a86a6d792e1140b3deec14535b` :
`indexer_kpool_select_tail=true` dans `src/llama-hparams.h` (ligne 319), et
initialisation de tous les indexeurs à « full » avant lecture optionnelle dans
`src/models/glm5-next.cpp` (lignes 44–47). Les lectures de clés optionnelles
absentes laissent les valeurs intactes dans `src/llama-model-loader.cpp`.
Les URL immuables, identités Git et extraits figurent dans
`indexer-defaults-reference.json`.

Le tokenizer embarqué est `gpt2`/`glm4`, avec 154 880 tokens et 321 649 merges.
Le template de conversation réel est conservé. Les grandes listes sont
représentées par leur nombre et leur SHA256 de sérialisation GGUF, sans copier
le vocabulaire dans chaque rapport. Les drapeaux `add_bos_token` et
`add_eos_token` ne sont pas présents dans AJ.

## Preuves et reproduction

`checkpoint-header-audit.json` contient les trois observations et les identités
des plages lues. Les fichiers `shard-*-header.json` permettent aussi de consulter
chaque observation séparément. L'inventaire, le résumé, les métadonnées
architecturales et celles du tokenizer sont conservés dans les JSON adjacents.
Les trois petits fichiers `sample-IQ*.bin` ont une provenance précise dans le
champ `samples` du résumé : révision, shard, tenseur, offset absolu et SHA256.

Commande reproductible, préparée avec le manifeste versionné :

```bash
python3 scripts/inspect_glm53_aj.py --manifest research/ronin_aj_iq3xxs/checkpoint-manifest.json --output-dir outputs/aj-header-audit --sample-blocks 8
```

La commande effectivement exécutée utilisait une copie temporaire des mêmes
champs de sélection ; cette égalité a été contrôlée. Elle figure dans
`header-audit-execution.json`, avec les coûts des sondes supplémentaires.

Les 11 tests de `tests_next/test_glm53_aj_inspection.py` passent. Ils couvrent le
refus d'une réponse 200 avant toute lecture du corps, les plages incohérentes,
les budgets avant ouverture et avant lecture, les dimensions et les captures
réelles. L'oracle C des codecs est une validation séparée.

Cet audit ne valide aucune génération du checkpoint complet, aucun débit, ni
l'exécution ou les ressources du MacBook Air M5.
