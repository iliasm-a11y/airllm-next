# Performances V4.1 / V5 — mesures et limites

Date : 23 octobre 2026. Plateforme : Linux x86_64, MLX 0.32.4 CPU,
Python 3.12.14. Aucun résultat de cette page n'est un benchmark M5 Metal ou
SSD externe de l'utilisateur. Les résultats historiques `evidence-final/`
restent séparés.

## Checkpoint complet dépassant réellement la RAM

Le CodeLlama-7b-hf original, révision
`6c284d1468fe6c413cf56183e69b194dcfa27fe6`, contient 13 477 093 376 octets
de payload BF16, au-delà des 10 451 439 616 octets de RAM exposée et du
cgroup configuré à 8 Gio. Trois processus frais ont généré les mêmes quatre
tokens sur le même prompt, sans conversion des poids ni modification du config.

| Mesure, une génération par moteur | V4.1 | V5 | Référence HF streamée indépendante |
|---|---:|---:|---:|
| Temps complet de 4 tokens | 165,164 s | 164,401 s | 44,709 s |
| Décodage hors premier token | 0,03616 tok/s | 0,03648 tok/s | 0,09499 tok/s |
| Pic RSS du processus | 1 405 104 128 B | 1 404 461 056 B | 1 370 660 864 B |
| Lectures physiques Linux | 50 575 691 776 B | 50 383 945 728 B | 41 399 136 256 B |

V4/V5 ont des logits identiques bit à bit dans cet essai. La référence, qui
exécute les couches Transformers une par une, correspond aux quatre tokens
et respecte les tolérances fixées (`atol=1e-4, rtol=5e-4`, écart maximal
5,5313e-5). Elle est plus rapide que Next CPU. Cette observation démontre une
génération entraînée hors mémoire, **sans accélération majeure V5, sans vitesse
pratique et sans résultat M5**. Le cache OS n'est pas purgé ; le modèle entier
ne peut y tenir. Les compteurs cgroup incluent les autres processus/cache de
fichiers. Sources : `evidence-v5/trained-large.json`, rapports individuels et
logits `.npz` ; détails dans `FINAL_VALIDATION_V5.md`.

## Contrôle entraîné indépendant

Checkpoint complet : `HuggingFaceTB/SmolLM2-135M-Instruct`, révision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.
SHA256 Safetensors : `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
Taille de fichier : 269 060 552 octets ; il tient dans la mémoire système.

Trois prompts, trois répétitions, seize nouveaux tokens gloutons, FP32,
masque explicite dans la référence. Neuf séquences V4.1 correspondent à la
référence Transformers indépendante. Écart maximal des derniers logits :
7,25×10⁻⁵, sous le critère fixé `atol=1e-4, rtol=5e-4`.
Les médianes de génération sont 5,919 s pour V4.1 et 1,007 s pour la référence
Torch en mémoire. **Le moteur streamé CPU est plus lent dans ce contrôle.**
Son RSS maximal est 939,4 Mo, contre 1 287,4 Mo pour la référence.

Ces médianes ne sont pas le dénominateur des ablations ci-dessous : les
prompts et les budgets sont différents. Source : `evidence-v5/trained-control.json`
et rapports/NPZ voisins. Les autres comparaisons entraînées, y compris le
contrôle final V5 et le checkpoint dépassant la RAM lorsqu'ils sont exécutés,
sont indexées dans `FINAL_VALIDATION_V5.md`.

## Ablations intégrées, même budget à l'intérieur de chaque paire

Deux prompts, deux répétitions chaudes par prompt, deux ordres inversés,
seize nouveaux tokens, FP32. Huit générations chaudes par variante ; un
premier passage par processus est conservé séparément. Les caches logiciels
sont vides au début de chaque processus, le cache de pages OS n'est pas purgé.
Les quatre variantes utilisent les mêmes poids et produisent les mêmes tokens
sur les quarante-huit générations enregistrées (768 tokens émis).

| Paire | Avant | Après | Interprétation |
|---|---:|---:|---|
| Budget poids total 300 Mio, temps complet médian | 8,320 s, cache décodé | 8,631 s, cache source compact | 0,964× ; environ 3,75 % plus lent |
| Même paire, octets logiques lus par génération chaude | 4 729 352 256 B | 0 B | Les lectures sont évitées, le redécodage reste coûteux |
| Même paire, maximum RSS des processus | 829,2 Mo | 772,4 Mo | RSS observé, pas une borne universelle |
| Budget poids total 600 Mio, temps complet médian | 6,889 s, sans résidence des tuiles | 6,502 s, tuiles admises | 1,059× observé sur ces essais |
| Même paire, octets logiques lus par génération chaude | 906 027 840 B | 58 176 B | Réduction de 99,9936 % des demandes au lecteur |
| Même paire, maximum RSS des processus | 939,3 Mo | 1 047,9 Mo | Coût de résidence réellement observé |

L'effet temporel des tuiles varie avec l'ordre : +10,4 % de vitesse dans le
premier ordre, puis environ −2,3 % dans le second. La médiane combinée n'est
pas une preuve d'accélération robuste sur toute charge, encore moins sur GLM.
Les options demeurent désactivées par défaut.

Les processus de mesure étaient distincts, mais l'environnement système
n'était pas totalement isolé : le téléchargement séparé du grand checkpoint
était en cours pendant ces ablations. Le premier passage exclu du premier
worker a aussi pu chevaucher cinq secondes de petits tests. Cette activité
de fond et l'ordre des essais limitent encore l'interprétation des temps ;
les séquences et compteurs d'octets restent les observations des exécutions.
Ces limites sont conservées dans `evidence-v5/weight-residency-analysis.json`.

Source brute complète : `evidence-v5/weight-residency-trained.json`, huit
rapports individuels référencés par ce fichier et journaux associés.
Reproduction :

```bash
PYTHONPATH=air_llm python benchmarks/benchmark_weight_residency_v5.py \
  --model-dir /chemin/SmolLM2-135M-Instruct \
  --tokens 16 --repeats 2 --output evidence-v5/weight-residency-trained.json
```

## Spéculation, reprise et situations défavorables

Les preuves de coût spéculatif et de prompt lookup sont dans
`evidence-v5/speculation*.json` et `research/v5/SPECULATION.md`. Les mesures de
lookup seules ne sont pas des facteurs d'accélération de génération. Les
brouillons sans correspondance, erronés et coûteux restent des contrôles
nécessaires ; les essais n'attribuent pas un gain à un brouillon non testé.

La file RONIN mesure des jobs d'inférence locaux, le temps de rejeu et les
octets des journaux. Une métrique de jobs/heure sur une courte session ne
signifie pas missions autonomes réelles/heure. Aucune mission RONIN externe
n'est exécutée par ce moteur.

## Ce que ces octets ne mesurent pas

Les octets logiques et encodés représentent des demandes aux lecteurs. Les
compteurs physiques Linux, lorsqu'ils existent, sont conservés séparément ;
un zéro après réutilisation des pages OS ne constitue pas un débit SSD.
Le débit USB-C froid, le swap macOS, les températures M5 et la NAND physique
ne sont pas déduits des chiffres ci-dessus. Aucun facteur 2×/5×/10× V5 de
génération géante n'est annoncé.
