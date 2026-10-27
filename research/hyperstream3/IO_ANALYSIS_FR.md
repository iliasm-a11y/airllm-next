# Analyse des lectures GGUF — HyperStream 3

27 octobre 2026. Les valeurs historiques ci-dessous sont recalculées depuis
les preuves héritées de H2/H1. **Aucune mesure de débit H3 sur le véritable
checkpoint ne découle de cette analyse ou des tests synthétiques.**

## Ce que représentent les 178 192 lectures directes historiques

Source locale reproductible :
[`historical-analysis.json`](../hyperstream2/historical-analysis.json),
champs `weights_counters_inclusive_overlapping`, `first_forward` et
`prefill_model_passes`. La version publiée de cette source est figée au
[commit de base H2](https://github.com/iliasm-a11y/airllm-next/blob/5e8234181e736c0319a64df25e6813e92dd3e6da/research/hyperstream2/historical-analysis.json).
Le document historique explique le périmètre dans
[`HYPERSTREAM_2_REPORT_FR.md`](../../HYPERSTREAM_2_REPORT_FR.md).

| Compteur historique | Valeur |
|---|---:|
| Tokens du prompt, template compris | 364 |
| Bloc de préfill | 16 |
| Passages de préfill | 23 |
| Couches MoE groupées pendant toute la génération | 1 000 |
| Utilisations d'experts, après union des positions du bloc | 56 783 |
| Produits ordinaires packed | 7 843 |
| Chargements directs packed | 178 192 |
| Produits groupés d'experts | 33 198 |
| Paires gate/up pipelinées | 11 066 |
| Octets de copies hôte déclarés évités par le transport direct | 274 460 540 928 |
| Durée inclusive déclarée de préparation/lecture packed | 52,672458 s |

L'identité arithmétique est exacte :

```text
3 × 56 783 + 7 843 = 178 192
```

Le chemin moteur conserve trois projections par expert choisi : gate, up,
down. Dans ces compteurs agrégés, 170 349 sélections correspondent ainsi au
terme experts, soit 95,5986 %, contre 7 843 produits ordinaires, soit 4,4014 %.
Les groupes contiennent les experts ayant le même nombre de positions
routées, par paquets de huit au maximum. Cela explique la multiplication des
préparations malgré les produits groupés ; ce ne sont pas 178 192 lectures de
scalaires ou de métadonnées. Ce rapprochement agrégé ne remplace pas une trace
individuelle des offsets, des hits de cache et des éventuelles tuiles.

Ces compteurs couvrent **toute la génération historique**, et pas uniquement
le préfill : `1 000 = 40 × (23 + 2)` couches sur les 23 passages du prompt et
les deux forwards de décodage suivants. Les trois IDs générés étaient `19`,
`17`, `248046`, soit la réponse « 42 » puis EOS.

Le code historique incrémente `packed_host_copy_bytes_avoided` de deux fois
la taille de chaque résultat direct. On peut donc retrouver pour ce chemin :

```text
Octets packed directs sur toute la génération = 274 460 540 928 / 2
                                             = 137 230 270 464
Taille moyenne par chargement direct          = 137 230 270 464 / 178 192
                                             = 770 125,878 octets
                                             = 752,076 Kio
```

Le premier forward seul rapporte 135 518 514 304 octets logiques, 145 363 525 632
octets de stockage imputés au processus par macOS, et 110,699902 s de TTFT
moteur. Ces domaines et périmètres sont différents. Aucun des deux compteurs
d'octets ne certifie le trafic physique de la NAND. La durée `packed_read_seconds`
inclut de la préparation et peut recouvrir les autres durées du moteur ; elle
ne mesure pas séparément les attentes SSD, le temps CPU ou le temps GPU.

## Changement exécutable du transport

[`gguf_packed_io.py`](../../air_llm/airllm/nextgen/gguf_packed_io.py) ajoute :

- `plan_packed_expert_reads` : validation des sélections et plan de régions,
  sans lecture ni allocation du payload ;
- `read_packed_experts_into` : lecture directe dans une seule allocation NumPy
  finale, contiguë et possédée, de forme `[experts, lignes, octets_par_ligne]`.

L'ordre des experts du demandeur est conservé. Une région n'est fusionnée que
si elle est contiguë dans le fichier **et** dans la destination finale ; aucun
intervalle ni expert non demandé n'est lu. Les sélections de lignes partielles
de deux experts consécutifs restent normalement deux lectures. Le plafond
`max_read_bytes` s'applique aussi aux régions fusionnées.

Le chemin évite la préparation historique d'un buffer temporaire par expert,
suivie de sa copie dans le buffer groupé du kernel. Les octets Q4_K/Q6_K/F32
ne sont ni décodés, ni transformés, ni réécrits sur disque par ce transport.
Les deux budgets, transitoire du lecteur et propriétaire des buffers packed,
sont vérifiés pour le groupe entier avant allocation/lecture. Les alias d'une
seule tranche conservent la charge de l'allocation complète. Les lectures
partielles sont poursuivies dans leur région ; EINTR/EAGAIN ont un nombre de
reprises borné et EOF échoue explicitement. Le verrou du descripteur couvre
l'opération complète afin qu'une fermeture ne recycle pas le descripteur.

L'intégration `grouped_direct_io` est désactivée par défaut et exige le
transport packed direct avec groupement des experts. Elle concerne le chemin
`linear_experts` et les préparations gate/up de `linear_expert_pair`.
Les pré-vérifications du workspace et de la réserve système restent en place.
Un budget propriétaire encore occupé après vidage du cache et synchronisation
refuse toujours la nouvelle allocation avant lecture.

**Compromis de cache explicite :** le groupe ne peuple et ne consulte pas le
cache historique des tuiles individuelles. Un groupe peut donc relire des poids
qui auraient été servis depuis ce cache. Le vidage sous contrainte peut aussi
réduire les hits ultérieurs. Le gain d'allocations/copies et les fusions de
lectures ne garantissent pas un gain de latence ou une baisse des octets lus.

## Vérifications exécutées sans MLX/Torch

Commande reproductible depuis la racine du dépôt, avec Python, NumPy, pytest
et psutil installés :

```sh
python -m pytest tests_hyperstream3/test_grouped_packed_io.py tests_hyperstream3/test_grouped_engine_plumbing.py -q
```

**51 tests réussis** dans cette exécution. Le premier fichier réalise 41
contrôles du transport ; le second réalise dix contrôles de l'intégration réelle
de `GGUFWeights` avec un substitut NumPy du calcul et de la synchronisation.
Les modules MLX et Torch ne sont pas chargés par ces imports, vérifié dans un
processus indépendant. Les principaux cas couverts sont :

- identité de tous les octets Q4_K/Q6_K/F32 et ordre d'experts quelconque ;
- régions adjacentes uniquement, plafonds et refus avant payload ;
- durée de vie complète des propriétaires, alias et tracebacks conservés ;
- lectures courtes, erreurs système, échec d'allocation et fermeture concurrente ;
- options incompatibles, contournement explicite du cache, refus lorsque les
  anciens propriétaires restent vivants après synchronisation simulée ;
- transmission du dispatcher multi-token et de sa taille de tuile au produit
  groupé, sans exécuter ni valider son calcul Metal.

Sur le fichier synthétique borné, les experts complets `[0, 1, 3, 4]` produisent
exactement deux appels de lecture au lieu de quatre et exactement le même
nombre d'octets demandés. Avec quatre lignes partielles, les quatre lectures
restent nécessaires au plan sans surlecture. **C'est une vérification du plan,
pas un benchmark de rapidité ni une mesure du SSD.**

## Mesures encore nécessaires

Les agrégats historiques ne permettent pas de compter exactement combien de
leurs lectures étaient adjacentes ou auraient été servies par un meilleur
cache. Il faut une trace nouvelle d'offsets et de tailles, des routes de chaque
position, des hits/misses, des réservations et des durées de lecture, préparation
et attente. `packed_group_*` compte les octets exacts, les régions prévues, les
fusions structurelles et le temps hôte des lectures ; `read_calls` reste le
nombre réel d'appels réussis, y compris leurs reprises sur lectures courtes.

Il faut comparer le chemin groupé au chemin historique sur les mêmes routes,
prompt, checkpoint et budgets, en incluant les effets du cache et toutes les
synchronisations. Les contrôles synthétiques n'établissent aucun chevauchement
SSD/GPU nouveau ; la paire gate/up existante reste une expérience dont le
chevauchement matériel doit être confirmé. Aucun double buffering nouveau,
disque externe, trafic SSD froid, gain TensorOps ou gain de génération réel
n'est validé par ce travail de transport. Les blocs de prompt plus grands et
la spéculation sont des moyens distincts de réduire les passages répétés sur
les poids ; ils nécessitent leur propre validation numérique et matérielle.
