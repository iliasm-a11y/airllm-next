#!/usr/bin/env python3
"""Generate the user-facing validation report from actual saved evidence."""
from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parents[1];ev=root/'evidence-next'
r=json.loads((ev/'benchmark-native.json').read_text());s=json.loads((ev/'validation-summary.json').read_text());bounds=json.loads((ev/'hardware-bounds.json').read_text())
rows=[]
for c in r['cases']:
    runs=c['runs']
    rows.append('| '+c['label']+' | '+f"{c['parameter_count']:,}".replace(',',' ')+' | '+f"{statistics.median(x['runtime']['ttft_seconds'] for x in runs)*1000:.2f}"+' | '+f"{statistics.median(x['runtime']['decode_tokens_per_second'] for x in runs):.2f}"+' | '+f"{c['quality_vs_original']['max_abs_logit_error']:.6g}"+' |')
start=json.loads((ev/'startup-process.json').read_text())
med={kind:statistics.median(x['seconds_process_to_first_token'] for x in start['measurements'] if x['backend']==kind) for kind in ('ultra','next')}
source=json.loads((ev/'glm-native-audit.json').read_text())
stress=json.loads((ev/'stress-cpu.json').read_text())
txt=f'''# Validation AirLLM Next — 22 octobre 2026

## État de la livraison

Le moteur natif et les kernels quantifiés ont été réellement exécutés sur
Linux x86_64 avec MLX CPU 0.32.4, Torch 2.14.1+cpu et Transformers 5.20.2.
Aucun GPU M5, SSD Fanxiang ni checkpoint GLM-5.3 complet n’était disponible.
Aucune mesure ci-dessous ne constitue leur validation.

Les corrections d’AirLLM ont été vérifiées face à main
`25778b37fcd63f62ad25e9eeef7fc0b891d3f40f` : 107 fichiers officiels comparés
au SHA Git de l’arbre épinglé, aucune différence. Base Ultra : tag v4.0.0
`ff35db207a0c559af9aa95d686057c3fe84f1d40`. Aucun push ni publication.

## Tests automatisés

| Suite | Tests principaux réussis | Ignorés | Échecs |
|---|---:|---:|---:|
| Nouveau backend Next | {s["next"]["passed_main"]} | {s["next"]["skipped"]} | 0 |
| Régression Ultra | 34 | 4 | 0 |
| Régression AirLLM officielle adaptée hors ligne | 66 | 4 | 0 |

Total : **{sum(v["passed_main"] for k,v in s.items() if k in ("next","ultra","upstream"))} tests principaux réussis, 9 ignorés et 5 sous-tests réussis**.
Les XML JUnit comptent séparément ces cinq sous-tests ; leur total diffère donc
du nombre de fonctions/tests principaux affiché par pytest. Les sauts portent
sur Metal/MPS ou CUDA indisponibles. Les tests CUDA historiques de compression
ne sont pas présentés comme validés.

Contrôles : logits et séquences, routeur et biais appris, indexeurs DSA partagés,
RoPE GLM, prefill découpé, KV incrémental et contextes bornés, sessions prolongées,
FP8 E4M3 sur les 256 motifs, blocs et bords non multiples, experts indépendants et
banques FP8 3D, vrais matmuls affines MLX 4/8 bits, quantification mixte,
projection finale tuilée, budgets saturés, déduplication de futures, lectures
courtes/interrompues, conversion interrompue/reprise et restauration du préfixe
KV après erreur. L’autotuning ABBA et sa réutilisation sont également testés.

Un test historique AutoModel utilisait des configurations distantes et attendait
d’anciennes classes consolidées par l’amont. Il a été remplacé par un test de
dispatch actuel sur configurations mockées, sans réseau. La suite est exécutée
avec `HF_HUB_OFFLINE=1` et `TRANSFORMERS_OFFLINE=1` ; le harnais manuel de grands
modèles n’est pas lancé. Les helpers de compression publics sont réexportés.

La wheel `dist-next/airllm-4.0.0+next.1-py3-none-any.whl` a été construite puis
installée dans un répertoire isolé ; une génération miniature a été exécutée.
Les fichiers Python de la wheel ont été comparés au source livré. `compileall`
et la vérification syntaxique des scripts shell ont réussi.

## Mesures CPU réelles, modèles miniatures

Trois générations de 32 tokens par configuration, batch 1, prefill de 8 tokens.
Le cache OS n’a pas été purgé ; cet environnement cloud partagé ne constitue
pas un banc matériel dédié, et les temps bruts des trois répétitions sont conservés. Les poids sont aléatoires ; ces modèles servent à
valider les calculs, pas à évaluer les compétences linguistiques de GLM-5.3.
TTFT et débit ci-dessous : médianes, processus déjà initialisé, caches logiciels
pouvant être chauds. Le débit exclut le prefill et le premier token.

| Fixture / format | Paramètres | TTFT (ms) | Décodage (tok/s) | Erreur max logits vs original |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

Un essai spécifique a mesuré une différence maximale de 0,772 sur un matmul
4 bits à échelles/accumulation BF16 face à la référence déquantifiée FP32. Le
chemin retenu conserve les échelles et l’accumulation FP32 : l’erreur du kernel
retombe à environ 1,15e-5 sur ce cas. Les métriques exactes sont conservées dans
`quantization-precision.json`. Cette décision évite de masquer le problème par
une tolérance large et reste à profiler sur Metal.

Les versions 4/8 bits ont une erreur non nulle face aux poids d’origine. Les
tests de kernels les comparent séparément à une référence utilisant exactement
les mêmes poids déquantifiés. L’accord de tokens sur ces fixtures ne certifie
aucune absence de perte de qualité sur GLM-5.3.

La mesure Ultra CPU de référence sur tiny64 donne
{r['ultra_cpu_baseline']['total_tokens_per_second_including_prefill']:.2f} tok/s,
prefill inclus. Elle ne se compare pas directement à la colonne « décodage ».
Ces microtests ne démontrent aucune supériorité générale de débit du nouveau
backend face à Ultra ; selon la configuration et le bruit de mesure, Ultra peut
rester plus rapide. Le principal résultat ici est une voie GLM native correcte
et les mesures explicites de ses coûts.

## Démarrage : phases réellement séparées

Dans trois nouveaux processus Python par backend, sur le même GLM miniature,
**sans tokenizer et avec cache OS chaud**, la médiane du lancement jusqu’au
premier vrai token est de **{med['ultra']:.3f} s pour Ultra** et
**{med['next']:.3f} s pour Next**. Le token produit est identique. Cela isole
surtout le coût logiciel des dépendances : il ne prédit pas l’accès à 755 Go
ni le TTFT du modèle complet. Avec le tokenizer Transformers, une partie de
son temps d’import revient lors du premier lancement.

L’import seul est mesuré à part dans `import-before.json` et `import-after.json` :
la commande Next passe d’environ 4,5 s à 0,11 s sur cet hôte après les imports
à la demande. Un import n’est pas une préparation des poids ni une génération.

Les tests de manifeste distinguent une première lecture des en-têtes et leur
réutilisation. Les générations successives conservent les caches de poids ;
les données exactes d’E/S, attente, TTFT, déquantification, RSS, swap, hits et
budgets figurent dans chaque enregistrement JSON.

## Stabilité répétée sur CPU

Le stress test miniature a généré **{stress['tokens_generated']} tokens** sur
20 requêtes de 128 tokens, en {stress['seconds']:.2f} secondes, avec croissance
KV au-delà d’un bloc de 128. Le RSS final diffère de
{stress['rss_growth_after_first_request_bytes']/1024**2:.2f} Mio par rapport à
l’après-première requête ; les budgets de cache et d’E/S restent respectés.
Il s’agit d’une mesure de stabilité courte sur petit modèle, sans évaluation
thermique du M5 et sans extrapolation au modèle complet.

## Lectures et politiques de cache

Les compteurs incluent les poids quantifiés lus dans le dérivé, et pas seulement
les fichiers d’origine. Les octets logiques de décodage sont séparés de ceux du
prefill. Sur tiny64 entièrement en cache, seules les lignes d’embedding relues
restent à 256 octets par token ; ce résultat n’est pas transposable au 744B.

36 configurations synthétiques couvrent uniforme, Zipf, balayage mêlé à un
ensemble chaud et changement d’ensemble chaud. À capacité 32 experts, les hits
LRU/SLRU/LFU valent respectivement : uniforme 12,78/12,50/12,24 % ; Zipf
69,06/74,40/77,99 % ; ensemble chaud + scan 75,76/76,84/78,16 % ; ensemble chaud
mobile 99,68/99,68/78,82 %. Aucune politique ne domine toutes les distributions.
Le fichier de replay et l’autotuning conservent cette distinction.

Les lectures `pread`/`mmap` et 1/2/4 workers sont réellement exécutées.
Les lectures physiques Linux peuvent rester nulles sur un stockage déjà en
cache. Sur macOS, leur compteur de processus n’est pas implémenté et reste
`null`. Les mesures logiques ne sont jamais annoncées comme débit du Fanxiang.
Sur Linux, le RSS est lu via `/proc/self/status` pour éviter les erreurs de
namespace PID de certains environnements conteneurisés.

## Limites physiques pour GLM-5.3

Le plan vérifie structurellement {source['metadata_keys_matched']:,} clés nécessaires
contre l’index officiel et les formes du modèle Transformers méta. Les poids
et leurs 141 en-têtes complets ne sont pas téléchargés. La taille officielle
issue de l’audit Ultra est d’environ 755,6 Go ; elle n’est pas une consommation
RAM. Les paramètres routés actifs des seules matrices d’experts représentent
22 649 241 600 paramètres par token.

En FP8 avec échelles : environ **22,655 Go d’experts actifs par token** sans
cache. En affine 4 bits groupe 64 avec échelles/biais FP32 sur disque :
**14,156 Go**. À 1 Go/s soutenu, ces experts seuls demandent au moins ce nombre
de secondes, avant cache, calcul, attention et sortie. Le profil exact et la
localité des routes déterminent ce qu’un petit cache peut économiser.

L’estimation architecturale ajoutant les grandes matrices hors experts donne,
sans cache, environ {bounds['full_decode_no_cache_estimates']['fp8_original_approx_bytes_per_token']/1e9:.2f} Go/token
au format d’origine, {bounds['full_decode_no_cache_estimates']['experts_int4_only_approx_bytes_per_token']/1e9:.2f} Go/token
avec les seuls experts 4 bits et {bounds['full_decode_no_cache_estimates']['mixed_int4_approx_bytes_per_token']/1e9:.2f} Go/token
en mixte 4 bits plus agressif. Ces estimations supposent des grandes projections
FP8 et une sortie/routeur BF16 ; les en-têtes complets et le débit réel restent
à mesurer. Les surcoûts de certaines normes/échelles sont exclus. Ce ne sont
pas des performances prédites garanties.

Le cache KV latent BF16 demande environ 95 232 octets par token pour les
78 couches et 21 indexeurs : **environ 99,86 Go pour 1 048 576 tokens**,
avant les autres mémoires. Le million de contexte ne peut donc pas être promis
sur 16 Go avec cette représentation. La limite locale est contrôlée.

La bande passante mémoire M5 annoncée par Apple, 153 Go/s, est distincte de celle
d’un SSD USB. Aucune optimisation de chargement ne rend ces deux débits égaux.
Les tableaux de scénarios à 0,5/1/2/3 Go/s dans `hardware-bounds.json` sont des
bornes théoriques optimistes, pas des mesures.

## Restant à certifier sur matériel

Metal et Neural Accelerators M5, coût des synchronisations, performances BF16
et entières sur les vraies formes GLM, débit/latence USB Fanxiang, pression
mémoire/swap avec le checkpoint complet, stabilité thermique soutenue et qualité
applicative des quantifications. Des scripts dédiés préparent ces mesures.

Le MTP n’est pas exécuté. Le speculative decoding n’est pas annoncé comme
implémenté. Aucun kernel Metal personnalisé supérieur aux primitives MLX n’a
été démontré. La persistance concerne index, conversions et profils ; elle ne
sauvegarde pas le KV complet d’une session entre deux processus.
'''
(root/'docs/NEXT_VALIDATION.md').write_text(txt)
print(root/'docs/NEXT_VALIDATION.md')
