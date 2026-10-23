#!/usr/bin/env python3
"""Analyze saved GGUF index and existing runs; performs no tensor reads or inference."""
from __future__ import annotations
from collections import Counter
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'evidence-native-gguf'
index_path=EVIDENCE/'inspect.json'
index=json.loads(index_path.read_text())
tensors=index['tensors'];active=index['metadata']['qwen35moe.expert_used_count'];experts=index['metadata']['qwen35moe.expert_count']
traffic=Counter();calls={rows:Counter() for rows in (512,2048,4096)}
for name,tensor in tensors.items():
 shape=tensor['shape'];size=tensor['nbytes']
 if name.startswith('blk.'):
  expert='_exps.' in name;role='active_experts' if expert else 'shared_blocks'
  if expert:assert size%experts==0
  traffic[role]+=size//experts*active if expert else size
  if tensor['quant_type'] in (12,14) and len(shape)>=2:
   for rows in calls:calls[rows][role]+=(active if expert else 1)*math.ceil(shape[-2]/rows)
 elif name=='output.weight':
  traffic['output_head']+=size
  for rows in calls:calls[rows]['output_head']+=math.ceil(shape[-2]/rows)
 elif name=='token_embd.weight':traffic['embedding_row']+=size//shape[0]
 elif name=='output_norm.weight':traffic['output_norm']+=size
cold=sum(traffic.values());sources={}
for p in [ROOT/'air_llm/airllm/nextgen'/n for n in ('gguf.py','gguf_qwen.py','gguf_metal.py','adaptive_cache.py','residency_cache.py')]:
 sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
runs=[]
for p in sorted(EVIDENCE.glob('*metal*.json')):
 if p.name.endswith('.guard.json'):continue
 try:run=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 if run.get('phase')!='generate' or 'metrics' not in run:continue
 steps=len(run.get('generated_ids',[]));metrics=run['metrics'];cache=metrics['cache'];weights=metrics['weights'];args=run['arguments'];duration=run.get('generation_seconds')
 if not steps:continue
 logical=run.get('logical_bytes',metrics['io']['tensor_bytes']);requests=cache.get('hits',0)+cache.get('misses',0)
 item={'path':str(p.relative_to(ROOT)),'status':run['status'],'steps':steps,'rows':args['rows'],
  'logical_bytes':logical,'logical_bytes_per_step':logical/steps,'cold_no_cache_per_step':cold,
  'encoded_bytes_saved_vs_cold':cold*steps-logical,'saved_fraction_vs_cold':1-logical/(cold*steps),
  'cache_hit_fraction':cache.get('hits',0)/requests if requests else None,
  'cache_hits':cache.get('hits',0),'cache_misses':cache.get('misses',0),
  'cacheable_requests_per_step':requests/steps,
  'packed_matmul_calls_per_step':weights['packed_matmul_calls']/steps,
  'predicted_calls_per_step':sum(calls.get(args['rows'],{}).values()),
  'routed_expert_uses':weights.get('routed_expert_uses'),
  'generation_seconds':duration,'tokens_per_second':run.get('generation_tokens_per_second'),
  'packed_read_seconds':weights.get('packed_read_seconds'),'matmul_seconds':weights.get('matmul_seconds'),
  'source_hash_differences_from_current':[name for name,sha in sources.items() if run['identity']['source_sha256'].get(name)!=sha]}
 runs.append(item)
result={'schema_version':1,'method':'Saved index arithmetic and saved run counters only; no tensor payload read, no inference',
 'source_index':str(index_path.relative_to(ROOT)),'index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),
 'current_source_sha256':sources,'cold_encoded_bytes_per_step':cold,'traffic_by_role':dict(traffic),
 'kernel_calls_per_step':{str(rows):dict(counts)|{'total':sum(counts.values())} for rows,counts in calls.items()},
 'cache_capacity_bytes':256<<20,'steady_encoded_bytes_lower_bound_at_capacity':cold-(256<<20),
 'cache_only_maximum_encoded_reduction_fraction':(256<<20)/cold,'runs':runs,
 'periodic_cache_model':{str(requests):{
   'requests_between_uses':requests,'decay_interval':1024,
   'steady_frequency_after_use':1/(1-2**(-requests/1024)),
   'requests_until_benefit_reaches_zero':1024*math.log2(1/(1-2**(-requests/1024)))}
   for requests in (3582,1722)},
 'limits':['Logical read bytes are not NAND bytes or physical SSD bandwidth.',
 'A cache hit still imports packed NumPy bytes into MLX and executes the same product.',
 'matmul_seconds includes host import, dispatch, assignment and synchronization, not GPU timestamp duration.',
 'Class-aware policy speed is unmeasured by this analysis; rows4096 is an arithmetic projection, not a run.']}
(EVIDENCE/'roofline.json').write_text(json.dumps(result,indent=2)+'\n')
short=next((r for r in runs if r['path'].endswith('short4-metal.json')),None)
lines=['# Roofline du chemin GGUF natif — analyse des preuves existantes','',
 'Source : `evidence-native-gguf/inspect.json`, index du fichier de 23 938 321 664 octets et 1 194 tenseurs. Cette analyse ne lit aucun poids et ne lance aucun benchmark. Les octets indiqués sont des tailles encodées GGUF et les appels des projections exactes du code courant.','',
 '## Octets nécessaires pour une position isolée sans cache','',
 '| Famille réellement exécutée | Octets encodés par position |', '|---|---:|']
labels={'active_experts':'Experts sélectionnés : 8 / 256 dans chacune des 40 couches','shared_blocks':'Poids partagés des 40 couches, routeurs et petits paramètres','output_head':'Projection finale complète de 248 320 lignes','embedding_row':'Une ligne de token embedding','output_norm':'Norme finale'}
for key,size in traffic.items():lines.append(f'| {labels[key]} | {size:,} |')
lines += [f'| **Total froid sans cache** | **{cold:,}** |','',
 'Les tenseurs vision (`v.*`) et MTP (`mtp.*`) sont exclus : ils ne sont pas exécutés par ce chemin texte. Une ligne de l’embedding suffit ; sa table complète ne doit pas entrer dans le volume par position.','',
 '## Taille des tuiles et nombre de lancements','',
 '| Lignes par tuile | Experts actifs | Blocs partagés | Sortie | Total / position |', '|---:|---:|---:|---:|---:|']
for rows,counts in calls.items():lines.append(f'| {rows} | {counts["active_experts"]} | {counts["shared_blocks"]} | {counts["output_head"]} | {sum(counts.values())} |')
lines += ['', 'Passer de 512 à 2 048 lignes réduit le compte théorique de 3 765 à 1 542 appels par position, soit 59,04 %. Les octets mathématiquement nécessaires et le routage restent identiques. Le débit réel doit être mesuré : cette baisse du nombre d’appels ne démontre pas une accélération du même facteur. La ligne 4 096 est uniquement une projection arithmétique.','',
 '## Lecture des mesures disponibles','', '| Preuve | État | Positions | Lignes | Octets / position | Appels / position | Débit global |', '|---|---|---:|---:|---:|---:|---:|']
for run in runs:
 rate='en cours' if run['tokens_per_second'] is None else f'{run["tokens_per_second"]:.4f} tok/s'
 lines.append(f'| `{Path(run["path"]).name}` | {run["status"]} | {run["steps"]} | {run["rows"]} | {run["logical_bytes_per_step"]:,.0f} | {run["packed_matmul_calls_per_step"]:,.0f} | {rate} |')
if short:
 lines += ['', f'La série courte à 512 lignes économise {short["encoded_bytes_saved_vs_cold"]/short["steps"]:,.0f} octets par position par rapport au parcours froid, soit {100*short["saved_fraction_vs_cold"]:.2f} %. Son taux de hits est {100*short["cache_hit_fraction"]:.2f} %. Les 15 060 appels observés correspondent exactement à quatre parcours de 3 765 appels : les hits de poids évitent certaines lectures, mais aucune multiplication.', '',
  'Dans cette série, `packed_read_seconds` vaut 1,281 s sur 8,647 s de génération (14,81 %) ; `matmul_seconds` vaut 5,381 s (62,23 %). Le second compteur couvre les imports MLX, lancements, affectations et synchronisations ; il ne mesure pas le temps GPU isolé. Le cache conserve des tableaux NumPy compactés : `packed_matmul` recrée une entrée MLX à chaque appel, y compris après un hit. Le compteur de stockage du processus conserve sa qualification OS et ne certifie pas les octets NAND.']
lines += ['', '## Cache : constat et prochain essai borné','',
 'Les clés de tuiles sont distinctes par nom, borne de lignes, représentation compactée et expert. Les clés expert sont des tuples `(packed:nom@début:compte, expert)` ; le classificateur existant les reconnaît déjà. Les hints de taille du chemin compacté valent bien `compte * row_bytes`. Aucun correctif de clé ou de taille n’est justifié par cette lecture.','',
 'Le cache adaptatif existant a une demi-vie de fréquence de 1 024 requêtes et une histoire non résidente de 4 096 clés. À 512 lignes, une position traverse 3 582 demandes cachables dans la série courte ; à 2 048 lignes, les séries 16/32 positions en traversent 1 722. Les changements d’experts ajoutent des clés et peuvent raccourcir la mémoire effective de l’historique. Les compteurs cumulés ne permettent pas d’attribuer chaque miss à une éviction d’historique.','',
 'Les séries de 16 et 32 positions montrent exactement **203 hits** et **190 257 152 octets économisés** chacune. Les 16 positions supplémentaires n’ajoutent donc aucun hit. Le cache reste rempli à 266 Mo (environ 254 Mio) : la capacité seule n’explique pas une absence totale de réutilisation de paramètres partagés.','',
 'Une cause structurelle plausible est le score existant `max(0, fréquence_décroissante - 1) * coût`. Pour une clé demandée une fois toutes les R requêtes avec une demi-vie D, son régime périodique idéal vaut `F = 1 / (1 - 2^(-R/D))` juste après utilisation. Son bénéfice devient nul après `D * log2(F)` requêtes. Avec R=1 722 et D=1 024, F vaut environ 1,45 et le bénéfice tombe à zéro après environ 552 requêtes, avant la prochaine réutilisation. Avec R=3 582, la fenêtre positive est d’environ 137 requêtes. Ce calcul explique un mécanisme possible de rotation des résidents ; il ne remplace pas une trace individuelle des évictions.','',
 'La projection finale vaut 417 177 600 octets, contre 268 435 456 octets pour tout le cache. Elle ne peut pas résider intégralement. `output_tile_cache=false` évite que son balayage entre en compétition avec les poids réutilisés. On peut retenir une fraction de ses tuiles, mais cela consomme le même budget que les autres poids partagés et ne diminue pas le plancher global.','',
 f'Même un cache idéal de 256 Mio ne peut éviter plus de {100*result["cache_only_maximum_encoded_reduction_fraction"]:.2f} % des octets d’une position isolée en régime établi, avec le chemin de calcul actuel : plancher {result["steady_encoded_bytes_lower_bound_at_capacity"]:,} octets / position. Ce plancher porte sur les lectures de poids de l’application, sans annoncer de débit SSD ou de durée.','',
 'Premier essai ciblé recommandé : conserver la politique adaptive et utiliser son paramètre existant `decay_interval=32768` face à 1 024, avec exactement le même budget de 256 Mio, les mêmes tuiles de 2 048 lignes, mêmes IDs d’entrée et limite de génération. Cette ablation change une seule constante de l’heuristique existante. Comparer IDs/logits, latences hors première position, octets, hits/misses et pic RSS. Elle teste le mécanisme décrit ci-dessus ; son débit supérieur n’est pas établi.','',
 'L’option existante `policy=shared` permet ensuite un essai indépendant de préférence de classe, avec les mêmes limites. La préférence shared est justifiée comme hypothèse par la réutilisation systématique des blocs, mais un balayage partagé de 894,8 Mo dépasse toujours le cache de 256 Mio ; la préférence seule peut donc continuer à tourner. Ne pas modifier simultanément la capacité, les mathématiques, le nombre d’experts ou le mode I/O.','',
 'Une priorité future pourrait conserver les tuiles compactées chaudes directement en MLX sous le même budget. Cela demanderait une mesure séparée des allocations Metal et des copies évitées ; ce changement n’est ni implémenté ni crédité ici.','',
 'Les résultats et hashes des sources lus sont enregistrés dans `evidence-native-gguf/roofline.json`. Les séries historiques ont des hashes GGUF-Qwen différents du fichier courant ; leurs chiffres restent attachés aux hashes enregistrés dans chacune de leurs preuves. Des séries de longueurs différentes, avec cache OS non contrôlé, ne constituent pas une ablation causale. Un run encore en cours conserve son état et ne doit pas être présenté comme une mesure finale.']
(ROOT/'research/native_gguf/ROOFLINE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'cold_bytes_per_position':cold,'calls512':sum(calls[512].values()),'calls2048':sum(calls[2048].values()),'runs':len(runs)}))
