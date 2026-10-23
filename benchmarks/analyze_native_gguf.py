"""Read-only aggregation of supervised, paired native GGUF research evidence.

No model imports or weight reads. Rates use sums of tokens divided by sums of
elapsed times; paired speedups require matching prompt, output IDs and budgets.
Missing/in-progress files stay pending. Full-vocabulary logit comparisons only
use an independently referenced position, never the last vector of an 8-step
run against the fourth reference vector.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EOS={248044,248046}
BUDGET_FIELDS=('cache_bytes','io_bytes','kv_bytes','workspace_bytes','reserve_bytes',
               'max_tensor_bytes','prefill_chunk','max_context','dtype','device')
CAVEATS=[
    'Checkpoint Qwen3.6 GGUF quantifié testé, sans certification de la qualité du checkpoint BF16 original.',
    'Trois prompts, deux répétitions, greedy batch un et huit tokens maximum; EOS peut raccourcir la génération.',
    'La référence indépendante exécute les blocs officiels Transformers/Torch mais partage le lecteur et les codecs GGUF.',
    'La référence Torch indépendante couvre quatre positions maximum par prompt; les tokens suivants sont comparés entre variantes natives seulement.',
    'Trois niveaux de coût: génération (préfill et boucle modèle), mur worker (inclut son initialisation), processus supervisé (imports Python, initialisation, génération et nettoyage, depuis guard.elapsed_seconds). Le dernier est le débit tous coûts processus de cette campagne.',
    'Les IDs d’entrée sont pré-calculés dans la campagne: la tokenisation et le rendu du chat sont validés séparément et restent exclus même du temps processus supervisé de ces essais.',
    'TTFT mesuré après initialisation; la somme initialisation+TTFT reste distincte du démarrage complet du processus.',
    'Cache OS non purgé et état non contrôlé, malgré ordre de campagne randomisé. Les compteurs stockage sont ceux du processus selon macOS, avec réadahead; ils ne certifient ni octets NAND ni bande passante physique du SSD.',
    'Même budget cache mais représentations différentes: référence déquantifiée, Metal compacté. Les gains mesurent le chemin complet et son cache, pas le kernel GPU isolé.',
    'cpu_seconds mesure du temps CPU processus. matmul_seconds inclut imports, lancements et synchronisations; aucun temps GPU isolé ou bilan énergétique ne peut en être déduit.',
    'RSS échantillonné et swap système complètent les réservations avant allocation. La croissance du swap est système, avec un swap initial déjà présent; absence de croissance ne signifie pas swap nul.',
    'Identité des tokens avec la référence ne prouve pas une réponse sémantiquement juste: le prompt brut 2+2= produit 5 puis EOS sur ce checkpoint.',
    'Chemin texte uniquement: vision et MTP non exécutés. Aucun résultat SSD externe, long contexte, thermique prolongé ou modèle 300–700B n’est établi.',
    'Ces mécanismes ont des antériorités documentées; les mesures locales ne prouvent aucune nouveauté algorithmique.'
]


def display_path(path):
    path=Path(path)
    try:return str(path.resolve().relative_to(ROOT))
    except ValueError:return str(path)


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError,OSError):return None


def positive(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0


def ratio(numerator,denominator):return numerator/denominator if denominator>0 else None


def aggregate(runs):
    """Pool work and elapsed time; never average per-run token rates."""
    if not runs:return {'runs':0}
    tokens=sum(len(r['generated_ids']) for r in runs)
    generation=sum(r['generation_seconds'] for r in runs)
    decode_tokens=sum(max(0,len(r['generated_ids'])-1) for r in runs)
    decode_seconds=sum(sum(r['token_seconds'][1:]) for r in runs)
    wall=sum(r['wall_seconds'] for r in runs)
    hits=sum(r.get('metrics',{}).get('cache',{}).get('hits',0) for r in runs)
    misses=sum(r.get('metrics',{}).get('cache',{}).get('misses',0) for r in runs)
    storage=[r.get('process_storage_read_bytes') for r in runs]
    logical=sum(r['logical_bytes'] for r in runs)
    cpu=[r.get('cpu_seconds') for r in runs]
    guards=[r['_guard'] for r in runs]
    elapsed=[g.get('elapsed_seconds') for g in guards]
    supervised=sum(elapsed) if all(positive(v) for v in elapsed) else None
    return {'runs':len(runs),'generated_tokens':tokens,'generation_seconds':generation,
        'pooled_generation_tokens_per_second':ratio(tokens,generation),
        'worker_wall_seconds':wall,'pooled_worker_wall_tokens_per_second':ratio(tokens,wall),
        'supervised_process_seconds':supervised,'supervised_elapsed_measured_runs':sum(positive(v) for v in elapsed),
        'pooled_supervised_process_tokens_per_second':ratio(tokens,supervised) if supervised is not None else None,
        'decode_tokens':decode_tokens,'decode_seconds':decode_seconds,
        'pooled_decode_tokens_per_second':ratio(decode_tokens,decode_seconds),
        'median_ttft_seconds':statistics.median(r['token_seconds'][0] for r in runs),
        'median_initialization_plus_ttft_seconds':statistics.median(r.get('initialization_seconds',0)+r['token_seconds'][0] for r in runs),
        'logical_weight_bytes':logical,'logical_bytes_per_generated_token':ratio(logical,tokens),
        'storage_counter_measured_runs':sum(v is not None for v in storage),
        'process_storage_read_bytes':sum(storage) if all(v is not None for v in storage) else None,
        'process_storage_read_bytes_observed_sum':sum(v for v in storage if v is not None),
        'cpu_seconds':sum(cpu) if all(v is not None for v in cpu) else None,
        'cpu_seconds_per_generated_token':ratio(sum(cpu),tokens) if all(v is not None for v in cpu) else None,
        'sampled_peak_rss_bytes':max(g['sampled_peak_rss_bytes'] for g in guards),
        'maximum_system_swap_growth_bytes':max(g['maximum_swap_growth_bytes'] for g in guards),
        'initial_system_swap_used_bytes_range':[min(g['before']['swap_used_bytes'] for g in guards),max(g['before']['swap_used_bytes'] for g in guards)],
        'minimum_system_available_bytes':min(g['minimum_available_bytes'] for g in guards),
        'cache_hits':hits,'cache_misses':misses,'pooled_cache_hit_rate':ratio(hits,hits+misses),
        'cache_peak_bytes':max(r.get('metrics',{}).get('cache',{}).get('peak_bytes',0) for r in runs),
        'matmul_host_wall_seconds':sum(r.get('metrics',{}).get('weights',{}).get('matmul_seconds',0) for r in runs),
        'packed_read_host_wall_seconds':sum(r.get('metrics',{}).get('weights',{}).get('packed_read_seconds',0) for r in runs),
        'read_dequant_host_wall_seconds':sum(r.get('metrics',{}).get('weights',{}).get('read_dequant_seconds',0) for r in runs),
        'gpu_command_seconds':None,'energy_joules':None,'physical_ssd_bandwidth_certified':False}


def paired_speedups(index,variants,baseline='reference'):
    result={}
    for variant in variants:
        if variant==baseline:continue
        pairs=[];excluded=[]
        for (case,repeat,name),run in sorted(index.items()):
            if name!=variant:continue
            other=index.get((case,repeat,baseline))
            if other is None:continue
            differences={key:[other.get('options',{}).get(key),run.get('options',{}).get(key)] for key in BUDGET_FIELDS
                         if other.get('options',{}).get(key)!=run.get('options',{}).get(key)}
            source_differences={key:[other.get('identity',{}).get('source_sha256',{}).get(key),run.get('identity',{}).get('source_sha256',{}).get(key)]
                for key in ('air_llm/airllm/nextgen/gguf.py','air_llm/airllm/nextgen/gguf_qwen.py','air_llm/airllm/nextgen/gguf_metal.py')
                if other.get('identity',{}).get('source_sha256',{}).get(key)!=run.get('identity',{}).get('source_sha256',{}).get(key)}
            same_ids=run['generated_ids']==other['generated_ids'] and run['input_ids']==other['input_ids']
            if differences or source_differences or not same_ids:
                excluded.append({'case':case,'repeat':repeat,'same_ids':same_ids,'budget_differences':differences,'source_differences':source_differences});continue
            pairs.append({'case':case,'repeat':repeat,'generated_tokens':len(run['generated_ids']),
                'baseline_generation_seconds':other['generation_seconds'],'variant_generation_seconds':run['generation_seconds'],
                'baseline_supervised_process_seconds':other['_guard']['elapsed_seconds'],'variant_supervised_process_seconds':run['_guard']['elapsed_seconds'],
                'baseline_decode_seconds':sum(other['token_seconds'][1:]),'variant_decode_seconds':sum(run['token_seconds'][1:]),
                'generation_speedup':other['generation_seconds']/run['generation_seconds'],
                'decode_speedup':ratio(sum(other['token_seconds'][1:]),sum(run['token_seconds'][1:])),
                'ttft_speedup':other['token_seconds'][0]/run['token_seconds'][0]})
        ratios=[p['generation_speedup'] for p in pairs]
        result[variant]={'baseline':baseline,'eligible_pairs':len(pairs),'excluded_pairs':excluded,'pairs':pairs,
            'pooled_generation_speedup':ratio(sum(p['baseline_generation_seconds'] for p in pairs),sum(p['variant_generation_seconds'] for p in pairs)),
            'pooled_decode_speedup':ratio(sum(p['baseline_decode_seconds'] for p in pairs),sum(p['variant_decode_seconds'] for p in pairs)),
            'pooled_supervised_process_speedup':ratio(sum(p['baseline_supervised_process_seconds'] for p in pairs),sum(p['variant_supervised_process_seconds'] for p in pairs)),
            'median_generation_speedup':statistics.median(ratios) if ratios else None,
            'geometric_mean_generation_speedup':math.exp(statistics.mean(math.log(v) for v in ratios)) if ratios else None}
    return result


def eos_prefix(ids,limit=4):
    result=[]
    for token in ids[:limit]:
        result.append(token)
        if token in EOS:break
    return result


def compare_logits(actual,expected,*,atol=2e-4,rtol=5e-5):
    a=np.asarray(actual,dtype=np.float64).reshape(-1);b=np.asarray(expected,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape:return {'status':'failed','reason':'shape_mismatch','actual_shape':list(a.shape),'reference_shape':list(b.shape)}
    if not len(a) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {'status':'failed','reason':'nonfinite_or_empty_logits'}
    error=np.abs(a-b);passed=bool(np.allclose(a,b,atol=atol,rtol=rtol))
    top=np.partition(b,-2)[-2:] if len(b)>1 else np.array([b[0],b[0]])
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return {'status':'passed' if passed else 'failed','vocabulary_entries':len(a),'atol':atol,'rtol':rtol,
        'maximum_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean()),
        'root_mean_square_error':float(np.sqrt(np.mean((a-b)**2))),
        'cosine_similarity':float(np.dot(a,b)/denominator) if denominator else None,
        'actual_argmax':int(a.argmax()),'reference_argmax':int(b.argmax()),'argmax_equal':bool(a.argmax()==b.argmax()),
        'reference_top1_margin':float(top[1]-top[0])}


def guarded(path):
    g=read_json(Path(path).with_suffix('.guard.json'))
    return bool(g and g.get('violation') is None and g.get('exit_code')==0),g


def compare_dump(native_path,dump_path,reference_path,reference_dump):
    native=read_json(native_path);reference=read_json(reference_path)
    if not native or not reference or not Path(dump_path).exists() or not Path(reference_dump).exists():
        return {'status':'pending','reason':'missing_report_or_logits'}
    if not guarded(native_path)[0] or not guarded(reference_path)[0]:return {'status':'pending','reason':'missing_successful_guard'}
    ids=native.get('generated_ids',[]);ref_ids=reference.get('generated_ids',[])
    if not ids or len(ids)>len(ref_ids):return {'status':'not_compared','reason':'final_position_exceeds_independent_reference','native_positions':len(ids),'reference_positions':len(ref_ids)}
    if native.get('input_ids')!=reference.get('input_ids') or ids!=ref_ids[:len(ids)]:
        return {'status':'failed','reason':'different_prompt_or_autoregressive_prefix'}
    try:
        with np.load(reference_dump,allow_pickle=False) as archive:
            rows=archive['logits'];saved=archive['generated_ids'].tolist()
            if saved!=ref_ids or rows.shape[0]!=len(ref_ids):return {'status':'failed','reason':'reference_dump_identity_mismatch'}
            expected=np.array(rows[len(ids)-1],copy=True)
        actual=np.load(dump_path,allow_pickle=False)
        expected_hash=native.get('last_logits_sha256')
        if expected_hash and hashlib.sha256(actual.tobytes()).hexdigest()!=expected_hash:
            return {'status':'failed','reason':'native_logit_hash_mismatch'}
        result=compare_logits(actual,expected)
        if result.get('actual_argmax')!=ids[-1] or result.get('reference_argmax')!=ref_ids[len(ids)-1]:
            result['status']='failed';result['reason']='argmax_does_not_match_saved_token'
        return result|{'native_report':display_path(native_path),'reference_report':display_path(reference_path),'position_1_based':len(ids)}
    except (ValueError,KeyError,OSError) as error:
        return {'status':'failed','reason':'invalid_logit_dump','detail':str(error)}


def long_runs(evidence,campaign_hashes):
    """Stability evidence is separate from short paired speed comparisons."""
    evidence=Path(evidence);refs={}
    for path in evidence.glob('torch-*.json'):
        ref=read_json(path)
        if ref and isinstance(ref.get('input_ids'),list) and ref.get('generated_ids') and guarded(path)[0]:
            key=tuple(ref['input_ids'])
            if key not in refs or len(ref['generated_ids'])>len(refs[key]['generated_ids']):refs[key]=ref
    result=[]
    for path in sorted(evidence.glob('*.json')):
        run=read_json(path)
        if not run or run.get('phase')!='generate' or run.get('arguments',{}).get('tokens',0)<32:continue
        ok,guard=guarded(path);ids=run.get('generated_ids',[]);requested=run['arguments']['tokens']
        sources=run.get('identity',{}).get('source_sha256',{})
        source_match=all(sources.get(f) in campaign_hashes.get(f,set()) for f in (
            'air_llm/airllm/nextgen/gguf.py','air_llm/airllm/nextgen/gguf_qwen.py','air_llm/airllm/nextgen/gguf_metal.py'))
        ref=refs.get(tuple(run.get('input_ids',[])));prefix=eos_prefix(ref['generated_ids']) if ref else []
        status='completed' if ok and run.get('status')=='completed' else 'failed' if (guard and (guard.get('violation') or guard.get('exit_code') not in (None,0))) or run.get('status')=='failed' else 'pending'
        result.append({'report':display_path(path),'status':status,'requested_tokens':requested,'generated_tokens':len(ids),
            'requested_length_reached':len(ids)>=requested if status=='completed' else False,
            'generation_seconds':run.get('generation_seconds'),'generation_tokens_per_second':run.get('generation_tokens_per_second'),
            'worker_wall_tokens_per_second':ratio(len(ids),run['wall_seconds']) if positive(run.get('wall_seconds')) else None,
            'supervised_process_seconds':guard.get('elapsed_seconds') if guard else None,
            'supervised_process_tokens_per_second':ratio(len(ids),guard['elapsed_seconds']) if guard and positive(guard.get('elapsed_seconds')) else None,
            'decode_tokens_per_second':run.get('decode_tokens_per_second'),'ttft_seconds':run.get('ttft_seconds'),
            'sampled_peak_rss_bytes':guard.get('sampled_peak_rss_bytes') if guard else None,
            'maximum_system_swap_growth_bytes':guard.get('maximum_swap_growth_bytes') if guard else None,
            'guard_violation':guard.get('violation') if guard else None,'core_source_matches_campaign':source_match,
            'independent_reference_prefix_positions':len(prefix),'independent_prefix_equal':ids[:len(prefix)]==prefix if prefix else None,
            'all_generated_positions_independently_validated':False,'logical_bytes':run.get('logical_bytes'),
            'cpu_seconds':run.get('cpu_seconds'),
            'process_storage_read_bytes':run.get('process_storage_read_bytes'),'cache':run.get('metrics',{}).get('cache')})
    return result


def long_comparison(evidence):
    evidence=Path(evidence);paths={name:evidence/f'{name}.json' for name in ('long64-baseline','long128-baseline','long128-decay')}
    runs={name:read_json(path) for name,path in paths.items()}
    if not all(runs.values()) or not all(guarded(path)[0] for path in paths.values()):return {'status':'pending'}
    base=runs['long128-baseline'];decay=runs['long128-decay'];short=runs['long64-baseline']
    equal=base.get('input_ids')==decay.get('input_ids')==short.get('input_ids') and base['generated_ids']==decay['generated_ids']
    prefix=base['generated_ids'][:len(short['generated_ids'])]==short['generated_ids']
    cpu_base=base.get('cpu_seconds');cpu_decay=decay.get('cpu_seconds')
    return {'status':'passed' if equal and prefix else 'failed','baseline_128_tokens':len(base['generated_ids']),
        'decay_128_tokens':len(decay['generated_ids']),'baseline_64_tokens':len(short['generated_ids']),
        'all_128_ids_equal':equal,'prefix_64_equal':prefix,
        'baseline_generation_tokens_per_second':base['generation_tokens_per_second'],
        'decay_generation_tokens_per_second':decay['generation_tokens_per_second'],
        'generation_throughput_change_percent':100*(decay['generation_tokens_per_second']/base['generation_tokens_per_second']-1),
        'decode_throughput_change_percent':100*(decay['decode_tokens_per_second']/base['decode_tokens_per_second']-1),
        'logical_byte_reduction_percent':100*(1-decay['logical_bytes']/base['logical_bytes']),
        'cpu_seconds_change_percent':100*(cpu_decay/cpu_base-1) if positive(cpu_base) and cpu_decay is not None else None,
        'default_change_supported':False,'scope':'One 128-token run per setting; OS cache and thermal state uncontrolled. No causal universal speedup/regression claim.',
        'decision':'Retain existing default; lower logical IO alone is insufficient when observed generation slows.'}


def storage_control(evidence):
    path=Path(evidence)/'nocache8.json';run=read_json(path);ok,guard=guarded(path)
    if not run or not ok or run.get('status')!='completed':return {'status':'pending'}
    n=len(run.get('generated_ids',[]));elapsed=guard.get('elapsed_seconds');logical=run.get('logical_bytes');storage=run.get('process_storage_read_bytes')
    return {'status':'completed','report':display_path(path),'io_mode':run.get('arguments',{}).get('io_mode'),
        'generated_tokens':n,'generation_tokens_per_second':run.get('generation_tokens_per_second'),
        'supervised_process_tokens_per_second':ratio(n,elapsed) if positive(elapsed) else None,
        'logical_bytes':logical,'process_storage_read_bytes':storage,'cpu_seconds':run.get('cpu_seconds'),
        'storage_to_logical_ratio':ratio(storage,logical) if storage is not None and positive(logical) else None,
        'sampled_peak_rss_bytes':guard.get('sampled_peak_rss_bytes'),'maximum_system_swap_growth_bytes':guard.get('maximum_swap_growth_bytes'),
        'storage_counter':run.get('storage_counter'),'physical_ssd_bandwidth_certified':False,
        'scope':'Separate short F_NOCACHE descriptor-policy run. Not a matched cache-purge experiment; OS process counter is not NAND traffic.'}


def analyze(evidence):
    evidence=Path(evidence);manifest_path=evidence/'paired/campaign.json';campaign=read_json(manifest_path)
    if not campaign:raise ValueError('Missing or incomplete campaign.json')
    variants=campaign['variants'];cases=campaign['cases'];repeats=campaign['repeats']
    expected=len(cases)*len(variants)*repeats;listed=set(campaign.get('completed',[]));index={};pending=[];failed=[]
    hashes=defaultdict(set)
    for case in cases:
        for repeat in range(repeats):
            for variant,spec in variants.items():
                name=f'{case["name"]}-r{repeat}-{variant}.json';path=evidence/'paired'/name
                run=read_json(path);ok,guard=guarded(path)
                if (run and run.get('status')=='failed') or (guard and (guard.get('violation') or guard.get('exit_code') not in (None,0))):
                    failed.append({'file':name,'status':run.get('status') if run else None,'guard_violation':guard.get('violation') if guard else None});continue
                if name not in listed or not run or run.get('status')!='completed' or not ok:
                    pending.append(name);continue
                ids=run.get('generated_ids');times=run.get('token_seconds',[]);args=run.get('arguments',{})
                checks=(isinstance(ids,list) and bool(ids) and len(times)==len(ids) and all(positive(t) for t in times)
                    and positive(run.get('generation_seconds')) and positive(run.get('wall_seconds'))
                    and positive(guard.get('elapsed_seconds'))
                    and run.get('input_ids')==case['input_ids'] and args.get('kernel')==spec[0]
                    and args.get('rows')==spec[1] and args.get('cache_decay')==spec[2])
                if not checks:failed.append({'file':name,'reason':'invalid_or_mismatched_run_identity'});continue
                run['_guard']=guard;run['_file']=str(path);index[(case['name'],repeat,variant)]=run
                for file,digest in run.get('identity',{}).get('source_sha256',{}).items():hashes[file].add(digest)
    all_runs=list(index.values());complete=len(all_runs)==expected and not failed and not pending
    groups={};prefix_checks=[]
    for case in cases:
        name=case['name'];group=[r for (c,_,_),r in index.items() if c==name]
        sequences=sorted({tuple(r['generated_ids']) for r in group})
        groups[name]={'completed_runs':len(group),'expected_runs':len(variants)*repeats,
            'all_available_sequences_identical':len(sequences)<=1,
            'all_variants_and_repeats_identical':len(sequences)==1 if len(group)==len(variants)*repeats else None,
            'sequences':[list(s) for s in sequences]}
        reference=read_json(evidence/f'torch-{name}-4.json')
        ref_guard=guarded(evidence/f'torch-{name}-4.json')[0]
        for run in group:
            expected_prefix=eos_prefix(reference['generated_ids']) if reference else []
            match=bool(reference and ref_guard and reference.get('input_ids')==case['input_ids'] and run['generated_ids'][:len(expected_prefix)]==expected_prefix)
            prefix_checks.append({'run':Path(run['_file']).name,'case':name,'reference_available':bool(reference and ref_guard),
                'expected_prefix':expected_prefix,'compared_positions':len(expected_prefix),
                'status':'passed' if match else 'failed' if reference and ref_guard else 'pending'})
    validation={case['name']:compare_dump(evidence/f'validation-{case["name"]}-4.json',evidence/f'validation-{case["name"]}-4.npy',
        evidence/f'torch-{case["name"]}-4.json',evidence/f'torch-{case["name"]}-4-logits.npz') for case in cases}
    campaign_logits={Path(r['_file']).name:compare_dump(r['_file'],Path(r['_file']).with_suffix('.npy'),
        evidence/f'torch-{case}-4.json',evidence/f'torch-{case}-4-logits.npz') for (case,_,_),r in index.items()}
    legacy={name:compare_dump(evidence/f'first-complete-{name}.json',evidence/f'first-complete-{name}-logits.npy',
        evidence/'torch-real1.json',evidence/'torch-real1-logits.npz') for name in ('reference','metal')}
    validations_passed=all(v['status']=='passed' for v in validation.values())
    numeric_failure=any(c['status']=='failed' for c in prefix_checks) or any(v['status']=='failed' for v in [*validation.values(),*campaign_logits.values(),*legacy.values()])
    sources_consistent=all(len(v)==1 for v in hashes.values())
    status='failed' if failed or numeric_failure or not sources_consistent else 'complete' if complete and validations_passed else 'pending'
    long_evidence=long_runs(evidence,hashes)
    if status=='complete' and any(r['status']=='pending' for r in long_evidence):status='pending'
    long_identity=long_comparison(evidence)
    if long_identity['status']=='failed':status='failed'
    ablations={}
    for label,base,candidate in [('tiling_512_to_2048','metal512','metal2048'),('cache_decay_1024_to_32768','metal2048','metal-decay')]:
        if base in variants and candidate in variants:
            ablations[label]=paired_speedups(index,[base,candidate],baseline=base)[candidate]
    return {'status':status,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'campaign_manifest':display_path(manifest_path),
        'campaign_complete':complete,'expected_runs':expected,'completed_runs':len(all_runs),'pending_runs':pending,'failed_runs':failed,
        'protocol':{'seed':campaign.get('seed'),'repeats':repeats,'tokens_limit':campaign['tokens_limit'],'variants':variants,'cases':cases,
        'rate_definition':'sum(tokens)/sum(seconds)','decode_definition':'sum(tokens-1)/sum(token_seconds[1:])',
            'all_process_costs_definition':'sum(tokens)/sum(guard.elapsed_seconds); inputs are pretokenized IDs'},
        'variants':{name:aggregate([r for (_,_,v),r in index.items() if v==name]) for name in variants},
        'per_case':{case['name']:{v:aggregate([r for (c,_,n),r in index.items() if c==case['name'] and n==v]) for v in variants} for case in cases},
        'paired_speedups':paired_speedups(index,variants),'ablations':ablations,'token_identity':groups,'independent_prefix_checks':prefix_checks,
        'independent_logits_validation':validation,'campaign_final_logits':campaign_logits,'legacy_first_position_logits':legacy,
        'long_generation_runs':long_evidence,
        'long_runs_pending':[r['report'] for r in long_evidence if r['status']=='pending'],
        'long_exact_identity_and_cache_tradeoff':long_identity,'storage_policy_control':storage_control(evidence),
        'independent_validation_positions_limit':4,'full_eight_positions_independently_validated':False,
        'source_consistency':{'consistent':sources_consistent,'sha256':{f:sorted(v) for f,v in sorted(hashes.items())}},
        'caveats':CAVEATS,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def markdown(report):
    def number(value,digits=3):return f'{value:.{digits}f}' if value is not None else 'n/d'
    provisional=report['status']!='complete'
    lines=['# Mesures GGUF natives — analyse appariée',
        '',f'**État : {"PROVISOIRE / EN ATTENTE" if provisional else "CAMPAGNE ET VALIDATIONS TERMINÉES"}** — {report["completed_runs"]}/{report["expected_runs"]} essais terminés et gardés.',
        '',f'Calcul régénéré depuis `{report["campaign_manifest"]}`. Aucun modèle ni poids n’est chargé par ce script.',
        '',f'Protocole : {len(report["protocol"]["cases"])} prompts, {report["protocol"]["repeats"]} répétitions, ordre randomisé (seed {report["protocol"]["seed"]}), huit tokens maximum avec arrêt EOS.',
        '', '| Variante | Essais | Tokens | Génération tok/s | Décode tok/s | Mur worker tok/s | Processus supervisé tok/s | TTFT médian s | RSS pic Mio | Swap croissance Mio | Hits cache |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in report['variants'].items():
        if not v['runs']:lines.append(f'| {name} | 0 | — | — | — | — | — | — | — | — | — |');continue
        lines.append(f'| {name} | {v["runs"]} | {v["generated_tokens"]} | {number(v["pooled_generation_tokens_per_second"])} | {number(v["pooled_decode_tokens_per_second"])} | {number(v["pooled_worker_wall_tokens_per_second"])} | {number(v["pooled_supervised_process_tokens_per_second"])} | {number(v["median_ttft_seconds"])} | {number(v["sampled_peak_rss_bytes"]/2**20,1)} | {number(v["maximum_system_swap_growth_bytes"]/2**20,3)} | {number(100*v["pooled_cache_hit_rate"],2) if v["pooled_cache_hit_rate"] is not None else "n/d"}% |')
    lines+=['','Les débits sont des rapports de sommes. La colonne décode exclut le premier token de chaque essai; génération inclut préfill et boucle modèle. Le mur worker ajoute son initialisation. Le **processus supervisé**, `somme(tokens) / somme(guard.elapsed_seconds)`, inclut imports Python, initialisation et nettoyage. Les IDs d’entrée sont pré-fournis : tokenisation et rendu du chat restent un essai séparé. Aucun débit génération n’est appelé ici débit complet tous coûts. La variante `reference` est le chemin AirLLM déquantification CPU + calcul MLX; Torch est un comparateur numérique indépendant, pas cette base de vitesse.','',
        '| Variante | Poids logiques Go/token | Stockage processus Go/token | CPU s/token | Cache pic Mio |',
        '|---|---:|---:|---:|---:|']
    for name,v in report['variants'].items():
        if not v['runs']:continue
        storage=ratio(v['process_storage_read_bytes'],v['generated_tokens']*1e9) if v['process_storage_read_bytes'] is not None else None
        lines.append(f'| {name} | {number(v["logical_bytes_per_generated_token"]/1e9)} | {number(storage)} | {number(v["cpu_seconds_per_generated_token"])} | {number(v["cache_peak_bytes"]/2**20,1)} |')
    lines+=['','Go = 10⁹ octets, Mio = 2²⁰ octets. Le stockage processus est un compteur macOS, pas une mesure NAND. Le temps CPU ne mesure pas les commandes GPU. Les métriques par prompt et répétition restent disponibles dans le JSON.','',
        '## Comparaisons appariées','', '| Variante / référence | Paires valides | Gain génération groupé | Gain décode groupé | Gain processus supervisé | Gain génération géométrique |',
        '|---|---:|---:|---:|---:|---:|']
    for name,p in report['paired_speedups'].items():
        lines.append(f'| {name} / {p["baseline"]} | {p["eligible_pairs"]} | {number(p["pooled_generation_speedup"])}× | {number(p["pooled_decode_speedup"])}× | {number(p["pooled_supervised_process_speedup"])}× | {number(p["geometric_mean_generation_speedup"])}× |')
    lines+=['','Les paires associent le même prompt et la même répétition, avec budgets, sources et IDs identiques. Des résultats partiels ne permettent pas de conclure au meilleur réglage. Le gain contre `reference` regroupe kernel, transport et représentation du cache.','',
        '| Ablation Metal | Paires valides | Gain génération groupé | Gain décode groupé | Gain processus supervisé |',
        '|---|---:|---:|---:|---:|']
    for name,p in report.get('ablations',{}).items():
        lines.append(f'| {name} | {p["eligible_pairs"]} | {number(p["pooled_generation_speedup"])}× | {number(p["pooled_decode_speedup"])}× | {number(p["pooled_supervised_process_speedup"])}× |')
    lines+=['','La première ablation change la taille des tuiles compactées; la seconde change seulement la demi-vie du cache adaptatif de 1 024 à 32 768 demandes, à tuiles et budget égaux. Un ratio inférieur à 1 indique un ralentissement.','',
        '## Exactitude et portée','', '| Prompt | Séries natives présentes / attendues | IDs identiques entre séries | Logits finaux ≤4 positions contre Torch | Écart absolu maximal |',
        '|---|---:|---|---|---:|']
    for case,g in report['token_identity'].items():
        v=report['independent_logits_validation'][case];same=g['all_variants_and_repeats_identical']
        lines.append(f'| {case} | {g["completed_runs"]}/{g["expected_runs"]} | {"oui" if same is True else "NON" if same is False else "en attente"} | {v["status"]} | {number(v.get("maximum_absolute_error"),8)} |')
    checks=report['independent_prefix_checks'];passed=sum(c['status']=='passed' for c in checks)
    lines+=[ '',f'{passed}/{len(checks)} préfixes disponibles correspondent à la référence Torch indépendante jusqu’à quatre positions, arrêt EOS inclus. Les NPY des séries de huit pas ne contiennent que le dernier vecteur : ils ne sont comparés à Torch que si sa position est couverte. Les positions cinq à huit ne reçoivent aucune validation indépendante ici.', '',
        f'Empreintes sources cohérentes entre essais présents : {"oui" if report["source_consistency"]["consistent"] else "NON"}. Les empreintes exactes et comparaisons du premier token historique figurent dans le JSON.', '', '## Limites et sources', '']
    long=report.get('long_generation_runs',[])
    if long:
        # Insert the separate stability table before caveats, preserving the
        # paired cohort and independent-reference limits above.
        lines=lines[:-2]+['## Générations prolongées — stabilité séparée','',
            '| Rapport | État | Tokens obtenus / demandés | Génération tok/s | Décode tok/s | Mur worker tok/s | Processus supervisé tok/s | RSS pic Mio | Swap croissance Mio | Même source que campagne |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in long:
            lines.append(f'| {Path(r["report"]).name} | {r["status"]} | {r["generated_tokens"]}/{r["requested_tokens"]} | {number(r["generation_tokens_per_second"])} | {number(r["decode_tokens_per_second"])} | {number(r["worker_wall_tokens_per_second"])} | {number(r["supervised_process_tokens_per_second"])} | {number(r["sampled_peak_rss_bytes"]/2**20,1) if r["sampled_peak_rss_bytes"] is not None else "n/d"} | {number(r["maximum_system_swap_growth_bytes"]/2**20) if r["maximum_system_swap_growth_bytes"] is not None else "n/d"} | {"oui" if r["core_source_matches_campaign"] else "non, historique"} |')
        lines+=['','Ces essais mesurent durée, débit et sûreté, sans validation indépendante de toutes les positions générées. Les historiques de source différente ne contribuent pas aux gains appariés; un préfixe validé ne certifie pas les 64 ou 128 tokens suivants.','']
        comparison=report.get('long_exact_identity_and_cache_tradeoff',{})
        if comparison.get('status')!='pending' and comparison:
            lines+=[f'Les 128 IDs baseline/decay sont {"identiques" if comparison["all_128_ids_equal"] else "DIFFÉRENTS"}; le préfixe de 64 IDs est {"identique" if comparison["prefix_64_equal"] else "DIFFÉRENT"}. À 128 tokens, decay réduit les lectures logiques de {number(comparison["logical_byte_reduction_percent"],2)} %, mais son débit génération varie de {number(comparison["generation_throughput_change_percent"],2)} % et son temps CPU de {number(comparison["cpu_seconds_change_percent"],2)} %. Résultat négatif conservé : aucun changement du défaut n’est justifié. Une seule répétition par réglage ne démontre pas une cause universelle.','']
        lines+=['## Limites et sources','']
    control=report.get('storage_policy_control',{})
    if control.get('status')=='completed':
        lines=lines[:-2]+['## Contrôle F_NOCACHE séparé','',
            f'{control["generated_tokens"]} tokens, {number(control["generation_tokens_per_second"])} tok/s dans la boucle, {number(control["supervised_process_tokens_per_second"])} tok/s processus supervisé. {number(control["logical_bytes"]/1e9)} Go logiques, {number(control["process_storage_read_bytes"]/1e9) if control["process_storage_read_bytes"] is not None else "n/d"} Go de stockage comptabilisés par macOS. Pic RSS {number(control["sampled_peak_rss_bytes"]/2**20,1)} Mio, croissance maximale du swap {number(control["maximum_system_swap_growth_bytes"]/2**20)} Mio.', '',
            'F_NOCACHE est une politique du descripteur. Cet essai court, non apparié à un cache purgé, ne certifie ni trafic NAND ni débit SSD physique; le décalage entre octets logiques et compteur OS doit rester visible.','','## Limites et sources','']
    lines+=['- '+c for c in report['caveats']]
    lines+=['','Sources de format et de calcul : [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [GGML C](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c), [Transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py), [Metal MLX](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). Les versions installées et SHA256 font foi pour les expériences. Antériorités détaillées dans `PRIMARY_SOURCES.md`; ces liens ne remplacent pas les mesures locales.','']
    if report['failed_runs']:lines+=['**Échecs ou identités invalides :** '+json.dumps(report['failed_runs'],ensure_ascii=False),'']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,default=ROOT/'evidence-native-gguf')
    p.add_argument('--output',type=Path,default=ROOT/'research/native_gguf/analysis.json')
    p.add_argument('--markdown',type=Path,default=ROOT/'research/native_gguf/ANALYSIS_FR.md')
    args=p.parse_args(argv)
    # Analysis is read-only with respect to all experimental evidence.
    targets=[args.output,args.markdown]
    if targets[0].resolve()==targets[1].resolve():raise ValueError('JSON and Markdown destinations must differ')
    for target in targets:
        if target.resolve().is_relative_to(args.evidence.resolve()):raise ValueError('Analysis output cannot overwrite evidence')
        if target.exists() and any(target.samefile(source) for source in args.evidence.rglob('*') if source.is_file()):
            raise ValueError('Analysis output aliases protected evidence')
    report=analyze(args.evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.markdown.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    args.markdown.write_text(markdown(report))
    print(json.dumps({'status':report['status'],'completed_runs':report['completed_runs'],'expected_runs':report['expected_runs'],
                      'output':str(args.output),'markdown':str(args.markdown)}))


if __name__=='__main__':main()
