#!/usr/bin/env python3
"""Validate exactly eight guarded workers: two prompts, two paired repetitions.

Standard library only. Reads recorded evidence, never models or live engines.
"""
import argparse
import hashlib
import json
from pathlib import Path

import analyze_research51 as shared


def controls(arguments):
    """Separate the only permitted treatment switches from invariant controls."""
    common=[];pipeline=False;buffer=0;seen=set();index=0
    while index<len(arguments):
        argument=arguments[index];index+=1
        if argument not in ('--expert-pipeline','--string-buffer-bytes'):
            common.append(argument);continue
        shared.require(argument not in seen,'Duplicate treatment switch');seen.add(argument)
        if argument=='--expert-pipeline':pipeline=True
        else:
            shared.require(index<len(arguments),'Missing string buffer value')
            buffer=int(arguments[index]);index+=1
            shared.require(0<=buffer<=65536,'Invalid string buffer value')
    return common,{'expert_pipeline':pipeline,'string_buffer_bytes':buffer}


def analyze_campaign(folder):
    folder=Path(folder).resolve();inputs=[];campaign=shared.load_json(folder/'campaign.json',inputs)
    shared.require(campaign.get('status')=='completed','Only a completed campaign can be analyzed')
    shared.require(campaign.get('protocol_unchanged') is True and campaign.get('engines_unchanged') is True,
                   'Campaign identities changed')
    manifests=campaign.get('engine_manifests');protocol=campaign.get('protocol_sha256')
    shared.require(isinstance(manifests,dict) and manifests and campaign.get('engine_manifests_after')==manifests,
                   'Engine before/after manifests differ')
    shared.require(isinstance(protocol,dict) and protocol and all(isinstance(v,str) and shared.SHA256.fullmatch(v)
                   for v in protocol.values()),'Missing protocol hashes')
    plan=campaign.get('plan');runs=campaign.get('runs')
    shared.require(isinstance(plan,list) and isinstance(runs,list) and len(plan)==len(runs)==8,
                   'Exactly eight workers are required')
    shared.require([j.get('id') for j in plan]==[r.get('id') for r in runs] and len({j['id'] for j in plan})==8,
                   'Missing, duplicated or reordered workers')
    rows=[];raw={};groups={};role_roots={'baseline':set(),'candidate':set()};common=limits=None;invariants={}
    for job,run in zip(plan,runs):
        ident=job['id'];match=shared.LABEL.fullmatch(ident)
        shared.require(match is not None and Path(ident).name==ident,'Invalid job identity')
        task,repeat,variant=match['task'],int(match['repeat']),match['variant']
        shared.require(repeat in (0,1) and run.get('status')=='completed' and type(run.get('exit_code')) is int
                       and run['exit_code']==0,'Incomplete run or invalid repetition')
        recorded=Path(shared.option_value(run['command'],'--output'))
        shared.require(recorded.name==ident+'.json' and Path(run.get('guard_file',''))==recorded.with_suffix('.guard.json'),
                       'Recorded output/guard identities differ')
        worker=shared.load_json(folder/(ident+'.json'),inputs)
        guard=shared.load_json(folder/(ident+'.guard.json'),inputs)
        shared.require(isinstance(worker.get('options'),dict),'Missing worker options')
        current_limits=shared.validate_guard(guard,run,worker)
        if limits is None:limits=current_limits
        shared.require(current_limits==limits,'Unequal resource guards')
        turn=shared.validate_worker(worker,guard,job,run,campaign)
        shared.require(turn['id']==task,'Job task label differs')
        identity={key:worker.get(key) for key in ('model','hardware','versions','cache_regime')}
        shared.require(isinstance(identity['model'],dict) and identity['model'].get('bytes',0)>0
                       and worker.get('physical_ssd_bandwidth_certified') is False,'Invalid model/storage identity')
        if common is None:common=identity
        shared.require(identity==common,'Unequal model, machine, versions or cache regime')
        invariant_args,treatment=controls(job['args']);backend=worker.get('backend_options')
        shared.require(isinstance(backend,dict),'Missing backend options')
        shared.require(all(backend.get(k,False if k=='expert_pipeline' else 0)==v for k,v in treatment.items()),
                       'Treatment controls differ from recorded backend')
        invariant=(worker['options'],{k:v for k,v in backend.items() if k not in treatment},invariant_args)
        if 'common' not in invariants:invariants['common']=invariant
        shared.require(invariant==invariants['common'],'Uncontrolled options differ')
        if variant not in invariants:invariants[variant]=treatment
        shared.require(invariants[variant]==treatment,'Treatment changed within a variant')
        row=shared.flat_row(worker,guard,turn,task,repeat,variant);rows.append(row);raw[ident]=(worker,turn)
        group=groups.setdefault((task,repeat),{});shared.require(variant not in group,'Duplicate paired variant')
        group[variant]=row;role_roots[variant].add(job['engine_root'])
    shared.require(len(groups)==4 and len({task for task,_ in groups})==2 and
                   all(set(group)=={'baseline','candidate'} for group in groups.values()),'Four exact pairs across two prompts required')
    shared.require(all(len(roots)==1 for roots in role_roots.values()) and role_roots['baseline']!=role_roots['candidate'],
                   'Variants must use distinct fixed engine checkouts')
    shared.require(invariants['baseline']==dict(expert_pipeline=False,string_buffer_bytes=0) and
                   invariants['candidate']==dict(expert_pipeline=True,string_buffer_bytes=65536),'Unexpected paired treatment')
    pairs=[]
    for (task,repeat),group in sorted(groups.items()):
        baseline,candidate=group['baseline'],group['candidate'];bt=raw[baseline['label']][1];ct=raw[candidate['label']][1]
        for key in ('prompt','full_input_ids','appended_input_ids','generated_ids','full_output_ids','eos_token_ids',
                    'generation_parameters','native_cache_position','text','raw_text'):
            shared.require(bt[key]==ct[key],f'Pair {task}.r{repeat} differs in {key}')
        for key in ('logits_sha256','routes_sha256','position'):
            shared.require([s[key] for s in bt['step_checks']]==[s[key] for s in ct['step_checks']],f'Pair drift in {key}')
        pairs.append(dict(task=task,repeat=repeat,baseline=baseline,candidate=candidate,
            decode_speedup=candidate['decode_tok_s']/baseline['decode_tok_s'],
            end_to_end_speedup=candidate['end_to_end_tok_s']/baseline['end_to_end_tok_s'],
            ttft_speedup=baseline['ttft_seconds']/candidate['ttft_seconds']))
    aggregate={variant:shared.aggregate(r for r in rows if r['variant']==variant) for variant in role_roots}
    for variant,values in aggregate.items():
        for field in ('initialization_seconds','tokenizer_seconds','tokenizer_read_calls'):
            values[field]=shared.describe(r[field] for r in rows if r['variant']==variant)
    sources=[Path(__file__),Path(shared.__file__)]
    summary=dict(kind='hyperstream_complete_guarded_paired_generation',fixture=campaign.get('fixture',False),
        pair_count=4,worker_count=8,prompt_count=2,repeats_per_prompt=2,all_tokens_logits_routes_positions_exact=True,
        all_complete_responses_exact=True,engine_manifests=manifests,protocol_sha256=protocol,inputs=inputs,
        analysis_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        environment_and_model_stat_identity=common,guard_limits=limits,treatments={v:invariants[v] for v in role_roots},
        aggregate=aggregate,pairs=pairs,aggregate_decode_speedup=aggregate['candidate']['decode_tok_s']/aggregate['baseline']['decode_tok_s'],
        aggregate_end_to_end_speedup=aggregate['candidate']['end_to_end_tok_s']/aggregate['baseline']['end_to_end_tok_s'],
        paired_decode_speedup=shared.describe(p['decode_speedup'] for p in pairs),
        paired_end_to_end_speedup=shared.describe(p['end_to_end_speedup'] for p in pairs),
        limitations=['Four descriptive pairs; no general quality or sustained thermal certification.',
                    'Fresh workers; OS page cache uncontrolled. No certified cold SSD or external SSD bandwidth.',
                    'Initialization includes tokenizer time. Runtime TTFT excludes process imports and initialization.',
                    'Process storage counters are OS-accounted, not NAND bytes. RSS/MLX memory overlap.',
                    'Model stat identity is checked; this analyzer never reads or hashes model weights.'])
    return summary,rows,raw


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('campaign',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    summary,rows,raw=analyze_campaign(args.campaign)
    shared.write_outputs(args.output,summary,rows,raw)
    responses=args.output/'responses.md'
    responses.write_text(responses.read_text().replace('campagne 5.1','campagne HyperStream',1),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('pair_count','aggregate_decode_speedup','aggregate_end_to_end_speedup')}))


if __name__=='__main__':main()
