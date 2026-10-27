#!/usr/bin/env python3
"""Four recorded workers isolating gate/up overlap; no model or engine import."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_hyperstream import controls
import analyze_research51 as shared

EXPECTED_HEAD='6960ef97ac4aa71d191087320362c1d636562aed'
ORDER=['explanation.r0.off','explanation.r0.on','explanation.r1.on','explanation.r1.off']


def analyze_campaign(folder,expected_head=EXPECTED_HEAD):
    shared.require(isinstance(expected_head,str) and len(expected_head)==40 and
                   all(c in '0123456789abcdef' for c in expected_head),'Expected engine head must be a full SHA-1')
    folder=Path(folder).resolve();inputs=[];campaign=shared.load_json(folder/'campaign.json',inputs)
    shared.require(campaign.get('status')=='completed' and campaign.get('protocol_unchanged') is True
                   and campaign.get('engines_unchanged') is True,'Incomplete or changed ablation campaign')
    manifests=campaign.get('engine_manifests');protocol=campaign.get('protocol_sha256')
    shared.require(isinstance(manifests,dict) and len(manifests)==1 and campaign.get('engine_manifests_after')==manifests,
                   'Ablation requires one fixed checkout with unchanged manifests')
    root,manifest=next(iter(manifests.items()))
    shared.require(manifest.get('git_head')==expected_head,'Unexpected ablation engine revision')
    shared.require(isinstance(protocol,dict) and protocol and all(isinstance(v,str) and shared.SHA256.fullmatch(v)
                   for v in protocol.values()),'Missing frozen protocol hashes')
    plan=campaign.get('plan');runs=campaign.get('runs')
    shared.require(isinstance(plan,list) and isinstance(runs,list) and [j.get('id') for j in plan]==ORDER
                   and [r.get('id') for r in runs]==ORDER,'Exactly four workers in off/on/on/off order required')
    rows=[];raw={};groups={0:{},1:{}};invariant=limits=identity=None
    for job,run in zip(plan,runs):
        ident=job['id'];_,repeat,variant=ident.split('.');repeat=int(repeat[1:])
        shared.require(job.get('engine_root')==root and run.get('status')=='completed'
                       and type(run.get('exit_code')) is int and run['exit_code']==0,'Unsuccessful or relocated worker')
        recorded=Path(shared.option_value(run['command'],'--output'))
        shared.require(recorded.name==ident+'.json' and Path(run.get('guard_file',''))==recorded.with_suffix('.guard.json'),
                       'Recorded worker or guard output differs')
        worker=shared.load_json(folder/(ident+'.json'),inputs)
        guard=shared.load_json(folder/(ident+'.guard.json'),inputs)
        shared.require(isinstance(worker.get('options'),dict),'Missing model options')
        current_limits=shared.validate_guard(guard,run,worker)
        if limits is None:limits=current_limits
        shared.require(current_limits==limits,'Unequal ablation resource guards')
        turn=shared.validate_worker(worker,guard,job,run,campaign)
        shared.require(turn['id']=='explanation','Ablation task must be explanation')
        current_identity={k:worker.get(k) for k in ('model','hardware','versions','cache_regime')}
        shared.require(isinstance(current_identity['model'],dict) and current_identity['model'].get('bytes',0)>0
                       and worker.get('physical_ssd_bandwidth_certified') is False,'Invalid model or storage identity')
        if identity is None:identity=current_identity
        shared.require(current_identity==identity,'Model, hardware, dependencies or cache regime changed')
        common_args,treatment=controls(job['args']);backend=worker.get('backend_options')
        shared.require(treatment==dict(expert_pipeline=variant=='on',string_buffer_bytes=65536)
                       and isinstance(backend,dict) and backend.get('expert_pipeline',False)==(variant=='on')
                       and backend.get('string_buffer_bytes')==65536,'Unexpected ablation treatment')
        current=(worker['options'],{k:v for k,v in backend.items() if k!='expert_pipeline'},common_args)
        if invariant is None:invariant=current
        shared.require(current==invariant,'Only expert_pipeline may differ within the ablation')
        exercised=turn['metrics']['weights'].get('expert_pipeline_pairs',0)
        shared.require(type(exercised) in (int,float) and (exercised>0 if variant=='on' else exercised==0),
                       'Requested pipeline treatment was not exercised')
        row=shared.flat_row(worker,guard,turn,'explanation',repeat,variant)
        rows.append(row);raw[ident]=(worker,turn);groups[repeat][variant]=row
    pairs=[]
    for repeat,group in groups.items():
        shared.require(set(group)=={'off','on'},'Missing paired treatment')
        off,on=group['off'],group['on'];ot=raw[off['label']][1];nt=raw[on['label']][1]
        for key in ('prompt','full_input_ids','appended_input_ids','generated_ids','full_output_ids','eos_token_ids',
                    'generation_parameters','native_cache_position','text','raw_text','logical_read_bytes'):
            shared.require(ot[key]==nt[key],f'Ablation pair differs in {key}')
        for key in ('logits_sha256','routes_sha256','position','logical_bytes'):
            shared.require([s[key] for s in ot['step_checks']]==[s[key] for s in nt['step_checks']],f'Per-step drift in {key}')
        for key in ('hits','misses','bytes','capacity'):
            shared.count(ot['metrics']['cache'].get(key),'cache '+key)
            shared.require(ot['metrics']['cache'][key]==nt['metrics']['cache'].get(key),'Unequal cache '+key)
        pairs.append(dict(repeat=repeat,off=off,on=on,decode_speedup=on['decode_tok_s']/off['decode_tok_s'],
                          end_to_end_speedup=on['end_to_end_tok_s']/off['end_to_end_tok_s'],
                          ttft_speedup=off['ttft_seconds']/on['ttft_seconds']))
    aggregate={v:shared.aggregate(r for r in rows if r['variant']==v) for v in ('off','on')}
    paths=[Path(__file__),Path(shared.__file__),Path(__file__).with_name('analyze_hyperstream.py')]
    summary=dict(kind='hyperstream_pipeline_ablation',fixture=campaign.get('fixture',False),pair_count=2,worker_count=4,
        prompt_count=1,repeats_per_prompt=2,all_tokens_logits_routes_positions_exact=True,all_complete_responses_exact=True,
        cache_counts_and_logical_reads_exact=True,engine_manifests=manifests,protocol_sha256=protocol,inputs=inputs,
        analysis_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},guard_limits=limits,
        environment_and_model_stat_identity=identity,expected_engine_head=expected_head,aggregate=aggregate,pairs=pairs,
        aggregate_decode_speedup=aggregate['on']['decode_tok_s']/aggregate['off']['decode_tok_s'],
        aggregate_end_to_end_speedup=aggregate['on']['end_to_end_tok_s']/aggregate['off']['end_to_end_tok_s'],
        paired_decode_speedup=shared.describe(p['decode_speedup'] for p in pairs),
        paired_end_to_end_speedup=shared.describe(p['end_to_end_speedup'] for p in pairs),
        limitations=['Two descriptive pairs on one prompt; a small attribution experiment, not general speed certification.',
                    'Same engine sources and string buffer 65536; only expert_pipeline off/on differs.',
                    'Fresh workers and uncontrolled OS page cache; no cold SSD, external SSD or sustained thermal claim.',
                    'Per-token logits/routes hashes and causal positions checked; full KV payload hashes were not captured.'])
    return summary,rows,raw


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('campaign',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-head',default=EXPECTED_HEAD,help='Exact recorded engine SHA; local validation revision by default')
    args=parser.parse_args()
    summary,rows,raw=analyze_campaign(args.campaign,args.expected_head);shared.write_outputs(args.output,summary,rows,raw)
    response=args.output/'responses.md'
    response.write_text(response.read_text().replace('campagne 5.1','ablation HyperStream gate/up',1),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('pair_count','aggregate_decode_speedup','aggregate_end_to_end_speedup')}))


if __name__=='__main__':main()
