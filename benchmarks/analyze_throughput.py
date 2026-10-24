#!/usr/bin/env python3
"""Recompute paired performance and exactness from primary JSON/guard evidence."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads((a.campaign/'plan.json').read_text());pairs=[];flat=[]
    for spec in plan['pairs']:
        pair={'id':spec['id'],'repeat':spec['repeat'],'order':spec['order']};raw={}
        for variant in ['reference','candidate']:
            path=a.campaign/f'{spec["id"]}-r{spec["repeat"]}-{variant}.json'
            d=json.loads(path.read_text());g=json.loads(path.with_suffix('.guard.json').read_text());t=d['turns'][0]
            if d['status']!='completed' or t['stop_reason']!='eos' or g['exit_code']!=0 or g['violation'] or g.get('cleanup_error'):
                raise ValueError(f'Incomplete/unsafe primary observation: {path}')
            if d['source_sha256']!=plan['sources']:raise ValueError('Primary worker source fingerprint mismatch')
            n=len(t['generated_ids']);steps=t['step_checks'];decode=steps[1:]
            row={'id':spec['id'],'repeat':spec['repeat'],'variant':variant,'input_tokens':len(t['full_input_ids']),
                'tokens':n,'stop':t['stop_reason'],'ttft_seconds':t['ttft_seconds'],'decode_tok_s':t['decode_tokens_per_second'],
                'generation_seconds':t['total_seconds'],'generation_tok_s':n/t['total_seconds'],
                'process_seconds':g['elapsed_seconds'],'e2e_tok_s':n/g['elapsed_seconds'],'cpu_seconds':t['cpu_seconds'],
                'logical_read_bytes':t['logical_read_bytes'],'pread_calls':t['pread_calls'],'storage_read_bytes':t['process_storage_read_bytes'],
                'decode_logical_bytes':sum(s['logical_read_bytes'] for s in decode),'decode_storage_bytes':sum(s['storage_read_bytes'] or 0 for s in decode),
                'peak_rss_bytes':g['sampled_peak_rss_bytes'],'minimum_available_bytes':g['minimum_available_bytes'],
                'swap_initial_bytes':g['before']['swap_used_bytes'],'swap_final_bytes':g['after']['swap_used_bytes'],
                'maximum_swap_growth_bytes':g['maximum_swap_growth_bytes'],'peak_pressure_level':g['peak_pressure_level']}
            pair[variant]=row;raw[variant]=t;flat.append(row)
        r,c=raw['reference'],raw['candidate']
        pair['tokens_equal']=r['generated_ids']==c['generated_ids'] and r['full_input_ids']==c['full_input_ids']
        pair['logits_bitwise_equal']=([s['logits_sha256'] for s in r['step_checks']]==[s['logits_sha256'] for s in c['step_checks']])
        pair['routes_equal']=([s['routes_sha256'] for s in r['step_checks']]==[s['routes_sha256'] for s in c['step_checks']])
        pair['positions_equal']=([s['position'] for s in r['step_checks']]==[s['position'] for s in c['step_checks']])
        pair['decode_speedup']=pair['candidate']['decode_tok_s']/pair['reference']['decode_tok_s']
        pair['e2e_speedup']=pair['candidate']['e2e_tok_s']/pair['reference']['e2e_tok_s']
        pair['ttft_speedup']=pair['reference']['ttft_seconds']/pair['candidate']['ttft_seconds']
        if not all(pair[k] for k in ['tokens_equal','logits_bitwise_equal','routes_equal','positions_equal']):
            raise ValueError(f'Fidelity failed for {spec["id"]} repetition {spec["repeat"]}')
        pairs.append(pair)
    aggregate={}
    for variant in ['reference','candidate']:
        rows=[r for r in flat if r['variant']==variant]
        decode_tokens=sum(r['tokens']-1 for r in rows)
        decode_time=sum((r['tokens']-1)/r['decode_tok_s'] for r in rows)
        all_tokens=sum(r['tokens'] for r in rows)
        aggregate[variant]={'decode_tok_s':decode_tokens/decode_time,'e2e_tok_s':all_tokens/sum(r['process_seconds'] for r in rows),
            'generation_tok_s':all_tokens/sum(r['generation_seconds'] for r in rows),'tokens':all_tokens,
            'ttft_median_seconds':statistics.median(r['ttft_seconds'] for r in rows),
            'process_seconds':sum(r['process_seconds'] for r in rows),'peak_rss_bytes':max(r['peak_rss_bytes'] for r in rows),
            'minimum_available_bytes':min(r['minimum_available_bytes'] for r in rows),'max_swap_growth_bytes':max(r['maximum_swap_growth_bytes'] for r in rows),
            'decode_logical_bytes_per_token':sum(r['decode_logical_bytes'] for r in rows)/decode_tokens,
            'decode_storage_bytes_per_token':sum(r['decode_storage_bytes'] for r in rows)/decode_tokens}
    ratios=[r['decode_speedup'] for r in pairs];rng=random.Random(9)
    boots=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
    result={'kind':'shuffled_counterbalanced_complete_generation','pairs':pairs,'aggregate':aggregate,
        'median_pair_decode_speedup':statistics.median(ratios),'paired_wins':sum(r>1 for r in ratios),'pair_count':len(pairs),
        'exploratory_pair_median_bootstrap_95':[boots[249],boots[9749]],
        'bootstrap_note':'Six pairs across three prompts, two repeats; descriptive paired bootstrap, not a general population guarantee',
        'all_tokens_logits_routes_positions_exact':True,'captured_state_scope':'Separate eight-step binary/state-hash captures; not every primary position',
        'conditional_logical_bandwidth_requirements_bytes_per_second':{str(rate):aggregate['candidate']['decode_logical_bytes_per_token']*rate for rate in [1.5,2,4,8]}}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (a.output/'benchmarks.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs',)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
