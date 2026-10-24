#!/usr/bin/env python3
"""Predeclared shuffled paired Qwen generation, guarded workers, no retries."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sources():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [*sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py')),ROOT/'benchmarks/throughput_gguf.py']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--simd-groups',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--seed',type=int,default=90241024)
    a=p.parse_args()
    if a.repeats<2:p.error('At least two paired repetitions required')
    a.output.mkdir(parents=True,exist_ok=False)
    tasks=json.loads(a.tasks.read_text());rng=random.Random(a.seed)
    pairs=[{'id':task['id'],'repeat':rep,'task':task} for task in tasks for rep in range(a.repeats)]
    rng.shuffle(pairs)
    frozen=sources();plan={'seed':a.seed,'repeats':a.repeats,'candidate':{'variant':'combined','rows':8192,'cache_policy':'lru','simd_groups':a.simd_groups},
        'reference':{'variant':'baseline','rows':2048,'cache_policy':'adaptive','simd_groups':1},
        'sources':frozen,'scope':'complete fresh-process generation until EOS or configured cap; fingerprint overhead identical; no OS cache purge',
        'pairs':[]}
    first_orders={task['id']:rng.choice([0,1]) for task in tasks}
    for pair in pairs:
        order=['reference','candidate']
        if (pair['repeat']+first_orders[pair['id']])%2:order.reverse()
        pair['order']=order;plan['pairs'].append(pair)
    (a.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    for pair in plan['pairs']:
        task_file=a.output/f'{pair["id"]}-tasks.json'
        if not task_file.exists():task_file.write_text(json.dumps([pair['task']],ensure_ascii=False,indent=2)+'\n')
        for variant in pair['order']:
            if sources()!=frozen:raise RuntimeError('Source changed during campaign; no further worker launched')
            cfg=plan[variant]
            output=a.output/f'{pair["id"]}-r{pair["repeat"]}-{variant}.json'
            cmd=[sys.executable,str(ROOT/'benchmarks/throughput_gguf.py'),'--model',str(a.model),'--tasks',str(task_file),
                '--output',str(output),'--variant',cfg['variant'],'--rows',str(cfg['rows']),
                '--cache-policy',cfg['cache_policy'],'--simd-groups',str(cfg['simd_groups'])]
            with (a.output/'commands.jsonl').open('a') as f:f.write(json.dumps(cmd)+'\n')
            process=subprocess.Popen(cmd)
            try:code=process.wait()
            except BaseException:
                # Let the supervisor run its finally block and stop only its
                # owned session. No automatic retries or personal PID filters.
                import signal
                if process.poll() is None:process.send_signal(signal.SIGINT)
                process.wait(timeout=10)
                raise
            if code:raise RuntimeError(f'Worker failed/guard refused ({code}); campaign stops without retry: {output}')
            d=json.loads(output.read_text());g=json.loads(output.with_suffix('.guard.json').read_text());t=d['turns'][0]
            print(json.dumps({'id':pair['id'],'repeat':pair['repeat'],'variant':variant,'stop':t['stop_reason'],
                'tokens':len(t['generated_ids']),'decode_tok_s':t['decode_tokens_per_second'],'ttft_seconds':t['ttft_seconds'],
                'swap_growth':g['maximum_swap_growth_bytes'],'violation':g['violation']},ensure_ascii=False),flush=True)
            if g['violation'] or t['stop_reason']!='eos':raise RuntimeError('Guard violation or incomplete response; campaign stops without retry')


if __name__=='__main__':main()
