#!/usr/bin/env python3
"""Two mirrored blocks of exact complete short generations, remove one knob."""
import argparse,json,random,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[{'name':'full','variant':'combined','rows':8192,'cache':'lru'},
        {'name':'no_expert_batch','variant':'concat','rows':8192,'cache':'lru'},
        {'name':'no_concat','variant':'expert-batch','rows':8192,'cache':'lru'},
        {'name':'adaptive_cache','variant':'combined','rows':8192,'cache':'adaptive'},
        {'name':'narrow_tiles','variant':'combined','rows':2048,'cache':'lru'}]
    random.Random(90249).shuffle(cases)
    plan={'scope':'Remove-one ablations on one short EOS task; two mirrored blocks; not independent additive contributions',
        'seed':90249,'blocks':[cases,list(reversed(cases))]}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for block,order in enumerate(plan['blocks']):
        for case in order:
            output=a.output/f'b{block}-{case["name"]}.json'
            command=[sys.executable,str(ROOT/'benchmarks/throughput_gguf.py'),'--model',str(a.model),
                '--tasks',str(ROOT/'research/throughput/profile-task.json'),'--output',str(output),
                '--variant',case['variant'],'--rows',str(case['rows']),'--cache-policy',case['cache']]
            with (a.output/'commands.jsonl').open('a') as f:f.write(json.dumps(command)+'\n')
            child=subprocess.Popen(command)
            try:code=child.wait()
            except BaseException:
                import signal
                if child.poll() is None:child.send_signal(signal.SIGINT)
                child.wait(timeout=10);raise
            if code:raise RuntimeError('Ablation failed or guard refused; no retry')
            d=json.loads(output.read_text());t=d['turns'][0]
            if t['stop_reason']!='eos':raise RuntimeError('Incomplete generation; no further run')
            print(json.dumps({'block':block,'case':case['name'],'decode_tok_s':t['decode_tokens_per_second'],'ttft':t['ttft_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
