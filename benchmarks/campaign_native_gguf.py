#!/usr/bin/env python3
"""Shuffled paired GGUF ablations; each experiment has its own strict guard."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={'reference':('reference',2048,None), 'metal512':('metal',512,None),
          'metal2048':('metal',2048,None), 'metal-decay':('metal',2048,32768)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cases',type=Path,default=ROOT/'research/native_gguf/CASES.json')
    p.add_argument('--tokens',type=int,default=8)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    args=p.parse_args()
    if args.tokens<1 or args.repeats<1:p.error('Positive token and repeat counts required')
    cases=json.loads(args.cases.read_text());args.output_dir.mkdir(parents=True,exist_ok=True)
    tasks=[(ci,ri,v) for ci in range(len(cases)) for ri in range(args.repeats) for v in args.variants]
    random.Random(20261023).shuffle(tasks)
    manifest={'tokens_limit':args.tokens,'repeats':args.repeats,'seed':20261023,
              'cases':cases,'variants':{v:VARIANTS[v] for v in args.variants},'order':tasks,'completed':[]}
    output=args.output_dir/'campaign.json'
    if output.exists():raise FileExistsError('Use a new campaign directory to preserve earlier results')
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    for i,(ci,ri,variant) in enumerate(tasks):
        kernel,rows,decay=VARIANTS[variant]
        destination=args.output_dir/f'{cases[ci]["name"]}-r{ri}-{variant}.json'
        command=[sys.executable,str(ROOT/'benchmarks/benchmark_native_gguf.py'),
                 '--model',str(args.model),'--output',str(destination),'--phase','generate',
                 '--token-ids',','.join(map(str,cases[ci]['input_ids'])),'--tokens',str(args.tokens),
                 '--kernel',kernel,'--rows',str(rows),'--cache-mib','256',
                 '--logits-output',str(destination.with_suffix('.npy'))]
        if decay:command+=['--cache-decay',str(decay)]
        print(f'{i+1}/{len(tasks)} {destination.name}',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        manifest['completed'].append(destination.name)
        output.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
