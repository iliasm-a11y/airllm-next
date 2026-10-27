#!/usr/bin/env python3
"""Write a counterbalanced real-checkpoint protocol; never launch inference."""
import argparse
import json
from pathlib import Path


def build_plan(baseline, candidate, *, pairs=10, maximum=128, numerical=True):
    if type(pairs) is not int or pairs < 1 or type(maximum) is not int or maximum < 1:
        raise ValueError('Positive integer pair count and chunk cap required')
    common = ['--expert-batch', '--concat-tiles', '--packed-direct-io', '--bulk-metadata',
              '--zero-copy-weights', '--expert-pipeline', '--string-buffer-bytes', '65536']
    if numerical:
        common.append('--numerical-evidence')
    jobs=[]
    for pair in range(pairs):
        for variant in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')):
            flags=common.copy()
            if variant == 'candidate':
                flags += ['--hyper-prefill', '--hyper-prefill-max-tokens', str(maximum)]
            jobs.append(dict(id=f'{variant}-p{pair:02d}',
                             engine_root=str(Path(baseline if variant=='baseline' else candidate).resolve()), args=flags))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=10)
    p.add_argument('--max-chunk',type=int,default=128)
    p.add_argument('--no-observer',action='store_true',help='Corroborating timing only; separate numerical campaign required')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=build_plan(a.baseline,a.candidate,pairs=a.pairs,maximum=a.max_chunk,numerical=not a.no_observer)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(plan,f,indent=2);f.write('\n')


if __name__=='__main__': main()
