#!/usr/bin/env python3
"""Launch one experiment with sampled process RSS and system-reserve guards."""
import argparse
from pathlib import Path
import sys
from benchmark_granite_moe import run_guarded

p=argparse.ArgumentParser()
p.add_argument('--max-rss-gib',type=float,default=5)
p.add_argument('--reserve-gib',type=float,default=1.5)
a,experiment=p.parse_known_args()
a.experiment=experiment
if not a.experiment:p.error('Supply benchmark_multitoken_moe.py arguments')
try:output=Path(a.experiment[a.experiment.index('--output')+1])
except (ValueError,IndexError):p.error('experiment requires --output')
output.parent.mkdir(parents=True,exist_ok=True)
run_guarded([sys.executable,str(Path(__file__).with_name('benchmark_multitoken_moe.py'))]+a.experiment,output,a)
