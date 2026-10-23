#!/usr/bin/env python3
"""Isolated exact lookup timing, explicitly not inference/SSD performance."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter
import types
from airllm.nextgen.prompt_draft import PromptSpanDraft

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=9);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
ref='9629c27b9d8bdf697ae866498ad468e11eb183d4'
source=subprocess.check_output(['git','show',ref+':air_llm/airllm/nextgen/prompt_draft.py'],cwd=root,text=True)
old=types.ModuleType('lookup_v4');exec(compile(source,'prompt_draft_v4.py','exec'),old.__dict__)
rng=random.Random(7)
workloads={'no_match':list(range(8192)),'random_tokens':[rng.randrange(4096) for _ in range(8192)],
           'repeat_pattern':[1,2,3,4]*2048,'repeat_one':[1]*8192}
records=[]
for name,prefix in workloads.items():
    cases={'v4':old.PromptSpanDraft(),'v5':PromptSpanDraft()};outputs={}
    for impl,lookup in cases.items():outputs[impl]=lookup(prefix,8)
    assert outputs['v4']==outputs['v5']
    for repetition in range(a.repeats):
        order=list(cases);rng.shuffle(order)
        for impl in order:
            start=perf_counter();output=cases[impl](prefix,8);seconds=perf_counter()-start
            assert output==outputs['v4']
            records.append(dict(workload=name,implementation=impl,repetition=repetition,seconds=seconds,output=output))
summary={}
for name in workloads:
    v={i:statistics.median(r['seconds'] for r in records if r['workload']==name and r['implementation']==i) for i in ['v4','v5']}
    summary[name]=v|{'v4_over_v5':v['v4']/v['v5']}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(kind='isolated CPU prompt lookup microbenchmark; not end-to-end inference',
    baseline_ref=ref,baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),summary=summary,records=records),indent=2)+'\n')
print(json.dumps(summary,indent=2))
