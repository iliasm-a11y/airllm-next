#!/usr/bin/env python3
"""Measure fresh-process imports, distinct from readiness and first-token time."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]/'air_llm'));p.add_argument('--output',default='evidence-next/import-current.json');a=p.parse_args()
rows=[]
for target in ('airllm','airllm.nextgen.cli'):
    samples=[]
    for i in range(3):
        env=dict(os.environ,PYTHONPATH=str(Path(a.source).resolve())+os.pathsep+os.environ.get('PYTHONPATH',''),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        start=time.perf_counter();r=subprocess.run([sys.executable,'-c',f'import {target}'],env=env,capture_output=True)
        samples.append(time.perf_counter()-start)
        if r.returncode:raise RuntimeError(r.stderr.decode())
    rows.append({'module':target,'seconds':samples})
Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
