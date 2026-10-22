#!/usr/bin/env python3
"""Fresh-process wall time to a real first miniature-model token, no tokenizer.

This cannot predict GLM-5.3 disk/compute time. Dependencies are preinstalled.
Parent and child inherit offline mode; source/checkpoint must already be local.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',default='evidence-next/startup-process.json');a=p.parse_args()
root=Path(__file__).resolve().parents[1];model=Path(a.model).resolve();meta=model.parent/'startup-metadata'
env=dict(os.environ,PYTHONPATH=str(root/'air_llm')+os.pathsep+os.environ.get('PYTHONPATH',''),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
next_code='''from airllm import AutoModel
from airllm.nextgen import NextConfig
import sys
with AutoModel.from_pretrained(sys.argv[1],backend='mlx-next',next_config=NextConfig(device='cpu',dtype='float32',load_tokenizer=False,cache_dir=sys.argv[2],cache_bytes=4*1024**2,io_bytes=4*1024**2,kv_bytes=8*1024**2,workspace_bytes=16*1024**2,reserve_bytes=0)) as model:
    print(model.generate([[1,8,9]],max_new_tokens=1).tolist())
'''
ultra_code='''import torch
from airllm.ultra import AirLLMUltraModel,UltraConfig
import sys
torch.set_num_threads(1)
with AirLLMUltraModel(sys.argv[1],device='cpu',dtype=torch.float32,ultra=UltraConfig(load_tokenizer=False,cache_dir=sys.argv[2],cache_bytes=4*1024**2,prefetch_bytes=4*1024**2,reserve_bytes=0)) as model:
    print(model.generate(torch.tensor([[1,8,9]]),max_new_tokens=1).tolist())
'''
rows=[]
for backend,code in [('ultra',ultra_code),('next',next_code)]:
    for i in range(3):
        begin=time.perf_counter();r=subprocess.run([sys.executable,'-c',code,str(model),str(meta/backend)],env=env,text=True,capture_output=True,timeout=60)
        elapsed=time.perf_counter()-begin
        if r.returncode:raise RuntimeError(r.stderr)
        rows.append({'backend':backend,'run':i,'seconds_process_to_first_token':elapsed,'stdout':r.stdout.strip()})
result={'scope':'fresh Python process + dependencies + miniature GLM token; software/OS cache warm, no tokenizer; not M5 or full GLM-5.3',
 'model':str(model),'measurements':rows}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
