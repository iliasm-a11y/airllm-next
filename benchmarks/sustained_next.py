#!/usr/bin/env python3
"""Time-bounded sustained inference on a user-supplied local checkpoint.

Records real wall times and memory; no invented GPU utilization or temperature.
Separate epochs allow comparison of early and late performance on fanless Macs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
os.environ.setdefault('MLX_ENABLE_TF32','0')
from airllm import AutoModel
from airllm.nextgen.tuning import auto_config,telemetry


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--prompt',default='Explique les limites de la bande passante.')
    p.add_argument('--device',choices=['cpu','gpu','auto'],default='auto');p.add_argument('--profile',choices=['balanced','performance'],default='balanced')
    p.add_argument('--backend',choices=['next','next-llama','next-qwen-experimental'],default='next')
    for name in ('cache','io','kv','workspace','reserve'):
        p.add_argument(f'--{name}-mib',type=int)
    p.add_argument('--output',default='evidence-next/sustained.json');p.add_argument('--quantized-dir');a=p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds<=0 or a.tokens<1:p.error('seconds and tokens must be positive')
    budgets={}
    for name in ('cache','io','kv','workspace','reserve'):
        value=getattr(a,name+'_mib')
        if value is not None:
            if value<0:p.error('Memory budgets must be nonnegative')
            budgets[name+'_bytes']=value*1024**2
    if 'io_bytes' in budgets:budgets['max_tensor_bytes']=max(1,budgets['io_bytes'])
    cfg=auto_config(a.profile,device=a.device,quantized_dir=a.quantized_dir,**budgets)
    with AutoModel.from_pretrained(a.model,backend=a.backend,next_config=cfg) as model:
        ids=[model.tokenizer.encode(a.prompt)];start=perf_counter();epochs=[]
        while perf_counter()-start<a.seconds:
            before=perf_counter();model.generate(ids,max_new_tokens=a.tokens,eos_token_id=[])
            allocator={}
            for field,method in (('active_bytes','get_active_memory'),('peak_bytes','get_peak_memory'),('cache_bytes','get_cache_memory')):
                try:allocator[field]=int(getattr(model.mx,method)())
                except (AttributeError,RuntimeError):allocator[field]=None
            allocator['scope']='Process-wide MLX allocator; peak is since initialization, not isolated model RSS'
            epochs.append({'elapsed':perf_counter()-start,'wall_seconds':perf_counter()-before,'metrics':model.metrics(),'telemetry':telemetry(),'mlx_allocator':allocator})
            Path(a.output).parent.mkdir(parents=True,exist_ok=True)
            Path(a.output).write_text(json.dumps({'epochs':epochs,'full_model_test':'user-supplied checkpoint','thermal_sensor_available':False},indent=2))
            print(f'{len(epochs)} epochs, {perf_counter()-start:.1f}s, {epochs[-1]["metrics"]["runtime"]["decode_tokens_per_second"]:.3f} tok/s',flush=True)


if __name__=='__main__':main()
