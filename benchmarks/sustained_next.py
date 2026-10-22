#!/usr/bin/env python3
"""Time-bounded sustained inference on a user-supplied local checkpoint.

Records real wall times and memory; no invented GPU utilization or temperature.
Separate epochs allow comparison of early and late performance on fanless Macs.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.tuning import auto_config,telemetry


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--prompt',default='Explique les limites de la bande passante.')
    p.add_argument('--device',choices=['cpu','gpu','auto'],default='auto');p.add_argument('--profile',choices=['balanced','performance'],default='balanced')
    p.add_argument('--output',default='evidence-next/sustained.json');p.add_argument('--quantized-dir');a=p.parse_args()
    cfg=auto_config(a.profile,device=a.device,quantized_dir=a.quantized_dir)
    with AirLLMNextModel(a.model,next_config=cfg) as model:
        ids=[model.tokenizer.encode(a.prompt)];start=perf_counter();epochs=[]
        while perf_counter()-start<a.seconds:
            before=perf_counter();model.generate(ids,max_new_tokens=a.tokens,eos_token_id=-1)
            epochs.append({'elapsed':perf_counter()-start,'wall_seconds':perf_counter()-before,'metrics':model.metrics(),'telemetry':telemetry()})
            Path(a.output).parent.mkdir(parents=True,exist_ok=True)
            Path(a.output).write_text(json.dumps({'epochs':epochs,'full_model_test':'user-supplied checkpoint','thermal_sensor_available':False},indent=2))
            print(f'{len(epochs)} epochs, {perf_counter()-start:.1f}s, {epochs[-1]["metrics"]["runtime"]["decode_tokens_per_second"]:.3f} tok/s',flush=True)


if __name__=='__main__':main()
