#!/usr/bin/env python3
"""Repeated real miniature decoding and KV block growth, not a quality benchmark."""
import argparse,json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
from airllm.nextgen import AirLLMNextModel,NextConfig
from airllm.nextgen.tuning import telemetry

p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--requests',type=int,default=20);p.add_argument('--tokens',type=int,default=128);p.add_argument('--output',default='evidence-next/stress-cpu.json');a=p.parse_args()
opts=NextConfig(device='cpu',dtype='float32',load_tokenizer=False,max_context=256,
  cache_bytes=4*1024**2,io_bytes=4*1024**2,kv_bytes=8*1024**2,max_tensor_bytes=4*1024**2,
  workspace_bytes=16*1024**2,reserve_bytes=0,cache_dir=str(Path(a.output).parent/'stress-metadata'))
rows=[];start=perf_counter()
with AirLLMNextModel(a.model,next_config=opts) as model:
    for i in range(a.requests):
        t=perf_counter();out=model.generate([[1,8,9,10,11,12,13,14]],max_new_tokens=a.tokens,eos_token_id=[])
        assert out.shape[1]==8+a.tokens
        assert model.reader.cache.bytes<=opts.cache_bytes and model.reader.io.peak_bytes<=opts.io_bytes
        rows.append({'request':i,'seconds':perf_counter()-t,'rss_bytes':telemetry()['rss_bytes'],'metrics':model.metrics()})
        if i%5==4:print('requests',i+1,'tokens',(i+1)*a.tokens,'seconds',perf_counter()-start,flush=True)
    report={'scope':'MLX CPU miniature repeated generation, no full GLM or M5 claim','tokens_generated':a.requests*a.tokens,
      'seconds':perf_counter()-start,'rss_growth_after_first_request_bytes':rows[-1]['rss_bytes']-rows[0]['rss_bytes'],
      'requests':rows,'metal':False,'M5':False}
Path(a.output).write_text(json.dumps(report,indent=2));print(a.output,flush=True)
