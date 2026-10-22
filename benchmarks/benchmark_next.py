#!/usr/bin/env python3
"""Reproducible synthetic GLM numerical/performance suite, no downloads.

Results are actual local host timings. Tiny random models are not evidence of
GLM-5.3 full-model throughput or of natural-language generation quality.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter,time_ns
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'air_llm'),str(Path(__file__).resolve().parents[1]/'tests_next'),str(Path(__file__).resolve().parents[1]/'tests_ultra')]
import numpy as np
import torch
import mlx.core as mx
from transformers import GlmMoeDsaConfig,GlmMoeDsaForCausalLM
from airllm.nextgen import AirLLMNextModel,NextConfig
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.tuning import telemetry,measure_storage
from airllm.nextgen.sidecar import convert_experts
from airllm.ultra import AirLLMUltraModel,UltraConfig
from test_fp8_model import save_glm64


def larger_fixture(path):
    torch.manual_seed(39)
    c=GlmMoeDsaConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=6,
       num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=512,
       n_routed_experts=16,num_experts_per_tok=4,n_shared_experts=1,first_k_dense_replace=1,
       moe_intermediate_size=64,kv_lora_rank=32,q_lora_rank=64,qk_nope_head_dim=16,qk_rope_head_dim=16,
       v_head_dim=16,index_n_heads=8,index_head_dim=32,index_topk=8,
       indexer_types=['full','shared','shared','full','shared','shared'],n_group=2,topk_group=1,
       rope_parameters={'rope_type':'default','rope_theta':10000.},eos_token_id=2,pad_token_id=0,bos_token_id=1)
    model=GlmMoeDsaForCausalLM(c).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if layer.self_attn.indexer is not None:layer.self_attn.indexer.weights_proj.weight.abs_()
    model.save_pretrained(path,max_shard_size='1MB');return model


def run_case(root,reference,kind,device,count,sidecar=None):
    opts=NextConfig(cache_bytes=4*1024**2,io_bytes=4*1024**2,kv_bytes=8*1024**2,
      max_tensor_bytes=4*1024**2,workspace_bytes=16*1024**2,reserve_bytes=0,
      prefill_chunk=8,max_context=256,device=device,dtype='float32',load_tokenizer=False,
      cache_dir=str(root.parent/'metadata'),quantized_dir=str(sidecar) if sidecar else None)
    before=perf_counter()
    model=AirLLMNextModel(root,next_config=opts)
    construction=perf_counter()-before
    ids=np.array([[1,8,9,10,11,12,13,14]],np.int32)
    with model,torch.inference_mode():
        before=perf_counter();a=np.array(model(ids).logits);prefill_time=perf_counter()-before
        b=reference(torch.from_numpy(ids)).logits.numpy();diff=a-b
        bf=b.astype(np.float64);p=np.exp(bf-bf.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        af=a.astype(np.float64);q=np.exp(af-af.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True)
        quality={'max_abs_logit_error':float(np.abs(diff).max()),'rms_logit_error':float(np.sqrt(np.mean(diff**2))),
          'argmax_agreement':float(np.mean(a.argmax(-1)==b.argmax(-1))),
          'mean_KL_ref_to_candidate':float(np.mean(np.sum(p.astype(np.float64)*np.log((p.astype(np.float64)+1e-30)/(q.astype(np.float64)+1e-30)),axis=-1))),
          'scope':'random miniature weights; not a linguistic quality evaluation of GLM-5.3'}
        records=[]
        for repeat in range(3):
            model.generate(ids,max_new_tokens=count,eos_token_id=[])
            records.append(model.metrics())
        return {'label':kind,'parameter_count':sum(p.numel() for p in reference.parameters()),
         'checkpoint_bytes':model.checkpoint.total_bytes,'construction_seconds':construction,
         'first_prefill_seconds':prefill_time,'quality_vs_original':quality,'runs':records,
         'cache_state':'first request has software cache empty; OS page cache not flushed',
         'full_GLM53_result':False,'M5_result':device=='gpu' and platform.machine()=='arm64'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='evidence-next/benchmark-native.json');p.add_argument('--work-dir',default='.benchmark-next')
    p.add_argument('--device',default='cpu',choices=['cpu','gpu']);p.add_argument('--tokens',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(1);mx.set_default_device(getattr(mx,a.device))
    work=Path(a.work_dir).resolve()/('run-'+str(time_ns()));work.mkdir(parents=True,exist_ok=False)
    result={'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'mlx':mx.__version__,
       'device':a.device,'metal_available':mx.metal.is_available(),'telemetry_begin':telemetry(),
       'scope':'synthetic GLM DSA fixtures; not full GLM-5.3 and not SSD Fanxiang','cases':[]}
    for size in ['tiny64','larger128','fp8_tiny64']:
        root=work/size;ref=larger_fixture(root) if size=='larger128' else save_glm64(root,size.startswith('fp8'))
        print('benchmark',size,flush=True)
        result['cases'].append(run_case(root,ref,size,a.device,a.tokens))
        if size=='tiny64':
            uopts=UltraConfig(cache_bytes=4*1024**2,prefetch_bytes=4*1024**2,max_group_bytes=4*1024**2,
                         reserve_bytes=0,load_tokenizer=False,cache_dir=str(work/'ultra-meta'))
            t=perf_counter()
            with AirLLMUltraModel(root,device='cpu',dtype=torch.float32,ultra=uopts) as baseline:
                construction=perf_counter()-t;ids=torch.tensor([[1,8,9,10,11,12,13,14]])
                start=perf_counter();baseline.generate(ids,max_new_tokens=a.tokens,eos_token_id=[]);elapsed=perf_counter()-start
                result['ultra_cpu_baseline']={'construction_seconds':construction,'generation_seconds':elapsed,
                  'tokens':a.tokens,'total_tokens_per_second_including_prefill':a.tokens/elapsed,
                  'logical_read_bytes':baseline.checkpoint.stats['logical_bytes'],
                  'scope':'same tiny64 fixture; baseline CPU vs Next device specified above'}
            with OffsetCheckpoint(root,work/'q-meta') as cp:
                for bits in (4,8):
                    destination=work/f'int{bits}';convert_experts(cp,ref.config.to_dict(),destination,bits,32)
                    result['cases'].append(run_case(root,ref,f'affine_int{bits}_tiny64',a.device,a.tokens,destination))
                result['storage_tuning']=measure_storage(cp,NextConfig(cache_dir=str(work/'q-meta')),samples=12)
    result['telemetry_end']=telemetry()
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2))
    print('wrote',target,flush=True)


if __name__=='__main__':main()
