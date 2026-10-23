#!/usr/bin/env python3
"""Ablate exact source residency/output tiles on a complete trained Llama.

Every variant runs in a fresh process, with the same explicit total cache
budget as its paired control. No downloads, OS-cache purge, artificial delay
or inference-weight modification. This is not a physical-SSD benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def worker(args):
    import numpy as np
    import psutil
    import resource
    from transformers import AutoTokenizer
    from airllm.nextgen.config import NextConfig, MiB
    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
    from airllm.nextgen.storage import physical_read_bytes

    variant={'decoded300':(300,0,False),'compact300':(300,300,False),
             'decoded600':(600,0,False),'tiles600':(600,0,True)}[args.worker]
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    prompts=['What is the capital of France?','Write a Python function that adds two numbers.']
    inputs=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,
                add_generation_prompt=True,return_dict=False) for p in prompts]
    if any(not isinstance(ids,list) for ids in inputs):
        raise TypeError('Expected flat token IDs from local tokenizer')
    with tempfile.TemporaryDirectory(prefix='airllm-weight-v5-meta-') as meta:
        cfg=NextConfig(device='cpu',dtype='float32',cache_bytes=variant[0]*MiB,
            compact_cache_bytes=variant[1]*MiB,output_tile_cache=variant[2],
            io_bytes=32*MiB,max_tensor_bytes=32*MiB,kv_bytes=32*MiB,
            workspace_bytes=128*MiB,reserve_bytes=256*MiB,max_context=512,
            prefetch=False,use_saved_tuning=False,load_tokenizer=False,cache_dir=meta)
        with AirLLMNextLlamaModel(args.model_dir,next_config=cfg) as model:
            rows=[]
            # The first generation sees empty software caches. Subsequent runs
            # reuse whatever each policy actually retains; OS pages are unknown.
            for repeat in range(args.repeats+1):
                for index,ids in enumerate(inputs):
                    cp=model.checkpoint;before=cp.stats['logical_bytes'];physical=physical_read_bytes()
                    encoded_before=cp.stats['read_calls'];swap=psutil.swap_memory().used
                    start=time.perf_counter()
                    output=model.generate([ids],max_new_tokens=args.tokens,eos_token_id=[]).tolist()[0]
                    seconds=time.perf_counter()-start
                    physical_end=physical_read_bytes()
                    metrics=model.metrics()
                    rows.append({'repeat':repeat,'prompt_index':index,'cold_software_cache':repeat==0 and index==0,
                        'token_ids':output,'generated_text':tokenizer.decode(output[len(ids):],skip_special_tokens=True),
                        'new_tokens':len(output)-len(ids),'generation_seconds':seconds,
                        'logical_file_bytes':cp.stats['logical_bytes']-before,
                        'read_calls':cp.stats['read_calls']-encoded_before,
                        'physical_read_bytes':physical_end-physical if physical is not None and physical_end is not None else None,
                        'swap_growth_bytes':max(0,psutil.swap_memory().used-swap),
                        'ttft_seconds':metrics['runtime']['ttft_seconds'],
                        'decode_tokens_per_second':metrics['runtime']['decode_tokens_per_second'],
                        'cache':metrics['cache'],
                        'compact_cache':dict(model.reader.compact_cache.stats)|{
                            'bytes':model.reader.compact_cache.bytes,'peak_bytes':model.reader.compact_cache.peak_bytes,
                            'capacity':model.reader.compact_cache.capacity}})
            result={'variant':args.worker,'config':cfg.as_dict(),'rows':rows,
                'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'platform':platform.platform(),'python':sys.version,
                'mlx_device':model.device,'weight_budget_bytes':cfg.cache_bytes}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tokens',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker',choices=['decoded300','compact300','decoded600','tiles600'])
    args=parser.parse_args()
    if args.tokens<1 or args.repeats<1:parser.error('tokens/repeats must be positive')
    if args.worker:return worker(args)
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    runs=[]
    # Reverse the complete order on the second round to expose order effects.
    for round_index,variants in enumerate([
        ['decoded300','compact300','decoded600','tiles600'],
        ['tiles600','decoded600','compact300','decoded300']]):
        for variant in variants:
            dest=output.with_name(output.stem+f'-round{round_index}-{variant}.json')
            cmd=[sys.executable,__file__,'--worker',variant,'--model-dir',args.model_dir,
                 '--output',str(dest),'--tokens',str(args.tokens),'--repeats',str(args.repeats)]
            done=subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                MLX_ENABLE_TF32='0'),capture_output=True,text=True)
            dest.with_suffix('.log').write_text(done.stdout+done.stderr)
            if done.returncode:raise RuntimeError(f'{variant} failed; see {dest.with_suffix(".log")}')
            runs.append(json.loads(dest.read_text())|{'round':round_index,'evidence_file':dest.name})
    summaries={}
    for variant in ('decoded300','compact300','decoded600','tiles600'):
        selected=[row for run in runs if run['variant']==variant for row in run['rows'] if row['repeat']>0]
        summaries[variant]={'warm_median_generation_seconds':statistics.median(r['generation_seconds'] for r in selected),
            'warm_median_logical_bytes':statistics.median(r['logical_file_bytes'] for r in selected),
            'warm_median_ttft_seconds':statistics.median(r['ttft_seconds'] for r in selected),
            'warm_median_decode_tokens_per_second':statistics.median(r['decode_tokens_per_second'] for r in selected),
            'process_peak_rss_bytes_max':max(run['process_peak_rss_bytes'] for run in runs if run['variant']==variant),
            'timed_warm_generations':len(selected)}
    pairs=[]
    for control,variant in [('decoded300','compact300'),('decoded600','tiles600')]:
        a,b=summaries[control],summaries[variant]
        pairs.append({'control':control,'variant':variant,
            'full_generation_speedup':a['warm_median_generation_seconds']/b['warm_median_generation_seconds'],
            'logical_file_bytes_reduction_fraction':1-b['warm_median_logical_bytes']/a['warm_median_logical_bytes'] if a['warm_median_logical_bytes'] else None})
    reference={row['prompt_index']:row['token_ids'] for row in runs[0]['rows'] if row['repeat']==0}
    equal=all(row['token_ids']==reference[row['prompt_index']] for run in runs for row in run['rows'])
    weight_path=Path(args.model_dir)/'model.safetensors'
    result={'model_path':str(Path(args.model_dir).resolve()),
        'checkpoint_sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        'checkpoint_bytes':weight_path.stat().st_size,'same_generated_sequences_all_variants':equal,
        'scope':'Complete trained checkpoint, MLX CPU; checkpoint fits RAM; no giant-model/OOC/SSD claim',
        'cache_state':'Software caches empty at each worker start; OS page cache uncontrolled; two reversed orders',
        'numerical_transport':'Source cache retains original bytes; no quantization or weight changes',
        'summaries':summaries,'pairs':pairs,
        'evidence_files':[r['evidence_file'] for r in runs]}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
