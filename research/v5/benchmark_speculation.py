#!/usr/bin/env python3
"""Ablate V4/V5 retrieval and adaptive verification on one trained checkpoint.

Both policies use the current weight engine and the same budget. This isolates
retrieval/policy changes, not every V5 weight-cache change. Timing excludes model
construction/tokenization; pages are not purged. No synthetic SSD is simulated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
from time import perf_counter
import types
import numpy as np
import mlx.core as mx
from airllm.nextgen.config import NextConfig,MiB
from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
from airllm.nextgen.prompt_draft import PromptSpanDraft
from airllm.nextgen.speculative import greedy_speculative_generate
from airllm.nextgen.storage import encoded_read_bytes,physical_read_bytes


def module_from_git(root,ref,filename,name):
    source=subprocess.check_output(['git','show',f'{ref}:{filename}'],cwd=root,text=True)
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(source,filename,'exec'),module.__dict__)
    return module,hashlib.sha256(source.encode()).hexdigest()


def run(model_path,ids,count,mode,old,old_lookup):
    options=NextConfig(device='cpu',dtype='float32',cache_bytes=8*MiB,
        io_bytes=32*MiB,kv_bytes=32*MiB,workspace_bytes=128*MiB,reserve_bytes=64*MiB,
        max_tensor_bytes=32*MiB,prefetch=False,row_chunk=2048,prefill_chunk=16,
        max_context=1024,load_tokenizer=False,use_saved_tuning=False)
    with AirLLMNextLlamaModel(model_path,next_config=options) as model:
        start_physical=physical_read_bytes();begin=perf_counter()
        if mode=='ordinary':
            output=model.generate([ids],max_new_tokens=count,eos_token_id=[]).tolist()[0]
            measurements=model.metrics()['runtime']
        else:
            fn=old.greedy_speculative_generate if mode=='v4_prompt_lookup' else greedy_speculative_generate
            draft=old_lookup.PromptSpanDraft() if mode=='v4_prompt_lookup' else PromptSpanDraft()
            if mode=='v5_wrong_draft':draft=lambda p,n:[0]*n
            def choose(logits):
                with mx.stream(model.stream):return mx.argmax(logits,axis=-1).tolist()
            kwargs={} if mode=='v4_prompt_lookup' else dict(draft_bytes_read=lambda:0,draft_encoded_bytes_read=lambda:0)
            result=fn(model,draft,[ids],count,eos_token_ids=(),max_depth=4,adaptive=True,
                argmax_tokens=choose,vocab_size=model.config['vocab_size'],
                bytes_read=lambda:model.checkpoint.stats['logical_bytes'],
                encoded_bytes_read=lambda:encoded_read_bytes(model.checkpoint),**kwargs)
            output=result.tokens;measurements=result.metrics
        elapsed=perf_counter()-begin;end_physical=physical_read_bytes()
        return dict(mode=mode,seconds=elapsed,output=output,metrics=measurements,
            logical_bytes=model.checkpoint.stats['logical_bytes'],
            encoded_file_bytes=encoded_read_bytes(model.checkpoint),
            physical_read_bytes=end_physical-start_physical if start_physical is not None and end_physical is not None else None,
            process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            options=options.as_dict())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-ref',default='9629c27b9d8bdf697ae866498ad468e11eb183d4')
    parser.add_argument('--repeats',type=int,default=2);parser.add_argument('--tokens',type=int,default=24)
    parser.add_argument('--prompt-limit',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    old,old_hash=module_from_git(root,args.baseline_ref,'air_llm/airllm/nextgen/speculative.py','airllm.nextgen._speculation_v4')
    old_lookup,lookup_hash=module_from_git(root,args.baseline_ref,'air_llm/airllm/nextgen/prompt_draft.py','airllm.nextgen._lookup_v4')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prompts=[('repeated_sentence','The capital of France is Paris. The capital of France is Paris. The capital of France is'),
             ('repeated_pattern','A B C A B C A B C A B C A B C A B C'),
             ('code','def factorial(n):\n    if n == 0:\n        return 1\n    return')]
    prompts=prompts[:args.prompt_limit]
    records=[];rng=random.Random(20261023)
    for rep in range(args.repeats):
        for label,prompt in prompts:
            ids=tokenizer.encode(prompt)
            modes=['ordinary','v4_prompt_lookup','v5_prompt_lookup']
            rng.shuffle(modes)
            group=[]
            for mode in modes:
                value=run(args.model,ids,args.tokens,mode,old,old_lookup)
                value.update(repetition=rep,prompt_name=label,prompt=prompt)
                value['text']=tokenizer.decode(value['output'][len(ids):],skip_special_tokens=True)
                records.append(value);group.append(value)
                print(f"{rep} {label} {mode}: {value['seconds']:.4f}s",flush=True)
            assert all(r['output']==group[0]['output'] for r in group),'Different greedy trajectories observed'
    # A deliberately wrong, zero-transport draft is a negative control, not a
    # useful trained draft or a speed result. Its proposal/computation cost is real.
    ids=tokenizer.encode(prompts[0][1]);wrong=run(args.model,ids,args.tokens,'v5_wrong_draft',old,old_lookup)
    wrong.update(repetition=0,prompt_name='wrong_draft_control',prompt=prompts[0][1])
    ordinary=next(r for r in records if r['repetition']==0 and r['prompt_name']==prompts[0][0] and r['mode']=='ordinary')
    assert wrong['output']==ordinary['output'];records.append(wrong)
    summary={}
    for label,_ in prompts:
        modes={mode:dict(median_seconds=statistics.median(r['seconds'] for r in records
                    if r['prompt_name']==label and r['mode']==mode),
                median_logical_bytes=statistics.median(r['logical_bytes'] for r in records
                    if r['prompt_name']==label and r['mode']==mode),
                median_encoded_file_bytes=statistics.median(r['encoded_file_bytes'] for r in records
                    if r['prompt_name']==label and r['mode']==mode))
               for mode in ('ordinary','v4_prompt_lookup','v5_prompt_lookup')}
        summary[label]=modes|{'v4_over_v5_generation_time':modes['v4_prompt_lookup']['median_seconds']/modes['v5_prompt_lookup']['median_seconds']}
    payload=dict(experiment='trained speculation/retrieval policy ablation',baseline_ref=args.baseline_ref,
        baseline_speculation_source_sha256=old_hash,baseline_lookup_source_sha256=lookup_hash,
        checkpoint=str(args.model),checkpoint_sha256=hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
        platform=sys.platform,device='MLX CPU',mlx_enable_tf32=os.environ.get('MLX_ENABLE_TF32'),
        limitations=['same current weight engine in every mode','checkpoint fits host RAM and OS cache',
            'no external SSD or Metal measured','timing excludes construction/tokenizer','no trained small-model draft evaluated',
            'maxrss is cumulative process high-water mark, not per-case RSS','physical reads are host observations, not logical bytes'],
        summary=summary,records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()
