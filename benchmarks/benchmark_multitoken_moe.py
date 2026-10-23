#!/usr/bin/env python3
"""Opt-in pinned Granite experiments; warm OS cache, logical and physical bytes separated.

All target arithmetic uses original BF16 weights decoded to FP32. This is a
small trained MoE campaign, never a giant-model or external-SSD validation.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
PROMPTS=[
 'What is the capital of France? Answer in one sentence.',
 'Complete the sequence: one, two, three, four, five,',
 'Return only Python code for a function add(a, b) that returns their sum.',
 'Repeat exactly this pattern several times: red blue green red blue green',
 'Explique en deux phrases pourquoi le ciel est bleu.',
 'List the first ten square numbers in ascending order.'
]
ATOL,RTOL=1e-4,5e-4
MiB=1024**2
CONTROL_TEXT='Paris is the capital of France. The city stands on the river Seine and contains many museums, parks, libraries and railway stations. Visitors can travel between its neighbourhoods using public transport and explore its history on foot.'


def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sources():
 return {str(p.relative_to(ROOT)):digest(p) for p in
  sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py'))+[Path(__file__).resolve()]}


def delta(after,before):
 return {k:v-before.get(k,0) for k,v in after.items() if type(v) in (int,float)}


def close(model):
 model.close(save_profile=False)
 import mlx.core as mx
 mx.clear_cache()


def clear_weights(model):
 model.reader.drain();model.reader.cache.clear();model.reader.compact_cache.clear()


def configuration(args,metadata,cache=None,policy='adaptive',prefetch=False):
 from airllm.nextgen.config import NextConfig
 return NextConfig(device=args.device,dtype='float32',cache_bytes=(cache or args.cache_mib)*MiB,
  io_bytes=32*MiB,kv_bytes=32*MiB,workspace_bytes=256*MiB,reserve_bytes=1536*MiB,
  max_tensor_bytes=32*MiB,row_chunk=2048,prefill_chunk=64,max_context=512,
  policy=policy,prefetch=prefetch,spec_prefetch=False,load_tokenizer=False,
  cache_dir=str(metadata),use_saved_tuning=False,io_mode=args.io_mode,output_tile_cache=True)


def construct(root,cfg,mode='union'):
 from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
 return AirLLMNextGraniteMoeModel(root,next_config=cfg,expert_execution=mode,expert_metrics_capacity=0)


def native_snapshot(model):
 return dict(io=dict(model.checkpoint.stats),experts=model.expert_batch_metrics()['totals'],
  cache=dict(model.reader.cache.stats),loads=dict(model.reader.stats))


def measure_generation(model,ids,length,draft=None,depth=4,adaptive=False,stop_at_eos=True):
 from airllm.nextgen.physical_io import physical_read_bytes
 before=native_snapshot(model);physical=physical_read_bytes();start=time.perf_counter()
 if draft is None:
  output=model.generate([ids],max_new_tokens=length,eos_token_id=None if stop_at_eos else [])
 else:
  output=model.generate_speculative([ids],draft,max_new_tokens=length,eos_token_id=None if stop_at_eos else [],
   max_depth=depth,adaptive=adaptive,candidate_depths=(2,4,8,16,32) if adaptive else None)
 values=output.tolist()[0];elapsed=time.perf_counter()-start;end=physical_read_bytes()
 metrics=model.metrics();after=native_snapshot(model)
 return dict(seconds=elapsed,tokens=values,new_tokens=len(values)-len(ids),
  tokens_per_second=(len(values)-len(ids))/elapsed,
  io_delta=delta(after['io'],before['io']),expert_delta=delta(after['experts'],before['experts']),
  runtime=metrics['runtime'],cache=metrics['cache'],io_scheduler=metrics['prefetch'],
  spec_prefetch=metrics['spec_prefetch'],stop_at_eos=stop_at_eos,
  physical_read_bytes=end-physical if physical is not None and end is not None else None,
  kv_peak_bytes=metrics['runtime']['kv_peak_bytes'],weights=metrics['weights'])


def block(model,context,values,execution):
 import numpy as np
 from airllm.nextgen.physical_io import physical_read_bytes
 cache=model.forward([context],logits_to_keep=1).past_key_values
 clear_weights(model);model.expert_batch_metrics(reset=True)
 before=native_snapshot(model);physical=physical_read_bytes();start=time.perf_counter()
 logits=[];routes={str(i):[] for i in range(len(model.mlps))}
 for piece in ([[v] for v in values] if execution=='tokenwise' else [values]):
  out=model.forward([piece],past_key_values=cache,logits_to_keep=0);cache=out.past_key_values
  logits.append(np.asarray(out.logits))
  for i,mlp in enumerate(model.mlps):routes[str(i)].extend(mlp.last_route_ids.tolist())
 elapsed=time.perf_counter()-start;end=physical_read_bytes();after=native_snapshot(model)
 return np.concatenate(logits,axis=1),routes,dict(seconds=elapsed,positions=len(values),
  io_delta=delta(after['io'],before['io']),expert_delta=delta(after['experts'],before['experts']),
  physical_read_bytes=end-physical if physical is not None and end is not None else None,
  kv_bytes=cache.bytes,kv_peak_bytes=cache.peak_bytes,cache=model.metrics()['cache'],
  io_peak_bytes=model.reader.io.peak_bytes)


def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True)
 p.add_argument('--draft-dir',type=Path);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--phase',choices=['batch','generation','cache','combination'],required=True)
 p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--cache-mib',type=int,default=256)
 p.add_argument('--draft-cache-mib',type=int,default=64);p.add_argument('--prompts',type=int,default=4)
 p.add_argument('--tokens',type=int,default=48);p.add_argument('--repeats',type=int,default=2)
 p.add_argument('--depths',type=int,nargs='+',default=[2,4,8,16,32]);p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,4,8,16,32])
 p.add_argument('--drafts',nargs='+',choices=['prompt','smollm'],default=['prompt'])
 p.add_argument('--policies',nargs='+',default=['adaptive','shared','expert','hybrid','batch-aware'])
 p.add_argument('--io-mode',choices=['pread','pread_nocache'],default='pread')
 args=p.parse_args()
 if args.prompts<1 or args.prompts>len(PROMPTS) or args.repeats<1 or args.tokens<1:
  p.error('positive tokens/repeats and 1..6 prompts required')
 import mlx.core as mx
 import numpy as np
 import psutil
 from transformers import AutoTokenizer
 from airllm.nextgen.prompt_draft import PromptSpanDraft
 from airllm.nextgen.physical_io import physical_read_counter_info
 mx.set_default_device(getattr(mx,args.device))
 identity=sources();root=args.model_dir.resolve();prov=json.loads((root/'download-provenance.json').read_text())
 for entry in prov['files']:
  if digest(root/entry['name'])!=entry['sha256']:raise ValueError('Checkpoint identity mismatch')
 tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
 ids=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=True,
  add_generation_prompt=True,return_dict=False) for s in PROMPTS[:args.prompts]]
 args.output.parent.mkdir(parents=True,exist_ok=True);records=[];arrays={};failures=[]
 report=dict(schema_version=1,phase=args.phase,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
  primary_campaign=True,eos_policy='teacher_forced_natural_text' if args.phase=='batch' else 'model_config_stop',
  batch_control_text=CONTROL_TEXT if args.phase=='batch' else None,
  source_sha256=identity,checkpoint=prov,host=platform.platform(),host_memory_bytes=psutil.virtual_memory().total,
  libraries={n:importlib.metadata.version(n) for n in ['mlx','torch','transformers','numpy','safetensors']},
  os_cache='checkpoint SHA256 verification warms pages; OS cache not purged; logical reads are not physical storage bytes',
  physical_counter=physical_read_counter_info(),physical_external_ssd_measured=False,giant_model_validated=False,
  original_checkpoint_fits_physical_ram=True,precision='original BF16 weights decoded to FP32',
  comparison_tolerance_predeclared=dict(atol=ATOL,rtol=RTOL),records=records,failures=failures,
  timing_scope='wall generation/verification includes draft, tokenizer conversions, route/logit copies; excludes model construction unless separately reported')
 if 'smollm' in args.drafts and args.phase in ('generation','combination'):
  if args.draft_dir is None:raise ValueError('Explicit draft checkpoint required')
  report['draft_checkpoint']={'model_id':'HuggingFaceTB/SmolLM2-135M-Instruct',
   'files_sha256':{str(f.relative_to(args.draft_dir)):digest(f) for f in sorted(args.draft_dir.glob('*')) if f.is_file()},
   'identity_scope':'local complete checkpoint hashes; official pinned revision verified separately in campaign manifest'}
 def save():
  report.update(source_unchanged=identity==sources(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   system_available_bytes=psutil.virtual_memory().available,swap_used_bytes=psutil.swap_memory().used)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
 def record(row):
  records.append(row);save();print(json.dumps({k:row[k] for k in ['kind','case','variant','depth','seconds'] if k in row}),flush=True)
 with tempfile.TemporaryDirectory(prefix='multitoken-') as td:
  cfg=configuration(args,td);report['baseline_options']=cfg.as_dict()
  baselines={}
  for case,prompt in enumerate(ids):
   if args.phase=='batch':
    continuation=tokenizer.encode(CONTROL_TEXT,add_special_tokens=False)
    if len(continuation)<max(args.batch_sizes):raise ValueError('Controlled text too short')
    baselines[case]=prompt+continuation
    continue
   model=construct(root,cfg)
   length=max(args.tokens,max(args.batch_sizes)+1) if args.phase=='batch' else args.tokens
   row=measure_generation(model,prompt,length,stop_at_eos=args.phase!='batch');baselines[case]=row['tokens']
   row.update(kind='sequence_seed',case=case,variant='autoregressive',prompt=PROMPTS[case],options=cfg.as_dict())
   record(row);close(model)
  if args.phase=='batch':
   tasks=[(repeat,case,n,variant) for repeat in range(args.repeats) for case in range(args.prompts)
    for n in args.batch_sizes for variant in ['tokenwise','serial_positions','union']]
   random.Random(20261023).shuffle(tasks);checks={}
   for repeat,case,n,variant in tasks:
    model=construct(root,cfg,'serial_positions' if variant=='serial_positions' else 'union')
    values=baselines[case][len(ids[case]):len(ids[case])+n]
    logits,routes,row=block(model,ids[case],values,variant)
    key=(repeat,case,n);checks.setdefault(key,{})[variant]=(logits,routes)
    stem=f'r{repeat}c{case}n{n}{variant}';arrays[stem+'_logits']=logits
    arrays[stem+'_routes']=np.asarray([routes[str(i)] for i in range(len(model.mlps))])
    row.update(kind='batch',case=case,repeat=repeat,variant=variant,batch_size=n,array_key=stem)
    record(row);close(model)
   for key,variants in checks.items():
    expected,routes=variants['tokenwise']
    for variant,(actual,actual_routes) in variants.items():
     result=dict(repeat=key[0],case=key[1],batch_size=key[2],variant=variant,
      max_abs_logit_error=float(np.max(np.abs(expected-actual))),
      logits_close=bool(np.allclose(expected,actual,atol=ATOL,rtol=RTOL)),
      argmax_equal=bool(np.array_equal(np.argmax(expected,axis=-1),np.argmax(actual,axis=-1))),
      route_sets_equal=all(np.array_equal(np.sort(routes[k],axis=-1),np.sort(actual_routes[k],axis=-1)) for k in routes))
     report.setdefault('comparisons',[]).append(result)
     if not all(result[k] for k in ['logits_close','argmax_equal','route_sets_equal']):failures.append(result)
   np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
   report['raw_arrays_sha256']=digest(args.output.with_suffix('.npz'))
  elif args.phase=='cache':
   tasks=[(repeat,case,policy) for repeat in range(args.repeats) for case in range(args.prompts) for policy in args.policies]
   random.Random(20261023).shuffle(tasks)
   for repeat,case,policy in tasks:
    current=replace(cfg,policy=policy);model=construct(root,current)
    row=measure_generation(model,ids[case],args.tokens)
    row.update(kind='cache',case=case,repeat=repeat,variant=policy,options=current.as_dict(),
     tokens_match_baseline=row['tokens']==baselines[case])
    if not row['tokens_match_baseline']:failures.append(dict(case=case,variant=policy,reason='token mismatch'))
    record(row);close(model)
  else:
   tasks=[(repeat,case,draft,depth,adaptive,prefetch,spec) for repeat in range(args.repeats)
    for case in range(args.prompts) for draft in args.drafts
    for depth,adaptive,prefetch,spec in ([(d,False,False,False) for d in args.depths]+[(32,True,False,False)]
     if args.phase=='generation' else [(4,False,False,False),(4,False,True,False),(4,False,True,True),(32,True,True,False)])]
   tasks += [(repeat,case,'autoregressive',0,False,False,False)
             for repeat in range(args.repeats) for case in range(args.prompts)]
   random.Random(20261023).shuffle(tasks)
   for repeat,case,draft_name,depth,adaptive,prefetch,spec in tasks:
    if draft_name=='autoregressive':
     model=construct(root,cfg);row=measure_generation(model,ids[case],args.tokens)
     row.update(kind='baseline',case=case,repeat=repeat,variant=draft_name,options=cfg.as_dict(),
      tokens_match_baseline=row['tokens']==baselines[case])
     if not row['tokens_match_baseline']:failures.append(dict(case=case,reason='baseline changed'))
     record(row);close(model);continue
    draft_model=None;current=replace(cfg,prefetch=prefetch,spec_prefetch=spec,spec_prefetch_bytes=8*MiB)
    if draft_name=='prompt':draft=PromptSpanDraft(min_match=2,max_match=32)
    else:
     if args.draft_dir is None:raise ValueError('smollm requires explicit --draft-dir')
     if args.draft_cache_mib>=args.cache_mib:raise ValueError('Draft must leave positive target cache budget')
     from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
     from airllm.nextgen.text_draft import ReencodedModelDraft
     # Equal total tensor/IO/KV/workspace budgets: draft partitions reference allowances.
     current=replace(current,cache_bytes=(args.cache_mib-args.draft_cache_mib)*MiB,
      io_bytes=24*MiB,kv_bytes=24*MiB,workspace_bytes=192*MiB)
     draft_cfg=replace(cfg,cache_bytes=args.draft_cache_mib*MiB,io_bytes=8*MiB,kv_bytes=8*MiB,
      workspace_bytes=64*MiB,load_tokenizer=True,max_tensor_bytes=16*MiB)
     draft_model=AirLLMNextLlamaModel(args.draft_dir,next_config=draft_cfg)
     draft=ReencodedModelDraft(draft_model,tokenizer)
    model=construct(root,current)
    if draft_model is not None and mx.metal.is_available():mx.set_memory_limit(current.mlx_memory_bytes+draft_cfg.mlx_memory_bytes)
    row=measure_generation(model,ids[case],args.tokens,draft,depth,adaptive)
    row.update(kind='speculation',case=case,repeat=repeat,variant=draft_name,depth=depth,
     adaptive=adaptive,prefetch=prefetch,spec_prefetch=spec,target_options=current.as_dict(),
     tokens_match_baseline=row['tokens']==baselines[case])
    if draft_model is not None:
     row['draft_options']=draft_cfg.as_dict();row['draft_metrics']=draft_model.metrics();row['reencoding']=dict(draft.stats)
    if not row['tokens_match_baseline']:failures.append(dict(case=case,variant=draft_name,depth=depth,reason='token mismatch'))
    record(row);close(model)
    if draft_model is not None:close(draft_model)
 save()
 if not report['source_unchanged']:raise SystemExit('Source changed during campaign')
 if failures:raise SystemExit('Numerical mismatch; evidence preserved, no tolerance changed')


if __name__=='__main__':main()
