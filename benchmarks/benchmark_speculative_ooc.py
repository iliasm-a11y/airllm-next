#!/usr/bin/env python3
"""Measure actual miniature GLM block verification and bytes/accepted token.

All variants use one local random miniature checkpoint at float32 and identical
target budgets. No network downloads occur. The accurate draft is an independent
local instance of that SAME miniature checkpoint: a correctness/stress fixture,
not a claim that a useful small trained draft exists for GLM-5.3. Its compute,
reads and time are included. The wrong draft supplies zeros to force rollback.
OS page cache and compiled kernels are not flushed; only the software weight
caches are cold at each generation. Results cannot predict M5/SSD/full-model
throughput or natural-language quality.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
import numpy as np
import mlx.core as mx
from airllm.nextgen import AirLLMNextModel,NextConfig
from airllm.nextgen.storage import physical_read_bytes
from airllm.nextgen.tuning import telemetry


def digest_files(root):
    result={}
    for path in sorted(root.glob('*.safetensors')):
        digest=sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):digest.update(block)
        result[path.name]={'sha256':digest.hexdigest(),'bytes':path.stat().st_size}
    result['config.json']={'sha256':sha256((root/'config.json').read_bytes()).hexdigest()}
    return result


def run(checkpoint,options,variant,depth,tokens,repeat,adaptive):
    init_start=perf_counter()
    target=AirLLMNextModel(checkpoint,next_config=options)
    target_startup=perf_counter()-init_start
    draft=None
    try:
        if variant=='same_checkpoint_local_draft':
            draft=AirLLMNextModel(checkpoint,next_config=replace(options,cache_bytes=8*1024**2,
                cache_dir=str(Path(options.cache_dir)/'draft')))
            proposal=draft
        elif variant=='wrong_zero_draft':proposal=lambda prefix,count:[0]*count
        else:proposal=None
        # New instances have no tensor cache entries; metadata files may be warm.
        assert target.reader.cache.bytes==0
        if draft is not None:assert draft.reader.cache.bytes==0
        begin_physical=physical_read_bytes()
        start=perf_counter()
        generated=np.asarray(target.generate([[1,8,9]],max_new_tokens=tokens,eos_token_id=[],
            draft_model=proposal,speculative_depth=depth,speculative_adaptive=adaptive))
        elapsed=perf_counter()-start
        end_physical=physical_read_bytes()
        actual=generated.shape[1]-3
        metrics=target.metrics()
        target_bytes=target.checkpoint.stats['logical_bytes']
        draft_bytes=draft.checkpoint.stats['logical_bytes'] if draft else 0
        return dict(variant=variant,depth=depth if proposal is not None else 0,
            adaptive=adaptive if proposal is not None else False,repeat=repeat,
            tokens=generated.tolist()[0],accepted_output_tokens=actual,
            generation_wall_seconds=elapsed,target_startup_seconds=target_startup,
            total_tokens_per_second_including_prefill=actual/elapsed,
            target_logical_read_bytes=target_bytes,draft_logical_read_bytes=draft_bytes,
            total_logical_read_bytes=target_bytes+draft_bytes,
            target_logical_bytes_per_accepted_token=target_bytes/actual,
            total_logical_bytes_per_accepted_token=(target_bytes+draft_bytes)/actual,
            physical_process_read_bytes=end_physical-begin_physical
                if begin_physical is not None and end_physical is not None else None,
            target_cache_peak_bytes=target.reader.cache.peak_bytes,
            target_io_peak_bytes=target.reader.io.peak_bytes,
            target_software_cache_cold_at_generation=True,
            draft_software_cache_cold_at_generation=True if draft else None,
            cache_state='OS page cache not flushed; software tensor caches cold; metadata/kernel caches may be warm',
            metrics=metrics)
    finally:
        if draft is not None:draft.close(save_profile=False)
        target.close(save_profile=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'fixtures/glm-mini')
    parser.add_argument('--output',type=Path,default=ROOT/'evidence-ooc/speculative.json')
    parser.add_argument('--tokens',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--target-cache-bytes',type=int,default=4096)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--timing-isolation-note',default='No external isolation enforced; concurrent host workloads may affect timing')
    args=parser.parse_args()
    if args.tokens<4 or args.repeats<1:parser.error('Use at least 4 tokens and one repeat')
    if args.device=='gpu' and not mx.metal.is_available():parser.error('Requested Metal GPU is unavailable')
    checkpoint=args.checkpoint.resolve()
    mx.set_default_device(getattr(mx,args.device))
    records=[]
    with tempfile.TemporaryDirectory(prefix='airllm-speculative-') as temporary:
        options=NextConfig(cache_bytes=args.target_cache_bytes,io_bytes=512*1024,
            kv_bytes=2*1024**2,workspace_bytes=16*1024**2,reserve_bytes=0,
            max_tensor_bytes=512*1024,max_context=256,prefill_chunk=16,row_chunk=32,
            device=args.device,dtype='float32',prefetch=False,load_tokenizer=False,
            cache_dir=temporary,use_saved_tuning=False)
        cases=[('autoregressive',0,False)]
        cases += [(variant,depth,False) for variant in
            ('same_checkpoint_local_draft','wrong_zero_draft') for depth in (1,2,4,8)]
        cases += [('wrong_zero_draft',4,True)]
        for repeat in range(args.repeats):
            order=list(cases);random.Random(20261022+repeat).shuffle(order)
            for variant,depth,adaptive in order:
                print(f'{repeat+1}/{args.repeats} {variant} depth={depth} adaptive={adaptive}',flush=True)
                record=run(checkpoint,options,variant,depth,args.tokens,repeat,adaptive)
                records.append(record)
    baseline=[r for r in records if r['variant']=='autoregressive']
    expected=baseline[0]['tokens']
    for record in records:
        record['exact_token_sequence_equal_to_autoregressive']=record['tokens']==expected
        if not record['exact_token_sequence_equal_to_autoregressive']:
            raise AssertionError('Speculative output differs from target autoregressive sequence')
        if record['target_cache_peak_bytes']>options.cache_bytes:
            raise AssertionError('Target cache exceeded its budget')
        if record['target_io_peak_bytes']>options.io_bytes:
            raise AssertionError('Target I/O exceeded its budget')
    summaries=[]
    base_bytes=statistics.median(r['target_logical_read_bytes'] for r in baseline)
    base_seconds=statistics.median(r['generation_wall_seconds'] for r in baseline)
    for variant,depth,adaptive in cases:
        group=[r for r in records if r['variant']==variant and r['depth']==depth and r['adaptive']==adaptive]
        seconds=statistics.median(r['generation_wall_seconds'] for r in group)
        target_bytes=statistics.median(r['target_logical_read_bytes'] for r in group)
        total_bytes=statistics.median(r['total_logical_read_bytes'] for r in group)
        summaries.append(dict(variant=variant,depth=depth,adaptive=adaptive,
            median_generation_seconds=seconds,measured_wall_speedup_vs_autoregressive=base_seconds/seconds,
            median_target_logical_read_bytes=target_bytes,
            median_total_logical_read_bytes=total_bytes,
            target_read_reduction_factor=base_bytes/target_bytes,
            total_read_reduction_factor=base_bytes/total_bytes,
            all_outputs_equal=True))
    output=dict(schema_version=1,scope='Actual random miniature GLM DSA float32 inference only; not full GLM-5.3, linguistic quality, external SSD, or M5 performance',
        platform=platform.platform(),machine=platform.machine(),python=sys.version,
        mlx_version=importlib.metadata.version('mlx'),device=args.device,
        metal_available=mx.metal.is_available(),hardware_m5_validated=False,
        checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        checkpoint_files=digest_files(checkpoint),target_options=options.as_dict(),
        draft_notes='Accurate draft is an independent copy of the same tiny checkpoint, cached in 8 MiB; intended to test verification, not representative of a smaller trained draft. Draft compute, reads, and proposal time included.',
        cache_protocol='Software weight caches cold for each generation; OS page cache and kernel cache not flushed. No physical SSD speed claim.',
        timing_isolation=args.timing_isolation_note,
        generation_timing_scope='Generation wall time includes draft inference and target verification, excludes construction of both already loaded models; target metadata/startup time is reported separately.',
        telemetry_end=telemetry(),tokens=args.tokens,repeats=args.repeats,
        summaries=summaries,raw_runs=records)
    output['target_options']['cache_dir']='temporary metadata directory'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
