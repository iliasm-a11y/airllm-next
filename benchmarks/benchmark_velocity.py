#!/usr/bin/env python3
"""Actual greedy generation, shuffled paired ablations, equal total budgets.

No preknown output tokens, no implicit download and no cache purge. The OS
process storage counter is not a per-file or NAND-bandwidth measurement.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
MiB = 1024**2
PROMPTS = [
    'What is the capital of France? Answer in one sentence.',
    'Complete the sequence: one, two, three, four, five,',
    'Return only Python code for a function add(a, b) that returns their sum.',
    'Repeat exactly this pattern several times: red blue green red blue green',
    'Explique en deux phrases pourquoi le ciel est bleu.',
    'List the first ten square numbers in ascending order.',
    ('Here is some context. A city has libraries, parks and trains. '*24)+
        'Explain in three sentences why libraries are useful.'
]


def identity():
    files = sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py'))+[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def numeric_delta(after, before):
    return {k:v-before.get(k, 0) for k,v in after.items() if type(v) in (int, float)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--draft-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','gpu'], default='gpu')
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--cases', type=int, nargs='+', default=[0,1,2,3,4,5])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--variants', nargs='+', default=['reference','prefetch','queued2','queued4'])
    p.add_argument('--cache-mib', type=int, default=256)
    p.add_argument('--io-mode', choices=['pread','pread_nocache'], default='pread')
    p.add_argument('--profile-io', action='store_true')
    p.add_argument('--max-context', type=int, default=768)
    args = p.parse_args()
    allowed = {'reference','prefetch','queued2','queued4','queued8','prompt4','draft-recompute','draft-incremental',
        'zero-copy','queued4-zero'}
    if (args.tokens<1 or args.repeats<1 or not args.cases or any(c not in range(len(PROMPTS)) for c in args.cases)
            or not set(args.variants)<=allowed or 'reference' not in args.variants):
        p.error('Positive counts, valid unique cases and variants with reference required')
    if len(set(args.cases)) != len(args.cases) or len(set(args.variants)) != len(args.variants):
        p.error('Duplicate cases/variants are not independent repetitions')
    os.environ.setdefault('MLX_ENABLE_TF32','0')
    import mlx.core as mx
    import psutil
    from transformers import AutoTokenizer
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel
    from airllm.nextgen.physical_io import physical_read_bytes, physical_read_counter_info
    mx.set_default_device(getattr(mx,args.device))
    source = identity()
    root = args.model_dir.resolve()
    prov = json.loads((root/'download-provenance.json').read_text())
    verification_start = time.perf_counter()
    for entry in prov['files']:
        f = root/entry['name']
        with f.open('rb') as stream:
            if f.stat().st_size != entry['bytes'] or hashlib.file_digest(stream,'sha256').hexdigest() != entry['sha256']:
                raise ValueError('Checkpoint identity mismatch: '+entry['name'])
    verification_seconds = time.perf_counter()-verification_start
    begin = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    prompts = {case:tokenizer.apply_chat_template([{'role':'user','content':PROMPTS[case]}],
        tokenize=True,add_generation_prompt=True,return_dict=False) for case in args.cases}
    tokenize_seconds = time.perf_counter()-begin
    records, failures = [], []
    report = dict(schema_version=1,kind='true_greedy_generation',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=source,source_sha256_scope='exact local files at measurement time; git commit recorded separately',
        checkpoint=prov,checkpoint_verification_seconds=verification_seconds,
        tokenizer_load_and_prompt_encoding_seconds=tokenize_seconds,
        host=platform.platform(),physical_memory_bytes=psutil.virtual_memory().total,
        libraries={n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','transformers','numpy','psutil']},
        os_cache='SHA256 verification warms checkpoint; not purged, uncontrolled OS cache',
        timing_scope='Each run starts with empty application weight/KV caches; generation wall includes prefill, draft and conversions. Startup and startup-inclusive throughput separate.',
        physical_counter=physical_read_counter_info(),gpu_time_seconds=None,
        gpu_time_unavailable_reason='No Metal command-buffer timing instrument attached; wall time is not GPU time',
        temperature_celsius=None,energy_joules=None,
        original_checkpoint_fits_physical_ram=sum(e['bytes'] for e in prov['files'] if e['name'].endswith('.safetensors'))<psutil.virtual_memory().total,
        eos_policy='stop at configured model EOS or requested ceiling; actual emitted token count',
        records=records,failures=failures)
    report['git_head_at_start'] = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    report['source_unchanged'] = True
    if any(v.startswith('draft-') for v in args.variants):
        if args.draft_dir is None:
            raise ValueError('Explicit local draft checkpoint required')
        draft_identity = {}
        for f in sorted(args.draft_dir.iterdir()):
            if f.is_file():
                with f.open('rb') as stream:
                    draft_identity[f.name] = dict(bytes=f.stat().st_size,
                        sha256=hashlib.file_digest(stream,'sha256').hexdigest())
        report['draft_checkpoint_local_identity'] = draft_identity
        report['draft_checkpoint_revision_provenance'] = 'See pinned official identity in evidence-multitoken/published/draft-provenance.json.gz; same local files must match.'
        with gzip.open(ROOT/'evidence-multitoken/published/draft-provenance.json.gz','rt') as stream:
            official = json.load(stream)
        for entry in official['files']:
            if draft_identity.get(entry['name']) != dict(bytes=entry['bytes'],sha256=entry['sha256']):
                raise ValueError('Draft differs from previously verified pinned official identity: '+entry['name'])
        report['draft_matches_pinned_official_identity'] = True
        report['draft_checkpoint_revision'] = official.get('revision')
    tasks = [(repeat,case,variant) for repeat in range(args.repeats) for case in args.cases for variant in args.variants]
    random.Random(20261023).shuffle(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report['source_unchanged'] = report['source_unchanged'] and source==identity()
        report['process_lifetime_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    with tempfile.TemporaryDirectory(prefix='velocity-') as td:
        cfg = NextConfig(device=args.device,dtype='float32',cache_bytes=args.cache_mib*MiB,
            io_bytes=32*MiB,kv_bytes=64*MiB,workspace_bytes=256*MiB,reserve_bytes=1536*MiB,
            max_tensor_bytes=32*MiB,row_chunk=2048,prefill_chunk=64,max_context=args.max_context,
            policy='adaptive',prefetch=False,spec_prefetch=False,load_tokenizer=False,
            cache_dir=td,use_saved_tuning=False,io_mode=args.io_mode,output_tile_cache=True,profile_io=args.profile_io)
        report['total_budgets'] = cfg.as_dict()
        for repeat,case,variant in tasks:
            draft = draft_model = model = None
            current = replace(cfg,prefetch=variant!='reference')
            group = {'queued2':2,'queued4':4,'queued8':8,'queued4-zero':4}.get(variant,1)
            if variant in ('zero-copy','queued4-zero'):
                current = replace(current,zero_copy_weights=True)
            if variant.startswith('draft-'):
                if args.draft_dir is None or args.cache_mib<=64:
                    raise ValueError('draft variants need --draft-dir and >64MiB total cache')
                current = replace(current,cache_bytes=(args.cache_mib-64)*MiB,
                    io_bytes=24*MiB,kv_bytes=48*MiB,workspace_bytes=192*MiB)
            output = None
            start = time.perf_counter()
            try:
                model = AirLLMNextGraniteMoeModel(root,next_config=current,expert_sync_group=group)
                if variant=='prompt4':
                    from airllm.nextgen.prompt_draft import PromptSpanDraft
                    draft = PromptSpanDraft(min_match=2,max_match=32)
                if variant.startswith('draft-'):
                    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
                    dc = replace(cfg,cache_bytes=64*MiB,io_bytes=8*MiB,kv_bytes=16*MiB,
                        workspace_bytes=64*MiB,load_tokenizer=True,prefetch=True)
                    draft_model = AirLLMNextLlamaModel(args.draft_dir,next_config=dc)
                    # This allocator limit is process-global: restore the sum
                    # of the two disjoint partitions after draft construction.
                    if mx.metal.is_available():
                        mx.set_memory_limit(current.mlx_memory_bytes+dc.mlx_memory_bytes)
                    if variant=='draft-recompute':
                        from airllm.nextgen.text_draft import ReencodedModelDraft
                        draft = ReencodedModelDraft(draft_model,tokenizer)
                    else:
                        from airllm.nextgen.incremental_draft import IncrementalReencodedModelDraft
                        draft = IncrementalReencodedModelDraft(draft_model,tokenizer)
                construction = time.perf_counter()-start
                before_io = dict(model.checkpoint.stats)
                disk_start = physical_read_bytes()
                swap_start = psutil.swap_memory().used
                cpu_start = time.process_time()
                start = time.perf_counter()
                if draft is None:
                    output = model.generate([prompts[case]],max_new_tokens=args.tokens)
                else:
                    output = model.generate_speculative([prompts[case]],draft,max_new_tokens=args.tokens,
                        max_depth=4,adaptive=False)
                values = output.tolist()[0]
                # Charge all actual asynchronous work to the generation that
                # requested it, including unused exact prefetch at its tail.
                model.reader.drain()
                if draft_model is not None:
                    draft_model.reader.drain()
                seconds = time.perf_counter()-start
                cpu_seconds = time.process_time()-cpu_start
                disk_end = physical_read_bytes()
                metrics = model.metrics()
                n = len(values)-len(prompts[case])
                row = dict(repeat=repeat,case=case,variant=variant,input_tokens=len(prompts[case]),
                    new_tokens=n,tokens=values,generated_text=tokenizer.decode(values[len(prompts[case]):],skip_special_tokens=True),
                    construction_seconds=construction,generation_seconds=seconds,
                    tokens_per_second=n/seconds,startup_inclusive_tokens_per_second=n/(seconds+construction),
                    cpu_process_seconds=cpu_seconds,swap_used_before_bytes=swap_start,
                    swap_used_after_bytes=psutil.swap_memory().used,
                    process_storage_read_bytes=disk_end-disk_start if disk_end is not None and disk_start is not None else None,
                    logical_io_delta=numeric_delta(model.checkpoint.stats,before_io),metrics=metrics,
                    sample_rss_bytes=psutil.Process().memory_info().rss,
                    available_system_bytes=psutil.virtual_memory().available)
                row['target_options'] = current.as_dict()
                if draft_model:
                    row['draft_metrics'] = draft_model.metrics()
                    row['draft_stats'] = dict(draft.stats)
                    row['draft_options'] = dc.as_dict()
                records.append(row);save()
                print(json.dumps({k:row[k] for k in ['repeat','case','variant','new_tokens','generation_seconds','tokens_per_second']}),flush=True)
            finally:
                for item in [model,draft_model]:
                    if item is not None:item.close(save_profile=False)
                del model,draft_model,draft,output
                mx.clear_cache()
    for repeat in range(args.repeats):
        for case in args.cases:
            pair = [r for r in records if r['repeat']==repeat and r['case']==case]
            baseline = next(r for r in pair if r['variant']=='reference')
            for row in pair:
                row['tokens_equal_reference'] = row['tokens']==baseline['tokens']
                row['speedup_vs_paired_reference'] = baseline['generation_seconds']/row['generation_seconds']
                if not row['tokens_equal_reference']:
                    failures.append(dict(repeat=repeat,case=case,variant=row['variant'],reason='token mismatch'))
    report['summary'] = {}
    for variant in args.variants:
        rows = [r for r in records if r['variant']==variant]
        report['summary'][variant] = dict(runs=len(rows),new_tokens=sum(r['new_tokens'] for r in rows),
            aggregate_tokens_per_second=sum(r['new_tokens'] for r in rows)/sum(r['generation_seconds'] for r in rows),
            median_paired_speedup=statistics.median(r['speedup_vs_paired_reference'] for r in rows),
            all_tokens_equal_reference=all(r['tokens_equal_reference'] for r in rows))
    save()
    if failures or not report['source_unchanged']:
        raise ValueError('Token divergence or source mutation; inspect retained report')


if __name__=='__main__':
    main()
