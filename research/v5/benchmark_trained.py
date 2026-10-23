#!/usr/bin/env python3
"""Isolated trained-checkpoint ablations: no downloads, synthetic latency, or GPU claims.

The baseline is a separate source tree. All variants use the same checkpoint,
precision, budgets, prompts and token count. Software-cold initial forwards and
subsequent warm generations are separate; the OS page cache is uncontrolled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

PROMPTS = ["What is the capital of France?", "Complète : un, deux, trois,", "Write a Python function that adds two numbers."]


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity(source):
    result = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'head': result.stdout.strip() if result.returncode == 0 else None,
            'production_sha256': {str(p.relative_to(source)): file_sha(p)
                for p in sorted((Path(source) / 'air_llm/airllm/nextgen').glob('*.py'))}}


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve() / 'air_llm'))
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    root = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    inputs = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
              tokenize=True, add_generation_prompt=True, return_dict=False) for prompt in PROMPTS]
    for i, ids in enumerate(inputs):
        if hasattr(ids, 'input_ids'): inputs[i] = ids.input_ids
    if args.worker == 'reference':
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(1)
        begin = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True,
                trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()
        construction = perf_counter() - begin
    else:
        import mlx.core as mx
        from airllm.nextgen import AirLLMNextLlamaModel, NextConfig
        from airllm.nextgen.storage import physical_read_bytes
        mx.set_default_device(mx.cpu)
        options = dict(device='cpu', dtype='float32', cache_bytes=args.cache_mib*1024**2,
            io_bytes=32*1024**2, kv_bytes=32*1024**2, workspace_bytes=128*1024**2,
            reserve_bytes=512*1024**2, max_tensor_bytes=32*1024**2, row_chunk=2048,
            prefill_chunk=16, max_context=512, load_tokenizer=False, prefetch=False,
            use_saved_tuning=False, cache_dir=args.metadata_dir)
        if args.worker == 'tiles': options['output_tile_cache'] = True
        if args.worker == 'compact': options['compact_cache_bytes'] = args.compact_mib*1024**2
        begin = perf_counter()
        model = AirLLMNextLlamaModel(root, next_config=NextConfig(**options))
        construction = perf_counter() - begin
    rows=[]; logits_arrays={}; swap_start=psutil.swap_memory().used
    for index, (prompt, values) in enumerate(zip(PROMPTS, inputs)):
        if args.worker == 'reference':
            ids = torch.tensor([values], dtype=torch.long)
            with torch.inference_mode():
                begin = perf_counter()
                logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False,
                               logits_to_keep=1).logits.detach().cpu().numpy()
                first_forward = perf_counter() - begin
        else:
            ids = np.array([values], dtype=np.int32)
            before = dict(model.checkpoint.stats)
            physical = physical_read_bytes()
            begin = perf_counter()
            logits = np.array(model(ids, use_cache=False, logits_to_keep=1).logits)
            first_forward = perf_counter() - begin
            physical_after = physical_read_bytes()
            cold_io = {key: value-before.get(key,0) for key,value in model.checkpoint.stats.items()
                       if isinstance(value,(float,int))}
            cold_io['physical_read_bytes'] = physical_after-physical if physical is not None and physical_after is not None else None
        logits_arrays[f'case_{index}'] = logits
        runs=[]
        for repeat in range(args.repeats):
            begin=perf_counter()
            if args.worker == 'reference':
                with torch.inference_mode():
                    output=model.generate(ids,attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.tokens,eos_token_id=[],do_sample=False).tolist()[0]
                metrics=None
            else:
                before=dict(model.checkpoint.stats)
                output=model.generate(ids,max_new_tokens=args.tokens,eos_token_id=[]).tolist()[0]
                metrics=model.metrics()
            seconds=perf_counter()-begin
            row={'repeat':repeat,'seconds':seconds,'token_ids':output,
                 'text':tokenizer.decode(output[len(values):],skip_special_tokens=True),
                 'new_tokens':len(output)-len(values),'metrics':metrics}
            if args.worker != 'reference':
                row['io_delta']={key:value-before.get(key,0) for key,value in model.checkpoint.stats.items()
                       if isinstance(value,(float,int))}
            runs.append(row)
        row={'prompt':prompt,'input_tokens':len(values),'first_forward_seconds':first_forward,'runs':runs,
             'first_forward_software_cache':'empty only for the first prompt of a fresh worker; later prompts retain cache entries'}
        if args.worker != 'reference': row['first_forward_io_delta']=cold_io
        rows.append(row)
    if hasattr(model,'close'):model.close()
    output=Path(args.output)
    np.savez_compressed(output.with_suffix('.npz'),**logits_arrays)
    result={'variant':args.worker,'source':source_identity(Path(args.source).resolve()),
        'construction_seconds':construction,'cases':rows,'platform':platform.platform(),
        'python':sys.version,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'swap_used_before_generation_bytes':swap_start,'swap_used_after_generation_bytes':psutil.swap_memory().used,
        'host_total_memory_bytes':psutil.virtual_memory().total,'cgroup_memory_max':None,
        'precision_environment':{'MLX_ENABLE_TF32':os.getenv('MLX_ENABLE_TF32')},
        'cache_state':'One fresh process per variant, fresh software cache at construction; each prompt has one forward before timed repeats; OS page cache uncontrolled',
        'checkpoint_payload_exceeds_physical_ram':False,'physical_external_ssd_measured':False,
        'logits_file':output.with_suffix('.npz').name,'logits_file_sha256':file_sha(output.with_suffix('.npz'))}
    cgroup=Path('/sys/fs/cgroup/memory.max')
    if cgroup.exists():result['cgroup_memory_max']=cgroup.read_text().strip()
    output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--candidate-source',required=True)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tokens',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--cache-mib',type=int,default=640)
    parser.add_argument('--compact-mib',type=int,default=128)
    parser.add_argument('--variants',nargs='+',choices=['reference','baseline','default','tiles','compact'],
                        default=['reference','baseline','default','tiles','compact'])
    parser.add_argument('--reverse-order',action='store_true')
    parser.add_argument('--worker',choices=['reference','baseline','default','tiles','compact'],help=argparse.SUPPRESS)
    parser.add_argument('--source',help=argparse.SUPPRESS)
    parser.add_argument('--metadata-dir',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.tokens<1 or args.repeats<1 or args.cache_mib<0 or not 0<=args.compact_mib<=args.cache_mib:
        parser.error('Invalid count or memory budget')
    if args.worker:return worker(args)
    if 'reference' not in args.variants or 'baseline' not in args.variants:
        parser.error('Reference and baseline are mandatory comparison controls')
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    reports={};paths={}
    variants=args.variants[:]
    if args.reverse_order:variants=['reference']+list(reversed([x for x in variants if x!='reference']))
    for variant in variants:
        source=args.baseline_source if variant=='baseline' else args.candidate_source
        path=output.with_name(output.stem+'-'+variant+'.json')
        print('running',variant,flush=True)
        with tempfile.TemporaryDirectory(prefix='airllm-v5-trained-') as meta:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',variant,
                '--source',source,'--baseline-source',args.baseline_source,'--candidate-source',args.candidate_source,
                '--model-dir',args.model_dir,'--output',str(path),'--tokens',str(args.tokens),'--repeats',str(args.repeats),
                '--cache-mib',str(args.cache_mib),'--compact-mib',str(args.compact_mib),'--metadata-dir',meta]
            result=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,MLX_ENABLE_TF32='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            path.with_suffix('.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f'{variant} failed; see {path.with_suffix(".log")}')
        reports[variant]=json.loads(path.read_text());paths[variant]=path
    import numpy as np
    reference=reports['reference'];ref_logits=np.load(paths['reference'].with_suffix('.npz'))
    summary={}
    for label,report in reports.items():
        observed=np.load(paths[label].with_suffix('.npz'));times=[];decode=[];ttft=[];logical=[];encoded=[]
        same=[];close=[];errors=[]
        for i,(actual,expected) in enumerate(zip(report['cases'],reference['cases'])):
            a=observed[f'case_{i}'];b=ref_logits[f'case_{i}'];close.append(bool(np.allclose(a,b,atol=1e-4,rtol=5e-4)));errors.append(float(np.max(np.abs(a-b))))
            for run in actual['runs']:
                same.append(run['token_ids']==expected['runs'][0]['token_ids']);times.append(run['seconds'])
                if run['metrics']:
                    m=run['metrics']['runtime'];decode.append(m['decode_tokens_per_second']);ttft.append(m['ttft_seconds'])
                    logical.append(m['decode_logical_bytes_per_token']);encoded.append(m['decode_encoded_file_bytes_per_token'])
        summary[label]={'all_sequences_match_reference':all(same),'logits_within_atol_1e4_rtol_5e4':all(close),
            'max_abs_logit_error':max(errors),'median_generation_seconds':statistics.median(times),
            'median_decode_tokens_per_second':statistics.median(decode) if decode else None,
            'median_ttft_seconds':statistics.median(ttft) if ttft else None,
            'median_decode_logical_bytes_per_token':statistics.median(logical) if logical else None,
            'median_decode_encoded_file_bytes_per_token':statistics.median(encoded) if encoded else None,
            'process_peak_rss_bytes':report['process_peak_rss_bytes'],'raw_report':paths[label].name}
    baseline=summary['baseline']['median_generation_seconds']
    for label,row in summary.items():row['generation_speedup_vs_v4_baseline']=baseline/row['median_generation_seconds']
    weights=Path(args.model_dir)/'model.safetensors'
    result={'model_id':'HuggingFaceTB/SmolLM2-135M-Instruct','revision':'12fd25f77366fa6b3b4b768ec3050bf629380bac',
        'checkpoint_sha256':file_sha(weights),'weight_file_bytes':weights.stat().st_size,
        'protocol':'Sequential isolated subprocesses; same trained checkpoint, FP32, budgets, prompts and greedy decoding at the requested fixed token count. Software initial forward measured separately. OS cache uncontrolled. No concurrent benchmark load intended.',
        'tokens_per_run':args.tokens,'repeats_per_prompt':args.repeats,'cache_mib':args.cache_mib,'compact_mib':args.compact_mib,
        'variant_order':variants,'summary':summary,'full_GLM53_generated':False,'full_Qwen38_generated':False,
        'checkpoint_exceeds_physical_ram':False,'metal_measured':False,'external_ssd_measured':False}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not all(r['all_sequences_match_reference'] and r['logits_within_atol_1e4_rtol_5e4'] for r in summary.values()):
        raise SystemExit('Trained generation or numerical parity failed')


if __name__=='__main__':main()
