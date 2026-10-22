#!/usr/bin/env python3
"""Trained Llama reference parity and isolated CPU/Metal timing, no downloads.

Run each backend in a fresh process. A cold software cache is distinguished
from the uncontrolled OS page cache. Measured model scope is recorded in JSON.
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
import time


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve() / 'air_llm'))
    import numpy as np
    import torch
    import psutil
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(1)
    root = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    prompts = ["What is the capital of France?", "Complète : un, deux, trois,", "Write a Python function that adds two numbers."]
    inputs = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                            tokenize=False, add_generation_prompt=True)
        inputs.append(tokenizer(text, return_tensors='pt')['input_ids'])
    start = time.perf_counter()
    if args.worker == 'reference':
        model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.float32, attn_implementation='eager').eval()
    elif args.worker == 'next':
        from airllm.nextgen import AirLLMNextLlamaModel, NextConfig
        model = AirLLMNextLlamaModel(root, next_config=NextConfig(
            device=args.device, dtype='float32', cache_bytes=args.cache_mib * 1024**2,
            io_bytes=32 * 1024**2, kv_bytes=32 * 1024**2, workspace_bytes=128 * 1024**2,
            reserve_bytes=512 * 1024**2, max_tensor_bytes=32 * 1024**2, row_chunk=args.row_chunk,
            prefill_chunk=16, max_context=512, load_tokenizer=False,
            use_saved_tuning=False, cache_dir=args.metadata_dir, prefetch=False))
    else:
        from airllm.ultra import AirLLMUltraModel, UltraConfig
        model = AirLLMUltraModel(root, device='cpu' if args.device == 'cpu' else 'mps', dtype=torch.float32,
            ultra=UltraConfig(cache_bytes=args.cache_mib * 1024**2, device_cache_bytes=0,
                 prefetch_bytes=32 * 1024**2, max_group_bytes=256 * 1024**2,
                 reserve_bytes=0, cache_dir=args.metadata_dir, load_tokenizer=False, prefetch=False))
    construction = time.perf_counter() - start
    reports = []
    with torch.inference_mode():
        for prompt, ids in zip(prompts, inputs):
            current = ids if args.worker != 'ultra' or args.device == 'cpu' else ids.to('mps')
            begin = time.perf_counter()
            logits = model(current, use_cache=False, logits_to_keep=1).logits
            logits = logits.detach().cpu().numpy() if hasattr(logits, 'detach') else np.array(logits)
            first_forward = time.perf_counter() - begin
            runs = []
            for repeat in range(args.repeats):
                before = time.perf_counter()
                if args.worker == 'next':
                    output = model.generate(current, max_new_tokens=args.tokens, eos_token_id=[])
                else:
                    # EOS and padding share id=2 in SmolLM2. With EOS disabled,
                    # HF's inferred mask otherwise removes real chat markers.
                    # These are unpadded sequences: every input token is valid.
                    output = model.generate(current, attention_mask=torch.ones_like(current),
                                            max_new_tokens=args.tokens, eos_token_id=[], do_sample=False)
                output = output.tolist()[0]
                elapsed = time.perf_counter() - before
                metrics = model.metrics() if hasattr(model, 'metrics') else None
                generated = len(output) - ids.shape[1]
                runs.append({'repeat': repeat, 'token_ids': output, 'new_tokens': generated,
                             'seconds': elapsed, 'tokens_per_second_including_prefill': generated / elapsed,
                             'output_text': tokenizer.decode(output[ids.shape[1]:], skip_special_tokens=True),
                             'metrics': metrics})
            reports.append({'prompt': prompt, 'input_tokens': ids.shape[1],
                            'first_forward_seconds': first_forward, 'last_logits': logits.tolist(), 'runs': runs})
    if hasattr(model, 'close'):
        model.close()
    import resource
    result = {'backend': args.worker, 'device': args.device if args.worker != 'reference' else 'cpu',
              'construction_seconds': construction, 'cases': reports,
              'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'swap_used_bytes_end': psutil.swap_memory().used, 'python': sys.version,
              'platform': platform.platform(), 'torch': torch.__version__,
              'precision_environment': {'MLX_ENABLE_TF32': os.environ.get('MLX_ENABLE_TF32')},
              'cache_state': 'Software cache initially empty; first_forward precedes timed generations; OS cache uncontrolled',
              'scope': 'Full local trained checkpoint in model_dir; not GLM-5.3 or Qwen3.8',
              'physical_storage_read_bytes': None, 'energy_joules': None}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--source', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--baseline-source', help='Optional prior tree for the Torch Ultra comparison')
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cache-mib', type=int, default=128)
    p.add_argument('--row-chunk', type=int, default=2048)
    p.add_argument('--devices', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    p.add_argument('--reverse-order', action='store_true', help='Reverse measured backend order after the reference')
    p.add_argument('--worker', choices=['reference', 'next', 'ultra'], help=argparse.SUPPRESS)
    p.add_argument('--device', default='cpu', help=argparse.SUPPRESS)
    p.add_argument('--metadata-dir', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.tokens < 1 or args.repeats < 1 or args.cache_mib < 0 or args.row_chunk < 1:
        p.error('tokens/repeats must be positive, cache-mib nonnegative')
    if args.worker:
        return worker(args)
    import numpy as np
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    variants = [('reference', 'cpu', args.source)]
    for device in args.devices:
        variants += [('next', device, args.source)]
        if args.baseline_source:
            variants += [('ultra', device, args.baseline_source)]
    if args.reverse_order:
        variants = variants[:1] + list(reversed(variants[1:]))
    # Explicit subprocess isolation includes a fresh MLX environment before any
    # import. Do not run other benchmarks concurrently with this experiment.
    import tempfile
    for kind, device, source in variants:
        label = f'{kind}-{device}'
        target = output.with_name(output.stem + '-' + label + '.json')
        with tempfile.TemporaryDirectory(prefix='airllm-release-meta-') as meta:
            command = [sys.executable, __file__, '--worker', kind, '--device', device,
                '--source', source, '--model-dir', args.model_dir, '--output', str(target),
                '--tokens', str(args.tokens), '--repeats', str(args.repeats),
                '--cache-mib', str(args.cache_mib), '--row-chunk', str(args.row_chunk), '--metadata-dir', meta]
            result = subprocess.run(command, env=dict(os.environ, MLX_ENABLE_TF32='0',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), capture_output=True, text=True)
            target.with_suffix('.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f'{label} failed ({result.returncode}); see {target.with_suffix(".log")}')
        reports[label] = json.loads(target.read_text())
    reference = reports['reference-cpu']
    summary = {}
    for label, report in reports.items():
        differences, matches, close, times, ttfts, decodes = [], [], [], [], [], []
        for actual, expected in zip(report['cases'], reference['cases']):
            differences.append(float(np.max(np.abs(np.array(actual['last_logits']) - np.array(expected['last_logits'])))))
            close.append(bool(np.allclose(actual['last_logits'], expected['last_logits'], atol=1e-4, rtol=5e-4)))
            for run in actual['runs']:
                matches.append(run['token_ids'] == expected['runs'][0]['token_ids'])
                times.append(run['seconds'])
                if run['metrics'] and 'ttft_seconds' in run['metrics']['runtime']:
                    ttfts.append(run['metrics']['runtime']['ttft_seconds'])
                    decodes.append(run['metrics']['runtime']['decode_tokens_per_second'])
        summary[label] = {'all_generated_sequences_match_reference': all(matches),
            'last_logits_within_atol_1e4_rtol_5e4': all(close),
            'max_abs_last_logit_error': max(differences), 'median_generation_seconds': statistics.median(times),
            'p50_generation_seconds': float(np.percentile(times, 50)),
            'p95_generation_seconds': float(np.percentile(times, 95)),
            'median_ttft_seconds': statistics.median(ttfts) if ttfts else None,
            'median_decode_tokens_per_second': statistics.median(decodes) if decodes else None,
            'process_peak_rss_bytes': report['process_peak_rss_bytes']}
    weights = Path(args.model_dir) / 'model.safetensors'
    result = {'model_dir': str(Path(args.model_dir).resolve()),
              'weight_sha256': hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest() if weights.is_file() else None,
              'tokens_per_run': args.tokens, 'repeats_per_prompt': args.repeats,
              'cache_mib': args.cache_mib, 'row_chunk': args.row_chunk, 'summary': summary,
              'variant_order': [f'{kind}-{device}' for kind, device, _ in variants],
              'protocol': 'Sequential fresh subprocess variants, same trained weights and prompts, explicit all-ones attention mask for unpadded HF inputs, greedy decoding, float32 arithmetic, no prefetch; TF32=0; OS page cache uncontrolled',
              'full_GLM53_validated': False, 'full_Qwen38_validated': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not all(x['all_generated_sequences_match_reference'] and x['last_logits_within_atol_1e4_rtol_5e4'] for x in summary.values()):
        raise SystemExit('Generation parity failed; inspect raw variant reports')


if __name__ == '__main__':
    main()
