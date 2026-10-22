#!/usr/bin/env python3
"""Compare two source trees on exactly the same local miniature checkpoints.

CPU MLX regression benchmark. No downloads, OS cache purge, Metal or full model
claims. Uses isolated subprocesses and fresh metadata directories for each case.
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
from time import perf_counter


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve() / 'air_llm'))
    import numpy as np
    import mlx.core as mx
    from airllm.nextgen import AirLLMNextModel, NextConfig
    mx.set_default_device(mx.cpu)
    root = Path(args.checkpoint_root).resolve()
    reports = []
    for label, folder, sidecar in [
        ('tiny64', 'tiny64', None),
        ('affine_int4_tiny64', 'tiny64', 'int4'),
        ('affine_int8_tiny64', 'tiny64', 'int8'),
        ('larger128', 'larger128', None),
        ('fp8_tiny64', 'fp8_tiny64', None),
    ]:
        with tempfile.TemporaryDirectory(prefix='airllm-revision-meta-') as meta:
            cfg = NextConfig(cache_bytes=4*1024**2, io_bytes=4*1024**2,
                kv_bytes=8*1024**2, max_tensor_bytes=4*1024**2,
                workspace_bytes=16*1024**2, reserve_bytes=0,
                prefill_chunk=8, max_context=256, device='cpu', dtype='float32',
                load_tokenizer=False, use_saved_tuning=False, cache_dir=meta,
                quantized_dir=str(root/sidecar) if sidecar else None)
            start = perf_counter()
            model = AirLLMNextModel(root/folder, next_config=cfg)
            construction = perf_counter()-start
            ids = np.array([[1, 8, 9, 10, 11, 12, 13, 14]], np.int32)
            with model:
                start = perf_counter()
                logits = np.array(model(ids).logits)
                first_prefill = perf_counter()-start
                runs = []
                for _ in range(args.repeats):
                    tokens = model.generate(ids, max_new_tokens=args.tokens,
                                            eos_token_id=[]).tolist()[0]
                    runs.append({'tokens': tokens, 'metrics': model.metrics()})
                reports.append({'label': label, 'construction_seconds': construction,
                    'first_prefill_seconds': first_prefill,
                    'logits': logits.tolist(), 'runs': runs})
    result = {'source': str(Path(args.source).resolve()), 'platform': platform.platform(),
        'python': sys.version, 'mlx': mx.__version__, 'numpy': np.__version__,
        'compute_device': 'cpu', 'metal_validated': False,
        'checkpoint_root': str(root), 'cases': reports}
    Path(args.output).write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-source')
    p.add_argument('--candidate-source')
    p.add_argument('--checkpoint-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--reverse-order', action='store_true')
    p.add_argument('--source', help=argparse.SUPPRESS)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        p.error('tokens must be >=2 and repeats >=1')
    if args.worker:
        return worker(args)
    if not args.baseline_source or not args.candidate_source:
        p.error('both source trees are required')
    import numpy as np
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    results = {}
    order = [('baseline', args.baseline_source), ('candidate', args.candidate_source)]
    if args.reverse_order:
        order.reverse()
    for label, source in order:
        path = output.with_name(output.stem+'-'+label+'.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
            '--source', str(Path(source).resolve()), '--checkpoint-root', args.checkpoint_root,
            '--tokens', str(args.tokens), '--repeats', str(args.repeats), '--output', str(path)]
        print('running', label, flush=True)
        subprocess.run(command, check=True, env=env)
        results[label] = json.loads(path.read_text())
    comparisons = []
    for before, after in zip(results['baseline']['cases'], results['candidate']['cases']):
        b = statistics.median(r['metrics']['runtime']['decode_tokens_per_second'] for r in before['runs'])
        a = statistics.median(r['metrics']['runtime']['decode_tokens_per_second'] for r in after['runs'])
        comparisons.append({'case': before['label'], 'baseline_median_decode_tokens_per_second': b,
            'candidate_median_decode_tokens_per_second': a, 'speed_ratio': a/b,
            'max_abs_logit_change': float(np.abs(np.array(before['logits'])-np.array(after['logits'])).max()),
            'generated_tokens_identical': all(x['tokens']==y['tokens'] for x,y in zip(before['runs'], after['runs']))})
    root = Path(args.checkpoint_root).resolve()
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(root.rglob('*')) if f.is_file() and
              (f.suffix in ('.safetensors', '.npy') or f.name == 'config.json')}
    result = {'scope': 'Random miniature CPU MLX checkpoints only; no full GLM-5.3, Metal or SSD hardware claim.',
        'cache_condition': 'Fresh metadata and software cache per case; first prefill before repeated generation. OS cache not purged.',
        'order': [label for label, _ in order], 'tokens_per_run': args.tokens,
        'repeats': args.repeats, 'comparisons': comparisons,
        'checkpoint_sha256': hashes,
        'raw_results': {'baseline': output.with_name(output.stem+'-baseline.json').name,
                        'candidate': output.with_name(output.stem+'-candidate.json').name}}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
