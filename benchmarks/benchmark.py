#!/usr/bin/env python3
"""Reproducible real timings on deterministic small models, no remote downloads.

Runs each engine in a fresh Python process, reports import and construction
separately, saves per-token latencies/OS counters. NOT a giant-model projection.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))


def os_counters():
    try:
        return {k: int(v) for k, v in (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def current_rss():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def worker(args):
    start = time.perf_counter()
    import psutil
    import torch
    torch.set_num_threads(args.threads)
    # Both paths import the same public package, so import costs are comparable.
    from airllm.airllm_base import AirLLMBaseModel
    from airllm.ultra import AirLLMUltraModel, UltraConfig
    imported = time.perf_counter()
    before = os_counters()
    construct_start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        if args.engine == 'upstream':
            model = AirLLMBaseModel(args.model, device='cpu', dtype=torch.float32,
                                    prefetching=True,
                                    layer_shards_saving_path=args.split_dir)
        else:
            model = AirLLMUltraModel(args.model, device='cpu', dtype=torch.float32,
                                     ultra=UltraConfig(cache_bytes=args.cache_mib * 1024**2,
                                                       prefetch_bytes=4*1024**2, reserve_bytes=0,
                                                       cache_dir=args.meta_dir, load_tokenizer=False))
    constructed = time.perf_counter()
    ids = torch.tensor([[1, 11, 12, 13, 14, 15, 16, 17]])
    current, kv = ids, None
    latencies, tokens = [], []
    with torch.inference_mode():
        for _ in range(args.tokens):
            t = time.perf_counter()
            out = model(current, use_cache=True, past_key_values=kv, logits_to_keep=1)
            current = out.logits[:, -1].argmax(-1)[:, None]
            tokens.append(current.item())
            kv = out.past_key_values
            latencies.append(time.perf_counter() - t)
    after = os_counters()
    result = {'engine': args.engine, 'cache_mib': args.cache_mib,
              'import_seconds': imported-start, 'construction_seconds': constructed-construct_start,
              'first_forward_seconds': latencies[0],
              'ttft_from_worker_start_seconds': constructed-start+latencies[0],
              'ttft_excluding_import_seconds': constructed-construct_start+latencies[0],
              'decode_tokens_per_second': (len(latencies)-1)/sum(latencies[1:]),
              'token_seconds': latencies, 'generated_token_ids': tokens,
              'rss_after_bytes': current_rss(),
              'os_read_bytes_delta': after.get('read_bytes', 0)-before.get('read_bytes', 0) if after and before else None,
              'os_write_bytes_delta': after.get('write_bytes', 0)-before.get('write_bytes', 0) if after and before else None,
              'torch': torch.__version__}
    import resource
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if args.engine == 'ultra':
        result['metrics'] = model.metrics()
        model.close()
    if args.split_dir:
        result['split_output_bytes'] = sum(p.stat().st_size for p in Path(args.split_dir).rglob('*') if p.is_file())
    print(json.dumps(result))


def create_model(path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(521)
    cfg = LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=768,
                       num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                       max_position_embeddings=512, bos_token_id=1, eos_token_id=2,
                       pad_token_id=0, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    model.save_pretrained(path, max_shard_size='4MB')
    tok = Tokenizer(WordLevel({'[UNK]': 0, 'hello': 1}, unk_token='[UNK]'))
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]').save_pretrained(path)
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'weight_bytes': sum(p.stat().st_size for p in Path(path).glob('*.safetensors')),
            'config': cfg.to_dict(), 'weights': 'deterministic randomly initialized; not trained'}


def main(args):
    import torch
    torch.set_num_threads(args.threads)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    logs = []
    with tempfile.TemporaryDirectory(prefix='airllm-bench-') as temp:
        temp = Path(temp)
        model_dir = temp / 'model'
        info = create_model(model_dir)
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
        # Stable engine-specific paths make all repeats reproducible. First use
        # includes preprocessing; reused runs retain split files/manifests only.
        specs = [('upstream_first', 'upstream', 0), ('upstream_reopen', 'upstream', 0),
                 ('ultra_first_cache0', 'ultra', 0), ('ultra_reopen_cache0', 'ultra', 0),
                 ('ultra_cache64', 'ultra', 64)]
        for trial in range(args.repeats):
            split = temp / f'split-{trial}'
            meta = temp / f'meta-{trial}'
            for label, engine, cache in specs:
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--engine', engine,
                       '--model', str(model_dir), '--tokens', str(args.tokens),
                       '--threads', str(args.threads), '--cache-mib', str(cache),
                       '--meta-dir', str(meta), '--split-dir', str(split)]
                proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=240)
                logs.append({'label': label, 'trial': trial, 'stderr': proc.stderr, 'returncode': proc.returncode})
                if proc.returncode:
                    raise RuntimeError(f'{label} failed:\n{proc.stderr}\n{proc.stdout}')
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row.update(label=label, trial=trial)
                results.append(row)
                print(f'{label} trial={trial} construct={row["construction_seconds"]:.4f}s '
                      f'TTFT={row["ttft_excluding_import_seconds"]:.4f}s '
                      f'decode={row["decode_tokens_per_second"]:.2f}tok/s', flush=True)
        assert all(r['generated_token_ids'] == results[0]['generated_token_ids'] for r in results)
    summary = {}
    for label, _, _ in specs:
        rows = [r for r in results if r['label'] == label]
        summary[label] = {k: statistics.median(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                          for k in ('construction_seconds', 'ttft_excluding_import_seconds',
                                    'ttft_from_worker_start_seconds', 'decode_tokens_per_second',
                                    'peak_rss_bytes', 'os_write_bytes_delta')}
    report = {'platform': platform.platform(), 'python': sys.version, 'threads': args.threads,
              'repeats': args.repeats, 'new_tokens_per_run': args.tokens,
              'model': info, 'all_generated_sequences_match': True,
              'limitations': ['Small synthetic dense Llama model on CPU; not representative giant-model tok/s.',
                              'No OS page-cache flush: fresh processes are not cold physical disk runs.',
                              'OS I/O counters differ from logical bytes and can be zero on warm cache.',
                              'Baseline uses the untouched v4.0.0 AirLLMBaseModel path, same dependencies.'],
              'summary_medians': summary, 'runs': results}
    output.write_text(json.dumps(report, indent=2) + '\n')
    output.with_suffix('.logs.json').write_text(json.dumps(logs, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(ROOT / 'benchmarks/results-cpu.json'))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--tokens', type=int, default=8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--engine', choices=['upstream', 'ultra'])
    p.add_argument('--model')
    p.add_argument('--cache-mib', type=int, default=0)
    p.add_argument('--meta-dir')
    p.add_argument('--split-dir')
    a = p.parse_args()
    worker(a) if a.worker else main(a)
