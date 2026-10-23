#!/usr/bin/env python3
"""Real autoregressive Qwen3 MoE generation, never known-token verification.

Requires a complete pinned trained checkpoint. This worker does not download
or load all weights and does not claim source-BF16 equivalence of quantization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import threading
from time import perf_counter, process_time

import numpy as np
import psutil

from airllm.nextgen.config import NextConfig, GiB, MiB
from airllm.nextgen.physical_io import physical_read_bytes, physical_read_counter_info
from airllm.nextgen.qwen3_mlx_runtime import ExperimentalQwen3MLXModel


def verify_complete_checkpoint(root, manifest_path):
    """All five immutable-revision shard hashes must pass before any forward."""
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('complete') is not True or
            manifest.get('model_id') != 'Qwen/Qwen3-30B-A3B-MLX-6bit' or
            manifest.get('revision') != 'b92024737fc38f0bce249aaefb7611dd22c38b58'):
        raise ValueError('A complete pinned official Qwen3 MLX download manifest is required')
    files = {item['name']: item for item in manifest['files']}
    expected_names = {f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)}
    if {name for name in files if name.endswith('.safetensors')} != expected_names:
        raise ValueError('Manifest must contain exactly all five model shards')
    if sum(files[name]['size'] for name in expected_names) != 23853636272:
        raise ValueError('Unexpected complete weight byte size')
    verified = []
    for name in sorted(expected_names):
        item = files[name]
        path = root/name
        if (item.get('verified_lfs_sha256') is not True or
                item.get('sha256') != item.get('expected_lfs_sha256') or
                not path.is_file() or path.stat().st_size != item['size']):
            raise ValueError(f'Incomplete or unverified shard: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4*MiB), b''):
                digest.update(block)
        if digest.hexdigest() != item['expected_lfs_sha256']:
            raise ValueError(f'Full trained shard SHA256 mismatch: {name}')
        verified.append(dict(name=name, size=item['size'], sha256=digest.hexdigest()))
    return verified


class GuardedAttention:
    def __init__(self, inner, guard):
        self.inner, self.guard = inner, guard

    def __call__(self, *args, **kwargs):
        self.guard()
        result = self.inner(*args, **kwargs)
        self.guard()
        return result


def run(args):
    output = dict(status='started', architecture='qwen3_moe',
        model_dir=str(args.model_dir.resolve()), requested_new_tokens=args.tokens,
        prompt=args.prompt, device=args.device, generated_ids=[], quality_status='not_evaluated',
        quantization_note='Existing checkpoint affine quantization; not lossless source BF16 equivalence.',
        physical_counter=physical_read_counter_info(), gpu_compute_seconds=None,
        gpu_compute_note='No GPU command timestamp profiler is attached; wall time includes required synchronization.',
        energy_joules=None, temperature_celsius=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(output, indent=2, default=str)+'\n')

    stop = threading.Event()
    samples = dict(rss_peak_bytes=0, minimum_available_bytes=psutil.virtual_memory().available,
                   swap_growth_peak_bytes=0)
    proc = psutil.Process()
    swap_start = psutil.swap_memory().used

    def sample():
        while not stop.is_set():
            samples['rss_peak_bytes'] = max(samples['rss_peak_bytes'], proc.memory_info().rss)
            samples['minimum_available_bytes'] = min(samples['minimum_available_bytes'], psutil.virtual_memory().available)
            samples['swap_growth_peak_bytes'] = max(samples['swap_growth_peak_bytes'], max(0, psutil.swap_memory().used-swap_start))
            stop.wait(.1)

    def guard():
        if psutil.virtual_memory().available < args.reserve_gib*GiB:
            raise MemoryError('Requested macOS memory reserve reached')
        if max(0, psutil.swap_memory().used-swap_start) > args.max_swap_growth_mib*MiB:
            raise MemoryError('Swap growth cap reached')

    monitor = threading.Thread(target=sample, daemon=True)
    model = None
    started, cpu_started = perf_counter(), process_time()
    physical_start = physical_read_bytes()
    monitor.start()
    save()
    try:
        guard()
        validation_start = perf_counter()
        output['validated_shards'] = verify_complete_checkpoint(args.model_dir, args.download_manifest)
        output['checkpoint_validation_seconds'] = perf_counter()-validation_start
        output['checkpoint_validation_note'] = 'Full SHA pass before this trial; its reads precede generation and can warm the OS cache.'
        load_start = perf_counter()
        options = NextConfig(cache_bytes=int(args.cache_gib*GiB), io_bytes=256*MiB,
            kv_bytes=128*MiB, workspace_bytes=512*MiB, reserve_bytes=int(args.reserve_gib*GiB),
            max_tensor_bytes=args.max_tensor_mib*MiB, io_workers=args.io_workers, row_chunk=4096,
            prefill_chunk=4, max_context=args.max_context, device=args.device,
            dtype=args.dtype, policy=args.policy, prefetch=args.prefetch,
            output_tile_cache=True, output_prefetch=args.output_prefetch,
            cache_dir=str(args.cache_dir), use_saved_tuning=False, io_mode=args.io_mode)
        from dataclasses import replace
        options = replace(options, io_bytes=args.io_mib*MiB)
        output['options'] = options.as_dict()
        model = ExperimentalQwen3MLXModel(args.model_dir, next_config=options)
        model.attention = [GuardedAttention(inner, guard) for inner in model.attention]
        loaded = perf_counter()
        output['load_seconds'] = loaded-load_start
        prompt = model.tokenizer.apply_chat_template([dict(role='user', content=args.prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids)+args.tokens > args.max_context:
            raise ValueError('Prompt plus generation exceeds max_context')
        output['prompt_ids'] = prompt_ids
        output['prompt_tokens'] = len(prompt_ids)
        cache = model.new_session()
        prefill_start = perf_counter()
        result = model.forward(np.asarray([prompt_ids]), past_key_values=cache, logits_to_keep=1)
        model.mx.eval(result.logits)
        guard()
        token = int(model.mx.argmax(result.logits[0, -1]).item())
        first = perf_counter()
        output['prefill_seconds'] = first-prefill_start
        output['time_to_first_token_excluding_load_seconds'] = first-loaded
        output['time_to_first_token_including_load_seconds'] = first-started
        output['generated_ids'].append(token)
        save()
        decode_started = perf_counter()
        eos = model.tokenizer.eos_token_id
        eos_set = set(eos if isinstance(eos, list) else [eos])
        while len(output['generated_ids']) < args.tokens and token not in eos_set:
            guard()
            step_start = perf_counter()
            result = model.forward(np.asarray([[token]]), past_key_values=cache, logits_to_keep=1)
            model.mx.eval(result.logits)
            token = int(model.mx.argmax(result.logits[0, -1]).item())
            output['generated_ids'].append(token)
            output.setdefault('decode_step_seconds', []).append(perf_counter()-step_start)
            if len(output['generated_ids']) % 4 == 0:
                save()
        # Charge outstanding exact/speculative work to the request that created
        # it. Capturing counters before this drain would omit actual I/O cost.
        model.reader.drain()
        guard()
        finished = perf_counter()
        output['decode_seconds'] = finished-decode_started
        output['decode_generated_tokens'] = len(output['generated_ids'])-1
        output['decode_tokens_per_second'] = output['decode_generated_tokens']/output['decode_seconds'] if output['decode_generated_tokens'] else None
        output['generation_seconds_excluding_load'] = finished-loaded
        output['end_to_end_seconds'] = finished-started
        output['generation_tokens_per_second_excluding_load'] = len(output['generated_ids'])/(finished-loaded)
        output['end_to_end_tokens_per_second'] = len(output['generated_ids'])/(finished-started)
        output['generated_text'] = model.tokenizer.decode(output['generated_ids'])
        output['stopped_on_eos'] = token in eos_set
        output['logical_read_bytes'] = int(model.checkpoint.stats['logical_bytes'])
        output['reader_stats'] = dict(model.reader.stats)
        output['runtime_stats'] = dict(model.stats)
        output['mlx_peak_bytes'] = int(model.mx.get_peak_memory())
        output['status'] = 'complete'
    except BaseException as exc:
        output.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop.set()
        monitor.join(timeout=2)
        output.update(samples)
        output['cpu_seconds'] = process_time()-cpu_started
        output['max_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        physical_end = physical_read_bytes()
        output['physical_read_bytes_delta'] = (physical_end-physical_start
            if physical_end is not None and physical_start is not None else None)
        close_error = None
        if model is not None:
            try:
                model.close(save_profile=False)
            except BaseException as exc:
                close_error = exc
                output['close_error'] = str(exc)
                if output['status'] == 'complete':
                    output.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save()
        if close_error is not None and output.get('error') == str(close_error):
            raise close_error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cache-dir', required=True, type=Path)
    p.add_argument('--download-manifest', required=True, type=Path)
    p.add_argument('--prompt', default='Explain in two sentences why the sky is blue.')
    p.add_argument('--tokens', type=int, default=4)
    p.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
    p.add_argument('--dtype', choices=('bfloat16', 'float32', 'float16'), default='bfloat16')
    p.add_argument('--cache-gib', type=float, default=2)
    p.add_argument('--reserve-gib', type=float, default=1.5)
    p.add_argument('--max-swap-growth-mib', type=int, default=256)
    p.add_argument('--max-context', type=int, default=512)
    p.add_argument('--io-workers', type=int, default=2)
    p.add_argument('--max-tensor-mib', type=int, default=32)
    p.add_argument('--io-mib', type=int, default=64)
    p.add_argument('--policy', default='slru', choices=('slru', 'lru', 'cost', 'adaptive', 'expert', 'shared', 'hybrid'))
    p.add_argument('--io-mode', choices=('pread', 'pread_nocache'), default='pread')
    p.add_argument('--prefetch', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--output-prefetch', action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    if args.tokens < 1 or args.reserve_gib < 1.5 or not 0 <= args.cache_gib <= 4:
        p.error('tokens >=1, reserve >=1.5GiB and cache <=4GiB are required')
    run(args)


if __name__ == '__main__':
    main()
