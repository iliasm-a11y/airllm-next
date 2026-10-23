#!/usr/bin/env python3
"""Real text conversations through native AirLLM GGUF, always supervised.

No generated code runs here. All user/model messages and partial outputs are
persisted, so interruption is evidence rather than a missing observation.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def identity():
    files = [*sorted((ROOT / 'air_llm/airllm/nextgen').glob('*.py')), Path(__file__)]
    versions = {}
    for package in ('mlx', 'numpy', 'tokenizers', 'transformers', 'psutil'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'platform': platform.platform(), 'python': sys.version, 'versions': versions,
            'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


class Sample:
    """Per-turn 50ms process RSS/system sampling; external guard enforces limits."""
    def __enter__(self):
        import psutil
        from airllm.nextgen.experiment_guard import system_snapshot
        self.before = system_snapshot()
        self.peak = psutil.Process().memory_info().rss
        self.swap = 0
        self.available = self.before['available_bytes']
        self.samples = 0
        self.done = threading.Event()
        def run():
            while not self.done.is_set():
                current = system_snapshot()
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
                self.swap = max(self.swap, current['swap_used_bytes'] - self.before['swap_used_bytes'])
                self.available = min(self.available, current['available_bytes'])
                self.samples += 1
                self.done.wait(.05)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        from airllm.nextgen.experiment_guard import system_snapshot
        self.done.set()
        self.thread.join()
        self.result = {'before': self.before, 'after': system_snapshot(),
                       'sampled_peak_rss_bytes': self.peak, 'maximum_swap_growth_bytes': self.swap,
                       'minimum_available_bytes': self.available, 'samples': self.samples, 'interval_seconds': .05}


def worker(args):
    # Capture launch delay separately from imports/tokenizer/model initialization.
    entry = perf_counter()
    import numpy as np
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    from airllm.nextgen.gguf_chat import GGUFChatSession
    from airllm.nextgen.physical_io import physical_read_bytes, physical_read_counter_info
    from airllm.nextgen.experiment_guard import system_snapshot
    selected = set(args.sessions.split(',')) if args.sessions else None
    sessions = json.loads(args.tasks.read_text())
    if selected:
        sessions = [s for s in sessions if s['id'] in selected]
        if {s['id'] for s in sessions} != selected:
            raise ValueError('Unknown session selection')
    report = {'kind': 'real_world_native_airllm_gguf', 'status': 'started', 'identity': identity(),
              'model_file_bytes': args.model.stat().st_size, 'model_sha256_previously_verified': args.model.name.removeprefix('sha256-'),
              'model_hash_scope': 'Existing full hash from PR4; not recomputed by this script',
              'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
              'settings': {'reuse_kv': not args.rebuild, 'temperature': 0., 'sampling': 'greedy', 'enable_thinking': False},
              'before': system_snapshot(), 'storage_counter': physical_read_counter_info(), 'turns': []}
    save(args.output, report)
    t = perf_counter()
    with GGUFReader(args.model, memory_budget_bytes=256 << 20, max_metadata_bytes=192 << 20) as cp:
        tokenizer = GGUFTokenizer(cp.metadata)
    report['tokenizer_seconds'] = perf_counter() - t
    del cp
    options = NextConfig(cache_bytes=args.cache_mib << 20, io_bytes=64 << 20, kv_bytes=args.kv_mib << 20,
        workspace_bytes=128 << 20, reserve_bytes=3 << 30, max_tensor_bytes=16 << 20,
        row_chunk=args.rows, prefill_chunk=16, max_context=args.max_context, policy='adaptive',
        device='gpu', dtype='float32', prefetch=False, load_tokenizer=False, io_mode='pread')
    report['options'] = asdict(options)
    t = perf_counter()
    try:
        with NativeGGUFQwenModel(args.model, options, kernel='metal') as model:
            report['model_initialization_seconds'] = perf_counter() - t
            report['worker_ready_seconds'] = perf_counter() - entry
            report['launch_to_ready_seconds'] = perf_counter() - args.launch_clock
            report['config'] = model.config
            save(args.output, report)
            for session_spec in sessions:
                chat = GGUFChatSession(model, tokenizer, system_prompt=session_spec.get('system'), reuse_kv=not args.rebuild)
                for spec in session_spec['turns']:
                    item = {'id': spec['id'], 'session_id': session_spec['id'], 'domain': session_spec['domain'],
                            'prompt': spec['prompt'], 'max_new_tokens': spec['max_new_tokens'], 'checks': spec.get('checks'),
                            'status': 'started', 'partial_generated_ids': [], 'partial_text': ''}
                    report['turns'].append(item)
                    save(args.output, report)
                    logical = model.checkpoint.stats['logical_bytes']
                    storage = physical_read_bytes()
                    cpu = process_time()
                    start = perf_counter()
                    def emitted(*event):
                        # API accepts token ID, optionally decoded data; raw IDs
                        # make partial output unambiguous across Unicode splits.
                        token = event[0]
                        if isinstance(token, dict):
                            token = token.get('token_id', token.get('token'))
                        item['partial_generated_ids'].append(int(token))
                        item['partial_text'] = tokenizer.decode(item['partial_generated_ids'], skip_special_tokens=True)
                        item['partial_elapsed_seconds'] = perf_counter() - start
                        save(args.output, report)
                    try:
                        with Sample() as sampling:
                            result = chat.reply(spec['prompt'], max_new_tokens=spec['max_new_tokens'], on_token=emitted)
                        item.update(result)
                        item['memory'] = sampling.result
                        item['response_wall_seconds'] = perf_counter() - start
                        item['cpu_seconds'] = process_time() - cpu
                        item['logical_read_bytes'] = model.checkpoint.stats['logical_bytes'] - logical
                        end_storage = physical_read_bytes()
                        item['process_storage_read_bytes'] = end_storage - storage if storage is not None and end_storage is not None else None
                        item['metrics'] = model.metrics()
                        item['cold_first_request'] = len(report['turns']) == 1
                        item['perceived_wait_seconds'] = item['response_wall_seconds'] + (report['launch_to_ready_seconds'] if item['cold_first_request'] else 0)
                        item['perceived_ttft_seconds'] = item['ttft_seconds'] + (report['launch_to_ready_seconds'] if item['cold_first_request'] else 0)
                        item['status'] = 'completed'
                        print(json.dumps({'id': item['id'], 'text': item['text'], 'seconds': item['response_wall_seconds'],
                                          'tokens': len(item['generated_ids']), 'stop': item['stop_reason']}, ensure_ascii=False), flush=True)
                    except BaseException as error:
                        item['status'] = 'failed'
                        item['error'] = f'{type(error).__name__}: {error}'
                        save(args.output, report)
                        raise
                    finally:
                        save(args.output, report)
                del chat
                gc.collect()
            report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['after'] = system_snapshot()
        report['worker_wall_seconds'] = perf_counter() - entry
        save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=ROOT / 'research/real_world/tasks.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', default='')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--kv-mib', type=int, default=320)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--max-context', type=int, default=1536)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--launch-clock', type=float)
    args = parser.parse_args()
    if min(args.cache_mib, args.kv_mib, args.rows, args.max_context, args.timeout) <= 0:
        parser.error('Positive budgets and timeout required')
    from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, validate_output_paths
    validate_output_paths(args.output, protected=[args.model, args.tasks])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
    else:
        # Same reserve/RSS/swap/pressure protections as PR4. Only bounded KV
        # payload is increased for the measured 1024-token/context extension.
        limits = ExperimentLimits(admitted_payload_bytes=(args.cache_mib + args.kv_mib + 448) << 20,
                                  timeout_seconds=args.timeout)
        supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker',
                   '--launch-clock', str(perf_counter())], args.output, limits,
                  env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))


if __name__ == '__main__':
    main()
