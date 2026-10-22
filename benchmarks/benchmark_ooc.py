#!/usr/bin/env python3
"""Bounded, reproducible offset-I/O comparison. Never an inference benchmark.

Synthetic fixtures measure Python/OS I/O overhead only. Warm reads can be served
entirely from RAM. Device throughput requires physical counters and a separate
large-file, cold-cache protocol; neither is inferred from logical throughput.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import struct
import sys
from time import perf_counter, process_time, time_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'air_llm'))
import numpy as np
from airllm.nextgen.storage import OffsetCheckpoint, physical_read_bytes

MiB = 1024 ** 2


def transfer_bounds(parameters, bits, active_parameters, bytes_per_second,
                    resident_active_bytes=0, accepted_per_pass=1):
    """Optimistic weight-only lower bound; no kernel, KV, metadata or quality cost."""
    if min(parameters, active_parameters, bytes_per_second, bits, accepted_per_pass) <= 0:
        raise ValueError('parameters, bits, rate and acceptance must be positive')
    if active_parameters > parameters or resident_active_bytes < 0:
        raise ValueError('invalid active or resident size')
    payload = parameters * bits / 8
    streamed = max(0, active_parameters * bits / 8 - resident_active_bytes)
    seconds = streamed / bytes_per_second / accepted_per_pass
    return {'parameter_payload_bytes_excluding_scales': payload,
            'assumed_streamed_weight_bytes_per_pass': streamed,
            'assumed_accepted_tokens_per_pass': accepted_per_pass,
            'optimistic_weight_transfer_seconds_per_accepted_token': seconds,
            'optimistic_weight_only_tokens_per_second': 1 / seconds if seconds else None,
            'scope': 'theoretical assumption; not measured model performance; scale/index overhead excluded'}


def make_fixture(root, tensors=256, tensor_bytes=65536):
    """Write actual safetensors bytes without requiring Torch or network access."""
    if tensors < 1 or tensor_bytes < 4 or tensor_bytes % 4:
        raise ValueError('positive tensor count and float32-aligned tensor bytes required')
    root.mkdir(parents=True, exist_ok=True)
    header = {}
    for i in range(tensors):
        header[f'experts.{i:06d}.weight'] = {
            'dtype': 'F32', 'shape': [1, tensor_bytes // 4],
            'data_offsets': [i * tensor_bytes, (i + 1) * tensor_bytes]}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with (root / 'model.safetensors').open('wb') as f:
        f.write(struct.pack('<Q', len(raw))); f.write(raw)
        for i in range(tensors):
            f.write((np.arange(tensor_bytes // 4, dtype=np.float32) + i).tobytes())
        f.flush(); os.fsync(f.fileno())


def _digest(arrays):
    digest = sha256()
    for value in arrays:
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def _cold_hint(cp):
    if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        return {'requested': True, 'available': False, 'guaranteed_cold': False}
    failures = []
    for name in sorted({v['file'] for v in cp.tensors.values()}):
        try:
            fd = os.open(cp.root / name, os.O_RDONLY)
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally: os.close(fd)
        except OSError as error:
            failures.append(str(error))
    return {'requested': True, 'available': True, 'failures': failures,
            'guaranteed_cold': False, 'method': 'POSIX_FADV_DONTNEED advisory only'}


def probe_storage(path, *, seconds=30., chunk_bytes=4 * MiB, random_access=False,
                  seed=20261022, max_bytes=512 * MiB):
    """Read existing data only; bounded buffers, explicit duration/transfer limits.

    F_NOCACHE is a public Darwin file-descriptor control, not a guarantee that
    every byte is fetched from physical NAND (controller caches still exist).
    Linux DONTNEED is an advisory and does not establish verified cold cache.
    """
    if seconds <= 0 or chunk_bytes <= 0 or max_bytes <= 0:
        raise ValueError('probe time, buffers and transfer cap must be positive')
    path = Path(path).resolve(); fd = os.open(path, os.O_RDONLY)
    count = os.fstat(fd).st_size
    if not count:
        os.close(fd); raise ValueError('cannot probe empty file')
    bypass = {'method': None, 'requested': False, 'succeeded': False}
    try:
        if sys.platform == 'darwin':
            import fcntl
            bypass = {'method': 'Darwin F_NOCACHE (48)', 'requested': True, 'succeeded': False,
                      'source': 'https://github.com/apple/darwin-xnu/blob/main/bsd/sys/fcntl.h'}
            try:
                fcntl.fcntl(fd, 48, 1); bypass['succeeded'] = True
            except OSError as error:
                bypass['error'] = str(error)
        elif hasattr(os, 'posix_fadvise'):
            bypass = {'method': 'POSIX_FADV_DONTNEED advisory per read', 'requested': True, 'succeeded': True}
        rng = random.Random(seed); logical = 0; offset = 0; rows = []
        physical = physical_read_bytes(); start = perf_counter(); cpu_start = process_time()
        window_start = start; window_bytes = 0
        while perf_counter() - start < seconds and logical < max_bytes:
            size = min(chunk_bytes, count, max_bytes - logical)
            if random_access:
                # Alignment keeps the request representative of row/expert ranges.
                slots = max(1, (count - size) // 4096 + 1)
                offset = rng.randrange(slots) * 4096
            elif offset + size > count:
                offset = 0
            if sys.platform != 'darwin' and bypass['succeeded']:
                try: os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
                except OSError as error:
                    bypass['succeeded'] = False; bypass['error'] = str(error)
            block = os.pread(fd, size, offset)
            if not block: raise EOFError('probe file changed or became unreadable')
            logical += len(block); window_bytes += len(block); offset += len(block)
            del block
            now = perf_counter()
            if now - window_start >= 1:
                try:
                    import psutil
                    memory = {'available_bytes': psutil.virtual_memory().available,
                              'swap_used_bytes': psutil.swap_memory().used}
                except ImportError:
                    memory = {'available_bytes': None, 'swap_used_bytes': None}
                rows.append({'elapsed_seconds': now - start, 'window_seconds': now - window_start,
                             'logical_bytes': window_bytes, **memory})
                window_start = now; window_bytes = 0
        end = perf_counter(); physical_end = physical_read_bytes()
        if window_bytes:
            rows.append({'elapsed_seconds': end - start, 'window_seconds': end - window_start,
                         'logical_bytes': window_bytes})
        return {'platform': platform.platform(), 'input_file': str(path), 'input_file_bytes': count,
                'input_size_is_small_enough_for_RAM_cache': count < 16 * 1024 ** 3,
                'read_pattern': 'seeded random aligned' if random_access else 'sequential wrapping',
                'duration_limit_seconds': seconds, 'transfer_limit_bytes': max_bytes,
                'seconds': end - start, 'cpu_seconds': process_time() - cpu_start,
                'logical_bytes': logical, 'logical_file_service_bytes_per_second': logical / (end - start),
                'physical_read_bytes': physical_end - physical
                    if physical is not None and physical_end is not None else None,
                'host_cache_bypass': bypass, 'physical_ssd_bandwidth_certified': False,
                'temperature_c': None, 'gpu_energy_joules': None,
                'scope': 'existing-file read-only probe; logical rate and cache controls reported separately; no inference',
                'seed': seed, 'samples': rows}
    finally:
        os.close(fd)


def run_comparison(root, *, metadata_dir=None, repeats=5, group_size=8,
                   max_bytes=16 * MiB, max_read_bytes=4 * MiB,
                   cache_state='warm', seed=20261022, baseline_storage=None):
    if repeats < 1 or group_size < 1 or max_bytes < 1 or max_read_bytes < 1:
        raise ValueError('benchmark budgets, group size and repeats must be positive')
    t = perf_counter()
    cp = OffsetCheckpoint(root, metadata_dir, max_read_bytes=max_read_bytes)
    metadata_seconds = perf_counter() - t
    baseline_cp = None; baseline_sha256 = None
    if baseline_storage is not None:
        try:
            baseline_storage = Path(baseline_storage).resolve()
            baseline_sha256 = sha256(baseline_storage.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location('airllm.nextgen._baseline_storage', baseline_storage)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            baseline_cp = module.OffsetCheckpoint(root, metadata_dir, max_read_bytes=max_read_bytes)
        except BaseException:
            cp.close(); raise
    try:
        # Preserve adjacency for a representative expert-weight read batch. Skip
        # tensors that cannot individually fit the declared bounded read budget.
        keys = sorted(cp.tensors, key=lambda k: (cp.tensors[k]['file'], cp.tensors[k]['offset']))
        selected = []; selected_bytes = 0
        for key in keys:
            size = cp.tensors[key]['bytes']
            if not size or size > max_read_bytes or selected_bytes + size > max_bytes:
                continue
            selected.append(key); selected_bytes += size
        if not selected:
            raise ValueError('No nonempty tensors fit the requested benchmark budget')
        groups = []; group = []; size = 0
        for key in selected:
            n = cp.tensors[key]['bytes']
            if group and (len(group) >= group_size or size + n > max_read_bytes):
                groups.append(group); group = []; size = 0
            group.append(key); size += n
        if group: groups.append(group)
        expected = []
        for group in groups:
            expected.append(_digest([cp.read_array(key) for key in group]))
        modes = ['separate_pread', 'separate_mmap', 'coalesced_pread', 'coalesced_mmap']
        rng = random.Random(seed); rows = []
        for repeat in range(repeats):
            order = list(modes); rng.shuffle(order)
            for mode in order:
                active_cp = baseline_cp if baseline_cp is not None and mode.startswith('separate') else cp
                if cache_state == 'warm':
                    for group in groups:
                        for key in group: active_cp.read_array(key)
                    hint = {'requested': False, 'guaranteed_cold': False}
                elif cache_state == 'cold-hint':
                    hint = _cold_hint(active_cp)
                else:
                    hint = {'requested': False, 'guaranteed_cold': False}
                before = dict(active_cp.stats); physical = physical_read_bytes()
                cpu = process_time(); begin = perf_counter(); read_seconds = 0.; verified = True
                for i, group in enumerate(groups):
                    start = perf_counter(); backend = mode.rsplit('_', 1)[1]
                    arrays = (active_cp.read_arrays(group, mode=backend, max_gap_bytes=0,
                                            max_batch_bytes=max_read_bytes)
                              if mode.startswith('coalesced') else
                              [active_cp.read_array(key, mode=backend) for key in group])
                    read_seconds += perf_counter() - start
                    verified = verified and _digest(arrays) == expected[i]
                    del arrays
                elapsed = perf_counter() - begin; cpu_seconds = process_time() - cpu
                physical_end = physical_read_bytes()
                delta = {k: v - before.get(k, 0) for k, v in active_cp.stats.items()
                         if isinstance(v, (float, int))}
                if not verified: raise AssertionError(f'Numerical payload mismatch in {mode}')
                rows.append({'repeat': repeat, 'order': order.index(mode), 'mode': mode,
                    'wall_seconds_including_sha256': elapsed, 'read_seconds': read_seconds,
                    'cpu_seconds_including_sha256': cpu_seconds,
                    'requested_tensor_bytes': selected_bytes,
                    'logical_bytes_per_second_read_timing': delta.get('logical_bytes', 0) / read_seconds,
                    'physical_read_bytes': (physical_end - physical)
                        if physical is not None and physical_end is not None else None,
                    'counter_delta': delta, 'payload_sha256_verified': verified,
                    'cache_hint': hint})
        summary = {}
        for mode in modes:
            cases = [r for r in rows if r['mode'] == mode]
            summary[mode] = {'median_read_seconds': statistics.median(r['read_seconds'] for r in cases),
                             'median_read_calls': statistics.median(r['counter_delta'].get('read_calls', 0) for r in cases)}
        baseline = summary['separate_pread']['median_read_seconds']
        for case in summary.values():
            case['read_time_speedup_vs_separate_pread'] = baseline / case['median_read_seconds']
        return {'schema_version': 1, 'platform': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'numpy': np.__version__, 'pid': os.getpid(),
            'checkpoint_fingerprint': cp.fingerprint,
            'baseline_storage_sha256': baseline_sha256,
            'baseline_scope': 'archive baseline storage module' if baseline_cp else 'same code, separate read_array calls',
            'metadata_seconds': metadata_seconds, 'metadata_stats': {k: v for k, v in cp.stats.items() if k in ('manifest_hit','header_bytes')},
            'input_tensor_count': len(selected), 'input_tensor_bytes': selected_bytes,
            'input_order_sha256': sha256('\n'.join(selected).encode()).hexdigest(),
            'group_count': len(groups), 'max_read_bytes': max_read_bytes,
            'cache_state': cache_state,
            'cache_state_explanation': {'warm': 'every selected tensor touched immediately before each candidate; RAM/page cache effects included',
                'uncontrolled': 'OS page cache uncontrolled; initial equality pass already touched selected tensors',
                'cold-hint': 'best-effort eviction advisory before each candidate; no verified cold-cache claim'}[cache_state],
            'physical_counters_scope': 'process-level Linux storage bytes, including readahead; unavailable platforms return null',
            'physical_ssd_bandwidth_certified': False, 'inference_performed': False,
            'scope': 'bounded offset-I/O microbenchmark; no model, Metal, quality, TTFT or full-checkpoint throughput claim',
            'peak_rss_process_lifetime_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            'seed': seed, 'summary': summary, 'raw_measurements': rows}
    finally:
        cp.close()
        if baseline_cp is not None: baseline_cp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='optional existing local checkpoint; never downloaded')
    parser.add_argument('--baseline-storage', type=Path, help='optional archived storage.py; separate reads use this exact source')
    parser.add_argument('--probe-file', type=Path, help='read-only SSD probe of one existing file; skips tensor benchmark')
    parser.add_argument('--probe-seconds', type=float, default=30.)
    parser.add_argument('--probe-max-mib', type=int, default=512, help='hard cap; deliberately increase for sustained probe')
    parser.add_argument('--probe-random', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=Path('.benchmark-ooc'))
    parser.add_argument('--output', type=Path, default=Path('evidence-ooc/io-host.json'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-mib', type=int, default=16)
    parser.add_argument('--max-read-mib', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--cache-state', choices=('warm', 'uncontrolled', 'cold-hint'), default='warm')
    args = parser.parse_args()
    if args.probe_file:
        result = probe_storage(args.probe_file, seconds=args.probe_seconds,
            max_bytes=args.probe_max_mib * MiB, random_access=args.probe_random)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'seconds': result['seconds'],
                          'logical_bytes': result['logical_bytes']}, indent=2))
        return
    work = args.work_dir.resolve() / ('run-' + str(time_ns())); work.mkdir(parents=True)
    root = args.checkpoint
    if root is None:
        root = work / 'fixture'; make_fixture(root)
    result = run_comparison(root, metadata_dir=work / 'metadata', repeats=args.repeats,
        group_size=args.group_size, max_bytes=args.max_mib * MiB,
        max_read_bytes=args.max_read_mib * MiB, cache_state=args.cache_state,
        baseline_storage=args.baseline_storage)
    result['input_kind'] = 'existing local checkpoint' if args.checkpoint else 'deterministic synthetic float32 tensor bytes'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'summary': result['summary']}, indent=2))


if __name__ == '__main__': main()
