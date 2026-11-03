"""Bounded read-only selected-range probe; no model, purge or NAND claim.

Use the same range plan for normal/nocache runs. F_NOCACHE is a requested
descriptor hint, not proof of a physically cold read. Global controller counters
include other processes. Existing v2 evidence is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.glm53_gpu_v2.storage_counters import block_snapshot, compare_snapshots
from research.glm53_gpu_v2.benchmark_lifecycle import interrupt_on_sigterm
from .process_io import ProcessIOObserver, compare_process, sample_owned
from .system_memory import SystemMemoryObserver

MIB = 1024**2
GIB = 1024**3
MAX_LOGICAL_BYTES = GIB
MAX_SCRATCH_BYTES = 8 * MIB


class ProbeGuardViolation(RuntimeError):
    pass


def validate_ranges(plan, source_bytes):
    rows = plan.get('ranges') if isinstance(plan, dict) else plan
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError('range plan must contain1..256 explicit ranges')
    validated = []
    total = expanded = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError('each range must be an object')
        offset, length, repetitions = row.get('offset'), row.get('length'), row.get('repetitions', 1)
        if (any(type(v) is not int for v in (offset, length, repetitions)) or
                offset < 0 or length <= 0 or not 1 <= repetitions <= 64):
            raise ValueError('range offsets/lengths/repetitions must be bounded integers')
        if offset > source_bytes or length > source_bytes - offset:
            raise ValueError('range exceeds the selected original file')
        total += length * repetitions
        expanded += repetitions
        if total > MAX_LOGICAL_BYTES or expanded > 1024:
            raise ValueError('range plan exceeds1GiB logical or1024 expanded-range budget')
        label = row.get('label', row.get('name', f'range-{index}'))
        if not isinstance(label, str) or len(label) > 512:
            raise ValueError('range label must be a bounded string')
        validated.append({'offset': offset, 'length': length, 'repetitions': repetitions, 'label': label})
    return validated, total


def advise_descriptor(fd, mode):
    if mode == 'nocache':
        if sys.platform != 'darwin':
            raise RuntimeError('F_NOCACHE probe requires Darwin')
        fcntl.fcntl(fd, 48, 1)  # Public Darwin SDK: F_NOCACHE, descriptor-local request.


def _identity(stat):
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _save(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_sample(sample, initial_swap, *, initial=False):
    if not sample.get('available') or not sample.get('process', {}).get('available') or not sample.get('system', {}).get('available'):
        raise ProbeGuardViolation('required owned-process/global observation unavailable')
    system, process = sample['system'], sample['process']['rusage']
    pressure = system['pressure_level']
    if pressure not in (1, 2, 4) or initial and pressure != 1:
        raise ProbeGuardViolation('launch requires normal measured memory pressure')
    if pressure == 4:
        raise ProbeGuardViolation('critical global memory pressure')
    if process['resident_bytes'] > int(8.5 * GIB):
        raise ProbeGuardViolation('owned probe RSS exceeds8.5GiB')
    if initial_swap is not None and system['swap_used_bytes'] - initial_swap > 32 * MIB:
        raise ProbeGuardViolation('global swap grew beyond32MiB')


def controller_ratio(before, after, logical_bytes):
    delta = compare_snapshots(before, after)
    rows = delta.get('devices', [])
    if (not delta.get('available') or not rows or
            any('Bytes (Read)' not in row.get('delta', {}) for row in rows)):
        return {'available': False, 'counter_delta': delta, 'global_controller_read_bytes': None,
                'controller_to_logical_ratio': None}
    total = sum(row['delta']['Bytes (Read)'] for row in rows)
    return {'available': True, 'counter_delta': delta, 'global_controller_read_bytes': total,
            'controller_to_logical_ratio': total / logical_bytes if logical_bytes else None,
            'scope': 'global controller bytes divided by probe logical bytes; may exceed1 and is not a cache-hit fraction or NAND ratio'}


def run_probe(source, plan, mode, output, *, scratch_bytes=MAX_SCRATCH_BYTES,
              sample_fn=None, block_fn=block_snapshot, fd_adviser=advise_descriptor,
              clock=time.monotonic):
    source, output = Path(source), Path(output)
    reads_path = output.with_suffix('.reads.jsonl')
    if output.exists() or reads_path.exists():
        raise ValueError('refusing to overwrite probe or read-latency evidence')
    if mode not in ('normal', 'nocache') or type(scratch_bytes) is not int or not 1 <= scratch_bytes <= MAX_SCRATCH_BYTES:
        raise ValueError('mode must be normal/nocache and scratch payload <=8MiB')
    original = _identity(source.stat())
    ranges, total = validate_ranges(plan, original['size'])
    if sample_fn is None:
        owned, system = ProcessIOObserver(os.getpid()), SystemMemoryObserver()
        sample_fn = lambda: sample_owned(owned, system)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 1, 'status': 'starting', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_file': source.name, 'source_identity_before': original, 'ranges': ranges,
              'range_order': 'listed range, repetition, increasing byte offset within range',
              'requested_logical_bytes': total, 'completed_logical_bytes': 0, 'mode': mode,
              'fd_nocache_requested': mode == 'nocache', 'fd_advice_succeeded': None,
              'scratch_payload_bytes': min(scratch_bytes, max(row['length'] for row in ranges)),
              'limits': {'logical_bytes': GIB, 'scratch_payload_bytes': MAX_SCRATCH_BYTES,
                         'rss_bytes': int(8.5 * GIB), 'global_swap_growth_bytes': 32 * MIB,
                         'launch_pressure_level': 1, 'critical_pressure_level': 4,
                         'guard_sampling_seconds': .25, 'max_read_calls': 4096},
              'reads_jsonl': reads_path.name, 'range_results': [], 'guard_samples': [],
              'guard_violation': None, 'peak_global_swap_growth_bytes': 0,
              'physical_nand_read_bytes': None, 'per_pid_physical_device_bytes': None,
              'file_cache_hit_bytes': None, 'physically_cold_proven': False,
              'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'selected original bytes through one read-only descriptor; OS/process and global controller bookkeeping, not NAND',
              'timing_limits': ['read-call latency excludes hash, guards and report writes',
                                'wall time includes hash, guards and diagnostics',
                                'normal and F_NOCACHE do not control prior macOS cache state',
                                'guards are sampled between bounded reads; caller must bound a stalled syscall/process']}
    _save(output, report)
    initial_swap = None
    started = clock()
    digest = hashlib.sha256()
    read_calls = 0
    read_seconds = 0.
    before = after = block_before = block_after = None
    try:
        before = sample_fn()
        report['initial_observation'] = before
        if before.get('system', {}).get('available'):
            initial_swap = before['system']['swap_used_bytes']
        check_sample(before, initial_swap, initial=True)
        report['guard_samples'].append(before)
        block_before = block_fn()
        report['global_block_before'] = block_before
        payload = bytearray(report['scratch_payload_bytes'])
        target = memoryview(payload)
        next_guard = clock() + .25
        with reads_path.open('x') as read_log, source.open('rb', buffering=0) as stream:
            if _identity(os.fstat(stream.fileno())) != original:
                raise RuntimeError('source identity changed before reading')
            fd_adviser(stream.fileno(), mode)
            report['fd_advice_succeeded'] = True
            for range_index, row in enumerate(ranges):
                for repetition in range(row['repetitions']):
                    stream.seek(row['offset'])
                    remaining = row['length']
                    range_digest = hashlib.sha256()
                    range_started = clock()
                    while remaining:
                        if read_calls >= 4096:
                            raise RuntimeError('bounded read-call metadata budget exceeded')
                        if clock() >= next_guard:
                            sample = sample_fn()
                            report['guard_samples'].append(sample)
                            if initial_swap is not None:
                                report['peak_global_swap_growth_bytes'] = max(report['peak_global_swap_growth_bytes'],
                                    sample.get('system', {}).get('swap_used_bytes', initial_swap) - initial_swap)
                            check_sample(sample, initial_swap)
                            next_guard = clock() + .25
                        count_requested = min(remaining, len(payload))
                        began = clock()
                        count = stream.readinto(target[:count_requested])
                        latency = clock() - began
                        if not count:
                            raise OSError('short read inside the selected original range')
                        read_calls += 1
                        read_seconds += latency
                        digest.update(target[:count])
                        range_digest.update(target[:count])
                        offset = row['offset'] + row['length'] - remaining
                        remaining -= count
                        report['completed_logical_bytes'] += count
                        read_log.write(json.dumps({'range_index': range_index, 'repetition': repetition,
                            'offset': offset, 'bytes': count, 'read_call_seconds': latency}) + '\n')
                    read_log.flush()
                    report['range_results'].append({'range_index': range_index, 'repetition': repetition,
                        'label': row['label'], 'offset': row['offset'], 'length': row['length'],
                        'sha256_original_bytes': range_digest.hexdigest(), 'wall_seconds': clock() - range_started})
                    _save(output, report)
            if _identity(os.fstat(stream.fileno())) != original:
                raise RuntimeError('opened source file stats changed during reads')
        report['status'] = 'complete'
    except ProbeGuardViolation as exc:
        report.update(status='stopped_by_guard', guard_violation=str(exc))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            after = sample_fn()
            report['final_observation'] = after
            report['guard_samples'].append(after)
            if initial_swap is not None:
                report['peak_global_swap_growth_bytes'] = max(report['peak_global_swap_growth_bytes'],
                    after.get('system', {}).get('swap_used_bytes', initial_swap) - initial_swap)
            check_sample(after, initial_swap)
        except Exception as exc:
            report['final_observation_error'] = f'{type(exc).__name__}: {exc}'
            if report['status'] == 'complete':
                report.update(status='stopped_by_guard', guard_violation=str(exc))
        try:
            final_identity = _identity(source.stat())
            report['source_identity_after'] = final_identity
            report['selected_file_stats_unchanged'] = original == final_identity
            if original != final_identity:
                report.update(status='failed', error='source file identity/size/mtime changed')
        except OSError as exc:
            report.update(status='failed', selected_file_stats_unchanged=False, error=str(exc))
        try:
            block_after = block_fn()
            report['global_block_after'] = block_after
            report['controller_observation'] = controller_ratio(block_before or {}, block_after,
                                                               report['completed_logical_bytes'])
        except Exception as exc:
            report['controller_observation'] = {'available': False, 'error': str(exc)}
        if before is not None and after is not None:
            report['owned_process_delta'] = compare_process(before.get('process', {}), after.get('process', {}))
        report.update(read_calls=read_calls, summed_read_call_seconds=read_seconds,
            logical_bytes_per_read_call_second=report['completed_logical_bytes'] / read_seconds if read_seconds > 0 else None,
            sha256_completed_ordered_stream=digest.hexdigest(), wall_seconds=clock() - started,
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        report['counter_observations_available'] = bool(report.get('owned_process_delta', {}).get('available') and
                                                       report.get('controller_observation', {}).get('available'))
        _save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--ranges', type=Path, required=True)
    parser.add_argument('--mode', choices=('normal', 'nocache'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.ranges.stat().st_size > 128 * 1024:
        parser.error('range JSON exceeds128KiB parsing budget')
    plan = json.loads(args.ranges.read_text())
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with interrupt_on_sigterm():
            report = run_probe(args.gguf, plan, args.mode, args.output)
    print(json.dumps({key: report[key] for key in ('status', 'completed_logical_bytes', 'guard_violation', 'selected_file_stats_unchanged')}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
