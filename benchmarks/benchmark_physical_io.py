#!/usr/bin/env python3
"""Bounded, paired pread/mmap/F_NOCACHE experiments; never claims NAND speed.

An owned incompressible safetensors fixture is fsynced, then read by fresh child
processes in shuffled order. Cache scenarios are explicitly uncontrolled or
preceded by a complete normal-pread warmup. No purge, pressure eviction, raw disk,
privileged operation, download, or write outside the output/fixture directory.
With --baseline-source, the original storage implementation is imported in its
own process. Its nocache row uses only a disclosed descriptor-policy harness;
the baseline data-read loop itself remains unchanged.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
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
import subprocess
import sys
import tempfile
from time import perf_counter

MiB = 1024 ** 2
MODES = ('pread', 'mmap', 'pread_nocache')
SCENARIOS = ('uncontrolled_after_creation_and_prior_jobs', 'explicit_normal_pread_warmup')


def delta(start, end):
    return end - start if start is not None and end is not None and end >= start else None


def system_disk_counters():
    import psutil
    counters = psutil.disk_io_counters(perdisk=True)
    return ({key: {'read_bytes': value.read_bytes, 'read_count': value.read_count}
             for key, value in counters.items()} if counters else {})


def disk_identity(path):
    if sys.platform != 'darwin': return {'path': str(path), 'detail': None}
    import plistlib
    # diskutil accepts a device/mount point, not an arbitrary nested directory.
    df = subprocess.run(['df', '-P', str(path)], capture_output=True, text=True)
    lines = df.stdout.splitlines()
    device = lines[-1].split()[0] if df.returncode == 0 and len(lines) > 1 else None
    details = []
    pending = [device] if device else []
    seen = set()
    keys = ('DeviceIdentifier', 'DeviceNode', 'DeviceLocation', 'Internal', 'MediaName',
            'BusProtocol', 'SolidState', 'MountPoint', 'APFSPhysicalStores', 'ParentWholeDisk')
    while pending and len(seen) < 8:
        candidate = pending.pop(0)
        if candidate in seen: continue
        seen.add(candidate)
        done = subprocess.run(['diskutil', 'info', '-plist', candidate], capture_output=True)
        if done.returncode: continue
        info = plistlib.loads(done.stdout)
        details.append({key: info[key] for key in keys if key in info})
        for store in info.get('APFSPhysicalStores', []):
            identifier = (store.get('DeviceIdentifier') or store.get('APFSPhysicalStore')) if isinstance(store, dict) else store
            if identifier: pending.append(identifier)
        if info.get('ParentWholeDisk'): pending.append(info['ParentWholeDisk'])
    return {'resolved_path': str(path), 'filesystem_device': device, 'device_chain': details}


def create_fixture(root, size_bytes, chunk_bytes, seed, write_policy='normal'):
    import numpy as np
    count = size_bytes // chunk_bytes
    header = json.dumps({'payload': {'dtype': 'U8', 'shape': [count, chunk_bytes],
                                    'data_offsets': [0, size_bytes]}}).encode()
    header += b' ' * (-len(header) % 8)
    prefix = struct.pack('<Q', len(header)) + header
    rng = np.random.default_rng(seed)
    file_hash = sha256(prefix)
    chunk_hashes = []
    with (root / 'model.safetensors').open('xb') as file:
        if write_policy == 'nocache':
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'air_llm'))
            from airllm.nextgen.physical_io import set_fd_nocache
            set_fd_nocache(file.fileno())
        file.write(prefix)
        for _ in range(count):
            chunk = rng.bytes(chunk_bytes)
            file.write(chunk)
            file_hash.update(chunk)
            chunk_hashes.append(sha256(chunk).digest())
        file.flush()
        os.fsync(file.fileno())
    order = list(range(count))
    random.Random(seed).shuffle(order)
    expected = sha256(b''.join(chunk_hashes[i] for i in order)).hexdigest()
    return {'file_sha256': file_hash.hexdigest(), 'request_order': order,
            'ordered_chunk_digest_sha256': expected, 'payload_bytes': size_bytes,
            'chunk_bytes': chunk_bytes, 'fsync_completed': True, 'writer_cache_policy': write_policy,
            'os_cache_evicted': False, 'incompressible_pseudorandom_seed': seed}


def worker(args):
    # Use the same current instrumentation for both revisions. Import storage
    # only after choosing the source tree, with a fresh interpreter per job.
    spec = importlib.util.spec_from_file_location('physical_evidence', Path(args.instrumentation))
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    sys.path.insert(0, str(Path(args.worker_source) / 'air_llm'))
    from airllm.nextgen.storage import OffsetCheckpoint
    import gc
    manifest = json.loads(Path(args.manifest).read_text())
    mode = args.mode
    cp = OffsetCheckpoint(args.fixture, max_read_bytes=manifest['chunk_bytes'])
    harness = False
    if args.baseline and mode == 'pread_nocache':
        # Baseline has no nocache enum. It is dedicated to this one mode/job;
        # setting policy on its borrowed fd also sets its shared open description.
        borrowed = cp.descriptor
        @contextmanager
        def nocache_descriptor(name):
            with borrowed(name) as fd:
                evidence.set_fd_nocache(fd)
                yield fd
        cp.descriptor = nocache_descriptor
        mode = 'pread'
        harness = True
    with cp:
        # Resolve lazy libproc and psutil imports before measurement starts.
        evidence.physical_read_bytes()
        system_disk_counters()
        warmup_stats = {'logical_bytes': 0, 'read_calls': 0}
        if args.scenario == 'explicit_normal_pread_warmup':
            if harness:
                # A separate ordinary reader is necessary: this cp owns nocache fds.
                with OffsetCheckpoint(args.fixture, max_read_bytes=manifest['chunk_bytes']) as warm:
                    for index in manifest['request_order']:
                        warm.read_array('payload', rows=(index, index + 1), mode='pread')
                    warmup_stats = dict(warm.stats)
            else:
                for index in manifest['request_order']:
                    cp.read_array('payload', rows=(index, index + 1), mode='pread')
                warmup_stats = dict(cp.stats)
        gc.collect()
        stats_start = dict(cp.stats)
        digest = sha256()
        disk_start = system_disk_counters()
        usage_start = resource.getrusage(resource.RUSAGE_SELF)
        physical_start = evidence.physical_read_bytes()
        start = perf_counter()
        read_seconds = hash_seconds = 0.
        for index in manifest['request_order']:
            before = perf_counter()
            array = cp.read_array('payload', rows=(index, index + 1), mode=mode)
            read_seconds += perf_counter() - before
            before = perf_counter()
            digest.update(sha256(memoryview(array)).digest())
            hash_seconds += perf_counter() - before
            del array
        wall_seconds = perf_counter() - start
        physical_end = evidence.physical_read_bytes()
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        disk_end = system_disk_counters()
        actual = digest.hexdigest()
        if actual != manifest['ordered_chunk_digest_sha256']:
            raise RuntimeError('Exactness failure: ordered payload digests differ')
        result = {'variant': args.label, 'mode': args.mode, 'repeat': args.repeat,
                  'cache_scenario': args.scenario,
                  'baseline_nocache_descriptor_harness': harness,
                  'source_root': args.worker_source,
                  'source_revision': subprocess.run(['git', '-C', args.worker_source, 'rev-parse', 'HEAD'],
                                                     capture_output=True, text=True).stdout.strip(),
                  'instrumentation_source_sha256': sha256(Path(args.instrumentation).read_bytes()).hexdigest(),
                  'storage_source_sha256': sha256((Path(args.worker_source) / 'air_llm/airllm/nextgen/storage.py').read_bytes()).hexdigest(),
                  'logical_read_bytes': cp.stats['logical_bytes'] - stats_start.get('logical_bytes', 0),
                  'read_calls': cp.stats['read_calls'] - stats_start.get('read_calls', 0),
                  'warmup_logical_read_bytes': warmup_stats['logical_bytes'],
                  'warmup_read_calls': warmup_stats['read_calls'],
                  'read_seconds': read_seconds, 'hash_seconds': hash_seconds, 'wall_seconds': wall_seconds,
                  'logical_mib_per_read_second': manifest['payload_bytes'] / MiB / read_seconds,
                  'physical_process_read_bytes': delta(physical_start, physical_end),
                  'physical_process_counter_start': physical_start, 'physical_process_counter_end': physical_end,
                  'physical_counter': evidence.physical_read_counter_info(),
                  'system_disk_read_byte_deltas': {key: delta(value['read_bytes'], disk_end.get(key, {}).get('read_bytes'))
                                                    for key, value in disk_start.items()},
                  'system_disk_counter_scope': 'whole-system device counters; concurrent unrelated I/O included; not attributed to this fixture',
                  'minor_page_faults': usage_end.ru_minflt - usage_start.ru_minflt,
                  'major_page_faults': usage_end.ru_majflt - usage_start.ru_majflt,
                  'peak_process_rss_bytes': int(usage_end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                  'ordered_chunk_digest_sha256': actual, 'exact': True,
                  'physical_ssd_bandwidth_certified': False}
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New report directory')
    parser.add_argument('--fixture-parent', help='Existing parent for temporary owned fixture (default report directory)')
    parser.add_argument('--size-mib', type=int, default=128)
    parser.add_argument('--chunk-kib', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--baseline-source')
    parser.add_argument('--fixture-write-policy', choices=['normal','nocache'], default='normal',
                        help='nocache requests macOS F_NOCACHE before writing the owned fixture; no eviction guarantee')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    # Internal subprocess parameters, deliberately separate from user workload.
    for name in ('worker-source', 'instrumentation', 'manifest', 'fixture', 'mode', 'label', 'scenario'):
        parser.add_argument('--' + name)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    if args.worker_source: return worker(args)
    if not 1 <= args.size_mib <= 256 or not 1 <= args.chunk_kib <= 8192 or not 1 <= args.repeats <= 100:
        parser.error('Fixture must be 1..256 MiB, chunks 1..8192 KiB, repeats 1..100')
    size, chunk = args.size_mib * MiB, args.chunk_kib * 1024
    if size % chunk: parser.error('Fixture size must be a multiple of chunk size')
    if len(set(args.modes)) != len(args.modes): parser.error('Duplicate modes')
    if len(set(args.scenarios)) != len(args.scenarios): parser.error('Duplicate scenarios')
    if args.fixture_write_policy == 'nocache' and sys.platform != 'darwin':
        parser.error('nocache fixture writer requires macOS')
    if 'pread_nocache' in args.modes and sys.platform != 'darwin':
        parser.error('pread_nocache experiment requires macOS; choose --modes pread mmap elsewhere')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture_parent = Path(args.fixture_parent).resolve(strict=True) if args.fixture_parent else output
    root = Path(__file__).resolve().parents[1]
    instrumentation = root / 'air_llm/airllm/nextgen/physical_io.py'
    variants = [('phase2', root)]
    if args.baseline_source:
        variants.insert(0, ('phase1', Path(args.baseline_source).resolve(strict=True)))
    jobs = [(repeat, label, source, mode, scenario)
            for repeat in range(args.repeats) for scenario in args.scenarios
            for label, source in variants for mode in args.modes]
    rng = random.Random(args.seed)
    # Shuffle within repeat blocks to balance temporal/thermal and cache effects.
    jobs = [job for repeat in range(args.repeats)
            for job in rng.sample([j for j in jobs if j[0] == repeat], len(jobs) // args.repeats)]
    rows = []
    with tempfile.TemporaryDirectory(prefix='airllm-physical-', dir=fixture_parent) as directory:
        fixture = Path(directory)
        manifest = create_fixture(fixture, size, chunk, args.seed, args.fixture_write_policy)
        manifest_path = output / 'fixture-manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        identity = disk_identity(fixture_parent)
        for repeat, label, source, mode, scenario in jobs:
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                       '--worker-source', str(source), '--instrumentation', str(instrumentation),
                       '--manifest', str(manifest_path), '--fixture', str(fixture),
                       '--mode', mode, '--label', label, '--scenario', scenario, '--repeat', str(repeat)]
            if label == 'phase1': command.append('--baseline')
            done = subprocess.run(command, capture_output=True, text=True, timeout=120)
            with (output / 'worker-stderr.log').open('a') as log:
                log.write(json.dumps({'repeat': repeat, 'variant': label, 'mode': mode, 'scenario': scenario}) + '\n' + done.stderr)
            if done.returncode: raise RuntimeError(f'Worker failed ({done.returncode}): {done.stderr}')
            row = json.loads(done.stdout)
            rows.append(row)
            with (output / 'rows.jsonl').open('a') as log: log.write(json.dumps(row) + '\n')
    summary = []
    for label, _ in variants:
        for mode in args.modes:
            for scenario in args.scenarios:
                group = [r for r in rows if (r['variant'], r['mode'], r['cache_scenario']) == (label, mode, scenario)]
                physical = [r['physical_process_read_bytes'] for r in group]
                summary.append({'variant': label, 'mode': mode, 'cache_scenario': scenario,
                                'runs': len(group), 'median_read_seconds': statistics.median(r['read_seconds'] for r in group),
                                'median_logical_mib_per_read_second': statistics.median(r['logical_mib_per_read_second'] for r in group),
                                'physical_process_read_bytes_each_run': physical, 'all_exact': all(r['exact'] for r in group),
                                'peak_process_rss_bytes': max(r['peak_process_rss_bytes'] for r in group)})
    report = {'schema_version': 1, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
              'fixture': manifest, 'fixture_disk_identity': identity, 'fixture_removed_after_run': True,
              'protocol': 'fresh worker per job; shuffled modes/revisions/cache scenarios per repeat; fixed offset order and payload; hashes outside read timing; fsynced creation; no purge',
              'cache_limit': 'The first scenario is uncontrolled after fixture creation/prior jobs. Warmup explicitly reads all payload with ordinary pread. Neither is called cold. F_NOCACHE requests policy; the process counter supplies separate evidence.',
              'physical_limit': 'OS-accounted process storage reads and system device totals are not NAND traffic, per-file attribution, or external SSD certification.',
              'physical_ssd_bandwidth_certified': False, 'summary': summary, 'rows': rows}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(output / 'report.json'), 'runs': len(rows), 'all_exact': all(r['exact'] for r in rows)}))


if __name__ == '__main__': main()
