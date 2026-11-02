#!/usr/bin/env python3
"""Bounded GLM native benchmark; counters describe their actual measurement scope."""
from __future__ import annotations

import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import psutil

GIB = 1024**3


class Usage(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        'user_ticks', 'system_ticks', 'idle_wakeups', 'interrupt_wakeups', 'pageins',
        'wired', 'resident', 'footprint', 'start', 'exit', 'child_user', 'child_system',
        'child_idle', 'child_interrupt', 'child_pageins', 'child_elapsed',
        'storage_read_bytes', 'storage_write_bytes')]


LIBPROC = ctypes.CDLL('/usr/lib/libproc.dylib')
LIBPROC.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


def usage(pid):
    value = Usage()
    if LIBPROC.proc_pid_rusage(pid, 2, ctypes.byref(value)):
        return None
    return {name: int(getattr(value, name)) for name in (
        'user_ticks', 'system_ticks', 'pageins', 'resident', 'footprint',
        'storage_read_bytes', 'storage_write_bytes')}


def command_text(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout.strip()


def pressure():
    value = command_text(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'])
    return int(value) if value else None


def gpu_stats():
    try:
        data = subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-d', '1', '-c', 'IOAccelerator'],
                              capture_output=True, timeout=5).stdout
        return plistlib.loads(data)[0].get('PerformanceStatistics', {})
    except (OSError, ValueError, IndexError, plistlib.InvalidFileException):
        return None


def atomic_save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def request(base, endpoint, body=None, timeout=1200):
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(base + endpoint, data=data,
        headers={'Content-Type': 'application/json'}), timeout=timeout)


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), '--offline', '--fit', 'off', '-m', str(args.model),
        '--load-mode', 'mmap', '--no-repack', '--no-warmup', '--ctx-size', str(args.context),
        '-b', str(args.batch), '-ub', str(args.batch), '-t', str(args.threads), '-tb', str(args.threads),
        '-ngl', str(args.gpu_layers), '-np', '1', '--host', '127.0.0.1', '--port', str(port),
        '--no-webui', '--no-mmproj', '--no-context-shift', '--perf', '--log-colors', 'off']
    if args.gpu_layers == 0:
        command += ['--device', 'none', '--no-op-offload']
    if args.cpu_moe:
        command += ['--cpu-moe', '--moe-cache-mib', str(args.moe_cache_mib)]
    if getattr(args, 'weights_only', False):
        command += ['--no-op-offload', '--no-kv-offload']
    command += args.extra
    report = dict(schema_version=1, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='running', command=command, model_revision='119034bc3d9d76eb41f91508e6f9fb19fde42672',
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        cache_scope='Process cold at launch; OS file cache uncontrolled. Repeats share process, KV reuse disabled.',
        hardware=command_text(['/usr/sbin/sysctl', 'hw.model', 'hw.memsize', 'machdep.cpu.brand_string']),
        os=command_text(['/usr/bin/sw_vers']), logical_weight_reads_bytes=None, physical_nand_reads_bytes=None,
        storage_scope='libproc process OS-accounted I/O, includes paging and readahead; not NAND bytes',
        gpu_scope='IORegistry system-wide utilization; includes desktop and other processes',
        cpu_counter_scope='rusage user/system are mach absolute ticks; multiply by 125/3 for nanoseconds on this M5',
        component_timings={'dequantization': None, 'router': None, 'gpu_compute': None,
            'synchronization': None, 'allocation': None},
        thermal_before=command_text(['/usr/bin/pmset', '-g', 'therm']),
        samples=[], requests=[], guard_violation=None, peak_rss_bytes=0, peak_footprint_bytes=0,
        swap_start_bytes=psutil.swap_memory().used)
    atomic_save(args.output, report)
    stop = threading.Event()
    t0 = time.perf_counter()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'AIRLLM_'))}
    report['mmap_demand'] = args.mmap_demand
    report['expert_prefetch'] = args.expert_prefetch
    report['dense_prefetch'] = args.dense_prefetch
    if args.mmap_demand:
        env['AIRLLM_MMAP_DEMAND'] = '1'
    if args.expert_prefetch:
        env['AIRLLM_EXPERT_PREFETCH'] = '1'
    if args.dense_prefetch:
        env['AIRLLM_DENSE_PREFETCH'] = '1'
    report['research_environment'] = {}
    for option, variable in (('layer_local_hc', 'AIRLLM_GLM_LAYER_LOCAL_HC'),
                             ('cache_lookup_split', 'AIRLLM_GLM_CACHE_SPLIT')):
        if getattr(args, option, False):
            env[variable] = '1'
            report['research_environment'][variable] = '1'
    if args.interposer:
        env['DYLD_INSERT_LIBRARIES'] = str(args.interposer.resolve())
    with args.output.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        report['owned_pid'] = child.pid
        process = psutil.Process(child.pid)
        def monitor():
            tick = 0
            try:
                while not stop.wait(.5) and child.poll() is None:
                    mem = psutil.virtual_memory()
                    rss = process.memory_info().rss
                    counters = usage(child.pid)
                    swap = psutil.swap_memory().used
                    row = dict(t=time.perf_counter()-t0, rss=rss, available=mem.available,
                               swap_used=swap, process=counters,
                               cpu_seconds=sum(process.cpu_times()[:2]))
                    if tick % 4 == 0:
                        row['pressure'] = pressure()
                        row['gpu_system'] = gpu_stats()
                    report['samples'].append(row)
                    if tick % 4 == 0:
                        atomic_save(args.output.with_suffix('.progress.json'), row)
                    report['peak_rss_bytes'] = max(report['peak_rss_bytes'], rss)
                    if counters:
                        report['peak_footprint_bytes'] = max(report['peak_footprint_bytes'], counters['footprint'])
                    if (rss > args.max_rss_gib * GIB or mem.available < 256 * 1024**2
                            or swap - report['swap_start_bytes'] > 64 * 1024**2
                            or row.get('pressure') == 4 or row['t'] > args.timeout):
                        report['guard_violation'] = row
                        os.killpg(child.pid, signal.SIGTERM)
                        return
                    tick += 1
            except psutil.NoSuchProcess:
                pass
            except Exception as exc:
                report['guard_violation'] = {'monitor_error': str(exc)}
                child.terminate()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        base = f'http://127.0.0.1:{port}'
        try:
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Server stopped before ready')
                try:
                    with request(base, '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.HTTPError):
                    pass
                if time.perf_counter() - t0 > 180:
                    raise TimeoutError('Server load timeout')
                time.sleep(.2)
            report['load_seconds'] = time.perf_counter()-t0
            for rep in range(args.repeats):
                body = dict(prompt=args.prompt, n_predict=args.tokens, temperature=0,
                    seed=42, stream=True, cache_prompt=False, return_tokens=True,
                    n_probs=args.n_probs, timings_per_token=True)
                row = dict(repetition=rep, body=body, events=[], usage_before=usage(child.pid))
                report['requests'].append(row)
                start = time.perf_counter()
                first = None
                with request(base, '/completion', body) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith('data: ') or line == 'data: [DONE]':
                            continue
                        event = json.loads(line[6:])
                        now = time.perf_counter()
                        row['events'].append(dict(t=now-start, response=event))
                        if event.get('tokens') and first is None:
                            first = now
                            row['ttft_seconds'] = first-start
                        if event.get('stop'):
                            row['final'] = event
                            break
                row['wall_seconds'] = time.perf_counter()-start
                row['usage_after'] = usage(child.pid)
                row['tokens'] = [token for event in row['events'] for token in event['response'].get('tokens', [])]
                row['text'] = ''.join(event['response'].get('content', '') for event in row['events'])
                emitted = [e['t'] for e in row['events'] if e['response'].get('tokens') and not e['response'].get('stop')]
                row['inter_token_seconds'] = [b-a for a, b in zip(emitted, emitted[1:])]
                row['observed_decode_tokens_per_second'] = ((len(emitted)-1)/(emitted[-1]-emitted[0])
                    if len(emitted) > 1 and emitted[-1] > emitted[0] else None)
                if not row.get('final', {}).get('stop'):
                    raise RuntimeError('Missing terminal SSE event')
                atomic_save(args.output, report)
                print(json.dumps(dict(output=str(args.output), repetition=rep, tokens=len(row['tokens']),
                    ttft=row.get('ttft_seconds'), decode=row['observed_decode_tokens_per_second'],
                    timings=row['final'].get('timings'))), flush=True)
            report['status'] = 'complete'
        except Exception as exc:
            report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        finally:
            report['usage_final'] = usage(child.pid)
            stop.set()
            watcher.join(timeout=6)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            report.update(exit_code=child.returncode, total_seconds=time.perf_counter()-t0,
                swap_final_bytes=psutil.swap_memory().used,
                thermal_after=command_text(['/usr/bin/pmset', '-g', 'therm']))
            atomic_save(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/Users/macbook/Models/GLM-5.3-Flash-Q2_K_S/GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt', default='La capitale de la France est')
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gpu-layers', type=int, default=0)
    p.add_argument('--cpu-moe', action='store_true')
    p.add_argument('--moe-cache-mib', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--n-probs', type=int, default=8)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--max-rss-gib', type=float, default=9)
    p.add_argument('--interposer', type=Path)
    p.add_argument('--mmap-demand', action='store_true')
    p.add_argument('--expert-prefetch', action='store_true')
    p.add_argument('--dense-prefetch', action='store_true')
    p.add_argument('--weights-only', action='store_true', help='Disable opportunistic GPU ops and KV offload')
    p.add_argument('--layer-local-hc', action='store_true', help='Experimental patched GLM HC placement')
    p.add_argument('--cache-lookup-split', action='store_true', help='Experimental patched MoE dependency split')
    p.add_argument('extra', nargs='*')
    args = p.parse_args()
    if args.context < 128 or args.tokens < 1 or args.batch < 1 or not 0 < args.max_rss_gib <= 10:
        p.error('Unsafe or invalid limits')
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = run(args)
    raise SystemExit(0 if report['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
