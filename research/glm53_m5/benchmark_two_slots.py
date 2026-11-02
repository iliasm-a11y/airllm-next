#!/usr/bin/env python3
"""Experimental two-slot pilot, not production or full-logit qualification.

Sequential then concurrent order shares one process; OS cache is uncontrolled.
Dense prefetch is bounded by the explicit column limit; batching changes eligibility.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))


def probability_rows(events):
    rows = [p for e in events for p in e.get('completion_probabilities', e.get('probs', []))]
    result = []
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Invalid logprob row')
        top = row.get('top_logprobs', [])
        valid = lambda p: isinstance(p, dict) and type(p.get('id')) is int and p['id'] >= 0 and type(p.get('logprob')) in (int, float) and math.isfinite(p['logprob'])
        if not isinstance(top, list) or len(top) != 8 or not all(valid(p) for p in [row] + top) or len({p['id'] for p in top}) != 8:
            raise ValueError('Missing top8 finite logprobs; pilot cannot verify fidelity')
        result.append((row['id'], row['logprob'], [(p['id'], p['logprob']) for p in top]))
    return result


def counter_delta(before, after):
    return {k: after[k]-before[k] for k in ('user_ticks', 'system_ticks', 'pageins',
        'storage_read_bytes', 'storage_write_bytes')} if before and after else None


def validate_slot(event, expected):
    # Pinned server partial responses leave id_slot at its -1 default.
    value = event.get('id_slot')
    if not event.get('stop') and ('id_slot' not in event or type(value) is int and value == -1):
        return
    if type(value) is not int or value != expected:
        raise ValueError('Requested slot was not respected in terminal or identified partial response')


def run(args, lock):
    import psutil
    from benchmark_glm_native import atomic_save, pressure, usage
    from run_guarded_probe import stop_owned
    if args.output.exists(): raise ValueError('Refusing to overwrite pilot evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
    def http(endpoint, body=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}' + endpoint,
            data=data, headers=headers), timeout=timeout)
    stopped, interrupted = threading.Event(), threading.Event()
    report = {'schema_version': 1, 'status': 'starting', 'phases': [], 'samples': [], 'guard_violation': None,
        'scope': 'Two-slot pilot only: tokens and reported top8 logprobs, no raw full logits or internal routes.',
        'configuration': {'context_total': 1024, 'context_per_job_expected': 512, 'slots': 2, 'batch': 8, 'threads': 10, 'dense_prefetch_max_cols': args.dense_prefetch_max_cols},
        'cache_scope': 'Same process, cache_prompt=false. Sequential phase precedes concurrent phase; OS file-cache/order effects uncontrolled.',
        'dense_prefetch_scope': f'Patched hint requires 1<=ne11<={args.dense_prefetch_max_cols}. Both phases use the same explicit limit; batching can change hint eligibility. System-throughput pilot, not isolated batching qualification.',
        'logical_weight_reads_bytes': None, 'physical_nand_reads_bytes': None,
        'counter_scope': 'libproc process OS-accounted bytes/pageins and Mach CPU ticks; concurrent per-request deltas overlap and cannot be attributed to individual jobs.',
        'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'peak_rss_bytes': 0,
        'model_path': str(args.model.resolve()), 'target_revision': '119034bc3d9d76eb41f91508e6f9fb19fde42672'}
    started = time.perf_counter(); initial_swap = psutil.swap_memory().used
    previous = {sig: signal.signal(sig, lambda *_: interrupted.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    with tempfile.TemporaryDirectory(prefix='glm-two-slot-') as private:
        key_file = Path(private) / 'key'; key_file.write_text(key + '\n'); key_file.chmod(0o600)
        command = [str(args.binary.resolve()), '--offline', '--fit', 'off', '-m', str(args.model.resolve()),
            '--load-mode', 'mmap', '--no-repack', '--no-warmup', '-c', '1024', '-np', '2', '--no-kv-unified',
            '-b', '8', '-ub', '8', '-t', '10', '-tb', '10', '-ngl', '0', '--device', 'none', '--no-op-offload',
            '--host', '127.0.0.1', '--port', str(port), '--api-key-file', str(key_file), '--no-webui',
            '--no-mmproj', '--no-context-shift', '--slots', '--perf', '--log-colors', 'off']
        report['command'] = command
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'AIRLLM_')) and k != 'DYLD_INSERT_LIBRARIES'}
        env.update(AIRLLM_MMAP_DEMAND='1', AIRLLM_EXPERT_PREFETCH='1', AIRLLM_DENSE_PREFETCH='1')
        env['AIRLLM_DENSE_PREFETCH_MAX_COLS'] = str(args.dense_prefetch_max_cols)
        child = None; watcher = None
        atomic_save(args.output, report)
        try:
            with args.output.with_suffix('.log').open('x') as log:
                child = subprocess_spawn(command, env, log, lock)
                report['owned_pid'] = child.pid; process = psutil.Process(child.pid)
                def watch():
                    try:
                        while not stopped.wait(.5) and child.poll() is None:
                            row = {'t': time.perf_counter()-started, 'rss': process.memory_info().rss,
                                'swap_growth': max(0, psutil.swap_memory().used-initial_swap),
                                'pressure': pressure(), 'process': usage(child.pid),
                                'cpu_seconds': sum(process.cpu_times()[:2])}
                            report['samples'].append(row); report['peak_rss_bytes'] = max(report['peak_rss_bytes'], row['rss'])
                            if row['pressure'] is None: raise RuntimeError('Memory pressure measurement unavailable')
                            violation = ('interrupted' if interrupted.is_set() else 'rss' if row['rss'] > 9*1024**3 else
                                'swap' if row['swap_growth'] > 64*1024**2 else 'pressure' if row['pressure'] == 4 else
                                'timeout' if row['t'] > 1200 else None)
                            atomic_save(args.output.with_suffix('.progress.json'), row)
                            if violation: report['guard_violation'] = violation; stop_owned(child); return
                    except psutil.NoSuchProcess: pass
                    except Exception as exc: report['guard_violation'] = str(exc); stop_owned(child)
                watcher = threading.Thread(target=watch, daemon=True); watcher.start()
                while True:
                    if child.poll() is not None: raise RuntimeError('Unsupported architecture/context or server stopped; inspect log')
                    try:
                        with http('/health', timeout=1) as response:
                            if response.status == 200: break
                    except (OSError, urllib.error.HTTPError): pass
                    time.sleep(.2)
                report['load_seconds'] = time.perf_counter()-started
                with http('/slots') as response: report['slots'] = json.load(response)
                if len(report['slots']) != 2 or any(s.get('n_ctx') != 512 for s in report['slots']):
                    raise RuntimeError('Server does not expose two independent 512-token slots')
                def job(index, barrier=None):
                    if barrier: barrier.wait(timeout=5)
                    row = {'slot': index, 'prompt': args.prompts[index], 'events': [], 'usage_before': usage(child.pid)}
                    t0 = time.perf_counter(); arrivals = []
                    try:
                        body = dict(prompt=args.prompts[index], id_slot=index, n_predict=8, temperature=0, seed=42,
                            stream=True, cache_prompt=False, return_tokens=True, n_probs=8, timings_per_token=True)
                        with http('/completion', body, timeout=1200) as response:
                            for raw in response:
                                line = raw.decode().strip()
                                if not line.startswith('data: ') or line == 'data: [DONE]': continue
                                event = json.loads(line[6:]); now = time.perf_counter(); row['events'].append(event)
                                validate_slot(event, index)
                                arrivals.extend([now] * len(event.get('tokens', [])))
                                if event.get('stop'): row['final'] = event; break
                        row['tokens'] = [t for e in row['events'] for t in e.get('tokens', [])]
                        row['text'] = ''.join(e.get('content', '') for e in row['events'])
                        row['top8'] = probability_rows(row['events'])
                        if not row.get('final', {}).get('stop') or not row['tokens'] or len(row['top8']) != len(row['tokens']):
                            raise ValueError('Incomplete native token/logprob stream')
                    except Exception as exc: row['error'] = type(exc).__name__ + ': ' + str(exc)
                    row.update(wall_seconds=time.perf_counter()-t0, ttft_wall_seconds=arrivals[0]-t0 if arrivals else None,
                        token_arrivals_monotonic=arrivals, usage_after=usage(child.pid))
                    row['process_counter_delta'] = counter_delta(row['usage_before'], row['usage_after'])
                    return row
                for concurrent in (False, True):
                    phase = {'kind': 'concurrent' if concurrent else 'sequential', 'usage_before': usage(child.pid)}
                    t0 = time.perf_counter()
                    if concurrent:
                        barrier = threading.Barrier(3)
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            futures = [pool.submit(job, i, barrier) for i in range(2)]; barrier.wait(timeout=5)
                            phase['jobs'] = [f.result() for f in futures]
                    else: phase['jobs'] = [job(i) for i in range(2)]
                    wall = time.perf_counter()-t0; jobs = phase['jobs']
                    arrivals = [r['token_arrivals_monotonic'] for r in jobs]
                    span = max(r[-1] for r in arrivals)-min(r[0] for r in arrivals) if all(arrivals) else None
                    phase.update(wall_seconds=wall, useful_tokens=sum(len(r.get('tokens', [])) for r in jobs),
                        usage_after=usage(child.pid), decode_pool_span_seconds=span,
                        decode_pool_numerator=sum(max(0, len(r)-1) for r in arrivals),
                        decode_pool_scope='Earliest first-token arrival to latest last-token arrival; numerator excludes each job first token. Span includes overlap/idle/prefill of the other job; not pure kernel decode.')
                    phase['useful_tokens_per_wall_second'] = phase['useful_tokens']/wall
                    phase['process_counter_delta'] = counter_delta(phase['usage_before'], phase['usage_after'])
                    phase['observed_decode_pool_tokens_per_second'] = phase['decode_pool_numerator']/span if span and span > 0 else None
                    report['phases'].append(phase); atomic_save(args.output, report)
                    if any('error' in r for r in jobs): raise RuntimeError('A pilot request failed; no concurrency fallback')
                sequential, concurrent = report['phases']
                report['strict_tokens_and_top8_equal'] = [a['tokens'] == b['tokens'] and json.dumps(a['top8']) == json.dumps(b['top8'])
                    for a, b in zip(sequential['jobs'], concurrent['jobs'])]
                report['pilot_useful_throughput_ratio'] = concurrent['useful_tokens_per_wall_second']/sequential['useful_tokens_per_wall_second']
                report['status'] = 'complete_pilot' if all(report['strict_tokens_and_top8_equal']) else 'fidelity_rejected_pilot'
        except Exception as exc: report.update(status='failed_pilot', error=type(exc).__name__ + ': ' + str(exc))
        finally:
            if child is not None: report['usage_final'] = usage(child.pid); stop_owned(child)
            stopped.set()
            if watcher: watcher.join(timeout=6)
            if report['guard_violation']: report['status'] = 'failed_pilot'
            report.update(server_exit_code=child.returncode if child else None, total_seconds=time.perf_counter()-started)
            for sig, handler in previous.items(): signal.signal(sig, handler)
            atomic_save(args.output, report)
    return report


def subprocess_spawn(command, env, log, lock):
    import subprocess
    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, pass_fds=(lock.fileno(),))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True); parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', nargs=2, default=['La capitale de la France est', 'Un petit programme Python peut'])
    parser.add_argument('--dense-prefetch-max-cols', type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args(argv)
    if args.prompts[0] == args.prompts[1]: parser.error('Two distinct free-generation prompts are required')
    if args.model.name != 'GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf' or not args.model.is_file(): parser.error('Pinned target first GGUF shard required')
    return args


def main():
    args = parse_args()
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); report = run(args, lock)
    print(json.dumps({k: report.get(k) for k in ('status', 'guard_violation', 'strict_tokens_and_top8_equal', 'pilot_useful_throughput_ratio')}, indent=2))
    return 0 if report['status'] == 'complete_pilot' else 2


if __name__ == '__main__': raise SystemExit(main())
