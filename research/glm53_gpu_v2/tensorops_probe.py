"""Bounded MPP int8 partial-dot probe; preparation never creates a Metal device."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROWS, WIDTH, COLUMNS = 16, 32, 8
MAX_PAYLOAD = 1 << 20
LOCK = Path('/tmp/airllm-glm53-benchmark.lock')
HERE = Path(__file__).resolve().parent


def validate_inputs(weights, activation):
    if not isinstance(weights, np.ndarray) or weights.dtype != np.dtype('int8') or weights.shape != (ROWS, WIDTH):
        raise ValueError('weights must be int8[16,32]')
    if not isinstance(activation, np.ndarray) or activation.dtype != np.dtype('int8') or activation.shape != (WIDTH,):
        raise ValueError('activation must be int8[32]')
    if not weights.flags.c_contiguous or not activation.flags.c_contiguous:
        raise ValueError('inputs must be contiguous')


def masked_columns(activation):
    if not isinstance(activation, np.ndarray) or activation.dtype != np.dtype('int8') or activation.shape != (WIDTH,):
        raise ValueError('activation must be int8[32]')
    b = np.zeros((WIDTH, COLUMNS), dtype=np.int8)
    for k in range(WIDTH):
        b[k, (k % 16) // 4] = activation[k]
    return b


def partial_reference(weights, activation):
    """Independent scalar Python-int grouping, without a matrix multiply oracle."""
    validate_inputs(weights, activation)
    result = np.zeros((ROWS, COLUMNS), dtype='<i4')
    for row in range(ROWS):
        for part in range(4):
            total = 0
            for half in (0, 16):
                for j in range(part * 4, part * 4 + 4):
                    total += int(weights[row, half + j]) * int(activation[half + j])
            result[row, part] = total
    return result


def compare(actual, expected):
    if not isinstance(actual, np.ndarray) or actual.dtype != np.dtype('<i4') or actual.shape != (ROWS, COLUMNS):
        return {'passed': False, 'reason': 'output must be int32[16,8]'}
    if not isinstance(expected, np.ndarray) or expected.dtype != np.dtype('<i4') or expected.shape != actual.shape:
        return {'passed': False, 'reason': 'reference must be int32[16,8]'}
    delta = actual.astype(np.int64) - expected.astype(np.int64)
    return {'passed': bool(np.array_equal(actual, expected)),
            'elements': int(actual.size), 'mismatches': int(np.count_nonzero(delta)),
            'max_abs_integer_error': int(np.abs(delta).max())}


def fixture_cases(seed=5305):
    rng = np.random.default_rng(seed)
    w = rng.integers(-128, 128, (ROWS, WIDTH), dtype=np.int8)
    x = rng.integers(-128, 128, WIDTH, dtype=np.int8)
    yield 'seeded_full_int8', w, x
    yield 'zeros', w.copy(), np.zeros(WIDTH, dtype=np.int8)
    w = np.empty((ROWS, WIDTH), dtype=np.int8)
    w[::2], w[1::2] = -128, 127
    x = np.tile(np.array([-128, 127], dtype=np.int8), WIDTH // 2)
    yield 'signed_extremes', w, x


def prepare(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cases, expected = [], {}
    charged = 0
    for name, weights, activation in fixture_cases():
        validate_inputs(weights, activation)
        b, ref = masked_columns(activation), partial_reference(weights, activation)
        # Account input arrays, device copies, reference, output and readback together.
        charged += 2 * (weights.nbytes + b.nbytes) + activation.nbytes + 3 * ref.nbytes
        if charged > MAX_PAYLOAD:
            raise MemoryError('fixture/device/readback payload exceeds 1MiB')
        wp, bp, output = (directory / f'{name}.{suffix}' for suffix in ('weights.i8', 'masked.i8', 'out.i32'))
        weights.tofile(wp)
        b.tofile(bp)
        cases.append({'name': name, 'weights': str(wp), 'masked': str(bp), 'output': str(output),
                      'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
                      'masked_sha256': hashlib.sha256(b.tobytes()).hexdigest()})
        expected[name] = ref
    manifest = directory / 'job.json'
    manifest.write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    return manifest, expected, charged


def accepted(report):
    native = report.get('native', {})
    cases = report.get('comparisons', {})
    return (report.get('mode') == 'execute' and report.get('host_compiled') is True
            and report.get('process_returncode') == 0 and report.get('guard_passed') is True
            and native.get('shader_library_compiled') is True and native.get('pipeline_compiled') is True
            and native.get('device_execution_attempted') is True and native.get('commands_completed') == 3
            and set(cases) == {'seeded_full_int8', 'zeros', 'signed_extremes'}
            and all(c.get('passed') is True and c.get('elements') == ROWS * COLUMNS for c in cases.values()))


def invoke_guarded(command, *, timeout=45):
    """Small process guard supplements fixed payload admission; never substitutes a GPU clock."""
    import psutil
    if psutil.virtual_memory().available < 2 << 30:
        raise MemoryError('less than 2GiB available before bounded probe')
    initial_swap = psutil.swap_memory().used
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'AIRLLM_METAL_', 'AIRLLM_GLM_'))}
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    peak_rss, peak_swap, reason = 0, 0, None
    while process.poll() is None:
        try:
            resident = psutil.Process(process.pid).memory_info().rss
            peak_rss = max(peak_rss, resident)
        except psutil.NoSuchProcess:
            pass
        peak_swap = max(peak_swap, max(0, psutil.swap_memory().used - initial_swap))
        if peak_rss > 1 << 30:
            reason = 'process RSS exceeded 1GiB'
        elif peak_swap > 8 << 20:
            reason = 'system swap growth exceeded 8MiB'
        elif psutil.virtual_memory().available < 1 << 30:
            reason = 'available system memory dropped below 1GiB'
        elif time.monotonic() - started > timeout:
            reason = 'probe timeout'
        if reason:
            process.kill()
            break
        time.sleep(0.02)
    stdout, stderr = process.communicate(timeout=5)
    return process.returncode, stdout, stderr, {
        'guard_passed': reason is None, 'guard_failure': reason,
        'sampled_peak_rss_bytes': peak_rss, 'sampled_system_swap_growth_bytes': peak_swap,
        'wall_seconds': time.monotonic() - started,
        'guard_limits': {'rss_bytes': 1 << 30, 'swap_growth_bytes': 8 << 20,
                         'available_memory_floor_bytes': 1 << 30, 'timeout_seconds': timeout}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'compile', 'execute'), default='prepare')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = {'schema_version': 1, 'mode': args.mode, 'accepted': False,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
              'scope': 'three fixed int8[16,32] by int8[32,8] partial-dot fixtures; no floating reduction/model test',
              'host_compiled': False, 'device_execution_attempted': False}
    try:
        manifest, expected, charged = prepare(args.directory)
        source = HERE / 'kernels_tensorops/int8_partials.metal'
        report.update(payload_charged_bytes=charged, shader_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      host_source_sha256=hashlib.sha256((HERE / 'tensorops_native.mm').read_bytes()).hexdigest())
        if args.mode != 'prepare':
            with LOCK.open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                binary = args.directory.resolve() / 'tensorops-native'
                command = ['xcrun', 'clang++', '-std=c++17', '-O2', '-fobjc-arc',
                           str(HERE / 'tensorops_native.mm'), '-framework', 'Foundation', '-framework', 'Metal',
                           '-o', str(binary)]
                start = time.monotonic()
                built = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
                report['host_compile_wall_seconds'] = time.monotonic() - start
                report['host_compile_returncode'] = built.returncode
                report['host_compile_diagnostics'] = built.stderr[-12000:]
                if built.returncode:
                    raise RuntimeError('Objective-C host adapter compilation failed')
                report['host_compiled'] = True
                sdk = subprocess.run(['xcrun', '--show-sdk-version'], capture_output=True, text=True, timeout=5)
                report['sdk_version'] = sdk.stdout.strip()
                mode = '--execute' if args.mode == 'execute' else '--compile-only'
                code, stdout, stderr, guard = invoke_guarded([str(binary), mode, str(source), str(manifest)])
                report.update(guard)
                report.update(process_returncode=code, process_stderr=stderr[-12000:])
                try:
                    report['native'] = json.loads(stdout)
                    report['device_execution_attempted'] = report['native'].get('device_execution_attempted') is True
                    static_bytes = report['native'].get('static_threadgroup_memory_bytes', 0)
                    if type(static_bytes) is not int or static_bytes < 0:
                        raise RuntimeError('invalid static threadgroup memory report')
                    report['payload_plus_static_threadgroup_bytes'] = charged + static_bytes
                    if charged + static_bytes > MAX_PAYLOAD:
                        raise MemoryError('fixture/device/static scratch exceeds 1MiB')
                except json.JSONDecodeError:
                    report['native_output_error'] = stdout[-12000:]
                if code or not guard['guard_passed']:
                    raise RuntimeError('native probe failed; inspect staged diagnostics')
                if args.mode == 'execute':
                    comparisons = {}
                    for case in json.loads(manifest.read_text())['cases']:
                        path = Path(case['output'])
                        if path.stat().st_size != ROWS * COLUMNS * 4:
                            raise RuntimeError('output byte count differs from fixed int32[16,8] contract')
                        actual = np.fromfile(path, dtype='<i4').reshape(ROWS, COLUMNS)
                        comparisons[case['name']] = compare(actual, expected[case['name']])
                    report['comparisons'] = comparisons
        report['accepted'] = accepted(report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': args.mode, 'accepted': report['accepted'], 'output': str(args.output),
                      'error': report.get('error'), 'native_stage': report.get('native', {}).get('stage')}))
    if report.get('error'):
        return 2
    if args.mode == 'execute' and not report['accepted']:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
