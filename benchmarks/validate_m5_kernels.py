#!/usr/bin/env python3
"""Admitted, isolated compilation and small FP32 checks of the existing M5 kernels.

This is tensor evidence only. Synchronous host intervals include GPU work and
dispatch/wait overhead. Pure GPU duration remains null unless measured by an
external Metal profiler; an optional capture is an artifact, not a timing result.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))
from airllm.nextgen.experiment_guard import (
    ExperimentLimits, admission, supervise, system_snapshot, validate_output_paths,
)

MIB = 1024**2
LIMITS = ExperimentLimits(rss_bytes=512*MIB, admitted_payload_bytes=256*MIB,
                          timeout_seconds=600, interval_seconds=.05)
ALLOCATOR_BYTES = 128*MIB
ATOL, RTOL = 1e-4, 5e-5


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def shapes(quick=False):
    # Partial row/token tiles and multiple reduction widths are intentional.
    return ([(19, 256, 16), (19, 512, 17)] if quick else
            [(19, 256, batch) for batch in (2, 16, 32, 64, 128)] +
            [(64, 512, 16), (64, 512, 64), (128, 1024, 128)])


def scalar_decode(raw, kind, width):
    """Scalar GGML block-layout oracle, independent of the Metal source strings."""
    import numpy as np
    result = np.zeros((len(raw), width), dtype=np.float32)
    block_size = 144 if kind == 12 else 210
    for row in range(len(raw)):
        for b in range(width // 256):
            p = raw[row, b*block_size:(b+1)*block_size]
            def half(pos):
                return np.frombuffer(p[pos:pos+2].tobytes(), dtype='<f2')[0].astype(np.float32)
            for k in range(256):
                if kind == 12:
                    g, lane = divmod(k, 32)
                    sc = p[4:16].astype(int)
                    scale = sc[g]&63 if g < 4 else (sc[g+4]&15)|((sc[g-4]>>6)<<4)
                    minimum = sc[g+4]&63 if g < 4 else (sc[g+4]>>4)|((sc[g]>>6)<<4)
                    byte = int(p[16+g//2*32+lane])
                    q = byte>>4 if g%2 else byte&15
                    value = np.float32(half(0)*scale)*np.float32(q)-np.float32(half(2)*minimum)
                else:
                    h, part, lane = k//128, (k%128)//32, k%32
                    low = int(p[h*64+part%2*32+lane])
                    high = int(p[128+h*32+lane])
                    q = ((low>>(part//2*4))&15)|(((high>>(part*2))&3)<<4)
                    scale = int(p[192+k//16])
                    if scale >= 128:
                        scale -= 256
                    value = np.float32(half(208)*scale)*np.float32(q-32)
                result[row, b*256+k] = value
    return result


def make_operands(kind, rows, width, batch, grouped, seed=20261027):
    import numpy as np
    rng = np.random.default_rng(seed)
    experts = 2 if grouped else 1
    block = 144 if kind == 12 else 210
    packed = rng.integers(0, 256, (experts, rows, width//256*block), dtype=np.uint8)
    for offset in range(0, width//256*block, block):
        for pos in ([0, 2] if kind == 12 else [208]):
            scales = np.resize(np.array([0., .001, -.002, .125], dtype='<f2'),
                               experts*rows).reshape(experts, rows)
            packed[:, :, offset+pos:offset+pos+2] = scales.view(np.uint8).reshape(experts, rows, 2)
    inputs = rng.normal(size=(experts, batch, width)).astype(np.float32)
    expected = np.stack([inputs[e] @ scalar_decode(packed[e], kind, width).T
                         for e in range(experts)])
    return (packed, inputs, expected) if grouped else (packed[0], inputs[0], expected[0])


def allocation_charge(packed, inputs, expected):
    # Charge host/device packed+inputs, scalar oracle, 6 outputs and tile staging.
    experts = packed.shape[0] if packed.ndim == 3 else 1
    rows, width = packed.shape[-2], inputs.shape[-1]
    return 2*(packed.nbytes+inputs.nbytes)+experts*rows*width*4+6*expected.nbytes+4096


def comparison(actual, expected):
    import numpy as np
    valid = actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    return {'accepted': bool(valid and np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
            'max_absolute_error': float(np.max(np.abs(actual-expected))) if valid else None,
            'atol': ATOL, 'rtol': RTOL}


def sources():
    paths = [Path(__file__), ROOT/'air_llm/airllm/nextgen/gguf_m5.py',
             ROOT/'air_llm/airllm/nextgen/gguf_metal.py',
             ROOT/'air_llm/airllm/nextgen/experiment_guard.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_worker(args):
    # Even invoking the private entry point directly cannot bypass admission.
    admission(LIMITS)
    token = os.environ.get('AIRLLM_M5_KERNEL_SUPERVISOR_TOKEN', '')
    if len(token) != 64 or any(c not in '0123456789abcdef' for c in token):
        raise RuntimeError('The kernel worker requires its isolated supervisor')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise RuntimeError('Strict FP32 requires MLX_ENABLE_TF32=0 before MLX import')
    import numpy as np
    import mlx.core as mx
    from airllm.nextgen import gguf_m5, gguf_metal
    if not mx.metal.is_available():
        raise RuntimeError('Metal is unavailable in this process')
    mx.set_memory_limit(ALLOCATOR_BYTES)
    mx.set_cache_limit(0)
    mx.reset_peak_memory()
    report = {'schema': 'airllm-m5-kernels-v1', 'status': 'running',
              'scope': 'small synthetic packed tensor compilation/numerics only',
              'real_checkpoint_validated': False, 'sources_before': sources(),
              'limits': asdict(LIMITS), 'before': system_snapshot(),
              'versions': {name: importlib.metadata.version(name) for name in ('mlx', 'numpy', 'psutil')},
              'device_info': mx.device_info(mx.gpu), 'strict_fp32': True,
              'allocator_guideline_bytes': ALLOCATOR_BYTES,
              'allocator_note': 'MLX limit is a guideline; explicit charge and global RSS/reserve/swap/pressure supervision also apply',
              'gpu_seconds': None, 'gpu_timing_reason': 'MLX 0.32.4 Python API exposes no per-command-buffer GPU timestamps',
              'host_timing_scope': 'dispatch plus synchronous eval plus GPU synchronize; includes GPU duration',
              'cases': []}
    atomic_json(args.output, report)
    capture_started = False
    if args.capture:
        mx.metal.start_capture(str(args.capture))
        capture_started = True
        report['capture'] = str(args.capture)
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            variants = [('reuse', 4), ('tensorops', 4)] if args.quick else [
                ('reuse', n) for n in (2, 4, 8, 16)] + [('tensorops', 4)]
            for kind in (12, 14):
                for grouped in (False, True):
                    for rows, width, batch in shapes(args.quick):
                        admission(LIMITS)
                        packed, inputs, expected = make_operands(kind, rows, width, batch, grouped)
                        charge = allocation_charge(packed, inputs, expected)
                        if charge > ALLOCATOR_BYTES//2:
                            raise MemoryError('Explicit tensor workspace exceeds conservative allocator allowance')
                        data, x = mx.array(packed), mx.array(inputs)
                        mx.eval(data, x)
                        mx.synchronize()
                        def baseline():
                            return gguf_metal._kernel(kind, grouped)(inputs=[data, x],
                                grid=(rows*32, batch, 2 if grouped else 1), threadgroup=(32, 1, 1),
                                output_shapes=[expected.shape], output_dtypes=[mx.float32])[0]
                        ref_started = time.perf_counter()
                        reference = baseline()
                        mx.eval(reference)
                        mx.synchronize()
                        reference_compile_and_first = time.perf_counter()-ref_started
                        ref_host = np.array(reference)
                        ref_check = comparison(ref_host, expected)
                        if not ref_check['accepted']:
                            raise RuntimeError('Existing GPU reference differs from independent scalar oracle')
                        for backend, tile in variants:
                            admission(LIMITS)
                            case = {'kind': kind, 'grouped': grouped, 'rows': rows, 'width': width,
                                    'batch': batch, 'backend': backend, 'token_tile': tile,
                                    'explicit_workspace_charge_bytes': charge,
                                    'reference_vs_scalar': ref_check,
                                    'reference_compile_and_first_host_seconds': reference_compile_and_first,
                                    'gpu_seconds': None, 'selected': False}
                            def candidate():
                                return gguf_m5.experimental_matmul(data, x, kind, backend=backend,
                                                                   grouped=grouped, token_tile=tile)
                            started = time.perf_counter()
                            try:
                                trial = candidate()
                                mx.eval(trial)
                                mx.synchronize()
                                case['compile_and_first_host_seconds'] = time.perf_counter()-started
                                trial_host = np.array(trial)
                                case['candidate_vs_scalar'] = comparison(trial_host, expected)
                                case['candidate_vs_reference'] = comparison(trial_host, ref_host)
                                del trial, trial_host
                                case['compiled_and_executed'] = True
                                valid = case['candidate_vs_scalar']['accepted'] and case['candidate_vs_reference']['accepted']
                                if valid:
                                    timings = [[], []]
                                    cpu = [[], []]
                                    order = []
                                    for repetition in range(args.repetitions):
                                        pair = (0, 1) if repetition%2 == 0 else (1, 0)
                                        order.append(list(pair))
                                        for index in pair:
                                            mx.synchronize()
                                            start, cpu_start = time.perf_counter(), time.process_time()
                                            value = baseline() if index == 0 else candidate()
                                            mx.eval(value)
                                            mx.synchronize()
                                            cpu[index].append(time.process_time()-cpu_start)
                                            timings[index].append(time.perf_counter()-start)
                                            del value
                                    case.update(host_seconds=timings, host_cpu_process_seconds=cpu,
                                                timing_order=order,
                                                host_median_speedup=statistics.median(timings[0])/statistics.median(timings[1]))
                                    case['selected'] = case['host_median_speedup'] > 1.03
                                    case['reason'] = 'measured_host_speed' if case['selected'] else 'not_faster'
                                else:
                                    case['reason'] = 'numerical_rejection'
                            except (RuntimeError, ValueError, TypeError) as error:
                                case.update(compiled_and_executed=False, reason='compilation_or_execution_unavailable',
                                            error=f'{type(error).__name__}: {error}',
                                            compile_and_first_host_seconds=time.perf_counter()-started)
                            case['mlx_active_bytes'] = mx.get_active_memory()
                            case['mlx_cache_bytes'] = mx.get_cache_memory()
                            case['mlx_peak_bytes'] = mx.get_peak_memory()
                            if case['mlx_active_bytes']+case['mlx_cache_bytes'] > ALLOCATOR_BYTES:
                                raise MemoryError('Observed MLX allocator exceeds conservative byte allowance')
                            report['cases'].append(case)
                            atomic_json(args.output, report)
                        del reference, ref_host, data, x, packed, inputs, expected
                        mx.synchronize()
                        mx.clear_cache()
    finally:
        if capture_started:
            mx.metal.stop_capture()
    report.update(status='completed', after=system_snapshot(), sources_after=sources(),
                  mlx_peak_bytes=mx.get_peak_memory())
    report['sources_unchanged'] = report['sources_before'] == report['sources_after']
    report['all_candidates_numerically_valid'] = all(c.get('candidate_vs_scalar', {}).get('accepted', False)
                                                   for c in report['cases'])
    report['selected_case_count'] = sum(c['selected'] for c in report['cases'])
    atomic_json(args.output, report)
    if not report['sources_unchanged']:
        raise RuntimeError('Kernel sources changed during validation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=6)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repetitions < 2 or args.repetitions > 100:
        parser.error('repetitions must be between 2 and 100')
    args.output = args.output.expanduser().absolute()
    if args.capture:
        args.capture = args.capture.expanduser().absolute()
        if args.capture.suffix != '.gputrace' or args.capture.exists() or args.capture.is_symlink():
            parser.error('capture must be a new .gputrace path')
    destinations = validate_output_paths(args.output, protected=[Path(__file__), *[
        ROOT/name for name in sources()]], extra_outputs=[args.output.with_name(args.output.name+'.partial'),
        *([args.capture] if args.capture else [])])
    # Active supervisor log and guard belong to the parent during worker entry.
    checked = [args.output] if args.worker else destinations
    if any(p.exists() or p.is_symlink() for p in checked):
        raise FileExistsError('Refusing to overwrite source or prior kernel evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args)
        return 0
    command = [sys.executable, str(Path(__file__).resolve()), '--output', str(args.output),
               '--repetitions', str(args.repetitions), '--worker']
    if args.quick:
        command.append('--quick')
    if args.capture:
        command.extend(['--capture', str(args.capture)])
    env = dict(os.environ, MLX_ENABLE_TF32='0',
               AIRLLM_M5_KERNEL_SUPERVISOR_TOKEN=secrets.token_hex(32))
    if args.capture:
        env['MTL_CAPTURE_ENABLED'] = '1'
    try:
        supervise(command, args.output, LIMITS, env=env)
    except MemoryError as error:
        atomic_json(args.output, {'schema': 'airllm-m5-kernels-v1', 'status': 'deferred',
            'model_worker_launched': False, 'gpu_worker_launched': False, 'reason': str(error),
            'limits': asdict(LIMITS), 'snapshot': system_snapshot(),
            'sources_before': sources(), 'cases': [], 'gpu_seconds': None,
            'real_checkpoint_validated': False})
        print(f'Kernel validation deferred: {error}', file=sys.stderr)
        return 75
    except RuntimeError as error:
        # Preserve partial case evidence while making interruption/failure
        # unambiguous to an independent caller. A partial run is not complete.
        report = json.loads(args.output.read_text()) if args.output.exists() else {
            'schema': 'airllm-m5-kernels-v1', 'cases': [], 'gpu_seconds': None,
            'real_checkpoint_validated': False}
        report.update(status='failed', error=str(error), after=system_snapshot())
        atomic_json(args.output, report)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
