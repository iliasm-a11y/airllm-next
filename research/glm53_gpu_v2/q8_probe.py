"""Bounded real-byte Q8_0/native-ARM fidelity microprobe, not GLM decoding."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np
from .q8_compatible import Q8CompatibleMatvec
from .q8_oracle import Q8NativeOracle
from .q8_reader import read_rows
from research.glm53_gpu.quantized_probe import compare


def timed(call, repetitions):
    times = []
    for _ in range(repetitions):
        begin = time.perf_counter()
        call()
        times.append(time.perf_counter()-begin)
    return {'wall_seconds': times, 'median_wall_seconds': statistics.median(times)}


def run(args):
    import mlx.core as mx
    mx.set_cache_limit(0)
    mx.set_memory_limit(128 << 20)
    packed, width, metadata = read_rows(args.gguf, args.tensor, args.first_row, args.rows)
    if args.activation_f32:
        if args.activation_f32.stat().st_size != width*4:
            raise ValueError('captured activation must contain exactly width FP32 values')
        x = np.fromfile(args.activation_f32, dtype='<f4')
        origin = {'kind': 'supplied captured FP32', 'file': args.activation_f32.name}
    else:
        x = np.random.default_rng(5305).normal(size=width).astype('f4')
        origin = {'kind': 'synthetic standard normal FP32', 'seed': 5305}
    oracle = Q8NativeOracle(args.libraries)
    native = oracle.matvec(packed, x)
    with Q8CompatibleMatvec(packed, width) as matrix:
        q, scales = matrix.quantize_activations(x)
        cpu_activation = oracle.activation(x)
        activation_exact = bool(np.array_equal(np.asarray(q).view('u1').reshape(-1,32), cpu_activation[:,2:])
            and np.array_equal(np.asarray(scales).view('u1').reshape(-1,2), cpu_activation[:,:2]))
        results = {}
        variants = [('prepared_activation', False, False), ('fused', True, False)]
        if width == 4096:
            variants.append(('shared_activation', False, True))
        for label, fused, shared in variants:
            actual = np.asarray(matrix(x, fused=fused, shared=shared))
            result = compare(actual, native)
            result['host_wall'] = timed(lambda: matrix(x, fused=fused, shared=shared), args.repetitions)
            if args.device_timer:
                from .q8_device_timer import measure
                device, output = measure(args.device_timer, args.work/label, packed, width, x,
                                         fused=fused, shared=shared, repetitions=args.repetitions)
                result['device'] = device
                result['device_output_comparison'] = compare(output, native)
            results[label] = result
        payload = matrix.payload_bytes
    report = {'schema_version':1, 'scope':'Selected original Q8_0 bytes + one activation; ARM nrc=1 component only',
        'measured_at_utc': datetime.now(timezone.utc).isoformat(), 'metadata': metadata,
        'activation': {**origin, 'sha256':hashlib.sha256(x.tobytes()).hexdigest()},
        'activation_bitexact':activation_exact, 'results':results, 'known_payload_bytes':payload,
        'mlx_peak_bytes':mx.get_peak_memory(), 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native_cpu_dylib_sha256':hashlib.sha256((args.libraries/'libggml-cpu.dylib').read_bytes()).hexdigest(),
        'mlx_version':mx.__version__, 'native_cpu_ctypes_one_caller':timed(lambda:oracle.matvec(packed,x),args.repetitions),
        'timing_limits':['prepared warm matrix','native CPU dot called once per row via ctypes; not CPU10','host wall includes submission+synchronization','only device field uses actual GPU timestamps','no full GLM fidelity or decode gain inferred']}
    report['passed'] = activation_exact and all(r['finite'] and r['bit_identical']
        and ('device_output_comparison' not in r or (r['device_output_comparison']['finite'] and r['device_output_comparison']['bit_identical'])) for r in results.values())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('passed','activation_bitexact','known_payload_bytes','mlx_peak_bytes')},indent=2))
    return 0 if report['passed'] else 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',type=Path,required=True)
    p.add_argument('--libraries',type=Path,required=True)
    p.add_argument('--tensor',default='output.weight')
    p.add_argument('--rows',type=int,choices=range(1,1025),default=32)
    p.add_argument('--first-row',type=int,default=0)
    p.add_argument('--activation-f32',type=Path)
    p.add_argument('--repetitions',type=int,choices=range(1,21),default=5)
    p.add_argument('--device-timer',type=Path)
    p.add_argument('--work',type=Path,default=Path('work/q8-v2/device'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run(args)


if __name__=='__main__':
    raise SystemExit(main())
