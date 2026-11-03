"""Bounded multi-token Q8_0/ARM nrc=2 component, never a model engine claim."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import numpy as np
from research.glm53_gpu_v2.q8_compatible import validate_activation, validate_matrix
from research.glm53_gpu_v2.q8_oracle import Q8NativeOracle


def representative_token_counts(token_max=32):
    if type(token_max) is not int or token_max not in (32, 64, 128):
        raise ValueError('representative token maximum must be 32, 64 or 128')
    return tuple(tokens for tokens in (8, 16, 32, 64, 128) if tokens <= token_max)


def q8_buffer_bytes(rows, width, tokens):
    """Exact timer buffers, excluding CPU oracle arrays, scratch and driver RSS."""
    if (any(type(value) is not int or value <= 0 for value in (rows, width, tokens))
            or width % 32):
        raise ValueError('buffer dimensions must be positive integers; width is a multiple of 32')
    return {'packed':rows * (width // 32) * 34, 'x':tokens * width * 4,
        'shape':12, 'qs':tokens * width, 'scales':tokens * (width // 32) * 2,
        'output':tokens * rows * 4}


def numerical_comparison(actual, expected):
    """Keep component bit fidelity separate from the existing Metal tolerances."""
    if (actual.shape != expected.shape or actual.dtype != np.float32
            or expected.dtype != np.float32 or not actual.size):
        raise ValueError('comparison requires equal nonempty FP32 shapes')
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    bit_mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    if finite:
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        tolerance = 1e-3 + 1e-4 * np.abs(expected.astype(np.float64))
        tolerance_mismatches = int(np.count_nonzero(error > tolerance))
    else:
        error = None
        tolerance_mismatches = int(actual.size)
    return {'finite':finite, 'bit_identical':finite and bit_mismatches == 0,
        'bit_mismatches':bit_mismatches, 'elements':int(actual.size),
        'max_abs':float(error.max()) if finite else None,
        'rmse':float(np.sqrt(np.mean(error*error))) if finite else None,
        'metal_tolerance':{'atol':1e-3, 'rtol':1e-4, 'reference':'native CPU',
            'passed':finite and tolerance_mismatches == 0,
            'elements_outside_tolerance':tolerance_mismatches,
            'scope':'component outputs only; no routes, model logits or recurrent states'}}


def validate(packed, x, budget_bytes=64 << 20):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.dtype != np.float32 or not x.flags.c_contiguous:
        raise ValueError('activations must be contiguous float32 [tokens,width]')
    tokens, width = x.shape
    validate_matrix(packed, width, budget_bytes)
    if tokens not in (2, 8, 16, 32, 64, 128) or packed.shape[0] % 2:
        raise ValueError('nrc=2 requires even rows and admitted token counts')
    for row in x:
        validate_activation(row, width)
    buffers = q8_buffer_bytes(packed.shape[0], width, tokens)
    required = 2 * (sum(buffers.values()) - buffers['shape'])
    if required > budget_bytes:
        raise MemoryError(f'known payload {required} exceeds {budget_bytes}')
    return required


def native_nrc2(oracle, packed, x):
    validate(packed, x)
    if not oracle.cpu.ggml_cpu_has_matmul_int8():
        raise RuntimeError('native nrc=2 requires ARM INT8 matmul support')
    tokens, width = x.shape
    q = np.stack([oracle.activation(a) for a in x])
    rows = packed.shape[0]
    output = np.full((tokens, rows), np.nan, np.float32)
    for token in range(0, tokens, 2):
        for row in range(0, rows, 2):
            oracle.dot(width, output[token, row:].ctypes.data, rows,
                       packed[row:].ctypes.data, packed.strides[0],
                       q[token:].ctypes.data, q.strides[0], 2)
    if not np.isfinite(output).all():
        raise RuntimeError('native nrc=2 oracle did not fill all outputs')
    return q, output


def measure(timer, directory, packed, x, repetitions=3):
    payload = validate(packed, x)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tokens, width = x.shape
    rows = packed.shape[0]
    shape = np.array([width, rows, tokens], np.uint32)
    buffers = []
    for name, value in [('packed', packed), ('x', x), ('shape', shape)]:
        path = root / (name + '.bin')
        path.write_bytes(value.tobytes())
        buffers.append({'bytes': value.nbytes, 'input': str(path)})
    buffers += [{'bytes': x.size, 'output': str(root/'qs.bin')},
                {'bytes': x.size//32*2, 'output': str(root/'scales.bin')},
                {'bytes': tokens*rows*4, 'output': str(root/'out.bin')}]
    source = Path(__file__).with_name('metal_q8.metal')
    stages = [
        {'entry':'q8_batch_quantize', 'bindings':[1,3,4,2], 'grid':[x.size,1,1], 'threadgroup':[32,1,1]},
        {'entry':'q8_batch_dot', 'bindings':[0,3,4,5,2], 'grid':[(rows+7)//8*8,(tokens+7)//8*8,1], 'threadgroup':[8,8,1]},
    ]
    manifest = root/'job.json'
    manifest.write_text(json.dumps({'source':str(source.resolve()), 'buffers':buffers,
        'stages':stages, 'repetitions':repetitions, 'warmups':1}, indent=2)+'\n')
    completed = subprocess.run([str(Path(timer).resolve()), str(manifest)], capture_output=True, text=True, timeout=60)
    (root/'timer-stdout.json').write_text(completed.stdout)
    (root/'timer-stderr.log').write_text(completed.stderr)
    (root/'timer-exit.json').write_text(json.dumps({'returncode':completed.returncode})+'\n')
    if completed.returncode:
        raise RuntimeError(completed.stderr[:3000])
    report = json.loads(completed.stdout)
    report.update(declared_prepared_data_accounting_bytes=payload,
        timer_buffer_bytes=q8_buffer_bytes(rows,width,tokens),
        timer_buffer_total_bytes=sum(b['bytes'] for b in buffers),
        payload_scope='declared prepared-data accounting; excludes CPU graph scratch, oracle copies and driver RSS',
        matmul_dimensions={'M_rows':rows,'N_tokens':tokens,'K_width':width},
        dot_dispatches_per_batch=1, weight_reuse_tokens_per_threadgroup=min(8,tokens),
        static_threadgroup_bytes=576, shader_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    q = np.fromfile(root/'qs.bin', np.int8).reshape(tokens,width//32,32)
    scales = np.fromfile(root/'scales.bin', np.uint16).reshape(tokens,width//32)
    output = np.fromfile(root/'out.bin', np.float32).reshape(tokens,rows)
    return report, q, scales, output


def guard_sample():
    import psutil
    power = subprocess.check_output(['pmset','-g','batt'],text=True,timeout=5)
    pressure = int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True,timeout=5))
    sample = {'ac_power':"'AC Power'" in power, 'pressure_level':pressure,
              'system_swap_used_bytes':psutil.swap_memory().used,
              'rss_bytes':psutil.Process().memory_info().rss}
    if not sample['ac_power'] or pressure != 1 or sample['rss_bytes'] > int(8.5*(1 << 30)):
        raise RuntimeError(f'component admission refused: {sample}')
    return sample


def representative_projection(oracle):
    """Full observed MLA projection shape; synthetic rows, native Q8 packing.

    Quantize one FP32 row at a time. Never allocate a full FP32 weight matrix
    or read the model. The existing64MiB prepared-matrix budget still applies.
    """
    width, rows, seed = 4096, 1536, 5301536
    row_bytes = width // 32 * 34
    if rows * row_bytes > 8 << 20:
        raise MemoryError('representative packed weights exceed8MiB generation bound')
    packed = np.empty((rows, row_bytes), np.uint8)
    rng = np.random.default_rng(seed)
    for row in range(rows):
        value = rng.standard_normal(width, dtype=np.float32)
        if row == 0:
            value.fill(0)
        packed[row] = oracle.activation(value).reshape(-1)
    validate_matrix(packed, width, 64 << 20)
    return packed, width, {'kind':'synthetic FP32 rows quantized by the pinned native CPU',
        'quantizer':'quantize_row_q8_0', 'seed':seed, 'numpy_version':np.__version__,
        'shape_matches_observed_tensor':'blk.3.attn_q_a.weight', 'M_rows':rows, 'K_width':width,
        'packed_bytes':packed.nbytes, 'packed_weights_sha256':hashlib.sha256(memoryview(packed).cast('B')).hexdigest(),
        'generation_float_row_bytes':width*4, 'model_weights_read':False,
        'scope':'complete observed matrix shape with synthetic weights; not a captured GLM layer execution'}


def run(args):
    representative = getattr(args, 'representative_projection', False)
    token_max = getattr(args, 'representative_token_max', 32)
    representative_counts = representative_token_counts(token_max)
    if representative and (args.gguf or not args.graph_oracle):
        raise ValueError('representative projection requires the CPU graph oracle and synthetic weights only')
    if not representative and token_max != 32:
        raise ValueError('representative token maximum requires --representative-projection')
    before = guard_sample()
    oracle = Q8NativeOracle(args.libraries)
    graph = None
    if args.graph_oracle:
        graph = ctypes.CDLL(str(args.graph_oracle.resolve())).airllm_q8_cpu_graph
        graph.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4
        graph.restype = ctypes.c_int
    rng = np.random.default_rng(530530)
    reports = []
    matrices = []
    if representative:
        matrices.append(representative_projection(oracle))
    elif args.gguf:
        from research.glm53_gpu_v2.q8_reader import read_rows
        packed,width,metadata = read_rows(args.gguf,args.tensor,0,args.rows,max_read_bytes=8 << 20)
        matrices.append((packed,width,metadata))
    else:
        for width, rows in ((32,10), (96,14), (4096,32)):
            packed = rng.integers(0,256,(rows,width//32,34),dtype=np.uint8)
            packed[:,:,:2] = rng.uniform(.0001,.03,(rows,width//32)).astype('<f2').view(np.uint8).reshape(rows,width//32,2)
            matrices.append((packed.reshape(rows,-1),width,{'kind':'synthetic original-format packed Q8_0'}))
    for packed,width,metadata in matrices:
        rows=packed.shape[0]
        for tokens in (representative_counts if representative else (8,16,32,64,128)):
            x = rng.normal(size=(tokens,width)).astype(np.float32)
            x[0] = 0
            q, expected = native_nrc2(oracle,packed,x)
            report, gpu_q, gpu_scale, actual = measure(args.timer,args.work/f'w{width}-n{tokens}',packed,x,args.repetitions)
            comparison = numerical_comparison(actual,expected)
            exact = comparison['bit_identical']
            activation_exact = bool(np.array_equal(gpu_q.view(np.uint8),q[:,:,2:]) and np.array_equal(gpu_scale.view(np.uint8).reshape(tokens,width//32,2),q[:,:,:2]))
            report.update(bit_identical=exact, activation_bit_identical=activation_exact,
                compared_outputs=int(actual.size), max_abs_error=comparison['max_abs'],
                native_nrc2_comparison=comparison,
                weights=metadata,
                activation_kind='synthetic FP32; not GLM captured activation',
                passed=exact and activation_exact)
            if graph:
                graph_output = np.empty_like(expected)
                status = graph(packed.ctypes.data,x.ctypes.data,graph_output.ctypes.data,width,rows,tokens,10)
                if status:
                    raise RuntimeError(f'CPU10 graph oracle failed: {status}')
                graph_comparison = numerical_comparison(actual,graph_output)
                graph_comparison['max_abs_error'] = graph_comparison['max_abs']
                report['cpu10_graph_comparison'] = graph_comparison
            (args.work/f'w{width}-n{tokens}'/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
            sampled = guard_sample()
            if sampled['system_swap_used_bytes'] - before['system_swap_used_bytes'] > (32 << 20):
                raise RuntimeError('component stopped on system swap growth; model attribution unknown')
            # Cases are sequential: do not retain the previous token size's
            # arrays while preparing the next CPU oracle and Metal payload.
            del q, expected, gpu_q, gpu_scale, actual, x
            if graph:
                del graph_output
    nrc2_passed = all(r['passed'] for r in reports)
    graph_passed = bool(graph) and all(r['cpu10_graph_comparison']['bit_identical'] for r in reports)
    result = {'schema_version':1, 'scope':'bounded Q8 component, native ARM nrc=2 oracle; not GLM prefill',
        'model_loaded':False, 'full_model_qualified':False,
        'representative_projection':representative,
        'representative_token_max':token_max if representative else None,
        'requested_token_counts':list(representative_counts) if representative else [8,16,32,64,128],
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'passed':nrc2_passed and (not graph or graph_passed), 'component_nrc2_passed':nrc2_passed,
        'passed_scope':'component bit-exact contract only; not global GPU qualification',
        'component_metal_tolerance_passed':all(r['activation_bit_identical']
            and r['native_nrc2_comparison']['metal_tolerance']['passed']
            and (not graph or r['cpu10_graph_comparison']['metal_tolerance']['passed']) for r in reports),
        'reports':reports, 'guards_before':before, 'guards_after':sampled,
        'native_cpu_dylib_sha256':hashlib.sha256((args.libraries/'libggml-cpu.dylib').read_bytes()).hexdigest(),
        'graph_oracle_sha256':hashlib.sha256(args.graph_oracle.read_bytes()).hexdigest() if args.graph_oracle else None,
        'timer_sha256':hashlib.sha256(args.timer.read_bytes()).hexdigest(),
        'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cpu10_graph_passed':graph_passed,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'timer_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'limits':['No complete model integration or speedup qualification','Warm prepared weights; no SSD claim',
                  'CPU oracle ctypes calls are not CPU10 timing','Only even-row nrc=2 arithmetic is implemented',
                  'Q2 experts, KDA, attention and mHC remain outside this component']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'cases':len(reports),'outputs':sum(r['compared_outputs'] for r in reports),'process_peak_rss_bytes':result['process_peak_rss_bytes']}))
    return 0 if result['passed'] else 3


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libraries',type=Path,required=True)
    parser.add_argument('--timer',type=Path,required=True)
    parser.add_argument('--graph-oracle',type=Path)
    parser.add_argument('--representative-projection',action='store_true',
        help='synthetic full M1536/K4096 projection; requires native CPU graph oracle')
    parser.add_argument('--representative-token-max',type=int,choices=(32,64,128),default=32,
        help='representative projection progresses through 8/16/32, then 64/128 only up to this explicit maximum (default: 32)')
    parser.add_argument('--gguf',type=Path)
    parser.add_argument('--tensor',default='output.weight')
    parser.add_argument('--rows',type=int,choices=range(2,1025,2),default=32)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,choices=range(1,6),default=3)
    args = parser.parse_args()
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'A guarded GLM experiment still owns the lock; no component started.\n')
        raise SystemExit(run(args))
