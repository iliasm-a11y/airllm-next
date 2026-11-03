"""Bounded multi-token Q8_0/ARM nrc=2 component, never a model engine claim."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import numpy as np
from research.glm53_gpu_v2.q8_compatible import validate_activation, validate_matrix
from research.glm53_gpu_v2.q8_oracle import Q8NativeOracle


def validate(packed, x, budget_bytes=64 << 20):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.dtype != np.float32 or not x.flags.c_contiguous:
        raise ValueError('activations must be contiguous float32 [tokens,width]')
    tokens, width = x.shape
    validate_matrix(packed, width, budget_bytes)
    if tokens not in (2, 8, 16, 32, 64, 128) or packed.shape[0] % 2:
        raise ValueError('nrc=2 requires even rows and admitted token counts')
    for row in x:
        validate_activation(row, width)
    required = 2 * (packed.nbytes + x.nbytes + x.size + tokens * width // 32 * 2 + tokens * packed.shape[0] * 4)
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
    if completed.returncode:
        raise RuntimeError(completed.stderr[:3000])
    report = json.loads(completed.stdout)
    report.update(known_host_device_payload_bound_bytes=payload,
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


def run(args):
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
    if args.gguf:
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
        for tokens in (8,16,32,64,128):
            x = rng.normal(size=(tokens,width)).astype(np.float32)
            x[0] = 0
            q, expected = native_nrc2(oracle,packed,x)
            report, gpu_q, gpu_scale, actual = measure(args.timer,args.work/f'w{width}-n{tokens}',packed,x,args.repetitions)
            exact = bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)))
            activation_exact = bool(np.array_equal(gpu_q.view(np.uint8),q[:,:,2:]) and np.array_equal(gpu_scale.view(np.uint8).reshape(tokens,width//32,2),q[:,:,:2]))
            report.update(bit_identical=exact, activation_bit_identical=activation_exact,
                compared_outputs=int(actual.size), max_abs_error=float(np.max(np.abs(actual-expected))),
                weights=metadata,
                activation_kind='synthetic FP32; not GLM captured activation',
                passed=exact and activation_exact)
            if graph:
                graph_output = np.empty_like(expected)
                status = graph(packed.ctypes.data,x.ctypes.data,graph_output.ctypes.data,width,rows,tokens,10)
                if status:
                    raise RuntimeError(f'CPU10 graph oracle failed: {status}')
                mismatches = int(np.count_nonzero(actual.view(np.uint32) != graph_output.view(np.uint32)))
                report['cpu10_graph_comparison'] = {'bit_identical':mismatches == 0,
                    'bit_mismatches':mismatches, 'max_abs_error':float(np.max(np.abs(actual-graph_output)))}
            reports.append(report)
            sampled = guard_sample()
            if sampled['system_swap_used_bytes'] - before['system_swap_used_bytes'] > (32 << 20):
                raise RuntimeError('component stopped on system swap growth; model attribution unknown')
    nrc2_passed = all(r['passed'] for r in reports)
    graph_passed = bool(graph) and all(r['cpu10_graph_comparison']['bit_identical'] for r in reports)
    result = {'schema_version':1, 'scope':'bounded Q8 component, native ARM nrc=2 oracle; not GLM prefill',
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'passed':nrc2_passed and (not graph or graph_passed), 'component_nrc2_passed':nrc2_passed,
        'reports':reports, 'guards_before':before, 'guards_after':sampled,
        'native_cpu_dylib_sha256':hashlib.sha256((args.libraries/'libggml-cpu.dylib').read_bytes()).hexdigest(),
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
    parser.add_argument('--gguf',type=Path)
    parser.add_argument('--tensor',default='output.weight')
    parser.add_argument('--rows',type=int,choices=range(2,1025,2),default=32)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,choices=range(1,6),default=3)
    raise SystemExit(run(parser.parse_args()))
