"""Bounded Q2_K expert x multiple tokens, component only; never model inference.

Prepared candidate, not compiled or executed as of 2026-11-03 publication.

The original packed format is retained. Synthetic Q2_K weights are the default;
an explicit GGUF option reads one bounded original expert with no conversion.
The CPU native quantizer prepares Q8_K activations before the single GPU dot
dispatch. This does not implement routing or ordered expert aggregation.
"""
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
import time
import numpy as np
from research.glm53_gpu.quantized_probe import NativeOracle, read_selected_rows
from research.glm53_batched_prefill.metal_probe import guard_sample, numerical_comparison

MAX_PAYLOAD = 64 << 20


def validate(packed, raw, width, tokens, budget_bytes=MAX_PAYLOAD):
    if type(width) is not int or width not in (2048, 4096):
        raise ValueError('representative expert width must be 2048 or 4096')
    if type(tokens) is not int or tokens not in (8, 16, 32):
        raise ValueError('multi-token component admits 8/16/32 tokens')
    if (not isinstance(packed,np.ndarray) or packed.dtype!=np.uint8 or packed.ndim!=2
            or not packed.flags.c_contiguous or not 1<=packed.shape[0]<=4096
            or packed.shape[1]!=width//256*84):
        raise ValueError('packed original Q2_K rows do not match bounded expert shape')
    if (not isinstance(raw,np.ndarray) or raw.dtype!=np.uint8 or not raw.flags.c_contiguous
            or raw.shape!=(tokens,width//256,292)):
        raise ValueError('native Q8_K activation bytes have incompatible shape')
    blocks=packed.reshape(packed.shape[0],width//256,84)
    weight_scales=blocks[...,80:84].copy().view('<f2')
    if (not np.isfinite(weight_scales).all() or np.any(
            (weight_scales!=0)&(np.abs(weight_scales)<np.finfo(np.float16).tiny))):
        raise ValueError('nonfinite/subnormal Q2_K weight scales are not qualified')
    scales=raw[...,:4].copy().view('<f4').reshape(tokens,width//256)
    if (not np.isfinite(scales).all() or
            np.any((scales!=0)&(np.abs(scales)<np.finfo(np.float32).tiny))):
        raise ValueError('nonfinite/subnormal Q8_K scales are not qualified')
    known=2*(packed.nbytes+raw.nbytes+tokens*width*4+tokens*packed.shape[0]*4)
    if type(budget_bytes) is not int or not 0<budget_bytes<=MAX_PAYLOAD or known>budget_bytes:
        raise MemoryError('known host/device payload exceeds 64 MiB budget')
    return known


def prepare_native(oracle, x):
    if (not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.ndim!=2
            or not x.flags.c_contiguous or not np.isfinite(x).all()):
        raise ValueError('finite contiguous FP32 activations required')
    # NativeOracle.activation zero-initializes the complete Q8_K block,
    # including bsums which the native zero-block fast path leaves unwritten.
    return np.stack([oracle.activation(row)[0] for row in x])


def native_outputs(oracle, packed, raw, width):
    tokens=raw.shape[0]
    validate(packed,raw,width,tokens)
    output=np.empty((tokens,packed.shape[0]),np.float32)
    for token in range(tokens):
        for row in range(packed.shape[0]):
            oracle.dot(width,output[token,row:].ctypes.data,0,packed[row:].ctypes.data,0,
                       raw[token:].ctypes.data,0,1)
    if not np.isfinite(output).all():
        raise RuntimeError('nonfinite native Q2_K oracle output')
    return output


def measure(timer, directory, packed, raw, width, repetitions):
    tokens,rows=raw.shape[0],packed.shape[0]
    known=validate(packed,raw,width,tokens)
    root=Path(directory).resolve()
    root.mkdir(parents=True,exist_ok=False)
    shape=np.array([width,rows,tokens],np.uint32)
    buffers=[]
    for name,values in [('packed',packed),('activation',raw)]:
        path=root/(name+'.bin')
        with path.open('wb') as out:out.write(memoryview(values).cast('B'))
        buffers.append({'bytes':values.nbytes,'input':str(path)})
    output=root/'output.f32'
    buffers.append({'bytes':tokens*rows*4,'output':str(output)})
    shape_path=root/'shape.bin';shape_path.write_bytes(shape.tobytes())
    buffers.append({'bytes':shape.nbytes,'input':str(shape_path)})
    shader=Path(__file__).with_name('metal_q2_batch.metal').resolve()
    job={'source':str(shader),'buffers':buffers,'warmups':1,'repetitions':repetitions,
        'stages':[{'entry':'q2_batch_dot','bindings':[0,1,2,3],
            'grid':[rows*256,tokens//8,1],'threadgroup':[256,1,1]}]}
    manifest=root/'job.json';manifest.write_text(json.dumps(job,indent=2)+'\n')
    child=subprocess.run([str(Path(timer).resolve()),str(manifest)],capture_output=True,text=True,timeout=60)
    (root/'timer-stdout.json').write_text(child.stdout)
    (root/'timer-stderr.log').write_text(child.stderr)
    (root/'timer-exit.json').write_text(json.dumps({'returncode':child.returncode})+'\n')
    if child.returncode:raise RuntimeError(child.stderr[:3000])
    report=json.loads(child.stdout)
    if report.get('thread_execution_width')!=32:
        raise RuntimeError('use the timer built with metal_q2_timer.patch; SIMD32 must be checked before dispatch')
    actual=np.fromfile(output,np.float32)
    if actual.size!=tokens*rows:raise RuntimeError('short Metal output')
    report.update(declared_prepared_data_accounting_bytes=known,dimensions={'M':rows,'K':width,'N':tokens},
        payload_scope='declared prepared-data accounting; excludes native oracle copies and driver RSS',
        dot_dispatches_per_batch=1,weight_reuse_tokens_per_threadgroup=8,
        declared_static_threadgroup_bytes=84,shader_sha256=hashlib.sha256(shader.read_bytes()).hexdigest())
    return report,actual.reshape(tokens,rows)


def run(args):
    if args.output.exists() or args.work.exists():
        raise FileExistsError('component output and work paths must be fresh')
    before=guard_sample()
    oracle=NativeOracle(args.libraries)
    oracle.quantize=oracle.cpu.quantize_row_q8_K
    oracle.quantize.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
    oracle.quantize.restype=None
    rng=np.random.default_rng(5302048)
    if args.gguf:
        packed,width,weight_metadata=read_selected_rows(args.gguf,args.tensor,args.expert,0,2048,max_read_bytes=8<<20)
    else:
        width,rows=4096,2048
        blocks=width//256
        packed=rng.integers(0,256,(rows,blocks,84),dtype=np.uint8)
        halves=rng.uniform(.0001,.03,(rows,blocks,2)).astype('<f2')
        packed[...,80:84]=halves.view(np.uint8).reshape(rows,blocks,4)
        packed[0]=0
        packed=packed.reshape(rows,-1)
        weight_metadata={'kind':'synthetic original-format Q2_K blocks','seed':5302048,
            'numpy_version':np.__version__,
            'shape_matches_observed_expert':'blk.3.ffn_gate_exps.weight / one expert',
            'model_weights_read':False,'packed_bytes':packed.nbytes,
            'packed_sha256':hashlib.sha256(memoryview(packed).cast('B')).hexdigest()}
    reports=[]
    for tokens in (8,16,32):
        x=rng.standard_normal((tokens,width),dtype=np.float32)
        x[0]=0
        quantization_begin=time.perf_counter()
        raw=prepare_native(oracle,x)
        quantization_wall=time.perf_counter()-quantization_begin
        known=validate(packed,raw,width,tokens)
        native_begin=time.perf_counter()
        expected=native_outputs(oracle,packed,raw,width)
        native_wall=time.perf_counter()-native_begin
        report,actual=measure(args.timer,args.work/f'n{tokens}',packed,raw,width,args.repetitions)
        report.update(comparison_to_native_arm=numerical_comparison(actual,expected),
            activation_quantization_wall_seconds=quantization_wall,
            native_reference_wall_seconds=native_wall,
            activation_quantizer='libggml-cpu quantize_row_q8_K',
            activation_raw_sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest(),
            activation_kind='seeded synthetic FP32, CPU-quantized Q8_K; not captured GLM activation',
            native_oracle_calls=tokens*packed.shape[0],cpu_oracle_scope='one native nrc=1 call per output; no CPU10 speed measurement')
        (args.work/f'n{tokens}'/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report)
        sampled=guard_sample()
        if sampled['system_swap_used_bytes']-before['system_swap_used_bytes']>32<<20:
            raise RuntimeError('system swap growth stopped component; exclusive attribution unknown')
    passed=all(r['comparison_to_native_arm']['finite'] and r['comparison_to_native_arm']['bit_identical'] for r in reports)
    result={'schema_version':1,'component_only':True,'model_loaded':False,'full_model_qualified':False,
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),'passed':passed,'weights':weight_metadata,
        'passed_scope':'component bit-exact contract only; not global GPU qualification',
        'component_metal_tolerance_passed':all(r['comparison_to_native_arm']['metal_tolerance']['passed'] for r in reports),
        'reports':reports,'guards_before':before,'guards_after':sampled,
        'native_cpu_dylib_sha256':hashlib.sha256((args.libraries/'libggml-cpu.dylib').read_bytes()).hexdigest(),
        'native_base_dylib_sha256':hashlib.sha256((args.libraries/'libggml-base.dylib').read_bytes()).hexdigest(),
        'timer_sha256':hashlib.sha256(args.timer.read_bytes()).hexdigest(),
        'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'timer_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'limits':['No complete GLM graph, router or recurrent state comparison',
            'Single fixed expert; no routing or ordered weighted expert aggregation',
            'Prepared weights/activations; not SSD or GLM prefill performance',
            'CPU activation quantization is outside device command timestamps',
            'No independent numerical or speed qualification from equal generated IDs']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':passed,'cases':len(reports),'elements':sum(r['comparison_to_native_arm']['elements'] for r in reports)}))
    return 0 if passed else 3


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('libraries','timer','work','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gguf',type=Path)
    p.add_argument('--tensor',default='blk.3.ffn_gate_exps.weight')
    p.add_argument('--expert',type=int,choices=range(288),default=0)
    p.add_argument('--repetitions',type=int,choices=range(1,6),default=3)
    args=p.parse_args()
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.exit(2,'A guarded GLM experiment still owns the lock; no component started.\n')
        raise SystemExit(run(args))
