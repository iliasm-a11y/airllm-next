"""Real-byte grouped Q2_K variant benchmark with native CPU and GPU timestamps."""
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import statistics
import tempfile
import time
import numpy as np
from research.glm53_gpu.quantized_probe import NativeOracle, compare, read_selected_rows
from .q2_kernels import Q2KVariants,prepare_activations,VARIANTS
from .q2_device_timer import measure_device_suite


class GroupOracle:
    def __init__(self,helper,packed,width,activation,nthreads,native):
        self.lib=ctypes.CDLL(str(Path(helper).resolve()))
        self.lib.airllm_q2_group_create.argtypes=[ctypes.c_int]*5
        self.lib.airllm_q2_group_create.restype=ctypes.c_void_p
        self.lib.airllm_q2_group_run.argtypes=[ctypes.c_void_p]*4
        self.lib.airllm_q2_group_run.restype=ctypes.c_int
        self.lib.airllm_q2_group_destroy.argtypes=[ctypes.c_void_p]
        self.lib.airllm_q2_group_destroy.restype=None
        e,r,_=packed.shape
        self.handle=self.lib.airllm_q2_group_create(e,r,width,activation.shape[0],nthreads)
        if not self.handle: raise ValueError('native grouped oracle admission failed')
        self.packed=packed
        self.raw=np.stack([native.activation(x)[0] for x in activation])
        self.out=np.empty((e,r),'f4')
    def run(self):
        if self.lib.airllm_q2_group_run(self.handle,self.packed.ctypes.data,self.raw.ctypes.data,self.out.ctypes.data):
            raise RuntimeError('native grouped oracle execution failed')
        return self.out
    def close(self):
        if self.handle: self.lib.airllm_q2_group_destroy(self.handle);self.handle=None


def summarize(samples):
    return {'seconds':samples,'median_seconds':statistics.median(samples),
            'min_seconds':min(samples),'max_seconds':max(samples)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',required=True)
    p.add_argument('--libraries',required=True)
    p.add_argument('--cpu-helper',required=True)
    p.add_argument('--metal-timer',required=True)
    p.add_argument('--work-directory',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--experts',default='25,124,274,131,76,240,262,21')
    p.add_argument('--tensor',default='blk.3.ffn_gate_exps.weight')
    p.add_argument('--rows',type=int,default=2048)
    p.add_argument('--repetitions',type=int,default=7)
    p.add_argument('--capture-metadata',help='JSON provenance of the provided native activation capture')
    p.add_argument('--activation',help='npy float32 [width] or [experts,width]; otherwise explicitly synthetic')
    args=p.parse_args(argv)
    experts=[int(e) for e in args.experts.split(',')]
    if not 1<=len(experts)<=8 or not 1<=args.rows<=8192 or not 3<=args.repetitions<=20:
        p.error('bounded expert/row/repetition range exceeded')
    import mlx.core as mx
    mx.set_cache_limit(0);mx.set_memory_limit(128<<20)
    read_start=time.perf_counter()
    selections=[read_selected_rows(args.gguf,args.tensor,e,0,args.rows) for e in experts]
    width=selections[0][1]
    packed=np.stack([v[0] for v in selections])
    metadata=[v[2] for v in selections]
    del selections
    read_time=time.perf_counter()-read_start
    if args.activation:
        x=np.load(args.activation,allow_pickle=False)
        if x.ndim==1: x=x.reshape(1,-1)
        x=np.ascontiguousarray(x)
        activation_source={'kind':'provided capture','file':Path(args.activation).name,
                           'sha256_file':hashlib.sha256(Path(args.activation).read_bytes()).hexdigest()}
    else:
        x=np.random.default_rng(5305).normal(size=(1,width)).astype('f4')
        activation_source={'kind':'seeded synthetic','seed':5305,'real_model_activation':False}
    if args.capture_metadata:
        provenance=json.loads(Path(args.capture_metadata).read_text())
        if (not args.activation or provenance.get('sha256_binary')!=hashlib.sha256(x.tobytes()).hexdigest()
                or provenance.get('ordered_expert_ids')!=experts
                or provenance.get('tensor')!=f'ffn_norm-{args.tensor.split(chr(46))[1]}'
                or not any(part in args.tensor for part in ('ffn_gate_exps','ffn_up_exps'))):
            raise ValueError('capture provenance does not match this exact activation/routes/projection')
        activation_source['capture_provenance']=provenance
    if x.dtype!=np.float32 or x.shape[1]!=width or x.shape[0] not in (1,len(experts)):
        raise ValueError('activation shape/type must exactly match selected expert operation')
    prep_start=time.perf_counter()
    matrix=Q2KVariants(packed,width)
    packed_import=time.perf_counter()-prep_start
    prep_start=time.perf_counter()
    activation=prepare_activations(x)
    activation_prep=time.perf_counter()-prep_start
    qs=np.array(activation.qs,dtype='i1')
    scales=np.array(activation.scales,dtype='f4')
    native=NativeOracle(args.libraries)
    native_activations=[native.activation(value) for value in x]
    q8_values_exact=bool(np.array_equal(qs,np.stack([value[1] for value in native_activations]).reshape(-1)))
    q8_scales_check=compare(scales,np.stack([value[2] for value in native_activations]).reshape(-1))
    cpu={n:GroupOracle(args.cpu_helper,packed,width,x,n,native) for n in (1,10)}
    expected=cpu[1].run().copy()
    oracle_agreement=compare(cpu[10].run(),expected)
    # Cross-check native adapter against the independently called pinned dot on
    # a bounded subset of each expert; no per-row Python calls are timed.
    sampled=np.stack([native.matvec(packed[e,:min(16,args.rows)],x[0 if x.shape[0]==1 else e])
                      for e in range(len(experts))])
    adapter_check=compare(expected[:,:min(16,args.rows)],sampled)
    correctness={}
    for variant in VARIANTS:
        correctness[variant]=compare(np.array(matrix.execute(activation,variant),dtype='f4'),expected)
    samples={v:[] for v in VARIANTS}
    samples.update({f'cpu{n}':[] for n in (1,10)})
    order=[]
    rng=np.random.default_rng(7053)
    for rep in range(args.repetitions):
        schedule=list(samples);rng.shuffle(schedule);order.append(schedule)
        for name in schedule:
            mx.synchronize()
            start=time.perf_counter()
            value=cpu[int(name[3:])].run() if name.startswith('cpu') else matrix.execute(activation,name)
            mx.synchronize()
            samples[name].append(time.perf_counter()-start)
    payload=matrix.payload_bytes
    matrix.close();del matrix
    activation.close();del activation
    mx.synchronize();mx.clear_cache()
    # Standalone Metal phase occurs after releasing the MLX packed matrix.
    with tempfile.TemporaryDirectory(prefix='q2-metal-suite-',dir=args.work_directory) as directory:
        device_suite,actual=measure_device_suite(args.metal_timer,directory,packed,width,qs,scales,VARIANTS,
                                                repetitions=args.repetitions)
        device=device_suite['experiments']
        for variant in VARIANTS:
            device[variant]['vs_native_arm']=compare(actual[variant],expected)
    for item in cpu.values(): item.close()
    accepted=bool(q8_values_exact and q8_scales_check['bit_identical'] and oracle_agreement['bit_identical'] and adapter_check['bit_identical'] and
                  all(c['finite'] and c['bit_identical'] for c in correctness.values()) and
                  all(d['vs_native_arm']['finite'] and d['vs_native_arm']['bit_identical'] for d in device.values()))
    report={'schema':1,'scope':'one original Q2_K selected expert projection; no GLM decode gain claim',
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),'mlx_version':mx.__version__,
        'experts_in_supplied_order':experts,'width':width,'rows':args.rows,'metadata':metadata,
        'activation':activation_source,'activation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'q2_packed_sha256':hashlib.sha256(memoryview(packed).cast('B')).hexdigest(),
        'known_payload_bytes':payload,'preparation':{'metadata_and_selected_file_read_seconds':read_time,
            'packed_import_sync_seconds':packed_import,'activation_import_quantize_sync_seconds':activation_prep},
        'q8_activation_values_exact':q8_values_exact,'q8_activation_scales_vs_native':q8_scales_check,
        'native_adapter_vs_cpu1_cpu10':oracle_agreement,'native_adapter_vs_independent_dot_sample':adapter_check,
        'correctness':correctness,'wall_clock':'perf_counter around execution followed by mx.synchronize',
        'wall_timings':{k:summarize(v) for k,v in samples.items()},'randomized_repetition_order':order,
        'metal_device_timings':device,'metal_suite_order':device_suite['randomized_order'],
        'metal_suite_source_sha256':device_suite['generated_source_sha256'],'acceptance':{'finite_bit_identical_all_variants':accepted},
        'mlx_peak_bytes':mx.get_peak_memory(),
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'libraries_sha256':{name:hashlib.sha256((Path(args.libraries)/name).read_bytes()).hexdigest()
                            for name in ('libggml-base.dylib','libggml-cpu.dylib')},
        'timer_sha256':hashlib.sha256(Path(args.metal_timer).read_bytes()).hexdigest(),
        'cpu_helper_sha256':hashlib.sha256(Path(args.cpu_helper).read_bytes()).hexdigest(),
        'limits':['prepared warm payload and prequantized activations are reused across repetitions',
                  'CPU1/10 use persistent grouped native adapter, not production GGML scheduler',
                  'GPU timestamps include command stages only; imports/quantization reported separately',
                  'MLX wall timings include dispatch, allocation and synchronization; not isolated GPU time',
                  'no true cold-SSD benchmark; no external SSD attribution; no full-model routes/logits certification']}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'acceptance':accepted,'payload_bytes':payload,
        'wall_medians_ms':{k:1000*statistics.median(v) for k,v in samples.items()},
        'gpu_medians_ms':{k:1000*v['median_gpu_seconds'] for k,v in device.items()}},indent=2))
    return 0 if accepted else 1


if __name__=='__main__': raise SystemExit(main())
