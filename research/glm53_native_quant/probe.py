"""Guarded real-weight CPU/native-Metal/activation-adapter/historical comparison.

Every measurement is a warm component. No model token rate or agent reuse is
inferred. Original weight slices live only in private temporary files.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from airllm.nextgen.experiment_guard import ExperimentLimits, admission, supervise, system_snapshot
from airllm.nextgen.glm_native import FILES, _identity, inspect_native_checkpoint
from research.glm53_batched_prefill.uid_supervisor import acquire_uid_lock
from research.glm53_gpu_v3.power_state import PowerGate, ScopedAwake
from research.glm53_ultra_decode.build import sha
from research.glm53_ultra_decode.campaign_limits import remaining_swap_budget

GIB = 1 << 30
TIMES = ('total','prepare_import','submit','terminal_wait','output_read')


def validate_selection(format, rows, width=4096):
    if format not in (2,8) or any(type(v) is not int for v in (rows,width)):
        raise ValueError('Q2_K/Q8_0 and integer dimensions required')
    if not 8 <= rows <= 8192 or rows % 8 or not 256 <= width <= 16384 or width % 256:
        raise ValueError('component dimensions outside native adapter bounds')
    weight_bytes = rows * (width//(256 if format == 2 else 32)) * (84 if format == 2 else 34)
    if weight_bytes > 16 << 20:
        raise MemoryError('selected packed weights exceed16MiB')
    return weight_bytes


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def verify_build(build, runtime):
    record = json.loads((build/'build.json').read_text())
    if record.get('passed') is not True:
        raise ValueError('successful build receipt required')
    for name, digest in record['inputs_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('build source changed: '+name)
    for name, digest in record['binaries_sha256'].items():
        if sha(build/name) != digest:
            raise ValueError('build executable changed: '+name)
    libraries = {p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    if libraries != record['native_libraries_sha256']:
        raise ValueError('runtime differs from build receipt')
    return sha(build/'build.json')


def limits(sample, baseline):
    return ExperimentLimits(rss_bytes=int(8.5*GIB),reserve_bytes=3*GIB,
        admitted_payload_bytes=256<<20,swap_growth_bytes=remaining_swap_budget(sample,baseline),
        timeout_seconds=180,interval_seconds=.1,max_processes=8)


def preflight(args):
    sample = system_snapshot()
    power = PowerGate(expected_source='ac').sample()
    reasons = []
    try:
        admission(limits(sample,args.campaign_swap_used_bytes),sample)
    except MemoryError as error:
        reasons.append(str(error))
    if sample['pressure_level'] != 1:
        reasons.append('normal macOS pressure required')
    disk = shutil.disk_usage(args.output).free
    if disk < 32*GIB:
        reasons.append('disk reserve below32GiB')
    if not power['passed']:
        reasons.append('AC/lid/clock admission failed')
    checkpoint = inspect_native_checkpoint(args.model,args.receipt)
    if not checkpoint['integrity_verified'] or not checkpoint['architecture_matches']:
        reasons.append('checkpoint integrity/architecture unverified')
    # This observation is not a lease. The worker acquires and retains its own
    # lock before importing arrays or invoking any native/device operation.
    try:
        holder, _ = acquire_uid_lock()
        holder.close()
    except RuntimeError as error:
        reasons.append(str(error))
    return {'sample':sample,'disk_free_bytes':disk,'power':power,
        'checkpoint':checkpoint,'blockers':reasons}


class Native:
    def __init__(self, lib, packed, width, format, gpu, threads):
        import numpy as np
        self.lib = lib
        count = ctypes.c_size_t()
        self.handle = lib.native_quant_create(format,packed.ctypes.data,width,len(packed),
            gpu,threads,ctypes.byref(count))
        if not self.handle:
            raise RuntimeError('native backend/shape/allocation refused')
        self.allocated_bytes = count.value
        self.output = np.empty(len(packed),np.float32)

    def run(self, x, adapted=0):
        timing = (ctypes.c_double*len(TIMES))()
        status = self.lib.native_quant_run(self.handle,x.ctypes.data,self.output.ctypes.data,
            adapted,timing)
        if status:
            raise RuntimeError('native execution failed: '+str(status))
        return self.output.copy(),dict(zip(TIMES,timing))

    def close(self):
        if self.handle:
            self.lib.native_quant_destroy(self.handle)
            self.handle = None


def measure_legacy(args, packed, x, holder):
    import numpy as np
    shader = Path(__file__).parents[1]/'glm53_ultra_decode/decode.metal'
    record = json.loads((args.legacy_build/'build.json').read_text())
    shader_hashes = [digest for path,digest in record['inputs_sha256'].items() if Path(path).name=='decode.metal']
    if shader_hashes != [sha(shader)]:
        raise ValueError('historical shader differs from bound PR24 source')
    with tempfile.TemporaryDirectory(prefix='private-weights-',dir=args.output) as directory:
        root = Path(directory)
        buffers = []
        for name, value in (('weights',packed),('x',x),('shape',np.array([x.size,len(packed),1],np.uint32))):
            path = root/(name+'.bin')
            path.write_bytes(value.tobytes())
            buffers.append({'bytes':value.nbytes,'input':str(path)})
        buffers += [{'bytes':x.size//(256 if args.format==2 else 32)*(292 if args.format==2 else 34)},
                    {'bytes':len(packed)*4,'output':str(root/'y.bin')}]
        lanes = 32 if args.format == 2 else 4
        job = {'source':str(shader),
            'buffers':buffers,'cpu_quantize':{'format':args.format,'width':x.size,'input':1,'output':3},
            'experiments':[{'name':'historical_tile1','stages':[{'entry':f'q{args.format}_tile1',
                'bindings':[0,0,3,4,2],'grid':[(len(packed)*lanes+31)//32*32,1,1],
                'threadgroup':[32,1,1]}]}],'repetitions':args.repetitions}
        (root/'job.json').write_text(json.dumps(job))
        result = subprocess.run([str(args.legacy_build/'decode-timer'),str(root/'job.json')],
            capture_output=True,text=True,timeout=90,pass_fds=(holder.fileno(),))
        (args.output/'legacy.stderr.log').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('historical timer failed: '+result.stderr[:1000])
        if shader_hashes != [sha(shader)]:
            raise ValueError('historical shader changed during execution')
        timer = json.loads(result.stdout)
        output = np.fromfile(root/'y.bin',np.float32)
        if output.shape != (len(packed),):
            raise ValueError('historical output size mismatch')
        return timer,output


def worker(args):
    # Array and device imports follow both outer and inner admission.
    initial = preflight(args)
    if initial['blockers']:
        write(args.output/'result.json',{'status':'worker_admission_refused',**initial})
        return 2
    try:
        holder, lock = acquire_uid_lock()
    except RuntimeError as error:
        write(args.output/'result.json',{'status':'worker_admission_refused','error':str(error)})
        return 2
    import numpy as np
    from research.glm53_batched_prefill.metal_probe import numerical_comparison
    from research.glm53_gpu.quantized_probe import read_selected_rows
    from research.glm53_gpu_v2.q8_reader import read_rows
    from research.glm53_ultra_decode.components import native_functions, summary
    gate = PowerGate(expected_source='ac')
    gate.sample()
    errors = []
    stop = threading.Event()
    class OwnProcess:
        pid = os.getpid()
        @staticmethod
        def poll(): return None
    handles = []
    report = {'schema':1,'utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'scope':'original GGUF packed weights, warm single-token component; no model generation',
        'uid_lock':lock,'format':'Q2_K' if args.format==2 else 'Q8_0','seed':args.seed,
        'model_decode_gain_qualified':False,'inter_agent_reuse_measured':False,
        'initial':initial,'numpy_version':np.__version__}
    sources = [Path(__file__),Path(__file__).with_name('build.py'),
        Path(__file__).parents[1]/'glm53_batched_prefill/metal_probe.py',
        Path(__file__).parents[1]/'glm53_gpu_v2/q8_reader.py',
        Path(__file__).parents[1]/'glm53_gpu/quantized_probe.py',
        Path(__file__).parents[1]/'glm53_ultra_decode/components.py']
    report['probe_source_sha256'] = {str(p):sha(p) for p in sources}
    write(args.output/'result.json',report)
    def watch():
        while not stop.wait(.25):
            try:
                if not gate.sample()['passed'] or shutil.disk_usage(args.output).free < 32*GIB:
                    errors.append('power/lid/clock/disk continuity failed')
                remaining_swap_budget(system_snapshot(),args.campaign_swap_used_bytes)
            except Exception as error:
                errors.append(str(error))
    monitor = threading.Thread(target=watch,daemon=True)
    monitor.start()
    try:
        with ScopedAwake(OwnProcess()) as awake:
            def check():
                awake.check()
                if errors:
                    raise RuntimeError(errors[0])
            bound = {label:verify_build(path,args.runtime) for label,path in
                (('native',args.build),('historical',args.legacy_build))}
            identities = {args.model/row['name']:row['identity'] for row in initial['checkpoint']['files']}
            shard = args.model/FILES[1][0]
            if args.format == 2:
                packed,width,metadata = read_selected_rows(shard,args.tensor,args.expert,0,args.rows,
                    max_read_bytes=16<<20)
            else:
                packed,width,metadata = read_rows(shard,args.tensor,0,args.rows,expert=args.expert,
                    max_read_bytes=16<<20)
            validate_selection(args.format,args.rows,width)
            if not all(_identity(p)==identity for p,identity in identities.items()):
                raise RuntimeError('checkpoint identity changed while reading selected rows')
            rng = np.random.default_rng(args.seed)
            if args.activation_f32:
                if args.activation_f32.stat().st_size != width*4:
                    raise ValueError('activation file must contain exactly one finite FP32 vector')
                x = np.fromfile(args.activation_f32,dtype=np.float32)
                activation = {'kind':'provided FP32 component vector','file':args.activation_f32.name}
            else:
                x = rng.standard_normal(width,dtype=np.float32)
                activation = {'kind':'synthetic deterministic FP32; not a captured GLM activation','seed':args.seed}
            if not np.isfinite(x).all():
                raise ValueError('nonfinite activation')
            activation['sha256'] = hashlib.sha256(x.tobytes()).hexdigest()
            report.update(weights=metadata,activation=activation,build_receipt_sha256=bound)
            lib = ctypes.CDLL(str(args.build/'native.dylib'))
            lib.native_quant_create.argtypes = [ctypes.c_int,ctypes.c_void_p]+[ctypes.c_int]*4+[ctypes.POINTER(ctypes.c_size_t)]
            lib.native_quant_create.restype = ctypes.c_void_p
            lib.native_quant_run.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
            lib.native_quant_run.restype = ctypes.c_int
            lib.native_quant_destroy.argtypes = [ctypes.c_void_p]
            lib.native_quant_destroy.restype = None
            cpu = {}
            for threads in (1,10):
                check()
                handle = Native(lib,packed,width,args.format,0,threads)
                handles.append(handle)
                samples = []; comparisons = []
                for i in range(args.repetitions+3):
                    check()
                    output,timing = handle.run(x)
                    if threads == 1 and i == 0: expected = output.copy()
                    comparisons.append(numerical_comparison(output,expected))
                    if i >= 3: samples.append(timing['total'])
                cpu[str(threads)] = {'timing':summary(samples),
                    'comparison':comparisons[-1],
                    'all_repetitions_bit_identical':all(c['bit_identical'] for c in comparisons),
                    'backend_allocated_bytes':handle.allocated_bytes}
                handle.close(); handles.remove(handle)
            _,quantize,dot = native_functions(args.runtime,args.format)
            q = np.empty(width//(256 if args.format==2 else 32)*(292 if args.format==2 else 34),np.uint8)
            quantize(x.ctypes.data,q.ctypes.data,width)
            independent = np.empty(min(16,len(packed)),np.float32)
            for row in range(len(independent)):
                dot(width,independent[row:].ctypes.data,0,packed[row].ctypes.data,0,q.ctypes.data,0,1)
            report.update(cpu=cpu,graph_vs_native_dot=numerical_comparison(expected[:len(independent)],independent))
            write(args.output/'result.json',report)
            check()
            begin = time.perf_counter()
            gpu = Native(lib,packed,width,args.format,1,10)
            handles.append(gpu)
            report['metal_setup_upload_seconds'] = time.perf_counter()-begin
            report['metal_backend_allocated_bytes'] = gpu.allocated_bytes
            timings = {0:[],1:[]}; checks = {0:[],1:[]}; order = []
            for i in range(args.repetitions+3):
                modes = [0,1]; rng.shuffle(modes)
                if i >= 3: order.append(modes)
                for mode in modes:
                    check()
                    output,timing = gpu.run(x,mode)
                    checks[mode].append(numerical_comparison(output,expected))
                    if i >= 3: timings[mode].append(timing)
            gpu.close(); handles.remove(gpu)
            report['metal'] = {name:{'timing':{key:summary([s[key] for s in timings[mode]]) for key in TIMES},
                'comparison':checks[mode][-1],
                'all_repetitions_bit_identical':all(c['bit_identical'] for c in checks[mode]),
                'all_repetitions_within_existing_tolerance':all(c['metal_tolerance']['passed'] for c in checks[mode])}
                for mode,name in ((0,'native'),(1,'activation_adapter'))}
            report['metal_mode_order'] = order
            write(args.output/'result.json',report)
            check()
            legacy,y = measure_legacy(args,packed,x,holder)
            report['historical'] = {'timer':legacy,'comparison':numerical_comparison(y,expected)}
            report['checkpoint_identities_unchanged'] = all(_identity(p)==identity for p,identity in identities.items())
            report['build_binding_unchanged'] = bound == {label:verify_build(path,args.runtime) for label,path in
                (('native',args.build),('historical',args.legacy_build))}
            report['probe_sources_unchanged'] = report['probe_source_sha256'] == {str(p):sha(p) for p in sources}
            report['cpu_reference_valid'] = all(c['all_repetitions_bit_identical'] for c in cpu.values()) and report['graph_vs_native_dot']['bit_identical']
            check()
            if not report['checkpoint_identities_unchanged'] or not report['build_binding_unchanged'] or not report['probe_sources_unchanged'] or not report['cpu_reference_valid']:
                raise RuntimeError('component reference or input/build identity failed')
            report['status'] = 'completed'
            report['awake'] = awake.record
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:
        for handle in handles: handle.close()
        stop.set(); monitor.join(timeout=5)
        report['power'] = gate.summary()
        report['continuity_errors'] = errors
        if errors or monitor.is_alive() or not gate.sample(final=True)['passed']:
            report['status'] = 'failed'
        report['power'] = gate.summary()
        holder.close()
        report['uid_fd_closed'] = True
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.output/'result.json',report)
    print(json.dumps({'status':report['status'],'model_decode_gain_qualified':False}))
    return 0 if report['status'] == 'completed' else 2


def run(args):
    validate_selection(args.format,args.rows)
    if not 3 <= args.repetitions <= 30 or not 0 <= args.expert < 288:
        raise ValueError('repetitions/expert outside bounds')
    for name in ('model','receipt','runtime','build','legacy_build','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    protected = [args.model,args.runtime,args.build,args.legacy_build,args.receipt.parent]
    if any(args.output.is_relative_to(p) or p.is_relative_to(args.output) for p in protected):
        raise ValueError('output overlaps a protected input directory')
    args.output.mkdir(parents=True,mode=0o700)
    initial = preflight(args)
    report = {'utc':datetime.now(timezone.utc).isoformat(),'status':'admission_refused' if initial['blockers'] else 'admitted',
        'component_worker_started':False,'model_process_started':False,
        'model_decode_gain_qualified':False,'preflight':initial}
    write(args.output/'admission.json',report)
    if initial['blockers']:
        print(json.dumps({'status':report['status'],'blockers':initial['blockers']}))
        return 2
    command = [sys.executable,'-m','research.glm53_native_quant.probe',*sys.argv[1:],'--worker']
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    env['AIRLLM_METAL_COMMAND_TIMING'] = '1'
    report['component_worker_started'] = True
    write(args.output/'admission.json',report)
    try:
        guard = supervise(command,args.output/'supervision',limits=limits(initial['sample'],args.campaign_swap_used_bytes),env=env)
        report.update(status='completed',guard=guard)
    except (MemoryError,RuntimeError) as error:
        report.update(status='failed',error=str(error))
    finally:
        guard_path = args.output/'supervision.guard.json'
        if guard_path.is_file(): report['guard'] = json.loads(guard_path.read_text())
        remaining = system_snapshot()
        report['after'] = remaining
        try: remaining_swap_budget(remaining,args.campaign_swap_used_bytes)
        except MemoryError as error: report.update(status='failed',error=str(error))
        write(args.output/'admission.json',report)
    print(json.dumps({'status':report['status'],'model_decode_gain_qualified':False}))
    return 0 if report['status']=='completed' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','receipt','runtime','build','legacy-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--format',type=int,choices=(2,8),required=True)
    p.add_argument('--tensor',required=True)
    p.add_argument('--rows',type=int,required=True)
    p.add_argument('--expert',type=int,default=0)
    p.add_argument('--seed',type=int,default=5304101)
    p.add_argument('--repetitions',type=int,default=21)
    p.add_argument('--activation-f32',type=Path)
    p.add_argument('--campaign-swap-used-bytes',type=int,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    raise SystemExit(run(p.parse_args()))
