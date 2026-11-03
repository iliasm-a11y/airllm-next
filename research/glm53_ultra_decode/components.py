"""Warm single-token native/Metal measurements, never whole-model qualification."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import numpy as np
from research.glm53_batched_prefill.metal_probe import numerical_comparison
from research.glm53_batched_prefill.uid_supervisor import acquire_uid_lock
from research.glm53_gpu_v3.power_state import PowerGate
from .build import sha

def validate_shape(format,rows,width,banks):
    if format not in (2,8) or any(type(v) is not int for v in (rows,width,banks)):
        raise ValueError('integer dimensions and Q2_K/Q8_0 required')
    if not 1<=rows<=8192 or not 256<=width<=16384 or width%256 or banks not in (1,2):
        raise ValueError('component shape outside bounds')
    size=rows*(width//(256 if format==2 else 32))*(84 if format==2 else 34)
    if size*banks>24<<20:raise MemoryError('original packed weights exceed24MiB')
    return size

def summary(samples):
    return {'seconds':samples,'median_seconds':statistics.median(samples),
            'min_seconds':min(samples),'max_seconds':max(samples)}

class CPU:
    def __init__(self,lib,packed,x,format,threads):
        self.lib=lib;self.packed=packed;self.x=x
        self.handle=lib.decode_component_create(format,packed[0].ctypes.data,packed[-1].ctypes.data,
            x.ctypes.data,x.size,packed.shape[1],packed.shape[0],threads)
        if not self.handle:raise RuntimeError('native CPU component admission refused')
        self.output=np.empty(packed.shape[:2],np.float32)
    def run(self):
        seconds=ctypes.c_double()
        if self.lib.decode_component_run(self.handle,self.output.ctypes.data,ctypes.byref(seconds)):
            raise RuntimeError('native CPU graph failed')
        return self.output,seconds.value
    def close(self):
        if self.handle:self.lib.decode_component_destroy(self.handle);self.handle=None

def native_functions(runtime,format):
    base=ctypes.CDLL(str(runtime/'libggml-base.dylib'))
    cpu=ctypes.CDLL(str(runtime/'libggml-cpu.dylib'))
    weight=getattr(base,'quantize_row_q2_K_ref') if format==2 else getattr(cpu,'quantize_row_q8_0')
    activation=getattr(base,'quantize_row_q8_K_ref') if format==2 else getattr(cpu,'quantize_row_q8_0')
    dot=getattr(cpu,'ggml_vec_dot_q2_K_q8_K' if format==2 else 'ggml_vec_dot_q8_0_q8_0')
    for f in (weight,activation):f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64];f.restype=None
    dot.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,
                  ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];dot.restype=None
    return weight,activation,dot

def run(args):
    format,rows,width,banks=args.format,args.rows,args.width,args.banks
    size=validate_shape(format,rows,width,banks)
    if not 3<=args.repetitions<=30:raise ValueError('repetitions outside3..30')
    root=args.output.resolve();runtime=args.runtime.resolve();build=args.build.resolve()
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    gate=PowerGate(expected_source='ac');initial=gate.sample()
    if not initial['passed']:raise RuntimeError('AC/lid/clock admission refused: '+str(initial))
    holder,lock=acquire_uid_lock()
    stop=threading.Event();power_errors=[]
    def watch():
        while not stop.wait(.5):
            try:
                if not gate.sample()['passed']:power_errors.append('power/lid/clock changed')
            except Exception as error:power_errors.append(str(error))
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    handles=[]
    report={'schema':1,'utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'scope':'warm synthetic original-format single-token component; no model weights read',
        'format':'Q2_K' if format==2 else 'Q8_0','rows':rows,'width':width,'banks':banks,
        'weight_bytes':size*banks,'model_decode_gain_qualified':False,'uid_lock':lock,
        'seed':args.seed,'numpy_version':np.__version__,'model_weights_read':False}
    try:
        source_root=Path(__file__).parent
        source_hashes={p.name:sha(p) for p in (source_root/'components.py',source_root/'decode.metal',source_root/'guard_components.py')}
        build_record=json.loads((build/'build.json').read_text())
        bound_inputs=all(Path(p).is_file() and sha(p)==digest for p,digest in build_record['inputs_sha256'].items())
        bound_binaries=all(sha(build/name)==digest for name,digest in build_record['binaries_sha256'].items())
        if not bound_inputs or not bound_binaries:raise ValueError('source/binary build binding changed')
        report['measured_sources_sha256']=source_hashes
        report['build_manifest_sha256']=sha(build/'build.json')
        report['source_binary_binding_passed']=True
        weight_quant,act_quant,dot=native_functions(runtime,format)
        rng=np.random.default_rng(args.seed)
        row_bytes=size//rows
        packed=np.empty((banks,rows,row_bytes),np.uint8)
        for bank in range(banks):
            for row in range(rows):
                value=rng.standard_normal(width,dtype=np.float32)
                if row==0:value.fill(0)
                weight_quant(value.ctypes.data,packed[bank,row].ctypes.data,width)
        x=rng.standard_normal(width,dtype=np.float32)
        activation=np.empty(width//(256 if format==2 else 32)*(292 if format==2 else 34),np.uint8)
        act_quant(x.ctypes.data,activation.ctypes.data,width)
        report['packed_sha256']=hashlib.sha256(packed.tobytes()).hexdigest()
        report['activation_f32_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
        lib=ctypes.CDLL(str(build/'component.dylib'))
        lib.decode_component_create.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3+[ctypes.c_int]*4
        lib.decode_component_create.restype=ctypes.c_void_p
        lib.decode_component_run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_double)]
        lib.decode_component_run.restype=ctypes.c_int
        lib.decode_component_destroy.argtypes=[ctypes.c_void_p];lib.decode_component_destroy.restype=None
        cpu={n:CPU(lib,packed,x,format,n) for n in (1,10)}
        handles=list(cpu.values())
        expected=cpu[1].run()[0].copy()
        check10=numerical_comparison(cpu[10].run()[0],expected)
        independent=np.empty((banks,min(rows,16)),np.float32)
        for b in range(banks):
            for r in range(independent.shape[1]):
                dot(width,independent[b,r:].ctypes.data,0,packed[b,r].ctypes.data,0,activation.ctypes.data,0,1)
        report['cpu10_vs_cpu1']=check10
        report['graph_vs_independent_nrc1_dot']=numerical_comparison(expected[:,:independent.shape[1]].copy(),independent)
        samples={1:[],10:[]};orders=[]
        for rep in range(args.repetitions+3):
            order=[1,10];rng.shuffle(order)
            if rep>=3:orders.append(order)
            for n in order:
                _,seconds=cpu[n].run()
                if rep>=3:samples[n].append(seconds)
        report['cpu']={str(n):summary(v) for n,v in samples.items()}
        report['cpu_order']=orders
        for handle in handles:handle.close()
        handles=[]
        buffers=[]
        def add(value=None,bytes=None,name=None,output=False):
            i=len(buffers);spec={'bytes':value.nbytes if value is not None else bytes}
            if value is not None:
                path=root/(name+'.bin');path.write_bytes(value.tobytes());spec['input']=str(path)
            if output:spec['output']=str(root/(name+'.out'))
            buffers.append(spec);return i
        w0=add(packed[0],name='w0');w1=add(packed[-1],name='w1') if banks==2 else w0
        xi=add(x,name='x');ai=add(bytes=activation.nbytes,name='quantized',output=True)
        si=add(np.array([width,rows,banks],np.uint32),name='shape')
        one=add(np.array([width,rows,1],np.uint32),name='one') if banks==2 else si
        experiments=[];outputs={}
        for tile in (1,2,4):
            name=f'tile{tile}';oi=add(bytes=rows*banks*4,name=name,output=True);outputs[name]=oi
            lanes=32 if format==2 else 4
            grid=((rows+tile-1)//tile*lanes+31)//32*32
            stage={'entry':f'q{format}_tile{tile}','bindings':[w0,w1,ai,oi,si],
                   'grid':[grid,1,banks],'threadgroup':[32,1,1]}
            experiments.append({'name':name,'stages':[stage]})
        if banks==2:
            for separate in (False,True):
                name='pair_two_commands' if separate else 'pair_two_dispatches'
                stages=[];outputs[name]=[]
                for b,wi in enumerate((w0,w1)):
                    oi=add(bytes=rows*4,name=name+str(b),output=True);outputs[name].append(oi)
                    stages.append({'entry':f'q{format}_tile1','bindings':[wi,wi,ai,oi,one],
                                   'grid':[((rows*(32 if format==2 else 4)+31)//32)*32,1,1],'threadgroup':[32,1,1]})
                experiments.append({'name':name,'stages':stages,'separate_commands':separate})
        job={'source':str(Path(__file__).with_name('decode.metal').resolve()),'buffers':buffers,
             'cpu_quantize':{'format':format,'width':width,'input':xi,'output':ai},
             'experiments':experiments,'repetitions':args.repetitions}
        path=root/'job.json';path.write_text(json.dumps(job,indent=2)+'\n')
        before_hashes={p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
        completed=subprocess.run([str(build/'decode-timer'),str(path)],capture_output=True,text=True,
                                 timeout=90,pass_fds=(holder.fileno(),))
        (root/'timer.stdout.json').write_text(completed.stdout)
        (root/'timer.stderr.log').write_text(completed.stderr)
        report['timer_returncode']=completed.returncode
        if completed.returncode:raise RuntimeError(completed.stderr[:3000])
        metal=json.loads(completed.stdout);checks={}
        for name,indices in outputs.items():
            if isinstance(indices,list):
                out=np.stack([np.fromfile(buffers[i]['output'],np.float32) for i in indices])
            else:out=np.fromfile(buffers[indices]['output'],np.float32).reshape(banks,rows)
            checks[name]=numerical_comparison(out,expected)
        qactual=np.fromfile(buffers[ai]['output'],np.uint8)
        report.update(metal=metal,metal_vs_native=checks,
            activation_quantization_bit_identical=bool(np.array_equal(activation,qactual)),
            binaries_sha256={p.name:sha(p) for p in (build/'component.dylib',build/'decode-timer')},
            shader_sha256=sha(Path(__file__).with_name('decode.metal')),
            native_libraries_sha256=before_hashes,
            runtime_unchanged=before_hashes=={p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()})
        report['measured_sources_unchanged']=source_hashes=={source_root.joinpath(name).name:sha(source_root/name) for name in source_hashes}
        final=gate.sample(final=True)
        accepted=all(c['bit_identical'] for c in checks.values()) and check10['bit_identical'] and report['graph_vs_independent_nrc1_dot']['bit_identical'] and report['activation_quantization_bit_identical'] and report['runtime_unchanged'] and report['measured_sources_unchanged'] and final['passed'] and not power_errors
        report.update(status='completed' if accepted else 'rejected',component_fidelity_passed=accepted)
        report['limits']=['weights/activation synthetic; no GLM routes/logits/recurrent-state comparison',
            'all weights resident and warm; no SSD or model token-rate conclusion',
            'CPU timings use persistent native GGML graph, include native activation quantization; output copy excluded',
            'Metal total wall includes CPU quantization, command creation, dispatch, synchronization; setup and weight import excluded',
            'CPU and GPU phases sequential; kernel orders randomized, not a whole-model ABBA campaign']
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error),component_fidelity_passed=False)
    finally:
        for handle in handles:handle.close()
        stop.set();monitor.join(timeout=5)
        report['power_gate']=gate.summary();report['power_monitor_errors']=power_errors
        if monitor.is_alive() or power_errors or not report['power_gate']['passed']:
            report.update(status='rejected',component_fidelity_passed=False)
            report.setdefault('error','power monitor did not close with valid continuity')
        holder.close();report['uid_fd_closed']=True
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        (root/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','component_fidelity_passed','error')}))
    return 0 if report['status']=='completed' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime','build','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--format',type=int,choices=(2,8),required=True)
    p.add_argument('--rows',type=int,default=1536);p.add_argument('--width',type=int,default=4096)
    p.add_argument('--banks',type=int,choices=(1,2),default=1)
    p.add_argument('--seed',type=int,default=5303101);p.add_argument('--repetitions',type=int,default=15)
    raise SystemExit(run(p.parse_args()))
