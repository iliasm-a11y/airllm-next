#!/usr/bin/env python3
"""Supervised native-GGUF profiling, generation and bounded parity captures.

GPU execution timestamps are not exposed by the installed MLX Python API.
Synchronous API wall time is explicitly not GPU-only time. Exclusive host
intervals do not overlap; inclusive component intervals must not be summed.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager,nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))


def save(path,obj):
    from tempfile import NamedTemporaryFile
    path=Path(path)
    with NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');name=f.name
    os.replace(name,path)


class Profile:
    def __init__(self):
        self.rows=defaultdict(lambda:dict(calls=0,wall=0.,cpu=0.,exclusive_wall=0.,exclusive_cpu=0.))
        self.stack=[];self.phase='prefill';self.steps=[];self.syncs=0

    @contextmanager
    def scope(self,label):
        frame=[perf_counter(),process_time(),0.,0.]
        self.stack.append(frame)
        try:yield
        finally:
            wall=perf_counter()-frame[0];cpu=process_time()-frame[1];self.stack.pop()
            r=self.rows[self.phase+'/'+label];r['calls']+=1;r['wall']+=wall;r['cpu']+=cpu
            r['exclusive_wall']+=max(0.,wall-frame[2]);r['exclusive_cpu']+=max(0.,cpu-frame[3])
            if self.stack:self.stack[-1][2]+=wall;self.stack[-1][3]+=cpu

    def wrap(self,obj,name,label):
        old=getattr(obj,name)
        def call(*a,**kw):
            with self.scope(label(*a,**kw) if callable(label) else label):return old(*a,**kw)
        setattr(obj,name,call)

    def install(self,model):
        import mlx.core as mx
        import airllm.nextgen.gguf_metal as metal
        import airllm.nextgen.gguf_qwen as qwen
        self.wrap(model.checkpoint,'_pread','pread')
        self.wrap(model.checkpoint,'read_rows','cpu_dequant_rows')
        self.wrap(model.reader,'_guard','allocation_guard')
        self.wrap(model.reader,'_packed_tile','packed_read_cache')
        self.wrap(model.reader,'_tile','float_read_cache')
        self.wrap(model.reader.cache,'get','software_cache_lookup')
        self.wrap(model.reader.cache,'put','software_cache_admit_evict')
        self.wrap(model.reader,'linear',lambda x,key,*a,**kw:
            'output_projection' if kw.get('output_role') else
            'selected_expert_projection' if 'exps.weight' in key else
            'shared_expert_projection' if 'shexp' in key or 'shared_expert' in key else
            'mixer_projection')
        self.wrap(model.reader,'linear_experts','selected_experts_group_projection')
        self.wrap(metal,'packed_matmul','packed_metal_call')
        self.wrap(metal,'packed_expert_matmul','grouped_packed_metal_call')
        self.wrap(qwen,'qwen_route','moe_routing')
        old_array=mx.array
        def array(*a,**kw):
            import numpy as np
            if a and isinstance(a[0],np.ndarray):
                with self.scope('host_to_mlx_array_api'):return old_array(*a,**kw)
            return old_array(*a,**kw)
        mx.array=array
        self.wrap(mx,'eval','eval_pending_graph_wall')
        self.wrap(mx,'async_eval','async_submit_api_wall')
        for i,mixer in enumerate(model.mixers):
            label='deltanet_total' if model.config['layer_types'][i]=='linear_attention' else 'attention_total'
            class Mixer:
                def __init__(self,inner,label,profile):self.inner=inner;self.label=label;self.profile=profile
                def __call__(self,*a,**kw):
                    with self.profile.scope(self.label):return self.inner(*a,**kw)
            model.mixers[i]=Mixer(mixer,label,self)

    def result(self):
        return {'groups':dict(self.rows),'scope':'exclusive host wall/process-CPU intervals; inclusive groups nest and must not be summed',
            'gpu_only_seconds':None,'gpu_timestamp_unavailable_reason':'Installed MLX Python API exposes capture but no per-command GPU timestamps',
            'os_page_cache_hits':None,'os_cache_reason':'No per-read OS cache hit counter; process storage bytes are reported separately'}


def worker(args):
    import numpy as np
    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    from airllm.nextgen.gguf_chat import GGUFChatSession
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.experiment_guard import system_snapshot
    from airllm.nextgen.physical_io import physical_read_bytes,physical_read_counter_info
    tasks=json.loads(args.tasks.read_text())
    report={'status':'started','variant':args.variant,'profile_enabled':args.profile,'capture_enabled':args.capture,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [*sorted((ROOT/'air_llm/airllm/nextgen').glob('*.py')),Path(__file__)]},
        'git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'model_file_bytes':args.model.stat().st_size,'model_path':str(args.model),'before':system_snapshot(),
        'storage_counter':physical_read_counter_info(),'turns':[]}
    save(args.output,report);entry=perf_counter()
    with GGUFReader(args.model,memory_budget_bytes=256<<20,max_metadata_bytes=192<<20) as cp:
        tokenizer=GGUFTokenizer(cp.metadata)
    del cp
    cfg=NextConfig(cache_bytes=64<<20,io_bytes=64<<20,kv_bytes=256<<20,workspace_bytes=128<<20,
        reserve_bytes=3<<30,max_tensor_bytes=16<<20,row_chunk=args.rows,prefill_chunk=16,max_context=1280,
        policy=args.cache_policy,device='gpu',dtype='float32',prefetch=False,load_tokenizer=False,io_mode='pread',zero_copy_weights=args.zero_copy_weights)
    report['options']=asdict(cfg)
    kwargs={'expert_batch':args.variant in ('expert-batch','combined','pipeline'),
        'concat_tiles':args.variant in ('concat','combined','pipeline'),'pipeline_tiles':args.variant=='pipeline','metal_simd_groups':args.simd_groups}
    try:
        with NativeGGUFQwenModel(args.model,cfg,kernel='metal',**kwargs) as model:
            profile=Profile() if args.profile else None
            if profile:profile.install(model)
            original=model.forward;step=[0]
            def forward(*a,**kw):
                if profile:profile.phase='prefill' if step[0]==0 else 'decode'
                logical_before=model.checkpoint.stats['logical_bytes'];reads_before=model.checkpoint.stats['read_calls']
                disk_before=physical_read_bytes();begin=perf_counter();cpu_begin=process_time()
                with (profile.scope('model_forward_total') if profile else nullcontext()):result=original(*a,**kw)
                forward_wall=perf_counter()-begin;forward_cpu=process_time()-cpu_begin;disk_after=physical_read_bytes()
                logits=np.asarray(result.logits)
                routes={str(i):m.routes for i,m in enumerate(model.mlps)}
                checks.append({'step':step[0],'logits_sha256':hashlib.sha256(logits.tobytes()).hexdigest(),
                    'routes_sha256':hashlib.sha256(json.dumps(routes,sort_keys=True).encode()).hexdigest(),
                    'position':result.past_key_values.position,'forward_wall_seconds':forward_wall,'forward_cpu_seconds':forward_cpu,
                    'logical_read_bytes':model.checkpoint.stats['logical_bytes']-logical_before,
                    'pread_calls':model.checkpoint.stats['read_calls']-reads_before,
                    'storage_read_bytes':None if disk_before is None or disk_after is None else disk_after-disk_before})
                if args.capture:
                    folder=args.output.with_suffix('.capture');folder.mkdir(exist_ok=True)
                    np.save(folder/f'{step[0]:04d}.npy',logits,allow_pickle=False)
                    state=hashlib.sha256()
                    for bank,values in sorted(result.past_key_values.banks.items()):
                        for layer,value in sorted(values.items()):
                            for array in value if isinstance(value,(tuple,list)) else [value]:
                                host=np.asarray(array);state.update(f'{bank}/{layer}/{host.shape}/{host.dtype}'.encode());state.update(host.tobytes())
                    captures.append({'step':step[0],'logits_sha256':hashlib.sha256(logits.tobytes()).hexdigest(),
                        'state_sha256':state.hexdigest(),'routes':routes,
                        'position':result.past_key_values.position})
                step[0]+=1
                return result
            model.forward=forward
            report['initialization_seconds']=perf_counter()-entry
            for spec in tasks:
                step[0]=0;captures=[];checks=[]
                chat=GGUFChatSession(model,tokenizer,system_prompt='Tu es un assistant utile. Réponds en français.')
                item=dict(spec,status='started',partial_ids=[],partial_text='');report['turns'].append(item)
                logical=model.checkpoint.stats['logical_bytes'];reads=model.checkpoint.stats['read_calls'];storage=physical_read_bytes()
                start=perf_counter();cpu=process_time()
                def emitted(event):
                    item['partial_ids'].append(event['token_id']);item['partial_text']=tokenizer.decode(item['partial_ids'],skip_special_tokens=True)
                    item['partial_elapsed_seconds']=perf_counter()-start;save(args.output,report)
                save(args.output,report)
                result=chat.reply(spec['prompt'],max_new_tokens=spec['max_new_tokens'],on_token=emitted)
                item.update(result);item['status']='completed';item['wall_seconds']=perf_counter()-start;item['cpu_seconds']=process_time()-cpu
                item['logical_read_bytes']=model.checkpoint.stats['logical_bytes']-logical
                item['pread_calls']=model.checkpoint.stats['read_calls']-reads
                storage_end=physical_read_bytes();item['process_storage_read_bytes']=None if storage is None or storage_end is None else storage_end-storage
                item['metrics']=model.metrics()
                item['step_checks']=checks
                if args.capture:item['captures']=captures
                if profile:item['profile']=profile.result()
                print(json.dumps({'id':spec['id'],'text':item['text'],'ttft':item['ttft_seconds'],
                    'decode_tok_s':item['decode_tokens_per_second'],'stop':item['stop_reason']},ensure_ascii=False),flush=True)
                save(args.output,report)
            report['status']='completed'
    except BaseException as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['after']=system_snapshot();report['worker_wall_seconds']=perf_counter()-entry;save(args.output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['baseline','expert-batch','concat','combined','pipeline'],default='baseline')
    p.add_argument('--profile',action='store_true');p.add_argument('--capture',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--rows',type=int,default=2048);p.add_argument('--timeout',type=float,default=900)
    p.add_argument('--zero-copy-weights',action='store_true');p.add_argument('--cache-policy',choices=['adaptive','lru'],default='adaptive')
    p.add_argument('--simd-groups',type=int,choices=[1,2,4,8],default=1)
    args=p.parse_args()
    from airllm.nextgen.experiment_guard import ExperimentLimits,supervise,validate_output_paths
    validate_output_paths(args.output,protected=[args.model,args.tasks])
    args.output.parent.mkdir(exist_ok=True,parents=True)
    if args.worker:worker(args)
    else:supervise([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],args.output,
        ExperimentLimits(admitted_payload_bytes=768<<20,timeout_seconds=args.timeout),
        env=dict(os.environ,MLX_ENABLE_TF32='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'))


if __name__=='__main__':main()
