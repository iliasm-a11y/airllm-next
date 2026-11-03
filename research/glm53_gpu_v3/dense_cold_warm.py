"""Prepare or explicitly execute one bounded original-matrix CPU diagnostic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from datetime import datetime,timezone
import psutil
from research.glm53_gpu_v2.storage_counters import block_snapshot,compare_snapshots
from research.glm53_gpu_v2.benchmark_lifecycle import stop_owned_server

ROOT=Path(__file__).parent


def prepare(args):
    inventory=json.loads(args.inventory.read_text())
    tensor=next(t for t in inventory['tensors'] if t['name']=='blk.0.attn_q.weight')
    if tensor['type']!='Q8_0' or tensor['dims']!=[4096,8192] or tensor['bytes_with_alignment']!=35651584:
        raise ValueError('pinned original Q8_0[4096,8192] descriptor required')
    header=next(f for f in inventory['files'] if f['file']==tensor['file'])
    records=[json.loads(line) for line in (args.capture/'tensors.jsonl').read_text().splitlines()]
    activation=next(r for r in records if r['step']==1 and r['name']=='attn_norm-40' and r['shape']==[4096,1,1,1]
                    and r['type']==0 and r['bytes']==16384 and r['strides'][0]==4)
    source=args.capture/activation['file'];data=source.read_bytes()
    if len(data)!=16384:raise ValueError('captured activation must contain exactly4096float32 values')
    args.directory.mkdir(parents=True)
    activation_path=args.directory/'activation.f32';activation_path.write_bytes(data)
    return {'tensor':tensor,'model_revision':inventory['revision'],'absolute_offset':header['data_offset']+tensor['offset'],
            'activation_file':str(activation_path.resolve()),'activation_source':str(source.resolve()),
            'activation_scope':'real captured layer40 decode activation, width-matched to layer0 matrix; not an actual layer0 model execution',
            'activation_sha256':hashlib.sha256(data).hexdigest(),
            'known_payload_bound_bytes':tensor['bytes_with_alignment']+16384+8*32768+1024**2,
            'weight_integrity_scope':'pinned descriptor plus unchanged file stats; no selected-weight read/hash before first phase'}


def run(args):
    if args.directory.exists() or args.output.exists():raise ValueError('refusing existing component evidence')
    report={'schema_version':1,'status':'preparing','started_utc':datetime.now(timezone.utc).isoformat(),
            'phases':[],'model_decoder_executed':False,'gpu_initialized':False,'guard_violation':None,
            'scope':'one original full Q8_0 matrix on native CPU10; first-use and repeated same-process phases, not model throughput'}
    try:
        report.update(prepare(args))
        report['persistent_cpu_threads']=10
        report['guard_limits']={'rss_bytes':256*1024**2,'system_swap_growth_bytes':32*1024**2,'timeout_seconds':60,'critical_pressure':4}
        if report['known_payload_bound_bytes']>128*1024**2:raise MemoryError('payload bound exceeds128MiB')
        shard=args.model_directory/report['tensor']['file'];before=shard.stat()
        report['source_file_stats_before']={'size':before.st_size,'mtime_ns':before.st_mtime_ns}
        if not args.execute:
            report['status']='prepared_no_checkpoint_payload_read'
            return report
        binary=args.directory/'dense-cold-warm'
        compile_command=['clang++','-std=c++17','-O2','-I'+str(args.source/'ggml/include'),
            str(ROOT/'dense_cold_warm.cpp'),'-L'+str(args.libraries),'-lggml-cpu','-lggml-base',
            '-Wl,-rpath,'+str(args.libraries.resolve()),'-o',str(binary)]
        subprocess.run(compile_command,check=True,capture_output=True,text=True,timeout=30)
        report['component_source_sha256']=hashlib.sha256((ROOT/'dense_cold_warm.cpp').read_bytes()).hexdigest()
        report['native_libraries_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.libraries.glob('*.dylib') if not p.is_symlink()}
        command=[str(binary.resolve()),str(shard.resolve()),str(report['absolute_offset']),report['activation_file'],str((args.directory/'native').resolve()),str(args.phases)]
        report['command']=command;report['swap_start_bytes']=psutil.swap_memory().used
        report['pressure_start']=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True,timeout=3))
        if report['pressure_start'] not in (1,2):raise MemoryError('memory pressure unknown or critical before component launch')
        report.update(peak_rss_bytes=0,peak_swap_growth_bytes=0)
        env={k:v for k,v in os.environ.items() if not k.startswith(('AIRLLM_','GGML_','DYLD_','LLAMA_ARG_'))}
        stop=threading.Event();child=None;start=time.monotonic()
        def monitor():
            tick=0
            try:
                while not stop.wait(.05) and child.poll() is None:
                    rss=psutil.Process(child.pid).memory_info().rss
                    swap=max(0,psutil.swap_memory().used-report['swap_start_bytes'])
                    report['peak_rss_bytes']=max(report['peak_rss_bytes'],rss)
                    report['peak_swap_growth_bytes']=max(report['peak_swap_growth_bytes'],swap)
                    level=(int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True,timeout=3)) if tick%4==0 else None)
                    if level is not None and level not in (1,2,4):raise RuntimeError('unknown memory pressure')
                    if rss>256*1024**2 or swap>32*1024**2 or level==4 or psutil.virtual_memory().available<256*1024**2 or time.monotonic()-start>60:
                        report['guard_violation']={'rss_bytes':rss,'system_swap_growth_bytes':swap,'pressure':level,'elapsed_seconds':time.monotonic()-start}
                        stop_owned_server(child);return
                    tick+=1
            except psutil.NoSuchProcess:pass
            except Exception as exc:
                report['guard_violation']={'monitor_error':str(exc)};stop_owned_server(child)
        watcher=None
        try:
            with (args.directory/'native.stderr.log').open('w') as stderr:
                child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,env=env,start_new_session=True)
                watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
                for index in range(args.phases):
                    ready=json.loads(child.stdout.readline())
                    if ready.get('ready')!=index:raise RuntimeError('native phase synchronization failed')
                    block_before=block_snapshot();child.stdin.write('run\n');child.stdin.flush()
                    phase=json.loads(child.stdout.readline());block_after=block_snapshot()
                    if phase.get('phase')!=index:raise RuntimeError('native phase result missing')
                    phase['block_device_before']=block_before;phase['block_device_after']=block_after
                    phase['block_device_delta']=compare_snapshots(block_before,block_after)
                    output=(args.directory/'native'/phase['output_file']).read_bytes()
                    if len(output)!=8192*4:raise RuntimeError('full8192-output vector missing')
                    phase['output_sha256']=hashlib.sha256(output).hexdigest()
                    phase['label']='first_use_observed_reads_not_proven_cold' if index==0 else 'same_process_repeat_not_assumed_storage_free'
                    report['phases'].append(phase)
                code=child.wait(timeout=5);report['native_returncode']=code
                if code or report['guard_violation']:raise RuntimeError('native component rejected')
        finally:
            if child is not None:
                stop_owned_server(child);report['native_returncode']=child.returncode
            stop.set()
            if watcher is not None:watcher.join(timeout=5)
            report['swap_final_bytes']=psutil.swap_memory().used
            report['peak_swap_growth_bytes']=max(report['peak_swap_growth_bytes'],report['swap_final_bytes']-report['swap_start_bytes'],0)
            if report['peak_swap_growth_bytes']>32*1024**2:
                report['guard_violation']=report['guard_violation'] or {'final_system_swap_growth_bytes':report['peak_swap_growth_bytes']}
        if report['guard_violation']:raise RuntimeError('component resource guard rejected run')
        after=shard.stat();report['source_file_stats_unchanged']=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        report['all_outputs_bit_identical']=len({p['output_sha256'] for p in report['phases']})==1
        if not report['source_file_stats_unchanged'] or not report['all_outputs_bit_identical']:raise RuntimeError('identity check failed')
        report['status']='complete'
    except BaseException as exc:report.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
    finally:report['finished_utc']=datetime.now(timezone.utc).isoformat()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);work=ROOT.resolve().parents[2]
    p.add_argument('--inventory',type=Path,default=ROOT.parents[0]/'glm53_m5/sources/gguf-headers.json')
    p.add_argument('--capture',type=Path,default=work/'v2/trace2-cpu')
    p.add_argument('--source',type=Path,default=work/'llama-v3');p.add_argument('--libraries',type=Path,default=work/'runtime-v3-build/bin')
    p.add_argument('--model-directory',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phases',type=int,choices=range(2,9),default=4)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    def interrupt(*unused):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);report=run(args)
    finally:signal.signal(signal.SIGTERM,previous)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'output':str(args.output),'error':report.get('error')}))
    return 0 if report['status'] in ('complete','prepared_no_checkpoint_payload_read') else 2


if __name__=='__main__':raise SystemExit(main())
