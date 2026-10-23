#!/usr/bin/env python3
"""Bounded test of an installed GGUF with the user's existing llama.cpp binary.

This is an external engine baseline, not native MLX AirLLM validation. An owned
subprocess is monitored from spawn through shutdown. No model download occurs.
"""
import argparse
import ctypes
from ctypes.util import find_library
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import psutil

GiB=1024**3
MiB=1024**2


def disk_counter(pid):
    try:
        class Usage(ctypes.Structure):
            _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in (
                'user','system','idle_wakeups','interrupt_wakeups','pageins','wired','resident','footprint',
                'start','exit','child_user','child_system','child_idle_wakeups','child_interrupt_wakeups',
                'child_pageins','child_elapsed','disk_read','disk_write')]
        lib=ctypes.CDLL(find_library('proc') or '/usr/lib/libproc.dylib')
        value=Usage()
        if lib.proc_pid_rusage(pid,2,ctypes.byref(value))==0:
            return int(value.disk_read)
    except (OSError,AttributeError):
        pass
    return None


def run(args):
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
    command=[str(args.binary),'-m',str(args.model), '--host','127.0.0.1','--port',str(port),
        '--no-webui','--offline','--no-mmproj','--no-repack','--load-mode','mmap',
        '--no-warmup','--lazy-mode','on','--fit','off','-c',str(args.context),'-np','1','-b','16','-ub','16',
        '-ngl',str(args.gpu_layers),'-t','4','-tb','4','--no-cache-prompt']
    if args.cpu_moe:
        command+=['--cpu-moe','--moe-cache-mib',str(args.moe_cache_mib)]
    report=dict(kind='installed_gguf_external_engine_baseline',native_airllm_validated=False,
        engine='existing bundled llama.cpp',binary=str(args.binary),model_path=str(args.model),
        weight_file_bytes=args.model.stat().st_size,physical_memory_bytes=psutil.virtual_memory().total,
        command=command,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        os_cache='Uncontrolled; an earlier Ollama attempt read the same blob. No cold SSD claim.',
        gpu_compute_seconds=None,energy_joules=None,temperature_celsius=None,
        logical_weight_read_bytes=None,status='started',rss_peak_bytes=0,swap_growth_peak_bytes=0,
        minimum_available_bytes=psutil.virtual_memory().available,guard_violation=None,
        physical_counter_scope='Owned worker process OS-accounted storage bytes; not per-file or NAND bandwidth.',
        response_chunks=[])
    worker_env=os.environ.copy()
    if args.prefetch_library is not None:
        if not args.prefetch_library.is_file():raise FileNotFoundError(args.prefetch_library)
        worker_env['DYLD_INSERT_LIBRARIES']=str(args.prefetch_library.resolve())
        report['process_only_prefetch_interposer']=str(args.prefetch_library.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    base_swap=psutil.swap_memory().used
    report['swap_start_bytes']=base_swap
    stop=threading.Event()
    started=time.perf_counter()
    worker=None
    with args.output.with_suffix('.log').open('w') as log:
        worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=worker_env)
        report['owned_worker_pid']=worker.pid
        process=psutil.Process(worker.pid)
        def monitor():
            while not stop.wait(.02):
                if worker.poll() is not None:return
                try:
                    rss=process.memory_info().rss
                    available=psutil.virtual_memory().available
                    growth=max(0,psutil.swap_memory().used-base_swap)
                    report['rss_peak_bytes']=max(report['rss_peak_bytes'],rss)
                    report['minimum_available_bytes']=min(report['minimum_available_bytes'],available)
                    report['swap_growth_peak_bytes']=max(report['swap_growth_peak_bytes'],growth)
                    report['process_storage_read_bytes']=disk_counter(worker.pid)
                    cpu=process.cpu_times()
                    report['worker_cpu_seconds']=cpu.user+cpu.system
                    if (rss>args.max_rss_gib*GiB or available<args.reserve_gib*GiB
                            or growth>args.max_swap_growth_mib*MiB):
                        report['guard_violation']=dict(rss_bytes=rss,available_bytes=available,swap_growth_bytes=growth)
                        worker.terminate();return
                except psutil.NoSuchProcess:
                    return
                except BaseException as exc:
                    report['guard_violation']=dict(monitor_error=str(exc))
                    worker.terminate();return
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        try:
            deadline=started+120
            while True:
                if worker.poll() is not None:
                    raise RuntimeError('Owned worker exited before serving; inspect log/guard')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as health:
                        if health.status==200:break
                except (OSError,urllib.error.HTTPError):pass
                if time.perf_counter()>deadline:
                    raise TimeoutError('Server initialization exceeded 120s')
                time.sleep(.1)
            ready=time.perf_counter()
            report['server_load_seconds']=ready-started
            prompt='<|im_start|>user\n'+args.prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            request=dict(prompt=prompt,n_predict=args.tokens,temperature=0,seed=20261023,
                stream=True,cache_prompt=False,return_tokens=True)
            report['request']=request
            request_start=time.perf_counter()
            req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',
                data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=600) as response:
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data: '):continue
                    if line=='data: [DONE]':break
                    row=json.loads(line[6:]);report['response_chunks'].append(row)
                    if row.get('content') and 'ttft_excluding_load_seconds' not in report:
                        now=time.perf_counter()
                        report['ttft_excluding_load_seconds']=now-request_start
                        report['ttft_including_load_seconds']=now-started
                    if row.get('stop'):
                        report['final_response']=row;break
            end=time.perf_counter()
            if report['guard_violation']:
                raise MemoryError('Owned worker stopped by resource guard')
            final=report.get('final_response',{})
            if not final.get('stop'):
                raise RuntimeError('No terminal generation result')
            count=final.get('tokens_predicted',final.get('timings',{}).get('predicted_n'))
            report.update(status='complete',generated_tokens=count,
                generation_seconds=end-request_start,end_to_end_seconds=end-started,
                generated_text=''.join(r.get('content','') for r in report['response_chunks']),
                end_to_end_tokens_per_second=count/(end-started) if count else None,
                generation_tokens_per_second=count/(end-request_start) if count else None,
                server_timings=final.get('timings'),process_storage_read_bytes=disk_counter(worker.pid))
        except BaseException as exc:
            report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        finally:
            stop.set();watcher.join(timeout=2)
            if worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=5)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
            report.update(exit_code=worker.returncode,swap_final_bytes=psutil.swap_memory().used,
                total_worker_lifetime_seconds=time.perf_counter()-started)
            save()
    print(json.dumps({k:report.get(k) for k in ['status','error','generated_tokens','server_load_seconds',
        'ttft_including_load_seconds','generation_tokens_per_second','end_to_end_tokens_per_second',
        'rss_peak_bytes','swap_growth_peak_bytes','guard_violation']}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=Path('/Applications/Ollama.app/Contents/Resources/llama-server'))
    p.add_argument('--model',type=Path,default=Path.home()/'.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=1)
    p.add_argument('--prefetch-library',type=Path)
    p.add_argument('--prompt',default='Explique en deux phrases pourquoi le ciel est bleu.')
    p.add_argument('--context',type=int,default=256)
    p.add_argument('--gpu-layers',type=int,default=0)
    p.add_argument('--cpu-moe',action='store_true')
    p.add_argument('--moe-cache-mib',type=int,default=256)
    p.add_argument('--max-rss-gib',type=float,default=5)
    p.add_argument('--reserve-gib',type=float,default=1.5)
    p.add_argument('--max-swap-growth-mib',type=int,default=128)
    args=p.parse_args()
    if args.tokens<1 or args.context<64 or args.reserve_gib<1.5 or not 0<args.max_rss_gib<=6:
        p.error('Invalid counts or unsafe memory limits')
    if run(args)['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
