#!/usr/bin/env python3
"""Guarded measurement of an existing local Ollama model, not native AirLLM.

Uses only localhost, never downloads. A new runner for the exact selected
model may be terminated if the configured system-memory/swap guard trips.
The pre-existing Ollama service and unrelated models are never terminated.
"""
import argparse
import ctypes
from ctypes.util import find_library
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import psutil

GiB = 1024**3
MiB = 1024**2
URL = 'http://127.0.0.1:11434/api/'


def api(name, data=None):
    request = urllib.request.Request(URL+name,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='qwen3.6:latest')
    p.add_argument('--tokens',type=int,default=1)
    p.add_argument('--prompt',default='Explique en deux phrases pourquoi le ciel est bleu.')
    p.add_argument('--num-gpu',type=int,default=0)
    p.add_argument('--context',type=int,default=256)
    p.add_argument('--reserve-gib',type=float,default=1.5)
    p.add_argument('--max-swap-growth-mib',type=int,default=256)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.tokens<1 or args.reserve_gib<1.5 or args.context<64:
        p.error('Positive token count, context>=64 and reserve>=1.5GiB required')
    tags = api('tags')
    selected = next((m for m in tags['models'] if m['name']==args.model),None)
    if selected is None:
        raise ValueError('Exact model is not already installed; no pull attempted')
    running = api('ps')
    if running['models']:
        raise RuntimeError('An Ollama model is already active; do not disturb another task')
    shown = api('show',{'model':args.model})
    manifest = Path.home()/'.ollama/models/manifests/registry.ollama.ai/library'/args.model.replace(':','/')
    manifest_json = json.loads(manifest.read_text())
    layers = [dict(entry,path=str(Path.home()/'.ollama/models/blobs'/entry['digest'].replace(':','-')))
        for entry in manifest_json['layers']]
    if any(Path(e['path']).stat().st_size!=e['size'] for e in layers):
        raise ValueError('Installed blob size differs from its manifest')
    identity = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if identity!=selected['digest']:
        raise ValueError('Manifest digest differs from Ollama tag')
    report = dict(kind='existing_model_ollama_system_test',engine='Ollama',
        native_airllm_validated=False,installed_model=selected,version=api('version'),
        model_info=shown['model_info'],details=shown['details'],license='Apache-2.0' if 'Apache License' in shown.get('license','') else 'See local show output',
        layers=layers,manifest_sha256=identity,full_blob_sha256_verified_this_worker=False,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        physical_memory_bytes=psutil.virtual_memory().total,swap_start_bytes=psutil.swap_memory().used,
        available_start_bytes=psutil.virtual_memory().available,status='started',
        sampling_seconds=.05,runner_peak_rss_bytes=0,minimum_available_bytes=psutil.virtual_memory().available,
        swap_growth_peak_bytes=0,guard_violation=None,response_chunks=[],
        gpu_time_seconds=None,physical_read_bytes=None,
        physical_read_limit='Runner OS read accounting unavailable in this worker; no NAND/per-file bandwidth claim.',
        logical_weight_read_bytes=None,logical_read_limit='Ollama does not expose exact logical weight reads through this API.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    started = time.perf_counter()
    stop = threading.Event()
    first_content = None
    blob_paths = [e['path'] for e in layers if e['mediaType']=='application/vnd.ollama.image.model']
    def runners():
        result=[]
        for proc in psutil.process_iter(['pid','cmdline','create_time','memory_info']):
            try:
                command=proc.info['cmdline'] or []
                executable=Path(command[0]).name if command else ''
                if (proc.info['create_time']>=time.time()-(time.perf_counter()-started)-2
                        and (executable=='llama-server' or 'runner' in command)
                        and any(b in command for b in blob_paths)):
                    result.append(proc)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        return result
    def monitor():
        while not stop.wait(.05):
            available=psutil.virtual_memory().available
            growth=max(0,psutil.swap_memory().used-report['swap_start_bytes'])
            report['minimum_available_bytes']=min(report['minimum_available_bytes'],available)
            report['swap_growth_peak_bytes']=max(report['swap_growth_peak_bytes'],growth)
            own=runners()
            rss=sum(q.memory_info().rss for q in own if q.is_running())
            report['runner_peak_rss_bytes']=max(report['runner_peak_rss_bytes'],rss)
            if available<args.reserve_gib*GiB or growth>args.max_swap_growth_mib*MiB:
                report['guard_violation']=dict(available_bytes=available,swap_growth_bytes=growth)
                for proc in own:
                    try:proc.terminate()
                    except psutil.NoSuchProcess:pass
                return
    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()
    request=dict(model=args.model,messages=[{'role':'user','content':args.prompt}],stream=True,
        think=False,keep_alive=0,options=dict(num_predict=args.tokens,num_ctx=args.context,
            num_batch=16,temperature=0,seed=20261023,num_thread=4,num_gpu=args.num_gpu,use_mmap=True))
    report['request']=request
    try:
        req=urllib.request.Request(URL+'chat',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as response:
            for raw in response:
                row=json.loads(raw)
                report['response_chunks'].append(row)
                message=row.get('message',{})
                if first_content is None and (message.get('content') or message.get('thinking')):
                    first_content=time.perf_counter()
                    report['observed_ttft_seconds']=first_content-started
                if row.get('error'):
                    raise RuntimeError(row['error'])
                if row.get('done'):
                    report['final_response']=row
                    break
        end=time.perf_counter()
        if report['guard_violation']:
            raise MemoryError('System memory or swap guard stopped this runner')
        final=report.get('final_response',{})
        if not final.get('done'):
            raise RuntimeError('No terminal generation result received')
        count,duration=final.get('eval_count',0),final.get('eval_duration',0)/1e9
        report.update(status='complete',wall_seconds=end-started,
            generated_tokens=count,server_decode_tokens_per_second=count/duration if duration else None,
            end_to_end_tokens_per_second=count/(end-started),
            generated_text=''.join(c.get('message',{}).get('content','') for c in report['response_chunks']))
    except urllib.error.HTTPError as exc:
        report.update(status='failed',http_status=exc.code,error=exc.read().decode(),wall_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),wall_seconds=time.perf_counter()-started)
    finally:
        stop.set();watcher.join(timeout=2)
        report['swap_final_bytes']=psutil.swap_memory().used
        report['ps_after']=api('ps')
        save()
    print(json.dumps({k:report.get(k) for k in ['status','error','generated_tokens','server_decode_tokens_per_second','end_to_end_tokens_per_second','observed_ttft_seconds','runner_peak_rss_bytes','guard_violation']}),flush=True)
    if report['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
