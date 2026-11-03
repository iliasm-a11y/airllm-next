"""Guarded contemporary CPU decode profiling, with a metadata-only admission path.

The32GiB storage and5.5GiB available-memory reservations retain the preceding
local research campaign's conservative admission envelope. They are policy
limits, not a claimed measured peak. No command-line option lowers them.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from airllm.nextgen.experiment_guard import ExperimentLimits,supervise,system_snapshot
from airllm.nextgen.glm_native import inspect_native_checkpoint,FILES,_identity
from research.glm53_gpu_v3.power_state import PowerGate,ScopedAwake
from research.glm53_batched_prefill.uid_supervisor import acquire_uid_lock
from research.glm53_batched_prefill.native_run import identify,stop_owned
from .build import sha
from .campaign_limits import remaining_swap_budget

GIB=1<<30
REQUIRED_MEMORY=int(5.5*GIB)
REQUIRED_DISK=32*GIB

def resource_blockers(sample,disk_free,power,campaign_swap_used_bytes=None):
    reasons=[]
    if sample.get('pressure_level')!=1:reasons.append('memory_pressure_not_normal')
    if sample.get('available_bytes',0)<REQUIRED_MEMORY:reasons.append('available_memory_below_5.5GiB')
    if disk_free<REQUIRED_DISK:reasons.append('disk_reserve_below_32GiB')
    if not power.get('passed'):reasons.append('power_lid_or_clock_admission')
    try:remaining_swap_budget(sample,campaign_swap_used_bytes)
    except MemoryError as error:reasons.append(str(error))
    return reasons

def worker(args):
    holder,lock=acquire_uid_lock();child=None;identity=None;report={'uid_lock':lock}
    gate=PowerGate(expected_source='ac')
    command=[str(args.probe.resolve()),str(args.model.resolve()/FILES[0][0]),
             str(args.output.resolve()/'capture'),'10','0',str(args.tokens),args.prompt,'-','cpu']
    env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    env.update(AIRLLM_MMAP_DEMAND='1',AIRLLM_EXPERT_PREFETCH='1',AIRLLM_DENSE_PREFETCH='1',
               AIRLLM_DENSE_PREFETCH_MAX_COLS='1',AIRLLM_GPU_CACHE_MIB='0')
    if args.diagnostic:
        env.update(AIRLLM_PROFILE_OPS='1',AIRLLM_PROFILE_PAGES='1',AIRLLM_STATE_HASHES='1',AIRLLM_SOURCE_PLAN='1')
    else:env['AIRLLM_DECODE_TIMING_ONLY']='1'
    try:
        reasons=resource_blockers(system_snapshot(),shutil.disk_usage(args.output).free,gate.sample(),
                                 getattr(args,'campaign_swap_used_bytes',None))
        checkpoint=inspect_native_checkpoint(args.model,args.receipt)
        if not checkpoint['integrity_verified'] or not checkpoint['architecture_matches']:
            reasons.append('worker_checkpoint_integrity')
        if reasons:raise RuntimeError('worker re-admission refused: '+','.join(reasons))
        with (args.output/'native.stdout.log').open('wb') as out,(args.output/'native.stderr.log').open('wb') as err:
            child=subprocess.Popen(command,env=env,stdout=out,stderr=err,pass_fds=(holder.fileno(),))
            # Retain a Popen child, then pin its birth and executable before any
            # signal. The inherited UID FD survives a parent failure.
            for _ in range(100):
                if child.poll() is not None:break
                try:identity=identify(child,args.probe);break
                except Exception:time.sleep(.01)
            if identity is None:raise RuntimeError('native child identity not established')
            report['native_identity']=identity
            with ScopedAwake(child) as awake:
                while child.poll() is None:
                    time.sleep(.5)
                    if child.poll() is not None:break
                    identify(child,args.probe,identity)
                    awake.check()
                    remaining_swap_budget(system_snapshot(),getattr(args,'campaign_swap_used_bytes',None))
                    if not gate.sample()['passed']:
                        raise RuntimeError('power/lid/clock/owned-awake continuity failed')
                    if shutil.disk_usage(args.output).free<REQUIRED_DISK:
                        raise RuntimeError('disk reserve fell below32GiB')
            report['native_returncode']=child.wait(timeout=5)
            report['status']='completed' if child.returncode==0 and gate.sample(final=True)['passed'] else 'failed'
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error))
        if isinstance(error,KeyboardInterrupt):report['cancelled']=True
    finally:
        if child is not None:
            if child.poll() is None and identity is not None:report['cleanup']=stop_owned(child,args.probe,identity)
            elif child.poll() is not None:report['native_returncode']=child.wait(timeout=5)
        report['power']=gate.summary();holder.close();report['uid_parent_fd_closed']=True
        report['surviving_child_retains_uid_fd']=True
        (args.output/'worker.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report.get('status')=='completed' else 2

def run(args):
    if args.worker:return worker(args)
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    # Validate paths before creating artifacts; a capture must never live in
    # the model directory or the measured runtime directory.
    if output.is_relative_to(args.model.resolve()) or output.is_relative_to(args.runtime.resolve()):
        raise ValueError('output must be outside model/runtime directories')
    output.mkdir(parents=True)
    gate=PowerGate(expected_source='ac');power=gate.sample();sample=system_snapshot()
    disk=shutil.disk_usage(output).free
    baseline=getattr(args,'campaign_swap_used_bytes',None)
    reasons=resource_blockers(sample,disk,power,baseline)
    inspection=inspect_native_checkpoint(args.model,args.receipt)
    if not inspection['integrity_verified'] or not inspection['architecture_matches']:
        reasons.append('checkpoint_integrity_or_architecture_unverified')
    binaries={args.probe.name:sha(args.probe)}
    libraries={p.name:sha(p) for p in args.runtime.glob('*.dylib') if not p.is_symlink()}
    report={'schema':1,'utc':datetime.now(timezone.utc).isoformat(),'status':'admission_refused' if reasons else 'admitted',
        'blockers':reasons,'native_process_launched':False,'performance_measured':False,
        'tokens_requested':args.tokens,'threads':10,'gpu_layers':0,'physical_slots':1,
        'diagnostic':args.diagnostic,'prompt':args.prompt,'resource_sample':sample,'disk_free_bytes':disk,
        'policy':{'required_available_memory_bytes':REQUIRED_MEMORY,'disk_reserve_bytes':REQUIRED_DISK,
            'max_rss_bytes':int(8.5*GIB),'max_system_swap_growth_bytes':32<<20,
            'campaign_swap_used_bytes':baseline},
        'power_preflight':power,'checkpoint':inspection,'binaries_sha256':binaries,
        'native_libraries_sha256':libraries,
        'scope':'contemporary raw deterministic CPU reference; diagnostics serialize callbacks and are not throughput',
        'decode_tokens_per_second':None,'physical_nand_bytes':None,'production_changed':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if reasons:
        print(json.dumps({'status':report['status'],'blockers':reasons,'native_process_launched':False}));return 2
    command=[sys.executable,'-m','research.glm53_ultra_decode.model_profile','--worker',
        '--model',str(args.model.resolve()),'--receipt',str(args.receipt.resolve()),
        '--runtime',str(args.runtime.resolve()),'--probe',str(args.probe.resolve()),
        '--output',str(output),'--tokens',str(args.tokens),'--prompt',args.prompt]
    if args.diagnostic:command.append('--diagnostic')
    if baseline is not None:command.extend(('--campaign-swap-used-bytes',str(baseline)))
    limits=ExperimentLimits(rss_bytes=int(8.5*GIB),reserve_bytes=3*GIB,
        swap_growth_bytes=remaining_swap_budget(sample,baseline),admitted_payload_bytes=int(2.5*GIB),
        timeout_seconds=2400,interval_seconds=.5,max_processes=8)
    report['owned_wrapper_started']=True
    identities={args.model.resolve()/row['name']:row['identity'] for row in inspection['files']}
    try:
        if not all(_identity(p)==identity for p,identity in identities.items()):
            raise RuntimeError('weight identities changed after integrity admission')
        guard=supervise(command,output/'supervision',limits=limits)
        worker_report=json.loads((output/'worker.json').read_text())
        report['native_process_launched']='native_identity' in worker_report
        report['worker']=worker_report;report['guard']=guard
        unchanged=all(_identity(p)==identity for p,identity in identities.items())
        report['weight_identities_unchanged']=unchanged
        report['native_artifacts_unchanged']=binaries=={args.probe.name:sha(args.probe)} and libraries=={p.name:sha(p) for p in args.runtime.glob('*.dylib') if not p.is_symlink()}
        remaining_swap_budget(system_snapshot(),baseline)
        complete=output/'capture/complete.json'
        report['status']='completed' if worker_report['status']=='completed' and unchanged and report['native_artifacts_unchanged'] and complete.is_file() else 'failed'
        if report['status']=='completed':
            from .profile_analysis import analyze_capture
            report['analysis']=analyze_capture(output/'capture',diagnostic=args.diagnostic)
            report['performance_measured']=not args.diagnostic
            report['decode_tokens_per_second']=report['analysis']['decode_tokens_per_second'] if not args.diagnostic else None
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error))
        if (output/'worker.json').is_file():
            report['worker']=json.loads((output/'worker.json').read_text())
            report['native_process_launched']='native_identity' in report['worker']
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','native_process_launched','performance_measured')}))
    return 0 if report['status']=='completed' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','receipt','runtime','probe','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--tokens',type=int,choices=(32,64,128),default=32)
    p.add_argument('--prompt',default='La capitale de la France est')
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--campaign-swap-used-bytes',type=int,help='Common initial swap usage in bytes; only tightens the unchanged32MiB per-experiment cap')
    raise SystemExit(run(p.parse_args()))
