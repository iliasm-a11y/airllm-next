"""Fresh two-shape ABBA controller. Plan only unless --execute is selected.

Reuses exact existing provider-fixture-warm-v2 binary/build. Never compiles or
loads a model. Existing fixture validators, native guards and sources stay intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
FROZEN=BRIDGE/'fixtures-warm-v2'
PREPARE=HERE/'prepare.json'
DATA_LIMIT=64<<20
RSS_LIMIT=128<<20
GLOBAL_SECONDS=240
CHILD_SECONDS=20
POLL_SECONDS=.1


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()

def read(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>8<<20:raise ValueError('bounded existing metadata required')
    return json.loads(path.read_text())

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)

def verify_file_pins(pins):
    for path,expected in pins.items():
        if sha(path)!=expected:raise ValueError('reviewed ABBA input changed: '+path)

def verify_pins():
    record=read(PREPARE)
    verify_file_pins(record['pins_sha256'])
    return record

def load_frozen():
    verify_pins()
    existing=sys.modules.get('gpu_contract')
    if existing is not None and Path(existing.__file__).resolve()!=BRIDGE/'gpu_contract.py':
        raise ValueError('GPU contract already imported from another candidate')
    spec=importlib.util.spec_from_file_location('abba_frozen_fixture',FROZEN/'provider_fixture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.gc.__file__).resolve()!=BRIDGE/'gpu_contract.py':raise ValueError('GPU contract path differs')
    if (module.RSS_LIMIT,module.POLL_SECONDS,module.GLOBAL_SECONDS,module.CHILD_SECONDS)!=(RSS_LIMIT,POLL_SECONDS,GLOBAL_SECONDS,CHILD_SECONDS):
        raise ValueError('frozen component guard constants differ')
    verify_pins()
    return module

def cases():
    return [{'id':f'm8192-k4096-n{n}-normal','M':8192,'K':4096,'N':n,'mode':'normal','expectation':'gpu'} for n in (8,16)]

def comparison_passed(on,result):
    return (type(on) is bool and result.get('all_finite') is True and
            result.get('metal_tolerance_passed' if on else 'cpu_bitexact') is True)

def plan():
    record=verify_pins()
    return {'schema_version':1,'status':'prepared_not_executed','cases':cases(),
        'order':['CPU','GPU','GPU','CPU'],'processes_max':8,'comparisons_per_case':4,
        'comparisons_scope':'first CPU self-finite validation, GPU1/oracle, GPU2/oracle, lastCPU/oracle',
        'repetitions_per_completed_graph':5,'first_iteration_includes_provider_initialization':True,
        'oracle':'first same-package ordinary CPU10 graph, bridgeOFF',
        'warm_scope':'four resident graph iterations; no model/SSD/end-to-end inference speed claim',
        'child_rss_ceiling_bytes':RSS_LIMIT,'controller_rss_ceiling_bytes':RSS_LIMIT,
        'native_lifetime_rss_required':True,'logical_fixture_limit_bytes':DATA_LIMIT,
        'metal_device_allocation_ceiling_bytes':DATA_LIMIT,'controller_additional_reservation_bytes':64<<20,
        'global_deadline_seconds':GLOBAL_SECONDS,'child_deadline_seconds':CHILD_SECONDS,
        'sampling_interval_seconds':POLL_SECONDS,'fresh_source_pins':record['pins_sha256'],
        'fixture_build_receipt_sha256':record['fixture_build_receipt_sha256'],
        'binary_sha256':record['binary_sha256'],'reused_exact_existing_binary':True,
        'build_performed':False,'model_loaded':False,'gpu_commands':0,'full_model_qualified':False,
        'measured_elapsed_seconds':None,'observed_peak_child_rss_bytes':None,
        'observed_peak_controller_rss_bytes':None,'scope':'two synthetic projection shapes only; preparation is not execution'}

def execute(args):
    # Existing native binary and validators are reused; never recompile.
    f=load_frozen();gc=f.gc
    validate_prepared=f.validate_prepared;validate_repetitions=f.validate_repetitions
    compare_raw=f.compare_raw;trace_rows=f.trace_rows
    sys.path.insert(0,str(BRIDGE));import native_gpu_run as n
    verify_pins()
    if Path(n.__file__).resolve()!=BRIDGE/'native_gpu_run.py':raise ValueError('native guard imported from another candidate')
    overall_started=time.monotonic()
    binary=Path(args.binary).resolve();package=Path(args.package_build).resolve();output=Path(args.output).resolve()
    prepared=verify_pins()
    gc.require(binary==Path(prepared['binary']).resolve() and sha(binary)==prepared['binary_sha256']
        and Path(args.fixture_build).resolve()==FROZEN/'build-receipt.json'
        and sha(args.fixture_build)==prepared['fixture_build_receipt_sha256'] and package==BRIDGE/'build-run1',
        'FIXTURE_BUILD','ABBA must reuse the exact reviewed existing binary/build/package')
    gc.require(not output.exists() and binary.is_file(),'FIXTURE_OUTPUT','Fresh fixture output and built native binary required')
    record=plan();record.update(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),results=[],gpu_commands=0)
    output.mkdir(mode=0o700,parents=True)
    provider=gc.read_json(package/'provider-build.json');driver=gc.read_json(package/'driver-build.json');document=gc.read_json(package/'gpu-profile.json')
    gc.validate_artifacts(document,provider,driver,libraries=package/'runtime/bin',binary=package/'native-prefill-gpu',
        helper_source=BRIDGE/'driver/research/glm53_batched_prefill/native_prefill.cpp',runner_source=BRIDGE/'native_gpu_run.py',
        guard_source=n.NATIVE_GUARD_SOURCE,provider_receipt_sha256=sha(package/'provider-build.json'))
    built=gc.read_json(args.fixture_build)
    roles=gc.validate_input_provenance(built,('fixture_source','build_script','controller_source',*('header:'+name for name in gc.HEADERS)))
    gc.require(built.get('status')=='built_metadata_only_smoked' and built.get('returncode')==0 and built.get('binary_sha256')==sha(binary)
        and built.get('provider_build_receipt_sha256')==sha(package/'provider-build.json')
        and built.get('native_libraries_sha256')==document['canonical_identity']['fingerprints']['native_libraries_sha256']
        and Path(roles['fixture_source']).resolve()==FROZEN/'provider_fixture.cpp'
        and Path(roles['controller_source']).resolve()==FROZEN/'provider_fixture.py','FIXTURE_BUILD','Exact fixture build/source/package receipt required')
    gc.require(Path(gc.profiles.__file__).resolve()==BRIDGE/'profile.py' and Path(n.__file__).resolve()==BRIDGE/'native_gpu_run.py',
        'FIXTURE_MODULE_SOURCE','Profile and runner must load from the actual isolated candidate')
    tracked={path:sha(path) for path in {str(binary),str(Path(__file__).resolve()),str(FROZEN/'provider_fixture.cpp'),str(FROZEN/'provider_fixture.py'),str(HERE/'prepare.json'),str(Path(args.fixture_build).resolve()),
        str(BRIDGE/'profile.py'),str(BRIDGE/'native_gpu_run.py'),
        str(package/'provider-build.json'),str(package/'driver-build.json'),str(package/'gpu-profile.json'),str(BRIDGE/'gpu_contract.py'),str(n.NATIVE_GUARD_SOURCE)}}
    record.update(profile_sha256=document['canonical_sha256'],binary_sha256=sha(binary),tracked_inputs_sha256=tracked,
        fixture_build_receipt_sha256=sha(args.fixture_build),native_libraries_sha256=provider['native_libraries_sha256'],
        reused_exact_existing_binary=True,frozen_controller_sha256=sha(FROZEN/'provider_fixture.py'),
        abba_controller_sha256=sha(__file__),prepared_sha256=sha(PREPARE))
    tracked.update({str(path.resolve()):sha(path) for path in (package/'runtime/bin').glob('*.dylib') if not path.is_symlink()})
    tracked.update(read(PREPARE)['pins_sha256'])
    identities={path:n._identity(Path(path)) for path in tracked}
    record['tracked_file_identities']=identities
    def unchanged():
        try:return all(n._identity(Path(path))==expected for path,expected in identities.items())
        except OSError:return False
    limits=n.NativeLimits(**{**vars(n.NativeLimits()),**gc.profiles.NATIVE_MEMORY_POLICY});guard=n.guard_model(limits)
    gate=n.PowerGate('ac');holder=None;child=None;identity=None;awake=None;started=overall_started;handlers={}
    max_child=max_controller=None;baseline=None
    def interrupted(signum,frame):raise n.qa.ProbeInterrupted(signum)
    def check(sample,*,startup=False):
        nonlocal max_child,max_controller
        gc.require(type(sample.get('available_bytes')) is int and type(sample.get('pressure_level')) is int
            and type(sample.get('swap_used_bytes')) is int,'GPU_RESOURCE_SAMPLER','Exact resource metadata required')
        rss=sample.get('rss_bytes',0)
        gc.require(type(rss) is int and rss>=0,'GPU_RESOURCE_SAMPLER','Exact child RSS metadata required')
        growth=max(0,sample['swap_used_bytes']-baseline['swap_used_bytes'])
        code=n.NativeGlmModel._guard_resource_sample(guard,sample,baseline,time.monotonic())
        if code:raise n.qa.QualificationError(code,'Original native watchdog refused component sample')
        gc.require(gc.sufficient_memory_sample({'pressure_level':sample['pressure_level'],
            'available_bytes':sample['available_bytes']-(64<<20),'rss_bytes':rss,'swap_growth_bytes':growth},baseline['swap_used_bytes']),
            'GPU_RESOURCE_ADMISSION','Sufficient native headroom plus controller64MiB reservation required')
        if startup:gc.require(sample['pressure_level']==1,'MEMORY_ADMISSION','Initial child admission remains normal pressure1')
        max_child=max(max_child or 0,rss) if 'rss_bytes' in sample else max_child
        controller=n.psutil.Process().memory_info().rss;max_controller=max(max_controller or 0,controller)
        gc.require(rss<=RSS_LIMIT and controller<=RSS_LIMIT,'FIXTURE_RSS_LIMIT','Component child/controller each limited to128MiB observedRSS')
        gc.require(gate.sample()['passed'],'POWER_CONTEXT_CHANGED','AC and open lid required')
        gc.require(unchanged(),'ARTIFACT_IDENTITY_CHANGED','Fixture source/binary/package changed')
        gc.require(time.monotonic()-started<=GLOBAL_SECONDS,'DEADLINE','Global fixture deadline240s')
        return {**sample,'swap_growth_bytes':growth,'controller_rss_bytes':controller}
    with n.qa.BENCHMARK_LOCK.open('a') as benchmark:
        fcntl.flock(benchmark,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
            gc.require(gate.sample()['passed'],'POWER_CONTEXT_UNAVAILABLE','Stable AC/open lid required')
            holder,record['uid_model_lock']=n.acquire_uid_model_lock()
            baseline=n.resource_snapshot();record['baseline']=baseline;check(baseline,startup=True)
            with (output/'samples.jsonl').open('x') as samples:
                for case in record['cases']:
                    pair={'case':case,'processes':[],'comparisons_to_first_cpu':[]};record['results'].append(pair)
                    for order_index,on in enumerate((False,True,True,False)):
                        check(n.resource_snapshot(),startup=True)
                        root=output/case['id']/(f'{order_index+1:02d}-'+('bridge-on' if on else 'bridge-off'));root.mkdir(parents=True,mode=0o700)
                        env,_=n.clean_environment({});env.update(document['environment'] if on else {'AIRLLM_CPU_METAL_Q8_BRIDGE':'0'})
                        env['AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES']=str(baseline['swap_used_bytes'])
                        read_fd,write_fd=os.pipe();env['AIRLLM_FIXTURE_START_FD']=str(read_fd)
                        command=[str(binary),str(root),str(case['M']),str(case['K']),str(case['N']),case['mode'],'on' if on else 'off']
                        job={'order_index':order_index,'abba_label':('A1','B1','B2','A2')[order_index],'output_root':str(root),'bridge':on,'command':command,'status':'starting','elapsed_seconds':None,'returncode':None,'actual_gpu':False}
                        pair['processes'].append(job);child=None;identity=None;awake=None;job_start=time.monotonic()
                        try:
                            with (root/'stdout.log').open('wb') as stdout,(root/'stderr.log').open('wb') as stderr:
                                child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,pass_fds=(holder.fileno(),read_fd))
                                os.close(read_fd);read_fd=-1
                                identity=n.identify(child,binary);job['owned_identity']=identity
                                job['pre_compute_sample']=check(n.resource_snapshot(child.pid))
                                observation=n.observe_uid_fd(child,binary,identity,record['uid_model_lock']);job['uid_fd_observation']=observation
                                gc.require(observation['observed_matching_lock_fd'],'UID_LOCK_INHERITANCE','Owned fixture must inherit the UID lock before compute')
                                awake=n.ScopedAwake(child);awake.__enter__();os.write(write_fd,b'G');os.close(write_fd);write_fd=-1
                                while child.poll() is None:
                                    try:
                                        n.identify(child,binary,identity);awake.check()
                                        current=n.resource_snapshot(child.pid)
                                    except (n.psutil.NoSuchProcess,n.qa.QualificationError,RuntimeError) as error:
                                        natural_exit=child.poll() is not None and (isinstance(error,(n.psutil.NoSuchProcess,RuntimeError)) or getattr(error,'code',None)=='OWNED_PROCESS_EXITED')
                                        if natural_exit:break
                                        raise
                                    sample=check(current);sample.update(case=case['id'],bridge=on,order_index=order_index,elapsed_seconds=time.monotonic()-started)
                                    samples.write(json.dumps(sample,allow_nan=False)+'\n');samples.flush()
                                    if time.monotonic()-job_start>CHILD_SECONDS:raise n.qa.QualificationError('DEADLINE','Fixture child deadline20s')
                                    if (root/'stderr.log').stat().st_size>32<<20:raise n.qa.QualificationError('GPU_TRACE_SIZE','Native log exceeds32MiB')
                                    time.sleep(POLL_SECONDS)
                                job.update(returncode=child.wait(timeout=5),status='completed',elapsed_seconds=time.monotonic()-job_start)
                        except BaseException as error:
                            job.update(status='failed',error_code=getattr(error,'code',type(error).__name__),error=str(error))
                            raise
                        finally:
                            saved={sig:signal.signal(sig,signal.SIG_IGN) for sig in handlers}
                            try:
                                for fd in (read_fd,write_fd):
                                    if fd>=0:os.close(fd)
                                if child is not None:job['cleanup']=n.stop_owned(child,binary,identity);job['returncode']=child.poll()
                                if awake is not None:
                                    awake.close();job['idle_assertion']=awake.record
                            except Exception as error:
                                job['cleanup_error']=str(error)
                                if awake is not None:job['idle_assertion']=awake.record
                                if 'error_code' not in job:raise
                            finally:
                                for sig,handler in saved.items():signal.signal(sig,handler)
                        job['post_compute_global_sample']=check(n.resource_snapshot())
                        prepared=gc.read_json(root/'prepared.json');validate_prepared(prepared,case,on,document,package/'runtime/bin')
                        job['prepared']=prepared;text=(root/'stderr.log').read_text(errors='replace');rows=trace_rows(text)
                        if on:
                            gc.require(all(pair['processes'][0]['prepared'][key]==prepared[key] for key in ('weight_sha256','input_sha256')),
                                'FIXTURE_INPUT_IDENTITY','Separate native processes produced different actual quantized inputs')
                        submissions=sum(row.get('event')=='submitted' for row in rows);record['gpu_commands']+=submissions;job['actual_gpu']=submissions>0
                        gc.require(job['returncode']==0,'NATIVE_NONZERO_EXIT','Unexpected native fixture failure')
                        job['result']=gc.read_json(root/'result.json');job['reuse']=validate_repetitions(job['result'])
                        if on:
                            job['gpu_trace']=gc.parse_trace(text,document,baseline_swap=baseline['swap_used_bytes'])
                            gc.require(job['gpu_trace']['published_operations']==5,'FIXTURE_REUSE_TRACE','Five actual final GPU publications required')
                        else:
                            gc.require(not any(row.get('event') in ('submitted','published','completed') for row in rows),
                                'FIXTURE_ORACLE','Ordinary CPU must complete without GPU work')
                        oracle=pair['processes'][0]
                        gc.require(all(oracle['prepared'][key]==prepared[key] for key in ('weight_sha256','input_sha256')),
                            'FIXTURE_INPUT_IDENTITY','Every ABBA process must reproduce the first CPU quantized inputs')
                        comparison=compare_raw(Path(oracle['output_root'])/'output.f32',root/'output.f32',case['M']*case['N'])
                        gc.require(comparison['reference_sha256']==oracle['result']['output_sha256'] and
                            comparison['candidate_sha256']==job['result']['output_sha256'],
                            'FIXTURE_RAW_IDENTITY','ABBA raw checksums differ from actual native serialization')
                        gc.require(comparison_passed(on,comparison),
                            'FIXTURE_NUMERIC_TOLERANCE' if on else 'FIXTURE_CPU_ORACLE_DRIFT',
                            'Original Metal criterion or final CPU bitexact oracle criterion failed')
                        job['comparison_to_first_cpu']=comparison
                        pair['comparisons_to_first_cpu'].append({'order_index':order_index,'abba_label':job['abba_label'],
                            'bridge':on,'reference_order_index':0,**comparison})
                        pair['expectation_met']=len(pair['comparisons_to_first_cpu'])==4
                        write(output/'receipt.json',record)
            gc.require(all(pair.get('expectation_met') for pair in record['results']) and len(record['results'])==len(record['cases']),
                'FIXTURE_INCOMPLETE','Every admitted case must meet its separate expectation')
            record['status']='passed_component_contract_only'
            numerical=[row for pair in record['results'] for row in pair['comparisons_to_first_cpu']]
            gc.require(len(numerical)==8,'FIXTURE_INCOMPLETE','Exactly four comparisons per each of two shapes required')
            record['comparisons_count']=len(numerical)
            record['cpu_bitexact_contract_met']=all(row['cpu_bitexact'] for row in numerical)
            record['original_metal_tolerance_passed']=all(row['metal_tolerance_passed'] for row in numerical)
            peaks=[job.get('gpu_trace',{}).get('device_allocated_observed_peak_bytes') for pair in record['results'] for job in pair['processes']]
            peaks=[value for value in peaks if type(value) is int]
            record['device_allocated_observed_peak_bytes']=max(peaks) if peaks else None
        except BaseException as error:
            record.update(status='cancelled' if isinstance(error,(KeyboardInterrupt,n.qa.ProbeInterrupted)) else 'failed_or_refused',
                error_code=getattr(error,'code',type(error).__name__),error=str(error))
        finally:
            for sig in handlers:signal.signal(sig,signal.SIG_IGN)
            def cleanup_error(phase,error):
                record.setdefault('cleanup_errors',[]).append({'phase':phase,'error':str(error)})
                if record['status']=='passed_component_contract_only':record.update(status='failed_or_refused',error_code='FINAL_EVIDENCE_REJECTED')
            try:
                if child is not None and child.poll() is None:record['last_cleanup']=n.stop_owned(child,binary,identity)
            except Exception as error:cleanup_error('owned_child',error)
            try:
                if holder is not None:holder.close();record['uid_parent_fd_closed']=True
            except Exception as error:cleanup_error('parent_uid_fd',error)
            try:gate.sample(final=True);record['power_gate']=gate.summary()
            except Exception as error:cleanup_error('power_finalize',error)
            try:intact=unchanged() and all(sha(path)==expected for path,expected in tracked.items())
            except OSError:intact=False
            record.update(measured_elapsed_seconds=time.monotonic()-started,observed_peak_child_rss_bytes=max_child,
                observed_peak_controller_rss_bytes=max_controller,finished_at_utc=datetime.now(timezone.utc).isoformat(),
                source_identity_unchanged=intact)
            peaks=[job.get('gpu_trace',{}).get('device_allocated_observed_peak_bytes') for pair in record['results'] for job in pair['processes']]
            peaks=[value for value in peaks if type(value) is int]
            record['device_allocated_observed_peak_bytes']=max(peaks) if peaks else None
            if record['status']=='passed_component_contract_only' and (not intact or record.get('power_gate',{}).get('passed') is not True):
                record.update(status='failed_or_refused',error_code='FINAL_EVIDENCE_REJECTED')
            write(output/'receipt.json',record)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan-output',type=Path,default=HERE/'plan.json')
    parser.add_argument('--binary',type=Path,default=BRIDGE/'build-run1/provider-fixture-warm-v2')
    parser.add_argument('--package-build',type=Path,default=BRIDGE/'build-run1')
    parser.add_argument('--fixture-build',type=Path,default=FROZEN/'build-receipt.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:
        if args.plan_output.exists():raise FileExistsError(args.plan_output)
        write(args.plan_output,plan());print(json.dumps({'status':'prepared_not_executed','plan':str(args.plan_output)}));return 0
    if args.output is None:parser.error('Fresh --output required for explicit coordinated execution')
    preexisting=args.output.exists()
    try:report=execute(args)
    except Exception as error:
        report={'schema_version':1,'status':'failed_or_refused','error_code':getattr(error,'code',type(error).__name__),
            'error':str(error),'model_loaded':False,'build_performed':False,'gpu_commands':None,
            'scope':'preflight/controller exception; no successful ABBA comparison claimed'}
        if not preexisting and args.output.is_dir() and not (args.output/'receipt.json').exists():
            write(args.output/'receipt.json',report)
    print(json.dumps({'status':report['status'],'receipt':str(args.output/'receipt.json'),'model_loaded':False}))
    return 0 if report['status']=='passed_component_contract_only' else 2

if __name__=='__main__':raise SystemExit(main())
