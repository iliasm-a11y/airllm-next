"""Plan/compare isolated provider graphs. Default never invokes native/GPU/build.

An explicitly admitted execution runs bridgeOFF then bridgeON in separate
processes of the same package. No GGUF, model or snapshot is ever loaded.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
sys.path.insert(0,str(BRIDGE))
import gpu_contract as gc
ATOL,RTOL=1e-3,1e-4
DATA_LIMIT=64<<20
RSS_LIMIT=128<<20
GLOBAL_SECONDS=240
CHILD_SECONDS=20
POLL_SECONDS=0.1

def sha(path):return gc.sha(path)
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(path)

def cases():
    result=[]
    def add(m,k,n,mode='normal',expect='gpu'):
        result.append({'id':f'm{m}-k{k}-n{n}-{mode}','M':m,'K':k,'N':n,'mode':mode,'expectation':expect})
    # Increasing token sizes for each large projection; both K values, six row tiles.
    for k in (2048,4096):
        for n in (8,16,32,64,128):add(8192,k,n)
    for n in (8,32,128):add(1536,4096,n)
    add(2048,2048,32)
    for k in (2048,4096):
        for n in (1,6,14):add(1536,k,n,expect='cpu_fallback')
    add(1536,4096,8,'subnormal-x','cpu_decline')
    add(1536,2048,8,'subnormal-w','cpu_decline')
    add(2048,4096,8,'late-subnormal-w','fatal_after_terminal_tile')
    return result

def logical_bound(case):
    m,k,n=(case[key] for key in ('M','K','N'))
    q=n*(k//32)*34;provider=((64+q+63)&~63)+m*n*4
    # Pinned ggml_graph_plan adds cache-line64 per CPU worker; no tiled Q8_0
    # workspace exists on ARM in this exact source. Alignment of input is64.
    graph=provider+64*10 if n in (8,16,32,64,128) else q+64*10
    owned=((k*n*4+63)&~63)+m*n*4
    return m*(k//32)*34+owned+(1<<20)+k*4+graph+gc.GPU_BYTES

def plan():
    suite=cases();bounds=[logical_bound(case) for case in suite]
    assert len(suite)==23 and max(bounds)<=DATA_LIMIT
    return {'schema_version':1,'status':'prepared_not_compiled_not_executed','cases':suite,
        'oracle':'ordinary CPU10 graph dispatcher, bridgeOFF, separate process, same new native package',
        'candidate':'bridgeON; CPU_Mapped + WEIGHTS; original native CPU Q8_0 row quantization',
        'processes_max':2*len(suite),'threads_per_process':10,'max_weight_bytes':8192*(4096//32)*34,
        'max_output_bytes_per_process':8192*128*4,'max_logical_fixture_payload_bytes':max(bounds),
        'logical_fixture_limit_bytes':DATA_LIMIT,'fixed_gpu_payload_bytes':gc.GPU_BYTES,
        'metal_device_allocation_ceiling_bytes':DATA_LIMIT,'child_rss_ceiling_bytes':RSS_LIMIT,
        'controller_rss_ceiling_bytes':RSS_LIMIT,'controller_additional_reservation_bytes':64<<20,
        'global_deadline_seconds':GLOBAL_SECONDS,'child_deadline_seconds':CHILD_SECONDS,
        'sampling_interval_seconds':POLL_SECONDS,'planned_raw_output_bytes':sum(c['M']*c['N']*4*(1 if c['expectation']=='fatal_after_terminal_tile' else 2) for c in suite),
        'bitexact_cpu_contract':'all float32 bits compared separately, including signed zero',
        'metal_tolerance':{'atol':ATOL,'rtol':RTOL,'formula':'abs(candidate-reference)<=atol+rtol*abs(reference)'},
        'use_ref':False,'full_model_qualified':False,'model_loaded':False,
        'measured_elapsed_seconds':None,'observed_peak_child_rss_bytes':None,'observed_peak_controller_rss_bytes':None,
        'device_allocated_observed_peak_bytes':None,'unmeasured_driver_residency_bytes':None,
        'physical_ssd_reads':None,'pending_timeout_injection_tested':False,'terminal_command_error_injection_tested':False,
        'scope':'finite synthetic dense projections and bounded deliberate late-scale failure; no routing/logits/KDA/EOS/GLM performance'}

def compare_raw(reference,candidate,count):
    reference,candidate=Path(reference),Path(candidate)
    gc.require(sys.byteorder=='little' and count>0 and count*4<=4<<20,'FIXTURE_RAW','Bounded little-endian float32 outputs required')
    gc.require(reference.stat().st_size==candidate.stat().st_size==count*4,'FIXTURE_RAW','Raw output length differs')
    hashes=[hashlib.sha256(),hashlib.sha256()];bits_differ=tolerance_failed=0;max_abs=max_relative=0.0;seen=0
    identities=[(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns) for path in (reference,candidate)]
    with reference.open('rb') as a,candidate.open('rb') as b:
        while True:
            aa=a.read(64<<10);bb=b.read(64<<10)
            if not aa and not bb:break
            gc.require(len(aa)==len(bb) and len(aa)%4==0,'FIXTURE_RAW','Truncated float32 block')
            hashes[0].update(aa);hashes[1].update(bb)
            av,bv=array('f'),array('f');av.frombytes(aa);bv.frombytes(bb)
            ai,bi=array('I'),array('I');ai.frombytes(aa);bi.frombytes(bb)
            for x,y,xb,yb in zip(av,bv,ai,bi):
                gc.require(math.isfinite(x) and math.isfinite(y),'FIXTURE_NONFINITE','Nonfinite raw output')
                difference=abs(y-x);max_abs=max(max_abs,difference)
                if x:max_relative=max(max_relative,difference/abs(x))
                bits_differ+=xb!=yb;tolerance_failed+=difference>ATOL+RTOL*abs(x);seen+=1
    gc.require(seen==count and identities==[(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns) for path in (reference,candidate)],
        'FIXTURE_RAW_IDENTITY','Raw output changed during comparison')
    return {'outputs':count,'all_finite':True,'cpu_bitexact':bits_differ==0,'different_float32_bits_count':bits_differ,
        'metal_tolerance_passed':tolerance_failed==0,'metal_tolerance_failures':tolerance_failed,'atol':ATOL,'rtol':RTOL,
        'max_abs_error':max_abs,'max_relative_error_nonzero_reference':max_relative,
        'reference_sha256':hashes[0].hexdigest(),'candidate_sha256':hashes[1].hexdigest()}

def trace_rows(text):
    gc.require(len(text.encode())<=32<<20,'GPU_TRACE_SIZE','Bounded native log required')
    return [dict(re.findall(r'([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)',line.split('AIRLLM_CPU_METAL_Q8 ',1)[1]))
        for line in text.splitlines() if 'AIRLLM_CPU_METAL_Q8 ' in line]

def zero_trace(text,document,baseline,allocated):
    rows=trace_rows(text)
    gc.require(not any(row.get('event') in ('submitted','completed','tile_done','published') for row in rows),'FIXTURE_UNEXPECTED_GPU','CPU case unexpectedly used GPU')
    enabled=[row for row in rows if row.get('enabled')=='1'];allocations=[row for row in rows if 'gpu_payload_peak' in row]
    gc.require(len(enabled)==1 and enabled[0].get('canonical_sha256')==document['canonical_sha256']
        and enabled[0].get('kernel_sha256')==document['canonical_identity']['fingerprints']['kernel_sha256'],
        'GPU_TRACE_IDENTITY','Enabled provider identity required for a candidate CPU fallback')
    gc.require(len(allocations)==int(allocated),'GPU_ALLOCATION_BOUND','Unexpected allocation for a declined/unsupported shape')
    if allocated:
        row=allocations[0]
        required={'allocation_count':4,'gpu_payload_current':gc.GPU_BYTES,'gpu_payload_peak':gc.GPU_BYTES,
            'weight_tile_bytes':gc.WEIGHT_TILE_BYTES,'native_swap_baseline_bytes':baseline,'device_allocated_limit':DATA_LIMIT}
        gc.require(all(row.get(key)==str(value) for key,value in required.items()),'GPU_ALLOCATION_BOUND','Decline allocation evidence differs')
        gc.require(all(re.fullmatch('[0-9]+',row.get(key,'')) for key in ('device_allocated_current','device_allocated_observed_peak')),
            'GPU_DEVICE_ALLOCATION_BOUND','Decline device measurement absent')
        gc.require(0<int(row['device_allocated_current'])<=int(row['device_allocated_observed_peak'])<=DATA_LIMIT,
            'GPU_DEVICE_ALLOCATION_BOUND','Decline device allocation exceeds bound')
        gc.require(any(row.get('declined_before_submit')=='1' for row in rows),'FIXTURE_DECLINE','Explicit native pre-submit decline required')
    return {'passed':True,'actual_gpu':False,'submitted_commands':0,'completed_commands':0,'allocation_observed':allocated}

def fatal_trace(text,document,baseline,case):
    rows=trace_rows(text)
    enabled=[row for row in rows if row.get('enabled')=='1'];allocations=[row for row in rows if 'gpu_payload_peak' in row]
    gc.require(len(enabled)==1 and enabled[0].get('backend')==gc.BACKEND
        and enabled[0].get('canonical_sha256')==document['canonical_sha256']
        and enabled[0].get('kernel_sha256')==document['canonical_identity']['fingerprints']['kernel_sha256'],
        'GPU_TRACE_IDENTITY','Expected fatal process requires its own canonical/kernel/backend trace')
    gc.require(len(allocations)==1,'GPU_ALLOCATION_BOUND','Expected fatal process requires one fixed allocation')
    allocation=allocations[0]
    expected={'allocation_count':4,'gpu_payload_current':gc.GPU_BYTES,'gpu_payload_peak':gc.GPU_BYTES,
        'weight_tile_bytes':gc.WEIGHT_TILE_BYTES,'native_swap_baseline_bytes':baseline,'device_allocated_limit':DATA_LIMIT}
    gc.require(all(allocation.get(key)==str(value) for key,value in expected.items()),'GPU_ALLOCATION_BOUND','Expected fatal allocation differs')
    def integer(row,key):
        gc.require(re.fullmatch('[0-9]+',row.get(key,'')) is not None,'GPU_TRACE_COUNTER','Exact counter required: '+key)
        return int(row[key])
    current=integer(allocation,'device_allocated_current');peak=integer(allocation,'device_allocated_observed_peak')
    gc.require(0<current<=peak<=DATA_LIMIT,'GPU_DEVICE_ALLOCATION_BOUND','Expected fatal device allocation missing/unbounded')
    submitted=[row for row in rows if row.get('event')=='submitted'];completed=[row for row in rows if row.get('event')=='completed']
    tiles=[row for row in rows if row.get('event')=='tile_done']
    gc.require(len(submitted)==len(completed)==len(tiles)==1 and not any(row.get('event')=='published' for row in rows),
        'GPU_TRACE_INCOMPLETE','One exact terminal tile prefix and no publication required')
    submission,completion,tile=submitted[0],completed[0],tiles[0]
    gc.require(rows.index(enabled[0])<rows.index(allocation)<rows.index(submission)<rows.index(completion)<rows.index(tile),
        'GPU_TRACE_INCOMPLETE','Fatal trace prefix ordering differs')
    shape={'weight':'fixture.q8.weight','M':case['M'],'K':case['K'],'N':case['N'],'tile_first':0,'tile_rows':1536}
    gc.require(all(submission.get(key)==tile.get(key)==str(value) for key,value in shape.items())
        and all(row.get('actual_gpu')=='1' for row in (submission,completion,tile))
        and submission.get('submission')==completion.get('submission')==completion.get('completed_commands')=='1'
        and completion.get('terminal_success')=='1','GPU_TRACE_SHAPE','Fatal completed command does not match the first actual tile')
    for key,value in {'attempt':1,'submissions':1,'completed_commands':1,'allocation_count':4,
            'operation_buffer_reuses':0,'tile_buffer_reuses':0,'gpu_payload_current':gc.GPU_BYTES,
            'copy_weight_bytes':1536*(case['K']//32)*34}.items():
        gc.require(integer(tile,key)==value,'GPU_TRACE_COUNTER','Fatal tile counter/copy differs')
    sample={'pressure_level':integer(submission,'pressure'),'available_bytes':integer(submission,'available_bytes'),
        'rss_bytes':integer(submission,'rss_bytes'),'swap_growth_bytes':integer(submission,'swap_growth_bytes')}
    gc.require(gc.sufficient_memory_sample(sample,baseline),'GPU_RESOURCE_ADMISSION','Fatal tile memory admission differs')
    for row,keys in ((completion,('gpu_seconds',)),(tile,('gpu_seconds','command_and_readback_us','copy_weight_us'))):
        for key in keys:
            try:value=float(row[key])
            except (KeyError,ValueError,TypeError) as error:raise gc.GpuContractError('GPU_TRACE_NUMBER','Finite fatal tile timing required') from error
            gc.require(math.isfinite(value) and value>=0,'GPU_TRACE_NUMBER','Finite fatal tile timing required')
    gc.require(completion['gpu_seconds']==tile['gpu_seconds'],'GPU_TRACE_INCOMPLETE','Fatal tile timing does not match completion')
    return {'passed':True,'actual_gpu':True,'submitted_commands':1,'completed_commands':1,'published_operations':0,
        'fixed_gpu_bytes':gc.GPU_BYTES,'device_allocated_observed_peak_bytes':peak,
        'scope':'strict terminal first-tile prefix before deliberate fatal; not a completed operation'}


def validate_prepared(row,case,on,document,libraries):
    gc.require(type(row) is dict and all(type(row.get(key)) is int and row.get(key)==case[key] for key in ('M','K','N')) and row.get('case')==case['mode']
        and type(row.get('threads')) is int and row.get('threads')==10 and row.get('uses_reference_mode') is False and row.get('bridge_requested') is on
        and row.get('cpu_metal_q8_bridge_compiled') is True,'FIXTURE_DESCRIPTOR','Actual fixture shape/backend descriptor differs')
    gc.require(row.get('weight_buffer_type')=='CPU_Mapped' and type(row.get('weight_buffer_usage')) is int and row.get('weight_buffer_usage')==1
        and row.get('weight_original_plain_layout') is True,'FIXTURE_MAPPED_WEIGHTS','Actual CPU_Mapped WEIGHTS buffer required')
    gc.require(type(row.get('logical_payload_bound_bytes')) is int and row['logical_payload_bound_bytes']<=logical_bound(case)<=DATA_LIMIT,
        'FIXTURE_ALLOCATION_BOUND','Actual planned fixture allocation exceeds source-derived bound')
    gc.validate_loaded_images(row.get('native_linkage'),document,libraries)
    for key in ('weight_sha256','input_sha256'):
        gc.require(type(row.get(key)) is str and re.fullmatch('[0-9a-f]{64}',row[key]),'FIXTURE_INPUT_IDENTITY','Actual native input hashes required')


def execute(args):
    # Delayed imports: plan/comparison tests never touch native/OS control paths.
    sys.path.insert(0,str(BRIDGE));import native_gpu_run as n
    overall_started=time.monotonic()
    binary=Path(args.binary).resolve();package=Path(args.package_build).resolve();output=Path(args.output).resolve()
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
        and Path(roles['fixture_source']).resolve()==HERE/'provider_fixture.cpp'
        and Path(roles['controller_source']).resolve()==Path(__file__).resolve(),'FIXTURE_BUILD','Exact fixture build/source/package receipt required')
    tracked={path:sha(path) for path in {str(binary),str(Path(__file__).resolve()),str(HERE/'provider_fixture.cpp'),str(Path(args.fixture_build).resolve()),
        str(package/'provider-build.json'),str(package/'driver-build.json'),str(package/'gpu-profile.json'),str(BRIDGE/'gpu_contract.py'),str(n.NATIVE_GUARD_SOURCE)}}
    record.update(profile_sha256=document['canonical_sha256'],binary_sha256=sha(binary),tracked_inputs_sha256=tracked,
        fixture_build_receipt_sha256=sha(args.fixture_build),native_libraries_sha256=provider['native_libraries_sha256'])
    tracked.update({str(path.resolve()):sha(path) for path in (package/'runtime/bin').glob('*.dylib') if not path.is_symlink()})
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
                    pair={'case':case,'processes':[]};record['results'].append(pair)
                    for on in (False,True):
                        check(n.resource_snapshot(),startup=True)
                        root=output/case['id']/('bridge-on' if on else 'bridge-off');root.mkdir(parents=True,mode=0o700)
                        env,_=n.clean_environment({});env.update(document['environment'] if on else {'AIRLLM_CPU_METAL_Q8_BRIDGE':'0'})
                        env['AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES']=str(baseline['swap_used_bytes'])
                        read_fd,write_fd=os.pipe();env['AIRLLM_FIXTURE_START_FD']=str(read_fd)
                        command=[str(binary),str(root),str(case['M']),str(case['K']),str(case['N']),case['mode'],'on' if on else 'off']
                        job={'bridge':on,'command':command,'status':'starting','elapsed_seconds':None,'returncode':None,'actual_gpu':False}
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
                                    sample=check(current);sample.update(case=case['id'],bridge=on,elapsed_seconds=time.monotonic()-started)
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
                        if not on:
                            gc.require(job['returncode']==0 and not any(row.get('event') in ('submitted','published','completed') for row in rows),
                                'FIXTURE_ORACLE','Ordinary CPU oracle must complete without GPU work')
                            job['result']=gc.read_json(root/'result.json')
                        elif case['expectation']=='fatal_after_terminal_tile':
                            fatal=gc.read_json(root/'fatal.json');job.update(status='expected_controlled_fatal',fatal=fatal)
                            job['gpu_trace']=fatal_trace(text,document,baseline['swap_used_bytes'],case)
                            oracle_path=output/case['id']/'bridge-off/output.f32'
                            oracle_check=compare_raw(oracle_path,oracle_path,case['M']*case['N'])
                            gc.require(oracle_check['reference_sha256']==pair['processes'][0]['result']['output_sha256'],
                                'FIXTURE_RAW_IDENTITY','Fatal-case ordinary CPU oracle checksum differs')
                            pair['ordinary_cpu_oracle_validation']=oracle_check
                            gc.require(job['returncode']==86 and submissions==1 and sum(row.get('event')=='completed' and row.get('terminal_success')=='1' for row in rows)==1
                                and not any(row.get('event')=='published' for row in rows) and not (root/'output.f32').exists()
                                and fatal.get('destination_unpublished') is True and fatal.get('cpu_fallback_performed') is False
                                and fatal.get('destination_bytes_checked')==case['M']*case['N']*4,
                                'FIXTURE_LATE_FATAL','Deliberate late refusal must stop after one terminal tile without destination publication')
                            pair.update(expectation_met=True,numerical_comparison=None)
                        else:
                            gc.require(job['returncode']==0,'NATIVE_NONZERO_EXIT','Unexpected native fixture failure')
                            job['result']=gc.read_json(root/'result.json')
                            if case['expectation']=='gpu':job['gpu_trace']=gc.parse_trace(text,document,baseline_swap=baseline['swap_used_bytes'])
                            else:job['gpu_trace']=zero_trace(text,document,baseline['swap_used_bytes'],case['expectation']=='cpu_decline')
                            oracle=pair['processes'][0]
                            gc.require(all(oracle['prepared'][key]==prepared[key] for key in ('weight_sha256','input_sha256')),
                                'FIXTURE_INPUT_IDENTITY','Separate native processes produced different actual quantized inputs')
                            compare=compare_raw(output/case['id']/'bridge-off/output.f32',root/'output.f32',case['M']*case['N'])
                            gc.require(compare['reference_sha256']==oracle['result']['output_sha256'] and compare['candidate_sha256']==job['result']['output_sha256'],
                                'FIXTURE_RAW_IDENTITY','Raw checksums differ from native serialization')
                            gc.require(compare['metal_tolerance_passed'],'FIXTURE_NUMERIC_TOLERANCE','Original Metal tolerance failed')
                            if case['expectation']!='gpu':gc.require(compare['cpu_bitexact'],'FIXTURE_CPU_FALLBACK_BITS','CPU fallback differs from normal same-package oracle')
                            pair.update(expectation_met=True,numerical_comparison=compare)
                        write(output/'receipt.json',record)
            gc.require(all(pair.get('expectation_met') for pair in record['results']) and len(record['results'])==len(record['cases']),
                'FIXTURE_INCOMPLETE','Every admitted case must meet its separate expectation')
            record['status']='passed_component_contract_only'
            numerical=[pair['numerical_comparison'] for pair in record['results'] if pair.get('numerical_comparison')]
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-output',type=Path,default=HERE/'fixture-plan.json')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--binary',type=Path);parser.add_argument('--package-build',type=Path)
    parser.add_argument('--fixture-build',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:
        if args.plan_output.exists():raise FileExistsError(args.plan_output)
        write(args.plan_output,plan());print(json.dumps({'status':'prepared_not_executed','cases':len(cases()),'plan':str(args.plan_output)}));return 0
    if any(getattr(args,key) is None for key in ('binary','package_build','fixture_build','output')):parser.error('Explicit binary/package/fixture receipt/new output required')
    try:record=execute(args)
    except Exception as error:
        record=plan();record.update(status='failed_or_refused',error_code='ENGINE_BUSY' if isinstance(error,BlockingIOError) else getattr(error,'code',type(error).__name__),error=str(error))
        if args.output.is_dir() and not (args.output/'receipt.json').exists():write(args.output/'receipt.json',record)
    print(json.dumps({'status':record['status'],'receipt':str(args.output/'receipt.json'),'model_loaded':False}))
    return 0 if record['status']=='passed_component_contract_only' else 2

if __name__=='__main__':raise SystemExit(main())
