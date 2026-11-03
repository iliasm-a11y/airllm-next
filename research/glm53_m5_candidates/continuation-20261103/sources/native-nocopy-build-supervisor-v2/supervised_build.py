"""SOURCE ONLY v2: one dedicated strict supervisor, no nested build locks.

Default only writes a plan. --execute is for a future reviewed zero-model window:
one C++ compile, signing and registry/parameter smokes (legacy Metal backend
initialization can use empty commands). No provider/library build and no model.
No output is ever written beneath the old NoCopy directory.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
BASE=WORK/'native-noblas-nocopy-candidate'
EXISTING=WORK/'metal-tensorops-q8-nocopy'
PACKAGE=EXISTING/'build-run1'
PROVIDER=PACKAGE/'provider-build.json'
OLD_DRIVER=PACKAGE/'driver-build.json'
PROVIDER_SHA='4fe68689c17b1c335f7b2fab2087aa99cf179d1dbe760d2a810d9cec1b01cb63'
OLD_DRIVER_SHA='a861e92a8e14a94e674c3cc9a4cb02401d6c0f73887c02066c3b068e135014ab'
HEADER_NAMES=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
HELPER=BASE/'driver/research/glm53_batched_prefill/native_prefill.cpp'
CPU_V4=WORK/'native-noblas208-v4-candidate'
GUARD=WORK/'airllm-next/air_llm/airllm/nextgen/glm_native.py'
sys.path.insert(0,str(HERE))
def exact_sibling(name):
    """Refuse another cached path, then load the exact current sibling bytes."""
    path=(BASE/(name+".py")).resolve()
    cached=sys.modules.get(name)
    if cached is not None and Path(getattr(cached,"__file__","")).resolve()!=path:
        raise ImportError("foreign cached module refused before native/build work: "+name)
    prepared=json.loads((HERE/'source-manifest.json').read_text())
    for candidate in (path,BASE/'profile.py'):
        expected=prepared['source_pins'].get(str(candidate.resolve()))
        if expected!=hashlib.sha256(candidate.read_bytes()).hexdigest():raise ImportError('unreviewed sibling source')
    payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
    tag="airllm_noblas_nocopy_"+name+"_"+digest
    cached=sys.modules.get(tag)
    if cached is not None:
        if Path(getattr(cached,"__file__","")).resolve()!=path:raise ImportError("foreign hashed module path")
        return cached
    spec=importlib.util.spec_from_file_location(tag,path)
    module=importlib.util.module_from_spec(spec);sys.modules[tag]=module
    try:exec(compile(payload,str(path),"exec"),module.__dict__)
    except BaseException:
        sys.modules.pop(tag,None);raise
    if path.read_bytes()!=payload:raise ImportError("module source changed while loading: "+name)
    return module

gc=exact_sibling("gpu_contract")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def hashes(paths):return {str(p):sha(p) for p in paths}

def unchanged(expected):
    if hashes(map(Path,expected))!=expected:raise ValueError('Tracked source/receipt/library changed')

def clean_env():
    return {k:v for k,v in os.environ.items() if not k.startswith(('AIRLLM_','GGML_','DYLD_','LLAMA_ARG_','RONIN_PREFILL_'))
        and k not in ('LLAMA_API_KEY','CFLAGS','CXXFLAGS','OBJCXXFLAGS','LDFLAGS','CC','CXX','OBJCXX')}

def plan(destination):
    destination=Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(HERE):raise ValueError('Fresh destination inside this candidate required')
    if sha(PROVIDER)!=PROVIDER_SHA or sha(OLD_DRIVER)!=OLD_DRIVER_SHA:raise ValueError('Exact closed provider/driver receipts required')
    provider=gc.read_json(PROVIDER);old=gc.read_json(OLD_DRIVER)
    if provider.get('status')!='built' or provider.get('compiled_feature_query_verified') is not True or provider.get('CPU_METAL_Q8_BRIDGE_COMPILED') is not True:
        raise ValueError('Existing actual compiled provider receipt required')
    required=('source_manifest','bridge_patch','reconstruction_receipt','v4_build_manifest','build_script','kernel_standalone',
        *('overlay:'+name for name in gc.OVERLAY_FILES),*('native_overlay:'+name for name in gc.OVERLAY_FILES),
        *('header:'+name for name in HEADER_NAMES),'cmake:CMakeLists.txt','cmake:ggml/CMakeLists.txt')
    proles=gc.validate_input_provenance(provider,required)
    libs=PACKAGE/'runtime/bin'
    actual={name:sha(libs/('lib'+name+'.dylib')) for name in provider['native_libraries_sha256']}
    if actual!=provider['native_libraries_sha256'] or actual!=old.get('native_libraries_sha256'):
        raise ValueError('Existing exact NoCopy libraries required')
    roles={'helper_source':HELPER,'snapshot_source':HELPER.with_name('snapshot.h'),
        'state_capture_source':BASE/'driver/research/glm53_gpu_v3/state_capture.inc',
        'build_script':Path(__file__),'profile_source':BASE/'profile.py','contract_source':BASE/'gpu_contract.py',
        'runner_source':BASE/'native_gpu_run.py','provider_receipt':PROVIDER,
        'cpu_v4_source':CPU_V4/'research/glm53_batched_prefill/native_prefill.cpp','cpu_v4_prepare_manifest':CPU_V4/'prepare-manifest.json',
        'guard_source':GUARD,'supervisor_source_manifest':HERE/'source-manifest.json',
        'original_planner_source':BASE/'build_driver.py','base_candidate_manifest':BASE/'prepare-manifest.json','runtime_manifest':PACKAGE/'runtime/manifest.json',
        **{'header:'+name:Path(proles['header:'+name]) for name in HEADER_NAMES}}
    roles={name:str(Path(path).resolve()) for name,path in roles.items()}
    cpu_prepared=gc.read_json(roles['cpu_v4_prepare_manifest'])
    if cpu_prepared.get('policy')!='cpu8-tail-global-fences208-no-blas-v4' or cpu_prepared.get('variant_helper_source_sha256')!=sha(roles['cpu_v4_source']):raise ValueError('Exact v4 geometry provenance required')
    fingerprints={'CPU_METAL_Q8_BRIDGE_COMPILED':True,'native_libraries_sha256':actual,
        'provider_patch_sha256':provider['provider_patch_sha256'],'kernel_sha256':provider['kernel_sha256'],
        'native_guard_policy_sha256':sha(GUARD),'provider_build_receipt_sha256':sha(PROVIDER),
        'cpu_geometry_source_sha256':sha(roles['cpu_v4_source']),'cpu_geometry_manifest_sha256':sha(roles['cpu_v4_prepare_manifest'])}
    for role in ('helper_source','snapshot_source','state_capture_source','runner_source','profile_source','contract_source'):
        fingerprints[role+'_sha256']=sha(roles[role])
    document=gc.profiles.profile(fingerprints)
    # Reuse the compiler, SDK, flags, includes, libraries and rpath from the actual
    # old driver build. Only the translation unit and output filename change.
    argv=old['command'][:]
    argv[argv.index(str(EXISTING/'driver/research/glm53_batched_prefill/native_prefill.cpp'))]=str(HELPER)
    argv[argv.index('-o')+1]=str(destination/'native-prefill-noblas-nocopy')
    tracked=hashes(Path(p) for p in roles.values())
    tracked.update(hashes((libs/('lib'+name+'.dylib')).resolve() for name in actual))
    closed=hashes((PROVIDER,OLD_DRIVER,PACKAGE/'gpu-profile.json',EXISTING/'source-manifest.json',EXISTING/'native_gpu_run.py',EXISTING/'gpu_contract.py',EXISTING/'profile.py'))
    return {'schema_version':2,'status':'planned_not_built','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'destination':str(destination),'binary':str(destination/'native-prefill-noblas-nocopy'),'command':argv,
        'input_roles':roles,'inputs_sha256':tracked,'closed_package_inputs_sha256':closed,
        'native_libraries_sha256':actual,'provider_build_receipt_sha256':sha(PROVIDER),'provider_rebuild':False,
        'prefill_chunk_policy':gc.TAIL_POLICY,'cpu_accelerator_policy':gc.CPU_ACCELERATOR_POLICY,
        'prefill_fences':gc.FENCES,'diagnostic_scope':gc.DIAGNOSTIC_SCOPE,'profile_document':document,
        'compile_timeout_seconds':120,'metadata_timeout_seconds_each':15,'metadata_smoke_count':6,
        'build_attempted':False,'binary_sha256':None,'parameter_smoke':{},'model_loaded':False,
        'actual_gpu_matrix_commands':None,'full_campaign_qualified':False,'numerical_fidelity_qualified':False,
        'scope':'fresh driver, dedicated supervisor owns benchmark+UID; no inherited qualification'}

# No caller-provided "already guarded" flag or environment bypass exists. This
# new coordinator owns both locks itself and must be launched directly, rather
# than nested beneath guarded_build_once.py. The old coordinator remains intact.
def policy_modules():
    import importlib
    from types import SimpleNamespace
    import psutil
    repository=WORK/'airllm-next'
    sys.path[:0]=[str(repository),str(repository/'air_llm')]
    for name,path in (('airllm.nextgen.glm_native',repository/'air_llm/airllm/nextgen/glm_native.py'),
            ('research.glm53_shared_prefill.qualify_agents',repository/'research/glm53_shared_prefill/qualify_agents.py'),
            ('research.glm53_gpu_v3.power_state',repository/'research/glm53_gpu_v3/power_state.py')):
        cached=sys.modules.get(name)
        if cached is not None and Path(getattr(cached,'__file__','')).resolve()!=path.resolve():raise ValueError('foreign cached policy before import')
    native=importlib.import_module('airllm.nextgen.glm_native')
    qa=importlib.import_module('research.glm53_shared_prefill.qualify_agents')
    power=importlib.import_module('research.glm53_gpu_v3.power_state')
    for module,path in ((native,repository/'air_llm/airllm/nextgen/glm_native.py'),
            (qa,repository/'research/glm53_shared_prefill/qualify_agents.py'),
            (power,repository/'research/glm53_gpu_v3/power_state.py')):
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('foreign cached policy module')
    return SimpleNamespace(native=native,qa=qa,power=power,psutil=psutil)

def verify_source_manifest():
    document=json.loads((HERE/'source-manifest.json').read_text())
    if document.get('status')!='source_prepared_not_executed':raise ValueError('source preparation required')
    for path,digest in document['source_pins'].items():
        if sha(path)!=digest:raise ValueError('source pin changed: '+path)
    if document['source_pins'].get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('supervisor source pin missing')
    return document

def kernel_lock(holder,path):
    value=os.fstat(holder.fileno())
    return {'path':str(Path(path).resolve()),'fd':holder.fileno(),'device':value.st_dev,'inode':value.st_ino,'uid':os.getuid()}

def observe_fds(process,locks,psutil):
    identity={'pid':process.pid,'create_time':process.create_time(),'executable':process.exe()}
    files=process.open_files();answer={}
    for name,row in locks.items():
        matches=[]
        for item in files:
            if item.fd!=row['fd']:continue
            try:
                stat=Path(item.path).stat()
                if (stat.st_dev,stat.st_ino)==(row['device'],row['inode']) and Path(item.path).resolve()==Path(row['path']):
                    matches.append({'fd':item.fd,'path':str(Path(item.path).resolve()),'device':stat.st_dev,'inode':stat.st_ino})
            except OSError:pass
        if len(matches)!=1:raise ValueError('actual inherited kernel FD not observed: '+name)
        answer[name]=matches[0]
    if not process.is_running()or process.create_time()!=identity['create_time']or process.exe()!=identity['executable']:
        raise ValueError('owned FD observation changed PID lifetime/executable')
    return {'identity':identity,'locks':answer,'scope':'actual owned PID/FD/path/device/inode, bracketed birth/executable; not an environment claim'}

def execute(record):
    import select
    import signal
    import tempfile
    source_manifest=verify_source_manifest();modules=policy_modules()
    native,psutil,power=modules.native,modules.psutil,modules.power
    destination=Path(record['destination']);destination.mkdir(parents=True,exist_ok=False)
    guard_path=destination/'build-guard.json';driver_path=destination/'driver-build.json'
    limits=native.NativeLimits(**json.loads((WORK/'cpu10-reference.json').read_text())['limits'])
    if limits.max_rss_bytes!=int(8.5*(1<<30))or limits.max_swap_growth_bytes!=32<<20:raise ValueError('unchanged approved hard caps required')
    checker=object.__new__(native.NativeGlmModel);checker.limits=limits
    checker._warning_started=checker._headroom_low_started=checker._pressure_context=None
    gate=power.PowerGate('ac');started=time.monotonic();benchmark=uid=None;locks={};children=[]
    guard={'schema_version':1,'status':'guard_pending','source_sha256':sha(__file__),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'native_limits':vars(limits),'samples':[],
        'tree_rss_cap_bytes':2<<30,'global_deadline_seconds':300,'guard_violation':None,'stages':[],
        'model_loaded':False,'bridge_matrix_commands_observed':None,'legacy_metadata_device_use_possible':True,
        'scope':'dedicated build supervisor only; original resource/power policy plus original conservative2GiB build tree cap'}
    record.update(status='guard_pending',inputs_before=hashes(map(Path,record['inputs_sha256'])),build_attempted=False)
    write(driver_path,record);write(guard_path,guard)
    baseline=None;success=False;failure=None;handlers={}
    def interrupted(signum,frame):raise KeyboardInterrupt('owned build cancelled by signal'+str(signum))
    def still_owned(row):
        try:
            p=psutil.Process(row['pid'])
            return p.is_running()and p.create_time()==row['create_time']and p.exe()==row['executable']
        except psutil.NoSuchProcess:return False
    def identify(p):return {'pid':p.pid,'create_time':p.create_time(),'executable':p.exe()}
    def check(pid=None,tree_rss=None):
        sample=native.resource_snapshot(os.getpid() if pid is None else pid)
        if tree_rss is not None:sample['rss_bytes']=tree_rss
        for name in ('rss_bytes','available_bytes','swap_used_bytes','pressure_level'):
            if name=='rss_bytes'and pid is None and name not in sample:sample[name]=0
            if type(sample.get(name))is not int or sample[name]<0:raise ValueError('finite integer resource metadata required: '+name)
        sample['swap_growth_bytes']=max(0,sample['swap_used_bytes']-baseline['swap_used_bytes'])
        sample['elapsed_seconds']=time.monotonic()-started
        sample['guard_monotonic']=time.monotonic()
        code=native.NativeGlmModel._guard_resource_sample(checker,sample,baseline,sample['guard_monotonic'])
        if sample['rss_bytes']>guard['tree_rss_cap_bytes']:code=code or'BUILD_TREE_RSS_LIMIT'
        if sample['elapsed_seconds']>300:code=code or'BUILD_DEADLINE'
        if not gate.sample()['passed']:code='POWER_CONTEXT_CHANGED'
        guard['samples'].append(sample)
        if code:
            guard['guard_violation']={'code':code,'sample':sample,'pressure_context':checker._pressure_context}
            raise ValueError(code)
        return sample
    def phase(label,argv,environment,timeout):
        # A trusted bootstrap blocks before exec. Both capabilities are inherited
        # and independently observed before GO, including compiler and metadata.
        bootstrap=('import os,sys; g=int(sys.argv[1]); r=int(sys.argv[2]); '
            'os.write(r,b"R"); os.close(r); token=os.read(g,1); os.close(g); '
            'assert token==b"G", "phase GO refused"; target=sys.argv[3:]; '
            'os.execve(target[0],target,os.environ)')
        go_r,go_w=os.pipe();ready_r,ready_w=os.pipe();child=None;awake=None;owned={};phase_start=time.monotonic()
        row={'label':label,'command':argv,'status':'starting','timeout_seconds':timeout,'signals':[]}
        guard['stages'].append(row);log=destination/(label+'.log')
        def remember(p):
            item=identify(p);old=owned.get(p.pid)
            if old and old['create_time']!=item['create_time']:raise ValueError('owned phase PID reused')
            if old and old['executable']!=item['executable']:row.setdefault('owned_exec_transitions',[]).append({'before':old,'after':item})
            owned[p.pid]=item;return item
        try:
            with log.with_suffix('.stdout.log').open('xb')as stdout,log.with_suffix('.stderr.log').open('xb')as stderr:
                # Isolated/no-site bootstrap cannot run user startup code before GO.
                child=subprocess.Popen([sys.executable,'-I','-S','-c',bootstrap,str(go_r),str(ready_w),*argv],env=environment,
                    stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,pass_fds=(locks['UID']['fd'],locks['BENCHMARK']['fd'],go_r,ready_w))
                children.append(child);os.close(go_r);go_r=None;os.close(ready_w);ready_w=None
                if not select.select([ready_r],[],[],5)[0]or os.read(ready_r,1)!=b'R':raise ValueError('phase bootstrap not ready')
                os.close(ready_r);ready_r=None
                p=psutil.Process(child.pid);before=remember(p);row['blocked_identity']=before
                allowed={str(Path(sys.executable).resolve()),str((Path(sys.base_prefix)/'Resources/Python.app/Contents/MacOS/Python').resolve())}
                if str(Path(before['executable']).resolve())not in allowed:raise ValueError('phase bootstrap interpreter differs')
                row['before_GO_FD_observation']=observe_fds(p,locks,psutil)
                check(child.pid);unchanged(record['inputs_sha256']);verify_source_manifest()
                awake=power.ScopedAwake(child);awake.__enter__()
                os.write(go_w,b'G');os.close(go_w);go_w=None;row['GO_after_actual_identity_and_both_FDs']=True
                target=str(Path(argv[0]).resolve());target_observed=False
                while child.poll()is None:
                    try:
                        awake.check()
                        parent=psutil.Process(child.pid);remember(parent)
                        descendants=[parent,*parent.children(recursive=True)]
                        rss=psutil.Process(os.getpid()).memory_info().rss
                        for process in descendants:
                            try:remember(process);rss+=process.memory_info().rss
                            except psutil.NoSuchProcess:pass
                        if awake.helper is not None and awake.helper.poll()is None:
                            try:rss+=psutil.Process(awake.helper.pid).memory_info().rss
                            except psutil.NoSuchProcess:pass
                        if not target_observed and str(Path(parent.exe()).resolve())==target:
                            row['actual_target_identity']=identify(parent)
                            row['target_FD_observation']=observe_fds(parent,locks,psutil);target_observed=True
                        check(child.pid,rss)
                    except psutil.NoSuchProcess:
                        if child.poll()is not None:break
                        raise
                    if time.monotonic()-phase_start>timeout:raise ValueError('phase deadline: '+label)
                    time.sleep(.1)
                row['returncode']=child.wait(timeout=5)
                row['target_exit_elapsed_seconds']=time.monotonic()-phase_start
                if row['returncode']!=0:raise ValueError('phase nonzero: '+label)
                if row['target_exit_elapsed_seconds']>timeout:raise ValueError('phase target exceeded deadline: '+label)
                if not target_observed:raise ValueError('actual target exe/FD observation missing: '+label)
            out=log.with_suffix('.stdout.log');err=log.with_suffix('.stderr.log')
            if out.stat().st_size>2<<20 or err.stat().st_size>2<<20:raise ValueError('bounded phase logs required')
            row.update(status='completed',stdout_sha256=sha(out),stderr_sha256=sha(err))
            return {'returncode':row['returncode'],'stdout':out.read_text(),'stderr':err.read_text()}
        except BaseException as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            for fd in (go_r,go_w,ready_r,ready_w):
                if fd is not None:os.close(fd)
            if child is not None:
                if child.poll()is None:
                    try:
                        parent=psutil.Process(child.pid);remember(parent)
                        for process in parent.children(recursive=True):remember(process)
                    except psutil.NoSuchProcess:pass
                # Observed descendants can outlive their Popen parent. They
                # remain ours by birth/exe and must not be skipped after exit.
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    for item in list(owned.values()):
                        try:
                            current=psutil.Process(item['pid'])
                            # A known process may have exec'ed since the last
                            # poll. Re-observe that same birth and record the
                            # actual exe before the immediate signal check.
                            if current.create_time()==item['create_time']:item=remember(current)
                        except psutil.NoSuchProcess:continue
                        if still_owned(item):
                            try:os.kill(item['pid'],sig)
                            except ProcessLookupError:continue
                            row['signals'].append({'pid':item['pid'],'birth':item['create_time'],'signal':sig.name})
                    if child.poll()is None:
                        try:child.wait(timeout=3)
                        except subprocess.TimeoutExpired:pass
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline and any(still_owned(item)for item in owned.values()):time.sleep(.05)
                    if not any(still_owned(item)for item in owned.values()):break
            if awake is not None:awake.close();row['idle_assertion']=awake.record
            row['owned_survivors']=[item for item in owned.values()if still_owned(item)]
            row['elapsed_seconds']=time.monotonic()-phase_start
            if row['owned_survivors']:row['status']='failed_surviving_owned_process'
            if row['signals']and row['status']=='completed':row['status']='failed_forced_descendant_cleanup'
            write(guard_path,guard)
            if row['owned_survivors']and not row.get('error'):raise ValueError('phase has surviving owned descendants')
            if row['signals']and not row.get('error'):raise ValueError('phase required forced owned-descendant cleanup')
            if awake is not None and (awake.record.get('exit_qualified')is not True or awake.record.get('validation_error')or awake.record.get('cleanup_error')):
                raise ValueError('phase idle/ownership postflight rejected')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
        benchmark=modules.qa.BENCHMARK_LOCK.open('a');fcntl.flock(benchmark,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root=Path(tempfile.gettempdir())/('airllm-native-locks-'+str(os.getuid()));root.mkdir(mode=0o700,exist_ok=True)
        key=native.NativeGlmModel._model_lock_key(object.__new__(native.NativeGlmModel));uid_path=root/(key+'.lock')
        uid=uid_path.open('a');fcntl.flock(uid,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.update(UID=kernel_lock(uid,uid_path),BENCHMARK=kernel_lock(benchmark,modules.qa.BENCHMARK_LOCK))
        guard['lock_capabilities']=locks;guard['locks_acquired_before_any_phase']=True
        guard['coordinator_identity']=identify(psutil.Process(os.getpid()))
        guard['power_admission']=gate.sample();baseline=native.resource_snapshot();guard['baseline']=baseline
        if not guard['power_admission']['passed']or baseline.get('pressure_level')!=1 or type(baseline.get('available_bytes'))is not int or baseline['available_bytes']<3<<30:raise ValueError('original normal-pressure/AC/3GiB build admission refused')
        unchanged(record['inputs_sha256']);unchanged(record['closed_package_inputs_sha256'])
        guard['status']='running';record['build_attempted']=True;write(guard_path,guard)
        env=clean_env();compiled=phase('compile',record['command'],env,120);record['returncode']=compiled['returncode']
        signed=phase('codesign',['/usr/bin/codesign','--force','--sign','-',record['binary']],env,30)
        record['codesign']=signed;record['binary_sha256']=sha(record['binary'])
        document=record['profile_document'];gc.validate_profile(document);write(destination/'gpu-profile.json',document)
        meta_env={**env,**document['environment'],'AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES':'0'}
        for ub in (8,16,32,64,128):
            row=phase('parameters-'+str(ub),[record['binary'],'--state-parameters',str(ub)],meta_env,15)
            record['parameter_smoke'][str(ub)]=row
            if 'event=submitted'in row['stderr']:raise ValueError('metadata unexpectedly submitted bridge GPU work')
            desc=json.loads(row['stdout'])
            for field in ('native_linkage','native_resource_context','native_backend_policy'):row[field]=desc.pop(field,None)
            row['state_parameters']=desc
            gc.validate_state(desc,document,ub);gc.validate_loaded_images(row['native_linkage'],document,PACKAGE/'runtime/bin')
            gc.validate_resource_context(row['native_resource_context'],0);gc.validate_backend_policy(row['native_backend_policy'])
            write(driver_path,record)  # guard_pending remains refused by all existing validators.
        row=phase('backend-policy',[record['binary'],'--backend-policy-selftest'],meta_env,15)
        if 'event=submitted'in row['stderr']:raise ValueError('registry smoke submitted bridge matrix work')
        gc.validate_backend_policy(json.loads(row['stdout']));record['backend_policy_smoke']=row
        record['inputs_after']=hashes(map(Path,record['inputs_sha256']))
        unchanged(record['inputs_sha256']);unchanged(record['closed_package_inputs_sha256']);verify_source_manifest()
        # Validate favorable schema IN MEMORY ONLY, before the final guard. Nothing
        # accepted can reach disk if this validation or the final postflight fails.
        accepted={**record,'status':'built_metadata_only_smoked',
            'actual_gpu_matrix_commands':None,'bridge_matrix_commands_observed':0}
        accepted['artifact_binding']=gc.validate_artifacts(document,gc.read_json(PROVIDER),accepted,
            libraries=PACKAGE/'runtime/bin',binary=record['binary'],helper_source=HELPER,
            runner_source=BASE/'native_gpu_run.py',guard_source=GUARD,provider_receipt_sha256=sha(PROVIDER))
        check();success=True
    except BaseException as error:
        failure=type(error).__name__+': '+str(error);guard.update(status='failed',error=failure)
    finally:
        # All phase-owned children are already reaped or explicitly accounted for.
        survivors=[child.pid for child in children if child.poll()is None]
        if survivors:success=False;guard.update(status='failed',error='unreaped owned children',surviving_owned_pids=survivors)
        if success:
            try:
                check();unchanged(record['inputs_sha256']);unchanged(record['closed_package_inputs_sha256']);verify_source_manifest()
                guard['postflight_source_integrity_passed']=True
            except BaseException as error:success=False;guard.update(status='failed',error='resource/source postflight: '+str(error))
        try:
            gate.sample(final=True);guard['power_gate']=gate.summary()
            if not guard['power_gate']['passed']:success=False;guard.update(status='failed',error='final AC/clock/continuity rejected')
        except BaseException as error:success=False;guard.update(status='failed',error='power postflight: '+str(error))
        for name,holder in (('UID',uid),('BENCHMARK',benchmark)):
            if holder is not None:
                try:holder.close();guard.setdefault('parent_FDs_closed',{})[name]=True
                except OSError as error:success=False;guard.update(status='failed',error='lock close failed: '+str(error))
        guard['elapsed_seconds']=time.monotonic()-started;guard['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        if guard['elapsed_seconds']>300:success=False;guard.update(status='failed',error='final build lifecycle exceeded300s')
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if not success:
        record.update(status='failed',error=guard.get('error',failure));write(guard_path,guard);write(driver_path,record)
        return record
    guard.update(status='completed',bridge_matrix_commands_observed=0,source_integrity_passed=True,
        all_phase_children_reaped=True,lock_window_closed=True,driver_receipt_was_pending_until_final_guard=True)
    write(guard_path,guard)
    accepted['supervision']={'kind':'dedicated_owner_benchmark_and_UID_v2','guard_receipt':str(guard_path),
        'guard_receipt_sha256':sha(guard_path),'supervisor_source_sha256':sha(__file__),
        'source_manifest_sha256':sha(HERE/'source-manifest.json'),'global_postflight_passed_before_publication':True}
    # LAST publication: before this atomic write, driver-build.json is pending.
    write(driver_path,accepted)
    return accepted

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=HERE/'build-run1')
    parser.add_argument('--plan',type=Path,default=HERE/'build-plan.json')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.plan.resolve().is_relative_to(HERE):raise ValueError('plan must stay inside this new candidate')
    if args.plan.exists():raise FileExistsError(args.plan)
    verify_source_manifest();record=plan(args.destination);write(args.plan,record)
    if args.execute:record=execute(record)
    print(json.dumps({'status':record['status'],'build_attempted':record['build_attempted'],'plan':str(args.plan)}))
    raise SystemExit(0 if record['status']in('planned_not_built','built_metadata_only_smoked')else 2)
