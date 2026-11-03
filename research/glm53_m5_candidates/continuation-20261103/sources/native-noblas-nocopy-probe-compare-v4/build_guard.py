"""Offline validation of the dedicated v2 build guard. Source only, unrun.

No process query, lock acquisition, native call or resource sample is performed
here. The exact original resource method is replayed on its recorded call times.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
BASE=WORK/'native-noblas-nocopy-candidate'
SUPERVISOR=WORK/'native-nocopy-build-supervisor-v2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read(path,limit=16<<20):
    path=Path(path)
    if not path.is_file()or path.stat().st_size>limit:raise ValueError('bounded actual build evidence required')
    return json.loads(path.read_text(),parse_constant=lambda v:(_ for _ in()).throw(ValueError('nonfinite build evidence: '+v)))

def finite(value,*,minimum=0):
    return type(value)in(int,float)and math.isfinite(value)and value>=minimum

def ident(value):
    if type(value)is not dict or set(value)!={'pid','create_time','executable'}:raise ValueError('exact build PID/birth/exe identity required')
    if type(value['pid'])is not int or value['pid']<=0 or not finite(value['create_time'],minimum=1)or type(value['executable'])is not str or not Path(value['executable']).is_absolute():raise ValueError('invalid build owned identity')
    return value

def fd_observation(value,identity,capabilities):
    if type(value)is not dict or value.get('identity')!=identity or set(value.get('locks',{}))!={'UID','BENCHMARK'}:raise ValueError('actual bracketed two-FD observation required')
    for name,capability in capabilities.items():
        expected={field:capability[field]for field in('fd','path','device','inode')}
        if value['locks'][name]!=expected:raise ValueError('actual inherited FD capability mismatch: '+name)

def validate(driver,driver_path,prepared,m):
    """Return a descriptor only after actual final build supervision is proven."""
    driver_path=Path(driver_path).resolve()
    if driver.get('status')!='built_metadata_only_smoked'or driver.get('build_attempted')is not True or driver.get('provider_rebuild')is not False or driver.get('model_loaded')is not False:raise ValueError('completed new metadata-only driver build required')
    if not(driver.get('inputs_before')==driver.get('inputs_sha256')==driver.get('inputs_after'))or not driver.get('inputs_sha256'):raise ValueError('build input hashes before/after differ')
    package=WORK/'metal-tensorops-q8-nocopy/build-run1';cpu=WORK/'native-noblas208-v4-candidate'
    provider_path=package/'provider-build.json'
    if sha(provider_path)!=prepared['provider_build_receipt_sha256']:raise ValueError('exact original provider receipt required')
    provider=read(provider_path);headers=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
    proles=m['gpu'].validate_input_provenance(provider,tuple('header:'+name for name in headers))
    helper=BASE/'driver/research/glm53_batched_prefill/native_prefill.cpp'
    expected_roles={'build_script':SUPERVISOR/'supervised_build.py','supervisor_source_manifest':SUPERVISOR/'source-manifest.json',
        'helper_source':helper,'snapshot_source':helper.with_name('snapshot.h'),'state_capture_source':BASE/'driver/research/glm53_gpu_v3/state_capture.inc',
        'runner_source':BASE/'native_gpu_run.py','profile_source':BASE/'profile.py','contract_source':BASE/'gpu_contract.py',
        'provider_receipt':provider_path,'cpu_v4_source':cpu/'research/glm53_batched_prefill/native_prefill.cpp','cpu_v4_prepare_manifest':cpu/'prepare-manifest.json',
        'guard_source':WORK/'airllm-next/air_llm/airllm/nextgen/glm_native.py','original_planner_source':BASE/'build_driver.py',
        'base_candidate_manifest':BASE/'prepare-manifest.json','runtime_manifest':package/'runtime/manifest.json',
        **{'header:'+name:Path(proles['header:'+name])for name in headers}}
    if set(driver.get('input_roles',{}))!=set(expected_roles):raise ValueError('complete exact driver source/header roles required')
    for role,path in expected_roles.items():
        if driver.get('input_roles',{}).get(role)!=str(path.resolve())or driver['inputs_sha256'].get(str(path.resolve()))!=sha(path):raise ValueError('reviewed exact build input role differs: '+role)
    expected_inputs={str(path.resolve())for path in expected_roles.values()}
    expected_inputs|={str((package/'runtime/bin'/('lib'+name+'.dylib')).resolve())for name in provider['native_libraries_sha256']}
    if set(driver['inputs_sha256'])!=expected_inputs:raise ValueError('complete driver input set differs from exact source/header/library allowlist')
    for path,digest in driver['inputs_sha256'].items():
        if type(digest)is not str or re.fullmatch('[0-9a-f]{64}',digest)is None or sha(path)!=digest:raise ValueError('actual build input changed')
    original=WORK/'metal-tensorops-q8-nocopy'
    closed_paths={str(path)for path in(provider_path,package/'driver-build.json',package/'gpu-profile.json',original/'source-manifest.json',original/'native_gpu_run.py',original/'gpu_contract.py',original/'profile.py')}
    if set(driver.get('closed_package_inputs_sha256',{}))!=closed_paths:raise ValueError('complete closed NoCopy provenance required')
    for path,digest in driver['closed_package_inputs_sha256'].items():
        if sha(path)!=digest:raise ValueError('closed original NoCopy package changed')
    original_command=read(package/'driver-build.json')['command'][:]
    original_command[original_command.index(str(original/'driver/research/glm53_batched_prefill/native_prefill.cpp'))]=str(helper)
    original_command[original_command.index('-o')+1]=str(driver_path.parent/'native-prefill-noblas-nocopy')
    if driver.get('command')!=original_command or driver.get('binary')!=original_command[original_command.index('-o')+1]:raise ValueError('compiler/SDK/flags/includes/libs/rpath differ from exact original command')
    supervision=driver.get('supervision',{})
    fields={'kind','guard_receipt','guard_receipt_sha256','supervisor_source_sha256','source_manifest_sha256','global_postflight_passed_before_publication'}
    if set(supervision)!=fields or supervision.get('kind')!='dedicated_owner_benchmark_and_UID_v2'or supervision.get('global_postflight_passed_before_publication')is not True:raise ValueError('final dedicated owner supervision required; guard_pending refused')
    guard_path=driver_path.with_name('build-guard.json')
    if supervision.get('guard_receipt')!=str(guard_path)or supervision.get('guard_receipt_sha256')!=sha(guard_path):raise ValueError('exact same-build final guard SHA required')
    for field,path in (('supervisor_source_sha256',SUPERVISOR/'supervised_build.py'),('source_manifest_sha256',SUPERVISOR/'source-manifest.json')):
        digest=sha(path)
        if supervision.get(field)!=digest or prepared['source_pins'].get(str(path.resolve()))!=digest:raise ValueError('reviewed build supervisor/manifest differs')
    source_manifest=read(SUPERVISOR/'source-manifest.json')
    if source_manifest.get('status')!='source_prepared_not_executed':raise ValueError('frozen source preparation required')
    for path,digest in source_manifest['source_pins'].items():
        if sha(path)!=digest:raise ValueError('build supervisor dependency differs')
    guard=read(guard_path)
    if guard.get('schema_version')!=1 or guard.get('status')!='completed'or guard.get('source_sha256')!=sha(SUPERVISOR/'supervised_build.py')or guard.get('guard_violation')is not None:raise ValueError('closed passed actual build guard required')
    for field in('source_integrity_passed','postflight_source_integrity_passed','all_phase_children_reaped','lock_window_closed','driver_receipt_was_pending_until_final_guard','locks_acquired_before_any_phase'):
        if guard.get(field)is not True:raise ValueError('build postflight/ownership proof missing: '+field)
    if guard.get('model_loaded')is not False or guard.get('bridge_matrix_commands_observed')!=0 or driver.get('bridge_matrix_commands_observed')!=0 or driver.get('actual_gpu_matrix_commands')is not None:raise ValueError('metadata scope differs; zero measured bridge commands is not zero driver activity')
    if guard.get('parent_FDs_closed')!={'UID':True,'BENCHMARK':True}or guard.get('surviving_owned_pids'):raise ValueError('all parent capabilities and phase processes must be closed')
    if guard.get('tree_rss_cap_bytes')!=2<<30 or guard.get('global_deadline_seconds')!=300 or not finite(guard.get('elapsed_seconds'),minimum=.000001)or guard['elapsed_seconds']>300:raise ValueError('unchanged bounded build envelope required')
    native=m['native_guard'];limits_doc=read(WORK/'cpu10-reference.json')['limits']
    if guard.get('native_limits')!=limits_doc or limits_doc['max_rss_bytes']!=int(8.5*(1<<30))or limits_doc['max_swap_growth_bytes']!=32<<20:raise ValueError('exact original configured native memory policy required')
    checker=object.__new__(native.NativeGlmModel);checker.limits=native.NativeLimits(**limits_doc)
    checker._warning_started=checker._headroom_low_started=checker._pressure_context=None
    baseline=guard.get('baseline',{})
    for field in('swap_used_bytes','available_bytes','pressure_level'):
        if type(baseline.get(field))is not int or baseline[field]<0:raise ValueError('complete integer build baseline required')
    if not finite(baseline.get('monotonic'),minimum=.000001)or baseline['pressure_level']!=1 or baseline['available_bytes']<3<<30:raise ValueError('original normal-pressure3GiB build admission required')
    samples=guard.get('samples')
    if type(samples)is not list or len(samples)<2:raise ValueError('actual build guard samples required')
    previous_time=baseline['monotonic'];previous_elapsed=-1
    for sample in samples:
        for field in('rss_bytes','available_bytes','swap_used_bytes','swap_growth_bytes','pressure_level'):
            if type(sample.get(field))is not int or sample[field]<0:raise ValueError('finite integer build sample required')
        when=sample.get('guard_monotonic');stamp=sample.get('monotonic');elapsed=sample.get('elapsed_seconds')
        if not finite(when)or not finite(stamp)or not finite(elapsed)or not previous_time<=when or not baseline['monotonic']<=stamp<=when or not previous_elapsed<=elapsed<=guard['elapsed_seconds']:raise ValueError('recorded original watchdog call times required')
        growth=max(0,sample['swap_used_bytes']-baseline['swap_used_bytes'])
        if sample['swap_growth_bytes']!=growth or sample['rss_bytes']>2<<30:raise ValueError('build tree RSS/swap accounting differs')
        code=native.NativeGlmModel._guard_resource_sample(checker,sample,baseline,when)
        if code:raise ValueError('original native resource replay rejects build: '+code)
        previous_time=when;previous_elapsed=elapsed
    owner=ident(guard.get('coordinator_identity'))
    caps=guard.get('lock_capabilities',{})
    if set(caps)!={'UID','BENCHMARK'}:raise ValueError('both original kernel capabilities required')
    key=native.NativeGlmModel._model_lock_key(object.__new__(native.NativeGlmModel))
    expected_paths={'UID':(Path(tempfile.gettempdir())/('airllm-native-locks-'+str(os.getuid()))/(key+'.lock')).resolve(),'BENCHMARK':Path('/tmp/airllm-glm53-benchmark.lock').resolve()}
    for name,cap in caps.items():
        if set(cap)!={'path','fd','device','inode','uid'}or cap.get('path')!=str(expected_paths[name])or cap.get('uid')!=os.getuid():raise ValueError('original lock identity/path required: '+name)
        if any(type(cap.get(field))is not int or cap[field]<0 for field in('fd','device','inode'))or cap['inode']==0:raise ValueError('complete kernel FD device/inode required')
    if caps['UID']['fd']==caps['BENCHMARK']['fd']or(caps['UID']['device'],caps['UID']['inode'])==(caps['BENCHMARK']['device'],caps['BENCHMARK']['inode']):raise ValueError('distinct two kernel lock capabilities required')
    stages=guard.get('stages');labels=['compile','codesign',*('parameters-'+str(ub)for ub in(8,16,32,64,128)),'backend-policy']
    if type(stages)is not list or [row.get('label')for row in stages]!=labels:raise ValueError('all eight exact build phases required')
    binary=driver['binary'];commands=[driver['command'],['/usr/bin/codesign','--force','--sign','-',binary],*([binary,'--state-parameters',str(ub)]for ub in(8,16,32,64,128)),[binary,'--backend-policy-selftest']]
    for row,command,timeout in zip(stages,commands,[120,30,15,15,15,15,15,15]):
        if row.get('status')!='completed'or row.get('returncode')!=0 or row.get('command')!=command or row.get('timeout_seconds')!=timeout or row.get('signals')!=[]or row.get('owned_survivors')!=[]or row.get('error'):raise ValueError('all exact phases must complete naturally without forced cleanup')
        if not finite(row.get('target_exit_elapsed_seconds'))or row['target_exit_elapsed_seconds']>timeout or not finite(row.get('elapsed_seconds'))or not row['target_exit_elapsed_seconds']<=row['elapsed_seconds']<=timeout+12:raise ValueError('bounded target and owned cleanup times required')
        blocked=ident(row.get('blocked_identity'));target=ident(row.get('actual_target_identity'))
        if target['pid']==owner['pid']or target['pid']!=blocked['pid']or target['create_time']!=blocked['create_time']or Path(target['executable']).resolve()!=Path(command[0]).resolve():raise ValueError('actual phase exec/PID/birth binding differs')
        fd_observation(row.get('before_GO_FD_observation'),blocked,caps);fd_observation(row.get('target_FD_observation'),target,caps)
        if row.get('GO_after_actual_identity_and_both_FDs')is not True:raise ValueError('phase computation did not follow observed two-FD GO')
        power_report={'owned_pid':target['pid'],'power_gate':{**guard.get('power_gate',{}),'idle_assertion':row.get('idle_assertion')}}
        if not m['power'].power_evidence_matches(power_report,'ac'):raise ValueError('actual phase idle + continuous final AC gate required')
        if row['idle_assertion'].get('target_create_time')!=target['create_time']:raise ValueError('phase idle assertion bound to different PID lifetime')
        for suffix,field in(('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256')):
            path=driver_path.parent/(row['label']+suffix)
            if not path.is_file()or path.stat().st_size>2<<20 or sha(path)!=row.get(field):raise ValueError('actual bounded phase logs differ')
            payload=path.read_text()
            if 'event=submitted'in payload:raise ValueError('build metadata unexpectedly submitted bridge matrix work')
            smoke=driver.get('parameter_smoke',{}).get(row['label'].removeprefix('parameters-'))if row['label'].startswith('parameters-')else driver.get('backend_policy_smoke')if row['label']=='backend-policy'else driver.get('codesign')if row['label']=='codesign'else None
            if smoke is not None and smoke.get('stdout'if field=='stdout_sha256'else'stderr')!=payload:raise ValueError('driver stdout/stderr differs from actual guarded phase')
    if guard.get('power_admission',{}).get('passed')is not True:raise ValueError('initial build AC admission missing')
    return {'kind':'validated_dedicated_owner_build_v2','guard_receipt':{'path':str(guard_path),'sha256':sha(guard_path),'bytes':guard_path.stat().st_size},
        'supervisor_source_sha256':sha(SUPERVISOR/'supervised_build.py'),'source_manifest_sha256':sha(SUPERVISOR/'source-manifest.json'),
        'original_resource_policy_replayed':True,'two_actual_inherited_FDs_before_GO_each_phase':True,'all_eight_phases_closed':True,
        'global_postflight_before_favorable_publication':True,'model_loaded':False,'qualification':False}
