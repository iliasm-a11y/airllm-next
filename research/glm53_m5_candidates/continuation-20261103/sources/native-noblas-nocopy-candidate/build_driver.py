"""SOURCE ONLY: plan a fresh driver against the exact, immutable NoCopy package.

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
EXISTING=WORK/'metal-tensorops-q8-nocopy'
PACKAGE=EXISTING/'build-run1'
PROVIDER=PACKAGE/'provider-build.json'
OLD_DRIVER=PACKAGE/'driver-build.json'
PROVIDER_SHA='4fe68689c17b1c335f7b2fab2087aa99cf179d1dbe760d2a810d9cec1b01cb63'
OLD_DRIVER_SHA='a861e92a8e14a94e674c3cc9a4cb02401d6c0f73887c02066c3b068e135014ab'
HEADER_NAMES=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
HELPER=HERE/'driver/research/glm53_batched_prefill/native_prefill.cpp'
CPU_V4=WORK/'native-noblas208-v4-candidate'
GUARD=WORK/'airllm-next/air_llm/airllm/nextgen/glm_native.py'
sys.path.insert(0,str(HERE))
def exact_sibling(name):
    """Refuse another cached path, then load the exact current sibling bytes."""
    path=(HERE/(name+".py")).resolve()
    cached=sys.modules.get(name)
    if cached is not None and Path(getattr(cached,"__file__","")).resolve()!=path:
        raise ImportError("foreign cached module refused before native/build work: "+name)
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
        'state_capture_source':HERE/'driver/research/glm53_gpu_v3/state_capture.inc',
        'build_script':Path(__file__),'profile_source':HERE/'profile.py','contract_source':HERE/'gpu_contract.py',
        'runner_source':HERE/'native_gpu_run.py','provider_receipt':PROVIDER,
        'cpu_v4_source':CPU_V4/'research/glm53_batched_prefill/native_prefill.cpp','cpu_v4_prepare_manifest':CPU_V4/'prepare-manifest.json',
        'guard_source':GUARD,'runtime_manifest':PACKAGE/'runtime/manifest.json',
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
        'scope':'fresh driver only; same exact provider libs; no inherited model qualification'}

def execute(record):
    # This function is prepared for later explicit coordination. It has NOT run.
    n=exact_sibling("native_gpu_run")
    dest=Path(record['destination']);dest.mkdir(parents=True,exist_ok=False)
    record['inputs_before']=hashes(map(Path,record['inputs_sha256']))
    unchanged(record['inputs_sha256']);unchanged(record['closed_package_inputs_sha256'])
    env=clean_env();uid=None;gate=n.PowerGate('ac');started=time.monotonic()
    with n.qa.BENCHMARK_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            record['power_admission']=gate.sample()
            if not record['power_admission']['passed']:raise ValueError('AC admission refused')
            uid,record['uid_model_lock']=n.acquire_uid_model_lock()
            record['build_attempted']=True
            result=subprocess.run(record['command'],env=env,capture_output=True,text=True,timeout=record['compile_timeout_seconds'])
            (dest/'compile.stdout.log').write_text(result.stdout);(dest/'compile.stderr.log').write_text(result.stderr)
            record['returncode']=result.returncode
            if result.returncode:raise ValueError('Driver compilation failed')
            sign=subprocess.run(['/usr/bin/codesign','--force','--sign','-',record['binary']],env=env,capture_output=True,text=True,timeout=30)
            record['codesign']={'returncode':sign.returncode,'stdout':sign.stdout,'stderr':sign.stderr}
            if sign.returncode:raise ValueError('Driver signing failed')
            record['binary_sha256']=sha(record['binary'])
            document=record['profile_document'];gc.validate_profile(document)
            write(dest/'gpu-profile.json',document)
            env.update(document['environment']);env['AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES']='0'
            for ub in (8,16,32,64,128):
                if not gate.sample()['passed']:raise ValueError('AC changed before metadata smoke')
                command=[record['binary'],'--state-parameters',str(ub)]
                child=subprocess.run(command,env=env,pass_fds=(uid.fileno(),),capture_output=True,text=True,timeout=15)
                row={'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
                record['parameter_smoke'][str(ub)]=row
                if child.returncode or 'event=submitted' in child.stderr:raise ValueError('Metadata smoke failed or submitted matrix GPU work')
                descriptor=json.loads(child.stdout)
                for field in ('native_linkage','native_resource_context','native_backend_policy'):row[field]=descriptor.pop(field,None)
                row['state_parameters']=descriptor
                gc.validate_state(descriptor,document,ub);gc.validate_backend_policy(row['native_backend_policy'])
                gc.validate_loaded_images(row['native_linkage'],document,PACKAGE/'runtime/bin')
                gc.validate_resource_context(row['native_resource_context'],0)
            child=subprocess.run([record['binary'],'--backend-policy-selftest'],env=env,pass_fds=(uid.fileno(),),capture_output=True,text=True,timeout=15)
            record['backend_policy_smoke']={'returncode':child.returncode,'stdout':child.stdout,'stderr':child.stderr}
            if child.returncode or 'event=submitted' in child.stderr:raise ValueError('Registry smoke failed or dispatched GPU matrix')
            gc.validate_backend_policy(json.loads(child.stdout))
            unchanged(record['inputs_sha256']);unchanged(record['closed_package_inputs_sha256'])
            record['inputs_after']=hashes(map(Path,record['inputs_sha256']))
            record['status']='built_metadata_only_smoked';record['actual_gpu_matrix_commands']=0
            record['artifact_binding']=gc.validate_artifacts(document,gc.read_json(PROVIDER),record,
                libraries=PACKAGE/'runtime/bin',binary=record['binary'],helper_source=HELPER,
                runner_source=HERE/'native_gpu_run.py',guard_source=GUARD,provider_receipt_sha256=sha(PROVIDER))
        except Exception as error:
            record.update(status='failed',error=type(error).__name__+': '+str(error));raise
        finally:
            if uid is not None:
                uid.close();record['uid_model_lock']['parent_fd_closed']=True
            gate.sample(final=True);record['power_gate']=gate.summary()
            if record['status']=='built_metadata_only_smoked' and not record['power_gate']['passed']:record.update(status='failed',error='Power postflight rejected')
            record['elapsed_seconds']=time.monotonic()-started
            write(dest/'driver-build.json',record)
    if record['status']!='built_metadata_only_smoked':raise ValueError('Build postflight rejected')
    write(dest/'build-binding.json',{'schema_version':1,'status':'driver_metadata_bound_only',
        'provider_build_receipt_sha256':sha(PROVIDER),'driver_build_receipt_sha256':sha(dest/'driver-build.json'),
        'profile_document_sha256':sha(dest/'gpu-profile.json'),'canonical_profile_sha256':record['profile_document']['canonical_sha256'],
        'binary_sha256':record['binary_sha256'],'native_libraries_sha256':record['native_libraries_sha256'],
        'same_existing_mathlibs':True,'model_loaded':False,'model_numeric_qualified':False,'full_campaign_qualified':False})
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=HERE/'build-run1')
    parser.add_argument('--plan',type=Path,default=HERE/'build-plan.json')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.plan.exists():raise FileExistsError(args.plan)
    result=plan(args.destination);write(args.plan,result)
    if args.execute:execute(result)
    print(json.dumps({'status':result['status'],'build_attempted':result['build_attempted'],'plan':str(args.plan)}))
