"""Prepare a pinned isolated package plan; --execute is coordination-only.

Default does not call any compiler, native helper, model or GPU. Execution
requires a new destination; original V4/reconstruction/active files are read-only.
A metadata smoke may load Metal libraries but submits no GPU command.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
V4=Path('/Users/macbook/Documents/Codex/2026-11-02/referenced-chatgpt-conversation-this-is-an-2/work/runtime-v4/manifest.json')
RECON=WORK/'metal-integration-v4-source/receipt.json'
V4_SHA='7b0ac6dddabf542aae42d7fb8bb8762d6b98e1705dc423b09ad071112d0c8750'
HEADERS=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')
TAIL_POLICY='cpu8-tail-compatible-v1'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def env_clean():
    return {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_ARG_','AIRLLM_','GGML_','DYLD_')) and k not in ('LLAMA_API_KEY','CFLAGS','CXXFLAGS','OBJCXXFLAGS','LDFLAGS','CC','CXX','OBJCXX')}

def plan(destination):
    destination=Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(HERE):raise ValueError('Fresh isolated destination under bridge candidate required')
    if sha(V4)!=V4_SHA:raise ValueError('Historical V4 build manifest changed')
    v4=json.loads(V4.read_text());recon=json.loads(RECON.read_text());manifest=json.loads((HERE/'source-manifest.json').read_text())
    if recon['status']!='source_exact' or recon['build_manifest_sha256']!=V4_SHA:raise ValueError('Exact V4 reconstruction required')
    if len(recon['source_checks'])!=13 or any(not row['matches'] or sha(Path(recon['source'])/name)!=row['expected_sha256'] for name,row in recon['source_checks'].items()):raise ValueError('V4 reconstructed sources changed')
    if sha(HERE/'cpu-metal-q8-bridge.patch')!=manifest['patch_sha256']:raise ValueError('Provider patch identity differs')
    source=destination/'source';build=destination/'build';package=destination/'runtime';binary=destination/'native-prefill-gpu'
    configure=next(row['argv'] for row in v4['commands'] if row['argv'][:2]==['cmake','-S'])[:]
    configure[configure.index('-S')+1]=str(source);configure[configure.index('-B')+1]=str(build)
    cxx=next(arg.split('=',1)[1] for arg in configure if arg.startswith('-DCMAKE_CXX_COMPILER='))
    cc=next(arg.split('=',1)[1] for arg in configure if arg.startswith('-DCMAKE_C_COMPILER='))
    # New Objective-C++ language uses the same pinned compiler, not a PATH default.
    configure+=['-DGGML_CPU_METAL_Q8_BRIDGE=ON','-DCMAKE_OBJCXX_COMPILER='+cxx,
                '-DCMAKE_OBJC_COMPILER='+cc,
                '-DCMAKE_PROJECT_INCLUDE='+str(HERE/'enable_bridge_objcxx.cmake')]
    compile_argv=next(row['argv'] for row in v4['commands'] if row['argv'][:2]==['cmake','--build'])[:]
    compile_argv[2]=str(build)
    helper=HERE/'driver/research/glm53_batched_prefill/native_prefill.cpp'
    driver=[cxx,'-isysroot',v4['macos_sdk_path'],'-std=c++17','-O3','-Wno-deprecated-declarations',
        '-I',str(source/'include'),'-I',str(source/'ggml/include'),'-I',str(source/'vendor'),str(helper),
        '-L',str(package/'bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(package/'bin'),'-o',str(binary)]
    roles={'source_manifest':str(HERE/'source-manifest.json'),'bridge_patch':str(HERE/'cpu-metal-q8-bridge.patch'),
        'reconstruction_receipt':str(RECON),'v4_build_manifest':str(V4),'build_script':str(Path(__file__).resolve()),
        'kernel_standalone':str(WORK/'airllm-next/research/glm53_batched_prefill/metal_q8.metal'),
        'root_objcxx_enable':str(HERE/'enable_bridge_objcxx.cmake')}
    for name,row in manifest['files'].items():
        path=HERE/'source'/name
        if sha(path)!=row['candidate_sha256']:raise ValueError('Overlay changed: '+name)
        roles['overlay:'+name]=str(path)
    pins=[]
    for row in v4['patches']:
        pin=RECON.parent/'inputs'/row['name']
        if sha(pin)!=row['sha256']:raise ValueError('Pinned patch changed: '+row['name'])
        roles['pin:'+row['name']]=str(pin);pins.append(str(pin))
    for name in recon['source_checks']:roles['v4_source:'+name]=str(Path(recon['source'])/name)
    for name in HEADERS:roles['v4_header:'+name]=str(Path(recon['source'])/name)
    return {'schema_version':2,'status':'planned_not_built','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'destination':str(destination),'source':str(source),'build':str(build),'package':str(package),'binary':str(binary),
        'historical_v4_manifest_sha256':V4_SHA,'historical_configure_command':next(row['argv'] for row in v4['commands'] if row['argv'][:2]==['cmake','-S']),
        'configure_command':configure,'compile_command':compile_argv,'driver_command':driver,'ordered_v4_patches':pins,
        'input_roles':roles,'inputs_sha256':{path:sha(path) for path in roles.values()},
        'provider_patch_sha256':manifest['patch_sha256'],'kernel_sha256':sha(roles['kernel_standalone']),
        'source_base_revision':v4['source_revision'],'original_compiler_version':v4['compiler_version'],
        'original_cmake_version':v4['cmake_version'],'historical_cmake_cache_available':False,
        'configure_timeout_seconds':120,'compile_timeout_seconds':600,'driver_timeout_seconds':120,
        'jobs':v4['jobs'],'build_performed':False,'model_loaded':False,'gpu_commands':0,
        'whole_model_qualified':False,'scope':'Historical configure/build options plus bridge ON, matching OBJCXX compiler and tracked root-language hook; isolated package'}

def verify_inputs(record):
    actual={path:sha(path) for path in record['inputs_sha256']}
    if actual!=record['inputs_sha256']:raise ValueError('A tracked source/include/pin changed')
    return actual

def execute(record):
    destination=Path(record['destination']);source=Path(record['source']);build=Path(record['build']);package=Path(record['package']);binary=Path(record['binary'])
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir();logs=destination/'logs';logs.mkdir();env=env_clean()
    record.update(status='building',started_at_utc=datetime.now(timezone.utc).isoformat(),commands=[],build_performed=True)
    def run(argv,timeout,cwd=None):
        started=time.perf_counter();index=len(record['commands']);stem=logs/f'{index:02d}'
        result=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
        stem.with_suffix('.stdout.log').write_text(result.stdout);stem.with_suffix('.stderr.log').write_text(result.stderr)
        record['commands'].append({'argv':argv,'cwd':str(cwd) if cwd else None,'returncode':result.returncode,
            'elapsed_seconds':time.perf_counter()-started,'stdout_file':str(stem.with_suffix('.stdout.log')),'stderr_file':str(stem.with_suffix('.stderr.log'))})
        if result.returncode:raise ValueError('Build step failed; see '+str(stem))
        return result
    record['inputs_before']=verify_inputs(record)
    compiler=record['driver_command'][0]
    version=run([compiler,'--version'],15).stdout.strip()
    if version!=record['original_compiler_version']:raise ValueError('Compiler version differs from recorded V4')
    if run(['cmake','--version'],15).stdout.splitlines()[0]!=record['original_cmake_version']:raise ValueError('CMake version differs from recorded V4')
    run(['git','clone','--shared','--no-checkout',json.loads(RECON.read_text())['source'],str(source)],60)
    run(['git','checkout','--detach',record['source_base_revision']],30,source)
    for path in [*record['ordered_v4_patches'],record['input_roles']['bridge_patch']]:
        run(['git','apply','--check',path],30,source);run(['git','apply',path],30,source)
    overlay=json.loads(Path(record['input_roles']['source_manifest']).read_text())['files']
    recon=json.loads(RECON.read_text())
    for name,row in recon['source_checks'].items():
        expected=overlay.get(name,{}).get('candidate_sha256',row['expected_sha256'])
        if sha(source/name)!=expected:raise ValueError('Staged native source differs: '+name)
        record['input_roles']['native:'+name]=str(source/name)
    for name,row in overlay.items():
        if sha(source/name)!=row['candidate_sha256']:raise ValueError('Staged overlay differs: '+name)
        record['input_roles']['native_overlay:'+name]=str(source/name)
    for name in HEADERS:record['input_roles']['header:'+name]=str(source/name)
    for name in ('CMakeLists.txt','ggml/CMakeLists.txt'):record['input_roles']['cmake:'+name]=str(source/name)
    record['inputs_sha256'].update({path:sha(path) for path in record['input_roles'].values()})
    record['inputs_before']=verify_inputs(record)
    run(record['configure_command'],record['configure_timeout_seconds'])
    record['cmake_cache_sha256']=sha(build/'CMakeCache.txt')
    run(record['compile_command'],record['compile_timeout_seconds'])
    (package/'bin').mkdir(parents=True)
    for path in sorted((build/'bin').iterdir()):
        if path.name.endswith('.dylib') or path.name in ('llama-server','llama-completion'):
            target=package/'bin'/path.name
            if path.is_symlink():target.symlink_to(os.readlink(path))
            elif path.is_file():shutil.copy2(path,target)
    for path in sorted((package/'bin').iterdir()):
        if not path.is_symlink():run(['/usr/bin/codesign','--force','--sign','-',str(path)],30)
    # Versioned real files and unversioned aliases identify the same bytes.
    libraries={name:sha(package/'bin'/('lib'+name+'.dylib')) for name in ('ggml','ggml-base','ggml-cpu','ggml-blas','ggml-metal','llama')}
    artifacts=[]
    for path in sorted((package/'bin').iterdir()):
        artifacts.append({'path':'bin/'+path.name,**({'symlink':os.readlink(path)} if path.is_symlink() else {'bytes':path.stat().st_size,'sha256':sha(path)})})
    write(package/'manifest.json',{'schema_version':1,'status':'built','source_revision':record['source_base_revision'],
        'artifacts':artifacts,'provider_patch_sha256':record['provider_patch_sha256'],'whole_model_qualified':False})
    record.update(status='compiled_feature_query_pending',native_libraries_sha256=libraries)
    helper=HERE/'driver/research/glm53_batched_prefill/native_prefill.cpp'
    driver_roles={'helper_source':str(helper),'snapshot_source':str(helper.with_name('snapshot.h')),
        'state_capture_source':str(HERE/'driver/research/glm53_gpu_v3/state_capture.inc'),'build_script':str(Path(__file__).resolve()),
        **{'header:'+name:str(source/name) for name in HEADERS}}
    driver={'schema_version':2,'status':'building','prefill_chunk_policy':TAIL_POLICY,'command':record['driver_command'],
        'input_roles':driver_roles,'inputs_sha256':{path:sha(path) for path in driver_roles.values()},
        'native_libraries_sha256':libraries,
        'parameter_smoke':{},'model_loaded':False,'gpu_commands':0}
    driver['inputs_before']=verify_inputs(driver)
    try:
        result=run(record['driver_command'],record['driver_timeout_seconds']);driver['returncode']=result.returncode
        run(['/usr/bin/codesign','--force','--sign','-',str(binary)],30);driver['binary_sha256']=sha(binary)
        # No model, buffer allocation or GPU dispatch: metadata only. The bridge
        # configuration registers the provider but creates its workspace lazily.
        import gpu_contract as gc
        document=gc.profiles.profile({'CPU_METAL_Q8_BRIDGE_COMPILED':True,'native_libraries_sha256':libraries,
            'provider_patch_sha256':record['provider_patch_sha256'],'kernel_sha256':record['kernel_sha256'],
            'runner_source_sha256':sha(HERE/'native_gpu_run.py'),
            'native_guard_policy_sha256':sha(WORK/'airllm-next/air_llm/airllm/nextgen/glm_native.py')})
        write(destination/'gpu-profile.json',document)
        smoke_env={**env,**document['environment'],'AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES':'0'}
        for ub in (8,16,32,64,128):
            result=subprocess.run([str(binary),'--state-parameters',str(ub)],env=smoke_env,capture_output=True,text=True,timeout=15)
            row={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
            driver['parameter_smoke'][str(ub)]=row
            if result.returncode:raise ValueError('Metadata smoke failed')
            descriptor=json.loads(result.stdout);row['state_parameters']={k:v for k,v in descriptor.items() if k not in ('native_linkage','native_resource_context')}
            row['native_linkage']=descriptor.get('native_linkage');row['native_resource_context']=descriptor.get('native_resource_context')
            gc.validate_state(row['state_parameters'],document,ub);gc.validate_loaded_images(row['native_linkage'],document,package/'bin');gc.validate_resource_context(row['native_resource_context'],0)
            if 'event=submitted' in result.stderr:raise ValueError('Metadata smoke unexpectedly submitted GPU work')
        if libraries!={name:sha(package/'bin'/('lib'+name+'.dylib')) for name in libraries}:raise ValueError('Native library changed during helper build/smokes')
        record.update(status='built',CPU_METAL_Q8_BRIDGE_COMPILED=True,inputs_after=verify_inputs(record),
            compiled_feature_query_verified=True,finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(destination/'provider-build.json',record)
        driver.update(status='built_metadata_only_smoked',inputs_after=verify_inputs(driver),
            provider_build_receipt_sha256=sha(destination/'provider-build.json'))
    except Exception as error:
        driver.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:write(destination/'driver-build.json',driver)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=HERE/'build-run1')
    parser.add_argument('--plan',type=Path,default=HERE/'build-plan.json')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.plan.exists():raise FileExistsError(args.plan)
    record=plan(args.destination);write(args.plan,record)
    if args.execute:
        try:execute(record)
        except Exception as error:
            record.update(status='failed',error=type(error).__name__+': '+str(error))
            write(Path(record['destination'])/'provider-failure.json',record);raise
    print(json.dumps({'status':record['status'],'build_performed':record['build_performed'],'plan':str(args.plan)}))

if __name__=='__main__':main()
