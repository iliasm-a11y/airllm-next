"""Isolated fixture compiler plan; no compiler/native execution without --execute."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;BRIDGE=HERE.parent;WORK=BRIDGE.parent
sys.path.insert(0,str(BRIDGE))
import gpu_contract as gc


def write(path,row):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def plan(package):
    package=Path(package).resolve()
    gc.require(package.is_relative_to(BRIDGE),'FIXTURE_PACKAGE','Future package must stay under isolated bridge candidate')
    v4=gc.read_json(WORK/'metal-integration-v4-source/inputs/runtime-build-manifest.json')
    configure=next(row['argv'] for row in v4['commands'] if row['argv'][:2]==['cmake','-S'])
    compiler=next(arg.split('=',1)[1] for arg in configure if arg.startswith('-DCMAKE_CXX_COMPILER='))
    source=package/'source';libraries=package/'runtime/bin';output=package/'provider-fixture'
    roles={'fixture_source':str(HERE/'provider_fixture.cpp'),'build_script':str(Path(__file__).resolve()),'controller_source':str(HERE/'provider_fixture.py')}
    originals={}
    for name in gc.HEADERS:
        original=(BRIDGE/'source'/name) if (BRIDGE/'source'/name).exists() else WORK/'metal-integration-v4-source/source'/name
        originals[name]={'path':str(original),'sha256':gc.sha(original)}
    command=[compiler,'-isysroot',v4['macos_sdk_path'],'-std=c++17','-O3','-Wno-deprecated-declarations',
        '-I',str(source/'include'),'-I',str(source/'ggml/include'),'-I',str(source/'vendor'),str(HERE/'provider_fixture.cpp'),
        '-L',str(libraries),'-Wl,-rpath,'+str(libraries),'-lllama','-lggml','-lggml-base','-lggml-cpu','-o',str(output)]
    return {'schema_version':2,'status':'prepared_not_compiled_not_executed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'binary':str(output),'package_build':str(package),'input_roles':roles,
        'inputs_sha256':{path:gc.sha(path) for path in roles.values()},'planned_headers':originals,
        'native_libraries_sha256':None,'compile_timeout_seconds':120,'metadata_smoke_timeout_seconds':15,
        'model_loaded':False,'gpu_commands':0,'build_performed':False,'measured_compile_seconds':None}


def execute(record):
    import native_gpu_run as n
    import time
    package=Path(record['package_build']);binary=Path(record['binary']);libraries=package/'runtime/bin'
    gc.require(not binary.exists(),'FIXTURE_OUTPUT','Fresh fixture binary required')
    provider=gc.read_json(package/'provider-build.json');driver=gc.read_json(package/'driver-build.json');document=gc.read_json(package/'gpu-profile.json')
    gc.validate_artifacts(document,provider,driver,libraries=libraries,binary=package/'native-prefill-gpu',
        helper_source=BRIDGE/'driver/research/glm53_batched_prefill/native_prefill.cpp',runner_source=BRIDGE/'native_gpu_run.py',
        guard_source=n.NATIVE_GUARD_SOURCE,provider_receipt_sha256=gc.sha(package/'provider-build.json'))
    for name,row in record['planned_headers'].items():
        path=package/'source'/name
        gc.require(gc.sha(row['path'])==row['sha256']==gc.sha(path),'FIXTURE_ABI','Expected staged/original ABI header differs')
        record['input_roles']['header:'+name]=str(path)
    for role,filename in (('provider_receipt','provider-build.json'),('driver_receipt','driver-build.json'),('profile','gpu-profile.json')):
        record['input_roles'][role]=str(package/filename)
    record['inputs_sha256'].update({path:gc.sha(path) for path in record['input_roles'].values()})
    def unchanged():return {path:gc.sha(path) for path in record['inputs_sha256']}
    record['inputs_before']=unchanged();env,_=n.clean_environment({});env['AIRLLM_CPU_METAL_Q8_BRIDGE']='0'
    version=subprocess.run([record['command'][0],'--version'],env=env,capture_output=True,text=True,timeout=15)
    gc.require(version.returncode==0 and version.stdout.strip()==provider['original_compiler_version'],'FIXTURE_COMPILER','Pinned V4 compiler version required')
    record['compiler_version']=version.stdout.strip();record['compiler_binary_sha256']=gc.sha(record['command'][0])
    start=time.perf_counter();record['build_performed']=True
    completed=subprocess.run(record['command'],env=env,capture_output=True,text=True,timeout=record['compile_timeout_seconds'])
    record.update(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,measured_compile_seconds=time.perf_counter()-start)
    gc.require(completed.returncode==0,'FIXTURE_BUILD_FAILED','Native fixture compiler failed')
    signed=subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(binary)],env=env,capture_output=True,text=True,timeout=30)
    record['codesign']={'returncode':signed.returncode,'stdout':signed.stdout,'stderr':signed.stderr}
    gc.require(signed.returncode==0,'FIXTURE_CODESIGN','Fixture signing failed')
    record['binary_sha256']=gc.sha(binary)
    smoke=subprocess.run([str(binary),'--describe'],env=env,capture_output=True,text=True,timeout=record['metadata_smoke_timeout_seconds'])
    record['metadata_smoke']={'returncode':smoke.returncode,'stdout':smoke.stdout,'stderr':smoke.stderr}
    gc.require(smoke.returncode==0 and 'event=submitted' not in smoke.stderr,'FIXTURE_SMOKE','Native metadata smoke failed')
    described=json.loads(smoke.stdout)
    gc.require(described.get('cpu_metal_q8_bridge_compiled') is True and described.get('threads')==10 and described.get('uses_reference_mode') is False,
        'FIXTURE_FEATURE','Actual compiled feature/ordinary CPU10 descriptor required')
    gc.validate_loaded_images(described.get('native_linkage'),document,libraries)
    record['native_libraries_sha256']={name:gc.sha(libraries/('lib'+name+'.dylib')) for name in provider['native_libraries_sha256']}
    gc.require(record['native_libraries_sha256']==provider['native_libraries_sha256'],'FIXTURE_LIBRARIES','Provider libraries changed during fixture build')
    record['inputs_after']=unchanged();gc.require(record['inputs_before']==record['inputs_sha256']==record['inputs_after'],'FIXTURE_SOURCE_IDENTITY','Fixture build inputs changed')
    record.update(status='built_metadata_only_smoked',provider_build_receipt_sha256=gc.sha(package/'provider-build.json'))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package-build',type=Path,default=BRIDGE/'build-run1')
    parser.add_argument('--receipt',type=Path,default=HERE/'fixture-build-plan.json');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.receipt.exists():raise FileExistsError(args.receipt)
    record=plan(args.package_build)
    if args.execute:
        try:execute(record)
        except Exception as error:record.update(status='failed',error_code=getattr(error,'code',type(error).__name__),error=str(error))
    write(args.receipt,record);print(json.dumps({'status':record['status'],'build_performed':record['build_performed'],'receipt':str(args.receipt)}))
    return 0 if record['status'] in ('prepared_not_compiled_not_executed','built_metadata_only_smoked') else 2

if __name__=='__main__':raise SystemExit(main())
