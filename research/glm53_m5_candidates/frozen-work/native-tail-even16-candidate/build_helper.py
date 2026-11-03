"""Plan by default; compile only with explicit --execute after coordination.

Builds one isolated scheduling helper against unchanged pinned V4 libraries.
No model or GPU workload is ever started by this script. The optional binary
smokes query only --state-parameters8/16/32/64/128 before any model/backend loading.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
POLICY='cpu8-tail-compatible-v1'
HEADER_NAMES=('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h','vendor/nlohmann/json.hpp')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()

def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def plan(native_source,libraries,output):
    manifest_path=ROOT/'prepare-manifest.json';prepared=json.loads(manifest_path.read_text())
    native_source,libraries,output=map(lambda p:Path(p).resolve(),[native_source,libraries,output])
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('fresh helper output must be inside this isolated candidate directory')
    original=Path(prepared['original_helper_source'])
    if sha(original)!=prepared['original_helper_source_sha256']:
        raise ValueError('original measured helper source changed')
    baseline_path=Path(prepared['original_build_receipt'])
    if sha(baseline_path)!=prepared['original_build_receipt_sha256']:
        raise ValueError('baseline build receipt changed')
    baseline=json.loads(baseline_path.read_text())
    if sha(prepared['original_helper_binary'])!=prepared['original_helper_binary_sha256'] or baseline['binary_sha256']!=prepared['original_helper_binary_sha256']:
        raise ValueError('original helper binary identity differs')
    source=ROOT/prepared['variant_helper_source']
    if sha(source)!=prepared['variant_helper_source_sha256']:
        raise ValueError('prepared candidate source changed')
    for name,expected in prepared['unchanged_copied_dependencies'].items():
        if sha(ROOT/name)!=expected:raise ValueError('copied unchanged helper dependency differs: '+name)
    reconstruction_path=Path(prepared['native_source_reconstruction_receipt'])
    if sha(reconstruction_path)!=prepared['native_source_reconstruction_receipt_sha256']:
        raise ValueError('source reconstruction receipt changed')
    reconstruction=json.loads(reconstruction_path.read_text())
    if reconstruction['status']!='source_exact' or len(reconstruction['source_checks'])!=13 or reconstruction['source_base_revision']!=prepared['native_base_revision']:
        raise ValueError('pinned exact V4 reconstruction receipt required')
    native_sources={}
    for name,row in reconstruction['source_checks'].items():
        actual=sha(native_source/name)
        if row['matches'] is not True or actual!=row['expected_sha256'] or actual!=row['actual_sha256']:
            raise ValueError('reconstructed native source differs: '+name)
        native_sources[name]=actual
    headers={}
    for relative in ('include/llama.h','ggml/include/ggml.h','ggml/include/ggml-backend.h'):
        expected=[value for path,value in baseline['inputs_sha256'].items() if path.endswith('/'+relative)]
        actual=sha(native_source/relative)
        if len(expected)!=1 or actual!=expected[0]:raise ValueError('pinned ABI header differs: '+relative)
        headers[relative]=actual
    json_pin=prepared['reference_vendor_json_header']
    if sha(json_pin['path'])!=json_pin['sha256']:raise ValueError('baseline JSON source header changed')
    headers['vendor/nlohmann/json.hpp']=sha(native_source/'vendor/nlohmann/json.hpp')
    if headers['vendor/nlohmann/json.hpp']!=json_pin['sha256']:raise ValueError('candidate JSON header differs from pinned baseline source')
    header_paths={name:str(native_source/name) for name in HEADER_NAMES}
    tracked_inputs={str(p):sha(p) for p in [source,source.with_name('snapshot.h'),ROOT/'research/glm53_gpu_v3/state_capture.inc',Path(__file__)]}
    tracked_inputs.update({header_paths[name]:headers[name] for name in HEADER_NAMES})
    for name in prepared['runtime_libraries_sha256']:
        path=libraries/name
        if path.is_symlink() or not path.is_file():raise ValueError('ordinary pinned runtime library required: '+name)
    command=['clang++','-std=c++17','-O2','-Wno-deprecated-declarations',
        '-I',str(native_source/'include'),'-I',str(native_source/'ggml/include'),'-I',str(native_source/'vendor'),str(source),
        '-L',str(libraries),'-Wl,-rpath,'+str(libraries),'-lllama','-lggml','-lggml-base','-o',str(output)]
    return {'schema_version':1,'status':'planned_not_built','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'policy':POLICY,'command':command,'output':str(output),'libraries':str(libraries),
        'native_base_revision':prepared['native_base_revision'],'source_reconstruction_receipt_sha256':sha(reconstruction_path),
        'original_build_receipt_sha256':sha(baseline_path),'original_helper_binary_sha256':prepared['original_helper_binary_sha256'],
        'inputs_sha256':tracked_inputs,'header_paths':header_paths,
        'header_sha256':headers,'native_sources_exact':native_sources,
        'runtime_libraries_expected_sha256':prepared['runtime_libraries_sha256'],
        'runtime_hashes_verified_for_this_build':False,'build_attempted':False,'build_performed':False,'binary_rebuilt':False,
        'model_loaded':False,'gpu_commands':0,'full_model_qualified':False,'numerical_fidelity_qualified':False,
        'scope':'isolated helper build plan; actual runtime hashes are checked before/after an explicitly requested compile; no inherited numerical qualification'}

def verify_tracked_inputs(record,phase):
    if set(record.get('header_sha256',{}))!=set(HEADER_NAMES) or set(record.get('header_paths',{}))!=set(HEADER_NAMES):
        raise ValueError('complete explicit helper header tracking is required')
    for relative in HEADER_NAMES:
        path=record['header_paths'][relative]
        expected=record['header_sha256'][relative]
        if record['inputs_sha256'].get(path)!=expected:
            raise ValueError('header declaration is outside tracked build inputs: '+relative)
    for path,expected in record['inputs_sha256'].items():
        if sha(path)!=expected:raise ValueError('tracked helper/header input changed '+phase+': '+path)


def compile_plan(record):
    libraries=Path(record['libraries']);output=Path(record['output'])
    if output.exists():raise FileExistsError(output)
    for name,expected in record['runtime_libraries_expected_sha256'].items():
        if sha(libraries/name)!=expected:raise ValueError('runtime library differs before compile: '+name)
    verify_tracked_inputs(record,'before compile')
    record['tracked_inputs_verified_before_compile']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    env={name:value for name,value in os.environ.items() if not name.startswith(('LLAMA_ARG_','AIRLLM_','GGML_','DYLD_')) and name!='LLAMA_API_KEY'}
    record['started_at_utc']=datetime.now(timezone.utc).isoformat()
    record['build_attempted']=True
    try:
        result=subprocess.run(record['command'],env=env,capture_output=True,text=True,timeout=120)
    except subprocess.TimeoutExpired as error:
        record.update(status='build_timeout',build_performed=True,compile_timeout_seconds=120,
            stdout=str(error.stdout or ''),stderr=str(error.stderr or ''),finished_at_utc=datetime.now(timezone.utc).isoformat())
        return record
    record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,build_performed=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    if result.returncode:
        record['status']='build_failed';return record
    record['binary_rebuilt']=True
    record['binary_sha256']=sha(output)
    verify_tracked_inputs(record,'after compile')
    record['tracked_inputs_verified_after_compile']=True
    actual={name:sha(libraries/name) for name in record['runtime_libraries_expected_sha256']}
    if actual!=record['runtime_libraries_expected_sha256']:raise ValueError('runtime library changed during helper compile')
    record.update(runtime_hashes_verified_for_this_build=True,native_libraries_sha256=actual)
    record['parameter_smoke']={}
    for ub in (8,16,32,64,128):
        smoke=subprocess.run([str(output),'--state-parameters',str(ub)],env=env,capture_output=True,text=True,timeout=15)
        record['parameter_smoke'][str(ub)]={'returncode':smoke.returncode,'stdout':smoke.stdout,'stderr':smoke.stderr}
        if smoke.returncode:raise ValueError('metadata-only parameter smoke failed')
        effective=json.loads(smoke.stdout)
        if effective.get('n_ctx')!=2048 or effective.get('n_batch')!=ub or effective.get('n_ubatch')!=ub or effective.get('prefill_chunk_policy')!=POLICY:
            raise ValueError('new binary metadata differs from the explicit candidate profile')
        record['parameter_smoke'][str(ub)]['state_parameters']=effective
    verify_tracked_inputs(record,'after metadata smoke')
    record['tracked_inputs_verified_after_smoke']=True
    record.update(status='built_metadata_only_smoked',binary_rebuilt=True)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-source',type=Path,required=True);p.add_argument('--libraries',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'build/native-prefill-cpu8-tail-compatible')
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--binding',type=Path,default=ROOT/'build-binding.json')
    p.add_argument('--execute',action='store_true',help='Explicitly compile helper after coordinator permission; default only writes a plan')
    args=p.parse_args()
    if args.receipt.exists():raise FileExistsError(args.receipt)
    record=plan(args.native_source,args.libraries,args.output)
    if args.execute:
        if args.binding.exists():raise FileExistsError(args.binding)
        try:record=compile_plan(record)
        except Exception as error:record.update(status='failed',error=type(error).__name__+': '+str(error))
    args.receipt.parent.mkdir(parents=True,exist_ok=True);write(args.receipt,record)
    if record['status']=='built_metadata_only_smoked':
        binding=json.loads((ROOT/'binding-template.json').read_text())
        binding.update(status='built_bound_not_numerically_qualified',created_at_utc=datetime.now(timezone.utc).isoformat())
        binding['candidate'].update(binary_sha256=record['binary_sha256'],build_receipt_sha256=sha(args.receipt))
        args.binding.parent.mkdir(parents=True,exist_ok=True);write(args.binding,binding)
    print(json.dumps({'status':record['status'],'policy':POLICY,'build_performed':record['build_performed'],'receipt':str(args.receipt)}))
    return 0 if record['status'] in ('planned_not_built','built_metadata_only_smoked') else 2

if __name__=='__main__':raise SystemExit(main())
