"""Compile only the research helper against the existing pinned native runtime."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

REVISION='3f8af15a76ac8675812e856da45454bafc90f12c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    source=args.source.resolve(); runtime=args.runtime.resolve(); output=args.output.resolve()
    if output.exists() or output.with_suffix('.build.json').exists():
        raise FileExistsError('existing build output will not be overwritten')
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISION:
        raise ValueError('llama.cpp base revision differs from the pinned runtime API')
    root=Path(__file__).resolve().parent
    inputs=[root/'native_prefill.cpp',root/'snapshot.h',root.parent/'glm53_gpu_v3/state_capture.inc',source/'include/llama.h',
        source/'ggml/include/ggml.h',source/'ggml/include/ggml-backend.h']
    before={str(p):sha(p) for p in inputs}
    libs={p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    if not all((runtime/name).is_file() for name in ('libllama.dylib','libggml-base.dylib')):
        raise ValueError('existing native runtime libraries are required')
    command=['clang++','-std=c++17','-O2','-Wno-deprecated-declarations']
    for directory in ('include','ggml/include','vendor'):
        command+=['-I',str(source/directory)]
    command += [str(root/'native_prefill.cpp'),'-L',str(runtime),'-Wl,-rpath,'+str(runtime),
        '-lllama','-lggml','-lggml-base','-o',str(output)]
    output.parent.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    record={'schema_version':1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'native_base_revision':revision,'inputs_sha256':before,'native_libraries_sha256':libs,
        'compiler':subprocess.check_output(['clang++','--version'],text=True),
        'scope':'new helper compiled and metadata-smoked; existing native libraries are not rebuilt or newly model-qualified'}
    if result.returncode==0:
        if before!={str(p):sha(p) for p in inputs}:
            raise ValueError('source changed during compilation')
        record['binary_sha256']=sha(output)
        record['parameter_smoke']={}
        for ub in (8,16,32,64,128):
            completed=subprocess.run([str(output),'--state-parameters',str(ub)],env=env,capture_output=True,text=True,check=True,timeout=5)
            state=json.loads(completed.stdout)
            if state['n_ubatch']!=ub or state['n_ctx']!=2048 or state['n_seq_max']!=1:
                raise ValueError('native parameter smoke failed')
            record['parameter_smoke'][str(ub)]=state
        invalid=subprocess.run([str(output),'--state-parameters','7'],env=env,capture_output=True,text=True,timeout=5)
        if invalid.returncode!=2:
            raise ValueError('invalid microbatch was not refused')
        record['invalid_batch_refused']=True
    output.with_suffix('.build.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(json.dumps({'binary_sha256':record['binary_sha256'],'smoke_profiles':5,'invalid_batch_refused':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
