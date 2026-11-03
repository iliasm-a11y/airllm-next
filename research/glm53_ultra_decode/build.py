"""Build bounded decode components against the existing pinned native runtime."""
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

def build(source, runtime, output):
    source,runtime,output=map(lambda p:Path(p).resolve(),(source,runtime,output))
    if output.exists():raise FileExistsError(output)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=REVISION:
        raise ValueError('pinned llama.cpp revision required')
    root=Path(__file__).resolve().parent
    inputs=[root/n for n in ('native_component.cpp','decode_timer.mm','decode.metal','build.py',
        'native_decode.cpp','timing.inc','residency.inc')]
    inputs += [root.parent/'glm53_gpu_v3'/n for n in ('state_capture.inc','source_capture.inc','self_vm.inc')]
    inputs += [source/n for n in ('ggml/include/ggml.h','ggml/include/ggml-cpu.h','ggml/include/ggml-backend.h')]
    before={str(p):sha(p) for p in inputs}
    libraries={p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    if not all((runtime/n).is_file() for n in ('libggml-cpu.dylib','libggml-base.dylib')):
        raise ValueError('existing native CPU/base libraries required')
    output.mkdir(parents=True)
    common=['clang++','-std=c++17','-O2','-I',str(source/'ggml/include'),'-L',str(runtime),
            '-Wl,-rpath,'+str(runtime),'-lggml-cpu','-lggml-base']
    commands=[common+['-dynamiclib',str(root/'native_component.cpp'),'-o',str(output/'component.dylib')],
              common+['-fobjc-arc','-framework','Foundation','-framework','Metal',
                      str(root/'decode_timer.mm'),'-o',str(output/'decode-timer')],
              common+['-Wno-deprecated-declarations','-I',str(source/'include'),'-I',str(source/'vendor'),
                      str(root/'native_decode.cpp'),'-lllama','-lggml','-o',str(output/'native-decode')]]
    env={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    report={'schema':1,'utc':datetime.now(timezone.utc).isoformat(),'revision':REVISION,
            'inputs_sha256':before,'native_libraries_sha256':libraries,'steps':[],
            'compiler':subprocess.check_output(['clang++','--version'],text=True),
            'scope':'new component binaries; no native model library rebuilt or requalified'}
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True,timeout=60,env=env)
        report['steps'].append({'command':command,'returncode':result.returncode,
                                'stdout':result.stdout,'stderr':result.stderr})
        (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        if result.returncode:raise RuntimeError(result.stderr)
    if before!={str(p):sha(p) for p in inputs}:raise ValueError('build inputs changed')
    if libraries!={p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}:
        raise ValueError('runtime changed during build')
    report['binaries_sha256']={p.name:sha(p) for p in (output/'component.dylib',output/'decode-timer',output/'native-decode')}
    smoke=subprocess.run([str(output/'native-decode'),'--parameters'],capture_output=True,text=True,check=True,timeout=5,env=env)
    report['parameter_smoke']=json.loads(smoke.stdout)
    report['passed']=True
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','runtime','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=build(a.source,a.runtime,a.output);print(json.dumps(r['binaries_sha256']))
