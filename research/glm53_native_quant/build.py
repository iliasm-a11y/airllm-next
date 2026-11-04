"""Compile only a small adapter against the fingerprinted V4 libraries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from research.glm53_ultra_decode.build import REVISION, sha


def build(source, runtime, output):
    source, runtime, output = (Path(p).resolve() for p in (source, runtime, output))
    if output.exists():
        raise FileExistsError(output)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip() != REVISION:
        raise ValueError('pinned llama.cpp revision required')
    root = Path(__file__).parent
    files = [root/'native.cpp', root/'build.py'] + [source/'ggml/include'/n for n in
        ('ggml.h','ggml-cpu.h','ggml-metal.h','ggml-backend.h','ggml-alloc.h')]
    before = {str(p):sha(p) for p in files}
    libraries = {p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    expected = json.loads((root.parent/'glm53_ultra_decode/evidence/resumed/build4.json').read_text())['native_libraries_sha256']
    if libraries != expected:
        raise ValueError('requires the complete fingerprinted V4 runtime, not an arbitrary rebuilt binary')
    output.mkdir(parents=True)
    command = ['clang++','-std=c++17','-O2','-dynamiclib','-I',str(source/'ggml/include'),
        str(root/'native.cpp'),'-L',str(runtime),'-Wl,-rpath,'+str(runtime),
        '-lggml-metal','-lggml-cpu','-lggml-base','-o',str(output/'native.dylib')]
    env = {k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','AIRLLM_','GGML_','LLAMA_ARG_'))}
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
    report = {'utc':datetime.now(timezone.utc).isoformat(),'revision':REVISION,'command':command,
        'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'inputs_sha256':before,'native_libraries_sha256':libraries,
        'compiler':subprocess.check_output(['clang++','--version'],text=True),
        'scope':'component adapter only; no native model library rebuilt, no inherited model qualification'}
    report['inputs_unchanged'] = before == {str(p):sha(p) for p in files}
    report['libraries_unchanged'] = libraries == {p.name:sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    report['passed'] = result.returncode == 0 and report['inputs_unchanged'] and report['libraries_unchanged']
    if not result.returncode:
        report['binaries_sha256'] = {'native.dylib':sha(output/'native.dylib')}
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError(result.stderr or 'inputs changed during build')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','runtime','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    print(json.dumps(build(args.source,args.runtime,args.output)['binaries_sha256']))
