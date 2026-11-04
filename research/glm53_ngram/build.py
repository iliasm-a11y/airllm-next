"""Compile the unmodified pinned n-gram implementation with a small GLM probe."""
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
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('pinned llama.cpp revision required')
    root = Path(__file__).resolve().parent
    inputs = [root/'native.cpp', root/'build.py', root/'log_bridge.cpp',
              root.parent/'glm53_gpu_v3/state_capture.inc',
              root.parent/'glm53_ultra_decode/timing.inc']
    inputs += [source/p for p in ('include/llama.h', 'common/ngram-map.cpp', 'common/ngram-map.h',
                                  'common/common.h', 'common/log.h')]
    before = {str(p): sha(p) for p in inputs}
    libraries = {p.name: sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    expected = json.loads((root.parent/'glm53_ultra_decode/evidence/resumed/build4.json').read_text())['native_libraries_sha256']
    if libraries != expected:
        raise ValueError('the fingerprinted V4 libraries are required; no inherited qualification for a rebuild')
    output.mkdir(parents=True)
    command = ['clang++', '-std=c++17', '-O2', '-Wno-deprecated-declarations',
               '-I', str(source/'include'), '-I', str(source/'ggml/include'),
               '-I', str(source/'common'), '-I', str(source/'vendor'),
               str(root/'native.cpp'), str(source/'common/ngram-map.cpp'), str(root/'log_bridge.cpp'),
               '-L', str(runtime), '-Wl,-rpath,'+str(runtime),
               '-lllama', '-lggml', '-lggml-base', '-lggml-cpu', '-o', str(output/'ngram-probe')]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DYLD_', 'AIRLLM_', 'GGML_', 'LLAMA_ARG_'))}
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=env)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'revision': REVISION,
              'inputs_sha256': before, 'native_libraries_sha256': libraries,
              'command': command, 'returncode': result.returncode, 'stderr': result.stderr,
              'compiler': subprocess.check_output(['clang++', '--version'], text=True),
              'scope': 'original common/ngram-map.cpp compiled unchanged; existing V4 model libraries unchanged'}
    report['inputs_unchanged'] = before == {str(p): sha(p) for p in inputs}
    report['libraries_unchanged'] = libraries == {p.name: sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}
    report['passed'] = result.returncode == 0 and report['inputs_unchanged'] and report['libraries_unchanged']
    if report['passed']:
        report['binaries_sha256'] = {'ngram-probe': sha(output/'ngram-probe')}
        smoke = subprocess.run([str(output/'ngram-probe'), '--parameters'], env=env, capture_output=True,
                               text=True, check=True, timeout=5)
        report['parameter_smoke'] = json.loads(smoke.stdout)
    (output/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError(result.stderr or 'build inputs changed')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'runtime', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.source, args.runtime, args.output)['binaries_sha256']))
