"""Opt-in small GGML CPU fixture; no model/GPU inference is launched."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).parent


def test_patch_does_not_change_arithmetic_or_public_plan_layout():
    text=(ROOT/'patches/llama-dense-lookahead.patch').read_text()
    assert 'llama-mmap.cpp' not in text and 'ggml-cpu.h' not in text
    assert 'ggml_cpu_dense_lookahead_notify' in text
    assert 'ggml-cpu-dense-lookahead.h' in text


def test_tiny_real_ggml_mmap_lifecycle(tmp_path):
    if os.environ.get('AIRLLM_TEST_DENSE_LOOKAHEAD')!='1':
        pytest.skip('explicit bounded CPU mmap/advice fixture')
    if sys.platform!='darwin' or not shutil.which('clang++'):pytest.skip('Darwin compiler required')
    work=ROOT.resolve().parents[2]
    source=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_SOURCE',work/'llama-v3'))
    libraries=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_LIBRARIES',work/'runtime-v3-build'/'bin'))
    if not (libraries/'libggml-cpu.dylib').exists():pytest.skip('pinned libraries required')
    binary=tmp_path/'dense-lookahead-fixture'
    subprocess.run(['clang++','-std=c++17','-O2','-I'+str(ROOT),'-I'+str(source/'ggml'/'include'),
                    str(ROOT/'dense_lookahead_fixture.cpp'),'-L'+str(libraries),'-lggml-cpu','-lggml-base',
                    '-Wl,-rpath,'+str(libraries),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    if os.environ.get('AIRLLM_DENSE_LOOKAHEAD_EVIDENCE'):
        path=Path(os.environ['AIRLLM_DENSE_LOOKAHEAD_EVIDENCE'])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['status']=='passed' and report['checks']>=25
    assert report['cpu_outputs_bit_identical'] and not report['gpu_initialized'] and not report['model_loaded']
