"""Patch preflight and explicit opt-in tiny real mapped-buffer integration test."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).parent
PATCH=ROOT/'patches'/'llama-metal-mapped-cpu-read.patch'


def runtime_paths():
    work=ROOT.resolve().parents[2]
    source=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_SOURCE',work/'llama-v2'))
    libraries=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_LIBRARIES',work/'runtime-v2-build'/'bin'))
    if not (source/'ggml'/'src'/'ggml-backend.cpp').exists() or not (libraries/'libggml-base.dylib').exists():
        pytest.skip('pinned native source/libraries required')
    return source,libraries


def guard_source():
    added='\n'.join(line[1:] for line in PATCH.read_text().splitlines()
                    if line.startswith('+') and not line.startswith('+++'))
    begin=added.index('// AIRLLM_METAL_CPU_MAPPED_GUARD_BEGIN')
    end=added.index('// AIRLLM_METAL_CPU_MAPPED_GUARD_END')
    return added[begin:end]+'\n'


def compile_fixture(directory):
    if sys.platform!='darwin' or not shutil.which('clang++'):pytest.skip('Darwin native compiler required')
    source,libraries=runtime_paths()
    (directory/'mapped_cpu_guard.inc').write_text(guard_source())
    binary=directory/'mapped-cpu-fixture'
    subprocess.run(['clang++','-std=c++17','-O2','-I'+str(directory),'-I'+str(source/'include'),
        '-I'+str(source/'ggml'/'include'),str(ROOT/'mapped_cpu_fixture.cpp'),'-L'+str(libraries),
        '-lllama','-lggml','-lggml-base','-lggml-cpu','-Wl,-rpath,'+str(libraries),'-o',str(binary)],check=True)
    return binary


def test_guard_contract_and_patch_scope():
    text=PATCH.read_text();guard=guard_source()
    assert text.count('--- a/')==1 and '--- a/ggml/src/ggml-backend.cpp' in text
    for required in ['AIRLLM_METAL_CPU_MAPPED_WEIGHTS','AIRLLM_GLM_Q8_ONLY',
                     'GGML_BACKEND_BUFFER_USAGE_WEIGHTS','"MTL0_Mapped"','VM_PROT_WRITE',
                     'GGML_BACKEND_DEVICE_TYPE_CPU','GGML_TYPE_Q8_0','GGML_OP_NONE']:
        assert required in guard
    assert 'GGML_TENSOR_FLAG_INPUT' in guard and 'GGML_TENSOR_FLAG_COMPUTE' in guard
    assert 'ggml_backend_buft_is_host' not in guard


def test_patch_applies_without_mutating_source(tmp_path):
    source,_=runtime_paths()
    backend=source/'ggml'/'src'/'ggml-backend.cpp'
    live_before=backend.read_bytes()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert revision=='3f8af15a76ac8675812e856da45454bafc90f12c'
    pristine=subprocess.check_output(['git','-C',str(source),'show','HEAD:ggml/src/ggml-backend.cpp'])
    isolated=tmp_path/'isolated-source'
    target=isolated/'ggml'/'src'/'ggml-backend.cpp'
    target.parent.mkdir(parents=True)
    target.write_bytes(pristine)
    # The mapped-read patch has no content dependency on the cache-split patch;
    # copy the pinned file rather than reapplying onto an already-patched tree.
    subprocess.run(['git','-C',str(isolated),'apply','--check',str(PATCH.resolve())],check=True)
    assert target.read_bytes()==pristine
    assert backend.read_bytes()==live_before


def test_real_guard_fixture_compiles(tmp_path):
    assert compile_fixture(tmp_path).exists()


def test_tiny_real_metal_mapped_cpu_read(tmp_path):
    if os.environ.get('AIRLLM_TEST_METAL_MAPPED')!='1':
        pytest.skip('explicit tiny Metal-context fixture; keep unrelated hardware profiles isolated')
    binary=compile_fixture(tmp_path)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    if result.returncode==77:pytest.skip('Metal device unavailable')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['status']=='passed' and report['checks']>=20
    assert report['alias_depth_checks']==7
    assert report['sixteen_alias_leaf_flags_rejected'] and report['seventeen_aliases_rejected']
    assert report['base_is_original_mapping'] and report['cpu_direct_outputs_bit_identical']
    assert report['scheduler_copy_elimination']=='not_tested'
