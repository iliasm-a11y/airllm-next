"""Metadata arithmetic + optional tiny real VM/GGML lock fixture; no model/GPU."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('dense_lock_selector',ROOT/'select_dense.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)


def test_full_page_budget_includes_gguf_edges_and_deduplicates_pages():
    page=16384;start=86720
    a,b=selector.page_cover(start,256*1024**2,page)
    assert a==81920 and b-a==256*1024**2+page
    one=selector.page_cover(start,256,page);two=selector.page_cover(start+512,128,page)
    spans=selector.union([('shard',*one),('shard',*two)])
    assert spans==[('shard',81920,98304)]
    spans=selector.union([('a',0,page),('a',page,2*page),('b',0,page)])
    assert spans==[('a',0,2*page),('b',0,page)]


def test_tiny_real_cpu_mapped_mlock_lifecycle(tmp_path):
    if os.environ.get('AIRLLM_TEST_DENSE_LOCK')!='1':pytest.skip('explicit real CPU VM lock fixture')
    if sys.platform!='darwin' or not shutil.which('clang++'):pytest.skip('Darwin compiler required')
    page=os.sysconf('SC_PAGESIZE');limit=resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    if limit!=resource.RLIM_INFINITY and limit<3*page:pytest.skip('existing memlock limit too small; never raised')
    work=ROOT.resolve().parents[2]
    source=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_SOURCE',work/'llama-v3'))
    libraries=Path(os.environ.get('AIRLLM_PROFILE_RUNTIME_LIBRARIES',work/'runtime-v3-build/bin'))
    if not (libraries/'libggml-base.dylib').is_file():pytest.skip('pinned libraries unavailable')
    with Path('/tmp/airllm-glm53-benchmark.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pytest.skip('shared physical experiment slot is busy')
        binary=tmp_path/'dense-lock-fixture'
        subprocess.run(['clang++','-std=c++17','-O2','-I'+str(ROOT),'-I'+str(source/'ggml/include'),
            str(ROOT/'dense_lock_fixture.cpp'),'-L'+str(libraries),'-lggml-cpu','-lggml-base',
            '-Wl,-rpath,'+str(libraries.resolve()),'-o',str(binary)],check=True)
        reports=[]
        for mode in ([],['--poison']):
            run=subprocess.run([str(binary),*mode],capture_output=True,text=True,timeout=30)
            assert run.returncode==0,run.stderr
            report=json.loads(run.stdout);assert report['status']=='passed' and report['rlimit_unchanged']
            reports.append({'report':report,'stdout':run.stdout,'stderr':run.stderr})
        if os.environ.get('AIRLLM_DENSE_LOCK_FIXTURE_EVIDENCE'):
            output=Path(os.environ['AIRLLM_DENSE_LOCK_FIXTURE_EVIDENCE']);output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(reports,indent=2)+'\n')
