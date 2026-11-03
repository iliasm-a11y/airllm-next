import json
from pathlib import Path
import shutil
import subprocess
import pytest

def test_timing_includes_actual_asynchronous_completion(tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler:pytest.skip('native C++ compiler unavailable')
    root=Path(__file__).parent;binary=tmp_path/'completion-fixture'
    subprocess.run([compiler,'-std=c++17','-O2','-pthread',str(root/'completion_fixture.cpp'),'-o',str(binary)],
                   capture_output=True,text=True,check=True,timeout=30)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=5)
    record=json.loads(result.stdout)
    assert record['completed_result']==42 and record['error_barrier_called']
