"""Tiny CPU fixtures check integrity output protection, not SSD throughput."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'benchmarks/hash_checkpoint_guarded.py'


def run(model,output):
    return subprocess.run([sys.executable,str(SCRIPT),'--model',str(model),
        '--output',str(output),'--worker'],capture_output=True,text=True)


def test_worker_hashes_entire_file_and_accepts_own_supervisor_log(tmp_path):
    data=bytes(range(256))*5000
    model=tmp_path/'model';model.write_bytes(data)
    output=tmp_path/'integrity.json'
    output.with_suffix('.log').write_text('active supervisor log')
    assert run(model,output).returncode==0
    assert json.loads(output.read_text())==dict(sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),stable_file=True)
    assert model.read_bytes()==data


def test_worker_rejects_existing_output_and_checkpoint_alias(tmp_path):
    model=tmp_path/'model';model.write_bytes(b'original checkpoint')
    output=tmp_path/'report';output.write_bytes(b'previous evidence')
    assert run(model,output).returncode!=0
    assert output.read_bytes()==b'previous evidence'
    assert run(model,model).returncode!=0
    alias=tmp_path/'alias';alias.symlink_to(model)
    assert run(model,alias).returncode!=0
    assert model.read_bytes()==b'original checkpoint'
