"""Tiny CPU fixtures check integrity output protection, not SSD throughput."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'benchmarks/hash_checkpoint_guarded.py'
spec=importlib.util.spec_from_file_location('hash_checkpoint_guarded',SCRIPT)
guarded_hash=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guarded_hash)


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
        bytes=len(data),stable_file=True,
        hash_io_policy='darwin_f_nocache' if sys.platform=='darwin' else 'ordinary_os_cache',
        hash_io_policy_applied=True,hash_read_buffer_bytes=1<<20,
        hash_io_scope='checkpoint_hash_descriptor_only')
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


def test_darwin_applies_existing_nocache_to_the_hash_descriptor_only(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'air_llm'))
    from airllm.nextgen import physical_io
    model=tmp_path/'model'
    data=bytes(range(256))*9000+b'final bytes past two blocks'
    model.write_bytes(data)
    calls=[]
    def nocache(fd):
        calls.append((fd,os.fstat(fd).st_ino,os.lseek(fd,0,os.SEEK_CUR)))
    monkeypatch.setattr(guarded_hash.sys,'platform','darwin')
    monkeypatch.setattr(physical_io,'set_fd_nocache',nocache)
    with model.open('rb') as unrelated:
        result=guarded_hash.checkpoint_hash(model)
        assert calls==[(calls[0][0],model.stat().st_ino,0)]
        assert calls[0][0]!=unrelated.fileno()
        assert unrelated.tell()==0
    assert result['sha256']==hashlib.sha256(data).hexdigest()
    assert result['bytes']==len(data) and result['stable_file']
    assert result['hash_io_policy']=='darwin_f_nocache'
    assert model.read_bytes()==data


def test_nondarwin_keeps_explicit_ordinary_policy_without_nocache_request(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'air_llm'))
    from airllm.nextgen import physical_io
    model=tmp_path/'model';model.write_bytes(b'ordinary portable hash')
    monkeypatch.setattr(guarded_hash.sys,'platform','linux')
    monkeypatch.setattr(physical_io,'set_fd_nocache',lambda fd:pytest.fail('Darwin policy requested on another OS'))
    result=guarded_hash.checkpoint_hash(model)
    assert result['hash_io_policy']=='ordinary_os_cache'
    assert result['sha256']==hashlib.sha256(model.read_bytes()).hexdigest()


def test_darwin_os_failure_has_no_fallback_or_checkpoint_or_evidence_mutation(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'air_llm'))
    from airllm.nextgen import physical_io
    model=tmp_path/'model';model.write_bytes(b'original checkpoint')
    previous=tmp_path/'previous.json';previous.write_bytes(b'sealed previous proof')
    output=tmp_path/'new.json'
    def refused(fd):
        raise OSError('injected F_NOCACHE refusal')
    monkeypatch.setattr(physical_io,'set_fd_nocache',refused)
    monkeypatch.setattr(guarded_hash.sys,'platform','darwin')
    monkeypatch.setattr(guarded_hash.sys,'argv',[str(SCRIPT),'--model',str(model),'--output',str(output),'--worker'])
    with pytest.raises(OSError,match='F_NOCACHE refusal'):
        guarded_hash.main()
    assert model.read_bytes()==b'original checkpoint'
    assert previous.read_bytes()==b'sealed previous proof'
    assert not output.exists()
