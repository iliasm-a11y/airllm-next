"""Finder metadata is not evidence; all fixtures and floats are synthetic."""
import errno
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'air_llm'))
import m5_manifest_recovery as recovery
import m5_validation as campaign
import verify_m5_validation as checker


def finder(path,payload=b'finder fixture'):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(checker.FINDER_METADATA_MAGIC+payload)
    return path


@pytest.fixture
def indexed(tmp_path):
    folder=tmp_path/'campaign';folder.mkdir()
    config=dict(origin=str(folder),experiments=[],tasks_by_path={})
    campaign.save(folder/'config.json',config)
    state=dict(status='prepared_no_worker_launched',sessions=[],jobs={},experiment_results={},
               config_sha256=campaign.digest(folder/'config.json'))
    campaign.save(folder/'state.json',state)
    (folder/'worker.step-000.f32').write_bytes(b'\x00\x00\x80\x3f')
    campaign.write_manifest(folder)
    return folder,config,state


def check(operation,indexed):
    folder,config,state=indexed
    if operation=='write':
        return campaign.write_manifest(folder)
    if operation=='verify':
        return checker.verify_manifest(folder)
    return recovery.inspect_index(folder,config,state)


def test_only_finder_databases_are_omitted_at_collection_and_mutable_afterward(indexed):
    folder,config,state=indexed
    root=finder(folder/'.DS_Store')
    nested=finder(folder/'sessions/.DS_Store')
    campaign.write_manifest(folder)
    index=(folder/'manifest.json').read_bytes()
    entries=json.loads(index)['files']
    assert set(entries)=={'config.json','state.json','worker.step-000.f32'}
    root.write_bytes(checker.FINDER_METADATA_MAGIC+b'changed Finder body')
    nested.unlink()  # Finder may also delete its optional database.
    finder(folder/'new-directory/.DS_Store',b'new Finder body')
    assert checker.verify_manifest(folder)==3
    assert recovery.inspect_index(folder,config,state)['needed'] is False
    assert (folder/'manifest.json').read_bytes()==index


def test_deferred_recovery_ignores_only_finder_metadata_without_promoting_evidence(indexed):
    folder,config,state=indexed
    state['status']='deferred_or_interrupted'
    campaign.save(folder/'state.json',state)
    finder(folder/'sessions/.DS_Store')
    result=recovery.inspect_index(folder,config,state)
    assert result['needed'] is True and result['added_files']==[]
    campaign.write_manifest(folder)
    assert checker.verify_manifest(folder)==3
    assert state['jobs']=={} and state['experiment_results']=={}


@pytest.mark.parametrize('operation',['verify','recover'])
def test_scientific_bytes_stay_immutable_despite_metadata_changes(indexed,operation):
    folder,_,_=indexed
    finder(folder/'.DS_Store',b'updated metadata')
    (folder/'worker.step-000.f32').write_bytes(b'\x00\x00\x00\x40')
    with pytest.raises(checker.RejectedEvidence,match='digest or length'):
        check(operation,indexed)


@pytest.mark.parametrize('operation',['write','verify','recover'])
@pytest.mark.parametrize('kind',['symlink','dangling-symlink','directory','fifo','hardlink','wrong-magic'])
def test_unsafe_metadata_entries_are_rejected_before_exclusion(indexed,operation,kind):
    folder,_,_=indexed
    path=folder/'.DS_Store'
    if kind=='symlink':
        path.symlink_to(folder/'worker.step-000.f32')
    elif kind=='dangling-symlink':
        path.symlink_to(folder/'absent')
    elif kind=='directory':
        path.mkdir()
    elif kind=='fifo':
        os.mkfifo(path)
    elif kind=='hardlink':
        # Even an inode carrying the magic cannot alias a retained evidence file.
        target=finder(folder/'scientific-proof')
        campaign.write_manifest(folder)
        os.link(target,path)
    else:
        path.write_bytes(b'scientific bytes, not a Finder database')
    with pytest.raises(checker.RejectedEvidence,match='Finder'):
        check(operation,indexed)


@pytest.mark.parametrize('operation',['write','verify','recover'])
@pytest.mark.parametrize('relative',['.DS_Store','sessions/.DS_Store'])
def test_existing_declared_evidence_cannot_be_dropped_or_relabelled(indexed,operation,relative):
    folder,_,_=indexed
    metadata=finder(folder/relative)
    path=folder/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest['files'][relative]=dict(bytes=metadata.stat().st_size,sha256=checker.digest(metadata))
    campaign.save(path,manifest)
    before=path.read_bytes()
    with pytest.raises(checker.RejectedEvidence,match='declared evidence'):
        check(operation,indexed)
    assert path.read_bytes()==before


@pytest.mark.parametrize('name',['.DS_Store.suffix','._DS_Store','.ds_store','.hidden-proof'])
@pytest.mark.parametrize('operation',['verify','recover'])
def test_other_hidden_names_remain_unlisted_evidence(indexed,name,operation):
    folder,_,_=indexed
    finder(folder/name)
    with pytest.raises(checker.RejectedEvidence,match='unlisted'):
        check(operation,indexed)


def test_other_hidden_names_are_hashed_when_collected(indexed):
    folder,_,_=indexed
    finder(folder/'._DS_Store')
    campaign.write_manifest(folder)
    assert '._DS_Store' in json.loads((folder/'manifest.json').read_text())['files']
    assert checker.verify_manifest(folder)==4


def test_manifest_extra_ignore_claim_never_waives_scientific_hash(indexed):
    folder,_,_=indexed
    path=folder/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest['ignore']=['worker.step-000.f32']
    campaign.save(path,manifest)
    (folder/'worker.step-000.f32').write_bytes(b'changed scientific bytes')
    with pytest.raises(checker.RejectedEvidence,match='digest or length'):
        checker.verify_manifest(folder)


def test_signature_read_is_bounded_nonfollowing_nonblocking_and_descriptor_is_closed(tmp_path,monkeypatch):
    path=finder(tmp_path/'.DS_Store',b'body'*1024)
    calls=[];real_open=os.open;real_read=os.read;real_close=os.close
    def opened(value,flags):
        calls.append(('open',value,flags));return real_open(value,flags)
    def read(fd,count):
        calls.append(('read',fd,count));return real_read(fd,count)
    def close(fd):
        calls.append(('close',fd));return real_close(fd)
    monkeypatch.setattr(checker.os,'open',opened)
    monkeypatch.setattr(checker.os,'read',read)
    monkeypatch.setattr(checker.os,'close',close)
    assert checker.finder_metadata_ignored(path)
    assert calls[0][2] & os.O_NOFOLLOW and calls[0][2] & os.O_NONBLOCK
    assert [call[2] for call in calls if call[0]=='read']==[8]
    assert calls[-1]==('close',calls[1][1])


@pytest.mark.parametrize('phase',['first-lstat','open','fstat','read','second-lstat'])
@pytest.mark.parametrize('kind',['permission','io','programming'])
def test_access_io_and_programming_errors_are_never_silently_ignored(tmp_path,monkeypatch,phase,kind):
    path=finder(tmp_path/'.DS_Store')
    error={'permission':PermissionError('denied'),
           'io':OSError(errno.EIO,'injected I/O failure'),
           'programming':RuntimeError('injected programming error')}[kind]
    real_lstat=Path.lstat;calls=0
    def lstat(value,*args,**kwargs):
        nonlocal calls
        if value==path:
            calls+=1
            if calls==({'first-lstat':1,'second-lstat':2}.get(phase)):
                raise error
        return real_lstat(value,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',lstat)
    if phase in ('open','fstat','read'):
        def fail(*args,**kwargs):
            raise error
        monkeypatch.setattr(checker.os,phase,fail)
    with pytest.raises(type(error)) as raised:
        checker.finder_metadata_ignored(path)
    assert raised.value is error


@pytest.mark.parametrize('phase',['before-open','after-read'])
def test_metadata_inode_replacement_is_refused(tmp_path,monkeypatch,phase):
    path=finder(tmp_path/'.DS_Store')
    replacement=finder(tmp_path/'replacement')
    real_open=os.open;real_read=os.read
    def opened(value,flags):
        if phase=='before-open':
            replacement.replace(path)
        return real_open(value,flags)
    def read(fd,count):
        data=real_read(fd,count)
        if phase=='after-read':
            replacement.replace(path)
        return data
    monkeypatch.setattr(checker.os,'open',opened)
    monkeypatch.setattr(checker.os,'read',read)
    with pytest.raises(checker.RejectedEvidence,match='identity changed'):
        checker.finder_metadata_ignored(path)


def test_replacement_by_fifo_between_lstat_and_open_cannot_block(tmp_path,monkeypatch):
    path=finder(tmp_path/'.DS_Store')
    real_open=os.open
    def opened(value,flags):
        path.unlink();os.mkfifo(path)
        assert flags & os.O_NONBLOCK
        return real_open(value,flags)
    monkeypatch.setattr(checker.os,'open',opened)
    with pytest.raises(checker.RejectedEvidence,match='identity changed'):
        checker.finder_metadata_ignored(path)
