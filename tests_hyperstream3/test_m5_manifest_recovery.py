"""Synthetic index/lifecycle regressions; no model, GPU or checkpoint reads."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import struct
import sys

import psutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'air_llm'))
import m5_manifest_recovery as recovery
import m5_validation as campaign
import m5_autonomous_supervisor as supervisor
import verify_m5_validation as checker


def stale_campaign(tmp_path,status='deferred_or_interrupted',max_new_tokens=2,speculative=False):
    folder=tmp_path/'campaign';folder.mkdir()
    task='fixture-task.json'
    flags=['--numerical-evidence']+(['--speculative-depth','8'] if speculative else [])
    ids=['baseline-p00','candidate-p00','candidate-p01','baseline-p01']
    plan=[dict(id=label,args=list(flags),tasks=task,engine_root=str(ROOT)) for label in ids]
    config=dict(origin=str(folder),experiments=[dict(name='control',plan=plan)],
                tasks_by_path={task:[dict(id='cpu',prompt='fixture',max_new_tokens=max_new_tokens)]},
                engine_roots=dict(candidate=str(ROOT)),protocol_sha256={
                    str(ROOT/name):campaign.digest(ROOT/name) for name in (
                        'benchmarks/m5_validation.py','benchmarks/verify_m5_validation.py',
                        'benchmarks/validate_m5_kernels.py','benchmarks/m5_manifest_recovery.py')})
    campaign.save(folder/'config.json',config)
    before=folder/'sessions/0000/hash-before/checkpoint.json'
    before.parent.mkdir(parents=True);before.write_text('{"fixture":"hash-before"}')
    session=dict(index=0,status='started',hash_before=str(before),integrity_blocks=[])
    state=dict(config_sha256=campaign.digest(folder/'config.json'),status='running',
               sessions=[session],jobs={},experiment_results={})
    campaign.save(folder/'state.json',state)
    campaign.write_manifest(folder)
    # Everything below simulates bytes/state persisted after the old index.
    for label in ids:
        output=folder/'sessions/0000/control'/label/'worker.json'
        output.parent.mkdir(parents=True)
        output.write_text('{"fixture":"completed recording"}')
        (output.parent/'worker.step-000.f32').write_bytes(struct.pack('<f',1.))
        (output.parent/'owned-processes.json').write_text('{"processes":[]}')
        state['jobs']['control/'+label]=dict(output=str(output.relative_to(folder)),session=0,
            integrity_block=0,started_unix=1.,ended_unix=2.)
    after=folder/'sessions/0000/integrity-0000/checkpoint.json'
    after.parent.mkdir();after.write_text('{"fixture":"hash-after"}')
    session['integrity_blocks']=[dict(index=0,status='sealed',jobs=list(state['jobs']),
        hash_before=str(before),hash_after=str(after),hash_before_completed_unix=0.,hash_after_started_unix=3.)]
    state['experiment_results']['control']=dict(status='accepted',fixture_only=True)
    state['status']=status
    if status.startswith('completed'):
        session['status']='completed'
        final=folder/'sessions/0000/hash-after/checkpoint.json'
        final.parent.mkdir();final.write_text('{"fixture":"final hash"}')
        session['hash_after']=str(final)
    campaign.save(folder/'state.json',state)
    return folder,config,state


def fake_independent_checks(monkeypatch,state,events):
    monkeypatch.setattr(recovery,'verify_hash',lambda *args:events.append('verify-hash'))
    monkeypatch.setattr(recovery,'validate_job',lambda *args:events.append('validate-job'))
    monkeypatch.setattr(recovery,'summarize_experiment',lambda *args:deepcopy(state['experiment_results']['control']))


@pytest.mark.parametrize('status',['running','deferred_or_interrupted','partial_selected_experiments_completed',
                                 'completed','completed_with_exclusions','completed_evidence_with_rejections'])
def test_known_stale_states_have_read_only_bounded_recovery_admission(tmp_path,status):
    folder,config,state=stale_campaign(tmp_path,status)
    before={p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    result=recovery.inspect_index(folder,config,state)
    assert result['needed'] and result['added_files']
    assert {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}==before


def test_supervisor_preparation_accepts_stale_deferred_index_without_rewriting(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path)
    monkeypatch.setattr(supervisor,'validate_config',lambda value:None)
    checked=[]
    monkeypatch.setattr(supervisor,'unchanged',lambda value:checked.append(value))
    before=(folder/'manifest.json').read_bytes()
    assert supervisor.preparation(folder)==(config,state)
    assert (folder/'manifest.json').read_bytes()==before
    assert checked==[config]


def test_modified_old_immutable_file_and_unknown_added_paths_remain_fatal(tmp_path):
    folder,config,state=stale_campaign(tmp_path)
    before=(folder/'manifest.json').read_bytes()
    (folder/'intruder.txt').write_text('unowned')
    with pytest.raises(checker.RejectedEvidence,match='outside recorded'):
        recovery.inspect_index(folder,config,state)
    (folder/'intruder.txt').unlink()
    original=folder/'sessions/0000/hash-before/checkpoint.json'
    original.write_text('changed sealed evidence')
    with pytest.raises(checker.RejectedEvidence,match='digest or length'):
        recovery.inspect_index(folder,config,state)
    assert (folder/'manifest.json').read_bytes()==before


def test_frozen_plan_and_exact_artifact_bounds_prevent_arbitrary_sidecars(tmp_path):
    folder,config,state=stale_campaign(tmp_path,max_new_tokens=384,speculative=True)
    path=folder/'sessions/0000/control/candidate-p00/worker.step-1000.f32'
    path.write_bytes(struct.pack('<f',1.))
    assert recovery.inspect_index(folder,config,state)['needed']
    path.with_name('worker.step-9999.f32').write_bytes(b'unowned')
    with pytest.raises(checker.RejectedEvidence,match='outside recorded'):
        recovery.inspect_index(folder,config,state)


def test_serial_recording_cannot_claim_speculative_extra_artifacts(tmp_path):
    folder,config,state=stale_campaign(tmp_path,max_new_tokens=2)
    (folder/'sessions/0000/control/candidate-p00/worker.step-002.f32').write_bytes(b'xxxx')
    with pytest.raises(checker.RejectedEvidence,match='outside recorded'):
        recovery.inspect_index(folder,config,state)


@pytest.mark.parametrize('name',['manifest.json','.runner.lock'])
def test_reserved_names_cannot_hide_unlisted_nested_files(tmp_path,name):
    folder,config,state=stale_campaign(tmp_path)
    path=folder/'unowned'/name;path.parent.mkdir();path.write_text('hidden unowned bytes')
    with pytest.raises(checker.RejectedEvidence,match='Reserved index/lock'):
        recovery.inspect_index(folder,config,state)


@pytest.mark.parametrize('mode',['unknown-state','nonfinite-state','symlink','wrong-job-output'])
def test_recovery_does_not_hide_malformed_or_unowned_data(tmp_path,mode):
    folder,config,state=stale_campaign(tmp_path)
    if mode=='unknown-state':
        state['status']='invented';campaign.save(folder/'state.json',state)
    elif mode=='nonfinite-state':
        (folder/'state.json').write_text(json.dumps(dict(state,bad=float('nan'))))
    elif mode=='symlink':
        (folder/'link').symlink_to(folder/'config.json')
    else:
        state['jobs']['control/baseline-p00']['output']='sessions/0000/other/baseline-p00/worker.json'
        campaign.save(folder/'state.json',state)
    with pytest.raises((checker.RejectedEvidence,ValueError,json.JSONDecodeError)):
        recovery.inspect_index(folder,config,state)


def test_completed_resume_cleans_leases_then_revalidates_reindexes_and_strictly_verifies(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path,'completed')
    state_raw=(folder/'state.json').read_bytes();config_raw=(folder/'config.json').read_bytes()
    old=json.loads((folder/'manifest.json').read_text())
    events=[]
    monkeypatch.setattr(checker,'validate_config',lambda value:None)
    monkeypatch.setattr(campaign,'unchanged',lambda value:events.append('inputs'))
    monkeypatch.setattr(campaign,'owned_cleanup',lambda values:events.append('cleanup'))
    monkeypatch.setattr(campaign,'require_leases_quiescent',lambda values:events.append('quiescent'))
    fake_independent_checks(monkeypatch,state,events)
    original_write=campaign.write_manifest
    def write(path):
        events.append('reindex');original_write(path)
    monkeypatch.setattr(campaign,'write_manifest',write)
    def completed(path):
        events.append('completed-verify')
        assert checker.verify_manifest(path)>len(old['files'])
        return dict(status='fixture-verified')
    monkeypatch.setattr(checker,'verify',completed)
    result=campaign.run(Namespace(campaign=folder,resource_wait_seconds=0.,experiment=None))
    assert result['status']=='fixture-verified'
    assert events.index('cleanup')<events.index('quiescent')<events.index('validate-job')<events.index('reindex')<events.index('completed-verify')
    assert (folder/'state.json').read_bytes()==state_raw and (folder/'config.json').read_bytes()==config_raw
    new=json.loads((folder/'manifest.json').read_text())
    assert all(new['files'][name]==entry for name,entry in old['files'].items() if name!='state.json')


def test_source_continuity_failure_prevents_cleanup_and_reindex(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path,'completed')
    before=(folder/'manifest.json').read_bytes()
    monkeypatch.setattr(checker,'validate_config',lambda value:None)
    def changed(value):
        raise RuntimeError('Frozen code changed')
    monkeypatch.setattr(campaign,'unchanged',changed)
    monkeypatch.setattr(campaign,'owned_cleanup',lambda *args:pytest.fail('cleanup before source validation'))
    with pytest.raises(RuntimeError,match='Frozen code changed'):
        campaign.run(Namespace(campaign=folder,resource_wait_seconds=0.,experiment=None))
    assert (folder/'manifest.json').read_bytes()==before


def test_independent_validation_failure_keeps_old_index_and_accepted_state_unchanged(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path,'completed')
    before=(folder/'manifest.json').read_bytes();state_before=(folder/'state.json').read_bytes()
    monkeypatch.setattr(checker,'validate_config',lambda value:None)
    monkeypatch.setattr(campaign,'unchanged',lambda value:None)
    monkeypatch.setattr(campaign,'owned_cleanup',lambda values:None)
    monkeypatch.setattr(recovery,'verify_hash',lambda *args:None)
    def nonfinite(*args):
        raise checker.RejectedEvidence('Non-finite logits in sealed recording')
    monkeypatch.setattr(recovery,'validate_job',nonfinite)
    with pytest.raises(checker.RejectedEvidence,match='Non-finite'):
        campaign.run(Namespace(campaign=folder,resource_wait_seconds=0.,experiment=None))
    assert (folder/'manifest.json').read_bytes()==before
    assert (folder/'state.json').read_bytes()==state_before


def test_index_recovery_never_promotes_unsealed_worker_or_changes_rejected_result(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path)
    key='control/baseline-p01'
    del state['jobs'][key]['integrity_block']
    state['sessions'][0]['integrity_blocks'][0]['jobs'].remove(key)
    state['experiment_results']={}
    campaign.save(folder/'state.json',state)
    before=(folder/'state.json').read_bytes();events=[]
    fake_independent_checks(monkeypatch,state,events)
    recovery.inspect_index(folder,config,state)
    recovery.validate_recovery_records(folder,config,state)
    assert events.count('validate-job')==3
    assert (folder/'state.json').read_bytes()==before
    assert 'integrity_block' not in state['jobs'][key]


def test_changed_recalculated_outcome_is_not_accepted_by_reindex(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path)
    events=[];fake_independent_checks(monkeypatch,state,events)
    monkeypatch.setattr(recovery,'summarize_experiment',lambda *args:dict(status='accepted',fixture_only=False))
    with pytest.raises(checker.RejectedEvidence,match='independent recalculation'):
        recovery.validate_recovery_records(folder,config,state)


def test_sealed_job_outside_hash_time_bracket_is_fatal_before_indexing(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path)
    events=[];fake_independent_checks(monkeypatch,state,events)
    state['jobs']['control/baseline-p00']['started_unix']=-1.
    with pytest.raises(checker.RejectedEvidence,match='outside its recorded hash bracket'):
        recovery.validate_recovery_records(folder,config,state)


def test_cleanup_uncertainty_cannot_reindex_an_owned_writer(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path,'completed')
    lease=folder/'sessions/0000/control/baseline-p00/owned-processes.json'
    lease.write_text('{"processes":[{"pid":12345,"created":10.0}]}')
    before=(folder/'manifest.json').read_bytes()
    monkeypatch.setattr(checker,'validate_config',lambda value:None)
    monkeypatch.setattr(campaign,'unchanged',lambda value:None)
    monkeypatch.setattr(campaign,'owned_cleanup',lambda values:None)
    monkeypatch.setattr(psutil,'Process',lambda pid:(_ for _ in ()).throw(psutil.AccessDenied(pid)))
    with pytest.raises(RuntimeError,match='identity is unavailable'):
        campaign.run(Namespace(campaign=folder,resource_wait_seconds=0.,experiment=None))
    assert (folder/'manifest.json').read_bytes()==before


def test_current_index_does_not_bypass_quiescence_before_completed_verify_or_resume(tmp_path,monkeypatch):
    folder,config,state=stale_campaign(tmp_path,'completed')
    lease=folder/'sessions/0000/control/baseline-p00/owned-processes.json'
    lease.write_text('{"processes":[{"pid":12345,"created":10.0}]}')
    campaign.write_manifest(folder)
    assert not recovery.inspect_index(folder,config,state)['needed']
    monkeypatch.setattr(checker,'validate_config',lambda value:None)
    monkeypatch.setattr(campaign,'unchanged',lambda value:None)
    monkeypatch.setattr(campaign,'owned_cleanup',lambda values:None)
    monkeypatch.setattr(psutil,'Process',lambda pid:(_ for _ in ()).throw(psutil.AccessDenied(pid)))
    monkeypatch.setattr(checker,'verify',lambda *args:pytest.fail('Live/unknown writer reached completed verification'))
    monkeypatch.setattr(campaign,'run_child',lambda *args:pytest.fail('Live/unknown writer reached worker launch'))
    with pytest.raises(RuntimeError,match='identity is unavailable'):
        campaign.run(Namespace(campaign=folder,resource_wait_seconds=0.,experiment=None))


@pytest.mark.parametrize('identities',[{},[dict(pid=0,created=1.)],[dict(pid=True,created=1.)],
    [dict(pid=123,created=float('nan'))],[dict(pid=123,created='1')],['malformed']])
def test_malformed_lease_is_rejected_before_any_process_lookup_or_signal(monkeypatch,identities):
    monkeypatch.setattr(psutil,'Process',lambda *args:pytest.fail('Unvalidated identity reached process control'))
    with pytest.raises(ValueError,match='Invalid saved process identit'):
        campaign.owned_cleanup(identities)


def test_failed_preparation_has_an_atomic_startup_journal_and_is_never_resource75(tmp_path,monkeypatch):
    folder=tmp_path/'campaign';folder.mkdir()
    output=tmp_path/'supervisor'
    def rejected(path):
        raise checker.RejectedEvidence('immutable evidence was changed')
    monkeypatch.setattr(supervisor,'preparation',rejected)
    with pytest.raises(checker.RejectedEvidence,match='immutable'):
        supervisor.run(Namespace(campaign=folder,output_dir=output,max_wait_seconds=1.,max_runtime_seconds=1.))
    journals=list(output.glob('startup-failure-*.json'))
    assert len(journals)==1 and not (output/'state.json').exists()
    record=json.loads(journals[0].read_text())
    assert record['exit_code']==1 and record['status']=='startup_failed'
