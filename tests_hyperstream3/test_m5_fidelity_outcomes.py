"""Fabricated proof controls only; no MLX/model or performance claims."""
from argparse import Namespace
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'air_llm'))
import m5_validation as bench
import verify_m5_validation as checker
from verify_gguf_speculation import RejectedEvidence as RecordingRejected
from airllm.nextgen.experiment_guard import ExperimentLimits


def experiment_fixture(tmp_path,monkeypatch):
    spec = importlib.util.spec_from_file_location('m5_fabricated_recordings',
        ROOT/'tests_next/test_verify_gguf_speculation.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    jobs,plan = {},[]
    for pair in range(2):
        folder = tmp_path/f'p{pair}'
        folder.mkdir()
        serial,candidate = fixtures.synthetic(folder)
        for path in (serial,candidate):
            report = json.loads(path.read_text())
            report.update(worker_wall_seconds=1.,initialization_seconds=.1)
            for turn in report['turns']:
                turn.update(wall_seconds=.9,ttft_seconds=.1,logical_read_bytes=1024,read_calls=2,
                            decode_tokens_per_second=5.,cpu_seconds=dict(user=.1,system=.1))
                for step in turn['step_checks']:
                    step['forward_seconds'] = .1
            bench.save(path,report)
        for variant in (('baseline','candidate') if pair%2 == 0 else ('candidate','baseline')):
            ident = f'{variant}-p{pair:02d}'
            path = serial if variant == 'baseline' else candidate
            plan.append(dict(id=ident,args=[]))
            jobs['control/'+ident] = dict(output=str(path.relative_to(tmp_path)),
                controller_wall_seconds=1.,session=0,integrity_block=0,started_unix=1.,ended_unix=2.)
    monkeypatch.setattr(checker,'validate_job',lambda folder,config,job,record:
                        json.loads((folder/record['output']).read_text()))
    return dict(pairs=2,numerical_tolerances=dict(atol=1e-4,rtol=1e-4)),dict(
        name='control',plan=plan,tasks_kind='fabricated',requires_prior_pass=[]),jobs


def change_route(path):
    report = json.loads(path.read_text())
    report['turns'][0]['step_checks'][0]['routing_by_layer_position']['0']['0'] = [1,2]
    bench.save(path,report)


def test_only_verified_cross_route_mismatch_is_a_typed_negative(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    change_route(tmp_path/'p0/candidate.json')
    with pytest.raises(checker.VerifiedFidelityRejection) as caught:
        checker.summarize_experiment(tmp_path,config,experiment,jobs)
    result = caught.value.result
    assert result['status'] == 'rejected_fidelity'
    assert result['rejected_pairs'][0]['category'] == 'verified_ordered_expert_routes_mismatch'
    assert result['accepted_pair_numbers'] == [1]
    assert result['performance_comparison_accepted'] is False
    assert result['eligible_as_combination_prerequisite'] is False
    assert 'median_total_speedup' not in result


def test_first_pair_mismatch_cannot_hide_later_nonfinite_recording(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    change_route(tmp_path/'p0/candidate.json')
    path = tmp_path/'p1/candidate.json'
    report = json.loads(path.read_text())
    step = report['turns'][0]['step_checks'][0]
    artifact = step['logits_artifact']
    target = path.parent/artifact['path']
    raw = bytearray(target.read_bytes())
    raw[:4] = struct.pack('<f',float('nan'))
    target.write_bytes(raw)
    artifact['sha256'] = step['logits_sha256'] = hashlib.sha256(raw).hexdigest()
    bench.save(path,report)
    with pytest.raises(RecordingRejected,match='Non-finite'):
        checker.summarize_experiment(tmp_path,config,experiment,jobs)


def test_verified_full_logit_difference_is_negative_without_speed_claim(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    path = tmp_path/'p0/candidate.json'
    report = json.loads(path.read_text())
    for step in report['turns'][0]['step_checks']:
        artifact = step['logits_artifact']
        target = path.parent/artifact['path']
        values = [x[0]+.25 for x in struct.iter_unpack('<f',target.read_bytes())]
        raw = struct.pack('<'+'f'*len(values),*values)
        target.write_bytes(raw)
        artifact['sha256'] = step['logits_sha256'] = hashlib.sha256(raw).hexdigest()
    bench.save(path,report)
    with pytest.raises(checker.VerifiedFidelityRejection) as caught:
        checker.summarize_experiment(tmp_path,config,experiment,jobs)
    assert caught.value.result['rejected_pairs'][0]['category'] == 'verified_full_logits_mismatch'
    assert 'median_total_speedup' not in caught.value.result


@pytest.mark.parametrize('coherent',[True,False])
def test_token_divergence_must_preserve_argmax_proof(tmp_path,monkeypatch,coherent):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    path = tmp_path/'p0/candidate.json'
    report = json.loads(path.read_text())
    turn = report['turns'][0]
    turn['generated_ids'][0] = 2
    turn['full_output_ids'][2] = 2
    turn['step_checks'][1]['input_ids'][0] = 2
    if coherent:
        step = turn['step_checks'][0]
        artifact = step['logits_artifact']
        target = path.parent/artifact['path']
        values = [x[0] for x in struct.iter_unpack('<f',target.read_bytes())]
        values[2],values[3] = values[3],values[2]
        raw = struct.pack('<'+'f'*len(values),*values)
        target.write_bytes(raw)
        artifact['sha256'] = step['logits_sha256'] = hashlib.sha256(raw).hexdigest()
    bench.save(path,report)
    if coherent:
        with pytest.raises(checker.VerifiedFidelityRejection) as caught:
            checker.summarize_experiment(tmp_path,config,experiment,jobs)
        assert caught.value.result['rejected_pairs'][0]['category'] == 'verified_generated_tokens_mismatch'
    else:
        with pytest.raises(RecordingRejected,match='argmax'):
            checker.summarize_experiment(tmp_path,config,experiment,jobs)


def test_prefix_difference_remains_fatal_protocol_error(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    path = tmp_path/'p0/candidate.json'
    report = json.loads(path.read_text())
    turn = report['turns'][0]
    turn['full_input_ids'][0] = turn['full_output_ids'][0] = 2
    turn['step_checks'][0]['input_ids'][0] = 2
    bench.save(path,report)
    with pytest.raises(RecordingRejected,match='prefix'):
        checker.summarize_experiment(tmp_path,config,experiment,jobs)


def test_proof_change_during_rejected_comparison_is_fatal(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    path = tmp_path/'p0/candidate.json'
    change_route(path)
    original = checker.verify_logits
    def compare(a,b,**kwargs):
        try:
            return original(a,b,**kwargs)
        except RecordingRejected:
            b.write_text(b.read_text()+'\n')
            raise
    monkeypatch.setattr(checker,'verify_logits',compare)
    with pytest.raises(checker.RejectedEvidence,match='changed during fidelity') as caught:
        checker.summarize_experiment(tmp_path,config,experiment,jobs)
    assert not isinstance(caught.value,checker.VerifiedFidelityRejection)


def test_guard_rejection_never_becomes_negative_fidelity(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    def reject(*args):
        raise checker.RejectedEvidence('Resource guard failed')
    monkeypatch.setattr(checker,'validate_job',reject)
    with pytest.raises(checker.RejectedEvidence,match='guard') as caught:
        checker.summarize_experiment(tmp_path,config,experiment,jobs)
    assert not isinstance(caught.value,checker.VerifiedFidelityRejection)


def test_independent_verifier_recomputes_stored_negative(tmp_path,monkeypatch):
    config,experiment,jobs = experiment_fixture(tmp_path,monkeypatch)
    change_route(tmp_path/'p0/candidate.json')
    with pytest.raises(checker.VerifiedFidelityRejection) as caught:
        checker.summarize_experiment(tmp_path,config,experiment,jobs)
    config.update(experiments=[experiment],expected_checkpoint_sha256='f'*64)
    bench.save(tmp_path/'config.json',config)
    state = dict(config_sha256=bench.digest(tmp_path/'config.json'),status='completed_evidence_with_rejections',
        sessions=[dict(index=0,status='completed',hash_before='before',hash_after='after',integrity_blocks=[
            dict(index=0,status='sealed',jobs=list(jobs),hash_before='before',hash_after='after',
                 hash_before_completed_unix=0.,hash_after_started_unix=3.)])],jobs=jobs,
        experiment_results=dict(control=caught.value.result))
    bench.save(tmp_path/'state.json',state)
    monkeypatch.setattr(checker,'verify_manifest',lambda folder:10)
    monkeypatch.setattr(checker,'validate_config',lambda config:None)
    monkeypatch.setattr(checker,'verify_hash',lambda *a:None)
    verified = checker.verify(tmp_path)
    assert verified['status'] == 'completed_evidence_with_rejections'
    assert verified['experiments'] == {} and 'control' in verified['rejected_fidelity']
    # A fabricated changed reason cannot be trusted merely because it is saved.
    state['experiment_results']['control']['rejected_pairs'][0]['checker_reason'] = 'invented'
    bench.save(tmp_path/'state.json',state)
    with pytest.raises(checker.RejectedEvidence,match='differs from independent'):
        checker.verify(tmp_path)


def test_controller_seals_negative_experiment_then_continues(tmp_path,monkeypatch):
    import airllm.nextgen.experiment_guard as guard
    config = dict(origin=str(tmp_path),limits=asdict(ExperimentLimits()),kernel_report='provided.json',
        model_identity=dict(path='dummy'),protocol_sha256={},engine_manifests={
            'engine':dict(git_head='a'*40,source_sha256={'x':'b'*64})},experiments=[])
    for name in ('negative','unrelated'):
        config['experiments'].append(dict(name=name,requires_prior_pass=[],plan=[
            dict(id='baseline-p00',args=[],tasks='task',engine_root='engine')]))
    bench.save(tmp_path/'config.json',config)
    bench.save(tmp_path/'state.json',dict(config_sha256=bench.digest(tmp_path/'config.json'),
        status='prepared_no_worker_launched',sessions=[],jobs={},experiment_results={}))
    monkeypatch.setattr(checker,'validate_config',lambda config:None)
    monkeypatch.setattr(checker,'verify_manifest',lambda *a,**k:2)
    monkeypatch.setattr(checker,'validate_job',lambda *a:None)
    monkeypatch.setattr(bench,'unchanged',lambda config:None)
    monkeypatch.setattr(bench,'require_kernels',lambda *a:None)
    monkeypatch.setattr(guard,'system_snapshot',lambda:dict(pressure_level=1,
        available_bytes=5 << 30,swap_used_bytes=0,physical_bytes=16 << 30))
    events = []
    def hash_checkpoint(config,folder):
        events.append('hash/'+folder.name)
        folder.mkdir()
        path = folder/'checkpoint.json'
        path.write_text('{}')
        return str(path)
    def child(command,folder):
        events.append('worker/'+folder.parent.name)
        return 0
    negative = dict(status='rejected_fidelity',rejected_pairs=[dict(checker_reason='verified route mismatch')],
                    performance_comparison_accepted=False,eligible_as_combination_prerequisite=False)
    def summary(folder,config,experiment,jobs):
        events.append('summary/'+experiment['name'])
        if experiment['name'] == 'negative':
            raise checker.VerifiedFidelityRejection(negative)
        return dict(status='accepted',median_total_speedup=1.,median_worker_total_speedup=1.)
    monkeypatch.setattr(bench,'guarded_hash',hash_checkpoint)
    monkeypatch.setattr(bench,'run_child',child)
    monkeypatch.setattr(checker,'summarize_experiment',summary)
    result = bench.run(Namespace(campaign=tmp_path,resource_wait_seconds=0.,experiment=None))
    state = json.loads((tmp_path/'state.json').read_text())
    assert result['status'] == 'completed_evidence_with_rejections'
    assert state['experiment_results']['negative']['status'] == 'rejected_fidelity'
    assert state['experiment_results']['unrelated']['status'] == 'accepted'
    assert len(state['sessions'][0]['integrity_blocks']) == 2
    assert events.index('hash/integrity-0000') < events.index('worker/unrelated')
