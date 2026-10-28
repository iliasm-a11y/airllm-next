"""Controller safety/evidence tests; these never launch MLX or a model."""
from argparse import Namespace
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
sys.path.insert(0, str(ROOT/'air_llm'))
import m5_validation as bench
import verify_m5_validation as checker
from airllm.nextgen.experiment_guard import ExperimentLimits


def test_counterbalanced_custom_jobs_keep_tasks_and_separate_flags(tmp_path):
    experiment = bench.custom_pairs('direct', tmp_path, tmp_path, ['--numerical-evidence'],
                                    ['--numerical-evidence', '--packed-direct-io'], 3, tmp_path/'tasks.json')
    assert [x['id'] for x in experiment['plan']] == [
        'baseline-p00','candidate-p00','candidate-p01','baseline-p01','baseline-p02','candidate-p02']
    assert all(x['tasks'] == str(tmp_path/'tasks.json') for x in experiment['plan'])
    assert '--packed-direct-io' not in experiment['plan'][0]['args']


def test_process_creation_cleanup_for_light_child(tmp_path):
    code = bench.run_child([sys.executable, '-c', 'print("light-control-only")'], tmp_path/'attempt')
    assert code == 0
    assert 'light-control-only' in (tmp_path/'attempt/controller-child.log').read_text()
    assert json.loads((tmp_path/'attempt/cleanup.json').read_text())['status'] == 'completed'


def test_process_cleanup_after_early_identity_error(tmp_path, monkeypatch):
    import psutil
    real_process = psutil.Process
    child = []
    real_popen = bench.subprocess.Popen
    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        child.append(process)
        return process
    def broken(pid):
        if child and pid == child[0].pid:
            raise RuntimeError('injected failure before first child observation')
        return real_process(pid)
    monkeypatch.setattr(bench.subprocess, 'Popen', popen)
    monkeypatch.setattr(psutil, 'Process', broken)
    with pytest.raises(RuntimeError, match='injected'):
        bench.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path/'attempt')
    assert child[0].poll() is not None


def resource_guard(tmp_path, **updates):
    output = tmp_path/'worker.json'
    command = ['python','benchmark_research51.py']
    limits = asdict(ExperimentLimits())
    raw = dict(command=command+['--worker'],limits=limits,violation='macos_memory_pressure',
               exit_code=-15,sample_count=3,peak_pressure_level=2,
               minimum_available_bytes=4 << 30,maximum_swap_growth_bytes=0)
    raw.update(updates)
    bench.save(output.with_suffix('.guard.json'),raw)
    return output,command,limits


def test_measured_worker_resource_guard_is_distinct_from_programming_error(tmp_path):
    output,command,limits = resource_guard(tmp_path)
    evidence = bench.worker_resource_refusal(output,command,limits)
    assert evidence['reason'] == 'macos_memory_pressure'
    assert evidence['guard_sha256'] == bench.digest(output.with_suffix('.guard.json'))


@pytest.mark.parametrize('updates', [dict(violation='timeout'),dict(violation='owned_process_tree_rss'),
    dict(exception='RuntimeError: programming bug'),dict(cleanup_error='failed'),
    dict(exit_code=0),dict(peak_pressure_level=1),dict(command=['another worker']),dict(sample_count=0)])
def test_worker_guard_errors_and_unmeasured_claims_are_never_resource_retries(tmp_path,updates):
    output,command,limits = resource_guard(tmp_path,**updates)
    assert bench.worker_resource_refusal(output,command,limits) is None


def test_real_admission_exception_requires_no_spawn_and_measured_pressure(tmp_path):
    output,command,limits = resource_guard(tmp_path,violation=None,exit_code=None,sample_count=0,
        exception='MemoryError: macOS memory pressure is above normal before launch',before=dict(pressure_level=2))
    assert bench.worker_resource_refusal(output,command,limits)['reason'] == 'worker_admission_resources'


def test_worker_failure_cannot_be_reclassified_from_resource_guard(tmp_path):
    output,command,limits = resource_guard(tmp_path)
    bench.save(output,dict(status='failed',error='FloatingPointError: nonfinite'))
    assert bench.worker_resource_refusal(output,command,limits) is None


def test_malformed_guard_remains_programming_or_integrity_failure(tmp_path):
    output,command,limits = resource_guard(tmp_path)
    output.with_suffix('.guard.json').write_text('broken')
    with pytest.raises(json.JSONDecodeError):
        bench.worker_resource_refusal(output,command,limits)


def test_owned_cleanup_ignores_reused_pid_identity(tmp_path):
    import psutil
    current = psutil.Process()
    bench.owned_cleanup([dict(pid=os.getpid(), created=current.create_time()-1)])
    assert current.is_running()


def test_one_controller_lock(tmp_path):
    with bench.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another runner'):
            with bench.campaign_lock(tmp_path):
                pass


def test_manifest_rejects_mutated_or_unlisted_bytes(tmp_path):
    (tmp_path/'config.json').write_text('{"x":1}')
    bench.write_manifest(tmp_path)
    assert checker.verify_manifest(tmp_path) == 1
    (tmp_path/'config.json').write_text('{"x":2}')
    with pytest.raises(checker.RejectedEvidence, match='digest'):
        checker.verify_manifest(tmp_path)


def test_manifest_crash_recovery_only_exempts_atomic_state(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'state.json').write_text('{"status":"prepared"}')
    bench.write_manifest(tmp_path)
    (tmp_path/'state.json').write_text('{"status":"running"}')
    (tmp_path/'new-partial.log').write_text('incomplete raw log')
    assert checker.verify_manifest(tmp_path, allow_interrupted=True) == 2
    (tmp_path/'config.json').write_text('tampered')
    with pytest.raises(checker.RejectedEvidence, match='digest'):
        checker.verify_manifest(tmp_path, allow_interrupted=True)


def test_manifest_rejects_file_escape(tmp_path):
    outside = tmp_path.parent/'outside.json'
    outside.write_text('{}')
    (tmp_path/'alias.json').symlink_to(outside)
    bench.write_manifest(tmp_path)
    with pytest.raises(checker.RejectedEvidence):
        checker.verify_manifest(tmp_path)


def normal_guard():
    limits = asdict(ExperimentLimits())
    snap = dict(pressure_level=1, available_bytes=5 << 30, swap_used_bytes=0)
    return limits, dict(command=['python','worker'], limits=limits, exit_code=0,
        violation=None, sample_count=3, sampled_peak_rss_bytes=50 << 20,
        minimum_available_bytes=5 << 30, maximum_swap_growth_bytes=0,
        sampled_peak_processes=1, peak_pressure_level=1, elapsed_seconds=.1,
        before=snap, after=snap)


@pytest.mark.parametrize('field,value', [('peak_pressure_level',2),('violation','memory'),
    ('exit_code',1),('sample_count',0),('maximum_swap_growth_bytes',129 << 20),
    ('sampled_peak_rss_bytes',3 << 30),('minimum_available_bytes',2 << 30),
    ('sampled_peak_processes',33),('cleanup_error','failed')])
def test_verifier_rejects_resource_failure(field, value):
    limits, guard = normal_guard()
    checker.guard_accepted(guard, limits, ['python','worker'])
    guard[field] = value
    with pytest.raises(checker.RejectedEvidence):
        checker.guard_accepted(guard, limits, ['python','worker'])


def kernel_fixture():
    """Synthetic proof fixture only: no claimed compilation or performance."""
    comparison = dict(accepted=True,atol=1e-4,rtol=5e-5,max_absolute_error=0.)
    cases = []
    shapes = [(19,256,b) for b in (2,16,32,64,128)]+[(64,512,16),(64,512,64),(128,1024,128)]
    for kind in (12,14):
        for grouped in (False,True):
            for rows,width,batch in shapes:
                for mode,tile in [('reuse',2),('reuse',4),('reuse',8),('reuse',16),('tensorops',4)]:
                    cases.append(dict(kind=kind,grouped=grouped,rows=rows,width=width,batch=batch,
                        backend=mode,token_tile=tile,compiled_and_executed=True,
                        reference_vs_scalar=dict(comparison),candidate_vs_scalar=dict(comparison),
                        candidate_vs_reference=dict(comparison),selected=False,
                        host_seconds=[[1.,1.],[2.,2.]],host_median_speedup=.5,timing_order=[[0,1],[1,0]]))
    return dict(cases=cases)


def test_reference_kept_when_kernel_uncompiled_wrong_or_slower():
    experiment = dict(plan=[dict(args=['--metal-batch-backend','tensorops'])])
    report = kernel_fixture()
    tensorops = [c for c in report['cases'] if c['backend'] == 'tensorops']
    for case in tensorops:
        case['compiled_and_executed'] = False
    assert 'No compiled' in checker.kernel_eligibility(report, experiment)
    for case in tensorops:
        case['compiled_and_executed'] = True
    assert 'No observed' in checker.kernel_eligibility(report, experiment)
    case = tensorops[0]
    case['selected'] = True
    case.update(host_seconds=[[2.,2.],[1.,1.]],host_median_speedup=2.)
    assert checker.kernel_eligibility(report, experiment) is None
    case['candidate_vs_reference']['accepted'] = False
    assert 'Some required' in checker.kernel_eligibility(report, experiment)


def test_kernel_tile_must_have_its_own_evidence():
    experiment = dict(plan=[dict(args=['--metal-batch-backend','reuse','--metal-token-tile','16'])])
    report = kernel_fixture()
    for case in report['cases']:
        if case['backend'] == 'reuse' and case['token_tile'] == 16:
            case['compiled_and_executed'] = False
    assert 'No compiled' in checker.kernel_eligibility(report, experiment)


def test_independent_verifier_rejects_incomplete_campaign(tmp_path):
    config = {}
    bench.save(tmp_path/'config.json', config)
    bench.save(tmp_path/'state.json', dict(config_sha256=bench.digest(tmp_path/'config.json'),
               status='deferred_or_interrupted'))
    bench.write_manifest(tmp_path)
    with pytest.raises(checker.RejectedEvidence, match='incomplete'):
        checker.verify(tmp_path)


def test_no_launch_under_warning_pressure(tmp_path, monkeypatch):
    import airllm.nextgen.experiment_guard as guard
    # This fixture injects an observation, not a lower guard threshold. Any call
    # to subprocess/model launch would make the test fail immediately.
    config = dict(origin=str(tmp_path),protocol_sha256={}, engine_manifests={}, model_identity={}, experiments=[],
                  limits=asdict(ExperimentLimits()))
    bench.save(tmp_path/'config.json', config)
    bench.save(tmp_path/'state.json', dict(config_sha256=bench.digest(tmp_path/'config.json'),
        status='prepared_no_worker_launched', sessions=[],jobs={},experiment_results={}))
    bench.write_manifest(tmp_path)
    monkeypatch.setattr(bench, 'unchanged', lambda config: None)
    monkeypatch.setattr(checker, 'validate_config', lambda config: None)
    monkeypatch.setattr(guard, 'system_snapshot', lambda: dict(pressure_level=2,
        available_bytes=8 << 30, swap_used_bytes=0, physical_bytes=16 << 30))
    monkeypatch.setattr(bench, 'run_child', lambda *a: pytest.fail('worker launched under warning pressure'))
    with pytest.raises(MemoryError, match='pressure'):
        bench.run(Namespace(campaign=tmp_path, resource_wait_seconds=0.,experiment=None))
    state = json.loads((tmp_path/'state.json').read_text())
    assert state['status'] == 'deferred_or_interrupted'
    assert state['jobs'] == {}
    assert state['sessions'][0]['before']['pressure_level'] == 2
    assert checker.verify_manifest(tmp_path) >= 2


def test_prepare_requires_real_expected_digest_and_counterbalance(tmp_path):
    with pytest.raises(ValueError, match='two pairs'):
        bench.prepare(Namespace(pairs=1))
    with pytest.raises(ValueError, match='SHA-256'):
        bench.prepare(Namespace(pairs=2,expected_sha256='model-name'))


def test_verifier_uses_only_standard_library():
    import ast
    source = ast.parse((ROOT/'benchmarks/verify_m5_validation.py').read_text())
    names = {n.names[0].name.split('.')[0] for n in ast.walk(source) if isinstance(n,ast.Import)}
    names.update(n.module.split('.')[0] for n in ast.walk(source) if isinstance(n,ast.ImportFrom))
    assert not (names & {'mlx','numpy','psutil','airllm','m5_validation','benchmark_research51'})


def test_rehashed_config_cannot_weaken_policy(tmp_path):
    config = dict(schema_version=1,origin=str(tmp_path),limits=dict(checker.RETAINED_LIMITS, reserve_bytes=0))
    with pytest.raises(checker.RejectedEvidence, match='safety policy'):
        checker.validate_config(config)


def test_relocated_evidence_cannot_escape_root(tmp_path):
    moved = tmp_path/'new'
    moved.mkdir()
    (moved/'report.json').write_text('{}')
    config = dict(origin='/original/campaign')
    assert checker.evidence_path(moved, '/original/campaign/report.json', config) == moved/'report.json'
    with pytest.raises(checker.RejectedEvidence, match='escapes'):
        checker.evidence_path(moved, '/another/report.json', config)


def test_independent_verifier_rejects_unsealed_worker(tmp_path,monkeypatch):
    experiment = dict(name='control',plan=[dict(id='baseline-p00',args=[])],requires_prior_pass=[])
    config = dict(experiments=[experiment])
    bench.save(tmp_path/'config.json',config)
    state = dict(config_sha256=bench.digest(tmp_path/'config.json'),status='completed',
        sessions=[dict(index=0,status='completed',hash_before='before',hash_after='after',integrity_blocks=[])],
        experiment_results=dict(control=dict(status='accepted')),
        jobs={'control/baseline-p00':dict(session=0,integrity_block=0)})
    bench.save(tmp_path/'state.json',state)
    monkeypatch.setattr(checker,'verify_manifest',lambda folder:2)
    monkeypatch.setattr(checker,'validate_config',lambda config:None)
    monkeypatch.setattr(checker,'verify_hash',lambda *a:None)
    with pytest.raises(checker.RejectedEvidence,match='sealed'):
        checker.verify(tmp_path)


def test_kernel_benefit_claim_requires_its_raw_measurements():
    experiment = dict(plan=[dict(args=['--metal-batch-backend','tensorops'])])
    report = kernel_fixture()
    for case in report['cases']:
        case.pop('host_seconds', None)
    with pytest.raises(checker.RejectedEvidence,match='timings'):
        checker.kernel_eligibility(report,experiment)


def test_quick_or_loose_kernel_report_cannot_qualify():
    experiment = dict(plan=[dict(args=['--metal-batch-backend','tensorops'])])
    report = kernel_fixture()
    report['cases'] = report['cases'][:1]
    with pytest.raises(checker.RejectedEvidence,match='incomplete'):
        checker.kernel_eligibility(report,experiment)
    report = kernel_fixture()
    report['cases'][0]['candidate_vs_reference']['atol'] = 100
    with pytest.raises(checker.RejectedEvidence,match='tolerances'):
        checker.kernel_eligibility(report,experiment)


def test_inherited_scalar_negative_is_evidence_but_excludes_strict_backend():
    report = kernel_fixture()
    case = next(c for c in report['cases'] if c['backend'] == 'tensorops')
    case['reference_vs_scalar']['accepted'] = False
    checker.validate_kernel_grid(report)
    reason = checker.kernel_eligibility(report,dict(plan=[dict(args=['--metal-batch-backend','tensorops'])]))
    assert 'Some required' in reason


def reference_exact_fixture():
    # Control-flow fixture stands for independently recomputed raw parity; the
    # artifact reader has separate tests below. It is no hardware measurement.
    report = kernel_fixture()
    cases = [c for c in report['cases'] if c['backend'] == 'reuse' and c['token_tile'] == 2]
    cases[0].update(selected=True,host_seconds=[[2.,2.],[1.,1.]],host_median_speedup=2.)
    cases[-1]['reference_vs_scalar']['accepted'] = False
    cases[-1]['candidate_vs_scalar']['accepted'] = False
    cases[-1].pop('host_seconds')
    report.update(_reviewed_qualification_policy=checker.KERNEL_POLICY,
        _recorded_c_diagnosis_verified=True,
        _verified_raw_reference_bitwise={checker.kernel_case_key(c):True for c in report['cases']})
    return report,cases


def test_v2_reuse_only_exact_reference_preserves_scalar_negative():
    report,cases = reference_exact_fixture()
    experiment = dict(plan=[dict(args=['--metal-batch-backend','reuse','--metal-token-tile','2'])])
    assert checker.kernel_eligibility(report,experiment) is None
    assert report['_qualification_modes']['reuse/2'] == 'reference_exact_with_inherited_scalar_disagreement'
    assert cases[-1]['reference_vs_scalar']['accepted'] is False
    report['_recorded_c_diagnosis_verified'] = False
    assert checker.kernel_eligibility(report,experiment) is not None
    report['_recorded_c_diagnosis_verified'] = True
    report['_verified_raw_reference_bitwise'][checker.kernel_case_key(cases[-1])] = False
    assert checker.kernel_eligibility(report,experiment) is not None


def test_reference_exact_alternative_never_qualifies_tensorops():
    report,cases = reference_exact_fixture()
    case = next(c for c in report['cases'] if c['backend'] == 'tensorops')
    case['reference_vs_scalar']['accepted'] = False
    assert checker.kernel_eligibility(report,dict(plan=[dict(args=['--metal-batch-backend','tensorops'])])) is not None


def raw_kernel_fixture(tmp_path):
    def tensor(name,raw,shape,dtype):
        path = tmp_path/name
        path.write_bytes(raw)
        return dict(file=name,shape=shape,dtype=dtype,bytes=len(raw),sha256=bench.digest(path))
    scalar = struct.pack('<2f',1.,2.01)
    reference = struct.pack('<2f',1.,2.)
    diff = abs(struct.unpack('<2f',scalar)[1]-2.)
    negative = dict(accepted=False,atol=1e-4,rtol=5e-5,max_absolute_error=diff)
    case = dict(kind=12,grouped=False,rows=1,width=256,batch=2,backend='reuse',token_tile=2,
        compiled_and_executed=True,reference_vs_scalar=dict(negative),candidate_vs_scalar=dict(negative),
        candidate_vs_reference=dict(accepted=True,atol=1e-4,rtol=5e-5,max_absolute_error=0.),
        numeric_evidence=dict(packed=tensor('packed.u8',bytes(144),[1,144],'|u1'),
            input=tensor('input.f32',bytes(2*256*4),[2,256],'<f4'),
            scalar=tensor('scalar.f32',scalar,[2,1],'<f4'),
            reference=tensor('reference.f32',reference,[2,1],'<f4')),
        candidate_evidence=tensor('candidate.f32',reference,[2,1],'<f4'))
    return dict(cases=[case])


def test_raw_kernel_parity_recomputed_and_false_comparison_rejected(tmp_path):
    report = raw_kernel_fixture(tmp_path)
    proof = checker.validate_kernel_artifacts(report,tmp_path)
    assert proof == {checker.kernel_case_key(report['cases'][0]):True}
    report['cases'][0]['candidate_vs_scalar']['accepted'] = True
    with pytest.raises(checker.RejectedEvidence,match='contradicts'):
        checker.validate_kernel_artifacts(report,tmp_path)


def test_raw_kernel_sidecars_survive_safe_campaign_copy(tmp_path):
    original = tmp_path/'original'
    original.mkdir()
    report = raw_kernel_fixture(original)
    source = original/'report.json'
    bench.save(source,report)
    target = tmp_path/'copied'/'report.json'
    target.parent.mkdir()
    assert bench.copy_kernel_sidecars(source,target) == 5
    proof = checker.validate_kernel_artifacts(report,target.parent)
    assert all(proof.values())
