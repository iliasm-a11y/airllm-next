"""Several individually small swap increases must not accumulate unnoticed."""
import pytest
import json
from types import SimpleNamespace
from .campaign_limits import MAX_SWAP_GROWTH, remaining_swap_budget
from .model_profile import REQUIRED_DISK, REQUIRED_MEMORY, resource_blockers


def test_two_individually_safe_arms_exhaust_common_campaign_budget():
    baseline = 3 << 30
    first_end = baseline + (20 << 20)
    assert remaining_swap_budget({'swap_used_bytes': first_end}, baseline) == 12 << 20
    second_end = first_end + (20 << 20)
    # Each20MiB increase is below the original per-arm32MiB limit.
    with pytest.raises(MemoryError, match='campaign_swap_growth_above_32MiB'):
        remaining_swap_budget({'swap_used_bytes': second_end}, baseline)
    sample = {'available_bytes': REQUIRED_MEMORY, 'pressure_level': 1,
              'swap_used_bytes': second_end}
    assert resource_blockers(sample, REQUIRED_DISK, {'passed': True}, baseline) == ['campaign_swap_growth_above_32MiB']


def test_swap_reduction_does_not_increase_original_per_arm_cap():
    assert remaining_swap_budget({'swap_used_bytes': 0}, 3 << 30) == MAX_SWAP_GROWTH
    assert remaining_swap_budget({}, None) == MAX_SWAP_GROWTH


@pytest.mark.parametrize('used', [None, -1, True, 1.5])
def test_common_budget_fails_closed_without_valid_swap_measurement(used):
    with pytest.raises(MemoryError, match='campaign_swap_measurement_unavailable'):
        remaining_swap_budget({'swap_used_bytes': used}, 0)


@pytest.mark.parametrize('baseline', [-1, True, 1.5])
def test_invalid_common_baseline_rejected(baseline):
    with pytest.raises(ValueError):
        remaining_swap_budget({'swap_used_bytes': 0}, baseline)


def test_profile_never_launches_after_common_budget_is_spent(tmp_path, monkeypatch):
    from . import model_profile as m
    model = tmp_path / 'model'; model.mkdir()
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    probe = tmp_path / 'probe'; probe.write_bytes(b'must never execute')
    baseline = 3 << 30
    monkeypatch.setattr(m, 'system_snapshot', lambda: {
        'pressure_level': 1, 'available_bytes': REQUIRED_MEMORY,
        'swap_used_bytes': baseline + MAX_SWAP_GROWTH + 1})
    monkeypatch.setattr(m, 'inspect_native_checkpoint', lambda *a: {
        'integrity_verified': True, 'architecture_matches': True})
    monkeypatch.setattr(m, 'PowerGate', lambda **kw: SimpleNamespace(sample=lambda: {'passed': True}))
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: SimpleNamespace(free=REQUIRED_DISK))
    monkeypatch.setattr(m, 'supervise', lambda *a, **kw: pytest.fail('spent campaign launched a native process'))
    args = SimpleNamespace(worker=False, output=tmp_path / 'out', model=model,
        runtime=runtime, probe=probe, receipt=tmp_path / 'receipt', tokens=32,
        diagnostic=False, prompt='test', campaign_swap_used_bytes=baseline)
    assert m.run(args) == 2
    report = json.loads((args.output / 'report.json').read_text())
    assert report['blockers'] == ['campaign_swap_growth_above_32MiB']
    assert not report['native_process_launched'] and not report['performance_measured']


def test_component_guard_retains_only_the_unspent_campaign_allowance(tmp_path, monkeypatch):
    from . import guard_components as m
    baseline = 3 << 30
    monkeypatch.setattr(m, 'system_snapshot', lambda: {'swap_used_bytes': baseline + (20 << 20)})
    observed = {}
    def supervisor(command, output, *, limits, env):
        observed['limits'] = limits
        return {'exit_code': 0, 'violation': None}
    monkeypatch.setattr(m, 'supervise', supervisor)
    output = tmp_path / 'out'
    args = SimpleNamespace(runtime=tmp_path, build=tmp_path, output=output,
        format=2, rows=2048, width=4096, banks=2, seed=1, repetitions=21,
        campaign_swap_used_bytes=baseline)
    # No synthetic result is created: this only checks the supervisor contract.
    assert m.run(args) == 2
    limits = observed['limits']
    assert limits.swap_growth_bytes == 12 << 20
    assert limits.rss_bytes == int(8.5 * (1 << 30))
    assert limits.reserve_bytes == 3 << 30


def test_worker_disk_abort_retains_outer_guard_and_input_checks(tmp_path, monkeypatch):
    from . import model_profile as m
    model=tmp_path/'model';model.mkdir()
    weights=model/'fixture.gguf';weights.write_bytes(b'not model weights; never executed')
    runtime=tmp_path/'runtime';runtime.mkdir()
    probe=tmp_path/'probe';probe.write_bytes(b'never executed')
    monkeypatch.setattr(m,'system_snapshot',lambda:{'pressure_level':1,'available_bytes':REQUIRED_MEMORY,'swap_used_bytes':0})
    monkeypatch.setattr(m,'PowerGate',lambda **kw:SimpleNamespace(sample=lambda:{'passed':True}))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=REQUIRED_DISK))
    monkeypatch.setattr(m,'inspect_native_checkpoint',lambda *a:{'integrity_verified':True,
        'architecture_matches':True,'files':[{'name':weights.name,'identity':m._identity(weights)}]})
    def stopped(command, output, *, limits):
        output.with_suffix('.guard.json').write_text(json.dumps({'violation':None,'exit_code':2,'maximum_swap_growth_bytes':0}))
        (output.parent/'worker.json').write_text(json.dumps({'status':'failed','error':'disk reserve fell below32GiB'}))
        raise RuntimeError('owned worker stopped')
    monkeypatch.setattr(m,'supervise',stopped)
    args=SimpleNamespace(worker=False,output=tmp_path/'out',model=model,runtime=runtime,
        probe=probe,receipt=tmp_path/'receipt',tokens=32,diagnostic=True,prompt='test')
    assert m.run(args)==2
    report=json.loads((args.output/'report.json').read_text())
    assert report['status']=='failed' and report['worker']['status']=='failed'
    assert report['guard']['exit_code']==2 and report['guard']['violation'] is None
    assert report['weight_identities_unchanged'] and report['native_artifacts_unchanged']
    assert not report['performance_measured']
