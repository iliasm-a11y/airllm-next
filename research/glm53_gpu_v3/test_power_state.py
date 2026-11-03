"""Portable power/clock/process fixtures: no model, native compile or wake command."""
from copy import deepcopy
import plistlib
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research.glm53_gpu_v3.power_state import (ClockContinuity, PowerGate, PowerStateObserver, ScopedAwake,
    parse_lid, parse_power, power_evidence_matches, power_preflight)


def snapshot(closed=False, source='battery', available=True):
    return {'available': available, 'lid_closed': closed, 'power_source': source}


def observer(row):
    return SimpleNamespace(snapshot=lambda: deepcopy(row))


def valid_evidence():
    edge = {'passed': True, 'clock': {'passed': True}, 'power': {'passed': True,
        'snapshot': snapshot()}}
    return {'owned_pid': 71, 'power_gate': {'passed': True, 'finalized': True, 'expected_power_source': 'battery',
        'violation': None, 'clock_samples': 5, 'power_samples': 3, 'first': edge, 'last': edge,
        'idle_assertion': {'started': True, 'cleaned_up': True, 'target_pid': 71, 'target_create_time': 1,
            'helper_pid': 72, 'helper_create_time': 2, 'helper_exit_code': -15, 'cleanup_kind': 'owner_term',
            'owner_signal': 'SIGTERM', 'helper_alive_before_owner_signal': True, 'exit_qualified': True,
            'command': ['/usr/bin/caffeinate', '-i', '-w', '71']}}}


@pytest.mark.parametrize('closed,source,available,passed', [
    (False, 'battery', True, True), (False, 'ac', True, True), (True, 'ac', True, False),
    (None, 'ac', True, False), (False, None, True, False), (False, 'ac', False, False)])
def test_admission_requires_known_open_lid_and_source(closed, source, available, passed):
    assert power_preflight(observer=observer(snapshot(closed, source, available)))['passed'] is passed


def test_bounded_plist_is_exact_boolean_and_power_is_unambiguous():
    assert parse_lid(plistlib.dumps([{'AppleClamshellState': False}])) is False
    assert parse_power(b"Now drawing from 'AC Power'\n") == 'ac'
    for row in ([{}], [{'AppleClamshellState': 0}], [{'AppleClamshellState': False}]*2):
        with pytest.raises(ValueError):
            parse_lid(plistlib.dumps(row))
    with pytest.raises(ValueError):
        parse_lid(b'x'*(128*1024+1))
    with pytest.raises(ValueError):
        parse_power(b"Now drawing from 'AC Power'\nNow drawing from 'Battery Power'\n")


def test_observer_failure_is_unavailable_and_never_assumed_awake():
    def fail(argv):
        raise TimeoutError('fixture timeout')
    value = PowerStateObserver(command=fail, system=lambda: 'Darwin').snapshot()
    assert value['available'] is False and value['lid_closed'] is None
    assert not power_preflight(observer=observer(value))['passed']


@pytest.mark.parametrize('wall,perf,passed', [
    ([100, 100.5, 101], [10, 10.5, 11], True),
    ([100, 100.5, 6000], [10, 10.5, 11], False),
    ([100, 100.5, 99], [10, 10.5, 11], False),
    ([100, 101, 102], [10, 11, 10.9], False),
    ([100, 101, 102], [10, 11, float('nan')], False)])
def test_clock_normal_sleep_backward_and_unknown_fail_closed(wall, perf, passed):
    a, b = iter(wall), iter(perf)
    clock = ClockContinuity(wall=lambda: next(a), perf=lambda: next(b))
    rows = [clock.sample() for _ in wall]
    assert rows[-1]['passed'] is passed
    if not passed:
        assert clock.violation


def test_cumulative_clock_drift_and_final_check_are_sticky():
    a, b = iter([100, 102, 104, 103]), iter([10, 10.5, 11, 13])
    gate = PowerGate(observer=observer(snapshot()), clock=ClockContinuity(wall=lambda: next(a), perf=lambda: next(b)))
    assert gate.sample()['passed']
    assert gate.sample(include_power=False)['passed']
    assert not gate.sample()['passed']
    assert not gate.sample(final=True)['passed']
    assert not gate.summary()['passed']


def test_source_change_rejects_and_success_requires_final_sample():
    rows = iter([snapshot(), snapshot(source='ac')])
    gate = PowerGate(observer=SimpleNamespace(snapshot=lambda: next(rows)))
    assert gate.sample()['passed'] and not gate.summary()['passed']
    assert not gate.sample(final=True)['passed'] and not gate.summary()['passed']


def test_awake_helper_owns_only_scoped_child_and_cleans_on_exception():
    target = Mock(pid=71); target.poll.return_value = None
    helper = Mock(pid=72, returncode=-15); helper.poll.return_value = None
    state = {71: (1, '/owned/server'), 72: (2, '/usr/bin/caffeinate')}
    def process(pid):
        created, exe = state[pid]
        return SimpleNamespace(create_time=lambda: created, exe=lambda: exe,
            is_running=lambda: True, status=lambda: 'running')
    launch = Mock(return_value=helper)
    with pytest.raises(ValueError):
        with ScopedAwake(target, popen=launch, process=process) as awake:
            assert awake.record['started']
            raise ValueError('fixture failure')
    assert launch.call_args.args[0] == ['/usr/bin/caffeinate', '-i', '-w', '71']
    helper.terminate.assert_called_once(); helper.wait.assert_called_once()
    target.terminate.assert_not_called()
    assert awake.record['cleaned_up']


def test_awake_helper_rejects_target_pid_reuse_and_cleans_its_helper():
    target = Mock(pid=81); target.poll.return_value = None
    helper = Mock(pid=82, returncode=-15); helper.poll.return_value = None
    times = iter([1, 2])
    def process(pid):
        created = next(times) if pid == 81 else 3
        return SimpleNamespace(create_time=lambda: created, exe=lambda: '/usr/bin/caffeinate',
            is_running=lambda: True, status=lambda: 'running')
    with pytest.raises(RuntimeError):
        with ScopedAwake(target, popen=lambda *a, **k: helper, process=process):
            pytest.fail('reused target must never enter')
    helper.terminate.assert_called_once()


def test_timing_qualification_rejects_missing_power_source_and_cleanup():
    report = valid_evidence()
    assert power_evidence_matches(report, 'battery')
    assert not power_evidence_matches(report, 'ac')
    assert not power_evidence_matches({}, 'battery')
    report['power_gate']['idle_assertion']['cleaned_up'] = False
    assert not power_evidence_matches(report, 'battery')


def test_qualification_rejects_wrong_owned_pid_and_false_open_edge():
    report = valid_evidence(); report['owned_pid'] = 999
    assert not power_evidence_matches(report, 'battery')
    report = valid_evidence(); report['power_gate']['last']['power']['snapshot']['lid_closed'] = True
    assert not power_evidence_matches(report, 'battery')


@pytest.mark.parametrize('code,target_exited,kind,owner_signal,qualified', [
    (1, True, 'unexpected_exit', None, False),
    (-15, True, 'unexpected_exit', None, False),
    (0, False, 'unexpected_exit', None, False),
    (0, True, 'natural_target_exit', None, True),
    (-15, False, 'owner_term', 'SIGTERM', True),
    (-9, True, 'owner_kill', 'SIGKILL', False)])
def test_helper_exit_classification(code, target_exited, kind, owner_signal, qualified):
    target = Mock(pid=71); target.poll.return_value = None
    helper = Mock(pid=72, returncode=None); helper.poll.return_value = None
    def process(pid):
        return SimpleNamespace(create_time=lambda: 1 if pid == 71 else 2,
            exe=lambda: '/usr/bin/caffeinate', is_running=lambda: True, status=lambda: 'running')
    awake = ScopedAwake(target, popen=lambda *a, **k: helper, process=process)
    awake.__enter__()
    target.poll.return_value = 0 if target_exited else None
    if owner_signal is None:
        helper.poll.return_value = code
        helper.returncode = code
    else:
        def reap(timeout):
            if owner_signal == 'SIGKILL' and not helper.kill.called:
                raise subprocess.TimeoutExpired('caffeinate', timeout)
            helper.returncode = code
            helper.poll.return_value = code
            return code
        helper.wait.side_effect = reap
    if qualified:
        awake.close()
    else:
        with pytest.raises(RuntimeError, match='did not qualify'):
            awake.close()
    assert awake.record['cleaned_up'] is True
    assert awake.record['exit_qualified'] is qualified
    assert awake.record['cleanup_kind'] == kind
    assert awake.record['owner_signal'] == owner_signal
    report = valid_evidence(); report['power_gate']['idle_assertion'] = awake.record
    assert power_evidence_matches(report, 'battery') is qualified
    if owner_signal == 'SIGKILL':
        helper.kill.assert_called_once()
    elif owner_signal is None:
        helper.terminate.assert_not_called()


def test_asserted_success_cannot_hide_abnormal_helper_exit_code():
    for code in (1, -9, -15):
        report = valid_evidence()
        helper = report['power_gate']['idle_assertion']
        helper.update(helper_exit_code=code, cleanup_kind='natural_target_exit', owner_signal=None,
                      target_exited_at_cleanup=True, exit_qualified=True)
        assert not power_evidence_matches(report, 'battery')
