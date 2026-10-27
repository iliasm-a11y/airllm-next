"""No inference: fake-clock admission windows and persisted campaign stops."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def window(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'air_llm'))
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks'))
    campaign = importlib.import_module('campaign_research51')
    guard = importlib.import_module('airllm.nextgen.experiment_guard')
    clock = Clock()
    monkeypatch.setattr(campaign, 'monotonic', clock.monotonic)
    monkeypatch.setattr(campaign, 'sleep', clock.sleep)
    limits = guard.ExperimentLimits(timeout_seconds=77.)
    return campaign, guard, clock, limits


def snapshot(limits, *, available=None, pressure=1):
    return dict(available_bytes=(limits.reserve_bytes + limits.admitted_payload_bytes + (256 << 20)
                                 if available is None else available),
                physical_bytes=16 << 30, swap_used_bytes=12345, pressure_level=pressure)


def test_default_zero_preserves_fail_fast_worker_path(window, monkeypatch):
    campaign, guard, clock, limits = window
    def forbidden():
        pytest.fail('Disabled window must leave admission to the worker')
    monkeypatch.setattr(guard, 'system_snapshot', forbidden)
    run = {}
    campaign.wait_before_job(limits, 0., run)
    assert run['launch_wait_seconds'] == 0.
    assert run['launch_wait_status'] == 'disabled'
    assert run['launch_wait_samples'] == []
    assert clock.sleeps == []


def test_ready_window_requires_five_observed_seconds_and_preserves_limits(window, monkeypatch):
    campaign, guard, clock, limits = window
    monkeypatch.setattr(guard, 'system_snapshot', lambda: snapshot(limits))
    run = {}
    campaign.wait_before_job(limits, 10., run)
    assert run['launch_wait_status'] == 'ready'
    assert run['launch_wait_seconds'] == 5.
    assert run['launch_wait_sample_count'] == 6
    assert run['launch_wait_policy']['admission_limits'] == asdict(limits)
    assert run['launch_wait_policy']['headroom_bytes'] == 256 << 20
    assert run['launch_wait_samples'][-1]['observed_stable_seconds'] == 5.
    assert all(delay == 1. for delay in clock.sleeps)


def test_headroom_above_unchanged_admission_is_required(window, monkeypatch):
    campaign, guard, clock, limits = window
    threshold = limits.reserve_bytes + limits.admitted_payload_bytes + (256 << 20)
    def current():
        return snapshot(limits, available=threshold - 1 if clock.now < 3 else threshold)
    monkeypatch.setattr(guard, 'system_snapshot', current)
    run = {}
    campaign.wait_before_job(limits, 10., run)
    assert run['launch_wait_seconds'] == 8.
    assert all(not sample['ready'] for sample in run['launch_wait_samples'][:3])
    assert all('256 MiB' in sample['reason'] for sample in run['launch_wait_samples'][:3])


def test_pressure_warning_restarts_the_consecutive_window(window, monkeypatch):
    campaign, guard, clock, limits = window
    monkeypatch.setattr(guard, 'system_snapshot',
                        lambda: snapshot(limits, pressure=2 if clock.now == 3 else 1))
    run = {}
    campaign.wait_before_job(limits, 12., run)
    assert run['launch_wait_seconds'] == 9.
    assert run['launch_wait_samples'][3]['ready'] is False
    assert run['launch_wait_samples'][4]['observed_stable_seconds'] == 0.


@pytest.mark.parametrize('pressure', [None, 2, 4])
def test_unknown_or_elevated_pressure_never_opens_a_window(window, monkeypatch, pressure):
    campaign, guard, clock, limits = window
    monkeypatch.setattr(guard, 'system_snapshot', lambda: snapshot(limits, pressure=pressure))
    run = {}
    with pytest.raises(MemoryError, match='wait expired'):
        campaign.wait_before_job(limits, 6., run)
    assert run['status'] == 'stopped'
    assert run['launch_wait_status'] == 'timed_out'
    assert run['launch_wait_seconds'] == 6.
    assert not any(sample['ready'] for sample in run['launch_wait_samples'])


def test_timeout_keeps_only_sixty_recent_samples(window, monkeypatch):
    campaign, guard, clock, limits = window
    monkeypatch.setattr(guard, 'system_snapshot', lambda: snapshot(limits, available=0))
    run = {}
    with pytest.raises(MemoryError):
        campaign.wait_before_job(limits, 75., run)
    assert run['launch_wait_seconds'] == 75.
    assert run['launch_wait_sample_count'] == 76
    assert len(run['launch_wait_samples']) == 60
    assert run['launch_wait_samples_truncated'] is True
    assert run['launch_wait_samples'][-1]['elapsed_seconds'] == 75.
    assert all('reserve' in sample['reason'] for sample in run['launch_wait_samples'])


def test_fractional_timeout_does_not_sleep_past_deadline(window, monkeypatch):
    campaign, guard, clock, limits = window
    monkeypatch.setattr(guard, 'system_snapshot', lambda: snapshot(limits))
    run = {}
    with pytest.raises(MemoryError):
        campaign.wait_before_job(limits, .5, run)
    assert clock.sleeps == [.5]
    assert run['launch_wait_seconds'] == .5


@pytest.mark.parametrize('duration', [-1, float('nan'), float('inf'), True, None])
def test_invalid_wait_duration_is_rejected(window, duration):
    campaign, _, _, limits = window
    with pytest.raises(ValueError, match='finite and nonnegative'):
        campaign.wait_before_job(limits, duration, {})


def test_timeout_persists_stopped_campaign_without_launching_worker(tmp_path, window, monkeypatch):
    campaign, guard, clock, limits = window
    engine = dict(git_head='frozen-head', source_sha256={'engine.py': 'digest'})
    states = {str(campaign.ROOT): engine}
    tasks = [dict(id='task', prompt='No model will run', max_new_tokens=8)]
    plan = [dict(id='candidate', tasks=str(tmp_path / 'tasks.json'),
                 engine_root=str(campaign.ROOT), args=['--timeout', '77'])]
    output = tmp_path / 'campaign'
    monkeypatch.setattr(sys, 'argv', [
        'campaign_research51.py', '--model', str(tmp_path / 'model'),
        '--tasks', str(tmp_path / 'tasks.json'), '--plan', str(tmp_path / 'plan.json'),
        '--output', str(output), '--resource-wait-seconds', '3',
    ])
    monkeypatch.setattr(campaign, 'prepare_campaign', lambda args: (plan, states, {'protocol': 'digest'}, {'candidate': tasks}))
    monkeypatch.setattr(campaign, 'capture_engine_states', lambda sources: states)
    monkeypatch.setattr(campaign, 'protocol_unchanged', lambda protocol: True)
    monkeypatch.setattr(guard, 'system_snapshot', lambda: snapshot(limits, available=0))
    monkeypatch.setattr(campaign.subprocess, 'run',
                        lambda *args, **kwargs: pytest.fail('Timed-out window must not launch a worker'))
    with pytest.raises(MemoryError):
        campaign.main()
    report = json.loads((output / 'campaign.json').read_text())
    assert report['status'] == 'stopped'
    run = report['runs'][0]
    assert run['status'] == 'stopped'
    assert run['launch_wait_seconds'] == 3.
    assert run['launch_wait_status'] == 'timed_out'
    assert run['launch_wait_policy']['admission_limits'] == asdict(limits)
    assert 'exit_code' not in run
