"""External-supervisor controls only: injected snapshots, no MLX or model runs."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
import m5_autonomous_supervisor as supervisor
from airllm.nextgen.experiment_guard import ExperimentLimits, admission as retained_admission


def snapshot(pressure=1,available=5 << 30):
    return dict(pressure_level=pressure,available_bytes=available,
                physical_bytes=16 << 30,swap_used_bytes=2 << 30)


class Clock:
    def __init__(self):
        self.value = 0.
    def monotonic(self):
        return self.value
    def sleep(self,seconds):
        self.value += seconds


@pytest.mark.parametrize('pressure',[None,2,4])
def test_pressure_requires_explicit_normal(pressure):
    ready,reason = supervisor.ready(snapshot(pressure),ExperimentLimits())
    assert not ready
    assert reason


def test_headroom_is_additional_and_cannot_be_lowered():
    limits = ExperimentLimits()
    required = (3 << 30)+(768 << 20)+(256 << 20)
    assert not supervisor.ready(snapshot(available=required-1),limits)[0]
    assert supervisor.ready(snapshot(available=required),limits)[0]


def test_finite_wait_stops_without_any_worker():
    clock = Clock()
    samples,record = [],{}
    ready = supervisor.wait_for_resources(ExperimentLimits(),3.,record,
        lambda:snapshot(2),samples.append,monotonic=clock.monotonic,sleeper=clock.sleep)
    assert ready is False
    assert record['waited_seconds'] == 3.
    assert len(samples) == 4
    assert all(x['ready'] is False for x in samples)


def test_five_continuous_seconds_required_and_warning_resets_window():
    clock = Clock()
    samples,record = [],{}
    ready = supervisor.wait_for_resources(ExperimentLimits(),12.,record,
        lambda:snapshot(2 if clock.value == 3 else 1),samples.append,
        monotonic=clock.monotonic,sleeper=clock.sleep)
    assert ready is True
    assert record['waited_seconds'] == 9.
    assert record['continuous_ready_seconds'] == 5.
    assert samples[3]['continuous_ready_seconds'] == 0.


def test_short_wait_cannot_claim_stability():
    clock = Clock()
    samples,record = [],{}
    assert not supervisor.wait_for_resources(ExperimentLimits(),4.9,record,
        lambda:snapshot(),samples.append,monotonic=clock.monotonic,sleeper=clock.sleep)
    assert record['continuous_ready_seconds'] == 4.9


def fixture_inputs(tmp_path,monkeypatch):
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    (campaign/'config.json').write_text('{}')
    config = dict(limits=asdict(ExperimentLimits()))
    monkeypatch.setattr(supervisor,'preparation',lambda folder:(config,dict(status='deferred_or_interrupted')))
    monkeypatch.setattr(supervisor,'inference_observation',lambda **kwargs:dict(
        process_inventory_complete=True,other_model_processes=[],process_inventory_errors=[],
        process_inventory_uncertain=[]))
    args = Namespace(campaign=campaign,output_dir=tmp_path/'supervisor',
                     max_wait_seconds=.002,max_runtime_seconds=60.)
    return args,config


def test_real_short_control_wait_refuses_without_launch(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(supervisor,'system_snapshot',lambda:snapshot(2))
    monkeypatch.setattr(supervisor,'run_child',lambda *a:pytest.fail('A worker was launched under warning pressure'))
    code,state = supervisor.run(args)
    assert code == 75
    assert state['status'] == 'resource_wait_expired'
    assert state['campaign_launched'] is False
    assert state['sample_count'] >= 1
    lines = (args.output_dir/'snapshots.jsonl').read_text().splitlines()
    assert lines and all(json.loads(x)['pressure_level'] == 2 for x in lines)
    assert (args.output_dir/'manifest.json').is_file()


def admitted_simulation(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    calls = []
    # These are injected observations for controller-flow testing. There is no
    # GPU import, worker backend, lowered threshold or actual model subprocess.
    monkeypatch.setattr(supervisor,'wait_for_resources',lambda *a,**k:True)
    monkeypatch.setattr(supervisor,'admission',lambda limits:retained_admission(limits,snapshot()))
    def child(command,folder,**kwargs):
        calls.append(command)
        Path(folder).mkdir()
        if '--output' in command:
            Path(command[command.index('--output')+1]).write_text('{"status":"accepted_evidence"}')
        return 0
    monkeypatch.setattr(supervisor,'run_child',child)
    return args,config,calls


def test_admitted_flow_invokes_one_campaign_then_independent_verifier(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    code,state = supervisor.run(args)
    assert code == 0
    assert state['status'] == 'completed_evidence_verified'
    assert len(calls) == 2
    assert calls[0][2] == 'run'
    assert calls[0][-2:] == ['--resource-wait-seconds','60']
    assert Path(calls[1][1]).name == 'verify_m5_validation.py'
    assert state['identity']['retained_limits'] == asdict(ExperimentLimits())
    assert state['identity']['inter_worker_wait_seconds'] == 60
    # Terminal invocations are idempotent: launchd cannot repeat heavy work.
    code,again = supervisor.run(args)
    assert code == 0 and len(calls) == 2


def test_explicit_300_second_recovery_wait_is_forwarded_and_pinned_without_guard_changes(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    args.inter_worker_wait_seconds=300.
    timers=[]
    monkeypatch.setattr(supervisor.signal,'setitimer',lambda kind,seconds:timers.append(seconds))
    code,state=supervisor.run(args)
    assert code==0 and calls[0][-2:]==['--resource-wait-seconds','300']
    assert state['identity']['inter_worker_wait_seconds']==300.
    assert state['identity']['retained_limits']==asdict(ExperimentLimits())
    assert state['identity']['headroom_bytes']==256<<20
    assert state['identity']['disk_reserve_bytes']==5<<30
    assert state['identity']['max_runtime_seconds']==60.
    assert timers==[60.,0]
    persisted=json.loads((args.output_dir/'state.json').read_text())
    assert persisted['identity']['inter_worker_wait_seconds']==300.
    assert persisted['attempts'][0]['command'][-2:]==['--resource-wait-seconds','300']
    # The same frozen options are idempotent; changing the recovery policy may
    # not reuse a prior supervisor's evidence/output identity.
    assert supervisor.run(args)[0]==0 and len(calls)==2
    args.inter_worker_wait_seconds=60.
    with pytest.raises(RuntimeError,match='different frozen inputs/options'):
        supervisor.run(args)
    assert len(calls)==2


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),float('-inf'),3600.01,True,'300'])
def test_invalid_inter_worker_wait_is_refused_before_any_campaign_or_worker(value):
    with pytest.raises(ValueError,match='inter_worker_wait_seconds'):
        supervisor.run(Namespace(max_wait_seconds=10.,max_runtime_seconds=60.,
                                 inter_worker_wait_seconds=value))


@pytest.mark.parametrize('value',[.01,3600.])
def test_positive_bounded_inter_worker_wait_endpoints_are_accepted(tmp_path,monkeypatch,value):
    args,config,calls=admitted_simulation(tmp_path,monkeypatch)
    args.inter_worker_wait_seconds=value
    code,state=supervisor.run(args)
    assert code==0
    assert float(calls[0][-1])==value
    assert state['identity']['inter_worker_wait_seconds']==value


def test_campaign_resource_refusal_has_no_retry(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    def refuse(command,folder,**kwargs):
        calls.append(command)
        return 75
    monkeypatch.setattr(supervisor,'run_child',refuse)
    code,state = supervisor.run(args)
    assert code == 75 and state['status'] == 'campaign_deferred_resources'
    assert len(calls) == 1


def test_runtime_timeout_is_terminal_and_keeps_evidence(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    def timeout(*a,**kwargs):
        raise supervisor.RuntimeBudgetExpired('test-control timer')
    monkeypatch.setattr(supervisor,'run_child',timeout)
    code,state = supervisor.run(args)
    assert code == 124 and state['status'] == 'runtime_budget_expired'
    assert (args.output_dir/'state.json').is_file()
    assert (args.output_dir/'manifest.json').is_file()


def test_verification_failure_never_reports_success(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    def reject(command,folder,**kwargs):
        calls.append(command)
        return 0 if len(calls) == 1 else 1
    monkeypatch.setattr(supervisor,'run_child',reject)
    code,state = supervisor.run(args)
    assert code == 1 and state['status'] == 'verification_failed'


def test_outputs_may_not_modify_campaign(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    args.output_dir = args.campaign/'supervisor'
    with pytest.raises(ValueError,match='outside'):
        supervisor.run(args)


@pytest.mark.parametrize('value',[0.,-1.,float('inf'),float('nan')])
def test_unbounded_or_invalid_wait_rejected(tmp_path,value):
    with pytest.raises(ValueError,match='finite'):
        supervisor.run(Namespace(max_wait_seconds=value,max_runtime_seconds=60.))


def test_supervisor_excludes_second_concurrent_owner(tmp_path):
    with supervisor.exclusive_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another'):
            with supervisor.exclusive_lock(tmp_path):
                pass


def test_source_changes_during_wait_prevent_launch(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    actual = supervisor.digest
    checks = []
    def changing(path):
        if str(path) == supervisor.__file__:
            checks.append(path)
            return '0'*64 if len(checks) == 1 else '1'*64
        return actual(path)
    monkeypatch.setattr(supervisor,'digest',changing)
    monkeypatch.setattr(supervisor,'system_snapshot',lambda:snapshot())
    monkeypatch.setattr(supervisor,'run_child',lambda *a:pytest.fail('changed source launched a worker'))
    with pytest.raises(RuntimeError,match='changed'):
        supervisor.run(args)
    state = json.loads((args.output_dir/'state.json').read_text())
    assert state['status'] == 'inputs_changed'
    assert state['campaign_launched'] is False


def test_running_crash_leases_recovered_before_campaign_lock(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    args.output_dir.mkdir()
    identity = dict(supervisor_sha256=supervisor.digest(supervisor.__file__),campaign=str(args.campaign),
                    campaign_config_sha256=supervisor.digest(args.campaign/'config.json'))
    supervisor.atomic_json(args.output_dir/'state.json',dict(identity=identity,status='campaign_running',campaign_launched=True))
    child = args.output_dir/'campaign-controller'
    child.mkdir()
    supervisor.atomic_json(child/'owned-processes.json',dict(processes=[dict(pid=999999,created=1.)]))
    calls = []
    monkeypatch.setattr(supervisor,'owned_cleanup',lambda identities:calls.append(identities))
    monkeypatch.setattr(supervisor,'preparation',lambda *a:pytest.fail('Orphan cleanup must precede the held campaign lock'))
    code,state = supervisor.run(args)
    assert code == 130 and state['status'] == 'interrupted'
    assert calls == [[dict(pid=999999,created=1.)]]


def test_terminal_evidence_corruption_prevents_idempotent_success(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    assert supervisor.run(args)[0] == 0
    (args.output_dir/'verification.json').write_text('{"status":"changed"}')
    with pytest.raises(RuntimeError,match='changed'):
        supervisor.run(args)
    assert len(calls) == 2


def test_completed_flow_has_one_finite_runtime_timer(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    timers = []
    monkeypatch.setattr(supervisor.signal,'setitimer',lambda kind,seconds:timers.append(seconds))
    code,state = supervisor.run(args)
    assert code == 0
    assert timers == [60.,0]


def test_low_disk_never_launches_owned_campaign(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    monkeypatch.setattr(supervisor,'disk_snapshot',lambda paths:dict(disk_free_bytes=(5 << 30)-1,
        disk_reserve_bytes=5 << 30,disk_free_by_path={str(args.campaign):(5 << 30)-1}))
    code,state = supervisor.run(args)
    assert code == 75 and state['status'] == 'disk_reserve'
    assert state['campaign_launched'] is False and calls == []


def test_runtime_disk_monitor_stops_only_through_owned_child_cleanup(tmp_path,monkeypatch):
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    samples = []
    def disks(paths):
        samples.append(paths)
        return dict(disk_free_bytes=6 << 30 if len(samples) == 1 else 4 << 30,
                    disk_reserve_bytes=5 << 30,disk_free_by_path={})
    monkeypatch.setattr(supervisor,'disk_snapshot',disks)
    def child(command,folder,monitor=None):
        calls.append(command)
        assert monitor is not None
        monitor()
        pytest.fail('Disk reserve violation did not interrupt owned child control')
    monkeypatch.setattr(supervisor,'run_child',child)
    code,state = supervisor.run(args)
    assert code == 75 and state['status'] == 'disk_reserve'
    assert len(calls) == 1 and state['last_disk_snapshot']['disk_free_bytes'] == 4 << 30


def test_uncertain_inventory_defers_without_worker_and_is_bounded():
    clock = Clock()
    samples,record = [],{}
    observation = dict(snapshot(),process_inventory_complete=False,process_inventory_uncertain=[dict(pid=4)])
    assert not supervisor.wait_for_resources(ExperimentLimits(),3.,record,
        lambda:observation,samples.append,monotonic=clock.monotonic,sleeper=clock.sleep)
    assert record['waited_seconds'] == 3.
    assert all('unavailable' in row['reason'] for row in samples)


def test_other_model_blocks_admission_without_signals():
    ready, reason = supervisor.ready(dict(snapshot(),other_model_processes=[dict(pid=4,created=10.)]),ExperimentLimits())
    assert not ready and 'Another model' in reason


def test_concurrent_output_directories_cannot_claim_same_campaign(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    args.output_dir = tmp_path/'another-output'
    with supervisor.campaign_supervisor_lock(args.campaign):
        with pytest.raises(RuntimeError,match='Another supervisor'):
            supervisor.run(args)
    assert not (args.output_dir/'state.json').exists()


def test_interrupted_wait_keeps_original_deadline_and_resumes(tmp_path,monkeypatch):
    args,config = fixture_inputs(tmp_path,monkeypatch)
    args.max_wait_seconds = 5.
    monkeypatch.setattr(supervisor,'system_snapshot',lambda:snapshot(2))
    original = supervisor.wait_for_resources
    def interrupted(limits,seconds,state,observe,persist):
        persist(dict(observe(),ready=False,reason='simulated interruption'))
        raise RuntimeError('simulated process crash')
    monkeypatch.setattr(supervisor,'wait_for_resources',interrupted)
    # Simulate an abrupt death after a persisted waiting state by restoring only
    # that saved state; no completed result is relabeled or reused.
    captured = []
    real_atomic = supervisor.atomic_json
    def capture(path,state):
        real_atomic(path,state)
        if Path(path).name == 'state.json' and state.get('status') == 'waiting_for_resources':
            captured.append(json.loads(Path(path).read_text()))
    monkeypatch.setattr(supervisor,'atomic_json',capture)
    with pytest.raises(RuntimeError,match='simulated'):
        supervisor.run(args)
    waiting = captured[-1]
    deadline = waiting['wait_deadline_unix']
    real_atomic(args.output_dir/'state.json',waiting)
    (args.output_dir/'manifest.json').unlink()
    remaining = []
    def expired(limits,seconds,state,*rest,**kwargs):
        remaining.append(seconds)
        return False
    monkeypatch.setattr(supervisor,'wait_for_resources',expired)
    code,state = supervisor.run(args)
    assert code == 75 and state['wait_deadline_unix'] == deadline
    assert 0 < remaining[0] <= args.max_wait_seconds


def test_atomic_state_write_failure_keeps_original_and_removes_temp(tmp_path):
    path = tmp_path/'state.json'
    supervisor.atomic_json(path,dict(status='valid'))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        supervisor.atomic_json(path,dict(value=float('nan')))
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ['state.json']


def test_new_owned_descendant_is_reconciled_before_persisted_lease(tmp_path,monkeypatch):
    import os
    args,config,calls = admitted_simulation(tmp_path,monkeypatch)
    actual = supervisor.psutil.Process
    own_birth = actual().create_time()
    spawned = []
    child_pid, child_birth = 9999999, own_birth+1
    class Own:
        pid = os.getpid()
        def create_time(self):
            return own_birth
        def children(self,recursive=False):
            return [Child()] if spawned else []
    class Child:
        pid = child_pid
        def create_time(self):
            return child_birth
        def parents(self):
            return [Own()]
    monkeypatch.setattr(supervisor.psutil,'Process',lambda pid=None:Child() if pid == child_pid else Own())
    def observation(**kwargs):
        return dict(process_inventory_complete=True,process_inventory_uncertain=[],
                    other_model_processes=[dict(pid=child_pid,created=child_birth)] if spawned else [])
    monkeypatch.setattr(supervisor,'inference_observation',observation)
    def child(command,folder,monitor=None):
        calls.append(command)
        Path(folder).mkdir()
        if len(calls) == 1:
            spawned.append(True)
            assert not list(args.output_dir.glob('**/owned-processes.json'))
            monitor()
        else:
            Path(command[command.index('--output')+1]).write_text('{"status":"accepted_evidence"}')
        return 0
    monkeypatch.setattr(supervisor,'run_child',child)
    code,state = supervisor.run(args)
    assert code == 0 and state['status'] == 'completed_evidence_verified'
    assert state['last_disk_snapshot']['other_model_processes'] == []
