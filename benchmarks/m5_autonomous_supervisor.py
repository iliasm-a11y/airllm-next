#!/usr/bin/env python3
"""One-shot external resource waiter and launcher for a frozen M5 campaign.

Suitable for a user LaunchAgent: this process has no dependency on Codex after
launch. It closes no applications, changes no guard and retries no heavy worker.
The existing campaign owns inference, kernel validation, hashes and resumption.
Between workers it observes resource recovery for 60 seconds by default, with
an explicit finite option bounded at one hour. The same five-second stable
window and all existing admission/runtime limits remain in force.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
from tempfile import NamedTemporaryFile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'air_llm'))
from airllm.nextgen.experiment_guard import ExperimentLimits, admission, system_snapshot
from m5_validation import (campaign_lock, digest, owned_cleanup, run_child,
                           unchanged)
from verify_m5_validation import validate_config, verify_manifest
from m5_process_inventory import inference_observation, expected_read_error
import psutil

HEADROOM_BYTES = 256 << 20
STABLE_SECONDS = 5.
POLL_SECONDS = 1.
INTER_WORKER_WAIT_SECONDS = 60
DISK_RESERVE_BYTES = 5 << 30
TERMINAL = {'completed_evidence_verified': 0, 'resource_wait_expired': 75,
            'campaign_deferred_resources': 75, 'campaign_failed': 1,
            'verification_failed': 1, 'runtime_budget_expired': 124,
            'interrupted': 130, 'inputs_changed': 1, 'disk_reserve': 75}


class RuntimeBudgetExpired(TimeoutError):
    pass


class DiskReserveError(OSError):
    pass


def disk_snapshot(paths):
    records = {str(path):shutil.disk_usage(path).free for path in paths}
    return dict(disk_free_bytes=min(records.values()),disk_free_by_path=records,
                disk_reserve_bytes=DISK_RESERVE_BYTES)


def require_disk(paths):
    snapshot = disk_snapshot(paths)
    if snapshot['disk_free_bytes'] < DISK_RESERVE_BYTES:
        raise DiskReserveError('disk_reserve: less than the retained 5 GiB disk floor remains')
    return snapshot


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_lock(folder):
    lock = Path(folder)/'.supervisor.lock'
    if lock.is_symlink():
        raise ValueError('Supervisor lock cannot be a symlink')
    with lock.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another external supervisor owns this output folder') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def campaign_supervisor_lock(campaign):
    """One waiter/owner per campaign, even with different output directories.

    The lock lives beside the immutable evidence folder; it is never unlinked.
    The child's existing runner lock remains an independent final launch guard.
    """
    campaign = Path(campaign).resolve()
    key = hashlib.sha256(os.fsencode(str(campaign))).hexdigest()
    path = campaign.parent/('.m5-supervisor-'+key+'.lock')
    if path.is_symlink():
        raise ValueError('Campaign supervisor lock cannot be a symlink')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another supervisor owns this campaign') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def preparation(campaign):
    """Read-only validation, including current code and model stat continuity."""
    campaign = Path(campaign).resolve(strict=True)
    with campaign_lock(campaign):
        config = json.loads((campaign/'config.json').read_text())
        state = json.loads((campaign/'state.json').read_text())
        if config.get('origin') != str(campaign):
            raise ValueError('Execution must use the original prepared campaign folder')
        validate_config(config)
        if config['engine_roots']['candidate'] != str(ROOT):
            raise ValueError('External supervisor must use its own frozen candidate checkout')
        for relative in ('benchmarks/m5_validation.py','benchmarks/verify_m5_validation.py',
                         'benchmarks/validate_m5_kernels.py'):
            path = ROOT/relative
            if config['protocol_sha256'].get(str(path)) != digest(path):
                raise RuntimeError('Actual supervisor child entrypoint is outside the frozen protocol: '+relative)
        if state.get('config_sha256') != digest(campaign/'config.json'):
            raise RuntimeError('Campaign configuration identity changed')
        verify_manifest(campaign, allow_interrupted=state.get('status') == 'running')
        unchanged(config)
    return config, state


def ready(snapshot, limits):
    """The waiter only adds safety headroom to the retained global admission."""
    try:
        admission(limits, snapshot)
        if snapshot.get('pressure_level') != 1:
            raise MemoryError('Observed macOS pressure must be exactly normal (1)')
        required = limits.reserve_bytes+limits.admitted_payload_bytes+HEADROOM_BYTES
        if snapshot['available_bytes'] < required:
            raise MemoryError('Stable launch requires reserve + payload + 256 MiB headroom')
        if snapshot.get('disk_free_bytes',DISK_RESERVE_BYTES) < DISK_RESERVE_BYTES:
            raise MemoryError('Stable launch requires at least 5 GiB free disk space')
        if not snapshot.get('process_inventory_complete', True):
            raise MemoryError('Same-user inference process identity is unavailable; admission deferred')
        if snapshot.get('other_model_processes'):
            raise MemoryError('Another model or campaign process is active; admission deferred')
    except (MemoryError, KeyError) as error:
        return False, str(error)
    return True, None


def wait_for_resources(limits, max_seconds, record, observe, persist,
                       *, monotonic=time.monotonic, sleeper=time.sleep):
    """Bounded polling, injectable for tests; never executes an inference path."""
    start = monotonic()
    stable_since = None
    while True:
        snapshot = observe()
        now = monotonic()
        elapsed = now-start
        allowed, reason = ready(snapshot, limits)
        stable_since = now if allowed and stable_since is None else stable_since
        if not allowed:
            stable_since = None
        stable = 0. if stable_since is None else now-stable_since
        sample = dict(snapshot, elapsed_seconds=elapsed, ready=allowed,
                      reason=reason, continuous_ready_seconds=stable)
        persist(sample)
        record.update(sample_count=record.get('sample_count',0)+1,
                      last_snapshot=snapshot, waited_seconds=elapsed,
                      continuous_ready_seconds=stable)
        if stable >= STABLE_SECONDS and elapsed <= max_seconds:
            return True
        if elapsed >= max_seconds:
            return False
        sleeper(min(POLL_SECONDS, max_seconds-elapsed))


def _interrupted(signum, frame):
    raise KeyboardInterrupt('External supervisor received signal '+str(signum))


def _runtime_expired(signum, frame):
    raise RuntimeBudgetExpired('The external campaign runtime budget expired')


def write_file_manifest(folder):
    folder = Path(folder)
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name not in ('manifest.json','.supervisor.lock'):
            if path.is_symlink():
                raise ValueError('Supervisor evidence cannot contain symlinks')
            files[str(path.relative_to(folder))] = dict(sha256=digest(path), bytes=path.stat().st_size)
    atomic_json(folder/'manifest.json', dict(schema_version=1, files=files,
                                           authenticity_certified=False))


def verify_file_manifest(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    files = manifest.get('files')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
              and p.name not in ('manifest.json','.supervisor.lock')}
    if not isinstance(files,dict) or actual != set(files):
        raise RuntimeError('External supervisor evidence manifest is incomplete or changed')
    for relative,value in files.items():
        path = (folder/relative).resolve(strict=True)
        if (Path(relative).is_absolute() or not path.is_relative_to(folder.resolve())
                or (folder/relative).is_symlink() or path.stat().st_size != value['bytes']
                or digest(path) != value['sha256']):
            raise RuntimeError('External supervisor evidence changed: '+relative)


def run(args):
    for name in ('max_wait_seconds','max_runtime_seconds'):
        value = getattr(args,name)
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be finite and positive')
    inter_worker_wait = getattr(args,'inter_worker_wait_seconds',INTER_WORKER_WAIT_SECONDS)
    if (type(inter_worker_wait) not in (int,float) or not math.isfinite(inter_worker_wait)
            or not 0 < inter_worker_wait <= 3600):
        raise ValueError('inter_worker_wait_seconds must be finite, positive and at most3600')
    wait_value=float(inter_worker_wait)
    wait_argument=str(int(wait_value)) if wait_value.is_integer() else repr(wait_value)
    campaign = args.campaign.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    if output.is_symlink():
        raise ValueError('Supervisor output cannot be a symlink')
    output = output.resolve()
    if output == campaign or output.is_relative_to(campaign):
        raise ValueError('Supervisor output must be outside the immutable campaign evidence folder')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output), campaign_supervisor_lock(campaign):
        if any(p.is_symlink() for p in output.rglob('*')):
            raise ValueError('Supervisor evidence cannot contain symlinks')
        # Recover our observed orphan leases before asking for the campaign
        # lock, which a surviving child controller may still hold after a crash.
        previous_path = output/'state.json'
        if previous_path.is_file():
            previous = json.loads(previous_path.read_text())
            prior_identity = previous.get('identity',{})
            if (previous.get('status') not in set(TERMINAL)|{'waiting_for_resources'}
                    and prior_identity.get('supervisor_sha256') == digest(__file__)
                    and prior_identity.get('campaign') == str(campaign)
                    and prior_identity.get('campaign_config_sha256') == digest(campaign/'config.json')):
                for lease in output.glob('**/owned-processes.json'):
                    owned_cleanup(json.loads(lease.read_text())['processes'])
                previous.update(status='interrupted',reason='Recovered observed child leases after supervisor interruption; start a new output folder to resume the sealed campaign')
                atomic_json(previous_path,previous)
                write_file_manifest(output)
                return 130,previous
        config, prior = preparation(campaign)
        identity = dict(supervisor_sha256=digest(__file__),
            campaign=str(campaign), campaign_config_sha256=digest(campaign/'config.json'),
            interpreter=str(Path(sys.executable).resolve()), max_wait_seconds=float(args.max_wait_seconds),
            max_runtime_seconds=float(args.max_runtime_seconds),
            required_pressure_level=1, stable_seconds=STABLE_SECONDS,
            poll_seconds=POLL_SECONDS, headroom_bytes=HEADROOM_BYTES,
            inter_worker_wait_seconds=float(inter_worker_wait),
            retained_limits=config['limits'])
        identity['disk_reserve_bytes'] = DISK_RESERVE_BYTES
        identity['process_inventory_sha256'] = digest(ROOT/'benchmarks/m5_process_inventory.py')
        state_path = output/'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('identity') != identity:
                raise RuntimeError('Existing supervisor evidence belongs to different frozen inputs/options')
            if state.get('status') in TERMINAL:
                verify_file_manifest(output)
                return TERMINAL[state['status']], state
            # A launchd restart/manual rerun resumes only the existing bounded
            # wait. A crashed running campaign is stopped, never blindly retried.
            if state.get('status') != 'waiting_for_resources':
                for lease in output.glob('**/owned-processes.json'):
                    owned_cleanup(json.loads(lease.read_text())['processes'])
                state.update(status='interrupted', reason='Previous controller interrupted; use a new supervisor output folder to resume the sealed campaign')
                atomic_json(state_path,state)
                write_file_manifest(output)
                return 130,state
        else:
            if any(p.name != '.supervisor.lock' for p in output.iterdir()):
                raise FileExistsError('Unrecognized existing supervisor output files')
            started = time.time()
            state = dict(schema_version=1, identity=identity, status='waiting_for_resources',
                started_unix=started, wait_deadline_unix=started+args.max_wait_seconds,
                sample_count=0, campaign_launched=False,
                prior_campaign_status=prior.get('status'), attempts=[])
            atomic_json(state_path,state)
        old_term = signal.signal(signal.SIGTERM,_interrupted)
        old_alarm = signal.signal(signal.SIGALRM,_runtime_expired)
        try:
            remaining = max(0.,state['wait_deadline_unix']-time.time())
            limits = ExperimentLimits(**config['limits'])
            frozen_config = identity['campaign_config_sha256']
            frozen_source = identity['supervisor_sha256']
            frozen_inventory = identity['process_inventory_sha256']
            own_identity = dict(pid=os.getpid(), created=psutil.Process().create_time())
            def process_observation():
                # Only observed PID+birth-time leases may be excluded. A reused
                # PID is a different process and remains visible to admission.
                excluded = [own_identity]
                for lease in output.glob('**/owned-processes.json'):
                    excluded.extend(json.loads(lease.read_text())['processes'])
                observed = inference_observation(excluded=excluded)
                # A guard may fork after run_child persisted its last lease.
                # Reconcile *after* scanning so a newly observed descendant is
                # not misclassified as a competing campaign. Trust only a live
                # ancestry chain reaching our exact PID/birth-time identity,
                # plus a fresh closing child birth-time check.
                try:
                    owner = psutil.Process(own_identity['pid'])
                    descendants = owner.children(recursive=True) if owner.create_time() == own_identity['created'] else []
                except Exception as error:
                    if not expected_read_error(error):
                        raise
                    descendants = []
                for child in descendants:
                    try:
                        born = child.create_time()
                        ancestors = child.parents()
                        if (any(p.pid == own_identity['pid'] and p.create_time() == own_identity['created'] for p in ancestors)
                            and psutil.Process(child.pid).create_time() == born):
                            excluded.append(dict(pid=child.pid,created=born))
                    except Exception as error:
                        if not expected_read_error(error):
                            raise
                leases = {(row['pid'],row['created']) for row in excluded}
                for key in ('other_model_processes','process_inventory_uncertain'):
                    observed[key] = [row for row in observed.get(key,[])
                                     if (row.get('pid'),row.get('created')) not in leases]
                observed['process_inventory_complete'] = not observed.get('process_inventory_uncertain')
                return observed
            def frozen():
                return (digest(campaign/'config.json') == frozen_config
                        and digest(__file__) == frozen_source
                        and digest(ROOT/'benchmarks/m5_process_inventory.py') == frozen_inventory)
            with (output/'snapshots.jsonl').open('a') as log:
                def persist(sample):
                    # No checkpoint reads or GPU contexts occur while waiting.
                    if not frozen():
                        raise RuntimeError('Frozen supervisor or campaign configuration changed while waiting')
                    sample['unix_seconds'] = time.time()
                    log.write(json.dumps(sample,sort_keys=True,allow_nan=False)+'\n')
                    log.flush()
                    os.fsync(log.fileno())
                    # wait_for_resources updates counters immediately after this
                    # callback; preserve the current observation before waiting.
                    state['last_observation'] = sample
                    state['last_updated_unix'] = sample['unix_seconds']
                    atomic_json(state_path,state)
                def observe():
                    return dict(system_snapshot(),**disk_snapshot([campaign,output]), **process_observation())
                available = wait_for_resources(limits,remaining,state,observe,persist)
            if not available:
                state.update(status='resource_wait_expired', reason='No stable normal-pressure resource window within the finite wait budget')
                return 75,state
            # Revalidate every frozen dependency after the external wait. The
            # campaign performs a fresh independent admission before its kernels.
            preparation(campaign)
            admission(limits)
            processes = process_observation()
            if processes['other_model_processes'] or not processes['process_inventory_complete']:
                state['process_admission_refusal'] = processes
                raise MemoryError('Process inventory changed or another model is active before launch')
            state['disk_before_launch'] = require_disk([campaign,output])
            if not frozen():
                raise RuntimeError('External supervisor source changed')
            state.update(status='campaign_running', campaign_launched=True,
                         campaign_started_unix=time.time())
            command = [sys.executable,str(ROOT/'benchmarks/m5_validation.py'),'run',
                       '--campaign',str(campaign),'--resource-wait-seconds',wait_argument]
            state['attempts'].append(dict(command=command, started_unix=time.time()))
            atomic_json(state_path,state)
            # This timer bounds the complete child/controller/verification phase;
            # per-worker 600s deadlines remain unchanged inside the campaign.
            signal.setitimer(signal.ITIMER_REAL,args.max_runtime_seconds)
            child = output/'campaign-controller'
            last_disk_sample = -math.inf
            with (output/'disk-snapshots.jsonl').open('a') as disk_log:
                def monitor():
                    nonlocal last_disk_sample
                    now = time.monotonic()
                    if now-last_disk_sample >= POLL_SECONDS:
                        last_disk_sample = now
                        observed = disk_snapshot([campaign,output])
                        observed.update(process_observation())
                        observed['unix_seconds'] = time.time()
                        disk_log.write(json.dumps(observed,sort_keys=True,allow_nan=False)+'\n')
                        disk_log.flush()
                        os.fsync(disk_log.fileno())
                        state['last_disk_snapshot'] = observed
                        if observed['disk_free_bytes'] < DISK_RESERVE_BYTES:
                            raise DiskReserveError('disk_reserve: less than 5 GiB free during the owned campaign')
                        if observed['other_model_processes'] or not observed['process_inventory_complete']:
                            raise MemoryError('A competing or unidentified inference process appeared')
                        if not frozen():
                            raise RuntimeError('Frozen supervisor/config/inventory changed while running')
                code = run_child(command,child,monitor=monitor)
            state['attempts'][-1].update(exit_code=code,ended_unix=time.time())
            if code:
                state.update(status='campaign_deferred_resources' if code == 75 else 'campaign_failed',
                             reason='The guarded campaign stopped; raw logs and sealed results are retained')
                return (75 if code == 75 else 1),state
            state['status'] = 'verifying_evidence'
            atomic_json(state_path,state)
            verification = output/'verification.json'
            verify_command = [sys.executable,str(ROOT/'benchmarks/verify_m5_validation.py'),
                              str(campaign),'--output',str(verification)]
            code = run_child(verify_command,output/'independent-verification',
                             monitor=lambda:require_disk([campaign,output]))
            state['verification'] = dict(command=verify_command,exit_code=code,
                report=str(verification) if verification.exists() else None)
            if code:
                state.update(status='verification_failed',reason='Independent verification rejected incomplete or changed evidence')
                return 1,state
            state.update(status='completed_evidence_verified', campaign_completed_unix=time.time(),
                         verification_sha256=digest(verification))
            return 0,state
        except MemoryError as error:
            state.update(status='campaign_deferred_resources',reason=str(error))
            return 75,state
        except DiskReserveError as error:
            state.update(status='disk_reserve',reason=str(error))
            return 75,state
        except RuntimeBudgetExpired as error:
            state.update(status='runtime_budget_expired',reason=str(error))
            return 124,state
        except KeyboardInterrupt as error:
            state.update(status='interrupted',reason=str(error))
            return 130,state
        except BaseException as error:
            state.update(status='inputs_changed' if 'changed' in str(error) else 'campaign_failed',
                         reason=f'{type(error).__name__}: {error}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGTERM,old_term)
            signal.signal(signal.SIGALRM,old_alarm)
            state['ended_unix'] = time.time()
            state['supervisor_source_unchanged'] = digest(__file__) == identity['supervisor_sha256']
            if not state['supervisor_source_unchanged']:
                state.update(status='inputs_changed',reason='External supervisor source changed during this attempt')
            atomic_json(state_path,state)
            write_file_manifest(output)
            if not state['supervisor_source_unchanged']:
                raise RuntimeError('External supervisor source changed during this attempt')


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-wait-seconds',type=float,default=1800.)
    p.add_argument('--max-runtime-seconds',type=float,default=28800.)
    p.add_argument('--inter-worker-wait-seconds',type=float,default=INTER_WORKER_WAIT_SECONDS,
                   help='Finite resource-recovery observation between workers, 0<seconds<=3600; included in the unchanged global runtime budget')
    args = p.parse_args()
    code,state = run(args)
    print(json.dumps(dict(status=state['status'],campaign=state['identity']['campaign'],
        campaign_launched=state['campaign_launched'],evidence=str(args.output_dir.resolve()),
        reason=state.get('reason')),ensure_ascii=False))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
