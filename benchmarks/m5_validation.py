#!/usr/bin/env python3
"""Freeze, execute and resume a sequential M5 hardware validation campaign.

This is a controller, not an inference backend. Every hash and model worker uses
the retained experiment_guard; pressure/reserve/RSS/swap limits are unchanged.
No daemon, model download, cache purge or application shutdown is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

from benchmark_research51 import (ROOT, engine_manifest, load_tasks,
                                 protocol_manifest, protocol_unchanged, save)
from campaign_research51 import validate_job_args, wait_before_job
from plan_hyperstream3 import build_experiments, apply_profile_jobs, PROFILES
from m5_process_inventory import expected_read_error
from m5_manifest_recovery import inspect_index, validate_recovery_records


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def model_identity(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


@contextmanager
def campaign_lock(folder):
    """A single retained inode is locked; never unlink it while a runner lives."""
    path = Path(folder) / '.runner.lock'
    if path.is_symlink():
        raise ValueError('Campaign lock may not be a symlink')
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another runner holds this campaign lock') from error
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def validate_lease_identities(identities):
    if not isinstance(identities,list):
        raise ValueError('Invalid saved process identities')
    for item in identities:
        if (not isinstance(item,dict) or type(item.get('pid')) is not int or item['pid'] <= 0
                or type(item.get('created')) not in (int,float)
                or not math.isfinite(item['created']) or item['created'] <= 0):
            raise ValueError('Invalid saved process identity')


def owned_cleanup(identities):
    """Only saved PID + creation-time identities from our child tree are eligible."""
    import psutil
    validate_lease_identities(identities)
    owned = []
    for item in identities:
        try:
            p = psutil.Process(item['pid'])
            if p.create_time() == item['created'] and p.pid != os.getpid():
                p.terminate()
                owned.append(p)
        except Exception as error:
            if not expected_read_error(error):
                raise
    _, alive = psutil.wait_procs(owned, timeout=2)
    for p in alive:
        try:
            expected = next(x['created'] for x in identities if x['pid'] == p.pid)
            # create_time on the old psutil instance is cached. A fresh handle
            # must confirm the saved lease before escalation after the wait.
            current = psutil.Process(p.pid)
            if current.create_time() == expected:
                current.kill()
        except Exception as error:
            if not expected_read_error(error):
                raise
    _, alive = psutil.wait_procs(alive, timeout=3)
    if alive:
        raise RuntimeError('Owned child cleanup did not complete')


def require_leases_quiescent(identities):
    """A denied optional cleanup read is not proof an owned writer has stopped."""
    import psutil
    validate_lease_identities(identities)
    for item in identities:
        try:
            process=psutil.Process(item['pid'])
            if process.create_time()!=item['created']:
                continue  # Reused PID belongs to someone else; never signal it.
            if process.status() not in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD) and process.is_running():
                raise RuntimeError('Owned campaign writer remains alive before index recovery')
        except (psutil.NoSuchProcess,ProcessLookupError):
            continue
        except Exception as error:
            if not expected_read_error(error):
                raise
            raise RuntimeError('Owned campaign writer identity is unavailable before index recovery') from error


def run_child(command, folder, monitor=None):
    """Track wrapper and worker identities, including independently-sessioned workers.

    The retained supervisor handles normal errors. This outer lease additionally
    cleans its observed descendants on Ctrl-C/SIGTERM and on the next resume after
    a controller crash. Abrupt SIGKILL during process creation remains an OS limit.
    """
    import psutil
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lease = folder / 'owned-processes.json'
    observed = {}
    diagnostic_errors = []
    diagnostic_counts = {}
    def diagnostic(pid,field,error):
        key = field+':'+type(error).__name__
        diagnostic_counts[key] = diagnostic_counts.get(key,0)+1
        # Aggregate repeated OS denials at 20 Hz; long bounded campaigns must
        # not grow an unbounded in-memory list of optional observations.
        if len(diagnostic_errors) < 64:
            row = dict(pid=pid,field=field,error_type=type(error).__name__)
            if row not in diagnostic_errors:
                diagnostic_errors.append(row)
    process = None
    with (folder / 'controller-child.log').open('x') as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, MLX_ENABLE_TF32='0', PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
            try:
                owner = psutil.Process(process.pid)
            except Exception as error:
                if not expected_read_error(error):
                    raise
                owner = None
                diagnostic(process.pid,'root_identity',error)
            while True:
                try:
                    members = [] if owner is None else [owner]
                    if owner is not None:
                        try:
                            members.extend(owner.children(recursive=True))
                        except Exception as error:
                            if not expected_read_error(error):
                                raise
                            diagnostic(owner.pid,'children',error)
                    changed = False
                    for p in members:
                        try:
                            identity = p.create_time()
                            if observed.get(p.pid) != identity:
                                observed[p.pid] = identity
                                changed = True
                        except Exception as error:
                            if not expected_read_error(error):
                                raise
                            diagnostic(p.pid,'birth_time',error)
                    if changed:
                        save(lease, dict(controller_pid=os.getpid(), processes=[
                            dict(pid=p, created=t) for p, t in observed.items()]))
                except Exception as error:
                    if not expected_read_error(error):
                        raise
                    diagnostic(process.pid,'child_inventory',error)
                if monitor is not None:
                    monitor()
                code = process.poll()
                if code is not None:
                    return code
                time.sleep(.05)
        finally:
            if process is not None:
                # Popen itself is a trusted unreaped-child lease, even when a
                # signal arrived before psutil's first successful observation.
                sys.path.insert(0, str(ROOT/'air_llm'))
                from airllm.nextgen.experiment_guard import _stop_group
                _stop_group(process, observed)
            owned_cleanup([dict(pid=p, created=t) for p, t in observed.items()])
            save(folder / 'cleanup.json', dict(status='completed', observed_processes=len(observed),
                                               process_inventory_errors=diagnostic_errors,
                                               process_inventory_error_counts=diagnostic_counts))


def worker_resource_refusal(output, command, limits):
    """Recognize only measured resource stops in this attempt's own raw guard.

    A wrapper's nonzero code alone cannot distinguish a temporary admission
    refusal from a programming error. Malformed guard evidence remains fatal.
    No checkpoint/logit/route acceptance criterion is changed by this classifier.
    """
    output = Path(output)
    guard_path = output.with_suffix('.guard.json')
    if not guard_path.is_file():
        return None
    guard = json.loads(guard_path.read_text())
    if (guard.get('command') != command+['--worker'] or guard.get('limits') != limits
            or guard.get('cleanup_error')):
        return None
    if output.is_file():
        worker = json.loads(output.read_text())
        if worker.get('status') in ('failed','invalid') or worker.get('error'):
            return None
        if any(worker.get(flag) is False for flag in (
                'source_unchanged','protocol_unchanged','model_stat_unchanged','engine_head_unchanged')):
            return None
    violation = guard.get('violation')
    exception = guard.get('exception')
    if exception is not None:
        # An admission failure happens before any child is created. Refuse to
        # reinterpret arbitrary exceptions or exceptions raised during a run.
        before = guard.get('before',{})
        known = {
            'MemoryError: macOS memory pressure is above normal before launch':
                before.get('pressure_level') in (2,4),
            'MemoryError: Requested payload exceeds available memory after system reserve':
                before.get('available_bytes', float('inf')) < limits['reserve_bytes']+limits['admitted_payload_bytes'],
        }
        if (known.get(exception) is not True or guard.get('sample_count') != 0
                or guard.get('exit_code') is not None or violation is not None):
            return None
        reason = 'worker_admission_resources'
    else:
        measured = {
            'macos_memory_pressure': guard.get('peak_pressure_level') in (2,4),
            'system_reserve': guard.get('minimum_available_bytes',float('inf')) < limits['reserve_bytes'],
            'system_swap_growth': guard.get('maximum_swap_growth_bytes',0) > limits['swap_growth_bytes'],
        }
        if (measured.get(violation) is not True or guard.get('exit_code') not in (-15,-9)
                or guard.get('sample_count',0) < 1):
            return None
        reason = violation
    return dict(reason=reason,guard_path=str(guard_path),guard_sha256=digest(guard_path))


def custom_pairs(name, baseline, candidate, base_args, candidate_args, pairs, tasks):
    jobs = []
    for pair in range(pairs):
        for variant in (('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')):
            root, flags = ((baseline, base_args) if variant == 'baseline' else (candidate, candidate_args))
            jobs.append(dict(id=f'{variant}-p{pair:02d}', engine_root=str(root),
                             args=list(flags), tasks=str(tasks)))
    return dict(name=name, speculative=False, requires_prior_pass=[],
                tasks_kind='historical_364', notes='Isolated transport/control comparison; no SSD NAND claim.',
                plan=jobs)


def copy_kernel_sidecars(original,target):
    """Copy only hash-declared relative raw tensors into campaign ownership."""
    report = json.loads(Path(original).read_text())
    seen = {}
    for case in report.get('cases',[]):
        descriptors = [*case.get('numeric_evidence',{}).values()]
        if case.get('candidate_evidence') is not None:
            descriptors.append(case['candidate_evidence'])
        for descriptor in descriptors:
            relative = Path(descriptor['file'])
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('Kernel sidecar path escapes its report directory')
            source = (Path(original).parent/relative).resolve(strict=True)
            if not source.is_relative_to(Path(original).parent.resolve()) or source.is_symlink():
                raise ValueError('Kernel sidecar path escapes its report directory')
            if (type(descriptor.get('bytes')) is not int or not 0 < descriptor['bytes'] <= 16 << 20
                    or source.stat().st_size != descriptor['bytes'] or digest(source) != descriptor.get('sha256')):
                raise ValueError('Kernel sidecar differs from recorded identity')
            if str(relative) in seen:
                if seen[str(relative)] != descriptor:
                    raise ValueError('Kernel sidecar is described inconsistently')
                continue
            destination = Path(target).parent/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
            if digest(destination) != descriptor['sha256']:
                raise ValueError('Copied kernel sidecar identity differs')
            seen[str(relative)] = descriptor
    return len(seen)


def prepare(args):
    if type(args.pairs) is not int or args.pairs < 2:
        raise ValueError('At least two pairs are needed for alternating AB/BA order')
    if not re.fullmatch('[0-9a-f]{64}', args.expected_sha256):
        raise ValueError('Expected full checkpoint SHA-256 is required')
    profile = getattr(args,'profile','retained')
    if profile not in PROFILES:
        raise ValueError('Unknown campaign profile: '+str(profile))
    roots = {key: Path(getattr(args, key)).expanduser().resolve(strict=True)
             for key in ('h1', 'h2', 'candidate')}
    folder = args.output.expanduser().absolute()
    # Existing/symlink destinations are deliberately rejected before writing.
    if folder.exists() or folder.is_symlink():
        raise FileExistsError('A campaign already exists; use run to resume it')
    model = args.model.expanduser().resolve(strict=True)
    identity = model_identity(model)
    if identity['bytes'] <= 0:
        raise ValueError('Empty checkpoint')
    tasks = roots['candidate'] / 'research/hyperstream2/context.json'
    experiments = build_experiments(roots['h1'], roots['h2'], roots['candidate'], pairs=args.pairs,profile=profile)
    numerical = ['--numerical-evidence', '--expert-batch', '--concat-tiles', '--bulk-metadata',
                 '--string-buffer-bytes', '65536', '--prefill-chunk', '16']
    direct = [*numerical, '--packed-direct-io']
    imported = [*direct, '--zero-copy-weights']
    pipeline = [*imported, '--expert-pipeline']
    for name, a, b in (('legacy_vs_direct', numerical, direct),
                       ('direct_vs_zero_copy_import', direct, imported),
                       ('gate_up_pipeline', imported, pipeline)):
        experiments.append(custom_pairs(name, roots['candidate'], roots['candidate'],
                                         a, b, args.pairs, tasks))
    long_tasks = roots['candidate']/'research/m5_validation/general-long.json'
    for depth in (0, 2, 4, 8, 16):
        extra = [] if depth == 0 else ['--speculative-depth', str(depth)]
        experiment = custom_pairs('general_long_'+('serial_control' if depth == 0 else f'speculation{depth}'),
            roots['candidate'], roots['candidate'], pipeline, [*pipeline, *extra], args.pairs, long_tasks)
        experiment.update(speculative=depth != 0, tasks_kind='general_long_noncopy',
            notes='Long non-copy generation; report truncation and observed length. Under 64 emitted tokens is insufficient sustained-decode evidence.')
        experiments.append(experiment)
    # A rejected/slower optional component is an explicit exclusion; it cannot
    # prevent unrelated isolated transport/speculation trials from being run.
    combinations = [x for x in experiments if x.get('requires_prior_pass')]
    experiments = [x for x in experiments if not x.get('requires_prior_pass')] + combinations
    # Apply centrally after custom transport/long-decode families are appended,
    # so every current-candidate control and treatment declares the same router.
    apply_profile_jobs(experiments,roots['candidate'],profile)
    manifests = {str(root): engine_manifest(root) for root in roots.values()}
    for label, expected in (('h1', 'c8f81e57463ed3d7fd3d7dba226a65500439a9ce'),
                            ('h2', '5e8234181e736c0319a64df25e6813e92dd3e6da')):
        if manifests[str(roots[label])]['git_head'] != expected:
            raise ValueError(label.upper()+' must point to its exact retained PR revision')
    if subprocess.run(['git', '-C', str(roots['candidate']), 'merge-base', '--is-ancestor',
                       '3b137621d69f3549a85b7d0cc20460bd07b11fb9',
                       manifests[str(roots['candidate'])]['git_head']], capture_output=True).returncode:
        raise ValueError('Candidate must descend from the retained PR #12 revision')
    paths = {Path(__file__).resolve(), ROOT / 'benchmarks/verify_m5_validation.py',
             ROOT / 'benchmarks/m5_autonomous_supervisor.py',
             ROOT / 'benchmarks/m5_process_inventory.py',
             ROOT / 'benchmarks/m5_manifest_recovery.py',
             ROOT / 'benchmarks/diagnose_m5_memory.py',
             ROOT / 'benchmarks/hash_checkpoint_guarded.py',
             ROOT / 'benchmarks/benchmark_research51.py',
             ROOT / 'benchmarks/campaign_research51.py', ROOT / 'benchmarks/plan_hyperstream3.py',
             ROOT / 'benchmarks/verify_gguf_speculation.py',
             ROOT / 'benchmarks/hyperstream2_observer.py',
             ROOT / 'air_llm/airllm/nextgen/request_trace.py',
             ROOT / 'air_llm/airllm/nextgen/experiment_guard.py', tasks}
    kernel_tool = ROOT/'benchmarks/validate_m5_kernels.py'
    if kernel_tool.is_file():
        paths.add(kernel_tool)
    for path in (ROOT/'benchmarks/diagnose_m5_numerics.py',ROOT/'tests_next/reference/ggml_codec_reference.c'):
        if path.is_file():
            paths.add(path)
    task_snapshots = {}
    for experiment in experiments:
        for job in experiment['plan']:
            job.setdefault('tasks', str(tasks))
            job['args'] = validate_job_args(job['args'])
            paths.add(Path(job['tasks']))
            task_snapshots[job['tasks']] = load_tasks(job['tasks'])
    # Capture all engine Python sources as well as the worker's narrower manifest.
    for root in roots.values():
        paths.update((root / 'air_llm').rglob('*.py'))
    from verify_m5_validation import KERNEL_POLICY, FIDELITY_FAILURE_POLICY
    config = dict(schema_version=1, origin=str(folder), expected_checkpoint_sha256=args.expected_sha256,profile=profile,
                  kernel_qualification_policy=KERNEL_POLICY,
                  fidelity_failure_policy=FIDELITY_FAILURE_POLICY,
                  model_identity=identity, engine_manifests=manifests, engine_roots={k:str(v) for k,v in roots.items()},
                  tasks_by_path=task_snapshots,
                  protocol_sha256=protocol_manifest(sorted(paths)), pairs=args.pairs,
                  experiments=experiments, kernel_report=None if args.kernel_report is None
                  else str(args.kernel_report.expanduser().resolve(strict=True)),
                  numerical_tolerances=dict(atol=1e-4, rtol=1e-4),
                  scope=dict(physical_ssd_certified=False, os_page_cache='uncontrolled; never purged',
                      isolated_gpu_time_required=False,
                      historical_reference_only=dict(decode_tokens_per_second=1.584,
                          historical_364_ttft_seconds=110.7, first_forward_logical_bytes=135500000000)),
                  limits=dict(rss_bytes=2 << 30, reserve_bytes=3 << 30,
                      swap_growth_bytes=128 << 20, admitted_payload_bytes=768 << 20,
                      timeout_seconds=600., interval_seconds=.05, max_processes=32))
    if config['kernel_report']:
        config['kernel_report_sha256'] = digest(config['kernel_report'])
    folder.mkdir(parents=True, exist_ok=False)
    if config['kernel_report']:
        original = Path(config['kernel_report'])
        config['kernel_report_origin'] = str(original)
        target = folder/'provided-kernels'/'kernels.json'
        target.parent.mkdir()
        for src, dst in ((original, target),
                         (original.with_suffix('.guard.json'), target.with_suffix('.guard.json'))):
            shutil.copyfile(src, dst)
        config['kernel_raw_sidecar_count'] = copy_kernel_sidecars(original,target)
        diagnostic = original.with_name(original.stem+'-numeric-diagnosis.json')
        if diagnostic.is_file():
            copied_diagnostic = target.with_name('kernel-numeric-diagnosis.json')
            shutil.copyfile(diagnostic,copied_diagnostic)
            config.update(kernel_diagnosis=str(copied_diagnostic),kernel_diagnosis_sha256=digest(copied_diagnostic))
        config['kernel_report'] = str(target)
    with (folder / 'config.json').open('x') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    save(folder / 'state.json', dict(schema_version=1, config_sha256=digest(folder/'config.json'),
        status='prepared_no_worker_launched', sessions=[], jobs={}, experiment_results={}))
    write_manifest(folder)
    return dict(status='prepared_no_worker_launched', folder=str(folder),
                experiment_count=len(experiments), jobs=sum(len(x['plan']) for x in experiments))


def write_manifest(folder):
    """Hash raw evidence without recursion or relying on its claims."""
    from verify_m5_validation import finder_metadata_ignored, read_json, require_no_finder_evidence
    folder = Path(folder)
    try:
        (folder/'manifest.json').lstat()
    except FileNotFoundError:
        pass
    else:
        require_no_finder_evidence(read_json(folder/'manifest.json').get('files'))
    paths=[path for path in sorted(folder.rglob('*')) if not finder_metadata_ignored(path)]
    files = {str(p.relative_to(folder)): dict(sha256=digest(p), bytes=p.stat().st_size)
             for p in paths if p.is_file()
             and p.name not in ('manifest.json', '.runner.lock') and not p.is_symlink()}
    save(folder / 'manifest.json', dict(schema_version=1, files=files,
                                      authenticity_certified=False))


def unchanged(config):
    if not protocol_unchanged(config['protocol_sha256']):
        raise RuntimeError('Frozen code or task content changed; prepare a new campaign')
    for root, expected in config['engine_manifests'].items():
        if engine_manifest(Path(root)) != expected:
            raise RuntimeError('Frozen engine commit or code changed')
    if model_identity(config['model_identity']['path']) != config['model_identity']:
        raise RuntimeError('Checkpoint file identity changed')
    if config.get('kernel_report') and digest(config['kernel_report']) != config['kernel_report_sha256']:
        raise RuntimeError('Kernel validation report changed')
    if config.get('kernel_diagnosis') and digest(config['kernel_diagnosis']) != config['kernel_diagnosis_sha256']:
        raise RuntimeError('Kernel C-codec diagnosis changed')


def require_kernels(config, experiment, generated_report=None):
    if not any('--metal-batch-backend' in job['args'] for job in experiment['plan']):
        return None
    path = config.get('kernel_report') or generated_report
    if path is None:
        raise RuntimeError('This experiment requires a separately admitted kernel validation report')
    from verify_m5_validation import validate_kernel_report, kernel_eligibility
    data = validate_kernel_report(path, config)
    return kernel_eligibility(data, experiment)


def guarded_hash(config, folder):
    output = folder / 'checkpoint.json'
    code = run_child([sys.executable, str(ROOT/'benchmarks/hash_checkpoint_guarded.py'),
        '--model', config['model_identity']['path'], '--output', str(output)], folder)
    if code:
        guard_path = output.with_suffix('.guard.json')
        guard = json.loads(guard_path.read_text()) if guard_path.is_file() else {}
        if str(guard.get('exception','')).startswith('MemoryError:') or guard.get('violation') in (
                'system_reserve','system_swap_growth','macos_memory_pressure','owned_process_tree_rss'):
            raise MemoryError('Full checkpoint hashing refused by retained resource guard')
        raise RuntimeError('Full checkpoint hashing failed; see raw guard evidence')
    data = json.loads(output.read_text())
    if data.get('sha256') != config['expected_checkpoint_sha256'] or data.get('stable_file') is not True:
        raise RuntimeError('Checkpoint SHA-256 differs from its required identity')
    return str(output)


def _signal_interrupt(signum, frame):
    raise KeyboardInterrupt('Controller received signal ' + str(signum))


def run(args):
    folder = args.campaign.expanduser().resolve(strict=True)
    if not math.isfinite(args.resource_wait_seconds) or args.resource_wait_seconds < 0:
        raise ValueError('Wait must be finite and nonnegative')
    with campaign_lock(folder):
        config = json.loads((folder/'config.json').read_text())
        if str(folder) != config.get('origin'):
            raise ValueError('Copied evidence can be verified, but resume must use the original campaign folder')
        state = json.loads((folder/'state.json').read_text())
        if state['config_sha256'] != digest(folder/'config.json'):
            raise RuntimeError('Frozen campaign configuration changed')
        from verify_m5_validation import (verify_manifest, validate_job, summarize_experiment,
                                         validate_config, VerifiedFidelityRejection, read_json)
        validate_config(config)
        if args.experiment and not set(args.experiment) <= {x['name'] for x in config['experiments']}:
            raise ValueError('Unknown experiment selection; inspect config.json for frozen names')
        recovery=inspect_index(folder,config,state)
        unchanged(config)
        # A stale process lease can only originate in our exclusively-created
        # attempt directories; creation-time identity prevents PID reuse kills.
        saved_leases=[]
        for lease in folder.glob('sessions/*/**/owned-processes.json'):
            identities=read_json(lease)['processes']
            saved_leases.extend(identities)
            owned_cleanup(identities)
        require_leases_quiescent(saved_leases)
        # A crash can persist its atomic state before the final global index.
        # Reindex only after owned writers are stopped, frozen inputs and all
        # old immutable hashes are rechecked, and sealed claims are revalidated.
        recovery=inspect_index(folder,config,state)
        unchanged(config)
        if recovery['needed']:
            validate_recovery_records(folder,config,state)
            unchanged(config)
            if (digest(folder/'state.json') != recovery['state_sha256']
                    or digest(folder/'config.json') != recovery['config_sha256']
                    or digest(folder/'manifest.json') != recovery['manifest_sha256']):
                raise RuntimeError('Campaign changed during index recovery')
            write_manifest(folder)
            verify_manifest(folder)
        if state['status'] in ('completed', 'completed_with_exclusions', 'completed_evidence_with_rejections'):
            from verify_m5_validation import verify
            return verify(folder)
        # A completed job is resumable only inside a completed pre/post-hash
        # bracket. Work from an aborted integrity session remains raw evidence,
        # but is rerun rather than mixed into accepted pairs.
        bracketed = {(s['index'], b['index']) for s in state['sessions']
                     for b in s.get('integrity_blocks', []) if b.get('status') == 'sealed'}
        unbracketed = {key: value for key, value in state['jobs'].items()
                      if (value['session'], value.get('integrity_block')) not in bracketed}
        if unbracketed:
            state.setdefault('discarded_unbracketed_jobs', []).append(unbracketed)
            for key in unbracketed:
                del state['jobs'][key]
                state['experiment_results'].pop(key.split('/')[0], None)
        index = len(state['sessions'])
        session_dir = folder/'sessions'/f'{index:04d}'
        session_dir.mkdir(parents=True, exist_ok=False)
        session = dict(index=index, status='started', started_unix=time.time(), integrity_blocks=[])
        state['sessions'].append(session)
        state['status'] = 'running'
        save(folder/'state.json', state)
        old_term = signal.signal(signal.SIGTERM, _signal_interrupt)
        try:
            sys.path.insert(0, str(ROOT/'air_llm'))
            from airllm.nextgen.experiment_guard import ExperimentLimits, admission, system_snapshot
            limits = ExperimentLimits(**config['limits'])
            # Fail fast before *any* hashing or inference parent. No hidden small
            # allocation path is admitted under warning/critical pressure.
            session['before'] = system_snapshot()
            admission(ExperimentLimits(admitted_payload_bytes=64 << 20), session['before'])
            wait_before_job(limits, args.resource_wait_seconds, session)
            unchanged(config)
            if config.get('kernel_report') is None:
                tool = ROOT/'benchmarks/validate_m5_kernels.py'
                if str(tool) not in config['protocol_sha256']:
                    raise RuntimeError('Kernel tool was absent when preparing the campaign; prepare after sources are complete')
                kernel_dir = session_dir/'kernels'
                kernel_dir.mkdir()
                kernel_output = kernel_dir/'kernels.json'
                code = run_child([sys.executable, str(tool), '--output', str(kernel_output)], kernel_dir)
                if code == 75:
                    raise MemoryError('Kernel validation refused by retained resource guard')
                if code:
                    raise RuntimeError('Kernel validation refused or failed')
                session['kernel_report'] = str(kernel_output)
                save(folder/'state.json', state)
            session['hash_before'] = guarded_hash(config, session_dir/'hash-before')
            session['hash_before_completed_unix'] = time.time()
            last_hash = session['hash_before']
            last_hash_completed = session['hash_before_completed_unix']
            for experiment in config['experiments']:
                name = experiment['name']
                if args.experiment and name not in args.experiment:
                    continue
                exclusion = None
                for prerequisite in experiment.get('requires_prior_pass', []):
                    result = state['experiment_results'].get(prerequisite, {})
                    if (result.get('status') != 'accepted' or result.get('median_total_speedup', 0) <= 1
                            or result.get('median_worker_total_speedup', 0) <= 1):
                        exclusion = 'Combination prerequisite lacks independently verified benefit: ' + prerequisite
                        break
                if exclusion is None:
                    exclusion = require_kernels(config, experiment, session.get('kernel_report'))
                if exclusion:
                    state['experiment_results'][name] = dict(status='excluded', reason=exclusion,
                        kernel_report=config.get('kernel_report') or session.get('kernel_report'), session=index)
                    save(folder/'state.json', state)
                    continue
                new_keys = []
                for job in experiment['plan']:
                    key = name + '/' + job['id']
                    if key in state['jobs']:
                        validate_job(folder, config, job, state['jobs'][key])
                        continue
                    unchanged(config)
                    wait = {}
                    wait_before_job(limits, args.resource_wait_seconds, wait)
                    admission(limits)
                    attempt = session_dir/name/job['id']
                    attempt.mkdir(parents=True, exist_ok=False)
                    output = attempt/'worker.json'
                    engine = config['engine_manifests'][job['engine_root']]
                    command = [sys.executable, str(ROOT/'benchmarks/benchmark_research51.py'),
                        '--model', config['model_identity']['path'], '--tasks', job['tasks'],
                        '--engine-root', job['engine_root'], '--output', str(output), '--label', job['id'],
                        '--expected-protocol-json', json.dumps(config['protocol_sha256'], sort_keys=True),
                        '--expected-engine-head', engine['git_head'],
                        '--expected-engine-sources-json', json.dumps(engine['source_sha256'], sort_keys=True),
                        *job['args']]
                    record = dict(output=str(output.relative_to(folder)), output_absolute=str(output), command=command,
                                  wait=wait, session=index, started_unix=time.time())
                    session['active'] = dict(experiment=name, job=job['id'], **record)
                    save(folder/'state.json', state)
                    process_start = time.monotonic()
                    code = run_child(command, attempt)
                    record.update(exit_code=code, ended_unix=time.time(),
                                  controller_wall_seconds=time.monotonic()-process_start)
                    resource_refusal = worker_resource_refusal(output,command,config['limits'])
                    if resource_refusal is not None:
                        session['active']['resource_refusal'] = resource_refusal
                        raise MemoryError('Worker deferred by verified resource guard: '+resource_refusal['reason'])
                    if code:
                        raise RuntimeError('Worker refused or failed: ' + key)
                    validate_job(folder, config, job, record)
                    unchanged(config)
                    state['jobs'][key] = record
                    new_keys.append(key)
                    session.pop('active', None)
                    save(folder/'state.json', state)
                try:
                    result = summarize_experiment(folder, config, experiment, state['jobs'])
                except VerifiedFidelityRejection as error:
                    # Only a fully validated cross-recording fidelity mismatch
                    # is a completed negative experiment. All guards, malformed
                    # evidence, nonfinite outputs and input drift still abort.
                    result = error.result
                if new_keys:
                    block_index = len(session['integrity_blocks'])
                    after_started = time.time()
                    after_hash = guarded_hash(config, session_dir/f'integrity-{block_index:04d}')
                    unchanged(config)
                    block = dict(index=block_index, status='sealed', jobs=new_keys,
                        hash_before=last_hash, hash_before_completed_unix=last_hash_completed,
                        hash_after=after_hash, hash_after_started_unix=after_started,
                        hash_after_completed_unix=time.time())
                    session['integrity_blocks'].append(block)
                    for key in new_keys:
                        state['jobs'][key]['integrity_block'] = block_index
                    last_hash, last_hash_completed = after_hash, block['hash_after_completed_unix']
                state['experiment_results'][name] = result
                save(folder/'state.json', state)
            unchanged(config)
            session['hash_after_started_unix'] = time.time()
            session['hash_after'] = guarded_hash(config, session_dir/'hash-after')
            unchanged(config)
            session['status'] = 'completed'
            session['after'] = system_snapshot()
            if len(state['experiment_results']) != len(config['experiments']):
                state['status'] = 'partial_selected_experiments_completed'
            else:
                if any(x['status'] == 'rejected_fidelity' for x in state['experiment_results'].values()):
                    state['status'] = 'completed_evidence_with_rejections'
                else:
                    state['status'] = ('completed_with_exclusions' if any(
                        x['status'] == 'excluded' for x in state['experiment_results'].values()) else 'completed')
        except BaseException as error:
            session['status'] = state['status'] = 'deferred_or_interrupted'
            session['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            session['ended_unix'] = time.time()
            signal.signal(signal.SIGTERM, old_term)
            save(folder/'state.json', state)
            write_manifest(folder)
    return dict(status=state['status'], completed_experiments=len(state['experiment_results']))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare', allow_abbrev=False)
    for option in ('model', 'h1', 'h2', 'candidate', 'output'):
        q.add_argument('--'+option, type=Path, required=True)
    q.add_argument('--expected-sha256', required=True)
    q.add_argument('--pairs', type=int, default=3)
    q.add_argument('--profile',choices=PROFILES,default='retained')
    q.add_argument('--kernel-report', type=Path)
    q = sub.add_parser('run', allow_abbrev=False)
    q.add_argument('--campaign', type=Path, required=True)
    q.add_argument('--resource-wait-seconds', type=float, default=0.,
                   help='Bounded optional recovery observation; defaults to immediate refusal')
    q.add_argument('--experiment', action='append', help='Execute only named experiments; resume others later')
    q = sub.add_parser('verify', allow_abbrev=False)
    q.add_argument('--campaign', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.action == 'prepare':
            result = prepare(args)
        elif args.action == 'run':
            result = run(args)
        else:
            from verify_m5_validation import verify
            result = verify(args.campaign)
    except MemoryError as error:
        print(json.dumps(dict(status='deferred_resources', reason=str(error),
              campaign=str(getattr(args,'campaign',''))), ensure_ascii=False))
        raise SystemExit(75)
    except KeyboardInterrupt:
        print(json.dumps(dict(status='interrupted', cleanup='observed owned processes stopped')))
        raise SystemExit(130)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
