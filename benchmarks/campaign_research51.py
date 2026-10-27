#!/usr/bin/env python3
"""Execute a frozen sequential plan, requiring complete independently guarded runs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from benchmark_research51 import (ROOT, engine_manifest, load_tasks, protocol_manifest,
                                 protocol_unchanged, save)


def validate_job_args(arguments):
    """Only comparison controls belong in a plan's raw argument list."""
    if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
        raise ValueError('Job args must be a list of strings')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument('--rows', type=int, default=8192)
    parser.add_argument('--prefill-chunk', type=int, default=16)
    parser.add_argument('--hyper-prefill-max-tokens', type=int, default=128)
    for flag in ('hyper-prefill', 'numerical-evidence', 'request-trace'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--cache-policy', choices=['adaptive', 'lru'])
    parser.add_argument('--io-mode', choices=['pread', 'pread_nocache'])
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--string-buffer-bytes', type=int, default=0)
    for flag in ('expert-batch', 'concat-tiles', 'packed-copy-free', 'packed-direct-io',
                 'bulk-metadata', 'zero-copy-weights', 'expert-pipeline'):
        parser.add_argument('--' + flag, action='store_true')
    try:
        parsed, unknown = parser.parse_known_args(arguments)
    except (argparse.ArgumentError, SystemExit, TypeError) as error:
        raise ValueError('Invalid comparison arguments') from error
    if unknown:
        raise ValueError('Reserved, abbreviated or unknown job arguments: ' + repr(unknown))
    if parsed.rows < 1 or not math.isfinite(parsed.timeout) or parsed.timeout <= 0:
        raise ValueError('Rows and timeout must be positive and finite')
    if parsed.prefill_chunk < 1 or parsed.hyper_prefill_max_tokens < 1:
        raise ValueError('Prefill sizes must be positive')
    if not 0 <= parsed.string_buffer_bytes <= 65536:
        raise ValueError('String read buffer must be in 0..65536 bytes')
    return list(arguments)


def prepare_campaign(args):
    """Resolve caller-relative paths and capture every engine and protocol input."""
    for field in ('model', 'tasks', 'plan', 'output'):
        setattr(args, field, Path(getattr(args, field)).expanduser().resolve())
    if not args.model.is_file():
        raise ValueError('Model must be an existing file')
    raw_plan = json.loads(args.plan.read_text())
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError('Campaign plan must be a nonempty list')
    # Validate the default too: an empty default must never make an empty run
    # appear complete through any([]), even when some jobs override its path.
    load_tasks(args.tasks)
    plan, tasks_by_job, ids = [], {}, set()
    roots = {ROOT.resolve()}
    protocol_paths = {Path(__file__).resolve(), ROOT / 'benchmarks/benchmark_research51.py',
                      ROOT / 'benchmarks/hyperstream2_observer.py',
                      ROOT / 'air_llm/airllm/nextgen/request_trace.py',
                      args.plan, args.tasks}
    for job in raw_plan:
        if not isinstance(job, dict) or set(job) - {'id', 'args', 'tasks', 'engine_root'}:
            raise ValueError('Unsupported campaign job fields')
        ident = job.get('id')
        if (not isinstance(ident, str) or not ident.strip() or ident in ids
                or ident in ('.', '..') or Path(ident).name != ident):
            raise ValueError('Unique plain filename job IDs required')
        arguments = validate_job_args(job.get('args', []))
        if '--packed-copy-free' in arguments and '--packed-direct-io' in arguments:
            raise ValueError('Choose one packed host transport per job')
        task_path = Path(job.get('tasks', args.tasks)).expanduser().resolve()
        engine_root = Path(job.get('engine_root', ROOT)).expanduser().resolve()
        tasks_by_job[ident] = load_tasks(task_path)
        protocol_paths.add(task_path)
        roots.add(engine_root)
        ids.add(ident)
        plan.append(dict(id=ident, args=arguments, tasks=str(task_path), engine_root=str(engine_root)))
    sources = {str(root): engine_manifest(root) for root in sorted(roots)}
    protocol = protocol_manifest(sorted(protocol_paths))
    return plan, sources, protocol, tasks_by_job


def validate_run_evidence(evidence, guard, expected_sources, expected_protocol, expected_tasks,
                          *, expected_head=None, expected_command=None, expected_limits=None):
    """Require exact work, unchanged inputs and a successful independent guard."""
    if not isinstance(evidence, dict) or not isinstance(guard, dict):
        raise ValueError('Worker and guard evidence must both exist')
    if (evidence.get('status') != 'completed' or evidence.get('source_unchanged') is not True
            or evidence.get('protocol_unchanged') is not True
            or evidence.get('model_stat_unchanged') is not True
            or evidence.get('source_sha256') != expected_sources
            or evidence.get('protocol_sha256') != expected_protocol):
        raise RuntimeError('Incomplete or changed worker evidence')
    if expected_head is not None and (evidence.get('git_head') != expected_head
            or evidence.get('engine_head_unchanged') is not True
            or evidence.get('git_head_after') != expected_head):
        raise RuntimeError('Engine revision changed or differs from the plan')
    if (type(guard.get('exit_code')) is not int or guard['exit_code'] != 0
            or 'violation' not in guard or guard['violation'] is not None
            or guard.get('exception') or guard.get('cleanup_error')
            or type(guard.get('sample_count')) is not int or guard['sample_count'] < 1
            or not isinstance(guard.get('limits'), dict) or not guard['limits']):
        raise RuntimeError('Missing, unsuccessful or deferred independent guard')
    if expected_command is not None and guard.get('command') != [*expected_command, '--worker']:
        raise RuntimeError('Guard did not supervise the requested worker command')
    if expected_limits is not None and guard['limits'] != expected_limits:
        raise RuntimeError('Guard limits differ from the recorded campaign limits')
    turns = evidence.get('turns')
    if (not expected_tasks or not isinstance(turns, list) or len(turns) != len(expected_tasks)):
        raise RuntimeError('No complete one-to-one task output')
    for task, turn in zip(expected_tasks, turns):
        if not isinstance(turn, dict) or any(turn.get(key) != task[key]
                for key in ('id', 'prompt', 'max_new_tokens')):
            raise RuntimeError('Output task differs from the frozen request')
        tokens, stops = turn.get('generated_ids'), turn.get('eos_token_ids')
        if (turn.get('status') != 'completed' or turn.get('stop_reason') != 'eos'
                or not isinstance(tokens, list) or not tokens or len(tokens) > task['max_new_tokens']
                or any(type(token) is not int or token < 0 for token in tokens)
                or not isinstance(stops, list) or not stops
                or any(type(token) is not int or token < 0 for token in stops)
                or tokens[-1] not in stops):
            raise RuntimeError('Output did not complete with its recorded EOS')


def capture_engine_states(sources):
    states = {}
    for root in sources:
        try:
            states[root] = engine_manifest(Path(root))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            states[root] = {'error': f'{type(error).__name__}: {error}'}
    return states


def wait_before_job(limits, max_wait_seconds, run):
    """Optionally wait for sampled recovery before launching any worker.

    The worker retains its own fresh admission and all runtime protections.
    Zero preserves the historical fail-fast path. Waiting never retries a
    refused inference: no worker exists until this window has been observed.
    """
    if (isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0):
        raise ValueError('Resource wait seconds must be finite and nonnegative')
    from dataclasses import asdict
    headroom = 256 << 20
    run.update(launch_wait_seconds=0., launch_wait_samples=[], launch_wait_sample_count=0,
               launch_wait_samples_truncated=False,
               launch_wait_status='disabled' if max_wait_seconds == 0 else 'waiting',
               launch_wait_policy=dict(max_wait_seconds=float(max_wait_seconds),
                   headroom_bytes=headroom, observed_stable_seconds=5., poll_seconds=1.,
                   required_pressure_level=1, admission_limits=asdict(limits)))
    if max_wait_seconds == 0:
        return
    sys.path.insert(0, str(ROOT / 'air_llm'))
    from airllm.nextgen.experiment_guard import admission, system_snapshot
    start = monotonic()
    stable_since = None
    try:
        while True:
            snapshot = system_snapshot()
            sampled_at = monotonic()
            elapsed = sampled_at - start
            reason = None
            try:
                admission(limits, snapshot)
                if snapshot['pressure_level'] != 1:
                    raise MemoryError('Normal pressure level 1 is required for the recovery window')
                if snapshot['available_bytes'] < limits.reserve_bytes + limits.admitted_payload_bytes + headroom:
                    raise MemoryError('Recovery window needs 256 MiB headroom above admission')
            except MemoryError as error:
                reason = str(error)
            if reason is None:
                if stable_since is None:
                    stable_since = sampled_at
            else:
                stable_since = None
            stable_seconds = 0. if stable_since is None else sampled_at - stable_since
            sample = dict(snapshot, elapsed_seconds=elapsed, ready=reason is None,
                          observed_stable_seconds=stable_seconds, reason=reason)
            run['launch_wait_sample_count'] += 1
            run['launch_wait_samples'].append(sample)
            if len(run['launch_wait_samples']) > 60:
                run['launch_wait_samples'].pop(0)
                run['launch_wait_samples_truncated'] = True
            run['launch_wait_seconds'] = elapsed
            if stable_seconds >= 5. and elapsed <= max_wait_seconds:
                run['launch_wait_status'] = 'ready'
                return
            if elapsed >= max_wait_seconds:
                run.update(status='stopped', launch_wait_status='timed_out')
                raise MemoryError('No stable resource window before the optional wait expired')
            sleep(min(1., max_wait_seconds - elapsed))
    except BaseException:
        if run['launch_wait_status'] == 'waiting':
            run.update(status='stopped', launch_wait_status='failed')
        raise
    finally:
        run['launch_wait_seconds'] = max(0., monotonic() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resource-wait-seconds', type=float, default=0.,
                   help='Optional recovery window before each job; zero keeps fail-fast admission')
    args = p.parse_args()
    if not math.isfinite(args.resource_wait_seconds) or args.resource_wait_seconds < 0:
        p.error('--resource-wait-seconds must be finite and nonnegative')
    plan, sources, protocol, tasks_by_job = prepare_campaign(args)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', plan=plan, engine_manifests=sources,
                  protocol_sha256=protocol, tasks_by_job=tasks_by_job, runs=[])
    manifest = args.output / 'campaign.json'
    save(manifest, report)
    try:
        for job in plan:
            if not protocol_unchanged(protocol):
                raise RuntimeError('Research protocol changed between runs')
            if capture_engine_states(sources) != sources:
                raise RuntimeError('Engine revision or source changed between runs')
            output = args.output / (job['id'] + '.json')
            engine = sources[job['engine_root']]
            command = [sys.executable, str(ROOT / 'benchmarks/benchmark_research51.py'),
                       '--model', str(args.model), '--tasks', job['tasks'],
                       '--engine-root', job['engine_root'], '--output', str(output), '--label', job['id'],
                       '--expected-protocol-json', json.dumps(protocol, sort_keys=True),
                       '--expected-engine-head', engine['git_head'],
                       '--expected-engine-sources-json', json.dumps(engine['source_sha256'], sort_keys=True),
                       *job['args']]
            # The same limits are checked again by the independent worker
            # supervisor after launch; the recovery window adds only headroom.
            sys.path.insert(0, str(ROOT / 'air_llm'))
            from dataclasses import asdict
            from airllm.nextgen.experiment_guard import ExperimentLimits
            job_limits = ExperimentLimits(timeout_seconds=_job_timeout(job['args']))
            run = dict(id=job['id'], command=command,
                       status='waiting_for_resources' if args.resource_wait_seconds else 'started')
            report['runs'].append(run)
            save(manifest, report)
            wait_before_job(job_limits, args.resource_wait_seconds, run)
            if args.resource_wait_seconds and (not protocol_unchanged(protocol)
                    or capture_engine_states(sources) != sources):
                run['status'] = 'stopped'
                raise RuntimeError('Engine or protocol changed while waiting for resources')
            run['status'] = 'started'
            save(manifest, report)
            result = subprocess.run(command, cwd=ROOT)
            run['exit_code'] = result.returncode
            if result.returncode:
                raise RuntimeError('Stopped after unsuccessful or deferred worker: ' + job['id'])
            try:
                evidence = json.loads(output.read_text())
                guard = json.loads(output.with_suffix('.guard.json').read_text())
            except (OSError, ValueError) as error:
                raise RuntimeError('Worker or guard evidence missing: ' + job['id']) from error
            # The parent supervisor belongs to ROOT even for a historical engine.
            limits = asdict(job_limits)
            validate_run_evidence(evidence, guard, engine['source_sha256'], protocol, tasks_by_job[job['id']],
                                  expected_head=engine['git_head'], expected_command=command, expected_limits=limits)
            if not protocol_unchanged(protocol) or capture_engine_states(sources) != sources:
                raise RuntimeError('Engine or protocol changed during a run')
            run['status'] = 'completed'
            run['guard_file'] = str(output.with_suffix('.guard.json'))
            save(manifest, report)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['protocol_unchanged'] = protocol_unchanged(protocol)
        report['engine_manifests_after'] = capture_engine_states(sources)
        report['engines_unchanged'] = report['engine_manifests_after'] == sources
        if report['status'] == 'completed' and not (
                report['protocol_unchanged'] and report['engines_unchanged']):
            report.update(status='stopped', error='Protocol or engine changed before campaign finalization')
        save(manifest, report)

    if report['status'] != 'completed':
        raise RuntimeError(report.get('error', 'Campaign did not complete'))


def _job_timeout(arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--timeout', type=float, default=600.)
    return parser.parse_known_args(arguments)[0].timeout


if __name__ == '__main__':
    main()
