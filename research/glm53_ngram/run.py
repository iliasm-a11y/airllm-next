"""Single owned model, original guards, and a mandatory fidelity gate for timing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, system_snapshot
from airllm.nextgen.glm_native import FILES, _identity, inspect_native_checkpoint
from research.glm53_batched_prefill.uid_supervisor import acquire_uid_lock
from research.glm53_batched_prefill.native_run import identify, stop_owned
from research.glm53_gpu_v3.power_state import PowerGate, ScopedAwake
from research.glm53_ultra_decode.build import sha
from research.glm53_ultra_decode.campaign_limits import remaining_swap_budget
from research.glm53_ultra_decode.model_profile import (
    GIB, REQUIRED_MEMORY, REQUIRED_DISK, resource_blockers,
    write_worker_report, retain_worker_report,
)


def fingerprints(probe, runtime):
    return {'probe': sha(probe), 'libraries': {p.name: sha(p) for p in runtime.glob('*.dylib') if not p.is_symlink()}}


def qualification(path, config, identity):
    """Only the completed audit of this executable/runtime admits timed drafts."""
    if path is None:
        raise ValueError('timed n-gram requires a completed fidelity receipt')
    outer = json.loads(path.read_text())
    native = json.loads((path.parent/'capture/report.json').read_text())
    wanted = {(d, label) for d in (config['draft_tokens'],)
              for label in ('all-accepted', 'reject-first', 'reject-middle')
              if label != 'reject-middle' or d > 1}
    observed = {(c.get('draft_tokens'), c.get('label', '').split('/')[-1]) for c in native.get('cases', [])}
    if (outer.get('status') != 'completed' or outer.get('artifacts') != identity or
            not outer.get('weight_identities_unchanged') or not outer.get('native_artifacts_unchanged') or
            not native.get('complete') or not native.get('qualification_passed') or
            native.get('config', {}).get('mode') != 'audit' or
            native['config']['draft_tokens'] != config['draft_tokens'] or
            native['config'].get('prompt') != config['prompt'] or not wanted <= observed or
            not native.get('cases') or not all(c.get('passed') for c in native['cases'])):
        raise ValueError('fidelity is incomplete, failed, or belongs to other native artifacts')
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def worker(args):
    holder, lock = acquire_uid_lock()
    child = identity = None
    report = {'uid_lock': lock, 'native_process_launched': False, 'finalized': False, 'status': 'admitting'}
    write_worker_report(args.output, report)
    gate = PowerGate(expected_source='ac')
    try:
        blockers = resource_blockers(system_snapshot(), shutil.disk_usage(args.output).free, gate.sample(),
                                     args.campaign_swap_used_bytes)
        inspection = inspect_native_checkpoint(args.model, args.receipt)
        if not inspection['integrity_verified'] or not inspection['architecture_matches']:
            blockers.append('worker_checkpoint_integrity')
        if blockers:
            raise RuntimeError('worker admission refused: '+','.join(blockers))
        command = [str(args.probe.resolve()), str(args.model.resolve()/FILES[0][0]),
                   str(args.output.resolve()/'config.json'), str(args.output.resolve()/'capture')]
        env = {k: v for k, v in os.environ.items() if not k.startswith(('DYLD_', 'AIRLLM_', 'GGML_', 'LLAMA_ARG_'))}
        env.update(AIRLLM_MMAP_DEMAND='1', AIRLLM_EXPERT_PREFETCH='1', AIRLLM_DENSE_PREFETCH='1',
                   AIRLLM_DENSE_PREFETCH_MAX_COLS='1', AIRLLM_GPU_CACHE_MIB='0')
        with (args.output/'native.stdout.log').open('wb') as out, (args.output/'native.stderr.log').open('wb') as err:
            report.update(status='spawning', native_process_launched=None)
            write_worker_report(args.output, report)
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err, pass_fds=(holder.fileno(),))
            report.update(status='identifying', native_process_launched=True, native_spawned_pid=child.pid)
            write_worker_report(args.output, report)
            for _ in range(100):
                if child.poll() is not None:
                    break
                try:
                    identity = identify(child, args.probe)
                    break
                except Exception:
                    time.sleep(.01)
            if identity is None:
                raise RuntimeError('native identity could not be established')
            report.update(status='running', native_identity=identity)
            write_worker_report(args.output, report)
            with ScopedAwake(child) as awake:
                while child.poll() is None:
                    time.sleep(.5)
                    if child.poll() is not None:
                        break
                    identify(child, args.probe, identity)
                    awake.check()
                    remaining_swap_budget(system_snapshot(), args.campaign_swap_used_bytes)
                    if not gate.sample()['passed']:
                        raise RuntimeError('power/lid/clock/awake continuity failed')
                    if shutil.disk_usage(args.output).free < REQUIRED_DISK:
                        raise RuntimeError('disk reserve fell below32GiB')
            report['native_returncode'] = child.wait(timeout=5)
            report['status'] = 'completed' if child.returncode == 0 and gate.sample(final=True)['passed'] else 'failed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
    finally:
        if child is not None:
            if child.poll() is None and identity is not None:
                report['cleanup'] = stop_owned(child, args.probe, identity)
            elif child.poll() is not None:
                report['native_returncode'] = child.wait(timeout=5)
        report.update(power=gate.summary(), uid_parent_fd_closed=True, surviving_child_retains_uid_fd=True, finalized=True)
        holder.close()
        write_worker_report(args.output, report)
    return 0 if report['status'] == 'completed' else 2


def run(args):
    if args.worker:
        return worker(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(args.model.resolve()) or output.is_relative_to(args.runtime.resolve()):
        raise ValueError('output must be outside model/runtime directories')
    config = {'mode': args.mode, 'tokens': args.tokens, 'ngram_size': args.ngram_size,
              'draft_tokens': args.draft_tokens, 'prompt': args.prompt}
    identity = fingerprints(args.probe, args.runtime)
    qualified = qualification(args.qualification, config, identity) if args.mode == 'ngram' else None
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    sample = system_snapshot()
    if args.campaign_swap_used_bytes is None:
        args.campaign_swap_used_bytes = sample['swap_used_bytes']
    reasons = resource_blockers(sample, shutil.disk_usage(output).free, PowerGate(expected_source='ac').sample(),
                                args.campaign_swap_used_bytes)
    inspection = inspect_native_checkpoint(args.model, args.receipt)
    if not inspection['integrity_verified'] or not inspection['architecture_matches']:
        reasons.append('checkpoint_integrity_or_architecture_unverified')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'config': config,
              'status': 'admission_refused' if reasons else 'admitted', 'blockers': reasons,
              'native_process_launched': False, 'performance_measured': False,
              'resource_sample': sample, 'disk_free_bytes': shutil.disk_usage(output).free,
              'artifacts': identity, 'qualification': qualified, 'checkpoint': inspection,
              'policy': {'required_available_memory_bytes': REQUIRED_MEMORY, 'disk_reserve_bytes': REQUIRED_DISK,
                         'max_rss_bytes': int(8.5*GIB), 'max_system_swap_growth_bytes': 32<<20,
                         'campaign_swap_used_bytes': args.campaign_swap_used_bytes}, 'production_changed': False}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if reasons:
        print(json.dumps({'status': report['status'], 'blockers': reasons, 'native_process_launched': False}))
        return 2
    command = [sys.executable, '-m', 'research.glm53_ngram.run', '--worker']
    for name in ('model', 'receipt', 'runtime', 'probe', 'output'):
        command.extend(('--'+name, str(getattr(args, name).resolve())))
    command.extend(('--campaign-swap-used-bytes', str(args.campaign_swap_used_bytes)))
    identities = {args.model.resolve()/row['name']: row['identity'] for row in inspection['files']}
    try:
        if not all(_identity(p) == stamp for p, stamp in identities.items()):
            raise RuntimeError('weight identity changed before spawn')
        report['native_process_launched'] = None
        guard = supervise(command, output/'supervision', limits=ExperimentLimits(
            rss_bytes=int(8.5*GIB), reserve_bytes=3*GIB,
            swap_growth_bytes=remaining_swap_budget(sample, args.campaign_swap_used_bytes),
            admitted_payload_bytes=int(2.5*GIB), timeout_seconds=2400, interval_seconds=.5, max_processes=8))
        report['guard'] = guard
        retain_worker_report(report, output)
        report['status'] = 'completed' if (report.get('worker', {}).get('status') == 'completed' and
            (output/'capture/complete.json').is_file()) else 'failed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
    finally:
        retain_worker_report(report, output)
        guard_file = output/'supervision.guard.json'
        if guard_file.is_file():
            report['guard'] = json.loads(guard_file.read_text())
        report['weight_identities_unchanged'] = all(_identity(p) == stamp for p, stamp in identities.items())
        report['native_artifacts_unchanged'] = identity == fingerprints(args.probe, args.runtime)
        if not report['weight_identities_unchanged'] or not report['native_artifacts_unchanged']:
            report['status'] = 'failed'
        report['performance_measured'] = report['status'] == 'completed' and args.mode != 'audit'
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'native_process_launched', 'performance_measured')}))
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'receipt', 'runtime', 'probe', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=('audit', 'reference', 'ngram'), default='audit')
    p.add_argument('--tokens', type=int, choices=(16, 32, 64, 128), default=32)
    p.add_argument('--ngram-size', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--draft-tokens', type=int, choices=(2, 4), default=2)
    p.add_argument('--prompt', default='La capitale de la France est')
    p.add_argument('--qualification', type=Path)
    p.add_argument('--campaign-swap-used-bytes', type=int)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    raise SystemExit(run(p.parse_args()))
