"""Fresh-server v3 ABBA campaigns with separately qualified CPU and candidate.

Captured fidelity is16..64 actual steps; timed jobs do not extend that full-logit/
state qualification automatically. Natural EOS may shorten a requested capture.
Production remains unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from research.glm53_gpu_v2.campaign import cancel_owned, interrupt_campaign, owned_process, save, summarize
from .dense_lock_plan import load_plan as load_dense_lock_plan, check_plan as check_dense_lock_plan
from .dense_lock_plan import lock_environment_matches, evidence_matches
from .completion_contract import validate_completion
from .power_state import power_preflight, power_evidence_matches

REPO = Path(__file__).resolve().parents[2]
CASES = ('cpu-stream', 'head-stream', 'tail-stream')
SOURCE_PATHS = ('research/glm53_gpu_v3/campaign.py', 'research/glm53_gpu_v3/benchmark_native.py',
                'research/glm53_gpu_v3/system_memory.py', 'research/glm53_gpu_v3/process_io.py',
                'research/glm53_gpu_v3/dense_lock_plan.py',
                'research/glm53_gpu_v3/completion_contract.py',
                'research/glm53_gpu_v3/power_state.py',
                'research/glm53_gpu_v2/campaign.py', 'research/glm53_gpu_v2/benchmark_lifecycle.py',
                'research/glm53_gpu_v2/storage_counters.py')


def _stat(path):
    data = Path(path).stat()
    return {'device': data.st_dev, 'inode': data.st_ino, 'size': data.st_size, 'mtime_ns': data.st_mtime_ns}


def _sha(path):
    # Avoid allocating an entire native dylib while another stage is quiescent.
    digest = hashlib.sha256()
    scratch = bytearray(min(max(1, Path(path).stat().st_size), 1024**2))
    view = memoryview(scratch)
    with Path(path).open('rb', buffering=0) as source:
        while count := source.readinto(scratch):
            digest.update(view[:count])
    return digest.hexdigest()


def _fingerprint(path):
    path = Path(path).resolve()
    before = _stat(path)
    digest = _sha(path)
    if before != _stat(path):
        raise ValueError('fingerprinted source/runtime changed during hashing')
    return {'path': str(path), 'stat': before, 'sha256': digest}


def _libraries(binary):
    rows = {path.name: _fingerprint(path) for path in Path(binary).resolve().parent.glob('*.dylib')
            if not path.is_symlink()}
    if not rows:
        raise ValueError('native runtime directory has no fingerprinted non-symlink dylibs')
    return rows


def _enabled(env, key):
    return env.get(key) not in (None, '', '0')


def fidelity_token_prefix(report):
    """Bind every actual captured natural greedy ID (16..64), not requested length."""
    capture = Path(report['command'][2]).resolve()
    tokens_path, complete_path = capture / 'tokens.jsonl', capture / 'complete.json'
    if tokens_path.stat().st_size > 2 * 1024**2 or complete_path.stat().st_size > 65536:
        raise ValueError('fidelity token/complete metadata exceeds parsing budget')
    marker = json.loads(complete_path.read_text())
    rows = [json.loads(line) for line in tokens_path.read_text().splitlines() if line.strip()]
    if (marker.get('completed') is not True or marker.get('captured_steps') != len(rows) or
            not 16 <= len(rows) <= 64 or report.get('full_logits', {}).get('compared') != len(rows)):
        raise ValueError('FID prefix requires complete actual16..64 token/logit rows, including natural EOS')
    prefix = []
    for step, row in enumerate(rows):
        token = row.get('token')
        if (type(row.get('step')) is not int or row['step'] != step or type(token) is not int or token < 0 or
                row.get('next_input_token', token) != token):
            raise ValueError('FID prefix is invalid or not a natural greedy trajectory')
        prefix.append(token)
    return {'token_ids': prefix, 'scope_steps': len(rows),
            'tokens_file': str(tokens_path), 'tokens_file_sha256': _sha(tokens_path),
            'complete_file': str(complete_path), 'complete_file_sha256': _sha(complete_path),
            'scope': 'natural generated-ID prefix only; no new full-logit/state proof beyond captured scope'}


def validate_timed_report(job, report, plan):
    expected_libraries = {name: row['sha256'] for name, row in plan['native_library_fingerprints'].items()}
    report = report or {}
    binary_matches = report.get('binary_sha256') == plan['binary_fingerprint']['sha256']
    libraries_match = report.get('native_libraries_sha256') == expected_libraries
    requests = report.get('requests', [])
    request = requests[0] if isinstance(requests, list) and len(requests) == 1 and isinstance(requests[0], dict) else {}
    completion = validate_completion(request, plan['tokens_requested'])
    server_normal = type(report.get('exit_code')) is int and report['exit_code'] == 0
    actual = request.get('tokens', [])
    expected = plan['fidelity'][job['arm']]['token_prefix']['token_ids']
    scope_steps = plan['fidelity'][job['arm']]['token_prefix']['scope_steps']
    checked = min(scope_steps, len(actual)) if isinstance(actual, list) else 0
    valid_ids = isinstance(actual, list) and bool(actual) and all(type(token) is int and token >= 0 for token in actual)
    prefix_matches = valid_ids and actual[:checked] == expected[:checked]
    gpu = request.get('gpu_command_times') or {}
    request_gpu = isinstance(gpu, dict) and type(gpu.get('valid_commands')) is int and gpu['valid_commands'] > 0
    gpu_matches = not job['gpu_commands_required'] or request_gpu
    dense_lock = plan.get('dense_lock_plan') if job['arm'] == 'B' else None
    dense_lock_matches = lock_environment_matches(report.get('research_environment', {}), dense_lock)
    if dense_lock:
        dense_lock_matches = dense_lock_matches and evidence_matches(report.get('dense_lock'), report.get('dense_lock_plan'), dense_lock)
        postrun = report.get('dense_lock_postrun', {})
        dense_lock_matches = dense_lock_matches and postrun.get('passed') is True and postrun.get('sha256') == dense_lock['sha256']
    elif report.get('dense_lock_plan') is not None or report.get('dense_lock') is not None:
        dense_lock_matches = False
    power_matches = power_evidence_matches(report, (plan.get('power_admission') or {}).get('snapshot', {}).get('power_source'))
    return {'passed': bool(binary_matches and libraries_match and prefix_matches and gpu_matches and dense_lock_matches and completion['passed'] and server_normal and power_matches),
            'binary_sha256_matches_plan': binary_matches, 'native_library_hashes_match_plan': libraries_match,
            'fidelity_prefix_matches': bool(prefix_matches), 'checked_prefix_steps': checked,
            'fidelity_prefix_scope_steps': scope_steps, 'generated_ids_logged': len(actual) if isinstance(actual, list) else None,
            'dense_lock_requirement_met': bool(dense_lock_matches),
            'completion_contract': completion, 'native_server_normal_exit': server_normal,
            'power_continuity_requirement_met': power_matches,
            'request_gpu_commands_observed': request_gpu, 'request_gpu_requirement_met': gpu_matches,
            'scope': 'all actually captured FID generated IDs plus timed runtime identity; later IDs are pair-compared only, not new full-logit fidelity'}


def _validate_report(path, model, libraries, *, prompt, require_states):
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError('fidelity experiment manifest exceeds parsing budget')
    report = json.loads(path.read_text())
    if (report.get('fidelity_passed') is not True or report.get('guard_status') != 'completed' or
            report.get('model_file_stats_unchanged') is not True):
        raise ValueError('fresh completed passed fidelity with unchanged model stats is required')
    current = {name: row['sha256'] for name, row in libraries.items()}
    if not report.get('native_libraries_sha256') or report['native_libraries_sha256'] != current:
        raise ValueError('fidelity runtime dylib hashes do not match the current server libraries')
    logits = report.get('full_logits', {})
    if type(logits.get('compared')) is not int or not 16 <= logits['compared'] <= 64:
        raise ValueError('each arm requires actual16..64 captured full-logit steps')
    if not {'routes', 'router', 'kda'}.issubset(report.get('coverage', {})):
        raise ValueError('fidelity must include ordered routes, router and KDA coverage')
    for label, metrics in [('logits', logits), *report['coverage'].items()]:
        if (metrics.get('passed') is not True or metrics.get('finite') is not True or
                metrics.get('bit_mismatch_count') != 0 or type(metrics.get('elements')) is not int or
                metrics['elements'] <= 0):
            raise ValueError('captured fidelity is not finite and bit-identical: ' + label)
    command = report.get('command', [])
    if (len(command) < 9 or Path(command[1]).resolve() != Path(model).resolve() or
            command[3:5] != ['10', '0'] or command[6] != prompt):
        raise ValueError('fidelity must match this model, CPU10/GPU0 placement count and exact prompt')
    expected_files = report.get('model_files')
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError('fidelity has no model file identity records')
    current_files = {path.name: _stat(path) for path in Path(model).parent.glob('*.gguf')}
    if set(current_files) != {row['file'] for row in expected_files}:
        raise ValueError('model file set differs from fidelity')
    for row in expected_files:
        current_stat = current_files[row['file']]
        if current_stat['size'] != row['size'] or current_stat['mtime_ns'] != row['mtime_ns']:
            raise ValueError('model stats differ from fidelity')
    states = report.get('observed_state_writes', {})
    if require_states and (states.get('passed') is not True or
            type(states.get('compared_events')) is not int or states['compared_events'] <= 0):
        raise ValueError('candidate requires nonempty bit-identical observed state-write hashes')
    env = report.get('environment', {})
    if env.get('AIRLLM_DIAGNOSTIC_REQUESTS') not in (None, '1'):
        raise ValueError('two-request reset captures cannot qualify one continuous campaign trajectory')
    if env.get('AIRLLM_GPU_CACHE_MIB', '0') != '0':
        raise ValueError('campaign requires no expert GPU cache')
    for key in ('AIRLLM_GLM_LAYER_LOCAL_HC', 'AIRLLM_GLM_CACHE_SPLIT', 'AIRLLM_EXPERT_DROP_AFTER_USE'):
        if _enabled(env, key):
            raise ValueError('unmatched extra native experiment: ' + key)
    if env.get('AIRLLM_GLM_HC_POST') not in (None, '1'):
        raise ValueError('modified HC post path is outside this campaign')
    evidence = {'report': str(path.resolve()), 'report_sha256': _sha(path), 'captured_steps': logits['compared'],
                'full_logits': logits, 'coverage': report['coverage'], 'observed_state_writes': states,
                'native_libraries_sha256': report['native_libraries_sha256'],
                'token_prefix': fidelity_token_prefix(report),
                'scope': 'captured trajectory and observed writes only; no uncaptured-state or longer-generation qualification'}
    return report, evidence, current_files


def validate_fidelities(args, libraries):
    reference, a_evidence, files = _validate_report(args.reference_fidelity, args.model, libraries,
                                                  prompt=args.prompt, require_states=False)
    candidate, b_evidence, b_files = _validate_report(args.candidate_fidelity, args.model, libraries,
                                                    prompt=args.prompt, require_states=True)
    if files != b_files:
        raise ValueError('CPU and candidate fidelity do not identify the same model files')
    a_env, b_env = reference.get('environment', {}), candidate.get('environment', {})
    dense_lock = load_dense_lock_plan(args.dense_lock_plan, args.model) if getattr(args, 'dense_lock_plan', None) else None
    if not lock_environment_matches(a_env, None) or reference.get('dense_lock_plan') is not None:
        raise ValueError('original CPU reference must have no dense lock experiment')
    if not lock_environment_matches(b_env, dense_lock):
        raise ValueError('candidate FID dense lock environment differs from exact selector plan')
    if dense_lock:
        prelaunch, postrun = candidate.get('dense_lock_prelaunch', {}), candidate.get('dense_lock_postrun', {})
        if (not evidence_matches(candidate.get('dense_lock'), candidate.get('dense_lock_plan'), dense_lock) or
                candidate.get('dense_lock_accepted') is not True or
                prelaunch.get('passed') is not True or postrun.get('passed') is not True or
                prelaunch.get('sha256') != dense_lock['sha256'] or postrun.get('sha256') != dense_lock['sha256']):
            raise ValueError('candidate FID did not prove exact retained dense lock plan')
        b_evidence['dense_lock_plan'] = dense_lock
        b_evidence['dense_lock'] = candidate['dense_lock']
    elif candidate.get('dense_lock_plan') is not None or candidate.get('dense_lock') is not None:
        raise ValueError('candidate FID has an unmatched dense lock experiment')
    if reference['command'][-1] != 'cpu':
        raise ValueError('reference fidelity must qualify the original CPU policy')
    for key in ('AIRLLM_MMAP_NOCACHE', 'AIRLLM_DENSE_LOOKAHEAD', 'AIRLLM_METAL_Q8_CPU_ARITH',
                'AIRLLM_METAL_Q8_SHARED', 'AIRLLM_METAL_CPU_MAPPED_WEIGHTS', 'AIRLLM_GLM_Q8_ONLY'):
        if _enabled(a_env, key):
            raise ValueError('CPU reference has a new experimental flag enabled: ' + key)
    if (b_env.get('AIRLLM_MMAP_NOCACHE') != '1' or b_env.get('AIRLLM_DENSE_LOOKAHEAD') != '1' or
            b_env.get('AIRLLM_DENSE_LOOKAHEAD_MIB') != '32'):
        raise ValueError('candidate must match nocache plus32MiB dense lookahead')
    flags = ['--mmap-nocache', '--dense-lookahead', '--lookahead-mib', '32']
    if dense_lock:
        flags += ['--dense-lock-plan', dense_lock['path']]
    policy = candidate['command'][-1]
    if args.candidate == 'cpu-stream':
        if policy != 'cpu' or any(_enabled(b_env, key) for key in (
            'AIRLLM_METAL_Q8_CPU_ARITH', 'AIRLLM_METAL_Q8_SHARED', 'AIRLLM_METAL_CPU_MAPPED_WEIGHTS', 'AIRLLM_GLM_Q8_ONLY')):
            raise ValueError('CPU-stream fidelity contains a GPU placement or arithmetic mode')
    elif args.candidate == 'head-stream':
        if (policy != 'head-only' or b_env.get('AIRLLM_METAL_Q8_CPU_ARITH') != '1' or
                any(_enabled(b_env, key) for key in ('AIRLLM_METAL_Q8_SHARED', 'AIRLLM_METAL_CPU_MAPPED_WEIGHTS', 'AIRLLM_GLM_Q8_ONLY'))):
            raise ValueError('head-stream fidelity must match the inline CPU-compatible head')
        flags += ['--head-only', '--q8-cpu-arith', '--metal-timing']
    elif args.candidate == 'tail-stream':
        from research.glm53_gpu_v2.placement import plan
        placement = plan(args.tail_layers)
        if (policy != 'quant-tail' or b_env.get('AIRLLM_METAL_Q8_CPU_ARITH') != '2' or
                b_env.get('AIRLLM_GLM_Q8_ONLY') != '1' or b_env.get('AIRLLM_METAL_CPU_MAPPED_WEIGHTS') != '1' or
                _enabled(b_env, 'AIRLLM_METAL_Q8_SHARED') or b_env.get('AIRLLM_Q8_TAIL_PATTERN') != placement['tensor_pattern']):
            raise ValueError('tail-stream fidelity does not match exact mapped Q8 placement')
        flags += ['--quant-tail', str(args.tail_layers), '--mapped-cpu-weights', '--metal-timing']
    else:
        raise ValueError('unknown v3 candidate')
    return flags, {'A': a_evidence, 'B': b_evidence}, files


def make_plan(args):
    if args.output.exists():
        raise ValueError('refusing to reuse a campaign output directory')
    if args.tokens not in (8, 16, 32, 64) or type(args.cycles) is not int or not 1 <= args.cycles <= 3:
        raise ValueError('tokens must be8/16/32/64 and cycles1..3')
    if args.candidate == 'tail-stream':
        if type(args.tail_layers) is not int or not 1 <= args.tail_layers <= 45:
            raise ValueError('tail-stream requires tail-layers1..45')
    elif args.tail_layers is not None:
        raise ValueError('tail-layers is only valid for tail-stream')
    admission = power_preflight()
    if not admission['passed']:
        raise ValueError('Campaign awake/power preflight declined: ' + '; '.join(admission['reasons']))
    libraries = _libraries(args.binary)
    flags, fidelity, models = validate_fidelities(args, libraries)
    sources = {name: _fingerprint(REPO / name) for name in SOURCE_PATHS}
    binary = _fingerprint(args.binary)
    common = [sys.executable, '-m', 'research.glm53_gpu_v3.benchmark_native',
              '--binary', str(args.binary.resolve()), '--model', str(args.model.resolve()),
              '--threads', '10', '--context', '512', '--batch', '8', '--gpu-layers', '0',
              '--tokens', str(args.tokens), '--repeats', '1', '--prompt', args.prompt,
              '--max-rss-gib', '8.5', '--max-swap-mib', '32', '--timeout', '900',
              '--mmap-demand', '--expert-prefetch', '--dense-prefetch',
              '--expected-power-source', admission['snapshot']['power_source']]
    jobs = []
    for cycle in range(args.cycles):
        for arm in 'ABBA':
            index = len(jobs)
            output = args.output.resolve() / 'runs' / f'{index:02d}-{arm}.json'
            jobs.append({'index': index, 'cycle': cycle, 'arm': arm,
                'candidate': 'original-cpu10' if arm == 'A' else args.candidate,
                'report': str(output), 'runner_log': str(output.with_suffix('.runner.log')),
                'gpu_commands_required': arm == 'B' and args.candidate in ('head-stream', 'tail-stream'),
                'command': common + ['--output', str(output)] + (flags if arm == 'B' else [])})
    return {'schema': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'planned',
            'sequence': 'ABBA' * args.cycles, 'jobs': jobs, 'tokens_requested': args.tokens,
            'power_admission': admission,
            'fidelity': fidelity, 'native_library_fingerprints': libraries, 'binary_fingerprint': binary,
            'dense_lock_plan': fidelity['B'].get('dense_lock_plan'),
            'source_fingerprints': sources, 'model_file_stats': models, 'production_changed': False,
            'limits': ['fresh owned server per stage; macOS file cache and other applications remain uncontrolled',
                       'full-logit/observed-state scope is the actual recorded16..64 trajectory, not requested generation length',
                       'no automatic optimization qualification, confidence interval or universal maximum claim']}


def check_before_launch(args, plan):
    awake = power_preflight(plan['power_admission']['snapshot']['power_source'])
    if not awake['passed']:
        raise ValueError('Campaign lid/power admission changed: ' + '; '.join(awake['reasons']))
    rehashed = []
    for group in (plan['native_library_fingerprints'], plan['source_fingerprints'], {'server': plan['binary_fingerprint']}):
        for key, row in group.items():
            if _stat(row['path']) != row['stat']:
                rehashed.append(key)
                if _sha(row['path']) != row['sha256']:
                    raise ValueError('runtime or measured runner changed after the fidelity gate: ' + key)
    current_names = {p.name for p in args.binary.resolve().parent.glob('*.dylib') if not p.is_symlink()}
    if current_names != set(plan['native_library_fingerprints']):
        raise ValueError('native library set changed after planning')
    if {p.name: _stat(p) for p in args.model.parent.glob('*.gguf')} != plan['model_file_stats']:
        raise ValueError('model identity/stats changed after planning')
    for evidence in plan['fidelity'].values():
        if _sha(evidence['report']) != evidence['report_sha256']:
            raise ValueError('fidelity manifest changed after planning')
        prefix = evidence['token_prefix']
        for kind in ('tokens', 'complete'):
            if _sha(prefix[kind + '_file']) != prefix[kind + '_file_sha256']:
                raise ValueError('fidelity capture prefix source changed after planning')
    if plan.get('dense_lock_plan'):
        check_dense_lock_plan(plan['dense_lock_plan'], args.model)
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
            'metadata_changed_hashes_rechecked': rehashed, 'power_admission': awake,
            'scope': 'file identities/stats each stage; unchanged metadata reuses prior streamed source/runtime hashes; no full weight hash'}


def classify_stage(job, code, report):
    gpu_observed = bool(report and report.get('gpu_command_times', {}).get('valid_commands', 0) > 0)
    passed = bool(code == 0 and report and report.get('status') == 'complete' and
                  type(report.get('exit_code')) is int and report['exit_code'] == 0 and
                  report.get('guard_violation') is None and report.get('model_file_stats_unchanged') is True and
                  (not job['gpu_commands_required'] or gpu_observed))
    return {'status': 'complete' if passed else 'failed', 'gpu_commands_required': job['gpu_commands_required'],
            'gpu_commands_observed': gpu_observed}


def run_campaign(args, plan):
    owned_process(os.getpid())  # Verify the launching Python's dependency before any server launch.
    args.output.mkdir(parents=True)
    (args.output / 'runs').mkdir()
    save(args.output / 'plan.json', plan)
    stages = []
    campaign = {'plan': plan, 'stages': stages, 'status': 'running'}
    save(args.output / 'campaign.json', campaign)
    child = identity = None
    try:
        for job in plan['jobs']:
            preflight = check_before_launch(args, plan)
            stage = {'index': job['index'], 'arm': job['arm'], 'status': 'running', 'prelaunch_identity': preflight,
                     'started_at_utc': datetime.now(timezone.utc).isoformat()}
            stages.append(stage)
            save(args.output / 'campaign.json', campaign)
            print(json.dumps({'stage': job['index'], 'arm': job['arm'], 'status': 'starting'}), flush=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(('AIRLLM_', 'GGML_', 'LLAMA_ARG_', 'DYLD_'))}
            env.pop('LLAMA_API_KEY', None)
            with Path(job['runner_log']).open('x') as output:
                child = subprocess.Popen(job['command'], cwd=REPO, env=env, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                identity = {'pid': child.pid, 'create_time': owned_process(child.pid).create_time()}
                stage['owned_benchmark'] = identity
                save(args.output / 'campaign.json', campaign)
                code = child.wait()
            stage['exit_code'] = code
            path = Path(job['report'])
            raw = json.loads(path.read_text()) if path.exists() else None
            stage['report_data'] = raw
            stage.update(classify_stage(job, code, raw))
            stage['timed_provenance'] = validate_timed_report(job, raw, plan)
            try:
                stage['postrun_identity'] = check_before_launch(args, plan)
            except (OSError, ValueError) as exc:
                stage['postrun_identity'] = {'passed': False, 'error': str(exc)}
            if not stage['timed_provenance']['passed'] or not stage['postrun_identity']['passed']:
                stage['status'] = 'failed'
            stage['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            child = identity = None
            save(args.output / 'campaign.json', campaign)
            print(json.dumps({'stage': job['index'], 'arm': job['arm'], 'status': stage['status'], 'report': job['report']}), flush=True)
            if stage['status'] != 'complete':
                raise RuntimeError('guard, FID prefix/runtime identity, GPU instrumentation or stage execution failed; remaining stages not launched')
        campaign['status'] = 'complete'
    except KeyboardInterrupt:
        campaign['status'] = 'cancelled'
        if child is not None:
            campaign['cancellation'] = cancel_owned(child, identity)
            if campaign['cancellation']['status'] not in ('already_exited', 'interrupted_and_exited'):
                campaign['status'] = 'cancelled_needs_attention'
        if stages and stages[-1]['status'] == 'running':
            stages[-1]['status'] = 'cancelled'
    except Exception as exc:
        campaign.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if child is not None:
            campaign['cancellation'] = cancel_owned(child, identity)
    finally:
        campaign['summary'] = summarize(plan, stages)
        campaign['summary']['fidelity_scope'] = {arm: evidence['captured_steps'] for arm, evidence in plan['fidelity'].items()}
        save(args.output / 'campaign.json', campaign)
    return 0 if campaign['status'] == 'complete' and campaign['summary']['comparison_valid_on_stated_scope'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-fidelity', type=Path, required=True)
    parser.add_argument('--candidate-fidelity', type=Path, required=True)
    parser.add_argument('--candidate', choices=CASES, required=True)
    parser.add_argument('--dense-lock-plan', type=Path, help='Optional candidate-only bounded original dense mmap retention plan')
    parser.add_argument('--tail-layers', type=int)
    parser.add_argument('--tokens', type=int, choices=(8, 16, 32, 64), default=16)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--prompt', default='La capitale de la France est')
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        with Path('/tmp/airllm-glm53-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = signal.signal(signal.SIGTERM, interrupt_campaign)
            try:
                return run_campaign(args, plan)
            finally:
                signal.signal(signal.SIGTERM, previous)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
