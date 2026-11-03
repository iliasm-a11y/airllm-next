"""Strict isolated CPU/GPU two-build comparison, or same-profile GPU restore.

Requires completed diagnostic captures, real bound builds and raw GPU traces.
No model, compiler or native backend is started by this command.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import provenance as p


def gpu_receipt(root, guard_path, report, receipt, binding, profile, modules):
    """Recheck real raw native GPU trace instead of accepting actual_gpu=true."""
    arm = binding['candidate']
    command = receipt.get('command')
    expected = [str(Path(arm['binary']['path']).resolve()), None,
                str((Path(guard_path).resolve().parent / 'run-prompts.json').resolve()),
                str(Path(root).resolve()), str(report['microbatch']), report['mode']]
    if not isinstance(command, list) or len(command) != 6 or any(command[i] != expected[i] for i in (0, 2, 3, 4, 5)):
        raise ValueError('GPU receipt command is not the exact bounded owned helper/capture')
    prompts = p.read(expected[2])
    if prompts.get('snapshot_key_sha256') != report.get('snapshot_key_sha256'):
        raise ValueError('actual GPU prompt input key differs from own report/key')
    uid = receipt.get('uid_model_lock', {})
    if (uid.get('acquired_before_spawn') is not True or uid.get('pass_fds_requested') is not True or
            uid.get('fd_observation', {}).get('observed_matching_lock_fd') is not True or
            uid.get('owned_child_pid') != receipt.get('owned_pid') or
            uid.get('parent_fd_closed') is not True or uid.get('child_poll_before_parent_fd_close') is None):
        raise ValueError('prelaunch owned UID lock and actual inherited FD evidence required')
    cleanup = receipt.get('cleanup', {})
    if cleanup.get('owned_server_stopped') is not True or type(cleanup.get('returncode')) is not int or cleanup.get('returncode') != 0:
        raise ValueError('owned GPU helper cleanup/completion evidence required')
    if receipt.get('gpu_execution_profile') != profile or receipt.get('actual_gpu') is not True:
        raise ValueError('GPU receipt profile or real-execution flag differs')
    trace_path = Path(guard_path).resolve().parent / 'native-stderr.log'
    if not trace_path.is_file() or trace_path.stat().st_size > 32 << 20:
        raise ValueError('bounded raw native GPU stderr trace required')
    modules['gpu'].validate_capture_report(report, receipt['preflight'], profile,
        binding['candidate_libraries_directory'], receipt['baseline']['swap_used_bytes'], report['microbatch'])
    trace = modules['gpu'].parse_trace(trace_path.read_text(errors='replace'), profile,
        baseline_swap=receipt['baseline']['swap_used_bytes'])
    if receipt.get('gpu_trace') != trace or receipt.get('actual_backend') != trace['actual_backend']:
        raise ValueError('receipt GPU summary differs from revalidated raw terminal commands')
    return {'passed': True, 'raw_trace_sha256': p.sha(trace_path), **trace}


def compare(reference, candidate, reference_guard, candidate_guard, binding_path, *, kind='cpu-gpu'):
    result = {'schema_version': 1, 'passed': False, 'errors': [], 'kind': kind,
              'full_model_qualified': False, 'performance_qualified': False, 'snapshot_exact_qualified': False,
              'scope': 'complete bounded native diagnostics only; no EOS, functional agents, performance or unobserved-state claim'}
    try:
        if kind not in ('cpu-gpu', 'gpu-restore'):
            raise ValueError('unknown strict two-build comparison kind')
        prepared = p.read(HERE / 'prepare.json')
        modules = p.load_modules(prepared)
        binding, builds, profile = p.load_pair(binding_path, prepared, modules)
        roots = [Path(reference).resolve(), Path(candidate).resolve()]
        data = [modules['native']._read(roots[0], reference_guard), modules['native']._read(roots[1], candidate_guard)]
        if kind == 'cpu-gpu':
            a = p.dependencies(data[0][0], data[0][3], binding['reference'], builds['reference'], modules)
            b = p.dependencies(data[1][0], data[1][3], binding['candidate'], builds['candidate'], modules, gpu_profile=profile)
            if a != b:
                raise ValueError('model/tokenizer/template/state/environment dependencies differ beyond the bound GPU builds, tail policy and batches')
            p.check_chunks(data[0][0], data[0][2])
            p.check_chunks(data[1][0], data[1][2], gpu=True)
            result['gpu_raw_trace'] = gpu_receipt(roots[1], candidate_guard, data[1][0], data[1][3], binding, profile, modules)
            p.verify_sources(prepared)
            spec = importlib.util.spec_from_file_location('bound_gpu_comparison_numerics', HERE / 'numerics.py')
            numerics = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(numerics)
            p.verify_sources(prepared)
            result.update(numerics.cpu_gpu(*roots, *data, modules))
        else:
            # The two GPU arms use the same actual GPU helper, source, runtime
            # and profile. A CPU key is never assigned or substituted here.
            keys = []
            normalized = []
            for index in (0, 1):
                report, _, _, receipt, _, _ = data[index]
                normalized.append(p.dependencies(report, receipt, binding['candidate'], builds['candidate'], modules, gpu_profile=profile))
                keys.append(report['snapshot_key_sha256'])
                result['gpu_raw_trace_' + str(index)] = gpu_receipt(roots[index], (reference_guard, candidate_guard)[index],
                    report, receipt, binding, profile, modules)
            if keys[0] != keys[1] or normalized[0] != normalized[1]:
                raise ValueError('GPU snapshot arms do not share their own exact complete dependency key/profile')
            pair = data[0][0]['mode'], data[1][0]['mode']
            if pair not in (('full', 'restore'), ('split', 'restore')):
                raise ValueError('same-profile GPU snapshot requires full/restore or split/restore')
            # Original exact states/writes/logits/ordered routes/continuations,
            # direct208 snapshot equality, replay and full coverage are retained.
            result['exact_gpu_restore'] = modules['native'].compare(*roots, reference_guard, candidate_guard)
            result['snapshot_exact_evidence_passed'] = result['exact_gpu_restore']['passed']
            result['passed'] = result['snapshot_exact_evidence_passed']
        result['two_build_binding_sha256'] = p.sha(binding_path)
        result['two_own_keys_recomputed'] = True
        result['builds_verified'] = True
        p.verify_sources(prepared)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        result['passed'] = False
        result['errors'].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference-guard', type=Path, required=True)
    parser.add_argument('--candidate-guard', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--kind', choices=('cpu-gpu', 'gpu-restore'), default='cpu-gpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.reference, args.candidate, args.reference_guard, args.candidate_guard, args.binding, kind=args.kind)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
