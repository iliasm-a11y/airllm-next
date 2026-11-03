"""Early rejection through the strict window reader and explicit real build pair.

Does not alter any measured files or existing comparator. A matching early
window never qualifies the complete arm, resource outcome, cache, EOS or speed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import psutil

WORK = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def compare(reference, candidate, tokens):
    early = module('original_early_window', WORK / 'compare_prefix_chunks.py')
    variant = module('bound_tail_variant', WORK / 'native-tail-even16-candidate/variant_compare.py')
    base = WORK / 'native-tail-even16-candidate'
    pair = variant.load_build_pair(
        WORK / 'native-prefill-final-controls.build.json',
        base / 'build-receipt.json', base / 'build-binding.json')
    roots = {reference.resolve(): 'reference', candidate.resolve(): 'candidate'}
    if len(roots) != 2:
        raise ValueError('Distinct exact reference and candidate roots required')
    native_key_observations = {}

    def unchanged_live_owner(guard, side):
        for name, expected in guard['preflight']['tracked_identities'].items():
            stat = Path(name).stat()
            actual = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            if actual != expected:
                raise ValueError('Measured input identity changed: ' + name)
        if side == 'candidate':
            if guard.get('status') != 'running' or guard.get('guard_violation') is not None:
                raise ValueError('Candidate is no longer an active clear-guard early window')
            owner = guard['owned_identity']
            process = psutil.Process(owner['pid'])
            if abs(process.create_time() - owner['create_time']) > .01 or process.exe() != owner['executable']:
                raise ValueError('Candidate owned process birth/executable changed')

    def bound_dependencies(root, window, unused_canonical_key):
        side = roots[Path(root).resolve()]
        guard = json.loads((root.parent / 'guard.json').read_text())
        report = json.loads((root / 'report.json').read_text())
        preflight = guard['preflight']
        manifest = preflight['snapshot_manifest']
        if window > len(manifest['prefix_tokens']):
            raise ValueError('Window exceeds the identified common prefix')
        unchanged_live_owner(guard, side)
        key, _ = variant.canonical_key(manifest)
        transmitted = json.loads((root.parent / 'run-prompts.json').read_text())
        if key != preflight.get('snapshot_key_sha256') or key != transmitted.get('snapshot_key_sha256'):
            raise ValueError('Recomputed preflight/transmitted prompt key differs')
        native_key = report.get('snapshot_key_sha256')
        if native_key is not None and native_key != key:
            raise ValueError('Observed native key differs')
        native_key_observations[side] = {'observed': native_key is not None,
                                         'value': native_key,
                                         'preflight_and_transmitted_key_verified': True}
        binding = pair['binding']
        sources = {**binding['shared_source_sha256'],
                   'helper_source': binding[side]['helper_source_sha256']}
        compound = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        binaries = {'helper': binding[side]['binary_sha256'], **binding['shared_native_libraries_sha256']}
        if (preflight.get('source_sha256') != sources or manifest['runtime_source_sha256'] != compound or
                manifest['runtime_binaries'] != binaries or preflight.get('runtime_binaries') != binaries):
            raise ValueError('Actual source/library/build identities differ')
        params = manifest['state_parameters']
        ub = report['microbatch']
        if (report.get('state_parameters') != params or report.get('logical_batch') != ub or
                params.get('n_batch') != ub or params.get('n_ubatch') != ub or
                params.get('n_threads') != 10 or params.get('n_threads_batch') != 10 or
                params.get('n_gpu_layers') != 0):
            raise ValueError('Actual keyed native parameters differ')
        if side == 'candidate':
            smoke = pair['builds'][side]['parameter_smoke'][str(ub)]
            if (smoke['returncode'] != 0 or smoke['state_parameters'] != params or
                    json.loads(smoke['stdout']) != params or
                    any(params.get(name) != value for name, value in variant.POLICY_PARAMETERS.items()) or
                    report.get('prefill_chunk_policy') != variant.POLICY):
                raise ValueError('Candidate actual binary smoke or policy differs')
        elif ub != 8 or pair['builds'][side]['parameter_smoke']['8'] != params:
            raise ValueError('Reference is not the measured CPU8 build')
        normalized = copy.deepcopy(manifest)
        for name in ('n_batch', 'n_ubatch'):
            del normalized['state_parameters'][name]
        if side == 'candidate':
            for name in variant.POLICY_PARAMETERS:
                del normalized['state_parameters'][name]
        original_sources = {**binding['shared_source_sha256'],
                            'helper_source': binding['reference']['helper_source_sha256']}
        normalized['runtime_source_sha256'] = hashlib.sha256(
            json.dumps(original_sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        normalized['runtime_binaries']['helper'] = binding['reference']['binary_sha256']
        return normalized

    # Only dependency normalization differs, after actual two-build verification.
    # The original raw SHA, dtype, stride, token coverage and numeric tests remain.
    early._load_dependencies = bound_dependencies
    args = argparse.Namespace(repo=variant.REPO, reference=reference,
                              candidate=candidate, tokens=tokens)
    result = early.compare(args)
    result['two_build_binding_validated'] = True
    result['native_key_observations'] = native_key_observations
    result['binding_sha256'] = variant.sha(base / 'build-binding.json')
    result['original_early_reader_sha256'] = variant.sha(WORK / 'compare_prefix_chunks.py')
    result['variant_comparator_sha256'] = variant.sha(base / 'variant_compare.py')
    result['early_variant_wrapper_sha256'] = variant.sha(__file__)
    variant.verify_delegated_modules()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.tokens)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}))
    raise SystemExit(0 if result['status'] == 'identical_window' else 2)
