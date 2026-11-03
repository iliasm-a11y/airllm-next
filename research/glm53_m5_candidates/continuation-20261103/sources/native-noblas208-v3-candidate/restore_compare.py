"""CPU8→fullV2 fidelity, then exact fullV2→restoreV2 under actual token fences.

No existing comparator or receipt is modified. All raw guards, captures,
continuations, states and replay requirements retain their CPU bitexact rules.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding_module():
    prepared = json.loads((ROOT / 'prepare-manifest.json').read_text())
    path = ROOT / 'variant_compare.py'
    if sha(path) != prepared['variant_comparator_sha256'] or sha(ROOT / 'geometry.py') != prepared['geometry_sha256']:
        raise ValueError('frozen v2 comparator/geometry source changed')
    spec = importlib.util.spec_from_file_location('fenced208_v2_build_binding', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_delegated_modules()
    return module


def compare(cpu_reference, reference_root, candidate_root, cpu_guard, reference_guard, candidate_guard,
            reference_build, candidate_build, binding):
    result = {'schema_version': 1, 'passed': False, 'errors': [], 'full_model_qualified': False,
              'functional_agents_qualified': False, 'performance_qualified': False,
              'snapshot_exact_evidence_passed': False, 'logits': [], 'sequence_states': [],
              'scope': 'bounded v2 explicit-fence CPU diagnostics; no EOS or private-history qualification'}
    try:
        v = binding_module()
        # Require the actual fullV2 arm to retain original CPU8 fidelity. No
        # standalone passed:true JSON can substitute for this raw comparison.
        fidelity = v.compare(cpu_reference, reference_root, cpu_guard, reference_guard,
                             reference_build, candidate_build, binding)
        result['cpu8_fullv2_fidelity'] = fidelity
        if fidelity.get('passed') is not True:
            raise ValueError('fullV2 has not passed complete raw original CPU8 fidelity')
        v.verify_delegated_modules()
        import numpy as np
        import sys
        sys.path.insert(0, str(ROOT))
        import geometry
        from research.glm53_m5.compare_fidelity import CRITERIA, compare_arrays, family, tensor_view
        from research.glm53_batched_prefill.native_compare import (
            _read, _compare_continuation, _check_replay, _compare_writes, EXPECTED_NAMES, FAMILIES)
        v.verify_delegated_modules()
        _read = v.checked_delegate(_read)
        _compare_continuation = v.checked_delegate(_compare_continuation)
        _check_replay = v.checked_delegate(_check_replay)
        _compare_writes = v.checked_delegate(_compare_writes)
        compare_arrays = v.checked_delegate(compare_arrays)
        tensor_view = v.checked_delegate(tensor_view)
        family = v.checked_delegate(family)
        reference_root, candidate_root = Path(reference_root), Path(candidate_root)
        a, at, ab, ag, aw, ac = _read(reference_root, reference_guard)
        b, bt, bb, bg, bw, bc = _read(candidate_root, candidate_guard)
        pair = v.load_build_pair(reference_build, candidate_build, binding)
        v.arm_dependencies(a, ag, pair['builds']['candidate'], pair, 'candidate')
        v.arm_dependencies(b, bg, pair['builds']['candidate'], pair, 'candidate')
        for name in ('microbatch', 'logical_batch', 'threads', 'gpu_layers', 'physical_slots', 'vocab_size', 'snapshot_key_sha256'):
            if name not in a or a[name] != b.get(name):
                raise ValueError('same actual v2 profile/key required: ' + name)
        result['actual_reference_geometry'] = v.checked_delegate(geometry.check)(a, ab)
        result['actual_candidate_geometry'] = v.checked_delegate(geometry.check)(b, bb)
        result['direct_restore_boundary'] = v.checked_delegate(geometry.direct_restore)(reference_root, a, b, ab, bb)
        result['continuation'] = _compare_continuation(reference_root, candidate_root, a, b, ac, bc)
        result['continuation_replays'] = {
            'reference': _compare_continuation(reference_root, reference_root, a, a, ac, ac, ((0, 2),)),
            'candidate': _compare_continuation(candidate_root, candidate_root, b, b, bc, bc, ((0, 2),))}
        result['same_agent_replays'] = {
            'reference': _check_replay(reference_root, a, at, ab, aw),
            'candidate': _check_replay(candidate_root, b, bt, bb, bw)}
        select_batches = lambda rows: [(r['request'], r['offset'], r['tokens']) for r in rows if r['offset'] >= 208]
        if select_batches(ab) != select_batches(bb):
            raise ValueError('actual fullV2/restoreV2 suffix physical boundaries differ')
        result['observed_state_writes'] = _compare_writes(aw, bw,
            [(r, offset) for r, offset, _ in select_batches(ab)], True)
        for request, (ar, br) in enumerate(zip(a['requests'], b['requests'])):
            for name in ('prompt_tokens', 'prompt_ids_sha256', 'next_position'):
                if ar[name] != br[name]:
                    raise ValueError('complete native prompt IDs or positions differ')
            result['logits'].append({'request': request, **compare_arrays(
                np.memmap(reference_root / f'logits-{request}.f32', dtype='<f4', mode='r'),
                np.memmap(candidate_root / f'logits-{request}.f32', dtype='<f4', mode='r'), CRITERIA['cpu'])})
            result['sequence_states'].append({'request': request,
                'passed': ar['sequence_bytes'] == br['sequence_bytes'] and ar['sequence_sha256'] == br['sequence_sha256']})
        select = lambda rows: {k: r for k, r in rows.items() if k[1] >= 208}
        at, bt = select(at), select(bt)
        if not at or set(at) != set(bt):
            raise ValueError('complete suffix capture keys differ')
        checks, coverage, per_batch, per_batch_names = [], {k: 0 for k in FAMILIES}, {}, {}
        for key, ar in at.items():
            label = family(ar['name'])
            check = compare_arrays(tensor_view(reference_root, ar), tensor_view(candidate_root, bt[key]),
                                   CRITERIA['cpu'], integer=label == 'routes')
            checks.append(check['passed'])
            coverage[label] += check['elements']
            per_batch.setdefault(key[:2], set()).add(label)
            per_batch_names.setdefault(key[:2], set()).add(key[2])
            if not check['passed'] and 'first_tensor_divergence' not in result:
                result['first_tensor_divergence'] = {'request': key[0], 'offset': key[1], 'name': key[2], **check}
        expected_batches = {(r, offset) for r, offset, _ in select_batches(ab)}
        if set(per_batch) != expected_batches or any(v != set(FAMILIES) for v in per_batch.values()):
            raise ValueError('routes/router/KDA suffix coverage missing')
        if any(not EXPECTED_NAMES <= names for names in per_batch_names.values()):
            raise ValueError('one or more original GLM layer captures missing')
        result['coverage'] = coverage
        result['criteria'] = {**CRITERIA['cpu'], 'ordered_routes_exact': True, 'tokens_exact': True, 'all_values_finite': True}
        result['snapshot_exact_evidence_passed'] = bool(result['direct_restore_boundary']['passed'] and
            result['continuation']['passed'] and all(r['passed'] for r in result['continuation_replays'].values()) and
            all(r['passed'] for r in result['same_agent_replays'].values()) and result['observed_state_writes']['passed'] and
            all(checks) and all(r['passed'] for r in result['logits'] + result['sequence_states']))
        result['passed'] = result['snapshot_exact_evidence_passed']
        result['v2_binary_sha256'] = pair['binding']['candidate']['binary_sha256']
        result['binding_sha256'] = sha(binding)
        result['delegated_module_sha256'] = v.verify_delegated_modules()
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        result['passed'] = False
        result['snapshot_exact_evidence_passed'] = False
        result['errors'].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-reference', 'reference', 'candidate', 'cpu-guard', 'reference-guard', 'candidate-guard',
                 'reference-build', 'candidate-build', 'binding', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.cpu_reference, args.reference, args.candidate, args.cpu_guard,
        args.reference_guard, args.candidate_guard, args.reference_build, args.candidate_build, args.binding)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
