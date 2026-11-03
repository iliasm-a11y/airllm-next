"""Unexecuted source candidate: complete CPU8→fullV4→restoreV4, profile32 only.

No existing comparator or receipt is modified. All raw guards, captures,
continuations, states and replay requirements retain their CPU bitexact rules.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TARGET_MICROBATCH = 32
PREFIX_BYTES = 156739588
PREFIX_SHA256 = '4c4d6f24b2cf3c4f30a7894b8609035960a3ec5687521c942aa6a578e2e5a3cd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_sources():
    prepared = json.loads((ROOT / 'prepare-manifest.json').read_text())
    for name, expected in prepared['frozen_candidate_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('frozen v4 dependency source changed: ' + name)
    if sha(__file__) != prepared['restore_comparator_sha256']:
        raise ValueError('frozen v4 restore comparator source changed')


def checkpoint_records(root, report, batches):
    """Exact checkpoints prescribed by actual v4 decode calls, including restore.

    Restore request0 includes the one common-prefix decode (128,208); requests
    1 and2 start from restored208 and therefore have only later checkpoints.
    No nonexistent repeated-prefix checkpoint is invented for those requests.
    """
    if report.get('mode') not in ('full', 'restore'):
        raise ValueError('complete full or restore arm required')
    path = Path(root) / 'state_checkpoints.jsonl'
    if not path.is_file() or not 0 < path.stat().st_size <= 1 << 20:
        raise ValueError('bounded complete-state checkpoint metadata required')
    expected = {(row['request'], row['offset'] + row['tokens']) for row in batches
                if (row['offset'] + row['tokens']) % 128 == 0 or row['offset'] + row['tokens'] == 208 or
                row['offset'] + row['tokens'] == report['requests'][row['request']]['prompt_tokens']}
    result = {}
    for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
        request, end = row.get('request'), row.get('token_end')
        if (type(request) is not int or request not in (0, 1, 2) or type(end) is not int or end < 1 or
                type(row.get('next_position')) is not int or row['next_position'] != end or
                row.get('status') != 'hashed' or type(row.get('state_seq_flags')) is not int or row['state_seq_flags'] != 0 or
                type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 256 << 20 or
                type(row.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise ValueError('invalid complete-state checkpoint descriptor')
        key = request, end
        if key in result:
            raise ValueError('duplicate complete-state checkpoint')
        result[key] = row
    if set(result) != expected:
        raise ValueError('exact native checkpoint coverage missing or extra')
    for request, row in enumerate(report['requests']):
        final = result.get((request, row['prompt_tokens']))
        if final is None or (final['bytes'], final['sha256']) != (row['sequence_bytes'], row['sequence_sha256']):
            raise ValueError('final checkpoint differs from verified raw sequence state')
    return result


def compare_checkpoints(a, b):
    if not set(b) <= set(a):
        raise ValueError('restore checkpoints have no full reference counterpart')
    rows = [{'request': key[0], 'token_end': key[1],
             'passed': (a[key]['bytes'], a[key]['sha256']) == (b[key]['bytes'], b[key]['sha256'])}
            for key in sorted(b)]
    # Identical-input replay only over checkpoints emitted in both restore0/2.
    first = {end: row for (request, end), row in b.items() if request == 0 and end > 208}
    third = {end: row for (request, end), row in b.items() if request == 2 and end > 208}
    replay = bool(first) and set(first) == set(third) and all(
        (first[end]['bytes'], first[end]['sha256']) == (third[end]['bytes'], third[end]['sha256']) for end in first)
    return {'passed': bool(rows) and all(row['passed'] for row in rows) and replay,
            'common_complete_states': rows, 'restore_suffix_checkpoint_replay': replay,
            'scope': 'every actually emitted restore checkpoint compared to full; opaque hashes, not finite-value claims'}


def own_batch_metadata(report, batches, tensors, writes):
    """Bind capture/write metadata to its own physical call, not the other arm's index."""
    table = {}
    for request in (0, 1, 2):
        selected = [row for row in batches if row['request'] == request]
        for index, row in enumerate(selected):
            key = request, row['offset']
            if key in table:
                raise ValueError('duplicate physical batch')
            table[key] = (index, row)
    for row in [*tensors.values(), *writes]:
        match = table.get((row['request'], row['offset']))
        if match is None:
            raise ValueError('capture/write has no actual physical batch')
        index, batch = match
        if (type(row.get('input_tokens')) is not int or row['input_tokens'] != batch['tokens'] or
                type(row.get('batch_index')) is not int or row['batch_index'] != index or row.get('phase') != batch['phase']):
            raise ValueError('capture/write token count, own index or phase differs from actual batch')
    return {'passed': True, 'physical_batches': len(table), 'captured_tensors': len(tensors),
            'observed_writes': len(writes), 'scope': 'own-arm callback metadata; full and restore phase/index differences remain explicit'}


def binding_module():
    verify_frozen_sources()
    prepared = json.loads((ROOT / 'prepare-manifest.json').read_text())
    path = ROOT / 'variant_compare.py'
    if sha(path) != prepared['variant_comparator_sha256'] or sha(ROOT / 'geometry.py') != prepared['geometry_sha256']:
        raise ValueError('frozen v4 comparator/geometry source changed')
    spec = importlib.util.spec_from_file_location('noblas208_v4_review_build_binding', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_delegated_modules()
    verify_frozen_sources()
    return module


def compare(cpu_reference, reference_root, candidate_root, cpu_guard, reference_guard, candidate_guard,
            reference_build, candidate_build, binding):
    result = {'schema_version': 1, 'passed': False, 'errors': [], 'full_model_qualified': False,
              'functional_agents_qualified': False, 'performance_qualified': False,
              'snapshot_exact_evidence_passed': False, 'logits': [], 'sequence_states': [],
              'scope': 'complete noBLAS v4 profile32 diagnostics; all native gates retained plus restore checkpoints; no EOS/private-history/performance qualification',
              'prefix_raw_blob_exported': False,
              'snapshot_scope': 'fixed helper in-memory complete snapshot SHA compared to full208; raw final/continuation files hashed independently; no prefix raw file is emitted'}
    source_at_start = sha(__file__)
    try:
        v = binding_module()
        # Require the actual fullV4 arm to retain original CPU8 fidelity. No
        # standalone passed:true JSON can substitute for this raw comparison.
        fidelity = v.compare(cpu_reference, reference_root, cpu_guard, reference_guard,
                             reference_build, candidate_build, binding)
        result['cpu8_fullv4_fidelity'] = fidelity
        if fidelity.get('passed') is not True:
            raise ValueError('fullV4 has not passed complete raw original CPU8 fidelity')
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
        if a.get('microbatch') != TARGET_MICROBATCH or b.get('microbatch') != TARGET_MICROBATCH:
            raise ValueError('this version admits only the exact future noBLAS32 profile')
        pair = v.load_build_pair(reference_build, candidate_build, binding)
        v.arm_dependencies(a, ag, pair['builds']['candidate'], pair, 'candidate')
        v.arm_dependencies(b, bg, pair['builds']['candidate'], pair, 'candidate')
        registry = a.get('backend_policy_observation')
        if registry != b.get('backend_policy_observation') or registry != pair['builds']['candidate'].get('backend_policy_smoke', {}).get('observation'):
            raise ValueError('full/restore actual backend registry differs from each other or the real build smoke')
        result['backend_policy_observations_identical'] = True
        for name in ('microbatch', 'logical_batch', 'threads', 'gpu_layers', 'physical_slots', 'vocab_size', 'snapshot_key_sha256'):
            if name not in a or a[name] != b.get(name):
                raise ValueError('same actual v4 profile/key required: ' + name)
        result['actual_reference_geometry'] = v.checked_delegate(geometry.check)(a, ab)
        result['actual_candidate_geometry'] = v.checked_delegate(geometry.check)(b, bb)
        result['direct_restore_boundary'] = v.checked_delegate(geometry.direct_restore)(reference_root, a, b, ab, bb)
        if (result['direct_restore_boundary']['snapshot_bytes'] != PREFIX_BYTES or
                result['direct_restore_boundary']['snapshot_sha256'] != PREFIX_SHA256):
            raise ValueError('snapshot208 differs from the fixed original CPU8 canonical payload')
        result['own_batch_metadata'] = {
            'full': own_batch_metadata(a, ab, at, aw), 'restore': own_batch_metadata(b, bb, bt, bw)}
        full_checkpoints = checkpoint_records(reference_root, a, ab)
        restore_checkpoints = checkpoint_records(candidate_root, b, bb)
        result['complete_checkpoint_comparison'] = compare_checkpoints(full_checkpoints, restore_checkpoints)
        result['continuation'] = _compare_continuation(reference_root, candidate_root, a, b, ac, bc)
        result['continuation_replays'] = {
            'reference': _compare_continuation(reference_root, reference_root, a, a, ac, ac, ((0, 2),)),
            'candidate': _compare_continuation(candidate_root, candidate_root, b, b, bc, bc, ((0, 2),))}
        result['same_agent_replays'] = {
            'reference': _check_replay(reference_root, a, at, ab, aw),
            'candidate': _check_replay(candidate_root, b, bt, bb, bw)}
        select_batches = lambda rows: [(r['request'], r['offset'], r['tokens']) for r in rows if r['offset'] >= 208]
        if select_batches(ab) != select_batches(bb):
            raise ValueError('actual fullV4/restoreV4 suffix physical boundaries differ')
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
            result['complete_checkpoint_comparison']['passed'] and
            result['continuation']['passed'] and all(r['passed'] for r in result['continuation_replays'].values()) and
            all(r['passed'] for r in result['same_agent_replays'].values()) and result['observed_state_writes']['passed'] and
            all(checks) and all(r['passed'] for r in result['logits'] + result['sequence_states']))
        result['passed'] = result['snapshot_exact_evidence_passed']
        result['v4_binary_sha256'] = pair['binding']['candidate']['binary_sha256']
        result['binding_sha256'] = sha(binding)
        result['delegated_module_sha256'] = v.verify_delegated_modules()
        verify_frozen_sources()
        if sha(__file__) != source_at_start:
            raise ValueError('review comparator source changed during comparison')
        result['review_comparator_source_sha256'] = source_at_start
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
