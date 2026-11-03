"""Two unchanged criteria over complete, already validated native captures.

No captured receipt, module function or dependency key is modified. Cross-GPU
serialized-state identity remains a separate, stricter verdict than allclose.
"""
from __future__ import annotations
import numpy as np


def both(reference, candidate, modules, *, integer=False):
    fidelity = modules['fidelity']
    return {mode: fidelity.compare_arrays(reference, candidate, fidelity.CRITERIA[mode], integer=integer)
            for mode in ('metal', 'cpu')}


def token_series(a_root, b_root, a_rows, b_rows, name, modules):
    f, batch = modules['fidelity'], modules['batch']
    axis = batch.TOKEN_AXES[f.family(name)]
    ai = bi = used_a = used_b = 0
    result = {'metal_passed': True, 'cpu_passed': True, 'elements': 0, 'tokens': 0,
              'metal_mismatches': 0, 'bit_mismatches': 0,
              'nonfinite_reference': 0, 'nonfinite_candidate': 0}
    while ai < len(a_rows) and bi < len(b_rows):
        ar, br = a_rows[ai], b_rows[bi]
        token = ar['offset'] + used_a
        if token != br['offset'] + used_b:
            raise ValueError('logical token positions differ during GPU stitching')
        count = min(ar['input_tokens'] - used_a, br['input_tokens'] - used_b)
        av, bv = f.tensor_view(a_root, ar), f.tensor_view(b_root, br)
        if any(av.shape[i] != bv.shape[i] for i in range(4) if i != axis):
            raise ValueError('non-token tensor dimensions differ')
        sa, sb = [slice(None)] * 4, [slice(None)] * 4
        sa[axis], sb[axis] = slice(used_a, used_a + count), slice(used_b, used_b + count)
        av, bv = av[tuple(sa)], bv[tuple(sb)]
        checks = both(av, bv, modules, integer=f.family(name) == 'routes')
        for mode in ('metal', 'cpu'):
            result[mode + '_passed'] &= checks[mode]['passed']
            if not checks[mode]['passed'] and 'first_' + mode + '_token_divergence' not in result:
                criteria = f.CRITERIA[mode]
                mask = (~np.isfinite(av) | ~np.isfinite(bv))
                if mode == 'cpu' or f.family(name) == 'routes':
                    mask |= av.view(np.uint32) != bv.view(np.uint32)
                else:
                    mask |= ~np.isclose(bv, av, rtol=criteria['rtol'], atol=criteria['atol'], equal_nan=False)
                affected = np.any(mask, axis=tuple(i for i in range(4) if i != axis))
                result['first_' + mode + '_token_divergence'] = token + int(np.flatnonzero(affected)[0])
        result['elements'] += checks['cpu']['elements']
        result['tokens'] += count
        result['metal_mismatches'] += checks['metal']['mismatch_count']
        result['bit_mismatches'] += checks['cpu']['bit_mismatch_count']
        result['nonfinite_reference'] += checks['cpu']['nonfinite_reference']
        result['nonfinite_candidate'] += checks['cpu']['nonfinite_candidate']
        used_a += count
        used_b += count
        if used_a == ar['input_tokens']:
            ai, used_a = ai + 1, 0
        if used_b == br['input_tokens']:
            bi, used_b = bi + 1, 0
    if ai != len(a_rows) or bi != len(b_rows):
        raise ValueError('one GPU token series ended before the other')
    return result


def continuation(a_root, b_root, a, b, ac, bc, modules):
    f, native = modules['fidelity'], modules['native']
    result = {'metal_numeric_passed': True, 'cpu_bitexact_passed': True,
              'serialized_states_exact': True, 'observed_state_writes_exact': True, 'requests': []}
    for request, (ar, br) in enumerate(zip(a['requests'], b['requests'])):
        aa, bb = ar['continuation'], br['continuation']
        tokens_equal = ([(r['token'], r['native_eog']) for r in aa] ==
                        [(r['token'], r['native_eog']) for r in bb] and
                        ar['continuation_stop'] == br['continuation_stop'])
        row = {'request': request, 'tokens_equal': tokens_equal, 'logits': [], 'tensors': [],
               'serialized_state_exact': ar['continuation_state_bytes'] == br['continuation_state_bytes'] and
                                         ar['continuation_state_sha256'] == br['continuation_state_sha256']}
        if len(aa) == len(bb):
            for step in range(len(aa)):
                row['logits'].append({'step': step, **both(
                    np.memmap(a_root / f'continuation-logits-{request}-{step}.f32', dtype='<f4', mode='r'),
                    np.memmap(b_root / f'continuation-logits-{request}-{step}.f32', dtype='<f4', mode='r'), modules)})
        select = lambda data: {k[1:]: v for k, v in data['tensors'].items() if k[0] == request}
        at, bt = select(ac), select(bc)
        if set(at) != set(bt):
            raise ValueError('complete continuation tensor key sets differ')
        for key, av in sorted(at.items()):
            row['tensors'].append({'key': key, **both(f.tensor_view(a_root, av), f.tensor_view(b_root, bt[key]),
                modules, integer=f.family(av['name']) == 'routes')})
        select_writes = lambda data: [{**r, 'request': 0} for r in data['writes'] if r['request'] == request]
        row['observed_state_writes'] = native._compare_writes(select_writes(ac), select_writes(bc),
                                                              [(0, n) for n in range(1, len(aa))], False)
        checks = row['logits'] + row['tensors']
        complete = tokens_equal and len(row['logits']) == len(aa)
        for mode, field in (('metal', 'metal_numeric_passed'), ('cpu', 'cpu_bitexact_passed')):
            row[field] = bool(complete and all(r[mode]['passed'] for r in checks))
            result[field] &= row[field]
        result['serialized_states_exact'] &= row['serialized_state_exact']
        result['observed_state_writes_exact'] &= row['observed_state_writes']['passed']
        result['requests'].append(row)
    return result


def cpu_gpu(reference_root, candidate_root, a_data, b_data, modules):
    """Raw validation is delegated first; no partial request can reach here."""
    a, at, ab, _, aw, ac = a_data
    b, bt, bb, _, bw, bc = b_data
    f, native, batch = modules['fidelity'], modules['native'], modules['batch']
    if a['mode'] != 'full' or b['mode'] != 'full':
        raise ValueError('CPU/GPU experiment requires complete full/full arms')
    for field in ('vocab_size', 'threads', 'gpu_layers', 'physical_slots'):
        if a[field] != b[field]:
            raise ValueError('incompatible native profile: ' + field)
    result = {'criteria': {'metal': f.CRITERIA['metal'], 'cpu_diagnostic': f.CRITERIA['cpu'],
                          'ordered_routes_exact': True, 'tokens_exact': True, 'all_values_finite': True},
              'logits': [], 'tensors': [], 'final_states': [], 'checkpoints': [],
              'coverage': {k: 0 for k in batch.TOKEN_AXES}, 'full_model_qualified': False,
              'performance_qualified': False, 'snapshot_exact_qualified': False,
              'cross_batch_state_writes': 'coverage only; unequal physical chunk boundaries are not compared as equal events'}
    checkpoints_a, checkpoints_b = batch._checkpoints(reference_root, a), batch._checkpoints(candidate_root, b)
    result['checkpoint_replays'] = {'reference': batch._checkpoint_replay(checkpoints_a),
                                    'candidate': batch._checkpoint_replay(checkpoints_b)}
    for key in sorted(set(checkpoints_a) & set(checkpoints_b)):
        x, y = checkpoints_a[key], checkpoints_b[key]
        result['checkpoints'].append({'request': key[0], 'token_end': key[1],
            'passed': x['bytes'] == y['bytes'] and x['sha256'] == y['sha256']})
    result['checkpoint208_compared'] = any(row['token_end'] == 208 for row in result['checkpoints'])
    result['same_agent_replays'] = {
        'reference': native._check_replay(reference_root, a, at, ab, aw),
        'candidate': native._check_replay(candidate_root, b, bt, bb, bw)}
    result['continuation_replays'] = {
        'reference': native._compare_continuation(reference_root, reference_root, a, a, ac, ac, ((0, 2),)),
        'candidate': native._compare_continuation(candidate_root, candidate_root, b, b, bc, bc, ((0, 2),))}
    result['write_coverage'] = {
        'reference': native._compare_writes(aw, aw, [(r['request'], r['offset']) for r in ab], False),
        'candidate': native._compare_writes(bw, bw, [(r['request'], r['offset']) for r in bb], False)}
    for request, (ar, br) in enumerate(zip(a['requests'], b['requests'])):
        for field in ('prompt_tokens', 'prompt_ids_sha256', 'next_position'):
            if ar[field] != br[field]:
                raise ValueError('complete native prompts/positions differ: ' + field)
        result['logits'].append({'request': request, **both(
            np.memmap(reference_root / f'logits-{request}.f32', dtype='<f4', mode='r'),
            np.memmap(candidate_root / f'logits-{request}.f32', dtype='<f4', mode='r'), modules)})
        result['final_states'].append({'request': request,
            'passed': ar['sequence_bytes'] == br['sequence_bytes'] and ar['sequence_sha256'] == br['sequence_sha256']})
    series_a, series_b = batch._series(a, at, ab), batch._series(b, bt, bb)
    if set(series_a) != set(series_b):
        raise ValueError('complete118-name token series differ')
    for request, name in sorted(series_a):
        check = token_series(reference_root, candidate_root, series_a[request, name], series_b[request, name], name, modules)
        result['tensors'].append({'request': request, 'name': name, **check})
        result['coverage'][f.family(name)] += check['elements']
    result['continuation'] = continuation(reference_root, candidate_root, a, b, ac, bc, modules)
    self_valid = (all(result['checkpoint_replays'].values()) and
                  all(r['passed'] for r in result['same_agent_replays'].values()) and
                  all(r['passed'] for r in result['continuation_replays'].values()) and
                  all(r['passed'] for r in result['write_coverage'].values()))
    result['metal_numeric_passed'] = bool(self_valid and all(r['metal']['passed'] for r in result['logits']) and
        all(r['metal_passed'] for r in result['tensors']) and result['continuation']['metal_numeric_passed'])
    result['cpu_gpu_serialized_state_exact'] = bool(all(r['passed'] for r in result['final_states'] + result['checkpoints']) and
        result['continuation']['serialized_states_exact'])
    result['cpu_bitexact_passed'] = bool(self_valid and all(r['cpu']['passed'] for r in result['logits']) and
        all(r['cpu_passed'] for r in result['tensors']) and result['continuation']['cpu_bitexact_passed'] and
        result['cpu_gpu_serialized_state_exact'] and result['continuation']['observed_state_writes_exact'])
    # This first Q8 bridge experiment additionally demands exact opaque native
    # state evidence. Metal allclose alone is reported honestly as insufficient.
    result['passed'] = bool(result['metal_numeric_passed'] and result['cpu_gpu_serialized_state_exact'] and
                            result['continuation']['observed_state_writes_exact'])
    result['state_values_unverified_if_only_allclose'] = not result['cpu_gpu_serialized_state_exact']
    return result
