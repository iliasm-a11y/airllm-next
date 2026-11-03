"""Explicit absolute token fences; portable geometry, not a fidelity claim."""
from __future__ import annotations
import json
from pathlib import Path
import re

POLICY = 'cpu8-tail-global-fences208-v2'
FENCES = (128, 208, 256, 384, 512)
BATCHES = (8, 16, 32, 64, 128)
PARAMETERS = {'prefill_chunk_policy': POLICY, 'prefill_chunk_policy_version': 2,
              'tail_reference_microbatch': 8, 'prefill_fences': list(FENCES)}


def plan(ub, start, end, phase='full'):
    if (type(ub) is not int or ub not in BATCHES or type(start) is not int or type(end) is not int or
            not 0 <= start < end <= 1800):
        raise ValueError('bounded positive native chunk geometry required')
    rows = []
    while start < end:
        stop = min([end] + [f for f in FENCES if f > start])
        unfenced, raw = min(ub, end - start), min(ub, stop - start)
        take = raw - raw % 8 if raw > 8 and raw % 2 else raw
        rows.append({'phase': phase, 'offset': start, 'tokens': take, 'unfenced_take': unfenced,
            'fence_end': stop, 'decode_end': end, 'fence_limited': raw != unfenced,
            'unadjusted_take': raw, 'odd_tail_split': take != raw, 'prefill_chunk_policy': POLICY})
        start += take
    return rows


def request_plan(ub, end, mode, request):
    if mode == 'full':
        return plan(ub, 0, end)
    if end <= 208 or mode not in ('split', 'restore'):
        raise ValueError('complete private suffix and known mode required')
    prefix = plan(ub, 0, 208, 'prefix' if mode == 'split' else 'common_prefix')
    return (prefix if mode == 'split' or request == 0 else []) + plan(ub, 208, end, 'suffix')


def check(report, batches):
    params = report.get('state_parameters', {})
    if (not isinstance(params.get('prefill_fences'), list) or
            any(type(value) is not int for value in params['prefill_fences']) or
            not isinstance(report.get('prefill_fences'), list) or
            any(type(value) is not int for value in report['prefill_fences'])):
        raise ValueError('explicit integer native-token fences required')
    for key, expected in PARAMETERS.items():
        if type(params.get(key)) is not type(expected) or params.get(key) != expected:
            raise ValueError('keyed v2 policy/fences missing or different')
    if (report.get('prefill_chunk_policy') != POLICY or report.get('prefill_chunk_policy_version') != 2 or
            report.get('prefill_fences') != list(FENCES) or report.get('prefill_chunk_policy_qualified') is not False):
        raise ValueError('report v2 policy/fence identity missing or falsely qualified')
    for request, row in enumerate(report['requests']):
        expected = request_plan(report['microbatch'], row['prompt_tokens'], report['mode'], request)
        actual = [item for item in batches if item['request'] == request]
        if len(actual) != len(expected):
            raise ValueError('actual v2 physical chunk count differs from global fences')
        for a, e in zip(actual, expected):
            if any(type(a.get(k)) is not type(value) or a.get(k) != value for k, value in e.items()):
                raise ValueError('actual v2 physical boundary/fence/tail receipt differs')
    return {'passed': True, 'policy': POLICY, 'absolute_fences': list(FENCES),
            'scope': 'actual geometry only; no numerical equivalence inferred'}


def checkpoints(root, report):
    """Require all checkpoints actually prescribed by the pinned helper source.

    The old helper already writes every128 tokens. V2 additionally guarantees
    reaching208 for every UB. Required coverage grows beyond512 for longer
    bounded prompts instead of silently ignoring additional native states.
    """
    if report['mode'] != 'full':
        raise ValueError('complete full reference required for checkpoint coverage')
    path = Path(root) / 'state_checkpoints.jsonl'
    if not path.is_file() or path.stat().st_size > 1 << 20:
        raise ValueError('bounded native checkpoint metadata required')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        request, end = row.get('request'), row.get('token_end')
        if (type(request) is not int or request not in (0, 1, 2) or type(end) is not int or end < 1 or
                row.get('next_position') != end or row.get('status') != 'hashed' or row.get('state_seq_flags') != 0 or
                type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 256 << 20 or
                type(row.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise ValueError('invalid complete-state checkpoint descriptor')
        key = request, end
        if key in result:
            raise ValueError('duplicate v2 state checkpoint')
        result[key] = row
    expected = {(r, n) for r, row in enumerate(report['requests'])
                for n in ({*range(128, row['prompt_tokens'] + 1, 128), row['prompt_tokens']} |
                          ({208} if row['prompt_tokens'] >= 208 else set()))}
    if set(result) != expected:
        raise ValueError('exact full checkpoint coverage missing or extra, including208')
    for request, row in enumerate(report['requests']):
        final = result[request, row['prompt_tokens']]
        if final['bytes'] != row['sequence_bytes'] or final['sha256'] != row['sequence_sha256']:
            raise ValueError('final checkpoint differs from verified raw state')
    return result


def direct_restore(root, reference, candidate, reference_batches, candidate_batches):
    """Replace static208%UB inference by observed complete physical boundaries."""
    check(reference, reference_batches)
    check(candidate, candidate_batches)
    if reference['mode'] != 'full' or candidate['mode'] != 'restore':
        raise ValueError('direct v2 reference must be full/restore')
    states = checkpoints(root, reference)
    prefix = [states[request, 208] for request in range(3)]
    first = prefix[0]
    if any((row['bytes'], row['sha256']) != (first['bytes'], first['sha256']) for row in prefix):
        raise ValueError('three full v2 states differ at the exact208 fence')
    snapshot = candidate.get('snapshot', {})
    if snapshot.get('bytes') != first['bytes'] or snapshot.get('sha256') != first['sha256']:
        raise ValueError('immutable snapshot differs from full208 complete state')
    select = lambda rows: [(r['request'], r['offset'], r['tokens']) for r in rows if r['offset'] >= 208]
    if select(reference_batches) != select(candidate_batches):
        raise ValueError('actual full/restore suffix physical boundaries differ')
    return {'passed': True, 'boundary': 208, 'reference_checkpoints': 3,
            'snapshot_bytes': first['bytes'], 'snapshot_sha256': first['sha256'],
            'actual_fences_and_suffix_boundaries_verified': True,
            'scope': 'explicit observed boundary and exact native state; no modulo-based inference'}
