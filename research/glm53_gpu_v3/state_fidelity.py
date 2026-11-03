"""Exact hashes of observed mutable-cache writes; not a claim of uncaptured state."""
import json
import math
import re
import struct
from pathlib import Path


def read_states(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('state hash capture is missing')
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError('state metadata exceeds parsing bound')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) > 65536:
        raise ValueError('state rows empty or exceed bound')
    for row in rows:
        if (not isinstance(row, dict) or row.get('status') != 'hashed' or row.get('finite') is not True or
                not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) or
                type(row.get('step')) is not int or row['step'] < 0 or
                type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= 512 * 1024**2):
            raise ValueError('unhashed, nonfinite or invalid cache-write record')
    return rows


def compare_states(reference, candidate):
    result = {'passed': False, 'scope': 'ordered observed CPY/SET_ROWS cache writes only; no proof for unobserved state',
              'reference': str(reference), 'candidate': str(candidate)}
    try:
        a, b = read_states(reference), read_states(candidate)
        keys = ('step', 'target', 'op', 'type', 'bytes', 'shape', 'sha256')
        mismatches = sum(any(x.get(k) != y.get(k) for k in keys) for x, y in zip(a, b)) + abs(len(a)-len(b))
        result.update(passed=mismatches == 0, reference_events=len(a), candidate_events=len(b),
                      compared_events=min(len(a),len(b)), mismatching_events=mismatches,
                      compared_logical_bytes=sum(r.get('logical_bytes',r['bytes']) for r in a) if mismatches == 0 else None,
                      compared_storage_span_bytes=sum(r['bytes'] for r in a) if mismatches == 0 else None)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result['error'] = str(exc)
    return result


def compare_reset(capture, per_request):
    """Two complete requests must repeat all full logits and observed cache writes."""
    root = Path(capture).resolve()
    result = {'passed': False, 'scope': 'same prompt repeated twice after llama_memory_clear; full logits and observed cache writes'}
    try:
        if type(per_request) is not int or not 1 <= per_request <= 64:
            raise ValueError('per_request must be an integer in1..64')

        def payload(name, limit):
            path = (root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('missing or escaping capture file: '+name)
            size = path.stat().st_size
            if size > limit:
                raise ValueError('capture file exceeds parsing bound: '+name)
            raw = path.read_bytes()
            if len(raw) != size or len(raw) > limit:
                raise ValueError('capture file changed while being read: '+name)
            return raw

        complete = json.loads(payload('complete.json', 65536))
        if (not isinstance(complete, dict) or complete.get('completed') is not True or
                type(complete.get('captured_steps')) is not int or complete['captured_steps'] != 2*per_request):
            raise ValueError('two-request capture is not complete')
        tokens = [json.loads(line) for line in payload('tokens.jsonl', 65536).splitlines() if line.strip()]
        if len(tokens) != 2*per_request:
            raise ValueError('token records must contain exactly two full requests')
        vocab = None
        for step, row in enumerate(tokens):
            if (not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != step or
                    type(row.get('request')) is not int or row['request'] != step//per_request or
                    type(row.get('vocab')) is not int or not 1 <= row['vocab'] <= 1_000_000):
                raise ValueError('invalid token step/request/vocabulary metadata')
            if vocab is None:
                vocab = row['vocab']
            if row['vocab'] != vocab:
                raise ValueError('vocabulary changes within the reset capture')
            for key in ('token', 'next_input_token'):
                if type(row.get(key)) is not int or not 0 <= row[key] < vocab:
                    raise ValueError('invalid recorded token ID: '+key)
        expected_names = {f'logits-{step}.f32' for step in range(2*per_request)}
        if {path.name for path in root.glob('logits-*.f32')} != expected_names:
            raise ValueError('full-logit files are missing or extra')
        expected_bytes = vocab*4
        same_logits = same_tokens = same_inputs = True
        for i in range(per_request):
            pair = []
            for step in (i, i+per_request):
                raw = payload(f'logits-{step}.f32', expected_bytes)
                if len(raw) != expected_bytes:
                    raise ValueError('full-logit byte length differs from recorded vocabulary')
                best, winner = -math.inf, -1
                for index, (value,) in enumerate(struct.iter_unpack('<f', raw)):
                    if not math.isfinite(value):
                        raise ValueError('nonfinite full-logit value')
                    if value > best:
                        best, winner = value, index
                if winner != tokens[step]['token']:
                    raise ValueError('recorded greedy token disagrees with full logits')
                pair.append(raw)
            same_logits &= pair[0] == pair[1]
            same_tokens &= tokens[i]['token'] == tokens[i+per_request]['token']
            same_inputs &= tokens[i]['next_input_token'] == tokens[i+per_request]['next_input_token']
        # Validate the fixed file path before handing it to the common reader.
        payload('state_hashes.jsonl', 16*1024**2)
        rows = read_states(root/'state_hashes.jsonl')
        state_steps = [row['step'] for row in rows]
        if set(state_steps) != set(range(2*per_request)) or state_steps != sorted(state_steps):
            raise ValueError('observed state steps are missing, extra or out of order')
        for row in rows:
            shape = row.get('shape')
            if (not isinstance(row.get('target'), str) or not row['target'].startswith('cache_') or
                    row.get('op') not in ('CPY', 'SET_ROWS') or type(row.get('type')) is not int or
                    row['type'] not in (0, 1, 30) or not isinstance(shape, list) or len(shape) != 4 or
                    any(type(n) is not int or n < 0 for n in shape)):
                raise ValueError('invalid observed cache-write descriptor')
            logical = (4 if row['type'] == 0 else 2)*math.prod(shape)
            if (logical > 512*1024**2 or logical > row['bytes'] or
                    ('logical_bytes' in row and (type(row['logical_bytes']) is not int or row['logical_bytes'] != logical))):
                raise ValueError('invalid observed state logical byte coverage')
        a = [{**r} for r in rows if r['step'] < per_request]
        b = [{**r, 'step':r['step']-per_request} for r in rows if per_request <= r['step'] < 2*per_request]
        keys = ('step','target','op','type','bytes','shape','logical_bytes','sha256')
        same_states = len(a)==len(b) and bool(a) and all(all(x.get(k)==y.get(k) for k in keys) for x,y in zip(a,b))
        result.update(passed=same_states and same_logits and same_tokens and same_inputs,
                      logits_bit_identical=same_logits, generated_ids_identical=same_tokens,
                      next_inputs_identical=same_inputs, finite_logit_vectors_validated=2*per_request,
                      vocab=vocab, expected_logit_bytes_per_vector=expected_bytes,
                      request_token_counts=[per_request,per_request],
                      observed_states_bit_identical=same_states, state_events_per_request=[len(a),len(b)])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result['error'] = str(exc)
    return result
