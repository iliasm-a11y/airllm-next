#!/usr/bin/env python3
"""Independently compare emitted speculative logits against a serial recording.

Only Python's standard library is used. Every FP32 artifact is shape/hash/finite
checked, including discarded speculative paths. Parity compares only rows whose
entire recorded consumed token prefix matches the final emitted transcript.
Rollback/replay may produce duplicate rows; every matching row must agree with
the serial reference. This checks recorded evidence, not its authenticity,
checkpoint SHA integrity, hidden recurrent arrays, timing or guard compliance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


class RejectedEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RejectedEvidence(message)


def integers(value, name, *, nonempty=False):
    require(isinstance(value, list) and (value or not nonempty), 'Missing '+name)
    require(all(type(token) is int and 0 <= token <= 2**31-1 for token in value),
            'Invalid token IDs: '+name)
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _row_values(reference):
    path, offset, width = reference
    with path.open('rb') as stream:
        stream.seek(offset)
        raw = stream.read(width*4)
    require(len(raw) == width*4, 'Truncated logits row')
    return [value[0] for value in struct.iter_unpack('<f', raw)]


def _argmax(values):
    require(values and all(math.isfinite(value) for value in values), 'Non-finite logits row')
    return max(range(len(values)), key=values.__getitem__)


def _recording(path, *, serial):
    path = Path(path).resolve()
    raw = path.read_bytes()
    try:
        report = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(RejectedEvidence(value)))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RejectedEvidence('Invalid worker JSON') from error
    require(isinstance(report, dict), 'Worker report must be an object')
    require(report.get('status') == 'completed', 'Worker did not complete')
    for flag in ('source_unchanged', 'protocol_unchanged', 'model_stat_unchanged', 'engine_head_unchanged'):
        require(report.get(flag) is True, 'Worker identity changed: '+flag)
    require(report.get('git_head') == report.get('git_head_after') and
            isinstance(report.get('git_head'), str) and
            re.fullmatch('[0-9a-f]{40}', report['git_head']), 'Invalid worker commit')
    hashes = report.get('source_sha256')
    require(isinstance(hashes, dict) and hashes and all(valid_hash(v) for v in hashes.values()),
            'Missing frozen source hashes')
    model = report.get('model', {})
    require(isinstance(model, dict), 'Missing checkpoint identity')
    require(isinstance(model.get('path'), str) and model['path'], 'Missing checkpoint path')
    for field in ('bytes', 'inode', 'mtime_ns'):
        require(type(model.get(field)) is int and model[field] > 0, 'Missing checkpoint stat: '+field)
    config = report.get('model_config', {})
    require(isinstance(config, dict), 'Missing model configuration')
    for field in ('vocab_size', 'num_hidden_layers', 'num_experts', 'num_experts_per_tok'):
        require(type(config.get(field)) is int and config[field] > 0, 'Invalid model configuration: '+field)
    vocab = config['vocab_size']
    require(config['num_experts_per_tok'] <= config['num_experts'], 'Invalid expert top-k')
    turns = report.get('turns')
    require(isinstance(turns, list) and turns, 'Missing completed turns')
    retained = []
    checked_artifacts = {}
    for turn in turns:
        require(isinstance(turn, dict), 'Turn must be an object')
        require(turn.get('status') == 'completed', 'Turn did not complete')
        prefix = integers(turn.get('full_input_ids'), 'full_input_ids', nonempty=True)
        generated = integers(turn.get('generated_ids'), 'generated_ids', nonempty=True)
        transcript = prefix+generated
        require(turn.get('full_output_ids') == transcript, 'Final output does not preserve input and output IDs')
        require(all(token < vocab for token in transcript), 'Transcript exceeds vocabulary')
        eos = integers(turn.get('eos_token_ids'), 'eos_token_ids')
        require(all(token < vocab for token in eos), 'EOS exceeds vocabulary')
        require(not any(token in eos for token in generated[:-1]), 'Generation continued after EOS')
        stop = 'eos' if generated[-1] in eos else 'max_new_tokens'
        require(turn.get('stop_reason') == stop, 'Stop reason disagrees with emitted EOS')
        require(isinstance(turn.get('text'), str) and isinstance(turn.get('raw_text'), str), 'Missing raw output text')
        steps = turn.get('step_checks')
        require(isinstance(steps, list) and steps, 'Missing captured forwards')
        if serial:
            require(len(steps) == len(generated), 'Reference must perform one forward per emitted token')
        history = []
        rows = {}
        routes = {}
        all_rows = valid_rows = 0
        for index, step in enumerate(steps):
            require(isinstance(step, dict), 'Captured forward must be an object')
            supplied = integers(step.get('input_ids'), 'forward input_ids', nonempty=True)
            require(all(token < vocab for token in supplied), 'Forward input exceeds vocabulary')
            start, end = step.get('start_position'), step.get('position')
            require(type(start) is int and 0 <= start <= len(history), 'Forward starts after an unrecorded prefix')
            require(type(end) is int and end == start+len(supplied), 'Forward position does not match input length')
            history = history[:start]+supplied
            if serial:
                require(start == (0 if index == 0 else len(prefix)+index-1) and
                        supplied == (prefix if index == 0 else [generated[index-1]]),
                        'Reference is not serial greedy generation')
            artifact = step.get('logits_artifact', {})
            require(isinstance(artifact, dict), 'Missing logits artifact')
            relative = artifact.get('path')
            require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                    'Logits artifact path must be relative')
            vector_path = (path.parent/relative).resolve()
            require(vector_path.is_relative_to(path.parent), 'Logits artifact escapes evidence folder')
            require(artifact.get('dtype') == 'float32', 'Logits must be little-endian float32')
            shape = artifact.get('shape')
            require(isinstance(shape, list) and len(shape) == 3 and
                    all(type(n) is int and n > 0 for n in shape) and
                    shape[0] == 1 and 1 <= shape[1] <= len(supplied) and shape[2] == vocab,
                    'Logits artifact shape does not match the forward')
            require(not serial or shape[1] == 1, 'Serial reference must retain one output position')
            require(vector_path.stat().st_size == math.prod(shape)*4, 'Logits artifact size differs from shape')
            declared = artifact.get('sha256')
            require(valid_hash(declared) and step.get('logits_sha256') == declared,
                    'Missing or inconsistent logits digest')
            actual = digest_file(vector_path)
            require(actual == declared, 'Logits digest differs from artifact')
            require(vector_path not in checked_artifacts or checked_artifacts[vector_path] == actual,
                    'A logits artifact changed during verification')
            checked_artifacts[vector_path] = actual
            first_position = end-shape[1]
            for row in range(shape[1]):
                position = first_position+row
                row_ref = (vector_path, row*vocab*4, vocab)
                values = _row_values(row_ref)
                prediction = _argmax(values)
                all_rows += 1
                # A row at input position p predicts the token at p+1.
                output_index = position+1-len(prefix)
                if (0 <= output_index < len(generated) and
                        history[:position+1] == transcript[:position+1]):
                    require(prediction == generated[output_index], 'Captured argmax differs from emitted token')
                    rows.setdefault(output_index, []).append(row_ref)
                    valid_rows += 1
            banks = step.get('routing_by_layer_position')
            require(isinstance(banks, dict) and set(banks) == {str(i) for i in range(config['num_hidden_layers'])},
                    'Missing per-forward routes for all layers')
            for layer, bank in banks.items():
                require(isinstance(bank, dict) and set(bank) == {str(p) for p in range(start, end)},
                        'Missing per-forward routes for every consumed position')
                for position_text, experts in bank.items():
                    position = int(position_text)
                    require(isinstance(experts, list) and len(experts) == config['num_experts_per_tok'] and
                            all(type(e) is int and 0 <= e < config['num_experts'] for e in experts) and
                            len(set(experts)) == len(experts),
                            'Invalid or incomplete selected experts')
                    if (position < len(transcript)-1 and
                            history[:position+1] == transcript[:position+1]):
                        key = (layer, position)
                        require(key not in routes or routes[key] == experts,
                                'Repeated accepted-prefix routes disagree')
                        routes[key] = experts
        require(set(rows) == set(range(len(generated))), 'Missing a verified logit row for an emitted token')
        expected_routes = {(str(layer), position) for layer in range(config['num_hidden_layers'])
                           for position in range(len(transcript)-1)}
        require(set(routes) == expected_routes, 'Missing accepted-prefix routes')
        retained.append(dict(id=turn.get('id'), prefix=prefix, generated=generated,
            eos=eos, stop=stop, rows=rows, routes=routes,
            all_rows=all_rows, matching_rows=valid_rows))
    return dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), model=model,
                config=config, turns=retained, artifacts=checked_artifacts)


def verify(serial_report, candidate_report, *, atol=1e-4, rtol=1e-4):
    for value in (atol, rtol):
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-4,
                'Numerical tolerances must be finite within 1e-4')
    serial = _recording(serial_report, serial=True)
    candidate = _recording(candidate_report, serial=False)
    require(serial['model'] == candidate['model'], 'Checkpoint recorded identities differ')
    require(serial['config'] == candidate['config'], 'Model configurations differ')
    require(len(serial['turns']) == len(candidate['turns']), 'Turn counts differ')
    comparisons = []
    for a, b in zip(serial['turns'], candidate['turns']):
        for key in ('id', 'prefix', 'generated', 'eos', 'stop'):
            require(a[key] == b[key], 'Turn transcripts or EOS differ: '+key)
        require(a['routes'] == b['routes'], 'Accepted-prefix expert routes differ')
        maximum_absolute = maximum_relative = 0.
        elements = 0
        bitwise = True
        for output_index, reference_rows in a['rows'].items():
            reference = _row_values(reference_rows[0])
            for row_ref in b['rows'][output_index]:
                values = _row_values(row_ref)
                for expected, observed in zip(reference, values):
                    error = abs(observed-expected)
                    relative = error/max(abs(expected), 1e-30)
                    require(error <= atol+rtol*abs(expected), 'Emitted full logits exceed numerical tolerance')
                    maximum_absolute = max(maximum_absolute, error)
                    maximum_relative = max(maximum_relative, relative)
                    elements += 1
                    bitwise = bitwise and struct.pack('<f', expected) == struct.pack('<f', observed)
        comparisons.append(dict(id=a['id'], emitted_tokens=len(a['generated']),
            full_logits_elements_checked=elements, logits_max_absolute_error=maximum_absolute,
            logits_max_relative_error=maximum_relative, logits_bitwise_exact=bitwise,
            candidate_rows_checked=b['all_rows'], candidate_rows_matching_emitted_prefix=b['matching_rows'],
            recorded_routes_checked=len(b['routes'])))
    # Detect modifications during the potentially long independent comparison.
    for recording in (serial, candidate):
        require(digest_file(recording['path']) == recording['sha256'], 'Worker report changed during verification')
        for path, expected in recording['artifacts'].items():
            require(digest_file(path) == expected, 'Logits changed during verification')
    return dict(status='accepted', scope='recorded emitted logits/tokens and accepted-prefix expert routes',
        serial_report_sha256=serial['sha256'], candidate_report_sha256=candidate['sha256'],
        atol=atol, rtol=rtol, exact_tokens=True, recorded_accepted_prefix_routes_equal=True,
        full_recurrent_state_parity_validated=False, checkpoint_hash_integrity_validated=False,
        guard_compliance_validated=False, timing_speedup_validated=False,
        comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serial_report', type=Path)
    parser.add_argument('candidate_report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.serial_report, args.candidate_report)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output is None:
        print(text, end='')
    else:
        # Exclusive creation prevents overwriting reports, artifacts or logs.
        with args.output.open('x') as stream:
            stream.write(text)


if __name__ == '__main__':
    main()
