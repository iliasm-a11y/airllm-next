#!/usr/bin/env python3
"""Extra exact-source checks, separated by the original prompt's input domain."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'benchmarks/check_real_world_code.py'
spec = importlib.util.spec_from_file_location('restricted_code_checker', CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def deduplicated(values):
    expected = []
    for value in values:
        if value not in expected:
            expected.append(value)
    return expected


def cases_b1():
    values = [[], [1, '1', [1], 1, '1', [1], [1, 2], [], [1, 2], []],
              [[[1], [2]], [[1], [2]], [[1], [3]], [[1], [2]]],
              [0, '', [0], [''], [[0]], 0, '', [0], [''], [[0]]],
              [[], [[]], [], [[]], [[[]]], [[[]]]],
              [1, -1, 0, 'a', [1], [[1]], ['a'], [1]]]
    rng = random.Random(20261023)
    pool = [-2, 0, 2, '', 'a', '2', [], [0], [2], ['a'], [[0]],
            [[], []], [[1], 2], [1, ['a']]]
    values += [[copy.deepcopy(rng.choice(pool)) for _ in range(rng.randrange(0, 32))]
               for _ in range(50)]
    cases = [{'id': f'B1-valid-{i + 1:02}', 'scope': 'original_contract',
              'args': [v], 'expected': deduplicated(v)} for i, v in enumerate(values)]
    # Tuples are outside the explicitly announced integer/string/list domain.
    for i, v in enumerate(([[1], (1,)], [(1,), [1]], [[], ()])):
        cases.append({'id': f'B1-tuple-extension-{i + 1}',
                      'scope': 'out_of_contract_tuple_extension',
                      'args': [v], 'expected': deduplicated(v)})
    return cases


def cases_b2():
    values = [[], [-6, -5, -2, 0, 3, 8, 10], [0, 0, 0],
              [-10**40, 1, 10**40, 6], [1, 3, 5, -7, -9]]
    rng = random.Random(20261024)
    values += [[rng.randrange(-1000, 1001) for _ in range(rng.randrange(0, 101))]
               for _ in range(30)]
    return [{'id': f'B2-valid-{i + 1:02}', 'scope': 'original_contract',
             'args': [v], 'expected': sum(n for n in v if n % 2 == 0)}
            for i, v in enumerate(values)]


def check_exact_turn(turn, cases, function_name, require_new_list):
    source, extraction = checker.extract_source(turn['text'])
    checker.validate_source(source)
    result = {'id': turn['id'], 'status': turn['status'], 'source': source,
              'source_sha256': checker.sha256(source),
              'response_sha256': checker.sha256(turn['text']),
              'extraction': extraction, 'function': function_name}
    # CASES_LITERAL is authored by this audit, never taken from generated text.
    # repr preserves tuples, which JSON alone would silently turn into lists.
    program = '''import builtins, copy, json
payload = json.load(open('payload.json'))
namespace = {'__builtins__': {name: getattr(builtins, name) for name in SAFE_BUILTINS}}
exec(compile(payload['source'], '<exact AirLLM response>', 'exec'), namespace, namespace)
function = namespace[FUNCTION_NAME]
cases = CASES_LITERAL
results = []
for case in cases:
    args = copy.deepcopy(case['args'])
    before = copy.deepcopy(args)
    entry = {'id': case['id'], 'scope': case['scope'], 'input_repr': repr(args),
             'expected_repr': repr(case['expected'])}
    try:
        actual = function(*args)
        entry.update(actual_repr=repr(actual), value_equal=actual == case['expected'],
                     input_unchanged=args == before,
                     same_object_as_first_argument=actual is args[0],
                     returned_new_list=isinstance(actual, list) and actual is not args[0])
        entry['passed'] = (entry['value_equal'] and entry['input_unchanged'] and
                           (not REQUIRE_NEW_LIST or entry['returned_new_list']))
    except BaseException as error:
        entry.update(passed=False, error=type(error).__name__ + ': ' + str(error))
    results.append(entry)
print(json.dumps({'cases': results}, ensure_ascii=False))
'''.replace('SAFE_BUILTINS', repr(checker.SAFE_BUILTINS)).replace(
        'FUNCTION_NAME', repr(function_name)).replace('CASES_LITERAL', repr(cases)).replace(
        'REQUIRE_NEW_LIST', repr(require_new_list))
    with tempfile.TemporaryDirectory(prefix='airllm-adversarial-code-') as directory:
        (Path(directory) / 'payload.json').write_text(json.dumps({'source': source}, ensure_ascii=False))
        result['execution'] = checker.run_restricted(directory, program, timeout=2)
    try:
        result['cases'] = json.loads(result['execution']['stdout'])['cases']
    except (ValueError, KeyError):
        result['cases'] = []
        result['error'] = 'Sandbox did not return all test evidence'
    result['summary'] = {}
    for scope in sorted({case['scope'] for case in cases}):
        selected = [case for case in result['cases'] if case['scope'] == scope]
        result['summary'][scope] = {
            'requested': sum(case['scope'] == scope for case in cases),
            'executed': len(selected), 'passed': sum(case['passed'] for case in selected),
            'all_inputs_unchanged': all(case.get('input_unchanged') is True for case in selected),
            'all_new_lists': all(case.get('returned_new_list') is True for case in selected)
                             if require_new_list else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, default=ROOT / 'evidence-real-world/campaign.json')
    parser.add_argument('--tasks', type=Path, default=ROOT / 'research/real_world/tasks.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', default='B1,B2')
    args = parser.parse_args()
    checker.validate_output_path(args.output, [args.responses, args.tasks])
    raw = args.responses.read_bytes()
    snapshot = json.loads(raw)
    probes = checker.verify_sandbox()
    report = {'kind': 'additional_exact_source_code_checks',
              'scope_note': 'Original-contract cases and tuple extensions are reported separately; extension failure is not B1 task failure.',
              'response_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
              'tasks_sha256': hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
              'checker_sha256': hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sandbox_probes': probes, 'results': []}
    if not probes['passed']:
        raise SystemExit('Sandbox probes failed; no generated code executed')
    turns = {turn['id']: turn for turn in snapshot['turns']}
    for turn_id in args.ids.split(','):
        turn = turns.get(turn_id)
        if not turn or turn.get('status') != 'completed':
            report['results'].append({'id': turn_id, 'executed': False,
                                      'status': turn.get('status') if turn else 'absent'})
            continue
        checked = check_exact_turn(turn, cases_b1() if turn_id == 'B1' else cases_b2(),
                                   'dedoublonne' if turn_id == 'B1' else 'somme_pairs', turn_id == 'B1')
        report['results'].append(checked)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({r['id']: r.get('summary', {'status': r['status']}) for r in report['results']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
