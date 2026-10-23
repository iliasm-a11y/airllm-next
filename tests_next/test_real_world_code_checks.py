"""Low-memory tests; can run with pytest --noconftest to avoid model imports."""
from pathlib import Path
import importlib.util

import pytest

_path = Path(__file__).resolve().parents[1] / 'benchmarks/check_real_world_code.py'
_spec = importlib.util.spec_from_file_location('real_world_code_checks', _path)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


@pytest.fixture(scope='module', autouse=True)
def sandbox_is_verified():
    result = checker.verify_sandbox()
    assert result['passed'], result


def test_exact_code_and_no_mutation():
    source = 'def dedoublonne(valeurs):\n    resultat = []\n    for v in valeurs:\n        if v not in resultat:\n            resultat.append(v)\n    return resultat\n'
    tests = {'function': 'dedoublonne', 'assert_input_unchanged': True,
             'cases': [{'args': [[3, 1, 3]], 'expected': [3, 1]},
                       {'args': [[[1], [1], [2]]], 'expected': [[1], [2]]}]}
    result = checker.check_response(source, tests)
    assert result['source'] == source
    assert result['source_sha256'] == checker.sha256(source)
    assert result['passed'], result
    assert result['process']['exit_code'] == 0
    assert all(case['same_object_as_first_argument'] is False
               for case in result['test_results']['cases'])


def test_fenced_code_is_exact_and_format_violation_separate():
    code = 'def f(values):\n    return sum(values)\n'
    text = 'Voici le code :\n```python\n' + code + '```\n'
    source, extraction = checker.extract_source(text)
    assert source == code
    assert not extraction['format_code_only']
    assert extraction['outside_fence'] == 'Voici le code :\n'


@pytest.mark.parametrize('alias', ['direct', 'symlink', 'hardlink'])
def test_output_input_alias_refused_without_mutation(tmp_path, alias):
    source = tmp_path / 'response.json'
    original = '{"turns": []}\n'
    source.write_text(original)
    output = tmp_path / 'checks.json'
    if alias == 'direct':
        output = source
    elif alias == 'symlink':
        output.symlink_to(source)
    else:
        import os
        os.link(source, output)
    with pytest.raises(ValueError, match='protected input'):
        checker.validate_output_path(output, [source])
    assert source.read_text() == original


def test_distinct_output_accepted(tmp_path):
    checker.validate_output_path(tmp_path / 'checks.json', [tmp_path / 'response.json'])


def test_wrong_generated_code_is_executed_without_repair():
    source = 'def f(values):\n    return list(set(values))\n'
    tests = {'function': 'f', 'cases': [{'args': [[[1], [1]]], 'expected': [[1]]}]}
    result = checker.check_response(source, tests)
    assert result['source'] == source
    assert result['executed'] and not result['passed']
    assert result['test_results']['cases'][0]['error'].startswith('TypeError:')


def test_input_mutation_fails_even_with_correct_return_value():
    tests = {'function': 'f', 'assert_input_unchanged': True,
             'cases': [{'args': [[2, 3]], 'expected': 2}]}
    result = checker.check_response('def f(values):\n    return values.pop(0)\n', tests)
    case = result['test_results']['cases'][0]
    assert case['result_equal'] and not case['input_unchanged']
    assert not result['passed']


def test_returned_input_alias_is_explicit_even_when_values_and_mutation_checks_pass():
    tests = {'function': 'f', 'assert_input_unchanged': True,
             'cases': [{'args': [[2, 3]], 'expected': [2, 3]}]}
    result = checker.check_response('def f(values):\n    return values\n', tests)
    case = result['test_results']['cases'][0]
    assert case['result_equal'] and case['input_unchanged']
    assert case['same_object_as_first_argument'] is True
    # Identity is reported separately because no new-list assertion was declared
    # in the frozen fixtures. Human instruction scoring must inspect this field.
    assert result['passed']


@pytest.mark.parametrize('response', [
    {'id': 'B1', 'status': 'failed', 'partial_text': 'def f(', 'partial_generated_ids': [1, 2],
     'error': 'MemoryError: supervised stop'},
    {'id': 'B1', 'status': 'started', 'partial_text': 'def f('},
    {'id': 'B1', 'status': 'started', 'text': 'def f(values):\n    return values\n'},
    {'id': 'B1', 'status': 'completed'},
])
def test_incomplete_response_retained_without_executing_partial_code(response, monkeypatch):
    def refuse_execution(*args):
        raise AssertionError('Incomplete code must not execute')
    monkeypatch.setattr(checker, 'check_response', refuse_execution)
    result = checker.check_turn_response(response, {'function': 'f', 'cases': []})
    assert result['executed'] is False and result['passed'] is False
    assert result['response_status'] == response['status']
    assert result['error'] == 'Response is not completed text; no code executed'
    if 'partial_text' in response:
        assert result['recorded_partial_text'] == response['partial_text']
        assert result['partial_text_sha256'] == checker.sha256(response['partial_text'])


def test_completed_response_still_executes_function_cases():
    result = checker.check_turn_response(
        {'status': 'completed', 'text': 'def f(values):\n    return sum(values)\n'},
        {'function': 'f', 'cases': [{'args': [[2, 3]], 'expected': 5}]})
    assert result['executed'] and result['passed']


@pytest.mark.parametrize('source', [
    'import os\ndef f():\n    return 1\n',
    'def f():\n    return open("x")\n',
    'def f():\n    return [].__class__\n',
    'f()\n',
    '@print\ndef f():\n    return 1\n',
])
def test_unsafe_ast_rejected_before_execution(source):
    result = checker.check_response(source, {'function': 'f', 'cases': []})
    assert not result['ast_accepted']
    assert not result['executed']


def test_infinite_generated_loop_is_terminated():
    result = checker.check_response('def f():\n    while True:\n        pass\n',
                                    {'function': 'f', 'cases': [{'args': [], 'expected': 1}],
                                     'timeout_seconds': 1})
    assert result['executed'] and not result['passed']
    assert result['process']['timed_out']
    assert result['process']['wall_seconds'] < 3
