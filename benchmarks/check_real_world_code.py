#!/usr/bin/env python3
"""Execute exact model-produced Python in a restricted, supervised subprocess.

Only the small import-free function tasks declared in tasks.json are supported.
The AST gate and explicit builtins limit available Python features; macOS Seatbelt
independently denies network, child processes and writes outside a temporary
directory. This is a focused evaluation harness, not a general Python sandbox.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import psutil

SAFE_BUILTINS = ('len', 'range', 'enumerate', 'zip', 'sum', 'min', 'max', 'sorted',
                 'reversed', 'abs', 'all', 'any', 'bool', 'int', 'float', 'str',
                 'list', 'tuple', 'dict', 'set', 'isinstance', 'round', 'map',
                 'filter', 'TypeError', 'ValueError', 'IndexError', 'Exception')
SAFE_ATTRIBUTES = frozenset(('append', 'extend', 'count', 'index', 'copy', 'pop',
                            'insert', 'remove', 'reverse', 'sort', 'keys', 'values',
                            'items', 'get', 'setdefault', 'update', 'add', 'discard',
                            'intersection', 'union', 'difference', 'issubset'))
FORBIDDEN_NAMES = frozenset(('eval', 'exec', 'compile', 'open', 'input', 'globals',
                             'locals', 'vars', 'getattr', 'setattr', 'delattr',
                             'hasattr', 'type', 'object', 'super', 'help',
                             'breakpoint', 'exit', 'quit', 'copyright', 'license'))
ALLOWED_NODES = frozenset((ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.While, ast.If,
    ast.Compare, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Expr, ast.Call,
    ast.Attribute, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub, ast.Not, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is,
    ast.IsNot, ast.IfExp, ast.Break, ast.Continue, ast.Pass, ast.keyword,
    ast.Raise, ast.Try, ast.ExceptHandler, ast.Lambda, ast.Starred, ast.NamedExpr))


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_output_path(output, protected=()):
    """Refuse input collisions, including symlinks and hardlinks, before work."""
    output = Path(output)
    for source in map(Path, protected):
        if output.resolve() == source.resolve() or (
                output.exists() and source.exists() and output.samefile(source)):
            raise ValueError('Code-check output aliases a protected input file')


def extract_source(text):
    """Remove only a single Markdown fence when present; never repair code."""
    fences = list(re.finditer(r'^```(?:python|py)?[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)',
                              text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE))
    if not fences:
        return text, {'mode': 'raw_response', 'format_code_only': True}
    if len(fences) != 1:
        raise ValueError('Multiple fenced blocks: code extraction is ambiguous')
    match = fences[0]
    return match.group(1), {'mode': 'single_markdown_fence', 'format_code_only': False,
                            'outside_fence': text[:match.start()] + text[match.end():]}


def validate_source(source):
    if len(source.encode('utf-8')) > 64 << 10:
        raise ValueError('Code exceeds the 64 KiB evaluation limit')
    tree = ast.parse(source, filename='<model response>', mode='exec')
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            raise ValueError('Only function definitions and docstrings are allowed at module level')
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError('Unsupported AST node: ' + type(node).__name__)
        if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN_NAMES):
            raise ValueError('Forbidden name: ' + node.id)
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            raise ValueError('Forbidden attribute: ' + node.attr)
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('_') or node.name in FORBIDDEN_NAMES or node.decorator_list:
                raise ValueError('Private/dangerous/decorated function definition is unsupported')
        if isinstance(node, ast.arg) and node.arg.startswith('_'):
            raise ValueError('Private function arguments are unsupported')
    return tree


def sandbox_profile(directory):
    path_literal = json.dumps(str(Path(directory).resolve()))
    return ('(version 1)\n(allow default)\n(deny network*)\n(deny process-fork)\n'
            '(deny file-write* (require-not (subpath ' + path_literal + ')))\n')


def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    except ValueError:
        # Darwin rejects RLIMIT_AS on the tested host. Record this explicitly
        # in the probes; DATA is supplemental when supported. The parent
        # enforces sampled RSS independently on every supported platform.
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (256 << 20, 256 << 20))
        except ValueError:
            pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_restricted(directory, program, timeout=4):
    sandbox = shutil.which('sandbox-exec')
    if not sandbox:
        raise RuntimeError('macOS sandbox-exec is required; refusing unsandboxed execution')
    program_path = Path(directory) / 'runner.py'
    program_path.write_text(program)
    started = time.monotonic()
    stdout_path, stderr_path = Path(directory) / 'stdout.txt', Path(directory) / 'stderr.txt'
    timed_out = memory_stopped = False
    peak_rss = 0
    # Files, unlike pipes, are bounded by the child's RLIMIT_FSIZE. Read them
    # only after exit, so a pathological return value cannot fill parent RAM.
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        child = subprocess.Popen([sandbox, '-p', sandbox_profile(directory), sys.executable,
                                  '-I', '-S', str(program_path)],
                                 cwd=directory, env={'PATH': '/usr/bin:/bin',
                                                     'TMPDIR': str(directory),
                                                     'PYTHONDONTWRITEBYTECODE': '1'},
                                 stdout=stdout, stderr=stderr, preexec_fn=set_limits)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
                peak_rss = max(peak_rss, rss)
                if rss > 256 << 20:
                    memory_stopped = True
                    child.kill()
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - started > min(5, max(1, timeout)):
                timed_out = True
                child.kill()
            time.sleep(.01)
        child.wait()
    return {'exit_code': child.returncode,
            'stdout': stdout_path.read_bytes()[:1 << 20].decode('utf-8', errors='replace'),
            'stderr': stderr_path.read_bytes()[:1 << 20].decode('utf-8', errors='replace'),
            'timed_out': timed_out, 'memory_stopped': memory_stopped,
            'sampled_peak_rss_bytes': peak_rss, 'rss_poll_interval_seconds': .01,
            'wall_seconds': time.monotonic() - started}


def verify_sandbox():
    """Probe only controlled temporary paths and loopback socket creation."""
    with tempfile.TemporaryDirectory(prefix='airllm-code-probes-') as outer:
        inner = Path(outer) / 'allowed'
        inner.mkdir()
        outside = Path(outer) / 'forbidden-sentinel'
        program = '''import json, os, socket, resource
result = {'rlimit_as': resource.getrlimit(resource.RLIMIT_AS),
          'rlimit_data': resource.getrlimit(resource.RLIMIT_DATA),
          'rlimit_cpu': resource.getrlimit(resource.RLIMIT_CPU),
          'rlimit_fsize': resource.getrlimit(resource.RLIMIT_FSIZE),
          'rlimit_nofile': resource.getrlimit(resource.RLIMIT_NOFILE)}
try:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(.1)
    connection.connect(('127.0.0.1', 9))
    result['network_denied'] = False
except PermissionError:
    result['network_denied'] = True
except OSError as error:
    result['network_denied'] = False
    result['network_error'] = type(error).__name__ + ': ' + str(error)
try:
    open(OUTSIDE, 'w').write('controlled probe')
    result['outside_write_denied'] = False
except PermissionError:
    result['outside_write_denied'] = True
try:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    result['fork_denied'] = False
except PermissionError:
    result['fork_denied'] = True
open('inside-sentinel', 'w').write('controlled probe')
result['inside_write_allowed'] = os.path.exists('inside-sentinel')
print(json.dumps(result))
'''.replace('OUTSIDE', repr(str(outside)))
        result = run_restricted(inner, program)
        try:
            checks = json.loads(result['stdout'])
        except (ValueError, TypeError):
            checks = {}
        result['checks'] = checks
        result['passed'] = (result['exit_code'] == 0 and checks
                            and all(checks.get(name) is True for name in (
                                'network_denied', 'outside_write_denied', 'fork_denied',
                                'inside_write_allowed')) and not outside.exists())
        return result


def check_response(text, tests):
    result = {'response_text': text, 'response_sha256': sha256(text), 'tests': tests,
              'executed': False, 'passed': False}
    try:
        source, extraction = extract_source(text)
        result.update(source=source, source_sha256=sha256(source), extraction=extraction)
        validate_source(source)
        result['ast_accepted'] = True
    except (SyntaxError, ValueError) as error:
        result.update(ast_accepted=False, error=type(error).__name__ + ': ' + str(error))
        return result
    with tempfile.TemporaryDirectory(prefix='airllm-code-test-') as directory:
        payload = Path(directory) / 'payload.json'
        payload.write_text(json.dumps({'source': source, 'tests': tests}, ensure_ascii=False))
        program = '''import builtins, copy, json
payload = json.load(open('payload.json'))
allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
namespace = {'__builtins__': allowed}
results = []
try:
    exec(compile(payload['source'], '<exact model response>', 'exec'), namespace, namespace)
    function = namespace[payload['tests']['function']]
    for case in payload['tests']['cases']:
        args = copy.deepcopy(case['args'])
        before = copy.deepcopy(args)
        entry = {'args': before, 'expected': case['expected']}
        try:
            actual = function(*args)
            entry.update(actual=actual, result_equal=actual == case['expected'],
                         input_unchanged=args == before,
                         same_object_as_first_argument=bool(args) and actual is args[0])
            entry['passed'] = entry['result_equal'] and (
                not payload['tests'].get('assert_input_unchanged') or entry['input_unchanged'])
        except BaseException as error:
            entry.update(passed=False, error=type(error).__name__ + ': ' + str(error))
        results.append(entry)
    output = {'cases': results, 'passed': all(case['passed'] for case in results)}
except BaseException as error:
    output = {'cases': results, 'passed': False, 'error': type(error).__name__ + ': ' + str(error)}
print(json.dumps(output, ensure_ascii=False))
'''.replace('SAFE_BUILTINS', repr(SAFE_BUILTINS))
        run = run_restricted(directory, program, timeout=tests.get('timeout_seconds', 2))
        result.update(executed=True, process=run)
        try:
            result['test_results'] = json.loads(run['stdout'])
            result['passed'] = (run['exit_code'] == 0 and result['test_results']['passed'])
        except (ValueError, KeyError, TypeError):
            result['error'] = 'Child did not return a valid test result'
        return result


def check_turn_response(response, tests):
    """Keep interrupted turns as explicit unexecuted evidence, never pass them."""
    if response.get('status') != 'completed' or not isinstance(response.get('text'), str):
        result = {'executed': False, 'passed': False,
                  'response_status': response.get('status'),
                  'error': 'Response is not completed text; no code executed'}
        for key in ('error', 'text', 'partial_text', 'partial_generated_ids'):
            if key in response:
                result['recorded_' + key] = response[key]
        if isinstance(response.get('partial_text'), str):
            result['partial_text_sha256'] = sha256(response['partial_text'])
        return result
    return check_response(response['text'], tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path)
    parser.add_argument('--tasks', type=Path, default=Path('research/real_world/tasks.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probes-only', action='store_true')
    args = parser.parse_args()
    validate_output_path(args.output, [args.tasks, *([args.responses] if args.responses else [])])
    probes = verify_sandbox()
    result = {'kind': 'exact_model_code_restricted_execution',
              'limits': {'cpu_seconds': 3, 'requested_address_space_bytes': 256 << 20,
                         'sampled_rss_stop_bytes': 256 << 20,
                         'address_space_limit_applied': probes.get('checks', {}).get('rlimit_as') == [256 << 20, 256 << 20],
                         'data_limit_applied': probes.get('checks', {}).get('rlimit_data') == [256 << 20, 256 << 20],
                         'darwin_fallback': 'parent RSS sampling when AS/DATA unsupported; not a strict process-wide allocation cap',
                         'file_size_bytes': 1 << 20, 'file_descriptors': 16},
              'restrictions': ['sandbox-exec: deny network', 'sandbox-exec: deny process-fork',
                               'sandbox-exec: deny file-write outside temporary directory',
                               'Python -I -S', 'AST allowlist', 'explicit safe builtins'],
              'sandbox_probes': probes, 'results': []}
    if probes['passed'] and not args.probes_only:
        if args.responses is None:
            parser.error('--responses is required unless --probes-only is used')
        tasks_raw = args.tasks.read_text()
        responses_raw = args.responses.read_text()
        result.update(tasks_file=str(args.tasks), tasks_sha256=sha256(tasks_raw),
                      responses_file=str(args.responses), responses_sha256=sha256(responses_raw))
        tests = {turn['id']: turn['checks']['python_tests']
                 for session in json.loads(tasks_raw) for turn in session['turns']
                 if 'python_tests' in turn.get('checks', {})}
        responses = {turn['id']: turn for turn in json.loads(responses_raw)['turns']}
        for turn_id, specification in tests.items():
            if turn_id not in responses:
                result['results'].append({'id': turn_id, 'executed': False,
                                          'passed': False, 'error': 'Response missing'})
                continue
            checked = check_turn_response(responses[turn_id], specification)
            checked['id'] = turn_id
            result['results'].append(checked)
    result['summary'] = {'sandbox_verified': bool(probes['passed']),
                         'evaluated': len(result['results']),
                         'executed': sum(r['executed'] for r in result['results']),
                         'passed': sum(r['passed'] for r in result['results'])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
    if not probes['passed']:
        raise SystemExit('Sandbox probes failed: no model code executed')


if __name__ == '__main__':
    main()
