"""Accounting/parser regressions: no GPU or benchmark admissions involved."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'diagnose_m5_memory', Path(__file__).resolve().parents[1] / 'benchmarks/diagnose_m5_memory.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_vm_compressor_and_file_backing_are_distinct_labeled_fields():
    result = diagnostic.vm_pages('''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 2.
Pages active: 5.
File-backed pages: 3.
Pages stored in compressor: 10.
Pages occupied by compressor: 4.
Bytes of compressed tags: 17.
''')
    assert result['derived_bytes'] == {
        'free_bytes': 32768, 'active_bytes': 81920,
        'file_backed_resident_bytes': 49152,
        'compressor_physical_bytes': 65536,
        'compressor_logical_uncompressed_bytes': 163840,
    }
    assert result['counts']['Bytes of compressed tags'] == 17


def test_unrecognized_vm_format_is_not_invented_as_zero_memory():
    assert diagnostic.vm_pages('not a macOS vm_stat report')['derived_bytes'] == {}


def test_integer_budget_parser_evaluates_only_literal_integer_operations():
    assert diagnostic.integer_expr(ast.parse('3 * 1024**3', mode='eval').body) == 3 << 30
    assert diagnostic.integer_expr(ast.parse('128 << 20', mode='eval').body) == 128 << 20
    for source in ('f()', 'budget', 'True', '0.05', '1024 / 2'):
        with pytest.raises(ValueError):
            diagnostic.integer_expr(ast.parse(source, mode='eval').body)


def test_reservations_are_from_current_sources_and_keep_original_guards():
    budgets = diagnostic.configured_budgets()
    guard = budgets['air_llm/airllm/nextgen/experiment_guard.py']['literal_integer_budgets']
    assert guard['reserve_bytes'] == 3 << 30
    assert guard['admitted_payload_bytes'] == 768 << 20
    assert guard['swap_growth_bytes'] == 128 << 20
    assert all(len(item['sha256']) == 64 for item in budgets.values())


def test_default_inventory_omits_private_arguments_and_executable_directories(monkeypatch):
    command = ['python','/private/model/qwen_terminal.py','--token','private-test-sentinel']
    row = dict(pid=123,created=1.,executable='python',command=command,
               full_executable='/private/runtime/bin/python',inference_candidate=True)
    monkeypatch.setattr(diagnostic,'inventory',lambda **kwargs:dict(processes=[row],errors=[],uncertain=[],complete=True))
    process = SimpleNamespace(pid=123,create_time=lambda:1.,ppid=lambda:1,status=lambda:'sleeping',
                              memory_info=lambda:SimpleNamespace(rss=100))
    monkeypatch.setattr(diagnostic.psutil,'Process',lambda pid:process)
    public = diagnostic.process_inventory()[0]
    assert public['exe'] == 'python'
    assert public['inference_candidate'] is True
    assert 'command' not in public and 'full_executable' not in public
    private = diagnostic.process_inventory(include_command_lines=True)[0]
    assert private['command'] == command
    assert private['full_executable'] == '/private/runtime/bin/python'


def test_accounting_permission_denied_is_unknown_not_zero_and_other_processes_survive(monkeypatch):
    import psutil
    row = dict(pid=123,created=1.,executable='python',command=[],full_executable='/python',inference_candidate=False)
    monkeypatch.setattr(diagnostic,'inventory',lambda **kwargs:dict(processes=[row],errors=[],uncertain=[],complete=True))
    def denied():
        raise psutil.AccessDenied(123)
    process = SimpleNamespace(pid=123,create_time=lambda:1.,ppid=lambda:1,status=lambda:'sleeping',memory_info=denied)
    monkeypatch.setattr(diagnostic.psutil,'Process',lambda pid:process)
    observations = {}
    rows = diagnostic.process_inventory(diagnostics=observations)
    assert rows[0]['rss_bytes'] is None
    assert observations['errors'][0]['field'] == 'memory_info'


def test_diagnostic_accounting_rechecks_birth_time_before_retaining_row(monkeypatch):
    row = dict(pid=123,created=1.,executable='python',command=[],full_executable='/python',inference_candidate=False)
    monkeypatch.setattr(diagnostic,'inventory',lambda **kwargs:dict(processes=[row],errors=[],uncertain=[],complete=True))
    monkeypatch.setattr(diagnostic.psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:2.))
    observations = {}
    assert diagnostic.process_inventory(diagnostics=observations) == []
    assert observations['errors'][0]['reason'] == 'pid_reused_before_accounting'
