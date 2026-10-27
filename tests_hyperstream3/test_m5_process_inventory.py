"""The observed Darwin exception, process lifetime races and admission safety."""
import errno
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
import m5_process_inventory as inventory


def darwin_cmdline_error():
    try:
        raise PermissionError(errno.EACCES, 'force permission denied (originated from sysctl(KERN_PROCARGS2) -> errno 0)')
    except PermissionError as cause:
        try:
            raise SystemError('<built-in function proc_cmdline> returned a result with an exception set') from cause
        except SystemError as error:
            return error


class Backend:
    """Each Process constructor is fresh; second birth checks can differ."""
    def __init__(self, entries):
        self.entries = entries
        self.calls = {}
    def pids(self):
        return list(self.entries)
    def Process(self, pid):
        self.calls[pid] = self.calls.get(pid, 0)+1
        entry = self.entries[pid]
        if 'constructor_error' in entry:
            raise entry['constructor_error']
        second = self.calls[pid] > 1
        class Process:
            def __init__(self):
                self.pid = pid
            def __getattr__(self, field):
                def read():
                    defaults = dict(create_time=10., uids=SimpleNamespace(real=os.getuid()),
                                    exe='/usr/bin/python3', name='python3', cmdline=['python3','idle.py'])
                    value = entry.get('closing_birth' if field == 'create_time' and second else field,
                                      defaults[field])
                    if isinstance(value, BaseException):
                        raise value
                    return value
                return read
        return Process()


def test_old_attributed_iterator_raises_before_inner_try():
    """Reproduce exactly why the historical per-process catch did not apply."""
    def old_iterator():
        raise darwin_cmdline_error()
        yield  # pragma: no cover - turns this into an iterator
    with pytest.raises(SystemError, match='proc_cmdline'):
        for process in old_iterator():
            try:
                assert process
            except (psutil.Error, OSError):
                pass


@pytest.mark.parametrize('error', [psutil.AccessDenied(4), darwin_cmdline_error()])
def test_denied_python_command_is_visible_and_defers_without_crash(error):
    backend = Backend({4: dict(cmdline=error), 5: dict(exe='/usr/bin/node',cmdline=['node','service.js'])})
    result = inventory.inference_observation(backend=backend)
    assert result['process_inventory_complete'] is False
    assert result['process_inventory_uncertain'] == [dict(pid=4,created=10.,reason='inference_identity_unavailable')]
    assert result['process_inventory_count'] == 2
    assert result['other_model_processes'] == []


def test_denied_optional_command_for_known_non_inference_executable_does_not_block():
    backend = Backend({4:dict(exe='/Applications/Codex.app/Contents/MacOS/Codex',cmdline=psutil.AccessDenied(4))})
    result = inventory.inference_observation(backend=backend)
    assert result['process_inventory_complete']
    assert result['process_inventory_errors'][0]['field'] == 'cmdline'


def test_privileged_system_arguments_are_not_requested():
    backend = Backend({4:dict(uids=SimpleNamespace(real=os.getuid()+1),exe='/sbin/launchd',
                             cmdline=AssertionError('privileged command was read'))})
    assert inventory.inference_observation(backend=backend)['process_inventory_complete']


@pytest.mark.parametrize('entry', [dict(constructor_error=psutil.NoSuchProcess(4)),
                                 dict(create_time=psutil.NoSuchProcess(4)),
                                 dict(closing_birth=psutil.NoSuchProcess(4))])
def test_disappearing_process_is_skipped_and_never_leased(entry):
    result = inventory.inventory(backend=Backend({4:entry}))
    assert result['processes'] == []
    assert result['uncertain'] == []


def test_reused_pid_is_uncertain_and_no_identity_is_exposed():
    backend = Backend({4:dict(closing_birth=11.,cmdline=['python3','benchmark_research51.py'])})
    result = inventory.inference_observation(backend=backend,excluded=[dict(pid=4,created=10.)])
    assert not result['process_inventory_complete']
    assert result['process_inventory_uncertain'] == [dict(pid=4,reason='pid_reused_during_inventory')]
    assert result['other_model_processes'] == []


def test_exclusions_need_exact_pid_and_birth_time():
    entry = dict(cmdline=['python3','/repo/benchmarks/m5_validation.py','run'])
    result = inventory.inference_observation(backend=Backend({4:entry}),excluded=[dict(pid=4,created=9.)])
    assert result['other_model_processes'][0]['pid'] == 4
    result = inventory.inference_observation(backend=Backend({4:entry}),excluded=[dict(pid=4,created=10.)])
    assert result['other_model_processes'] == []


def test_name_fallback_wrapped_permission_failure_is_nonfatal():
    entry = dict(exe=psutil.AccessDenied(4), name=darwin_cmdline_error(), cmdline=psutil.AccessDenied(4))
    result = inventory.inference_observation(backend=Backend({4:entry}))
    assert not result['process_inventory_complete']
    assert {e['field'] for e in result['process_inventory_errors']} == {'exe','name','cmdline'}


@pytest.mark.parametrize('error', [SystemError('unrelated extension bug'),
                                 SystemError('<built-in function proc_cmdline> returned a result with an exception set'),
                                 RuntimeError('bad code'), OSError(errno.EIO,'I/O failure')])
def test_programming_and_unrelated_os_errors_are_not_hidden(error):
    with pytest.raises(type(error)):
        inventory.inventory(backend=Backend({4:dict(cmdline=error)}))


def test_unknown_owner_is_fail_closed():
    result = inventory.inference_observation(backend=Backend({4:dict(uids=psutil.AccessDenied(4))}))
    assert not result['process_inventory_complete']


def test_public_observation_never_contains_command_lines():
    result = inventory.inference_observation(backend=Backend({4:dict(cmdline=['python3','qwen_terminal.py','private prompt'])}))
    assert result['other_model_processes']
    assert 'private prompt' not in str(result)
