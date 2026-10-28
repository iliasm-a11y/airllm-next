"""Race-aware, read-only process inventory for M5 supervision.

Enumerate numeric PIDs, never ``process_iter(attrs=...)``: Darwin can raise a
wrapped proc_cmdline permission failure while that iterator is advancing. A
fresh Process is used for the closing birth-time check (psutil caches it).
Nothing in this module signals a process. Missing optional diagnostics remain
visible; an unidentified same-user inference-capable process prevents launch.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import psutil


WORKERS = frozenset({
    'benchmark_research51.py', 'validate_m5_kernels.py', 'qwen_terminal.py',
    'interactive_airllm.py', 'gguf_cli.py', 'gguf_chat.py', 'profile_native_gguf.py',
    'campaign_research51.py', 'm5_validation.py', 'm5_autonomous_supervisor.py',
    'bounded_recovery_keeper.py', 'benchmark_native_gguf.py',
})
NATIVE = frozenset({'llama-server', 'llama-cli', 'llama-bench'})


def expected_read_error(error):
    """Recognize access/lifetime failures, including the observed Darwin bug.

    A plain SystemError or unrelated OSError is a programming/OS failure and is
    deliberately not swallowed. Only proc_cmdline's documented exception chain
    is treated as an access/lifetime observation.
    """
    if isinstance(error, (psutil.AccessDenied, psutil.NoSuchProcess,
                          psutil.ZombieProcess, ProcessLookupError, PermissionError)):
        return True
    if isinstance(error, OSError) and error.errno in (errno.EACCES, errno.EPERM, errno.ESRCH):
        return True
    if not (isinstance(error, SystemError) and
            str(error) == '<built-in function proc_cmdline> returned a result with an exception set'):
        return False
    cause = error.__cause__ or error.__context__
    return cause is not None and not isinstance(cause, SystemError) and expected_read_error(cause)


def _lifetime_error(error):
    """Recognize a departed process, rather than an inaccessible live one."""
    if isinstance(error,(psutil.NoSuchProcess,psutil.ZombieProcess,ProcessLookupError)):
        return True
    if isinstance(error,OSError) and error.errno == errno.ESRCH:
        return True
    if (isinstance(error,SystemError)
            and str(error) == '<built-in function proc_cmdline> returned a result with an exception set'):
        cause = error.__cause__ or error.__context__
        return cause is not None and not isinstance(cause,SystemError) and _lifetime_error(cause)
    return False


class _ProcessDisappeared(Exception):
    """Internal control flow: abandon this PID for the current snapshot."""


def _read(process, field, errors):
    try:
        return getattr(process, field)()
    except Exception as error:
        if not expected_read_error(error):
            raise
        errors.append(dict(pid=process.pid, field=field, error_type=type(error).__name__,
                           reason='disappeared' if _lifetime_error(error)
                           else 'access_or_lifetime_unavailable'))
        if _lifetime_error(error):
            # Do not combine stale birth-time data with later permission errors
            # from an exited/reused PID and invent an unidentified live model.
            raise _ProcessDisappeared from None
        return None


def _status(process,errors):
    status = _read(process,'status',errors)
    if status in ('zombie','dead'):
        errors.append(dict(pid=process.pid,field='status',error_type='ProcessStatus',reason='disappeared'))
        raise _ProcessDisappeared
    return status


def _inspect(process,backend,errors,include_commands):
    pid = process.pid
    born = _read(process,'create_time',errors)
    _status(process,errors)  # Known zombies cannot be concurrent inference.
    uids = _read(process,'uids',errors)
    uid = None if uids is None else uids.real
    executable = _read(process,'exe',errors)
    # Read status before cmdline, which commonly becomes inaccessible for a
    # zombie on Darwin. Privileged system arguments remain optional/unread.
    command = _read(process,'cmdline',errors) if uid == os.getuid() else None
    name = Path(executable).name if executable else _read(process,'name',errors)
    try:
        closing = backend.Process(pid)
    except Exception as error:
        if not expected_read_error(error):
            raise
        if _lifetime_error(error):
            errors.append(dict(pid=pid,field='closing_identity',error_type=type(error).__name__,reason='disappeared'))
            raise _ProcessDisappeared from None
        closing_born = None
    else:
        closing_born = _read(closing,'create_time',errors)
        _status(closing,errors)
    if born is None or closing_born is None:
        unknown = dict(pid=pid,reason='birth_time_unavailable') if uid in (None,os.getuid()) else None
        return None,unknown
    if born != closing_born:
        return None,dict(pid=pid,reason='pid_reused_during_inventory')
    lower_name = (name or '').lower()
    possible = (not name or lower_name.startswith('python') or name in NATIVE or name == 'ollama')
    unknown = None
    if uid is None or (uid == os.getuid() and possible and command is None):
        unknown = dict(pid=pid,created=born,reason='inference_identity_unavailable')
    selected = (name in NATIVE or (name == 'ollama' and command and 'runner' in command[1:])
                or (command and any(Path(arg).name in WORKERS or arg in (
                    'airllm.nextgen.gguf_cli','airllm.nextgen.gguf_chat') for arg in command[1:])))
    row = dict(pid=pid,created=born,uid=uid,executable=name,
               inference_candidate=bool(selected),identity_verified=True)
    if include_commands:
        row['command'] = command
        row['full_executable'] = executable
    return row,unknown


def inventory(*, backend=psutil, include_commands=False):
    """Return trusted rows and explicit uncertainty, without retaining PID caches."""
    rows, errors, uncertain = [], [], []
    try:
        pids = backend.pids()
    except Exception as error:
        if not expected_read_error(error):
            raise
        return dict(processes=[], errors=[dict(field='pids', error_type=type(error).__name__)],
                    uncertain=[dict(reason='pid_enumeration_unavailable')], complete=False)
    for pid in pids:
        try:
            process = backend.Process(pid)
        except Exception as error:
            if not expected_read_error(error):
                raise
            if not _lifetime_error(error):
                uncertain.append(dict(pid=pid, reason='identity_unavailable'))
            continue
        try:
            row,unknown = _inspect(process,backend,errors,include_commands)
        except _ProcessDisappeared:
            continue
        if unknown is not None:
            uncertain.append(unknown)
        if row is not None:
            rows.append(row)
    return dict(processes=rows, errors=errors, uncertain=uncertain, complete=not uncertain)


def inference_observation(*, excluded=(), backend=psutil):
    """Exclude only exact PID/birth-time leases, never a bare PID."""
    result = inventory(backend=backend)
    leases = {(item['pid'], item['created']) for item in excluded}
    found = [row for row in result['processes'] if row['inference_candidate']
             and (row['pid'], row['created']) not in leases]
    unknown = [row for row in result['uncertain']
               if (row.get('pid'), row.get('created')) not in leases]
    return dict(other_model_processes=found, process_inventory_errors=result['errors'],
                process_inventory_uncertain=unknown, process_inventory_complete=not unknown,
                process_inventory_count=len(result['processes']))
