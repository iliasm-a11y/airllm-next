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


def _read(process, field, errors):
    try:
        return getattr(process, field)()
    except Exception as error:
        if not expected_read_error(error):
            raise
        errors.append(dict(pid=process.pid, field=field, error_type=type(error).__name__,
                           reason='disappeared' if isinstance(error, (psutil.NoSuchProcess, ProcessLookupError))
                           else 'access_or_lifetime_unavailable'))
        return None


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
            if not isinstance(error, (psutil.NoSuchProcess, ProcessLookupError)):
                uncertain.append(dict(pid=pid, reason='identity_unavailable'))
            continue
        born = _read(process, 'create_time', errors)
        uids = _read(process, 'uids', errors)
        uid = None if uids is None else uids.real
        executable = _read(process, 'exe', errors)
        # A command line is an optional diagnostic for another user's process.
        # Avoid reading privileged system arguments at all: no authority to
        # signal those processes is derived from this inventory.
        command = _read(process, 'cmdline', errors) if uid == os.getuid() else None
        name = Path(executable).name if executable else _read(process, 'name', errors)
        try:
            closing = backend.Process(pid)
            closing_born = _read(closing, 'create_time', errors)
        except Exception as error:
            if not expected_read_error(error):
                raise
            if isinstance(error, (psutil.NoSuchProcess, ProcessLookupError)):
                continue
            closing_born = None
        if born is None or closing_born is None:
            # A vanished PID is harmless; unavailable identity is never a lease.
            gone = any(e.get('pid') == pid and e.get('field') == 'create_time'
                       and e.get('reason') == 'disappeared' for e in errors)
            if not gone and uid in (None, os.getuid()):
                uncertain.append(dict(pid=pid, reason='birth_time_unavailable'))
            continue
        if born != closing_born:
            uncertain.append(dict(pid=pid, reason='pid_reused_during_inventory'))
            continue
        lower_name = (name or '').lower()
        possible = (not name or lower_name.startswith('python') or name in NATIVE or name == 'ollama')
        if uid is None or (uid == os.getuid() and possible and command is None):
            uncertain.append(dict(pid=pid, created=born, reason='inference_identity_unavailable'))
        selected = (name in NATIVE or (name == 'ollama' and command and 'runner' in command[1:])
                    or (command and any(Path(arg).name in WORKERS or arg in (
                        'airllm.nextgen.gguf_cli', 'airllm.nextgen.gguf_chat') for arg in command[1:])))
        row = dict(pid=pid, created=born, uid=uid, executable=name,
                   inference_candidate=bool(selected), identity_verified=True)
        if include_commands:
            row['command'] = command
            row['full_executable'] = executable
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
