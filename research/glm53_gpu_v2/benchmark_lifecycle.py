"""Owned subprocess cleanup and final global-swap accounting, without devices."""
from contextlib import contextmanager
import os
import signal
import subprocess
import threading
import weakref

_STOP_LOCKS = weakref.WeakKeyDictionary()
_STOP_LOCKS_LOCK = threading.Lock()


def _stop_lock(child):
    with _STOP_LOCKS_LOCK:
        if child not in _STOP_LOCKS:
            _STOP_LOCKS[child] = threading.Lock()
        return _STOP_LOCKS[child]


def stop_owned_server(child, *, kill_group=None):
    """The Popen child owns the new session; never target an unrelated PID."""
    if kill_group is None:
        kill_group = os.killpg
    # The monitor, request-finally and outer context can converge here. Only
    # one may signal/wait; later callers observe the same child's reaped state.
    with _stop_lock(child):
        if child.poll() is None:
            try:
                kill_group(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    try:
                        kill_group(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                child.wait(timeout=10)


def stop_for_guard(child, report, violation, *, stop_server=None):
    """Escalate during the breach itself, while a response reader may block."""
    if report.get('guard_violation') is None:
        report['guard_violation'] = violation
    if stop_server is None:
        stop_server = stop_owned_server
    try:
        stop_server(child)
    except Exception as exc:
        # Keep the initiating breach separate from an unsuccessful cleanup.
        report['guard_stop_error'] = f'{type(exc).__name__}: {exc}'


def _interrupt_on_term(*unused):
    raise KeyboardInterrupt


@contextmanager
def interrupt_on_sigterm():
    """Give a standalone CLI's owned-process finally the normal interrupt path."""
    previous = signal.signal(signal.SIGTERM, _interrupt_on_term)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def owned_server(command, *, stdout, env, after_cleanup=None, popen=None,
                 stop_server=None):
    """Install cleanup before post-spawn metadata or watcher setup can fail."""
    child = None
    error = None
    if popen is None:
        popen = subprocess.Popen
    if stop_server is None:
        stop_server = stop_owned_server
    try:
        child = popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                      env=env, start_new_session=True)
        yield child
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if child is not None:
                stop_server(child)
        except BaseException as exc:
            if error is None:
                error = exc
            raise
        finally:
            if after_cleanup is not None:
                after_cleanup(child, error)


def account_final_swap(report, final_swap_bytes):
    report["swap_final_bytes"] = final_swap_bytes
    report["peak_swap_growth_bytes"] = max(report.get("peak_swap_growth_bytes", 0),
                                           final_swap_bytes - report["swap_start_bytes"], 0)
