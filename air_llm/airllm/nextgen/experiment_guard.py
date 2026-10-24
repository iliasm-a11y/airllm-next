"""Conservative admission and owned-process supervision for local research.

The external sampler complements (and cannot replace) allocation checks in the
reader, KV cache and MLX allocator. Existing swap is recorded, never hidden.
"""
from dataclasses import asdict, dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def validate_output_paths(output, protected=(), extra_outputs=()):
    """Reject direct, symlink and hardlink collisions before opening a log."""
    report = Path(output)
    destinations = [report, report.with_suffix('.log'), report.with_suffix('.guard.json'),
                    *map(Path, extra_outputs)]
    def aliases(a, b):
        if a.resolve() == b.resolve():
            return True
        return a.exists() and b.exists() and a.samefile(b)
    for i, destination in enumerate(destinations):
        if any(aliases(destination, previous) for previous in destinations[:i]):
            raise ValueError('Experiment report/log/guard outputs must be distinct files')
        if any(aliases(destination, Path(source)) for source in protected):
            raise ValueError('Experiment output aliases a protected input file')
    return destinations


def pressure_level():
    """Darwin public sysctl: 1 normal, 2 warning, 4 critical; None elsewhere."""
    if sys.platform != 'darwin':
        return None
    value = ctypes.c_int()
    length = ctypes.c_size_t(ctypes.sizeof(value))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.sysctlbyname(b'kern.memorystatus_vm_pressure_level',
                        ctypes.byref(value), ctypes.byref(length), None, 0):
        return None
    return value.value


def system_snapshot():
    memory = psutil.virtual_memory()
    return {'available_bytes': memory.available, 'physical_bytes': memory.total,
            'swap_used_bytes': psutil.swap_memory().used,
            'pressure_level': pressure_level()}


@dataclass(frozen=True)
class ExperimentLimits:
    rss_bytes: int = 2 * 1024**3
    reserve_bytes: int = 3 * 1024**3
    swap_growth_bytes: int = 128 * 1024**2
    admitted_payload_bytes: int = 768 * 1024**2
    timeout_seconds: float = 900.
    interval_seconds: float = .05
    max_processes: int = 32

    def __post_init__(self):
        for field in ('rss_bytes', 'reserve_bytes', 'swap_growth_bytes', 'admitted_payload_bytes'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        if self.rss_bytes < 1:
            raise ValueError('rss_bytes must be positive')
        if type(self.max_processes) is not int or self.max_processes < 1:
            raise ValueError('max_processes must be a positive integer')
        for field in ('timeout_seconds', 'interval_seconds'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field} must be finite and positive')
        if self.interval_seconds > 1:
            raise ValueError('Sampling interval must be at most one second')


def admission(limits, snapshot=None):
    snapshot = snapshot or system_snapshot()
    if snapshot['pressure_level'] not in (None, 1):
        raise MemoryError('macOS memory pressure is above normal before launch')
    if limits.admitted_payload_bytes + limits.reserve_bytes > snapshot['available_bytes']:
        raise MemoryError('Requested payload exceeds available memory after system reserve')
    if limits.admitted_payload_bytes > limits.rss_bytes:
        raise MemoryError('Requested payload exceeds supervised RSS ceiling')
    return snapshot


def violation(limits, before, current, rss, elapsed, process_count=1):
    if process_count > limits.max_processes:
        return 'owned_process_count'
    if rss > limits.rss_bytes:
        return 'owned_process_tree_rss'
    if current['available_bytes'] < limits.reserve_bytes:
        return 'system_reserve'
    if current['swap_used_bytes'] - before['swap_used_bytes'] > limits.swap_growth_bytes:
        return 'system_swap_growth'
    if current['pressure_level'] not in (None, 1):
        return 'macos_memory_pressure'
    if elapsed > limits.timeout_seconds:
        return 'timeout'
    return None


def _stop_group(process, observed):
    """Signal only the new session created for this experiment, including orphans."""
    def owned_group_alive():
        # poll() only returns None for our unreaped child, whose PID cannot
        # yet be reused. After exit, retain the group lease only through an
        # observed member with the same creation identity and process group.
        if process.poll() is None:return True
        for pid,created in observed.items():
            try:
                member=psutil.Process(pid)
                if member.create_time()==created and os.getpgid(pid)==process.pid:return True
            except (psutil.NoSuchProcess,psutil.AccessDenied,ProcessLookupError):pass
        return False
    if owned_group_alive():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Remember identities while the parent lives: a child can detach into its
    # own session and then disappear from parent.children(). Never signal a PID
    # that has since been reused by another application.
    survivors = []
    for pid, created in observed.items():
        try:
            child = psutil.Process(pid)
            if child.create_time() == created:
                child.terminate()
                survivors.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The parent may already have exited while descendants still live.
    if owned_group_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in survivors:
        try:
            if child.is_running() and child.create_time() == observed[child.pid]:
                child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return process.wait(timeout=5)


def supervise(command, output, limits=None, env=None):
    """Run one owned process group; always persist guard evidence, even on failure."""
    limits = limits or ExperimentLimits()
    output = Path(output)
    validate_output_paths(output, protected=[command[0]])
    output.parent.mkdir(parents=True, exist_ok=True)
    before = system_snapshot()
    report = {'command': list(map(str, command)), 'limits': asdict(limits),
              'before': before, 'sampling_scope': 'sum of owned process and descendant RSS; shared pages may be counted twice',
              'swap_scope': 'system wide; other applications can cause a conservative stop',
              'violation': None, 'exit_code': None, 'sample_count': 0,
              'sampled_peak_rss_bytes': 0, 'minimum_available_bytes': before['available_bytes'],
              'maximum_swap_growth_bytes': 0, 'peak_pressure_level': before['pressure_level']}
    process = None
    observed = {}
    begin = time.monotonic()
    try:
        admission(limits, before)
        with output.with_suffix('.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            owner = psutil.Process(process.pid)
            while True:
                current = system_snapshot()
                rss = 0
                try:
                    members = [owner] + owner.children(recursive=True)
                    for member in members:
                        try:
                            if member.pid != owner.pid:
                                observed[member.pid] = member.create_time()
                            rss += member.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                active_pids = {member.pid for member in members} if 'members' in locals() else set()
                process_count = len(active_pids)
                for pid, created in list(observed.items()):
                    if pid in active_pids:
                        continue
                    try:
                        detached = psutil.Process(pid)
                        if detached.create_time() == created:
                            rss += detached.memory_info().rss
                            process_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                report['sample_count'] += 1
                report['sampled_peak_processes'] = max(report.get('sampled_peak_processes', 0), process_count)
                report['sampled_peak_rss_bytes'] = max(report['sampled_peak_rss_bytes'], rss)
                report['minimum_available_bytes'] = min(report['minimum_available_bytes'], current['available_bytes'])
                report['maximum_swap_growth_bytes'] = max(report['maximum_swap_growth_bytes'],
                                                         current['swap_used_bytes'] - before['swap_used_bytes'])
                if current['pressure_level'] is not None:
                    report['peak_pressure_level'] = max(report['peak_pressure_level'] or 0, current['pressure_level'])
                reason = violation(limits, before, current, rss, time.monotonic() - begin, process_count)
                if reason:
                    report['violation'] = reason
                    break
                if process.poll() is not None:
                    break
                time.sleep(limits.interval_seconds)
    except BaseException as error:
        report['exception'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if process is not None:
            try:
                report['exit_code'] = _stop_group(process, observed)
            except BaseException as error:
                # A cleanup failure is a failed experiment, never a reason to
                # lose admission/sampling evidence or report a clean finish.
                report['cleanup_error']=f'{type(error).__name__}: {error}'
                report['exit_code']=process.poll()
        report['observed_descendant_count'] = len(observed)
        report['after'] = system_snapshot()
        report['elapsed_seconds'] = time.monotonic() - begin
        output.with_suffix('.guard.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['violation'] or report['exit_code'] or report.get('cleanup_error'):
        raise RuntimeError(f"Experiment stopped ({report.get('cleanup_error') or report['violation'] or report['exit_code']}); see {output.with_suffix('.guard.json')}")
    return report
