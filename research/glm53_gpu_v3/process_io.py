"""Read one authorized PID's libproc counters; never enumerate command lines.

Storage bytes are OS-accounted process IO, not per-PID NAND traffic. Faults,
pageins and CPU time do not isolate file-cache hits or pure weight wait time.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time


RUSAGE_V2_FIELDS = ('user_ticks', 'system_ticks', 'idle_wakeups', 'interrupt_wakeups', 'pageins',
    'wired_bytes', 'resident_bytes', 'physical_footprint_bytes', 'start_abstime', 'exit_abstime',
    'child_user_ticks', 'child_system_ticks', 'child_idle_wakeups', 'child_interrupt_wakeups',
    'child_pageins', 'child_elapsed_ticks', 'os_storage_read_bytes', 'os_storage_write_bytes')
RUSAGE_V3_FIELDS = ('qos_default', 'qos_maintenance', 'qos_background', 'qos_utility', 'qos_legacy',
    'qos_user_initiated', 'qos_user_interactive', 'billed_system_ticks', 'serviced_system_ticks')
RUSAGE_V4_FIELDS = ('logical_writes', 'lifetime_max_footprint_bytes', 'instructions', 'cycles',
    'billed_energy', 'serviced_energy', 'interval_max_footprint_bytes', 'runnable_ticks')


class UsageV4(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in
        (*RUSAGE_V2_FIELDS, *RUSAGE_V3_FIELDS, *RUSAGE_V4_FIELDS)]


class TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('virtual_bytes', 'resident_bytes', 'total_user',
        'total_system', 'threads_user', 'threads_system')] + [(name, ctypes.c_int32) for name in
        ('policy', 'faults', 'pageins', 'cow_faults', 'messages_sent', 'messages_received',
         'syscalls_mach', 'syscalls_unix', 'context_switches', 'thread_count', 'running_threads', 'priority')]


class Timebase(ctypes.Structure):
    _fields_ = [('numer', ctypes.c_uint32), ('denom', ctypes.c_uint32)]


class TaskVMRev1(ctypes.Structure):
    _layout_ = 'ms'  # ctypes' explicit packed layout; Mach declares pragma pack(4).
    _pack_ = 4
    _fields_ = [('virtual_bytes', ctypes.c_uint64), ('region_count', ctypes.c_int32),
        ('page_size', ctypes.c_int32)] + [(name, ctypes.c_uint64) for name in
        ('resident_bytes', 'resident_peak_bytes', 'device_bytes', 'device_peak_bytes',
         'internal_bytes', 'internal_peak_bytes', 'external_bytes', 'external_peak_bytes',
         'reusable_bytes', 'reusable_peak_bytes', 'purgeable_volatile_pmap_bytes',
         'purgeable_volatile_resident_bytes', 'purgeable_volatile_virtual_bytes',
         'compressed_bytes', 'compressed_peak_bytes', 'compressed_lifetime_bytes', 'physical_footprint_bytes')]


def self_task_vm_snapshot(libsystem=None):
    """TASK_VM_INFO for this observer only; no task_for_pid privilege request."""
    try:
        lib = libsystem or ctypes.CDLL(None)
        port = ctypes.c_uint32.in_dll(lib, 'mach_task_self_').value
        lib.task_info.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        lib.task_info.restype = ctypes.c_int
        data = TaskVMRev1()
        count = ctypes.c_uint32(ctypes.sizeof(data) // 4)
        status = lib.task_info(port, 22, ctypes.byref(data), ctypes.byref(count))
        if status:
            return {'available': False, 'reason': 'TASK_VM_INFO returned ' + str(status)}
        fields = {name: int(getattr(data, name)) if getattr(TaskVMRev1, name).offset + ctypes.sizeof(kind) <= count.value * 4 else None
                  for name, kind in TaskVMRev1._fields_}
        return {'available': True, 'returned_words': count.value, **fields,
                'scope': 'observer process only; external model PID compressed/device fields require a native self-report'}
    except (OSError, ValueError, AttributeError) as exc:
        return {'available': False, 'reason': f'{type(exc).__name__}: {exc}'}


class ProcessIOObserver:
    def __init__(self, pid, *, libproc=None, libsystem=None):
        if type(pid) is not int or pid <= 0:
            raise ValueError('an explicitly authorized positive PID is required')
        self.pid = pid
        self.libproc = libproc or ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.libsystem = libsystem or ctypes.CDLL(None)
        self.libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.libproc.proc_pid_rusage.restype = ctypes.c_int
        self.libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        self.libproc.proc_pidinfo.restype = ctypes.c_int
        self.start_identity = None
        base = Timebase()
        self.libsystem.mach_timebase_info.argtypes = [ctypes.c_void_p]
        status = self.libsystem.mach_timebase_info(ctypes.byref(base))
        self.timebase = {'numer': base.numer, 'denom': base.denom} if not status and base.denom else None

    def _usage(self):
        for flavor in (4, 2):
            data = UsageV4()
            if self.libproc.proc_pid_rusage(self.pid, flavor, ctypes.byref(data)) == 0:
                fields = (*RUSAGE_V2_FIELDS, *RUSAGE_V3_FIELDS, *RUSAGE_V4_FIELDS) if flavor == 4 else RUSAGE_V2_FIELDS
                values = {name: int(getattr(data, name)) for name in fields}
                values['flavor'] = flavor
                return values
        raise OSError(ctypes.get_errno(), 'proc_pid_rusage unavailable for the authorized PID')

    def snapshot(self):
        start = time.monotonic()
        result = {'available': False, 'pid': self.pid, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'monotonic_start': start, 'scope': 'one authorized PID, read-only libproc bookkeeping'}
        try:
            first = self._usage()
            task = TaskInfo()
            size = ctypes.sizeof(task)
            copied = self.libproc.proc_pidinfo(self.pid, 4, 0, ctypes.byref(task), size)
            counters = self._usage()
            if first['start_abstime'] != counters['start_abstime']:
                raise ValueError('PID identity changed inside the snapshot')
            identity = counters['start_abstime']
            if not identity or self.start_identity is not None and self.start_identity != identity:
                raise ValueError('PID start identity changed or is unavailable')
            self.start_identity = identity
            result.update(available=True, identity={'pid': self.pid, 'start_abstime': identity}, rusage=counters,
                          mach_timebase=self.timebase)
            task_counters = ('faults', 'pageins', 'cow_faults', 'syscalls_mach', 'syscalls_unix', 'context_switches')
            if copied == size and all(getattr(task, key) >= 0 for key in task_counters):
                result['task_info'] = {'available': True, **{name: int(getattr(task, name)) for name, _ in task._fields_}}
            else:
                result['task_info'] = {'available': False,
                    'reason': 'proc_pidinfo unavailable, truncated, or signed32 counter overflow; no fault estimate substituted'}
            if self.timebase:
                scale = self.timebase['numer'] / self.timebase['denom'] / 1e9
                result['cpu_seconds'] = (counters['user_ticks'] + counters['system_ticks']) * scale
            else:
                result['cpu_seconds'] = None
            result['task_vm'] = (self_task_vm_snapshot(self.libsystem) if self.pid == os.getpid() else
                {'available': False, 'compressed_bytes': None, 'device_bytes': None,
                 'reason': 'external PID TASK_VM_INFO not requested; no privileged task attachment'})
        except (OSError, ValueError, AttributeError) as exc:
            result.update(available=False, error=f'{type(exc).__name__}: {exc}')
        result.update(monotonic_end=time.monotonic(), atomic_snapshot=False,
            physical_nand_read_bytes=None, physical_device_transfer_bytes=None, file_cache_hit_bytes=None,
            allocation_bytes=None, cpu_gpu_copy_bytes=None, weight_wait_seconds=None,
            unknown_scope='Counters do not isolate NAND/cache-hit bytes, weight waits, allocations, CPU/GPU copies, or GPU pinned allocations')
        result['snapshot_span_seconds'] = result['monotonic_end'] - start
        return result


def compare_process(before, after):
    if not before.get('available') or not after.get('available'):
        return {'available': False, 'reason': 'a process snapshot is unavailable'}
    if before['identity'] != after['identity']:
        return {'available': False, 'reason': 'PID identity changed'}
    if after['monotonic_start'] < before['monotonic_end']:
        return {'available': False, 'reason': 'process snapshot intervals overlap or are reversed'}
    keys = ('user_ticks', 'system_ticks', 'pageins', 'os_storage_read_bytes', 'os_storage_write_bytes')
    delta = {k: after['rusage'][k] - before['rusage'][k] for k in keys}
    if any(value < 0 for value in delta.values()):
        return {'available': False, 'reason': 'process cumulative counter reset or wrap'}
    task = None
    if before['task_info'].get('available') and after['task_info'].get('available'):
        task = {k: after['task_info'][k] - before['task_info'][k]
                for k in ('faults', 'pageins', 'cow_faults', 'context_switches')}
        if any(value < 0 for value in task.values()):
            return {'available': False, 'reason': 'process task counter reset or signed32 wrap'}
    seconds = (after.get('cpu_seconds') - before.get('cpu_seconds')
               if before.get('cpu_seconds') is not None and after.get('cpu_seconds') is not None else None)
    return {'available': True, 'rusage_counter_delta': delta, 'task_counter_delta': task,
            'cpu_core_seconds': seconds,
            'rss_change_bytes': after['rusage']['resident_bytes'] - before['rusage']['resident_bytes'],
            'footprint_change_bytes': after['rusage']['physical_footprint_bytes'] - before['rusage']['physical_footprint_bytes'],
            'wired_change_bytes': after['rusage']['wired_bytes'] - before['rusage']['wired_bytes'],
            'scope': 'OS-accounted process storage/pageins and CPU core-seconds; not NAND, pure weight wait or cache-hit bytes'}


def tool_capabilities():
    return {name: {'path': shutil.which(name), 'permission_scope': scope} for name, scope in (
        ('fs_usage', 'requires root on this Mac; not launched by observer; no sudo requested'),
        ('sample', 'optional owned-PID stack sampling; availability does not establish permission or zero overhead'),
        ('vmmap', 'optional owned-PID VM summary; not enabled in timed samples'),
        ('footprint', 'optional owned-PID footprint inspection; permission not assumed'))}


def sample_owned(process, system, *, block_device_observer=None):
    start = time.monotonic()
    stamp = datetime.now(timezone.utc).isoformat()
    global_values = system.snapshot()
    owned_values = process.snapshot()
    device_values = block_device_observer() if block_device_observer is not None else None
    return {'timestamp_utc': stamp, 'monotonic_start': start, 'monotonic_end': time.monotonic(),
            'available': global_values.get('available', False) and owned_values.get('available', False),
            'system': global_values, 'process': owned_values,
            'global_block_device': device_values,
            'scope': 'one bracketed sequential sample; non-atomic, no process suspension'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owned-pid', type=int, help='explicitly authorized target; default is observer itself')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('refusing to overwrite observations')
    from .system_memory import SystemMemoryObserver, compare_system
    observer = ProcessIOObserver(args.owned_pid if args.owned_pid is not None else os.getpid())
    system = SystemMemoryObserver()
    before = sample_owned(observer, system)
    after = sample_owned(observer, system)
    report = {'schema': 1, 'scope': 'read-only counter smoke; no model/GPU/weight IO generated',
              'before': before, 'after': after,
              'process_delta': compare_process(before['process'], after['process']),
              'global_delta': compare_system(before['system'], after['system']), 'tools': tool_capabilities(),
              'source_code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ('process_io.py', 'system_memory.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'process_available': after['process']['available'], 'system_available': after['system']['available'],
                      'process_delta_available': report['process_delta']['available']}))
    return 0 if after['process']['available'] and after['system']['available'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
