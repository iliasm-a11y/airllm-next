"""Opt-in platform controls and OS-accounted process storage read evidence.

These counters cannot certify NAND traffic or identify a particular device.
F_NOCACHE is an fd policy, not a privileged cache purge or a cold-cache claim.
"""
from functools import lru_cache
import os
from pathlib import Path
import sys


def physical_read_counter_info():
    if sys.platform == 'darwin':
        source = 'libproc.proc_pid_rusage(RUSAGE_INFO_V2).ri_diskio_bytesread'
    elif sys.platform.startswith('linux'):
        source = '/proc/self/io:read_bytes'
    else:
        source = None
    return {'source': source,
            'scope': 'process-wide OS-accounted storage reads; includes unrelated reads and readahead; not per-file or NAND bytes',
            'unavailable': None, 'physical_ssd_bandwidth_certified': False}


@lru_cache(maxsize=1)
def _darwin_reader():
    # Public ABI in Apple's sys/resource.h and libproc.h (RUSAGE_INFO_V2=2).
    # Keep ctypes/library loading off the disabled-profiler fast path.
    import ctypes
    from ctypes.util import find_library

    class RUsageInfoV2(ctypes.Structure):
        _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in (
                'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups',
                'ri_interrupt_wkups', 'ri_pageins', 'ri_wired_size',
                'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime',
                'ri_proc_exit_abstime', 'ri_child_user_time', 'ri_child_system_time',
                'ri_child_pkg_idle_wkups', 'ri_child_interrupt_wkups',
                'ri_child_pageins', 'ri_child_elapsed_abstime',
                'ri_diskio_bytesread', 'ri_diskio_byteswritten')]

    library = ctypes.CDLL(find_library('proc') or '/usr/lib/libproc.dylib', use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    function.restype = ctypes.c_int

    def read():
        result = RUsageInfoV2()
        if function(os.getpid(), 2, ctypes.byref(result)) != 0:
            return None
        return int(result.ri_diskio_bytesread)
    return read


def physical_read_bytes():
    """Return cumulative OS-accounted storage-read bytes, or None if unavailable.

    A zero delta is evidence only about this process counter in that interval.
    It does not establish that every byte came from cache, or that no device read
    occurred elsewhere. Apple may return zero when task I/O stats are absent.
    """
    try:
        if sys.platform == 'darwin':
            return _darwin_reader()()
        if sys.platform.startswith('linux'):
            lines = Path('/proc/self/io').read_text().splitlines()
            return int(next(s.split(':', 1)[1] for s in lines if s.startswith('read_bytes:')))
    except (OSError, AttributeError, StopIteration, ValueError):
        pass
    return None


def set_fd_nocache(fd):
    """Request macOS F_NOCACHE for an exclusively opened descriptor; fail closed."""
    if sys.platform != 'darwin':
        raise NotImplementedError('pread_nocache requires macOS F_NOCACHE; no portable fallback is implied')
    import fcntl
    # Python does not expose F_NOCACHE on every macOS build. The public Darwin
    # sys/fcntl.h ABI defines 48 as turning data caching off/on for this fd.
    fcntl.fcntl(fd, getattr(fcntl, 'F_NOCACHE', 48), 1)
