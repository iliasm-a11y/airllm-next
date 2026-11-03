"""Read-only macOS global memory observations, with explicit non-atomic scope."""
from __future__ import annotations

from datetime import datetime, timezone
import re
import subprocess
import time

MIB = 1024**2
GIB = 1024**3
PAGE_GAUGES = {
    'Pages free': 'free_bytes', 'Pages active': 'active_bytes',
    'Pages inactive': 'inactive_bytes', 'Pages speculative': 'speculative_bytes',
    'Pages wired down': 'wired_bytes', 'Pages purgeable': 'purgeable_bytes',
    'File-backed pages': 'file_backed_resident_bytes', 'Anonymous pages': 'anonymous_resident_bytes',
    'Pages stored in compressor': 'compressor_logical_page_bytes',
    'Pages occupied by compressor': 'compressor_physical_bytes',
}
COUNTERS = ('Translation faults', 'Pages copy-on-write', 'Pages zero filled', 'Pages reactivated',
            'Pages purged', 'Decompressions', 'Compressions', 'Pageins', 'Pageouts', 'Swapins', 'Swapouts')


def parse_vm_stat(text):
    page = re.search(r'page size of (\d+) bytes', text)
    if page is None or int(page[1]) < 1:
        raise ValueError('vm_stat omitted a valid page size')
    values = {}
    for line in text.splitlines()[1:]:
        label, separator, value = line.partition(':')
        if separator and re.fullmatch(r'\s*\d+\.\s*', value):
            values[label.strip().strip('"')] = int(value.strip().rstrip('.'))
    required = set(PAGE_GAUGES) | set(COUNTERS)
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError('vm_stat omitted required fields: ' + ', '.join(missing))
    page_size = int(page[1])
    gauges = {name: values[label] * page_size for label, name in PAGE_GAUGES.items()}
    gauges['available_reclaimable_estimate_bytes'] = sum(gauges[k] for k in (
        'free_bytes', 'inactive_bytes', 'speculative_bytes'))
    return {'page_size': page_size, 'gauges': gauges,
            'cumulative_counters': {k: values[k] for k in COUNTERS},
            'counter_units': 'raw vm_stat kernel counters; no counter-to-storage-byte conversion',
            'file_cache_hit_bytes': None,
            'available_estimate_scope': 'free+inactive+speculative pages; reclaimability estimate, not an allocation guarantee',
            'file_backed_scope': 'resident file-backed pages system-wide, including active mappings/libraries; not unused cache alone'}


def parse_sysctl(text):
    memory = re.search(r'^hw\.memsize:\s*(\d+)\s*$', text, re.M)
    pressure = re.search(r'^kern\.memorystatus_vm_pressure_level:\s*(\d+)\s*$', text, re.M)
    swap = re.search(r'^vm\.swapusage:.*?used\s*=\s*([\d.]+)([KMGTB])', text, re.M)
    boot = re.search(r'^kern\.boottime:.*?sec\s*=\s*(\d+).*?usec\s*=\s*(\d+)', text, re.M)
    if not all((memory, pressure, swap, boot)) or int(pressure[1]) not in (1, 2, 4):
        raise ValueError('sysctl omitted valid memory/pressure/swap/boot values')
    return {'physical_memory_bytes': int(memory[1]), 'pressure_level': int(pressure[1]),
            'swap_used_bytes': int(float(swap[1]) * {'B': 1, 'K': 1024, 'M': MIB, 'G': GIB, 'T': 1024**4}[swap[2]]),
            'boot_identity': [int(boot[1]), int(boot[2])]}


def _command(argv):
    return subprocess.check_output(argv, text=True, timeout=5)


class SystemMemoryObserver:
    def __init__(self, command=_command):
        self.command = command

    def snapshot(self):
        start = time.monotonic()
        result = {'available': False, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'monotonic_start': start, 'scope': 'global system counters; not attributed to any process'}
        try:
            result.update(parse_vm_stat(self.command(['/usr/bin/vm_stat'])))
            result.update(parse_sysctl(self.command(['/usr/sbin/sysctl', 'hw.memsize', 'vm.swapusage',
                'kern.memorystatus_vm_pressure_level', 'kern.boottime'])))
            result['available'] = True
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['monotonic_end'] = time.monotonic()
        result['snapshot_span_seconds'] = result['monotonic_end'] - start
        result['atomic_snapshot'] = False
        return result


def compare_system(before, after):
    if not before.get('available') or not after.get('available'):
        return {'available': False, 'reason': 'a system snapshot is unavailable'}
    if before['boot_identity'] != after['boot_identity']:
        return {'available': False, 'reason': 'system boot identity changed'}
    if after['monotonic_start'] < before['monotonic_end']:
        return {'available': False, 'reason': 'system snapshot intervals overlap or are reversed'}
    delta = {k: after['cumulative_counters'][k] - before['cumulative_counters'][k] for k in COUNTERS}
    if any(v < 0 for v in delta.values()):
        return {'available': False, 'reason': 'system counter reset or wrap'}
    return {'available': True, 'cumulative_counter_delta': delta,
            'gauge_delta_bytes': {k: after['gauges'][k] - before['gauges'][k] for k in before['gauges']},
            'global_swap_change_bytes': after['swap_used_bytes'] - before['swap_used_bytes'],
            'scope': 'system-wide changes; not native-process swap, NAND, or file-cache hit attribution'}


def partial_dense_cache_admission(samples, *, baseline_peak_rss_bytes, cache_bytes,
                                   inflight_bytes=0, device_increment_bytes=0,
                                   host_duplication_factor=2, max_rss_bytes=int(8.5 * GIB),
                                   headroom_bytes=2 * GIB, min_stable_seconds=5):
    """Conservative experimental preflight; existing live guards remain mandatory.

    Default factor2 charges both a new packed host copy and its still-resident
    mapped source. Use factor1 only after a measured direct-buffer path proves
    that no second payload is resident. This is a policy estimate, not a claim
    that RSS alone bounds compressed/driver/physical memory.
    """
    reasons = []
    for name, value in (('baseline_peak_rss_bytes', baseline_peak_rss_bytes), ('cache_bytes', cache_bytes),
                        ('inflight_bytes', inflight_bytes), ('device_increment_bytes', device_increment_bytes),
                        ('max_rss_bytes', max_rss_bytes), ('headroom_bytes', headroom_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    if type(host_duplication_factor) is not int or host_duplication_factor not in (1, 2):
        raise ValueError('host duplication factor must be1or2')
    if not 0 < max_rss_bytes <= int(8.5 * GIB):
        raise ValueError('the established RSS cap must not be increased')
    increment = cache_bytes * host_duplication_factor + inflight_bytes + device_increment_bytes
    if len(samples) < 2 or any(not s.get('available') for s in samples):
        return {'admitted': False, 'reasons': ['at least two valid global observations are required']}
    first, last = samples[0], samples[-1]
    for before, after in zip(samples, samples[1:]):
        delta = compare_system(before, after)
        if not delta['available']:
            reasons.append(delta['reason'])
    if last['monotonic_start'] - first['monotonic_end'] < min_stable_seconds:
        reasons.append('observed quiet interval is too short')
    if any(s['pressure_level'] != 1 for s in samples):
        reasons.append('memory pressure was not normal throughout the observations')
    if max(s['swap_used_bytes'] for s in samples) > first['swap_used_bytes']:
        reasons.append('global swap grew during the observations')
    if baseline_peak_rss_bytes + increment > max_rss_bytes:
        reasons.append('measured baseline RSS plus conservative new payload exceeds the existing RSS cap')
    available = min(s['gauges']['available_reclaimable_estimate_bytes'] for s in samples)
    if available < headroom_bytes + increment:
        reasons.append('estimated reclaimable memory cannot preserve the requested system headroom')
    return {'admitted': not reasons, 'reasons': reasons,
            'conservative_increment_bytes': increment, 'baseline_peak_rss_bytes': baseline_peak_rss_bytes,
            'host_duplication_factor': host_duplication_factor, 'minimum_available_estimate_bytes': available,
            'headroom_bytes': headroom_bytes, 'max_rss_bytes': max_rss_bytes,
            'compressor_physical_change_bytes': last['gauges']['compressor_physical_bytes'] - first['gauges']['compressor_physical_bytes'],
            'qualification': 'experimental preflight estimate only; retain RSS/swap/pressure guards during every run'}
