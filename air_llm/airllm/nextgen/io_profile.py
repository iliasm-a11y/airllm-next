"""Opt-in, bounded logical offset-I/O measurements; never device bandwidth.

Intervals identify serialized bytes, including repeated and overlapping reads.
The two interval unions have fixed metadata budgets. On saturation their exact
unique/reread totals become unavailable, with explicitly labelled lower bounds.
No tensor payload, token history, or unbounded per-tensor trace is retained.
"""
from bisect import bisect_left
import re
from threading import RLock


class _Intervals:
    def __init__(self, capacity):
        self.capacity = capacity
        self.files = {}
        self.count = 0
        self.unique = 0
        self.repeated = 0
        self.saturated = False

    def add(self, filename, start, size):
        if not size:
            return
        end = start + size
        entries = self.files.get(filename, [])
        i = bisect_left(entries, (start,))
        if i and entries[i - 1][1] >= start:
            i -= 1
        j = i
        overlap = 0
        lo, hi = start, end
        while j < len(entries) and entries[j][0] <= hi:
            a, b = entries[j]
            overlap += max(0, min(end, b) - max(start, a))
            lo, hi = min(lo, a), max(hi, b)
            j += 1
        self.repeated += overlap
        if self.saturated:
            return
        count = self.count + 1 - (j - i)
        if count > self.capacity:
            self.saturated = True
            return
        entries[i:j] = [(lo, hi)]
        self.files[filename] = entries
        self.count = count
        self.unique += size - overlap

    def snapshot(self):
        return {'unique_bytes': None if self.saturated else self.unique,
                'reread_bytes': None if self.saturated else self.repeated,
                'unique_bytes_lower_bound': self.unique,
                'reread_bytes_lower_bound': self.repeated,
                'interval_count': self.count, 'saturated': self.saturated}


def _tag(key, expert):
    layer = re.search(r'(?:^|\.)(?:layers|blocks|h)\.(\d{1,12})(?:\.|$)', key)
    number = re.search(r'(?:^|\.)experts\.(\d{1,12})(?:\.|$)', key)
    layer = layer.group(1) if layer else None
    expert = str(expert) if expert is not None else number.group(1) if number else None
    if expert is None and 'shared_expert' in key:
        expert = 'shared'
    return f'layer={layer if layer is not None else "other"}/expert={expert if expert is not None else "other"}'


class IOProfiler:
    def __init__(self, *, max_intervals=8192, max_groups=512, physical_counter=None, physical_counter_source=None):
        for name, value in (('max_intervals', max_intervals), ('max_groups', max_groups)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.lock = RLock()
        self.max_intervals = max_intervals
        self.max_groups = max_groups
        self.requested = _Intervals(max_intervals)
        self.read = _Intervals(max_intervals)
        self.groups = {}
        self.overflow_groups = False
        self.requested_bytes = self.read_bytes = self.request_count = self.read_calls = 0
        self.failed_read_calls = self.unattributed_read_bytes = 0
        self.io_seconds = 0.
        self.modes = {'pread': {'read_calls': 0, 'read_bytes': 0, 'io_seconds': 0.},
                      'mmap': {'read_calls': 0, 'read_bytes': 0, 'io_seconds': 0.},
                      'pread_nocache': {'read_calls': 0, 'read_bytes': 0, 'io_seconds': 0.}}
        self.physical_counter = physical_counter
        self.physical_counter_source = physical_counter_source
        self.physical_start = physical_counter() if physical_counter else None

    def requests(self, regions):
        """Register validated attempted selections and return bounded attribution.

        Regions contain (filename, offset, size, key, expert). Only short derived
        layer/expert labels survive this call. Duplicate selections count twice.
        """
        tagged = []
        with self.lock:
            for filename, offset, size, key, expert in regions:
                tag = _tag(key, expert)
                if tag not in self.groups:
                    # Reserve one fixed overflow bucket within the group budget.
                    if len(self.groups) >= self.max_groups - 1:
                        tag = 'other_groups'
                        self.overflow_groups = True
                    if tag not in self.groups:
                        self.groups[tag] = {'request_count': 0, 'requested_bytes': 0,
                                            'read_calls': 0, 'read_bytes': 0}
                group = self.groups[tag]
                group['request_count'] += 1
                group['requested_bytes'] += size
                self.request_count += 1
                self.requested_bytes += size
                self.requested.add(filename, offset, size)
                tagged.append((offset, offset + size, tag))
        return tagged

    def record_read(self, filename, offset, size, seconds, mode, regions=(), failed=False):
        with self.lock:
            self.read_calls += 1
            self.failed_read_calls += int(failed)
            self.read_bytes += size
            self.io_seconds += seconds
            self.read.add(filename, offset, size)
            by_mode = self.modes[mode]
            by_mode['read_calls'] += 1
            by_mode['read_bytes'] += size
            by_mode['io_seconds'] += seconds
            # Coalescing/duplicate selections must not multiply attributed bytes.
            # Regions are already in offset order. Distinct labels that overlap
            # share attribution with the first label, keeping the total exact.
            covered = offset
            attributed = 0
            touched = set()
            for start, end, tag in sorted(regions):
                start, end = max(offset, start, covered), min(offset + size, end)
                if end > start:
                    self.groups[tag]['read_bytes'] += end - start
                    attributed += end - start
                    covered = end
                    touched.add(tag)
            for tag in touched:
                self.groups[tag]['read_calls'] += 1
            self.unattributed_read_bytes += size - attributed

    def snapshot(self):
        with self.lock:
            physical_end = self.physical_counter() if self.physical_counter else None
            physical = (physical_end - self.physical_start
                        if physical_end is not None and self.physical_start is not None
                        and physical_end >= self.physical_start else None)
            return {'schema_version': 1,
                    'scope': 'serialized logical bytes; OS page-cache service is included; not SSD bandwidth',
                    'requested_bytes': self.requested_bytes,
                    'requested_semantics': 'validated selections attempted, including duplicates and failed transfers',
                    'read_bytes': self.read_bytes,
                    'read_semantics': 'bytes copied by completed or partially completed pread/mmap/pread_nocache calls; gaps included',
                    'request_count': self.request_count, 'read_calls': self.read_calls,
                    'failed_read_calls': self.failed_read_calls, 'io_seconds': self.io_seconds,
                    'requested_intervals': self.requested.snapshot(),
                    'read_intervals': self.read.snapshot(),
                    'modes': {k: dict(v) for k, v in self.modes.items()},
                    'by_layer_expert': {k: dict(v) for k, v in sorted(self.groups.items())},
                    'unattributed_read_bytes': self.unattributed_read_bytes,
                    'group_attribution': 'overlaps assigned once to first sorted region; gaps unattributed',
                    'group_limit_reached': self.overflow_groups,
                    'metadata_limits': {'max_intervals_per_stream': self.max_intervals,
                                        'max_layer_expert_groups': self.max_groups},
                    'physical_process_read_bytes': physical,
                    'physical_counter_source': self.physical_counter_source,
                    'cache_policy': 'pread/mmap use normal OS cache policy; pread_nocache requests macOS F_NOCACHE per separately opened fd; no cache purge',
                    'physical_scope': 'process-wide OS storage counter; includes unrelated reads/readahead; unavailable is null',
                    'physical_ssd_bandwidth_certified': False}
