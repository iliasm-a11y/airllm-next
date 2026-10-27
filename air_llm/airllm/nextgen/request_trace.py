"""Bounded opt-in monotonic wall-time tracing for streamed requests.

A span around MLX submission measures host wall time, not GPU kernel execution.
Use a separate synchronization span for host waits; obtaining GPU utilization or
pure kernel time requires independent Metal instrumentation. Category unions
may overlap and must not be added. No profiling synchronization is introduced.
"""
from threading import Lock, get_ident, local
from time import perf_counter_ns


class _NoSpan:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_NO_SPAN = _NoSpan()


def _union_ns(intervals):
    """Length of the union of half-open intervals, independent of their order."""
    intervals = sorted((start, end) for start, end in intervals if end > start)
    total = 0
    if not intervals:
        return total
    start, end = intervals[0]
    for following_start, following_end in intervals[1:]:
        if following_start > end:
            total += end - start
            start, end = following_start, following_end
        else:
            end = max(end, following_end)
    return total + end - start


class _Span:
    def __init__(self, trace, name, category, phase, metadata):
        self.trace = trace
        self.name = name
        self.category = category
        self.phase = phase
        self.metadata = metadata
        self.record = None

    def __enter__(self):
        trace = self.trace
        stack = trace._stack()
        parent = stack[-1] if stack else None
        phase = self.phase if self.phase is not None else (parent.phase if parent else None)
        self.phase = phase
        start = trace._now()
        self.record = trace._append(dict(
            kind='span', name=self.name, category=self.category, phase=phase,
            start_ns=start, end_ns=None, thread_id=get_ident(),
            parent_id=parent.record['id'] if parent and parent.record else None,
            metadata=dict(self.metadata), outcome=None))
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        trace = self.trace
        end = trace._now()
        stack = trace._stack()
        if not stack or stack[-1] is not self:
            raise RuntimeError('Trace spans must exit in stack order on their entering thread')
        stack.pop()
        if self.record is not None:
            with trace._lock:
                self.record['end_ns'] = max(self.record['start_ns'], end)
                self.record['outcome'] = 'error' if exc_type else 'ok'
                if exc_type:
                    self.record['error_type'] = exc_type.__name__
        return False


class RequestTrace:
    """Collect bounded diagnostic spans without changing model synchronization.

    ``span(name, category='engine', phase=None, **metadata)`` is a context
    manager. Nested spans inherit the enclosing phase. ``event`` records an
    instant, ``count`` accumulates a numeric counter, and ``export`` returns a
    JSON-compatible snapshot. Disabled traces never consult the clock.

    If the record bound is reached, collection continues counting omissions but
    the export is explicitly incomplete; partial timing is never called total.
    All timestamps are offsets from ``origin_monotonic_ns``. ``exclusive_ns``
    subtracts the union of nested direct children on that thread. Across threads
    even exclusive sums may exceed elapsed time; only interval unions express
    observed wall-time coverage.
    """
    def __init__(self, enabled=True, max_records=100_000, clock=perf_counter_ns):
        if type(max_records) is not int or max_records < 1:
            raise ValueError('max_records must be a positive integer')
        self.enabled = bool(enabled)
        self.max_records = max_records
        self._clock = clock
        self.origin_monotonic_ns = clock() if self.enabled else None
        self._records = []
        self._counters = {}
        self._dropped_records = 0
        self._next_id = 0
        self._local = local()
        self._lock = Lock()

    def _now(self):
        return self._clock() - self.origin_monotonic_ns

    def _stack(self):
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def _append(self, record):
        with self._lock:
            if len(self._records) >= self.max_records:
                self._dropped_records += 1
                return None
            record['id'] = self._next_id
            self._next_id += 1
            self._records.append(record)
            return record

    def span(self, name, category='engine', phase=None, **metadata):
        if not self.enabled:
            return _NO_SPAN
        return _Span(self, name, category, phase, metadata)

    def event(self, name, category='engine', phase=None, **metadata):
        if not self.enabled:
            return
        stack = self._stack()
        parent = stack[-1] if stack else None
        self._append(dict(
            kind='event', name=name, category=category,
            phase=phase if phase is not None else (parent.phase if parent else None),
            at_ns=self._now(), thread_id=get_ident(),
            parent_id=parent.record['id'] if parent and parent.record else None,
            metadata=dict(metadata)))

    def count(self, name, amount=1):
        if not self.enabled:
            return
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            raise TypeError('Counter amounts must be numeric')
        from math import isfinite
        if not isfinite(amount):
            raise ValueError('Counter amounts must be finite')
        with self._lock:
            value = self._counters.get(name, 0) + amount
            if not isfinite(value):
                raise ValueError('Counter total must be finite')
            self._counters[name] = value

    def export(self):
        with self._lock:
            records = [dict(record, metadata=dict(record['metadata'])) for record in self._records]
            counters = dict(self._counters)
            dropped = self._dropped_records
        spans = [record for record in records if record['kind'] == 'span' and record['end_ns'] is not None]
        open_spans = sum(record['kind'] == 'span' and record['end_ns'] is None for record in records)
        children = {}
        for record in spans:
            children.setdefault(record['parent_id'], []).append(record)
        categories = {}
        phases = {}
        for record in spans:
            start, end = record['start_ns'], record['end_ns']
            record['inclusive_ns'] = end - start
            nested = [(max(start, child['start_ns']), min(end, child['end_ns']))
                      for child in children.get(record['id'], [])]
            record['exclusive_ns'] = max(0, end - start - _union_ns(nested))
            for groups, key in ((categories, record['category']), (phases, record['phase'])):
                if key is None:
                    key = 'unspecified'
                group = groups.setdefault(key, dict(spans=0, inclusive_sum_ns=0, exclusive_sum_ns=0,
                                                    intervals=[]))
                group['spans'] += 1
                group['inclusive_sum_ns'] += record['inclusive_ns']
                group['exclusive_sum_ns'] += record['exclusive_ns']
                group['intervals'].append((start, end))
        for groups in (categories, phases):
            for group in groups.values():
                group['union_ns'] = _union_ns(group.pop('intervals'))
        intervals = [(span['start_ns'], span['end_ns']) for span in spans]
        first = min((start for start, end in intervals), default=0)
        last = max((end for start, end in intervals), default=0)
        return dict(
            schema_version=1, enabled=self.enabled,
            complete=self.enabled and dropped == 0 and open_spans == 0,
            origin_monotonic_ns=self.origin_monotonic_ns,
            clock='perf_counter_ns (monotonic offsets)',
            max_records=self.max_records, dropped_records=dropped, open_spans=open_spans,
            records=records, counters=counters,
            summary=dict(observed_span_union_ns=_union_ns(intervals),
                         first_span_start_ns=first, last_span_end_ns=last,
                         observed_span_extent_ns=last-first,
                         categories=categories, phases=phases),
            attribution=dict(
                host_wall_time=True, pure_cpu_time=False, pure_gpu_time=False,
                gpu_utilization=False, storage_device_time=False,
                note='Submission and blocking waits are host wall times. Category and phase '
                     'unions may overlap; do not add them. Missing categories are uninstrumented, '
                     'not proven free. No synchronization is added by tracing.'))
