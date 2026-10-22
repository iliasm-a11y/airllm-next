"""Byte-budgeted, scan-resistant two-segment cache and bounded I/O scheduler."""
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import RLock


def tensor_bytes(state):
    return sum(t.numel() * t.element_size() if hasattr(t, 'numel') else t.nbytes
               for t in state.values())


class AdaptiveCache:
    """Promote reused entries; don't let a scan evict the entire hot expert set.

    Budgets count materialized tensor storage, excluding Python object overhead.
    This is a host-weight budget, not a bound on model activations or the KV cache.
    """
    def __init__(self, capacity):
        self.capacity = max(0, int(capacity))
        self.probation = OrderedDict()
        self.protected = OrderedDict()
        self.sizes = {}
        self.bytes = 0
        self.peak_bytes = 0
        self.lock = RLock()
        self.stats = defaultdict(int)

    def get(self, key):
        with self.lock:
            if key in self.protected:
                self.protected.move_to_end(key)
                self.stats['hits'] += 1
                return self.protected[key]
            if key in self.probation:
                value = self.probation.pop(key)
                self.protected[key] = value
                self.stats['hits'] += 1
                return value
            self.stats['misses'] += 1
            return None

    def contains(self, key):
        with self.lock:
            return key in self.sizes

    def put(self, key, value):
        size = tensor_bytes(value)
        with self.lock:
            if key in self.sizes or size > self.capacity or self.capacity == 0:
                return False
            while self.bytes + size > self.capacity:
                source = self.probation if self.probation else self.protected
                victim, _ = source.popitem(last=False)
                self.bytes -= self.sizes.pop(victim)
                self.stats['evictions'] += 1
            self.probation[key] = value
            self.sizes[key] = size
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            return True

    def resize(self, capacity):
        with self.lock:
            self.capacity = max(0, int(capacity))
            while self.bytes > self.capacity:
                source = self.probation if self.probation else self.protected
                key, _ = source.popitem(last=False)
                self.bytes -= self.sizes.pop(key)
                self.stats['evictions'] += 1

    def clear(self):
        with self.lock:
            self.probation.clear()
            self.protected.clear()
            self.sizes.clear()
            self.bytes = 0


class BoundedPrefetch:
    """One sequential I/O worker, deduplicated futures, hard byte reservations.

    Completed speculative results still occupy reservations until consumed or
    discarded. This prevents futures from retaining unbounded host tensors.
    """
    def __init__(self, budget, max_tasks=8):
        self.budget = max(0, int(budget))
        self.max_tasks = max_tasks
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='airllm-io')
        self.pending = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.closed = False
        self.stats = defaultdict(int)

    def submit(self, key, size, loader):
        if self.closed or key in self.pending or size > self.budget or size <= 0:
            return False
        # Drop only completed speculation; do not abandon a running read and lose
        # track of its memory reservation.
        for old in list(self.pending):
            if len(self.pending) < self.max_tasks and self.bytes + size <= self.budget:
                break
            future, amount = self.pending[old]
            if future.done() or future.cancel():
                self.pending.pop(old)
                self.bytes -= amount
                self.stats['discarded'] += 1
        if len(self.pending) >= self.max_tasks or self.bytes + size > self.budget:
            self.stats['rejected'] += 1
            return False
        self.pending[key] = (self.executor.submit(loader), size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.stats['submitted'] += 1
        return True

    def take(self, key):
        item = self.pending.pop(key, None)
        if item is None:
            return None
        future, size = item
        try:
            result = future.result()
            if tensor_bytes(result) > size:
                raise RuntimeError('Prefetch loader exceeded its reserved tensor bytes')
            self.stats['consumed'] += 1
            return result
        finally:
            self.bytes -= size

    def drain(self):
        for future, _ in self.pending.values():
            future.cancel()
        error = None
        for future, _ in self.pending.values():
            if not future.cancelled():
                try:
                    future.result()
                except BaseException as exc:
                    if error is None:
                        error = exc
        self.pending.clear()
        self.bytes = 0
        if error is not None:
            raise error

    def close(self):
        if self.closed:
            return
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
