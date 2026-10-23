"""A byte-bounded tier for exact, still-encoded source tensor selections.

This is a transport cache: FP8 bytes and their block scales are retained before
decode, so it neither requantizes weights nor changes numerical semantics. Its
budget is a partition of the existing weight-cache budget, not extra headroom.
Arrays may be views into coalesced reads; count their complete unique backing
buffers rather than only the visible array bytes. Admission makes no large copy.
"""
from collections import Counter, OrderedDict
from threading import RLock

import numpy as np


def backing_bytes(arrays):
    """Return unique retained buffer bytes, including storage outside a view."""
    owners = {}
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError('Compact weight entries must contain NumPy arrays')
        owner = array
        seen = set()
        while id(owner) not in seen:
            seen.add(id(owner))
            if isinstance(owner, np.ndarray) and owner.base is not None:
                owner = owner.base
            elif isinstance(owner, memoryview):
                owner = owner.obj
            else:
                break
        try:
            nbytes = memoryview(owner).nbytes
        except TypeError:
            # An external array owner without the buffer protocol cannot be
            # safely charged by visible shape alone. Do not retain such views.
            raise ValueError('Cannot determine compact-array backing allocation')
        owners[id(owner)] = nbytes
    return sum(owners.values())


class CompactWeightCache:
    """LRU source-array cache, independent of decoded admission policy.

    Entries are internal read results only. Exposing them to mutable callers is
    deliberately unsupported. Eviction releases this tier's references; an
    in-flight decoder still has its separately reserved I/O workspace.
    """
    def __init__(self, capacity, max_entries=4096):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Compact cache capacity must be a nonnegative integer')
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('Compact cache entry limit must be a positive integer')
        self.capacity = capacity
        self.max_entries = max_entries
        self.data = OrderedDict()
        self.sizes = {}
        self.bytes = 0
        self.peak_bytes = 0
        self.stats = Counter()
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            value = self.data.get(key)
            if value is None:
                self.stats['misses'] += 1
                return None
            self.data.move_to_end(key)
            self.stats['hits'] += 1
            self.stats['source_bytes_reused'] += sum(a.nbytes for a in value)
            return value

    def put(self, key, arrays):
        arrays = tuple(arrays)
        # Disabled tier does not inspect ownership, mutate arrays or retain data.
        if self.capacity == 0:
            return False
        nbytes = backing_bytes(arrays)
        with self.lock:
            if key in self.data:
                return False
            if nbytes == 0 or nbytes > self.capacity:
                self.stats['rejected'] += 1
                return False
            while self.bytes + nbytes > self.capacity or len(self.data) >= self.max_entries:
                old, _ = self.data.popitem(last=False)
                self.bytes -= self.sizes.pop(old)
                self.stats['evictions'] += 1
            for array in arrays:
                array.flags.writeable = False
            self.data[key] = arrays
            self.sizes[key] = nbytes
            self.bytes += nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            return True

    def resize(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Compact cache capacity must be a nonnegative integer')
        with self.lock:
            self.capacity = capacity
            while self.bytes > capacity:
                key, _ = self.data.popitem(last=False)
                self.bytes -= self.sizes.pop(key)
                self.stats['evictions'] += 1

    def clear(self):
        with self.lock:
            self.data.clear()
            self.sizes.clear()
            self.bytes = 0
