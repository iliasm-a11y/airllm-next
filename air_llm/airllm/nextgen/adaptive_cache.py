"""Optional, scan-resistant cache using observed reuse and reload cost.

All tensor/expert keys share one byte budget, so hot experts in one layer can
occupy space formerly used by another layer without fixed per-layer quotas.
Demand counts decay with a half-life measured in requests. Admission under
pressure requires repeat demand and more estimated avoided reload work than
the *sum* lost by evicting the selected residents. An expensive first-touch
scan therefore cannot evict a resident just because its load was slow.

This is a heuristic, not a prediction of throughput. Load-cost observations are
EWMA timings supplied by the caller; unknown costs use the cache's observed
mean, or a neutral equal-cost rank before any timings exist. No bandwidth or
hardware speed is assumed. Prefetch loads record cost but do not count as use.

Resident entries, nonresident observations and replay traces each have explicit
count limits. The byte budget covers retained tensor payloads; Python metadata,
the caller's live references and the backend's allocator remain outside it.
NumPy backing buffers are charged conservatively, without copying weights.
"""
from collections import Counter, OrderedDict, deque
from math import isfinite
from threading import RLock

import numpy as np

from .compact_cache import backing_bytes
from .quantization import PackedMatrix


def retained_size(value):
    """Charge complete NumPy owners and backend-reported tensor payloads.

    Owners shared within an entry count once. Sharing between entries may be
    charged twice, which is conservative. MLX values use the existing evaluated
    tensor ``nbytes`` contract; this does not measure Metal allocator residency.
    """
    arrays = []
    backend_bytes = 0

    def visit(item):
        nonlocal backend_bytes
        if isinstance(item, np.ndarray):
            arrays.append(item)
        elif isinstance(item, PackedMatrix):
            for component in (item.weight, item.scales, item.biases):
                visit(component)
        elif isinstance(item, dict):
            for component in item.values():
                visit(component)
        elif isinstance(item, (tuple, list)):
            for component in item:
                visit(component)
        else:
            size = item.nbytes
            if isinstance(size, bool) or int(size) != size or size < 0:
                raise ValueError('Tensor byte size must be a nonnegative integer')
            backend_bytes += int(size)

    visit(value)
    return backend_bytes + backing_bytes(arrays)


class AdaptiveWeightCache:
    """WeightCache-compatible optional policy with bounded observation history.

    ``max_metadata_entries`` bounds *nonresident* observations; resident metadata
    is additionally bounded by ``max_entries``. Frequency/cost dictionaries can
    therefore hold at most their sum. A full cache still has room to observe a
    new hot set. Trace deques each retain at most 4096 requests or load events.
    """
    def __init__(self, capacity, policy='adaptive', *, max_metadata_entries=4096,
                 decay_interval=1024, max_entries=4096):
        self._validate_capacity(capacity)
        if policy != 'adaptive':
            raise ValueError('AdaptiveWeightCache requires adaptive policy')
        for name, value in (('metadata limit', max_metadata_entries),
                            ('decay interval', decay_interval),
                            ('entry limit', max_entries)):
            if type(value) is not int or value < 1:
                raise ValueError(f'Adaptive cache {name} must be a positive integer')
        self.capacity = capacity
        self.policy = policy
        self.max_metadata_entries = max_metadata_entries
        self.max_entries = max_entries
        self.decay_interval = decay_interval
        self.data = OrderedDict()
        self.sizes = {}
        self.freq = Counter()
        self.costs = {}
        self._last_seen = {}
        self._ghosts = OrderedDict()
        self._clock = 0
        self._mean_cost = None
        self.bytes = 0
        self.peak_bytes = 0
        self.stats = Counter()
        self.lock = RLock()
        self.trace = deque(maxlen=4096)
        self.cost_trace = deque(maxlen=4096)
        self.load_trace = deque(maxlen=4096)

    @staticmethod
    def _validate_capacity(capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Adaptive cache capacity must be a nonnegative integer')

    def _trim_history(self):
        while len(self._ghosts) > self.max_metadata_entries:
            key, _ = self._ghosts.popitem(last=False)
            self.freq.pop(key, None)
            self.costs.pop(key, None)
            self._last_seen.pop(key, None)
            self.stats['history_evictions'] += 1

    def _observe_key(self, key):
        if key not in self._last_seen:
            self._last_seen[key] = self._clock
            self.freq[key] = 0.
        if key not in self.data:
            self._ghosts[key] = None
            self._ghosts.move_to_end(key)
        self._trim_history()

    def _decayed_frequency(self, key):
        age = self._clock - self._last_seen.get(key, self._clock)
        # Avoid enormous integer-to-float conversion after extremely long runs.
        if age >= 1024 * self.decay_interval:
            return 0.
        return self.freq.get(key, 0.) * 2. ** (-age / self.decay_interval)

    def record_load(self, key, seconds):
        """Record successful demand/prefetch service without inventing a use."""
        seconds = float(seconds)
        if not isfinite(seconds) or seconds < 0:
            raise ValueError('Load time must be finite and nonnegative')
        with self.lock:
            self._observe_key(key)
            previous = self.costs.get(key)
            self.costs[key] = seconds if previous is None else previous * .75 + seconds * .25
            self._mean_cost = (seconds if self._mean_cost is None
                               else self._mean_cost * .75 + seconds * .25)
            self.load_trace.append((key, seconds))
            self.stats['load_observations'] += 1

    def get(self, key, size_hint=None):
        with self.lock:
            self._clock += 1
            self._observe_key(key)
            self.freq[key] = self._decayed_frequency(key) + 1.
            self._last_seen[key] = self._clock
            if size_hint is not None:
                self.trace.append((key, int(size_hint)))
                self.cost_trace.append((key, int(size_hint), self.costs.get(key)))
            if key not in self.data:
                self.stats['misses'] += 1
                return None
            self.stats['hits'] += 1
            self.data.move_to_end(key)
            return self.data[key]

    def contains(self, key):
        with self.lock:
            return key in self.data

    def _benefit(self, key):
        # One observation alone supplies no evidence of future reuse. Decay lets
        # a previously hot resident yield space after a workload phase change.
        reuse = max(0., self._decayed_frequency(key) - 1.)
        fallback = self._mean_cost if self._mean_cost else 1.
        return reuse * self.costs.get(key, fallback)

    def _ranked_residents(self):
        # OrderedDict order gives deterministic LRU tie-breaking without ever
        # comparing potentially heterogeneous caller keys.
        return sorted(self.data, key=lambda key: self._benefit(key) / max(1, self.sizes[key]))

    def _remove(self, key):
        self.data.pop(key)
        self.bytes -= self.sizes.pop(key)
        self._ghosts[key] = None
        self._ghosts.move_to_end(key)
        self.stats['evictions'] += 1
        self._trim_history()

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                return False
            if self.capacity == 0:
                self.stats['rejected'] += 1
                return False
            nbytes = retained_size(value)
            if nbytes == 0 or nbytes > self.capacity:
                self.stats['rejected'] += 1
                return False
            self._observe_key(key)
            victims = []
            freed = 0
            if self.bytes + nbytes > self.capacity or len(self.data) >= self.max_entries:
                candidate_benefit = self._benefit(key)
                if candidate_benefit <= 0:
                    self.stats['rejected'] += 1
                    self.stats['cold_rejected'] += 1
                    return False
                for victim in self._ranked_residents():
                    victims.append(victim)
                    freed += self.sizes[victim]
                    if (self.bytes + nbytes - freed <= self.capacity and
                            len(self.data) + 1 - len(victims) <= self.max_entries):
                        break
                lost_benefit = sum(self._benefit(victim) for victim in victims)
                if candidate_benefit <= lost_benefit:
                    self.stats['rejected'] += 1
                    return False
            # Protect the candidate's observation while removed residents enter
            # the bounded ghost queue. It becomes a resident below.
            self._ghosts.pop(key, None)
            for victim in victims:
                self._remove(victim)
            self.data[key] = value
            self.sizes[key] = nbytes
            self.bytes += nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.stats['admitted'] += 1
            if key not in self.costs:
                self.stats['admitted_without_cost'] += 1
            return True

    def resize(self, capacity):
        self._validate_capacity(capacity)
        with self.lock:
            self.capacity = capacity
            # Admission does not apply to a forced budget reduction.
            for key in self._ranked_residents():
                if self.bytes <= self.capacity:
                    break
                self._remove(key)

    def clear(self):
        """Release values and observations; retain cumulative stats and peak."""
        with self.lock:
            self.data.clear()
            self.sizes.clear()
            self.freq.clear()
            self.costs.clear()
            self._last_seen.clear()
            self._ghosts.clear()
            self.trace.clear()
            self.cost_trace.clear()
            self.load_trace.clear()
            self.bytes = 0
            self._clock = 0
            self._mean_cost = None
