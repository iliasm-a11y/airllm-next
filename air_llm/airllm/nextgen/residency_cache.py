"""Opt-in class-aware residency inside the existing decoded-weight budget.

These are ablation policies, not throughput guarantees. ``shared`` and
``expert`` prefer one class under pressure; lower-priority entries cannot
displace the preferred class. ``hybrid`` uses measured reuse/reload benefit
with a shared-first tie break. ``batch-aware`` temporarily prioritizes the
*actual* expert union supplied by the router, then shared weights. Its batch
metadata contains keys/counts only, never extra weight buffers.

All modes use one freely shared byte budget. Output tiles remain shared
weights, although their counters are separate from attention/router/norm.
"""
from collections import Counter
from numbers import Integral
import re

from .adaptive_cache import AdaptiveWeightCache, retained_size


RESIDENCY_POLICIES = ('shared', 'expert', 'hybrid', 'batch-aware')


def cache_key_class(key):
    """Classify real reader identities without treating every tuple as expert.

    Expert references may be ``(tensor_name, expert_id)``, native identities
    ``tensor_name#expert_id``, or individually named ``.experts.<id>.`` tensors.
    Output tile tuples contain row bounds; these are shared, not expert ids.
    Unknown caller keys are conservatively shared.
    """
    if isinstance(key, tuple):
        if key and key[0] == 'output':
            return 'output'
        if (len(key) == 2 and isinstance(key[0], str) and
                isinstance(key[1], Integral) and not isinstance(key[1], bool) and key[1] >= 0):
            return 'expert'
        return 'shared'
    if isinstance(key, str):
        if re.search(r'#\d+$', key) or re.search(r'\.experts\.\d+(?:\.|$)', key):
            return 'expert'
    return 'shared'


class ResidencyWeightCache(AdaptiveWeightCache):
    """Class preference and finite batch protection, with bounded metadata."""
    def __init__(self, capacity, policy='hybrid', **options):
        if policy not in RESIDENCY_POLICIES:
            raise ValueError('Unknown residency cache policy')
        super().__init__(capacity, **options)
        self.policy = policy
        self.batch_counts = Counter()

    def set_batch(self, keys):
        """Declare exact routed identities; duplicates are observed multiplicity.

        A bounded declaration is built before replacing the old declaration.
        Rejecting an oversized declaration cannot alter the cache or reserve
        memory. The caller must clear it after the routed group, even on error.
        """
        counts = Counter()
        for key in keys:
            if cache_key_class(key) != 'expert':
                raise ValueError('A batch declaration must contain expert identities')
            counts[key] += 1
            if len(counts) > self.max_metadata_entries:
                raise ValueError('Batch expert union exceeds cache metadata limit')
        with self.lock:
            self.batch_counts = counts
            self.stats['batch_declarations'] += 1
            self.stats['batch_union_peak'] = max(self.stats['batch_union_peak'], len(counts))
            self.stats['batch_references'] += sum(counts.values())

    def clear_batch(self):
        with self.lock:
            self.batch_counts.clear()

    def _preference(self, key):
        shared = cache_key_class(key) != 'expert'
        if self.policy == 'shared':
            return int(shared)
        if self.policy == 'expert':
            return int(not shared)
        if self.policy == 'batch-aware':
            return 2 if key in self.batch_counts else int(shared)
        return 0

    def _benefit(self, key):
        benefit = super()._benefit(key)
        if self.policy == 'batch-aware' and key in self.batch_counts:
            # Multiplicity is observed routing, not a fabricated future demand.
            benefit *= self.batch_counts[key]
        return benefit

    def _ranked_residents(self):
        # Shared tie-break in hybrid is subordinate to measured benefit/byte.
        return sorted(self.data, key=lambda key: (
            self._preference(key), self._benefit(key) / max(1, self.sizes[key]),
            int(self.policy == 'hybrid' and cache_key_class(key) != 'expert')))

    def get(self, key, size_hint=None):
        with self.lock:
            value = super().get(key, size_hint)
            self.stats[cache_key_class(key) + ('_misses' if value is None else '_hits')] += 1
            return value

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
            victims, freed = [], 0
            if self.bytes + nbytes > self.capacity or len(self.data) >= self.max_entries:
                for victim in self._ranked_residents():
                    victims.append(victim)
                    freed += self.sizes[victim]
                    if (self.bytes + nbytes - freed <= self.capacity and
                            len(self.data) + 1 - len(victims) <= self.max_entries):
                        break
                if any(self._preference(victim) > self._preference(key) for victim in victims):
                    self.stats['rejected'] += 1
                    self.stats['class_rejected'] += 1
                    return False
                if self.policy == 'hybrid':
                    candidate, lost = self._benefit(key), sum(self._benefit(v) for v in victims)
                    # Permit an exactly equal nonzero benefit only when every
                    # victim is expert and the candidate is shared.
                    shared_tie = (candidate > 0 and candidate == lost and
                        cache_key_class(key) != 'expert' and
                        all(cache_key_class(v) == 'expert' for v in victims))
                    if candidate <= lost and not shared_tie:
                        self.stats['rejected'] += 1
                        if candidate <= 0:
                            self.stats['cold_rejected'] += 1
                        return False
            self._ghosts.pop(key, None)
            for victim in victims:
                self.stats[cache_key_class(victim) + '_evictions'] += 1
                self._remove(victim)
            self.data[key] = value
            self.sizes[key] = nbytes
            self.bytes += nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.stats['admitted'] += 1
            self.stats[cache_key_class(key) + '_admitted_bytes'] += nbytes
            return True

    def clear(self):
        with self.lock:
            super().clear()
            self.batch_counts.clear()
