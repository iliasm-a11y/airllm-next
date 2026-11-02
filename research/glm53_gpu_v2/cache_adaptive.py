"""Exact-byte, single-slot adaptive staging cache; experimental and default off.

Reuses the audited payload reservations and read-only leases from PR18. A
selected request pins its routed experts before asynchronous reads start. The
consumer may process each ready expert while the reader fills later experts.
No route prediction, quantization change or production scheduler is installed.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from time import perf_counter
import os

from research.glm53_gpu.expert_cache import (
    ByteSpan, CacheBudgetExceeded, CacheUnavailable, ExpertByteCache,
    ExpertLease, ExpertRef)

POLICIES = ("lru", "slru", "lfu", "weighted")


class ReusePolicy:
    """Bounded observed reuse history; weighted mode needs measured read cost.

    No future trace is consulted. Cold weighted entries have zero measured
    transfer cost until a real load completes. A trace-only replay therefore
    must not invent a weighted ranking from a supposed SSD speed.
    """
    def __init__(self, name="lru", history_limit=4096):
        if name not in POLICIES:
            raise ValueError("unknown eviction policy")
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.name = name
        self.limit = history_limit
        self.clock = 0
        self.history = OrderedDict()
        self.protected = OrderedDict()

    def observe(self, refs, resident_keys, capacity):
        unique = {r.key: r for r in refs}
        for key, ref in unique.items():
            self.clock += 1
            old = self.history.get(key, (0, 0, None))
            self.history[key] = (old[0] + 1, self.clock, old[2])
            self.history.move_to_end(key)
            if key in resident_keys and old[0] > 0:
                self.protected[key] = ref.nbytes
                self.protected.move_to_end(key)
        protected_limit = capacity // 2
        while self.protected and sum(self.protected.values()) > protected_limit:
            self.protected.popitem(last=False)
        while len(self.history) > self.limit:
            # Bounded history can forget a resident's prior frequency. That
            # affects eviction preference only, never its bytes or routes.
            forgotten, _ = self.history.popitem(last=False)
            self.protected.pop(forgotten, None)

    def record_load(self, key, seconds):
        if key in self.history:
            freq, recency, old = self.history[key]
            self.history[key] = (freq, recency, seconds if old is None else .75 * old + .25 * seconds)

    def forget_resident(self, key):
        self.protected.pop(key, None)

    def score(self, key, nbytes):
        frequency, recency, cost = self.history.get(key, (0, 0, None))
        if self.name == "slru":
            return (key in self.protected, recency)
        if self.name == "lfu":
            return (frequency, recency)
        if self.name == "weighted":
            return (frequency * (cost or 0.0) / max(1, nbytes), recency)
        return (recency,)


class _PinnedLease(ExpertLease):
    def release(self):
        # The shared cache is thread-aware even though the model slot is one.
        with self._owner._lock:
            super().release()


class SelectedRequest:
    """One routed selection, pinned from submission through synchronized use.

    `iter_ready()` yields experts in original route order. It waits only for the
    next expert, allowing independent CPU/GPU work to overlap later reads.
    Synchronize a GPU consumer before closing this request. Do not retain views.
    """
    def __init__(self, cache, entries):
        self.cache = cache
        self.entries = entries
        self.leases = tuple(_PinnedLease(cache, entry) for entry in entries)
        self.closed = False

    def iter_ready(self):
        if self.closed:
            raise CacheUnavailable("selected request is closed")
        for entry, lease in zip(self.entries, self.leases):
            start = perf_counter()
            entry.future.result()
            waited = perf_counter() - start
            with self.cache._lock:
                self.cache._metrics["consumer_wait_seconds"] += waited
                if self.cache._closed:
                    raise CacheUnavailable("cache closed while waiting for selected bytes")
                if entry.error:
                    raise OSError(entry.error)
                if entry.payload is None:
                    raise CacheUnavailable("selected fill was cancelled")
            yield lease

    def close(self):
        with self.cache._lock:
            if self.closed:
                return
            self.closed = True
        # Keep every reservation until the associated read has actually exited.
        try:
            for entry in self.entries:
                if not entry.future.cancelled():
                    entry.future.result()
        finally:
            for lease in self.leases:
                lease.release()
            with self.cache._lock:
                if self.cache._request is self:
                    self.cache._request = None
                self.cache._rebalance()

    cancel = close

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def adjacent_spans(spans):
    """Merge exact adjacent source spans in payload order, with no gap reads."""
    groups = []
    for span in spans:
        if groups and groups[-1].offset + groups[-1].length == span.offset:
            previous = groups.pop()
            groups.append(ByteSpan(previous.offset, previous.length + span.length))
        else:
            groups.append(span)
    return groups


class AdaptiveExpertCache(ExpertByteCache):
    """Policy-aware cache with deferred safe shrinking and an exact IO pipeline.

    max_capacity_bytes is a hard payload ceiling, <=128 MiB. resize() may request
    a smaller target while a consumer/read pins bytes. It then rejects new
    requests and retains the old effective reservation ceiling until those
    users drain. No reservation is released early to make a target look met.
    Payload bytes exclude file-system cache, Python and GPU runtime overhead.
    """
    def __init__(self, source, capacity_bytes=0, *, policy="lru",
                 max_capacity_bytes=128 * 1024**2, history_limit=4096,
                 coalesce=True, **kwargs):
        if (type(max_capacity_bytes) is not int or
                not 0 <= capacity_bytes <= max_capacity_bytes <= 128 * 1024**2):
            raise ValueError("capacity must fit the hard <=128 MiB payload ceiling")
        self.policy = ReusePolicy(policy, history_limit)
        self.target_capacity_bytes = capacity_bytes
        self.max_capacity_bytes = max_capacity_bytes
        self.coalesce = coalesce
        self._metrics = Counter()
        self._request = None
        super().__init__(source, capacity_bytes, **kwargs)
        self._inflight_ceiling = (max_capacity_bytes if kwargs.get("max_inflight_bytes") is None
                                 else kwargs["max_inflight_bytes"])

    def _sort_victims(self):
        keys = sorted(self._entries, key=lambda k: self.policy.score(k, self._entries[k].ref.nbytes))
        self._entries = OrderedDict((key, self._entries[key]) for key in keys)

    def _drop(self, entry):
        super()._drop(entry)
        self.policy.forget_resident(entry.ref.key)

    def request_selected(self, refs):
        refs = tuple(refs)
        with self._lock:
            self._check_open()
            if self._request is not None:
                raise CacheUnavailable("one physical selected-request slot is already active")
            self._rebalance()
            if self._bytes > self.target_capacity_bytes:
                raise CacheBudgetExceeded("adaptive shrink is waiting for existing readers or leases")
            resident = {key for key, entry in self._entries.items() if entry.state == "ready"}
            self._sort_victims()
            entries = super()._prepare(refs, pin=True)
            unique = {r.key: r for r in refs}
            self._metrics["route_requests"] += 1
            self._metrics["selected_experts"] += len(refs)
            self._metrics["unique_selected_experts"] += len(unique)
            self._metrics["hits"] += sum(key in resident for key in unique)
            self._metrics["misses"] += sum(key not in resident for key in unique)
            self.policy.observe(refs, resident, self.target_capacity_bytes)
            request = SelectedRequest(self, entries)
            self._request = request
            return request

    def _load(self, entry):
        started = perf_counter()
        super()._load(entry)
        with self._lock:
            if not entry.error:
                self.policy.record_load(entry.ref.key, perf_counter() - started)
            self._rebalance()

    def _read_into(self, ref, payload):
        groups = adjacent_spans(ref.spans) if self.coalesce else ref.spans
        calls = 0
        try:
            with self.source.open("rb", buffering=0) as source:
                if self._file_identity(os.fstat(source.fileno())) != self._identity:
                    raise OSError("source shard identity changed")
                target = memoryview(payload)
                position = 0
                for span in groups:
                    source.seek(span.offset)
                    end = position + span.length
                    while position < end:
                        count = source.readinto(target[position:end])
                        calls += 1
                        if not count:
                            raise OSError("short read of exact selected bytes")
                        position += count
                if self._file_identity(os.fstat(source.fileno())) != self._identity:
                    raise OSError("source shard changed during read")
        finally:
            with self._lock:
                self._metrics["read_syscalls"] += calls
                self._metrics["source_spans"] += len(ref.spans)
                self._metrics["coalesced_spans"] += len(groups)

    def _rebalance(self):
        if self._bytes > self.target_capacity_bytes:
            self._sort_victims()
            for entry in list(self._entries.values()):
                if self._bytes <= self.target_capacity_bytes:
                    break
                if not entry.pins and entry.state != "pending":
                    self._drop(entry)
                    self._metrics["adaptive_evictions"] += 1
        if self._bytes <= self.target_capacity_bytes:
            self.capacity_bytes = self.target_capacity_bytes
            self.max_inflight_bytes = min(self._inflight_ceiling, self.capacity_bytes)

    def _unpin(self, entry):
        with self._lock:
            super()._unpin(entry)
            self._rebalance()

    def resize(self, capacity_bytes):
        if type(capacity_bytes) is not int or not 0 <= capacity_bytes <= self.max_capacity_bytes:
            raise ValueError("adaptive target must fit the configured hard ceiling")
        with self._lock:
            self._check_open()
            self.target_capacity_bytes = capacity_bytes
            self._metrics["resize_requests"] += 1
            self._rebalance()
            return self._bytes <= capacity_bytes

    def adapt_to_memory(self, *, available_bytes, pressure_level, headroom_bytes=512 * 1024**2):
        if any(type(v) is not int or v < 0 for v in (available_bytes, headroom_bytes)):
            raise ValueError("memory observations must be nonnegative integer bytes")
        if type(pressure_level) is not int or pressure_level not in (1, 2, 4):
            raise CacheUnavailable("unknown memory pressure observation")
        with self._lock:
            target = (min(self.max_capacity_bytes, max(0, self._bytes + available_bytes - headroom_bytes))
                      if pressure_level == 1 else 0)
        return self.resize(target)

    def snapshot(self):
        with self._lock:
            result = super().snapshot()
            result.update(self._metrics)
            result.update(policy=self.policy.name, target_capacity_bytes=self.target_capacity_bytes,
                          max_capacity_bytes=self.max_capacity_bytes,
                          deferred_shrink_bytes=max(0, self._bytes - self.target_capacity_bytes),
                          history_entries=len(self.policy.history),
                          measured_load_cost_entries=sum(v[2] is not None for v in self.policy.history.values()),
                          active_physical_requests=int(self._request is not None),
                          coalescing_reads_no_gaps=True)
            return result
