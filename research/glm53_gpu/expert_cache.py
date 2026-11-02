"""Opt-in, byte-exact GGUF expert cache for bounded Metal experiments.

This does not install a llama.cpp hook or predict future routes. The caller must
pass the *already selected* experts and preserve their order in computation.
Each expert can concatenate several original GGUF ranges (gate/up/down). Fills
use readinto(), retaining the packed quantization unchanged. Read-only leases
pin payloads until the caller has synchronized any asynchronous GPU consumer.

capacity_bytes bounds packed payloads, including queued/running fills and pinned
leases. It is not an RSS limit: Python overhead, GPU copies, page cache, model
state and other applications require the caller's separate memory admission.
There is no unbounded fallback allocation when admission fails.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Callable, Hashable, Iterable
import os


class CacheUnavailable(RuntimeError):
    """The experiment is disabled, closed, or rejected by memory admission."""


class CacheBudgetExceeded(MemoryError):
    """The exact selected set does not fit the current payload reservations."""


@dataclass(frozen=True)
class ByteSpan:
    offset: int
    length: int

    def __post_init__(self):
        for name, value in (("offset", self.offset), ("length", self.length)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.offset < 0 or self.length <= 0:
            raise ValueError("range offset must be nonnegative and length positive")


@dataclass(frozen=True)
class ExpertRef:
    key: Hashable
    spans: tuple[ByteSpan, ...]

    def __post_init__(self):
        hash(self.key)
        object.__setattr__(self, "spans", tuple(self.spans))
        if not self.spans or not all(isinstance(s, ByteSpan) for s in self.spans):
            raise ValueError("an expert needs at least one ByteSpan")

    @property
    def nbytes(self) -> int:
        return sum(s.length for s in self.spans)


@dataclass
class _Entry:
    ref: ExpertRef
    future: Future | None = None
    payload: bytearray | None = None
    pins: int = 0
    state: str = "pending"
    error: str | None = None
    reserved: bool = True


class ExpertLease:
    """Pinned packed bytes. Release only after the GPU has finished reading.

    Views are valid until release(). Do not retain a memoryview obtained from a
    lease after releasing it: Python cannot revoke independently retained views.
    The cache's budget covers managed leases, not deliberately retained views.
    """

    def __init__(self, owner: "ExpertByteCache", entry: _Entry):
        self._owner = owner
        self._entry = entry
        self._released = False

    @property
    def key(self):
        return self._entry.ref.key

    @property
    def data(self) -> memoryview:
        if self._released:
            raise CacheUnavailable("lease has been released")
        return memoryview(self._entry.payload).toreadonly()

    def parts(self) -> tuple[memoryview, ...]:
        data = self.data
        offset = 0
        result = []
        for span in self._entry.ref.spans:
            result.append(data[offset:offset + span.length])
            offset += span.length
        return tuple(result)

    def release(self):
        if not self._released:
            self._released = True
            self._owner._unpin(self._entry)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.release()


class ExpertSelection:
    """Ordered leases, including duplicate route IDs without duplicate fills."""

    def __init__(self, leases: list[ExpertLease]):
        self.leases = tuple(leases)

    def release(self):
        for lease in self.leases:
            lease.release()

    def __enter__(self):
        return self.leases

    def __exit__(self, *unused):
        self.release()


class ExpertByteCache:
    """Bound resident + in-flight packed bytes in one admission domain.

    Defaults to disabled. Enabling requires an external admission callback; its
    argument is the number of newly reserved packed bytes. Callback failure is
    fail-closed. It should account for the application's remaining RSS/swap and
    GPU working set. The callback must not re-enter this cache.

    A single physical model slot can submit its selected experts, overlap reads
    with independent CPU/GPU work, then acquire in original route order. Nothing
    here changes the application's 600 logical identities or slot count.
    """

    def __init__(self, source: str | Path, capacity_bytes: int = 0, *,
                 max_inflight_bytes: int | None = None, workers: int = 1,
                 max_entries: int = 128, enabled: bool = False,
                 admission: Callable[[int], bool] | None = None):
        for name, value, minimum in (("capacity_bytes", capacity_bytes, 0),
                                     ("workers", workers, 1),
                                     ("max_entries", max_entries, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if max_inflight_bytes is None:
            max_inflight_bytes = capacity_bytes
        if (not isinstance(max_inflight_bytes, int) or isinstance(max_inflight_bytes, bool)
                or not 0 <= max_inflight_bytes <= capacity_bytes):
            raise ValueError("in-flight budget must lie within total payload budget")
        if enabled and not callable(admission):
            raise ValueError("enabled cache requires external memory admission")
        self.source = Path(source)
        stat = self.source.stat()
        self._identity = self._file_identity(stat)
        self.capacity_bytes = capacity_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_entries = max_entries
        self.enabled = enabled
        self.admission = admission
        self._lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="glm-exact-io")
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._closed = False
        self._bytes = 0
        self._inflight = 0
        self._peak_bytes = 0
        self._stats = Counter()

    @staticmethod
    def _file_identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _check_open(self):
        if not self.enabled:
            raise CacheUnavailable("expert byte cache is disabled")
        if self._closed:
            raise CacheUnavailable("expert byte cache is closed")

    def _validate(self, ref: ExpertRef):
        if not isinstance(ref, ExpertRef):
            raise TypeError("selected experts must be ExpertRef instances")
        for span in ref.spans:
            if span.offset + span.length > self._identity[2]:
                raise ValueError("expert range exceeds the source shard")

    def _drop(self, entry: _Entry):
        # Only a completed/cancelled reader may relinquish its reservation.
        if entry.pins or entry.state == "pending":
            raise RuntimeError("attempted to evict an active expert")
        if self._entries.get(entry.ref.key) is entry:
            del self._entries[entry.ref.key]
        if entry.reserved:
            self._bytes -= entry.ref.nbytes
            entry.reserved = False
        entry.payload = None

    def _prepare(self, refs: Iterable[ExpertRef], pin: bool) -> list[_Entry]:
        refs = tuple(refs)
        with self._lock:
            self._check_open()
            unique = {}
            for ref in refs:
                self._validate(ref)
                previous = unique.setdefault(ref.key, ref)
                if previous != ref:
                    raise ValueError("a selected key refers to different byte ranges")
                current = self._entries.get(ref.key)
                if current is not None and current.ref != ref:
                    raise ValueError("cached key refers to different byte ranges")
            if len(unique) > self.max_entries:
                raise CacheBudgetExceeded("selected experts exceed metadata entry bound")
            new = [r for r in unique.values() if r.key not in self._entries]
            needed = sum(r.nbytes for r in new)
            if sum(r.nbytes for r in unique.values()) > self.capacity_bytes:
                raise CacheBudgetExceeded("selected experts exceed the total payload budget")
            if self._inflight + needed > self.max_inflight_bytes:
                raise CacheBudgetExceeded("selected reads exceed the in-flight payload budget")
            victims = [e for key, e in self._entries.items()
                       if key not in unique and not e.pins and e.state != "pending"]
            reclaim = 0
            victim_count = 0
            while (self._bytes - reclaim + needed > self.capacity_bytes or
                   len(self._entries) - victim_count + len(new) > self.max_entries):
                if victim_count == len(victims):
                    raise CacheBudgetExceeded("payload budget is pinned or still being read")
                reclaim += victims[victim_count].ref.nbytes if victims[victim_count].reserved else 0
                victim_count += 1
            if needed:
                try:
                    approved = bool(self.admission(needed))
                except Exception as exc:
                    raise CacheUnavailable("external memory admission failed") from exc
                if not approved:
                    raise CacheUnavailable("external memory admission rejected packed bytes")
            for victim in victims[:victim_count]:
                self._drop(victim)
                self._stats["evictions"] += 1
            added = []
            try:
                for ref in new:
                    entry = _Entry(ref)
                    self._entries[ref.key] = entry
                    self._bytes += ref.nbytes
                    self._inflight += ref.nbytes
                    self._peak_bytes = max(self._peak_bytes, self._bytes)
                    added.append(entry)
                    entry.future = self._pool.submit(self._load, entry)
                    self._stats["fills_submitted"] += 1
            except BaseException:
                # Running readers keep their reservation. Queued readers can be
                # cancelled; no untracked background allocation is permitted.
                for entry in added:
                    if entry.future is None or entry.future.cancel():
                        entry.state = "cancelled"
                        self._inflight -= entry.ref.nbytes
                        self._drop(entry)
                raise
            result = []
            new_keys = {r.key for r in new}
            for ref in refs:
                entry = self._entries[ref.key]
                if pin:
                    entry.pins += 1
                self._entries.move_to_end(ref.key)
                result.append(entry)
                self._stats["selected_requests"] += 1
                if ref.key not in new_keys:
                    self._stats["reused_requests"] += 1
            return result

    def _read_into(self, ref: ExpertRef, payload: bytearray):
        # Each worker owns its file position. readinto avoids os.pread's extra
        # whole-range bytes allocation. Read-only open cannot mutate the GGUF.
        with self.source.open("rb", buffering=0) as source:
            if self._file_identity(os.fstat(source.fileno())) != self._identity:
                raise OSError("source shard identity, length, or modification time changed")
            target = memoryview(payload)
            offset = 0
            for span in ref.spans:
                source.seek(span.offset)
                end = offset + span.length
                while offset < end:
                    count = source.readinto(target[offset:end])
                    if not count:
                        raise OSError("short read of selected expert bytes")
                    offset += count
            # Detect a source edited while this read was in progress as well.
            if self._file_identity(os.fstat(source.fileno())) != self._identity:
                raise OSError("source shard changed during expert read")

    def _load(self, entry: _Entry):
        started = perf_counter()
        payload = None
        error = None
        try:
            payload = bytearray(entry.ref.nbytes)
            self._read_into(entry.ref, payload)
        except BaseException as exc:
            # Do not store exception tracebacks retaining the allocated payload.
            error = f"{type(exc).__name__}: {exc}"
            payload = None
        with self._lock:
            entry.payload = payload
            entry.error = error
            entry.state = "failed" if error else "ready"
            self._inflight -= entry.ref.nbytes
            self._stats["read_seconds"] += perf_counter() - started
            if error:
                self._stats["read_failures"] += 1
                if entry.reserved:
                    self._bytes -= entry.ref.nbytes
                    entry.reserved = False
                if not entry.pins:
                    self._drop(entry)
            else:
                self._stats["logical_bytes_read"] += entry.ref.nbytes

    def prefetch_selected(self, refs: Iterable[ExpertRef]):
        """Start bounded reads only for these already-routed experts.

        Returns after submission, not after reading. A later acquire waits only
        for unfinished fills. The function never guesses the next layer's IDs.
        """
        self._prepare(refs, pin=False)

    def acquire_selected(self, refs: Iterable[ExpertRef]) -> ExpertSelection:
        """Pin and wait for exactly these experts, preserving selection order."""
        entries = self._prepare(refs, pin=True)
        started = perf_counter()
        try:
            for entry in entries:
                entry.future.result()
                with self._lock:
                    if self._closed:
                        raise CacheUnavailable("cache closed while waiting for expert bytes")
                    if entry.error:
                        raise OSError(entry.error)
                    if entry.payload is None:
                        raise CacheUnavailable("selected fill was cancelled")
            return ExpertSelection([ExpertLease(self, e) for e in entries])
        except BaseException:
            for entry in entries:
                self._unpin(entry)
            raise
        finally:
            with self._lock:
                self._stats["acquire_wait_seconds"] += perf_counter() - started

    def acquire(self, ref: ExpertRef) -> ExpertLease:
        return self.acquire_selected([ref]).leases[0]

    def _unpin(self, entry: _Entry):
        with self._lock:
            if entry.pins <= 0:
                raise RuntimeError("unbalanced expert lease release")
            entry.pins -= 1
            if (not entry.pins and entry.state != "pending" and
                    (self._closed or entry.error or entry.state == "cancelled")):
                self._drop(entry)

    def trim(self):
        """Drop completed unpinned payloads; never abandon running reads."""
        with self._lock:
            for entry in list(self._entries.values()):
                if not entry.pins and entry.state != "pending":
                    self._drop(entry)

    def cancel_prefetch(self):
        """Cancel queued unpinned fills; drain running fills before discarding.

        Waiting is intentional: releasing a reservation before its worker exits
        could allow a second allocation to exceed the budget.
        """
        with self._lock:
            entries = [e for e in self._entries.values() if not e.pins]
            for entry in entries:
                if entry.state == "pending" and entry.future.cancel():
                    entry.state = "cancelled"
                    self._inflight -= entry.ref.nbytes
        for entry in entries:
            if not entry.future.cancelled():
                entry.future.result()
            with self._lock:
                if not entry.pins and entry.state != "pending":
                    self._drop(entry)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._stats, capacity_bytes=self.capacity_bytes,
                        reserved_bytes=self._bytes, inflight_bytes=self._inflight,
                        peak_reserved_bytes=self._peak_bytes,
                        resident_bytes=self._bytes - self._inflight,
                        pinned_leases=sum(e.pins for e in self._entries.values()),
                        entries=len(self._entries), enabled=self.enabled, closed=self._closed,
                        logical_bytes_are_not_physical_ssd_traffic=True)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # This drains running reads before their reservations can be released.
        self.cancel_prefetch()
        # Pinned queued fills may have an acquire() waiter. Let them finish so
        # its closed-cache error can release a correctly completed reservation.
        self._pool.shutdown(wait=True, cancel_futures=False)
        self.trim()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
