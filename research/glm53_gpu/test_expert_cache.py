"""Payload accounting and byte fidelity tests; not model/speed evidence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
import unittest

from expert_cache import (ByteSpan, CacheBudgetExceeded, CacheUnavailable,
                          ExpertByteCache, ExpertRef)


class PausedCache(ExpertByteCache):
    def __init__(self, *args, **kwargs):
        self.started = Event()
        self.resume = Event()
        super().__init__(*args, **kwargs)

    def _read_into(self, ref, payload):
        self.started.set()
        if not self.resume.wait(5):
            raise TimeoutError("test did not resume reader")
        return super()._read_into(ref, payload)


class ObservedClosingCache(PausedCache):
    def __init__(self, *args, **kwargs):
        self.pins_ready = Event()
        self.closing = Event()
        super().__init__(*args, **kwargs)

    def _prepare(self, refs, pin):
        result = super()._prepare(refs, pin)
        if pin:
            self.pins_ready.set()
        return result

    def cancel_prefetch(self):
        if self.snapshot()["closed"]:
            self.closing.set()
        return super().cancel_prefetch()


class ExpertCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.source = Path(self.temp.name) / "weights.gguf"
        self.source.write_bytes(bytes(range(256)) * 4)
        self.caches = []

    def tearDown(self):
        for cache in self.caches:
            if isinstance(cache, PausedCache):
                cache.resume.set()
            cache.close()
        self.temp.cleanup()

    def cache(self, size=32, kind=ExpertByteCache, **kwargs):
        cache = kind(self.source, size, enabled=True,
                     admission=kwargs.pop("admission", lambda n: True), **kwargs)
        self.caches.append(cache)
        return cache

    def ref(self, key, offset=0, size=16):
        return ExpertRef(key, (ByteSpan(offset, size),))

    def test_disabled_by_default_and_enabled_needs_admission(self):
        cache = ExpertByteCache(self.source, 32)
        self.caches.append(cache)
        with self.assertRaises(CacheUnavailable):
            cache.prefetch_selected([self.ref("a")])
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        with self.assertRaises(ValueError):
            ExpertByteCache(self.source, 32, enabled=True)

    def test_original_ranges_and_route_order_with_duplicates(self):
        cache = self.cache(32)
        a = ExpertRef((3, 9), (ByteSpan(9, 4), ByteSpan(128, 4)))
        b = self.ref((3, 2), 32, 8)
        with cache.acquire_selected([b, a, b]) as leases:
            self.assertEqual([v.key for v in leases], [b.key, a.key, b.key])
            self.assertEqual(bytes(leases[0].data), bytes(range(32, 40)))
            self.assertEqual([bytes(v) for v in leases[1].parts()],
                             [bytes(range(9, 13)), bytes(range(128, 132))])
            self.assertTrue(leases[0].data.readonly)
            with self.assertRaises(TypeError):
                leases[0].data[0] = 255
        stats = cache.snapshot()
        self.assertEqual(stats["fills_submitted"], 2)
        self.assertEqual(stats["logical_bytes_read"], 16)
        self.assertEqual(stats["pinned_leases"], 0)

    def test_prefetch_deduplicates_and_counts_running_payload(self):
        cache = self.cache(16, kind=PausedCache)
        ref = self.ref("a")
        cache.prefetch_selected([ref])
        self.assertTrue(cache.started.wait(2))
        cache.prefetch_selected([ref])
        stats = cache.snapshot()
        self.assertEqual(stats["reserved_bytes"], 16)
        self.assertEqual(stats["inflight_bytes"], 16)
        self.assertEqual(stats["fills_submitted"], 1)
        with self.assertRaises(CacheBudgetExceeded):
            cache.prefetch_selected([self.ref("b", 16)])
        self.assertEqual(cache.snapshot()["peak_reserved_bytes"], 16)
        cache.resume.set()
        with cache.acquire(ref) as lease:
            self.assertEqual(bytes(lease.data), bytes(range(16)))

    def test_selected_set_is_admitted_atomically(self):
        calls = []
        cache = self.cache(16, admission=lambda n: calls.append(n) or True)
        with self.assertRaises(CacheBudgetExceeded):
            cache.prefetch_selected([self.ref("a"), self.ref("b", 16)])
        self.assertFalse(calls)
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)

    def test_distinct_inflight_budget_rejects_without_overallocation(self):
        cache = self.cache(32, max_inflight_bytes=8)
        with self.assertRaises(CacheBudgetExceeded):
            cache.prefetch_selected([self.ref("a")])
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)

    def test_memory_admission_rejection_or_error_is_fail_closed(self):
        for admission in (lambda n: False, lambda n: 1 / 0):
            cache = self.cache(admission=admission)
            with self.assertRaises(CacheUnavailable):
                cache.acquire(self.ref("a"))
            self.assertEqual(cache.snapshot()["reserved_bytes"], 0)

    def test_pinned_payload_cannot_be_evicted(self):
        cache = self.cache(16)
        lease = cache.acquire(self.ref("a"))
        with self.assertRaises(CacheBudgetExceeded):
            cache.acquire(self.ref("b", 16))
        self.assertEqual(bytes(lease.data), bytes(range(16)))
        lease.release()
        with cache.acquire(self.ref("b", 16)) as other:
            self.assertEqual(bytes(other.data), bytes(range(16, 32)))
        self.assertEqual(cache.snapshot()["evictions"], 1)

    def test_completed_leases_survive_close_until_release(self):
        cache = self.cache(16)
        lease = cache.acquire(self.ref("a"))
        cache.close()
        self.assertEqual(cache.snapshot()["reserved_bytes"], 16)
        self.assertEqual(bytes(lease.data), bytes(range(16)))
        lease.release()
        lease.release()  # idempotent
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        with self.assertRaises(CacheUnavailable):
            _ = lease.data
        with self.assertRaises(CacheUnavailable):
            cache.acquire(self.ref("b"))

    def test_trim_releases_only_unpinned_completed_payloads(self):
        cache = self.cache(32)
        lease = cache.acquire(self.ref("a"))
        with cache.acquire(self.ref("b", 16)):
            pass
        cache.trim()
        self.assertEqual(cache.snapshot()["reserved_bytes"], 16)
        lease.release()
        cache.trim()
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)

    def test_source_truncation_fails_without_retaining_reservation(self):
        cache = self.cache(16)
        self.source.write_bytes(b"short")
        with self.assertRaisesRegex(OSError, "source shard"):
            cache.acquire(self.ref("a"))
        stats = cache.snapshot()
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["entries"], 0)

    def test_wrong_key_range_is_rejected(self):
        cache = self.cache(32)
        with cache.acquire(self.ref("a")):
            with self.assertRaises(ValueError):
                cache.acquire(self.ref("a", 16))
        with self.assertRaises(ValueError):
            cache.acquire_selected([self.ref("b"), self.ref("b", 16)])

    def test_range_bounds_and_metadata_bounds(self):
        cache = self.cache(64, max_entries=1)
        with self.assertRaises(ValueError):
            cache.acquire(self.ref("bad", 1020))
        with self.assertRaises(CacheBudgetExceeded):
            cache.acquire_selected([self.ref("a"), self.ref("b", 16)])
        with cache.acquire(self.ref("a")):
            pass
        with cache.acquire(self.ref("b", 16)):
            pass
        self.assertEqual(cache.snapshot()["entries"], 1)

    def test_cancel_prefetch_waits_for_running_reader_and_releases_bytes(self):
        cache = self.cache(32, kind=PausedCache)
        cache.prefetch_selected([self.ref("a"), self.ref("b", 16)])
        self.assertTrue(cache.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            done = pool.submit(cache.cancel_prefetch)
            # Reader is running: its reservation must remain until it exits.
            self.assertGreaterEqual(cache.snapshot()["reserved_bytes"], 16)
            self.assertFalse(done.done())
            cache.resume.set()
            done.result(timeout=2)
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        self.assertEqual(cache.snapshot()["inflight_bytes"], 0)

    def test_close_waits_for_running_prefetch(self):
        cache = self.cache(16, kind=PausedCache)
        cache.prefetch_selected([self.ref("a")])
        self.assertTrue(cache.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            done = pool.submit(cache.close)
            self.assertFalse(done.done())
            self.assertEqual(cache.snapshot()["reserved_bytes"], 16)
            cache.resume.set()
            done.result(timeout=2)
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)

    def test_close_drains_queued_pinned_acquires_without_leaking_reservations(self):
        cache = self.cache(32, kind=ObservedClosingCache)
        with ThreadPoolExecutor(2) as pool:
            acquire = pool.submit(cache.acquire_selected,
                                  [self.ref("a"), self.ref("b", 16)])
            self.assertTrue(cache.started.wait(2))
            self.assertTrue(cache.pins_ready.wait(2))
            close = pool.submit(cache.close)
            self.assertTrue(cache.closing.wait(2))
            cache.resume.set()
            close.result(timeout=2)
            with self.assertRaises(CacheUnavailable):
                acquire.result(timeout=2)
        stats = cache.snapshot()
        self.assertEqual(stats["reserved_bytes"], 0)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["pinned_leases"], 0)

    def test_invalid_configuration(self):
        for kwargs in ({"capacity_bytes": -1}, {"workers": 0},
                       {"capacity_bytes": True}, {"max_entries": 0},
                       {"capacity_bytes": 8, "max_inflight_bytes": 9}):
            with self.assertRaises(ValueError):
                ExpertByteCache(self.source, **kwargs)
        for args in ((-1, 1), (0, 0), (False, 1)):
            with self.assertRaises(ValueError):
                ByteSpan(*args)


if __name__ == "__main__":
    unittest.main()
