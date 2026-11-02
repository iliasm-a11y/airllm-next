"""Portable exact-byte/policy/reservation tests; no GLM or GPU execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from research.glm53_gpu.expert_cache import ByteSpan, CacheBudgetExceeded, CacheUnavailable, ExpertRef
from research.glm53_gpu_v2.cache_adaptive import AdaptiveExpertCache, ReusePolicy, adjacent_spans


class PausedCache(AdaptiveExpertCache):
    def __init__(self, *args, **kwargs):
        self.started = Event()
        self.resume = Event()
        super().__init__(*args, **kwargs)

    def _read_into(self, ref, payload):
        self.started.set()
        if not self.resume.wait(5):
            raise TimeoutError("fixture did not release exact read")
        super()._read_into(ref, payload)


class AdaptiveCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.source = Path(self.temp.name) / "source.gguf"
        self.source.write_bytes(bytes(range(256)))
        self.caches = []

    def tearDown(self):
        for cache in self.caches:
            if isinstance(cache, PausedCache):
                cache.resume.set()
            if cache._request:
                cache._request.close()
            cache.close()
        self.temp.cleanup()

    def cache(self, size=16, **kwargs):
        cls = kwargs.pop("cls", AdaptiveExpertCache)
        cache = cls(self.source, size, max_capacity_bytes=64,
                    enabled=True, admission=lambda n: True, **kwargs)
        self.caches.append(cache)
        return cache

    def ref(self, key, offset=0, size=4):
        return ExpertRef(key, (ByteSpan(offset, size),))

    def consume(self, cache, refs):
        with cache.request_selected(refs) as request:
            return [bytes(lease.data) for lease in request.iter_ready()]

    def test_exact_original_order_and_duplicate_ids(self):
        cache = self.cache()
        a, b = self.ref("a", 9), self.ref("b", 32)
        self.assertEqual(self.consume(cache, [b, a, b]), [bytes(range(32, 36)), bytes(range(9, 13)), bytes(range(32, 36))])
        self.assertEqual(cache.snapshot()["fills_submitted"], 2)
        self.assertEqual(cache.snapshot()["logical_bytes_read"], 8)
        self.consume(cache, [a, b])
        self.assertEqual(cache.snapshot()["hits"], 2)
        self.assertEqual(cache.snapshot()["misses"], 2)

    def test_measured_reuse_changes_eviction_without_changing_bytes(self):
        results = {}
        for policy in ("lru", "slru", "lfu"):
            cache = self.cache(8, policy=policy)
            for key, offset in (("a", 0), ("a", 0), ("a", 0), ("b", 4), ("c", 8)):
                self.consume(cache, [self.ref(key, offset)])
            results[policy] = set(cache._entries)
        self.assertEqual(results["lru"], {"b", "c"})
        self.assertEqual(results["slru"], {"a", "c"})
        self.assertEqual(results["lfu"], {"a", "c"})

    def test_weighted_uses_observed_cost_and_history_is_bounded(self):
        policy = ReusePolicy("weighted", history_limit=2)
        refs = [self.ref("a"), self.ref("b", 4)]
        policy.observe(refs, set(), 8)
        policy.record_load("a", .001)
        policy.record_load("b", .1)
        self.assertLess(policy.score("a", 4), policy.score("b", 4))
        policy.observe([self.ref("c", 8)], set(), 8)
        self.assertEqual(len(policy.history), 2)

    def test_adjacent_only_coalescing_preserves_payload_without_gap_reads(self):
        spans = (ByteSpan(8, 4), ByteSpan(12, 4), ByteSpan(64, 4))
        self.assertEqual(adjacent_spans(spans), [ByteSpan(8, 8), ByteSpan(64, 4)])
        expected = bytes(range(8, 16)) + bytes(range(64, 68))
        calls = {}
        for enabled in (False, True):
            cache = self.cache(16, coalesce=enabled)
            self.assertEqual(self.consume(cache, [ExpertRef("joined", spans)]), [expected])
            calls[enabled] = cache.snapshot()["read_syscalls"]
            self.assertEqual(cache.snapshot()["logical_bytes_read"], 12)
        self.assertEqual(calls, {False: 3, True: 2})

    def test_shrink_defers_while_pinned_and_rejects_new_requests(self):
        cache = self.cache(16, cls=PausedCache)
        request = cache.request_selected([self.ref("a", size=8)])
        self.assertTrue(cache.started.wait(2))
        self.assertFalse(cache.resize(0))
        self.assertEqual(cache.snapshot()["reserved_bytes"], 8)
        self.assertEqual(cache.snapshot()["deferred_shrink_bytes"], 8)
        self.assertEqual(cache.snapshot()["capacity_bytes"], 16)
        with self.assertRaises(CacheUnavailable):
            cache.request_selected([self.ref("b", 8)])
        cache.resume.set()
        request.close()
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        self.assertEqual(cache.snapshot()["capacity_bytes"], 0)
        with self.assertRaises(CacheBudgetExceeded):
            cache.request_selected([self.ref("b", 8)])

    def test_memory_adaptation_can_grow_bounded_inflight_or_reclaim_on_pressure(self):
        cache = self.cache(8)
        self.assertTrue(cache.adapt_to_memory(available_bytes=64, pressure_level=1, headroom_bytes=0))
        self.assertEqual(cache.snapshot()["capacity_bytes"], 64)
        self.assertEqual(self.consume(cache, [self.ref("a", size=32)]), [bytes(range(32))])
        self.assertTrue(cache.adapt_to_memory(available_bytes=64, pressure_level=2, headroom_bytes=0))
        self.assertEqual(cache.snapshot()["capacity_bytes"], 0)
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        with self.assertRaises(CacheUnavailable):
            cache.adapt_to_memory(available_bytes=64, pressure_level=99)
        with self.assertRaises(ValueError):
            cache.resize(65)

    def test_cancel_waits_for_read_before_releasing_reservation(self):
        cache = self.cache(16, cls=PausedCache)
        request = cache.request_selected([self.ref("a", size=8)])
        self.assertTrue(cache.started.wait(2))
        with ThreadPoolExecutor(1) as pool:
            close = pool.submit(request.cancel)
            self.assertFalse(close.done())
            self.assertEqual(cache.snapshot()["inflight_bytes"], 8)
            cache.resume.set()
            close.result(timeout=2)
        cache.resize(0)
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)
        self.assertEqual(cache.snapshot()["inflight_bytes"], 0)

    def test_cache_close_keeps_active_payload_until_lease_owner_finishes(self):
        cache = self.cache(16)
        request = cache.request_selected([self.ref("a", size=8)])
        leases = list(request.iter_ready())
        cache.close()
        self.assertEqual(bytes(leases[0].data), bytes(range(8)))
        self.assertEqual(cache.snapshot()["reserved_bytes"], 8)
        request.close()
        self.assertEqual(cache.snapshot()["reserved_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
