"""Cache invariants and deterministic trace replays, not hardware benchmarks."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.adaptive_cache import AdaptiveWeightCache, retained_size
from airllm.nextgen.cache import WeightCache
from airllm.nextgen.quantization import PackedMatrix


def access(cache, key, *, size=8, seconds=1.):
    value = cache.get(key, size)
    if value is None:
        cache.record_load(key, seconds)
        value = SimpleNamespace(nbytes=size)
        cache.put(key, value)
    return value


def replay(trace, capacity, **adaptive_options):
    """Equal synthetic requests/costs for every policy; no I/O takes place."""
    rows = {}
    for policy in ('lru', 'lfu', 'cost', 'adaptive'):
        cache = (AdaptiveWeightCache(capacity, **adaptive_options)
                 if policy == 'adaptive' else WeightCache(capacity, policy))
        miss_bytes = service = 0
        for key, size, seconds in trace:
            if cache.get(key, size) is None:
                miss_bytes += size
                service += seconds
                cache.record_load(key, seconds)
                cache.put(key, SimpleNamespace(nbytes=size))
            assert cache.bytes <= capacity
        rows[policy] = {'misses': cache.stats['misses'], 'miss_weight_bytes': miss_bytes,
                        'synthetic_reload_work': service, 'peak_payload_bytes': cache.peak_bytes}
    return rows


def test_weights_reused_as_exact_objects_without_conversion_or_mutation():
    raw = np.array([0x7fc00001, 0x80000000, 0x3f800000], dtype=np.uint32)
    value = raw.view(np.float32)
    before = raw.copy()
    cache = AdaptiveWeightCache(raw.nbytes)
    cache.get('exact', value.nbytes)
    cache.record_load('exact', .001)
    assert cache.put('exact', value)
    assert cache.get('exact') is value
    np.testing.assert_array_equal(value.view(np.uint32), before)
    assert value.flags.writeable  # Cache admission never changes caller arrays.


def test_numpy_views_charge_complete_unique_backing_without_copy():
    base = np.arange(128, dtype=np.uint8)
    small, other = base[:2], base[50:52]
    assert retained_size((small, other)) == base.nbytes
    cache = AdaptiveWeightCache(127)
    assert not cache.put('view', small)
    assert cache.bytes == 0
    cache.resize(128)
    assert cache.put('views', (small, other))
    assert cache.bytes == 128 and cache.data['views'][0] is small
    packed = PackedMatrix(small, other, small, (1, 2), 4, 32)
    assert retained_size(packed) == 128


def test_one_off_scan_cannot_displace_hot_residents_even_with_slow_loads():
    cache = AdaptiveWeightCache(32, decay_interval=16)
    hot = [('layers.0.experts', i) for i in range(4)]
    for _ in range(8):
        for key in hot:
            access(cache, key)
    original = dict(cache.data)
    for i in range(256):
        access(cache, ('cold_scan', i), seconds=100.)
    assert set(cache.data) == set(hot)
    assert all(cache.data[key] is value for key, value in original.items())
    assert cache.stats['cold_rejected'] == 256
    assert cache.bytes == 32


def test_budget_migrates_between_layers_after_observed_demand_change():
    cache = AdaptiveWeightCache(32, decay_interval=16)
    old = [('layers.0.experts', i) for i in range(4)]
    new = [('layers.8.experts', i) for i in range(4)]
    for _ in range(50):
        for key in old:
            access(cache, key)
    before = cache.stats['hits']
    for _ in range(20):
        for key in new:
            access(cache, key)
    assert set(cache.data) == set(new)
    assert cache.stats['hits'] - before >= 60
    assert cache.bytes == cache.peak_bytes == cache.capacity


def test_measured_reload_cost_can_outweigh_equal_size_frequency():
    cache = AdaptiveWeightCache(8, decay_interval=64)
    for _ in range(6):
        access(cache, 'cheap', seconds=.001)
    access(cache, 'expensive', seconds=1.)
    assert cache.contains('cheap')
    access(cache, 'expensive', seconds=1.)
    assert cache.contains('expensive') and not cache.contains('cheap')


def test_large_candidate_must_pay_for_all_victims_and_rejection_is_atomic():
    cache = AdaptiveWeightCache(32, decay_interval=100000)
    for _ in range(3):
        for key in range(4):
            access(cache, key)
    before = dict(cache.data)
    # A single 32-byte candidate could outrank each 8-byte victim by absolute
    # benefit while still destroying more useful reload work in aggregate.
    for _ in range(4):
        access(cache, 'large', size=32)
    assert dict(cache.data) == before
    assert cache.stats['evictions'] == 0
    assert cache.bytes == 32
    for _ in range(12):
        access(cache, 'large', size=32)
    assert list(cache.data) == ['large']


def test_prefetch_cost_observations_do_not_fabricate_repeated_demand():
    cache = AdaptiveWeightCache(8)
    access(cache, 'resident')
    for _ in range(100):
        cache.record_load('speculative', 100.)
    assert not cache.put('speculative', SimpleNamespace(nbytes=8))
    access(cache, 'speculative', seconds=100.)
    assert cache.contains('resident')
    assert not cache.contains('speculative')


def test_admission_keeps_observed_cost_with_tiny_history_and_multiple_victims():
    cache = AdaptiveWeightCache(16, max_metadata_entries=1)
    access(cache, 'first')
    access(cache, 'second')
    access(cache, 'large', size=16, seconds=.25)
    access(cache, 'large', size=16, seconds=.25)
    assert list(cache.data) == ['large']
    assert cache.costs['large'] == .25
    assert cache.freq['large'] > 1
    assert len(cache.freq) <= len(cache.data) + 1


def test_history_entry_count_trace_limits_resize_and_clear_are_bounded():
    cache = AdaptiveWeightCache(10_000, max_entries=3, max_metadata_entries=7)
    for key in range(4300):
        access(cache, key, size=1)
        assert len(cache.data) <= 3
        assert len(cache.freq) <= 10
        assert len(cache.costs) <= 10
        assert len(cache._last_seen) <= 10
        assert len(cache._ghosts) <= 7
    assert len(cache.trace) == len(cache.cost_trace) == len(cache.load_trace) == 4096
    cache.resize(1)
    assert cache.bytes <= 1
    assert len(cache.freq) <= len(cache.data) + cache.max_metadata_entries
    cache.clear()
    assert not cache.data and not cache.sizes and not cache.freq and not cache.costs
    assert not cache._last_seen and not cache._ghosts
    assert not cache.trace and not cache.cost_trace and not cache.load_trace
    assert cache.bytes == 0 and cache.peak_bytes == 3


def test_concurrent_load_observations_and_pressure_preserve_bounds():
    cache = AdaptiveWeightCache(64, max_entries=8, max_metadata_entries=8, decay_interval=8)
    def worker(index):
        for offset in range(40):
            access(cache, (index, offset % 8))
            if offset % 7 == 0:
                cache.resize(32 if index % 2 else 64)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    assert cache.bytes == sum(cache.sizes.values()) <= cache.capacity
    assert len(cache.freq) <= cache.max_entries + cache.max_metadata_entries
    assert set(cache.data) <= set(cache.freq)


@pytest.mark.parametrize('capacity', [-1, True, 1.5, float('nan')])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError, match='capacity'):
        AdaptiveWeightCache(capacity)
    with pytest.raises(ValueError, match='capacity'):
        AdaptiveWeightCache(8).resize(capacity)


@pytest.mark.parametrize('option', ['max_entries', 'max_metadata_entries', 'decay_interval'])
def test_invalid_history_options_rejected(option):
    with pytest.raises(ValueError, match='positive integer'):
        AdaptiveWeightCache(8, **{option: 0})


@pytest.mark.parametrize('seconds', [-.1, float('nan'), float('inf')])
def test_invalid_reload_observation_cannot_change_cache(seconds):
    cache = AdaptiveWeightCache(8)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        cache.record_load('x', seconds)
    assert not cache.freq and not cache.costs and not cache.load_trace


def test_disabled_cache_duplicate_and_oversized_entries():
    cache = AdaptiveWeightCache(0)
    assert not cache.put('disabled', object())  # No allocation inspection.
    cache.resize(8)
    value = access(cache, 'small')
    assert not cache.put('small', SimpleNamespace(nbytes=1))
    assert not cache.put('large', SimpleNamespace(nbytes=9))
    assert cache.get('small') is value
    assert not cache.put('empty', SimpleNamespace(nbytes=0))
    cache.resize(0)
    assert not cache.data and cache.bytes == 0


def test_deterministic_trace_comparison_exposes_specific_strengths_and_limits():
    hot = [('hot', i) for i in range(4)]
    scan = [(key, 8, .01) for _ in range(20) for key in hot]
    scan += [(('scan', i), 8, 1.) for i in range(100)]
    scan += [(key, 8, .01) for _ in range(4) for key in hot]
    scan_rows = replay(scan, 32, decay_interval=16)
    assert scan_rows['adaptive']['misses'] < scan_rows['lru']['misses']
    assert scan_rows['adaptive']['misses'] < scan_rows['cost']['misses']

    phase = [(('old', i), 8, 1.) for _ in range(50) for i in range(4)]
    phase += [(('new', i), 8, 1.) for _ in range(20) for i in range(4)]
    phase_rows = replay(phase, 32, decay_interval=16)
    assert phase_rows['adaptive']['misses'] < phase_rows['lfu']['misses']
    # Conservative admission has a real cost: LRU adapts immediately to a
    # brand-new phase, whereas adaptive needs evidence of repeated demand.
    assert phase_rows['lru']['misses'] < phase_rows['adaptive']['misses']
