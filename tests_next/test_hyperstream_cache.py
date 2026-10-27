"""Demand-history bounds under sustained streamed-weight cache churn."""
from types import SimpleNamespace

import pytest

from airllm.nextgen.cache import WeightCache


@pytest.mark.parametrize('policy', ['lru', 'slru', 'lfu', 'cost'])
def test_even_resident_scan_bounds_observations_across_multiple_prune_periods(policy):
    cache = WeightCache(2, policy)
    payload = SimpleNamespace(nbytes=1)
    for number in range(32_769):
        key = f'expert-tile-{number}'
        assert cache.get(key, 1) is None
        cache.record_load(key, .001)
        assert cache.put(key, payload)
        if number % 257 == 0:
            assert len(cache.freq) <= len(cache.data) + 8192
            assert len(cache.costs) <= len(cache.data) + 8192
    assert len(cache.data) == 2
    assert cache.bytes == 2
    assert cache.stats['misses'] == 32_769
    assert cache.stats['evictions'] == 32_767
    assert len(cache.freq) <= len(cache.data) + 8192
    assert len(cache.costs) <= len(cache.data) + 8192
    assert set(cache.data) <= set(cache.freq)
    assert set(cache.data) <= set(cache.costs)
    assert len(cache.trace) == len(cache.cost_trace) == len(cache.load_trace) == 4096


@pytest.mark.parametrize('policy', ['lru', 'slru', 'lfu', 'cost'])
def test_current_miss_keeps_demand_and_load_cost_at_prune_boundary(policy):
    cache = WeightCache(2, policy)
    for _ in range(8191):
        assert cache.get('nonresident') is None
    assert cache.get('current', 1) is None
    cache.record_load('current', .2)
    assert cache.freq['current'] == 1
    assert 'nonresident' not in cache.freq
    payload = SimpleNamespace(nbytes=1)
    assert cache.put('current', payload)
    assert cache.get('current', 1) is payload
    assert cache.costs['current'] == .2
    if policy == 'cost':
        assert cache.priorities['current'] > 0


@pytest.mark.parametrize('policy', ['lru', 'slru', 'lfu', 'cost'])
def test_pressure_clears_do_not_restart_history_countdown(policy):
    cache = WeightCache(2, policy)
    payload = SimpleNamespace(nbytes=1)
    for number in range(20_000):
        key = f'tile-{number}'
        cache.get(key, 1)
        cache.record_load(key, .001)
        cache.put(key, payload)
        if number % 101 == 0:
            cache.clear()
    assert len(cache.freq) <= len(cache.data) + 8192
    assert len(cache.costs) <= len(cache.data) + 8192


@pytest.mark.parametrize('policy', ['lru', 'slru', 'lfu', 'cost'])
def test_replay_retains_reused_resident_and_original_payload(policy):
    cache = WeightCache(2, policy)
    hot = SimpleNamespace(nbytes=1)
    cold = SimpleNamespace(nbytes=1)
    for key, payload, cost in [('hot', hot, .2), ('cold', cold, .001)]:
        assert cache.get(key, 1) is None
        cache.record_load(key, cost)
        assert cache.put(key, payload)
    assert cache.get('hot', 1) is hot
    assert cache.get('next', 1) is None
    cache.record_load('next', .001)
    assert cache.put('next', cold)
    assert cache.contains('hot')
    assert not cache.contains('cold')
    assert cache.bytes == 2
    assert cache.get('hot', 1) is hot
