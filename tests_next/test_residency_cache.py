"""Equal-budget residency policies; these tests do not measure hardware speed."""
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.residency_cache import ResidencyWeightCache, cache_key_class
from airllm.nextgen.weights import NativeWeights


def request(cache, key, nbytes=8, seconds=.01):
    value = cache.get(key, nbytes)
    if value is None:
        cache.record_load(key, seconds)
        value = SimpleNamespace(nbytes=nbytes)
        cache.put(key, value)
    assert cache.bytes == sum(cache.sizes.values()) <= cache.capacity
    return value


@pytest.mark.parametrize('key,kind', [
    ('model.layers.0.self_attn.q_proj.weight', 'shared'),
    ('model.layers.0.block_sparse_moe.router.layer.weight', 'shared'),
    ('model.norm.weight', 'shared'),
    ('model.layers.0.block_sparse_moe.input_linear.weight#4', 'expert'),
    ('model.layers.1.mlp.experts.12.down_proj.weight', 'expert'),
    (('model.layers.0.block_sparse_moe.input_linear.weight', 4), 'expert'),
    (('model.layers.0.block_sparse_moe.input_linear.weight', np.int64(4)), 'expert'),
    (('output', 'model.embed_tokens.weight', 0, 2048), 'output'),
    (('output', 'lm_head.weight', 2048, 4096), 'output'),
    (('embedding', 4, 5), 'shared'),
    (('norm', True), 'shared'),
])
def test_reader_key_classification_distinguishes_expert_ids_from_output_rows(key, kind):
    assert cache_key_class(key) == kind


@pytest.mark.parametrize('policy,preferred,other', [
    ('shared', 'model.norm.weight', ('experts', 1)),
    ('shared', ('output', 'lm_head.weight', 0, 8), ('experts', 1)),
    ('expert', ('experts', 1), 'model.layers.0.self_attn.q_proj.weight'),
])
def test_preferred_class_resists_scan_without_extra_budget(policy, preferred, other):
    cache = ResidencyWeightCache(8, policy)
    exact = request(cache, preferred)
    for _ in range(10):
        request(cache, other)
    assert cache.get(preferred) is exact
    assert not cache.contains(other)
    assert cache.peak_bytes == cache.capacity == 8
    assert cache.stats['class_rejected'] == 10


def test_shared_policy_can_displace_expert_and_freed_space_is_reusable():
    cache = ResidencyWeightCache(16, 'shared')
    request(cache, ('expert', 0))
    request(cache, ('expert', 1))
    request(cache, 'norm', nbytes=16)
    assert list(cache.data) == ['norm']
    assert cache.bytes == cache.peak_bytes == 16
    assert cache.stats['expert_evictions'] == 2
    cache.resize(4)
    assert cache.bytes == 0
    assert request(cache, ('expert', 3), nbytes=4) is cache.get(('expert', 3))


def test_hybrid_needs_reuse_and_can_migrate_full_budget_between_classes():
    cache = ResidencyWeightCache(8, 'hybrid')
    for _ in range(4):
        request(cache, 'shared', seconds=.01)
    request(cache, ('expert', 0), seconds=1.)
    assert cache.contains('shared')
    request(cache, ('expert', 0), seconds=1.)
    assert cache.contains(('expert', 0))
    assert not cache.contains('shared')
    assert cache.bytes == 8


def test_batch_union_priority_is_finite_and_multiplicity_is_observed():
    cache = ResidencyWeightCache(16, 'batch-aware')
    request(cache, 'norm')
    request(cache, ('expert', 0))
    cache.set_batch(key for key in [('expert', 1), ('expert', 1), ('expert', 2)])
    request(cache, ('expert', 1))
    assert cache.contains('norm') and cache.contains(('expert', 1))
    request(cache, ('expert', 9))
    assert not cache.contains(('expert', 9))
    assert cache.batch_counts == {('expert', 1): 2, ('expert', 2): 1}
    assert cache.stats['batch_union_peak'] == 2
    assert cache.bytes == cache.peak_bytes == cache.capacity
    cache.clear_batch()
    request(cache, 'another_shared')
    assert cache.contains('another_shared') and not cache.contains(('expert', 1))
    cache.clear()
    assert not cache.batch_counts and cache.bytes == 0


def test_batch_declaration_failure_is_atomic_and_history_remains_bounded():
    cache = ResidencyWeightCache(8, 'batch-aware', max_metadata_entries=2, max_entries=2)
    cache.set_batch([('expert', 0)])
    with pytest.raises(ValueError, match='metadata limit'):
        cache.set_batch(('expert', i) for i in range(3))
    assert cache.batch_counts == {('expert', 0): 1}
    with pytest.raises(ValueError, match='expert identities'):
        cache.set_batch([('output', 'lm_head.weight', 0, 8)])
    for i in range(100):
        request(cache, ('expert', i))
        assert len(cache.freq) <= cache.max_entries + cache.max_metadata_entries
        assert len(cache.batch_counts) <= cache.max_metadata_entries


@pytest.mark.parametrize('policy', ['shared', 'expert', 'hybrid', 'batch-aware'])
def test_retained_numpy_owner_and_all_policy_budgets(policy):
    cache = ResidencyWeightCache(16, policy)
    owner = np.arange(32, dtype=np.uint8)
    assert not cache.put('small_view_large_owner', owner[:1])
    for i in range(30):
        request(cache, ('tensor', i), nbytes=5)
    cache.resize(5)
    assert cache.bytes <= 5
    cache.resize(0)
    assert cache.bytes == 0
    assert not cache.put('disabled', object())


@pytest.mark.parametrize('policy', ['shared', 'expert', 'hybrid', 'batch-aware'])
def test_native_reader_shares_original_decoded_and_compact_allowance(policy):
    cfg = NextConfig(policy=policy, dtype='float32', cache_bytes=16,
                     compact_cache_bytes=4, io_bytes=8)
    reader = NativeWeights(SimpleNamespace(), np, cfg, {})
    try:
        assert reader.cache.capacity == 12
        assert reader.compact_cache.capacity == 4
        assert reader.cache.capacity + reader.compact_cache.capacity == cfg.cache_bytes
        assert cfg.mlx_memory_bytes == 12 + cfg.io_bytes + cfg.kv_bytes + cfg.workspace_bytes
        reader.observe_batch((('bank', int(e)) for e in [1, 1, 3]))
        if policy == 'batch-aware':
            assert reader.cache.batch_counts == {'bank#1': 2, 'bank#3': 1}
        reader.clear_batch()
        if policy == 'batch-aware':
            assert not reader.cache.batch_counts
        reader.trim()
        assert reader.cache.capacity + reader.compact_cache.capacity == 8
    finally:
        reader.close()


@pytest.mark.parametrize('policy', ['shared', 'expert', 'hybrid', 'batch-aware'])
def test_cli_accepts_explicit_residency_policy_and_disables_saved_tuning(policy, monkeypatch):
    from airllm.nextgen import cli
    class ParsedOnly(Exception):
        pass
    captured = {}
    original = cli.config_for
    monkeypatch.setattr(cli, 'auto_config', lambda profile, **values: NextConfig(profile=profile, **values))
    def parsed(args):
        captured['config'] = original(args)
        raise ParsedOnly
    monkeypatch.setattr(cli, 'config_for', parsed)
    with pytest.raises(ParsedOnly):
        cli.main(['run', '/unused-local-checkpoint', '--token-ids', '1', '--device', 'cpu',
                  '--cache-mib', '16', '--cache-policy', policy])
    assert captured['config'].policy == policy
    assert captured['config'].cache_bytes == 16*1024**2
    assert captured['config'].use_saved_tuning is False
