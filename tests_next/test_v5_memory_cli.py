"""Admission and peak-memory regressions specific to V5's shared budgets."""
from argparse import Namespace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from airllm.nextgen.attention import LatentKV
from airllm.nextgen.cli import config_for, main
from airllm.nextgen.config import NextConfig


@pytest.mark.parametrize('value', [True, -1, 1.5, '1024'])
def test_compact_partition_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match='compact_cache_bytes'):
        NextConfig(compact_cache_bytes=value)


def test_cache_tiers_share_budget_and_cli_partition_is_explicit():
    with pytest.raises(ValueError, match='inside cache_bytes'):
        NextConfig(cache_bytes=100, compact_cache_bytes=101)
    cfg = config_for(Namespace(command='prepare', profile='balanced', device='cpu',
        cache_mib=8, compact_cache_mib=3, output_tile_cache=True))
    assert cfg.cache_bytes == 8 * 1024**2
    assert cfg.compact_cache_bytes == 3 * 1024**2
    assert cfg.output_tile_cache is True


def test_numpy_source_reservation_is_not_reoffered_to_mlx():
    from airllm.nextgen.cli import combined_allocator_budget
    cfg = NextConfig(cache_bytes=100, compact_cache_bytes=40, io_bytes=20,
                     kv_bytes=30, workspace_bytes=40, reserve_bytes=50)
    assert cfg.mlx_memory_bytes == 150
    budget = combined_allocator_budget((cfg,), available_bytes=240)
    assert budget['memory_limit_bytes'] == 150
    assert budget['managed_payload_bytes_including_compact'] == 190
    with pytest.raises(MemoryError, match='Combined target and draft'):
        combined_allocator_budget((cfg,), available_bytes=239)


def test_glm_growth_rejects_overlap_and_preserves_committed_prefix():
    # Two layers fit at128 positions. Growing one requires both that old
    # layer's bank and its replacement, even though steady-state5120 fits.
    cache = LatentKV(mx, {}, budget=5120, limit=256, dtype=mx.float32)
    first = mx.ones((1, 128, 2))
    for layer in (0, 1):
        cache.append(layer, first, first)
    cache.position = 128
    assert cache.bytes == 4096
    old = cache.layers[0]
    with pytest.raises(MemoryError, match='during growth'):
        cache.append(0, first[:, :1], first[:, :1])
    assert cache.layers[0] is old
    assert cache.position == 128 and cache.bytes == 4096
    cache.budget = 8192
    cache.append(0, first[:, :1], first[:, :1])
    assert cache.peak_bytes == 8192
    assert cache.bytes == 6144
    assert cache.layers[0][0][0, :128].tolist() == first[0].tolist()


@pytest.mark.parametrize('arguments', [
    ['agent-serve', 'absent', '--state-dir', '/tmp/journal', '--port', '-1'],
    ['agent-serve', 'absent', '--state-dir', '/tmp/journal', '--token-ids', '1'],
    ['agent-serve', 'absent', '--state-dir', '/tmp/journal', '--quantum-tokens', '0'],
    ['preflight', 'absent', '--available-memory-mib', '-1'],
])
def test_v5_invalid_cli_never_loads_checkpoint(arguments):
    with pytest.raises(SystemExit) as result:
        main(arguments)
    assert result.value.code == 2


def test_cli_preflight_reports_supplied_target_separately(tiny_glm, capsys):
    import json
    root, _ = tiny_glm
    main(['preflight', str(root), '--tokens', '4', '--available-memory-mib', '16384'])
    report = json.loads(capsys.readouterr().out)
    assert report['preflight_passed']
    assert report['memory']['source'] == 'supplied scenario'
    assert report['memory']['available_bytes'] == 16 * 1024**3
    assert not report['generation_validated'] and not report['writes_performed']
