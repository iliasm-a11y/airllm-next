"""Transport integration: exact packed Metal math, eviction and live owners."""
from dataclasses import replace
import gc

import mlx.core as mx
import numpy as np
import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.gguf import GGUFReader
from airllm.nextgen.gguf_metal import packed_matmul
from airllm.nextgen.gguf_qwen import GGUFWeights
from test_gguf import _fixture
from test_gguf_throughput import packed_fixture


@pytest.mark.parametrize('kind', [12, 14])
@pytest.mark.parametrize('transport', ['readonly', 'direct'])
@pytest.mark.parametrize('zero_copy', [False, True])
@pytest.mark.parametrize('policy', ['adaptive', 'lru'])
def test_transport_preserves_metal_results_under_cache_eviction(tmp_path, kind, transport, zero_copy, policy):
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    rng, packed = packed_fixture(kind, 4, 13, 512)
    name = 'blk.0.ffn_gate_exps.weight'
    path = _fixture(tmp_path / 'transport.gguf', [(name, (512, 13, 4), kind, packed.tobytes())])
    cfg = NextConfig(cache_bytes=2048, io_bytes=1 << 20, workspace_bytes=1 << 20,
                     kv_bytes=1 << 20, reserve_bytes=0, max_tensor_bytes=4096,
                     row_chunk=4, policy=policy, device='gpu', dtype='float32',
                     prefetch=False, zero_copy_weights=zero_copy)
    config = dict(linear_num_key_heads=2, linear_num_value_heads=4,
                  linear_value_head_dim=8, gguf_v_head_reordered=False)
    experts = [3, 0, 2]
    x = rng.normal(size=(3, 2, 512)).astype(np.float32)
    with GGUFReader(path, memory_budget_bytes=1 << 20, max_read_bytes=4096) as cp:
        weights = GGUFWeights(cp, mx, cfg, config, kernel='metal', expert_batch=True,
                              packed_copy_free=transport == 'readonly', packed_direct_io=transport == 'direct')
        with mx.stream(mx.new_stream(mx.gpu)):
            expected = np.stack([np.asarray(packed_matmul(packed[e], mx.array(x[i]), kind))
                                 for i, e in enumerate(experts)])
            for _ in range(3):
                individual = np.stack([np.asarray(weights.linear(mx.array(x[i]), name, e))
                                       for i, e in enumerate(experts)])
                grouped = np.asarray(weights.linear_experts(mx.array(x), name, experts))
                np.testing.assert_array_equal(individual, expected)
                np.testing.assert_array_equal(grouped, expected)
                assert weights.cache.bytes <= cfg.cache_bytes
                assert weights.packed_budget.live_bytes <= weights.packed_budget.capacity_bytes
            weights.close()
            mx.synchronize()
        gc.collect()
        assert weights.packed_budget.live_bytes == 0
        assert path.read_bytes().endswith(packed.tobytes())


def test_transport_flags_reject_invalid_combinations():
    cfg = NextConfig(reserve_bytes=0)
    for flag in (1, 'yes', None):
        with pytest.raises(ValueError, match='boolean'):
            GGUFWeights(None, mx, cfg, {}, packed_direct_io=flag)
    with pytest.raises(ValueError, match='one packed'):
        GGUFWeights(None, mx, cfg, {}, packed_direct_io=True, packed_copy_free=True)


def test_cli_rejects_two_host_transports_before_model_open(tmp_path):
    from airllm.nextgen.gguf_cli import main
    output = tmp_path / 'report.json'
    with pytest.raises(SystemExit) as error:
        main(['gguf-run', str(tmp_path / 'missing.gguf'), '--report', str(output),
              '--packed-direct-io', '--packed-copy-free'])
    assert error.value.code == 2
    assert not output.exists()
