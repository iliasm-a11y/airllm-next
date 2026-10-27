"""Exercise real GGUFWeights transport plumbing with a CPU NumPy stand-in.

No claim about actual Metal computation follows from these tests.
"""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
from airllm.nextgen.gguf import GGUFMemoryError, PackedBufferBudget
from airllm.nextgen.gguf_qwen import GGUFWeights
from airllm.nextgen import gguf_metal
from test_grouped_packed_io import fixture


class FakeMX:
    float32 = np.float32
    metal = SimpleNamespace(is_available=lambda: True)
    def __init__(self):
        self.synchronizations = 0
    def synchronize(self):
        self.synchronizations += 1
    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)
    @staticmethod
    def eval(*args):
        pass


def weights(reader, **overrides):
    mx = FakeMX()
    options = SimpleNamespace(dtype="float32", policy="lru", compact_cache_bytes=0,
                              cache_bytes=65536, max_tensor_bytes=65536,
                              workspace_bytes=262144, row_chunk=8192,
                              reserve_bytes=0, zero_copy_weights=False)
    config = dict(linear_num_key_heads=1, linear_num_value_heads=1,
                  linear_value_head_dim=1, gguf_v_head_reordered=False)
    kwargs = dict(kernel="metal", packed_direct_io=True, expert_batch=True,
                  grouped_direct_io=True)
    kwargs.update(overrides)
    return GGUFWeights(reader, mx, options, config, **kwargs)


@pytest.mark.parametrize("options", [{"grouped_direct_io": 1},
                                     {"packed_direct_io": False},
                                     {"expert_batch": False},
                                     {"metal_batch_backend": "reuse", "kernel": "reference", "expert_batch": False},
                                     {"metal_token_tile": True},
                                     {"metal_token_tile": 3}])
def test_engine_refuses_incompatible_experimental_options(tmp_path, options):
    reader, _ = fixture(tmp_path)
    with reader:
        before = dict(reader.stats)
        with pytest.raises(ValueError):
            weights(reader, **options)
        assert reader.stats == before


def test_complete_group_path_uses_final_owner_and_explicitly_bypasses_tile_cache(tmp_path, monkeypatch):
    reader, raw = fixture(tmp_path)
    with reader:
        engine = weights(reader)
        monkeypatch.setattr(engine, "_packed_tile", lambda *a, **kw: pytest.fail("Per-expert scratch/caching used"))
        before = dict(reader.stats)
        packed = engine._pack_expert_rows("bank", 9, [0, 1, 3])
        np.testing.assert_array_equal(packed, raw[[0, 1, 3]])
        assert packed.base.flags.owndata
        assert engine.packed_budget.live_bytes == packed.nbytes
        assert engine.cache.bytes == 0
        assert reader.stats["read_calls"] - before["read_calls"] == 2
        assert engine.stats["grouped_direct_io_loads"] == 1
        assert engine.stats["grouped_direct_io_bytes"] == packed.nbytes
        del packed
        gc.collect()
        assert engine.packed_budget.live_bytes == 0


def test_live_group_leases_still_refuse_after_trim_and_device_drain(tmp_path, monkeypatch):
    reader, _ = fixture(tmp_path)
    with reader:
        engine = weights(reader)
        size = 2 * 9 * reader.tensors["bank"].row_bytes
        engine.packed_budget = PackedBufferBudget(size)
        held = engine._grouped_packed_tile("bank", 0, 9, [0, 1])
        before = dict(reader.stats)
        with pytest.raises(GGUFMemoryError):
            engine._grouped_packed_tile("bank", 0, 9, [2, 3])
        assert engine.mx.synchronizations == 1
        assert engine.packed_budget.live_bytes == size
        assert reader.stats["read_calls"] == before["read_calls"]
        assert reader._reserved == 0
        del held
        gc.collect()
        assert engine.packed_budget.live_bytes == 0


def test_grouped_linear_preserves_order_and_forwards_batch_dispatcher(tmp_path, monkeypatch):
    reader, raw = fixture(tmp_path)
    with reader:
        engine = weights(reader, metal_batch_backend="reuse", metal_token_tile=8)
        calls = []
        def multiply(packed, x, kind, **kwargs):
            np.testing.assert_array_equal(packed, raw[[0, 1, 3]])
            calls.append(kwargs)
            assert kind == 12
            return np.stack([np.full((x.shape[1], packed.shape[1]), index, dtype=np.float32)
                             for index in range(x.shape[0])])
        monkeypatch.setattr(gguf_metal, "packed_expert_matmul", multiply)
        x = np.zeros((3, 2, 512), dtype=np.float32)
        result = engine.linear_experts(x, "bank", [0, 1, 3])
        assert result.shape == (3, 2, 9)
        np.testing.assert_array_equal(result[:, 0, 0], [0, 1, 2])
        assert len(calls) == 1 and calls[0]["batch_dispatch"] is engine.batch_dispatch
        assert engine.batch_dispatch.token_tile == 8
        assert engine.stats["expert_batch_matmul_calls"] == 1
        gc.collect()
        assert engine.packed_budget.live_bytes == 0


def test_unchanged_default_does_not_construct_a_dispatcher_or_group_transport(tmp_path):
    reader, _ = fixture(tmp_path)
    with reader:
        engine = weights(reader, grouped_direct_io=False)
        assert engine.grouped_direct_io is False and engine.batch_dispatch is None
