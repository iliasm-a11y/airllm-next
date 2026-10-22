import json
import os
import struct
import time
import pytest
import torch
from safetensors.torch import save_file
from airllm.ultra.checkpoint import DirectCheckpoint
from airllm.ultra.cache import AdaptiveCache, BoundedPrefetch
from airllm.ultra.weights import dequantize_fp8, WeightReader


def test_random_access_single_file_and_bounded_handles(tmp_path):
    for i in range(5):
        save_file({f'model.layers.{i}.weight': torch.full((8, 8), float(i))},
                  tmp_path / f'shard{i}.safetensors')
    with DirectCheckpoint(tmp_path, max_open_files=2) as cp:
        assert cp.stats['tensor_reads'] == 0
        for i in range(5):
            k = f'model.layers.{i}.weight'
            assert cp.read([k])[k][0, 0].item() == i
            assert len(cp.handles) <= 2
        assert len(cp.keys('model.layers.1')) == 1
        assert len(cp.keys('model.layers.10')) == 0
    with pytest.raises(RuntimeError):
        cp.read(['model.layers.0.weight'])


def test_manifest_invalidates_on_weight_change(tmp_path):
    root = tmp_path / 'weights'
    root.mkdir()
    dest = root / 'model.safetensors'
    save_file({'w': torch.ones(4)}, dest)
    with DirectCheckpoint(root, tmp_path / 'meta') as cp:
        first = cp.fingerprint
    save_file({'w': torch.zeros(5)}, dest)
    with DirectCheckpoint(root, tmp_path / 'meta') as cp:
        assert cp.fingerprint != first
        assert cp.stats['manifest_hit'] == 0
        assert cp.read(['w'])['w'].numel() == 5


@pytest.mark.parametrize('value', [b'', struct.pack('<Q', 10**12)])
def test_corrupt_header_rejected(tmp_path, value):
    (tmp_path / 'model.safetensors').write_bytes(value)
    with pytest.raises(ValueError):
        DirectCheckpoint(tmp_path)


def test_traversal_index_rejected(tmp_path):
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': '../outside'}}))
    with pytest.raises(ValueError, match='path'):
        DirectCheckpoint(tmp_path)


def test_cache_budget_and_scan_resistance():
    cache = AdaptiveCache(3 * 64)
    state = {'w': torch.ones(16)}
    cache.put('hot', state)
    assert cache.get('hot') is state
    for i in range(100):
        cache.put(i, state)
        assert cache.bytes <= cache.capacity
    assert cache.get('hot') is state
    assert not cache.put('oversize', {'w': torch.ones(1000)})
    cache.resize(0)
    assert cache.bytes == 0


def test_prefetch_bound_dedup_and_exception_propagation():
    state = {'w': torch.ones(16)}
    scheduler = BoundedPrefetch(64, max_tasks=1)
    try:
        assert scheduler.submit('a', 64, lambda: state)
        assert not scheduler.submit('a', 64, lambda: state)
        assert not scheduler.submit('big', 65, lambda: state)
        assert scheduler.take('a') is state
        assert scheduler.bytes == 0
        def fail():
            raise IOError('test I/O failure')
        scheduler.submit('bad', 64, fail)
        with pytest.raises(IOError):
            scheduler.take('bad')
        assert scheduler.bytes == 0
    finally:
        scheduler.close()
    assert not scheduler.submit('a', 64, lambda: state)


def test_fp8_dequant_edges_and_reader(tmp_path):
    w = torch.randn(7, 11).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3) + 0.1
    expected = w.float() * scale.repeat_interleave(4, 0).repeat_interleave(4, 1)[:7, :11]
    torch.testing.assert_close(dequantize_fp8(w, scale, [4, 4], torch.float32), expected)
    save_file({'layer.weight': w, 'layer.weight_scale_inv': scale}, tmp_path / 'model.safetensors')
    with DirectCheckpoint(tmp_path) as cp:
        reader = WeightReader(cp, torch.float32, {'quant_method': 'fp8', 'weight_block_size': [4, 4]})
        got = reader.load(cp.keys())
        assert set(got) == {'layer.weight'}
        torch.testing.assert_close(got['layer.weight'], expected)


def test_prefetch_reservation_violation_rejected():
    with pytest.raises(RuntimeError, match='reserved'):
        scheduler = BoundedPrefetch(16)
        try:
            scheduler.submit('bad', 8, lambda: {'w': torch.ones(4)})
            scheduler.take('bad')
        finally:
            scheduler.close()


def test_close_releases_all_pending_results_after_worker_failure():
    scheduler = BoundedPrefetch(1024)
    def fail():
        raise OSError('worker failure during shutdown')
    scheduler.submit('bad', 64, fail)
    # Ensure the failure is actually running/completed, not merely cancellable.
    try:
        scheduler.pending['bad'][0].result()
    except OSError:
        pass
    scheduler.submit('good', 64, lambda: {'w': torch.ones(16)})
    with pytest.raises(OSError, match='shutdown'):
        scheduler.close()
    assert not scheduler.pending
    assert scheduler.bytes == 0
    assert scheduler.closed
