"""Exact logical interval accounting, bounded metadata, and disabled fast path."""
from concurrent.futures import ThreadPoolExecutor
import json
import random
import struct

import numpy as np
import pytest

from airllm.nextgen.io_profile import IOProfiler
from airllm.nextgen.storage import OffsetCheckpoint


def checkpoint(tmp_path, **kwargs):
    root = tmp_path / 'cp'
    root.mkdir()
    key = 'model.layers.2.mlp.experts.3.weight'
    values = np.arange(6 * 8, dtype='<f4').reshape(6, 8)
    raw = json.dumps({key: {'dtype': 'F32', 'shape': list(values.shape),
                             'data_offsets': [0, values.nbytes]}}).encode()
    raw += b' ' * (-len(raw) % 8)
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + values.tobytes())
    return OffsetCheckpoint(root, max_read_bytes=1024, **kwargs), key, values


@pytest.mark.parametrize('mode', ['pread', 'mmap'])
def test_interval_rereads_overlap_and_attribution_are_actual(tmp_path, mode):
    cp, key, values = checkpoint(tmp_path, profile_io=True)
    with cp:
        np.testing.assert_array_equal(cp.read_array(key, rows=(0, 3), mode=mode), values[:3])
        np.testing.assert_array_equal(cp.read_array(key, rows=(2, 5), mode=mode), values[2:5])
        profile = cp.io_profile()
        assert profile['requested_bytes'] == profile['read_bytes'] == 192
        for stream in ('requested_intervals', 'read_intervals'):
            assert profile[stream]['unique_bytes'] == 160
            assert profile[stream]['reread_bytes'] == 32
            assert not profile[stream]['saturated']
        group = profile['by_layer_expert']['layer=2/expert=3']
        assert group == {'request_count': 2, 'requested_bytes': 192,
                         'read_calls': 2, 'read_bytes': 192}
        assert profile['modes'][mode]['read_bytes'] == 192
        assert not profile['physical_ssd_bandwidth_certified']


def test_coalesced_gap_and_duplicates_do_not_multiply_read_bytes(tmp_path):
    cp, key, _ = checkpoint(tmp_path, profile_io=True)
    with cp:
        cp.read_arrays([{'key': key, 'rows': (0, 2)},
                        {'key': key, 'rows': (0, 2)},
                        {'key': key, 'rows': (3, 5)}], max_gap_bytes=32)
        profile = cp.io_profile()
        assert profile['requested_bytes'] == 192
        assert profile['read_bytes'] == 160
        assert profile['requested_intervals']['unique_bytes'] == 128
        assert profile['requested_intervals']['reread_bytes'] == 64
        assert profile['read_intervals']['unique_bytes'] == 160
        assert profile['unattributed_read_bytes'] == 32
        assert profile['by_layer_expert']['layer=2/expert=3']['read_bytes'] == 128
        assert profile['read_calls'] == 1


def test_preflight_failure_does_not_register_requests(tmp_path):
    cp, key, _ = checkpoint(tmp_path, profile_io=True)
    with cp:
        with pytest.raises(MemoryError):
            cp.read_arrays([key], max_batch_bytes=1)
        with pytest.raises(ValueError):
            cp.read_array(key, mode='invalid')
        assert cp.io_profile()['requested_bytes'] == 0
        assert cp.io_profile()['read_calls'] == 0


def test_bounded_union_saturation_reports_null_not_false_precision():
    profile = IOProfiler(max_intervals=2, max_groups=2, physical_counter=lambda: None)
    for i in (0, 10, 20, 0, 20):
        regions = profile.requests([('file', i, 4, f'layers.{i}.experts.{i}.w', None)])
        profile.record_read('file', i, 4, .01, 'pread', regions)
    result = profile.snapshot()
    assert result['requested_bytes'] == result['read_bytes'] == 20
    for stream in ('requested_intervals', 'read_intervals'):
        assert result[stream]['unique_bytes'] is None
        assert result[stream]['reread_bytes'] is None
        assert result[stream]['unique_bytes_lower_bound'] == 8
        assert result[stream]['reread_bytes_lower_bound'] == 4
        assert result[stream]['interval_count'] == 2
    assert len(result['by_layer_expert']) == 2
    assert result['group_limit_reached']
    assert sum(row['read_bytes'] for row in result['by_layer_expert'].values()) == 20
    assert result['physical_process_read_bytes'] is None


def test_default_does_not_construct_profiler_or_sample_physical_counter(tmp_path, monkeypatch):
    import airllm.nextgen.storage as storage
    monkeypatch.setattr(storage, 'physical_read_bytes', lambda: pytest.fail('disabled profiler sampled OS'))
    cp, key, _ = checkpoint(tmp_path)
    with cp:
        assert cp.io_profile() is None
        cp.read_array(key)
        cp.read_arrays([key])


def test_partial_failed_read_counts_actual_copied_bytes(tmp_path):
    cp, key, _ = checkpoint(tmp_path, profile_io=True)
    with cp:
        # A checkpoint can be truncated after header validation. Distinguish the
        # attempted selection from the successfully copied prefix before EOF.
        descriptor = cp.tensors[key]
        with (cp.root / descriptor['file']).open('r+b') as file:
            file.truncate(descriptor['offset'] + 32)
        with pytest.raises(EOFError):
            cp.read_array(key)
        profile = cp.io_profile()
        assert profile['requested_bytes'] == 192
        assert profile['read_bytes'] == 32
        assert profile['failed_read_calls'] == 1


def test_concurrent_profile_totals_and_detached_snapshot(tmp_path):
    cp, key, _ = checkpoint(tmp_path, profile_io=True)
    with cp, ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: cp.read_array(key), range(24)))
        profile = cp.io_profile()
        assert profile['read_bytes'] == 24 * 192
        assert profile['read_intervals']['unique_bytes'] == 192
        assert profile['read_intervals']['reread_bytes'] == 23 * 192
        profile['by_layer_expert'].clear()
        assert cp.io_profile()['by_layer_expert']


def test_packed_expert_selector_has_finite_layer_group(tmp_path):
    profile = IOProfiler(max_groups=4, physical_counter=lambda: 15)
    profile.requests([('file', 0, 4, 'model.layers.7.mlp.experts.gate_up_proj', 11)])
    result = profile.snapshot()
    assert 'layer=7/expert=11' in result['by_layer_expert']
    assert result['physical_process_read_bytes'] == 0
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError): IOProfiler(max_intervals=bad)
        with pytest.raises(ValueError): IOProfiler(max_groups=bad)


def test_randomized_interval_unions_match_independent_byte_set_oracle():
    profile = IOProfiler(max_intervals=128)
    requested, read = set(), set()
    requested_count = read_count = 0
    rng = random.Random(521)
    for _ in range(200):
        filename = f'file{rng.randrange(3)}'
        start, size = rng.randrange(96), rng.randrange(33)
        regions = profile.requests([(filename, start, size, 'layers.0.w', None)])
        requested.update((filename, offset) for offset in range(start, start + size))
        requested_count += size
        # Deliberately include some short copies and some reads of gaps.
        copied = max(0, size + rng.randrange(-8, 9))
        profile.record_read(filename, start, copied, .001, 'pread', regions)
        read.update((filename, offset) for offset in range(start, start + copied))
        read_count += copied
    result = profile.snapshot()
    assert result['requested_intervals']['unique_bytes'] == len(requested)
    assert result['requested_intervals']['reread_bytes'] == requested_count - len(requested)
    assert result['read_intervals']['unique_bytes'] == len(read)
    assert result['read_intervals']['reread_bytes'] == read_count - len(read)
    assert sum(row['read_bytes'] for row in result['by_layer_expert'].values()) + result['unattributed_read_bytes'] == read_count
