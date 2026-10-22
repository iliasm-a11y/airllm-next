"""Benchmark correctness, scope and byte-budget regressions; no timing assertions."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('benchmark_ooc', Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_ooc.py')
_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_benchmark)


def test_bounded_comparison_checks_equal_payloads(tmp_path):
    root = tmp_path / 'fixture'
    _benchmark.make_fixture(root, tensors=12, tensor_bytes=4096)
    result = _benchmark.run_comparison(root, repeats=1, group_size=4,
        max_bytes=48 * 1024, max_read_bytes=16 * 1024)
    assert result['inference_performed'] is False
    assert result['physical_ssd_bandwidth_certified'] is False
    assert len(result['raw_measurements']) == 4
    assert all(row['payload_sha256_verified'] for row in result['raw_measurements'])
    assert result['summary']['separate_pread']['median_read_calls'] == 12
    assert result['summary']['coalesced_pread']['median_read_calls'] == 3


def test_transfer_limits_are_explicit_theory():
    bounds = _benchmark.transfer_bounds(1e12, 4, 50e9, 1e9, accepted_per_pass=2)
    assert bounds['parameter_payload_bytes_excluding_scales'] == 500e9
    assert bounds['optimistic_weight_transfer_seconds_per_accepted_token'] == 12.5
    assert 'theoretical' in bounds['scope']
    with pytest.raises(ValueError):
        _benchmark.transfer_bounds(1e12, 4, 2e12, 1e9)


def test_readonly_probe_respects_transfer_cap(tmp_path):
    path = tmp_path / 'bytes'
    before = bytes(range(256)) * 64
    path.write_bytes(before)
    result = _benchmark.probe_storage(path, seconds=1, chunk_bytes=4096, max_bytes=8192)
    assert result['logical_bytes'] == 8192
    assert result['physical_ssd_bandwidth_certified'] is False
    assert path.read_bytes() == before


def test_invalid_probe_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _benchmark.probe_storage(tmp_path / 'missing', seconds=0)
