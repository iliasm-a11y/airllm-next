"""Protocol regressions: honest cache labels and enforced numerical equivalence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('io_wall_benchmark',
    Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_io_wall.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_randomized_rounds_cover_identical_variants_and_pair_baseline():
    variants = ['reference', 'adaptive', 'all']
    jobs = benchmark.ordered_jobs(variants, 3, 521, '/read-only/baseline')
    assert jobs == benchmark.ordered_jobs(variants, 3, 521, '/read-only/baseline')
    for repeat in range(3):
        assert sorted(name for round_index, name in jobs if round_index == repeat) == sorted(variants + ['baseline'])
    for invalid in ([], ['reference', 'reference'], ['unknown']):
        with pytest.raises(ValueError): benchmark.ordered_jobs(invalid, 1, 521)


def test_first_prompt_only_is_software_cold_and_all_policies_share_budgets():
    assert benchmark.cache_state(0, 0) == 'process_cold_software_weights'
    for pair in ((0, 1), (1, 0), (1, 1)):
        assert benchmark.cache_state(*pair) == 'reused_software_weights_os_cache_uncontrolled'
    for name in (*benchmark.VARIANTS, *benchmark.OPTIONAL_VARIANTS, 'baseline'):
        options = benchmark.variant_options(name, 4096)
        assert not any(key.endswith('_bytes') for key in options if key != 'spec_prefetch_bytes')
        assert options['spec_prefetch_bytes'] <= 4096


def test_nocache_is_explicit_and_preserves_equal_memory_envelopes():
    plain = benchmark.variant_options('reference', 4096)
    controlled = benchmark.variant_options('pread_nocache', 4096)
    assert plain['io_mode'] == 'pread'
    assert controlled == plain | {'io_mode': 'pread_nocache'}
    assert benchmark.variant_options('adaptive_pipeline_nocache', 4096) == (
        benchmark.variant_options('adaptive_pipeline', 4096) | {'io_mode': 'pread_nocache'})


def reports(tmp_path, values, output_ids=None):
    reference = tmp_path / 'reference.npz'
    observed = tmp_path / 'observed.npz'
    np.savez(reference, repeat_0_prompt_0=np.array([[1., 2., 3.]], dtype='f4'))
    np.savez(observed, repeat_0_prompt_0=np.array(values, dtype='f4'))
    case = {'case': 'repeat_0_prompt_0', 'output_ids': [1, 2, 3]}
    return ([{'variant': 'reference', 'artifact': 'reference', 'rows': [case]},
             {'variant': 'all', 'artifact': 'all', 'rows': [case | {'output_ids': output_ids or [1, 2, 3]}]}],
            {'reference': reference, 'all': observed})


def test_comparison_checks_logits_and_greedy_tokens_independently(tmp_path):
    result = benchmark.compare_reports(*reports(tmp_path, [[1., 2., 3.]]))
    assert all(row['identical_logits'] and row['same_greedy_sequences'] for row in result)
    with pytest.raises(AssertionError, match='equivalence'):
        benchmark.compare_reports(*reports(tmp_path, [[1., 2., 3.]], [1, 2, 4]))
    with pytest.raises(AssertionError, match='equivalence'):
        benchmark.compare_reports(*reports(tmp_path, [[1., 2., 3.1]]))
    with pytest.raises(AssertionError, match='nonfinite'):
        benchmark.compare_reports(*reports(tmp_path, [[1., 2., float('nan')]]))
    with pytest.raises(AssertionError, match='shape'):
        benchmark.compare_reports(*reports(tmp_path, [[1., 2.]]))


def test_bounded_storage_probe_proves_read_coalescing_without_ssd_claim(tmp_path):
    report = benchmark.storage_comparison(tmp_path / 'probe', repeats=1, seed=521)
    assert 'no model' in report['scope'].lower()
    assert report['input_tensor_bytes'] <= benchmark.MiB
    assert report['summary']['coalesced_pread']['median_read_calls'] < report['summary']['separate_pread']['median_read_calls']
    assert all(row['payload_sha256_verified'] for row in report['raw_measurements'])
