"""Portable checks; Metal execution is explicit in tensorops_probe, never implicit here."""
import json
import numpy as np
import pytest
from research.glm53_gpu_v2.tensorops_probe import (
    ROWS, WIDTH, COLUMNS, MAX_PAYLOAD, accepted, compare, fixture_cases,
    masked_columns, partial_reference, prepare, validate_inputs,
)


@pytest.mark.parametrize('name,w,x', list(fixture_cases()))
def test_partial_groups_match_independent_integer_matrix_product(name, w, x):
    masked = masked_columns(x)
    assert masked.dtype == np.dtype('int8')
    assert masked.shape == (32, 8)
    # Int64 NumPy matmul has a different grouping from the scalar Python oracle.
    expected = w.astype(np.int64) @ masked.astype(np.int64)
    np.testing.assert_array_equal(partial_reference(w, x), expected)
    assert np.count_nonzero(masked[:, 4:]) == 0
    np.testing.assert_array_equal(masked.sum(axis=1, dtype=np.int64), x.astype(np.int64))


def test_fixed_contract_rejects_wrong_shapes_and_types():
    _, w, x = next(fixture_cases())
    for bad in (w.astype(np.float32), w[:, :16], w[:, ::-1], w.tolist()):
        with pytest.raises(ValueError):
            validate_inputs(bad, x)
    for bad in (x.astype(np.int16), x[:16], x[::-1], x.tolist()):
        with pytest.raises(ValueError):
            validate_inputs(w, bad)


def test_reference_keeps_four_partials_and_signed_extremes():
    w = np.full((ROWS, WIDTH), -128, dtype=np.int8)
    x = np.full(WIDTH, -128, dtype=np.int8)
    ref = partial_reference(w, x)
    np.testing.assert_array_equal(ref[:, :4], np.full((ROWS, 4), 131072, dtype=np.int32))
    assert (ref[:, 4:] == 0).all()
    assert ref.sum(axis=1)[0] == 524288
    assert compare(ref.copy(), ref)['passed']
    bad = ref.copy()
    bad[0, 1] += 1
    assert not compare(bad, ref)['passed']
    assert not compare(ref[:, :1], ref)['passed']
    assert not compare(ref.astype(np.float32), ref)['passed']


def test_preparation_is_bounded_and_contains_exact_payload(tmp_path):
    manifest, refs, charged = prepare(tmp_path)
    assert charged <= MAX_PAYLOAD
    job = json.loads(manifest.read_text())
    assert len(job['cases']) == 3
    for case in job['cases']:
        from pathlib import Path
        assert Path(case['weights']).stat().st_size == ROWS * WIDTH
        assert Path(case['masked']).stat().st_size == WIDTH * COLUMNS
        assert refs[case['name']].shape == (ROWS, COLUMNS)
        assert not Path(case['output']).exists()


def test_acceptance_requires_actual_execution_and_complete_exact_results():
    report = {'mode': 'execute', 'host_compiled': True, 'process_returncode': 0, 'guard_passed': True,
              'native': {'shader_library_compiled': True, 'pipeline_compiled': True,
                         'device_execution_attempted': True, 'commands_completed': 3},
              'comparisons': {name: {'passed': True, 'elements': 128} for name, _, _ in fixture_cases()}}
    assert accepted(report)
    for key in ('host_compiled', 'guard_passed'):
        bad = json.loads(json.dumps(report)); bad[key] = False
        assert not accepted(bad)
    for key in ('shader_library_compiled', 'pipeline_compiled', 'device_execution_attempted'):
        bad = json.loads(json.dumps(report)); bad['native'][key] = False
        assert not accepted(bad)
    for mode in ('compile', 'prepare'):
        assert not accepted({**report, 'mode': mode})
    bad = json.loads(json.dumps(report)); del bad['comparisons']['zeros']
    assert not accepted(bad)
    bad = json.loads(json.dumps(report)); bad['comparisons']['zeros']['passed'] = False
    assert not accepted(bad)
    assert not accepted({})
