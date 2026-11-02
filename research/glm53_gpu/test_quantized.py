"""Portable admission tests and optional independent native/Metal oracle tests."""
import os
from pathlib import Path
import numpy as np
import pytest
from research.glm53_gpu.quantized import Q2KMetalMatvec, dequantize_reference, validate_shapes
from research.glm53_gpu.quantized_probe import NativeOracle, read_selected_rows, compare, evaluate_acceptance


def fixture(rows=3, width=512, seed=5):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (rows, width // 256 * 84), dtype=np.uint8)
    blocks = packed.reshape(rows, -1, 84)
    scales = rng.uniform(0.0001, 0.1, (rows, width // 256, 2)).astype('<f2')
    blocks[..., 80:84] = scales.view('u1').reshape(rows, width // 256, 4)
    return packed


def test_admission_before_import():
    packed = fixture()
    required = validate_shapes(packed, 512, 1 << 20)
    assert required < 1 << 20
    with pytest.raises(MemoryError):
        validate_shapes(packed, 512, required - 1)
    for width in [0, 255, 257, 65537, True]:
        with pytest.raises(ValueError):
            validate_shapes(packed, width, 1 << 20)
    with pytest.raises(ValueError):
        validate_shapes(packed[:, ::2], 256, 1 << 20)
    packed[0, 80:82] = np.frombuffer(np.float16(np.inf).tobytes(), 'u1')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_shapes(packed, 512, 1 << 20)


def test_literal_layout_known_groups():
    p = np.zeros((1, 84), 'u1')
    p[0, :16] = np.arange(16) | (np.arange(16) << 4)
    p[0, 16:80] = 0b11100100
    p[0, 80:84] = np.frombuffer(np.array([0.5, 0.25], '<f2').tobytes(), 'u1')
    decoded = dequantize_reference(p, 256)[0]
    # Bytes encode groups in two 128-value halves and four 2-bit planes.
    assert decoded[0] == 0
    assert decoded[16] == -0.25
    assert decoded[32] == 0.5
    assert decoded[64] == 3
    assert decoded[128] == -2
    assert decoded[224] == 17.5


def oracle():
    directory = os.environ.get('AIRLLM_GGML_LIBRARIES')
    if not directory:
        pytest.skip('set AIRLLM_GGML_LIBRARIES for independent native oracle')
    return NativeOracle(directory)


def test_decode_against_native():
    native = oracle()
    for width in [256, 512, 4096]:
        p = fixture(rows=5, width=width)
        assert np.array_equal(dequantize_reference(p, width), native.dequantize(p, width))


@pytest.mark.parametrize('width', [256, 512, 4096])
def test_gpu_against_native_q8k_and_dot(width):
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    native = oracle()
    rng = np.random.default_rng(13)
    packed = fixture(rows=7, width=width)
    inputs = [rng.normal(size=width).astype('f4'), np.zeros(width, 'f4')]
    tied = np.zeros(width, 'f4')
    tied[0], tied[1] = -2, 2  # GGML chooses first equal-absolute maximum.
    tied[2:10] = np.arange(-4, 4, dtype='f4') / 127
    inputs.append(tied)
    with Q2KMetalMatvec(packed, width) as matrix:
        for x in inputs:
            q, scale = matrix.quantize_activations(x)
            mx.eval(q, scale)
            _, qref, sref = native.activation(x)
            assert np.array_equal(np.asarray(q), qref)
            assert np.array_equal(np.asarray(scale).view('u4'), sref.view('u4'))
            actual = np.asarray(matrix(x))
            expected = native.matvec(packed, x)
            # CPU ARM FMA order is represented explicitly by this mode.
            np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'))
            floating = np.asarray(matrix(x, mode='float'))
            np.testing.assert_allclose(floating, native.dequantize(packed, width) @ x,
                                       rtol=2e-5, atol=2e-5)
        with pytest.raises(ValueError):
            matrix(inputs[0], synchronize=False)
        with pytest.raises(ValueError):
            matrix(np.full(width, np.nan, 'f4'))
    with pytest.raises(RuntimeError):
        matrix(inputs[0])


def test_reader_rejects_non_gguf(tmp_path):
    p = tmp_path / 'wrong.gguf'
    p.write_bytes(b'other file')
    with pytest.raises(ValueError):
        read_selected_rows(p, 'x', 0, 0, 1)


def test_acceptance_rejects_wrong_operator_and_nonfinite():
    def valid():
        return {'q8_values_exact': True, 'q8_scales': {'finite': True, 'bit_identical': True},
                'decode_vs_native': {'finite': True, 'bit_identical': True},
                'cpu_compatible_vs_native_arm_dot': {'finite': True, 'bit_identical': True}}
    assert evaluate_acceptance(valid())
    for key in ['q8_scales', 'decode_vs_native', 'cpu_compatible_vs_native_arm_dot']:
        for field in ['finite', 'bit_identical']:
            value = valid()
            value[key][field] = False
            assert not evaluate_acceptance(value)
    value = valid()
    del value['cpu_compatible_vs_native_arm_dot']
    assert not evaluate_acceptance(value)
    value = valid()
    value['native_matrix_validation'] = {'10': {'finite': True, 'bit_identical': False}}
    assert not evaluate_acceptance(value)
    assert not compare(np.array([np.nan], 'f4'), np.array([np.nan], 'f4'))['finite']
    with pytest.raises(ValueError):
        compare(np.zeros(2, 'f4'), np.zeros((2, 1), 'f4'))
