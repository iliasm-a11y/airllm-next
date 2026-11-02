"""Q8 CPU compatibility: independent native oracle, no GLM speed claim."""
import os
from pathlib import Path
import numpy as np
import pytest
from research.glm53_gpu_v2.q8_compatible import Q8CompatibleMatvec, validate_activation, validate_matrix
from research.glm53_gpu_v2.q8_oracle import Q8NativeOracle


def fixture(rows=13, width=96, seed=91):
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (rows, width // 32 * 34), dtype=np.uint8)
    blocks = packed.reshape(rows, -1, 34)
    scale = rng.uniform(0.001, 0.05, blocks.shape[:2]).astype('<f2')
    blocks[..., :2] = scale.view('u1').reshape(*scale.shape, 2)
    return packed


def test_payload_and_input_guards():
    packed = fixture()
    charged = validate_matrix(packed, 96, 1 << 20)
    with pytest.raises(MemoryError):
        validate_matrix(packed, 96, charged - 1)
    with pytest.raises(ValueError):
        validate_matrix(packed[:, ::2], 96, 1 << 20)
    for width in (True, 0, 33, 65568):
        with pytest.raises(ValueError):
            validate_matrix(packed, width, 1 << 20)
    for value in (np.full(96, np.inf, 'f4'), np.full(96, 1e-40, 'f4'), np.full(96, 1e20, 'f4')):
        with pytest.raises(ValueError):
            validate_activation(value, 96)


@pytest.mark.parametrize('width', [32, 64, 96, 256, 512, 4096])
def test_gpu_q8_against_native_arm(width):
    directory = os.environ.get('AIRLLM_GGML_LIBRARIES')
    if not directory:
        pytest.skip('set AIRLLM_GGML_LIBRARIES for independent native oracle')
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    native = Q8NativeOracle(directory)
    packed = fixture(width=width)
    rng = np.random.default_rng(5305)
    inputs = [rng.normal(size=width).astype('f4'), np.zeros(width, 'f4')]
    tied = np.zeros(width, 'f4')
    tied[:6] = np.array([127, -127, 0.5, 1.5, 2.5, -2.5], 'f4')
    inputs.append(tied)
    with Q8CompatibleMatvec(packed, width) as matrix:
        for x in inputs:
            qs, scales = matrix.quantize_activations(x)
            cpu_activation = native.activation(x)
            np.testing.assert_array_equal(np.asarray(qs).view('u1').reshape(-1, 32), cpu_activation[:, 2:])
            np.testing.assert_array_equal(np.asarray(scales).view('u1').reshape(-1, 2), cpu_activation[:, :2])
            actual = np.asarray(matrix(x))
            expected = native.matvec(packed, x)
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual.view('u4'), expected.view('u4'))
            fused = np.asarray(matrix(x, fused=True))
            np.testing.assert_array_equal(fused.view('u4'), expected.view('u4'))
            if width == 4096:
                shared = np.asarray(matrix(x, shared=True))
                np.testing.assert_array_equal(shared.view('u4'), expected.view('u4'))
    with pytest.raises(RuntimeError):
        matrix(inputs[0])
