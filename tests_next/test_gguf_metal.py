"""Independent scalar block layout versus fused packed Metal dot products."""
import numpy as np
import pytest

from airllm.nextgen.gguf_metal import packed_matmul


def scalar_decode(raw, kind, width):
    result = np.zeros((len(raw), width), dtype=np.float32)
    block_size = 144 if kind == 12 else 210
    for row in range(len(raw)):
        for b in range(width // 256):
            p = raw[row, b * block_size:(b + 1) * block_size]
            half = lambda pos: np.frombuffer(p[pos:pos + 2].tobytes(), dtype='<f2')[0].astype(np.float32)
            for k in range(256):
                if kind == 12:
                    g, l = divmod(k, 32)
                    sc = p[4:16].astype(int)
                    scale = sc[g] & 63 if g < 4 else (sc[g + 4] & 15) | ((sc[g - 4] >> 6) << 4)
                    minimum = sc[g + 4] & 63 if g < 4 else (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4)
                    byte = int(p[16 + g // 2 * 32 + l])
                    q = byte >> 4 if g % 2 else byte & 15
                    value = np.float32(half(0) * scale) * np.float32(q) - np.float32(half(2) * minimum)
                else:
                    h, part, lane = k // 128, (k % 128) // 32, k % 32
                    low = int(p[h * 64 + part % 2 * 32 + lane])
                    high = int(p[128 + h * 32 + lane])
                    q = ((low >> (part // 2 * 4)) & 15) | (((high >> (part * 2)) & 3) << 4)
                    scale = int(p[192 + k // 16])
                    if scale >= 128:
                        scale -= 256
                    value = np.float32(half(208) * scale) * np.float32(q - 32)
                result[row, b * 256 + k] = value
    return result


@pytest.mark.parametrize('kind', [12, 14])
@pytest.mark.parametrize('batch', [1, 3])
def test_fused_packed_dot_matches_independent_scalar(kind, batch):
    mx = pytest.importorskip('mlx.core')
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    rng = np.random.default_rng(142)
    width, rows = 768, 5
    block_size = 144 if kind == 12 else 210
    packed = rng.integers(0, 256, size=(rows, width // 256 * block_size), dtype=np.uint8)
    for b in range(width // 256):
        for pos in ([0, 2] if kind == 12 else [208]):
            packed[:, b * block_size + pos:b * block_size + pos + 2] = np.frombuffer(np.float16(.001).tobytes(), dtype=np.uint8)
    x = rng.normal(size=(batch, width)).astype(np.float32)
    expected = x @ scalar_decode(packed, kind, width).T
    with mx.stream(mx.new_stream(mx.gpu)):
        actual = np.asarray(packed_matmul(packed, mx.array(x), kind))
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-5)


def test_packed_dot_rejects_before_import():
    mx = pytest.importorskip('mlx.core')
    packed = np.zeros((2, 144), np.uint8)
    x = mx.zeros((1, 256))
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    with pytest.raises(MemoryError):
        packed_matmul(packed, x, 12, max_workspace_bytes=1)
    with pytest.raises(ValueError):
        packed_matmul(packed, mx.zeros((1, 255)), 12)
