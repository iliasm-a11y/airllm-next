"""GLM-5.3's distributed codecs, checked against independent GGML layouts.

The scalar oracle follows dequantize_row_q2_K/dequantize_row_q8_0 in
ggml-org/llama.cpp@74a4fb0037243f1b6196d0f990ff0aff031356cf,
ggml/src/ggml-quants.c (MIT). It does not call the production decoder.
"""
import gc
import struct

import numpy as np
import pytest

from airllm.nextgen.gguf import (
    GGMLType, GGUFError, GGUFMemoryError, GGUFReader, dequantize,
)
from test_gguf import _fixture


def blocks(kind, count=37):
    rng = np.random.default_rng(5380)
    size = 84 if kind == GGMLType.Q2_K else 34
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    # Exercise arbitrary signed FP16 scales while excluding NaN/Inf, including
    # zero, tiny, negative and large scales, rather than one convenient power.
    scales = np.resize(np.array([0., 0.03125, -0.1875, 0.3333, 2**-14, 15.9],
                                dtype='<f2'), count)
    if kind == GGMLType.Q2_K:
        raw[:, 80:82] = scales.view(np.uint8).reshape(-1, 2)
        raw[:, 82:84] = scales[::-1].copy().view(np.uint8).reshape(-1, 2)
    else:
        raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        # Explicit signed extrema, in addition to random signed bytes.
        raw[:, 2:6] = [0, 127, 128, 255]
    return raw


def scalar_ggml(raw, kind):
    result = []
    for block in raw:
        data = block.tobytes()
        if kind == GGMLType.Q8_0:
            d = np.float32(struct.unpack('<e', data[:2])[0])
            result.extend(np.float32(d * q) for q in struct.unpack('<32b', data[2:]))
            continue
        d, dmin = (np.float32(v) for v in struct.unpack('<ee', data[80:84]))
        scale_index = 0
        for n in range(0, 256, 128):
            for shift in range(0, 8, 2):
                for lane in range(2):
                    sc = data[scale_index]
                    scale_index += 1
                    dl = np.float32(d * (sc & 15))
                    ml = np.float32(dmin * (sc >> 4))
                    for l in range(16):
                        q = (data[16 + n // 4 + lane * 16 + l] >> shift) & 3
                        result.append(np.float32(np.float32(dl * q) - ml))
    return np.asarray(result, dtype=np.float32)


@pytest.mark.parametrize('kind', [GGMLType.Q2_K, GGMLType.Q8_0])
def test_scalar_ggml_exact_parity(kind):
    raw = blocks(kind)
    np.testing.assert_array_equal(dequantize(raw.tobytes(), kind), scalar_ggml(raw, kind))


@pytest.mark.parametrize('kind', [GGMLType.Q2_K, GGMLType.Q8_0])
def test_upstream_gguf_python_exact_parity(kind):
    gguf = pytest.importorskip('gguf')
    from gguf.quants import dequantize as upstream
    raw = blocks(kind)
    expected = upstream(raw, gguf.GGMLQuantizationType(int(kind))).reshape(-1)
    np.testing.assert_array_equal(dequantize(raw.tobytes(), kind), expected)


@pytest.mark.parametrize('kind,width,size', [(GGMLType.Q2_K, 256, 84),
                                          (GGMLType.Q8_0, 32, 34)])
def test_selective_expert_rows_and_packed_bytes(tmp_path, kind, width, size):
    raw = blocks(kind, 12)
    path = _fixture(tmp_path / 'experts.gguf', [('experts', (width, 3, 4), kind, raw.tobytes())])
    with GGUFReader(path) as reader:
        info = reader.tensors['experts']
        assert info.row_bytes == size
        assert info.nbytes == len(raw) * size
        assert reader.stats['tensor_bytes'] == 0
        value = reader.read_rows('experts', 1, 2, expert=2)
        np.testing.assert_array_equal(value, scalar_ggml(raw[7:9], kind).reshape(2, width))
        assert reader.stats['tensor_bytes'] == 2 * size
        packed = reader.read_packed_rows('experts', 1, 2, expert=2, copy=False)
        assert packed.tobytes() == raw[7:9].tobytes()
        assert reader.read_rows('experts', 3, 0, expert=1).shape == (0, width)


@pytest.mark.parametrize('kind,width', [(GGMLType.Q2_K, 256), (GGMLType.Q8_0, 32)])
def test_budget_checked_before_quantized_payload_read(tmp_path, monkeypatch, kind, width):
    raw = blocks(kind, 3)
    path = _fixture(tmp_path / 'budget.gguf', [('w', (width, 3), kind, raw.tobytes())])
    # 8192 scratch bytes are reserved for both codecs. A budget that fits the
    # encoded bytes and output alone must still refuse before a tensor read.
    with GGUFReader(path, memory_budget_bytes=8192) as reader:
        before = dict(reader.stats)
        assert reader.max_rows('w') == 0
        original = reader._pread
        def guarded(count, offset, *, tensor=False):
            assert not tensor, 'quantized payload read preceded allocation guard'
            return original(count, offset, tensor=tensor)
        monkeypatch.setattr(reader, '_pread', guarded)
        with pytest.raises(GGUFMemoryError):
            reader.read_rows('w', 0, 1)
        assert reader.stats == before


@pytest.mark.parametrize('kind', [GGMLType.Q2_K, GGMLType.Q8_0])
def test_truncated_blocks_and_live_output_accounting(tmp_path, kind):
    raw = blocks(kind, 2)
    with pytest.raises(GGUFError, match='Truncated'):
        dequantize(raw.tobytes()[:-1], kind)
    width = 256 if kind == GGMLType.Q2_K else 32
    path = _fixture(tmp_path / 'live.gguf', [('w', (width, 2), kind, raw.tobytes())])
    with GGUFReader(path) as reader:
        available = reader.available_memory_bytes
        value = reader.read_rows('w', 0, 2)
        assert reader.available_memory_bytes == available - value.nbytes
        del value
        gc.collect()
        assert reader.available_memory_bytes == available


def test_bfloat16_preserves_every_bit_pattern():
    # Exhaust the 16-bit format, including both zeros, subnormals, infinities,
    # signaling/quiet NaNs and their payload/sign bits.
    stored = np.arange(65536, dtype='<u2')
    decoded = dequantize(stored.tobytes(), GGMLType.BF16)
    np.testing.assert_array_equal(decoded.view(np.uint32), stored.astype(np.uint32) << 16)


def test_bfloat16_upstream_gguf_bitwise_parity():
    gguf = pytest.importorskip('gguf')
    from gguf.quants import dequantize as upstream
    stored = np.arange(65536, dtype='<u2')
    expected = upstream(stored.view(np.uint8).reshape(-1, 2), gguf.GGMLQuantizationType.BF16)
    decoded = dequantize(stored.tobytes(), GGMLType.BF16)
    np.testing.assert_array_equal(decoded.view(np.uint32), expected.reshape(-1).view(np.uint32))


def test_bfloat16_selective_rows_without_quantized_scratch(tmp_path):
    stored = np.array([0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80,
                       0x7fc1, 0xffc1, 0x0001, 0x8001, 0x3eaa, 0xbeaa], dtype='<u2')
    path = _fixture(tmp_path / 'bf16.gguf', [('w', (3, 4), GGMLType.BF16, stored.tobytes())])
    with GGUFReader(path, memory_budget_bytes=2048) as reader:
        assert reader.tensors['w'].row_bytes == 6
        assert reader.estimate_read_bytes('w', 2) == 2 * 3 * (2 + 4)
        assert reader.max_rows('w') == 4
        decoded = reader.read_rows('w', 1, 2)
        np.testing.assert_array_equal(decoded.reshape(-1).view(np.uint32),
                                      stored[3:9].astype(np.uint32) << 16)
        assert reader.stats['tensor_bytes'] == 12
    with pytest.raises(GGUFError, match='Truncated'):
        dequantize(b'\x00', GGMLType.BF16)
