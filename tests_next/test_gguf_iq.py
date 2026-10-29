"""Mixed IQ GGUF reads, signed values, codebook accounting and rejection guards."""
import gc

import numpy as np
import pytest

from airllm.nextgen import gguf as production
from airllm.nextgen.gguf import (
    GGMLType, GGUFError, GGUFMemoryError, GGUFReader, dequantize,
    IQ_CODEBOOK_BUDGET_BYTES,
)
from test_gguf import _fixture


CODECS = [(GGMLType.IQ2_S, 82), (GGMLType.IQ3_XXS, 98), (GGMLType.IQ3_S, 110)]


def iq_blocks(kind, count=127):
    size = dict(CODECS)[kind]
    rng = np.random.default_rng(53000 + int(kind))
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    scales = np.resize(np.array(
        [0., -0., 2**-24, -2**-24, 2**-14, -0.1875, 0.3333, 15.9, 65504],
        dtype="<f2"), count)
    raw[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    return raw


def upstream_values(raw, kind):
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize as upstream
    return upstream(raw, gguf.GGMLQuantizationType(int(kind))).reshape(-1)


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_python_upstream_all_positions_and_signed_zeros(kind, size):
    raw = iq_blocks(kind)
    actual = dequantize(raw.tobytes(), kind)
    expected = upstream_values(raw, kind)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual.size == len(raw) * 256


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_selective_expert_rows_preserve_layout(tmp_path, kind, size):
    raw = iq_blocks(kind, 24)
    path = _fixture(tmp_path / "expert.gguf",
                    [("experts", (512, 3, 4), kind, raw.tobytes())])
    with GGUFReader(path) as reader:
        assert reader.tensors["experts"].shape == (4, 3, 512)
        assert reader.tensors["experts"].row_bytes == 2 * size
        assert reader.metadata_bytes >= IQ_CODEBOOK_BUDGET_BYTES
        assert reader.stats["tensor_bytes"] == 0
        actual = reader.read_rows("experts", 1, 2, expert=2)
        expected = upstream_values(raw[14:18], kind).reshape(2, 512)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert reader.stats["tensor_bytes"] == 4 * size
        packed = reader.read_packed_rows("experts", 1, 2, expert=2, copy=False)
        assert packed.tobytes() == raw[14:18].tobytes()
        assert reader.read_rows("experts", 3, 0, expert=2).shape == (0, 512)
        with pytest.raises(IndexError):
            reader.read_rows("experts", 2, 2, expert=2)
        with pytest.raises(ValueError):
            reader.read_rows("experts", 0, 1, expert=4)


def test_iq_codebook_charge_is_once_for_mixed_reader(tmp_path):
    raw = [(kind.name, (256, 1), kind, iq_blocks(kind, 1).tobytes())
           for kind, _ in CODECS]
    path = _fixture(tmp_path / "mixed.gguf", raw)
    with GGUFReader(path) as reader:
        # Tensor/name metadata adds less than 2 KiB. The shared grids must not
        # be charged once per tensor or codec, nor omitted from the budget.
        assert IQ_CODEBOOK_BUDGET_BYTES < reader.metadata_bytes < IQ_CODEBOOK_BUDGET_BYTES + 2048
        for kind, _ in CODECS:
            actual = reader.read(kind.name)
            expected = upstream_values(iq_blocks(kind, 1), kind).reshape(1, 256)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_budget_refusal_precedes_payload_read(tmp_path, monkeypatch, kind, size):
    path = _fixture(tmp_path / "budget.gguf",
                    [("w", (256, 2), kind, iq_blocks(kind, 2).tobytes())])
    with GGUFReader(path, memory_budget_bytes=25 << 10) as reader:
        assert reader.max_rows("w") == 0
        before = dict(reader.stats)
        original = reader._pread
        def guarded(count, offset, *, tensor=False):
            assert not tensor, "IQ tensor read happened before budget rejection"
            return original(count, offset, tensor=tensor)
        monkeypatch.setattr(reader, "_pread", guarded)
        with pytest.raises(GGUFMemoryError):
            reader.read_rows("w", 0, 1)
        assert reader.stats == before


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_direct_budget_refusal_precedes_decoder(monkeypatch, kind, size):
    def forbidden(*args):
        pytest.fail("decoder ran before fixed codebook budget was admitted")
    monkeypatch.setattr(production, "_decode_into", forbidden)
    with pytest.raises(GGUFMemoryError):
        dequantize(iq_blocks(kind, 1).tobytes(), kind,
                   memory_budget_bytes=IQ_CODEBOOK_BUDGET_BYTES)


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_live_outputs_release_only_payload_charge(tmp_path, kind, size):
    path = _fixture(tmp_path / "live.gguf",
                    [("w", (256, 2), kind, iq_blocks(kind, 2).tobytes())])
    with GGUFReader(path) as reader:
        initial = reader.available_memory_bytes
        decoded = reader.read_rows("w", 0, 2)
        assert reader.available_memory_bytes == initial - decoded.nbytes
        del decoded
        gc.collect()
        assert reader.available_memory_bytes == initial
        assert reader.metadata_bytes >= IQ_CODEBOOK_BUDGET_BYTES


@pytest.mark.parametrize("kind,size", CODECS)
def test_iq_truncation_and_invalid_row_width_fail_closed(tmp_path, kind, size):
    with pytest.raises(GGUFError, match="Truncated"):
        dequantize(iq_blocks(kind, 1).tobytes()[:-1], kind)
    path = _fixture(tmp_path / "badwidth.gguf",
                    [("w", (255, 1), kind, bytes(size))])
    with pytest.raises(GGUFError, match="not divisible by 256"):
        GGUFReader(path)
    with pytest.raises(GGUFError, match="Unsupported quantization"):
        dequantize(b"", 999)


def test_iq_packed_codebooks_are_shared_immutable_views():
    from airllm.nextgen import gguf_iq_tables as tables
    assert tables.TABLE_PAYLOAD_BYTES == 11400
    assert tables.TABLE_PAYLOAD_BYTES < IQ_CODEBOOK_BUDGET_BYTES
    for name in ("KMASK_IQ2XS", "KSIGNS_IQ2XS", "IQ2S_GRID", "IQ3XXS_GRID", "IQ3S_GRID"):
        array = getattr(tables, name)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0

