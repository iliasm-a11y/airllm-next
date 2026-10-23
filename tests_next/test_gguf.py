"""Small synthetic fixtures plus optional upstream and real-checkpoint parity."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import struct
import threading

import numpy as np
import pytest

from airllm.nextgen.gguf import GGMLType, GGUFError, GGUFMemoryError, GGUFReader, dequantize


def _string(value):
    value = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(value)) + value


def _metadata(key, kind, value):
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}
    if kind == 8:
        payload = _string(value)
    elif kind == 9:
        subtype, values = value
        payload = struct.pack("<IQ", subtype, len(values))
        payload += b"".join(_string(v) if subtype == 8 else struct.pack("<" + formats[subtype], v) for v in values)
    else:
        payload = struct.pack("<" + formats[kind], value)
    return _string(key) + struct.pack("<I", kind) + payload


def _fixture(path, tensors, metadata=(), *, offsets=None, alignment=32, version=3):
    """tensors are (name, fastest-first dims, type_id, raw bytes)."""
    meta = b"".join(_metadata(*m) for m in metadata)
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)) + meta
    records, payload = [], bytearray()
    for index, (name, dims, kind, raw) in enumerate(tensors):
        relative = (len(payload) + alignment - 1) // alignment * alignment
        if offsets is not None:
            relative = offsets[index]
        records.append(_string(name) + struct.pack("<I", len(dims)) +
                       b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", kind, relative))
        if relative > len(payload):
            payload.extend(bytes(relative - len(payload)))
        end = relative + len(raw)
        if end > len(payload):
            payload.extend(bytes(end - len(payload)))
        payload[relative:end] = raw
    header += b"".join(records)
    header += bytes((-len(header)) % alignment)
    path.write_bytes(header + payload)
    return path


def _blocks(kind, count=7):
    rng = np.random.default_rng(481)
    size = 144 if kind == GGMLType.Q4_K else 210
    raw = rng.integers(0, 256, (count, size), dtype=np.uint8)
    if kind == GGMLType.Q4_K:
        raw[:, :2] = np.frombuffer(np.float16(0.03125).tobytes(), dtype=np.uint8)
        raw[:, 2:4] = np.frombuffer(np.float16(0.015625).tobytes(), dtype=np.uint8)
    else:
        raw[:, 208:210] = np.frombuffer(np.float16(0.03125).tobytes(), dtype=np.uint8)
    return raw


def test_tokenizer_identity_and_stops_survive_skipping_vocabulary(tmp_path):
    path=_fixture(tmp_path/'metadata.gguf',[],metadata=[
        ('tokenizer.ggml.model',8,'gpt2'),('tokenizer.ggml.pre',8,'qwen35'),
        ('tokenizer.ggml.eos_token_ids',9,(4,[248046,248044])),
        ('tokenizer.ggml.tokens',9,(8,['unneeded']))])
    with GGUFReader(path,include_tokenizer_metadata=False) as reader:
        assert reader.metadata['tokenizer.ggml.pre']=='qwen35'
        assert reader.metadata['tokenizer.ggml.eos_token_ids']==[248046,248044]
        assert 'tokenizer.ggml.tokens' not in reader.metadata


def _scalar_c_reference(raw, kind):
    """Scalar layout independently transcribed from GGML dequantize_row_*_K."""
    result = []
    for block in raw:
        values = np.zeros(256, dtype=np.float32)
        if kind == GGMLType.Q4_K:
            d, dm = struct.unpack("<ee", block[:4].tobytes())
            scales = block[4:16].tolist()
            for j in range(8):
                if j < 4:
                    scale, minimum = scales[j] & 63, scales[j + 4] & 63
                else:
                    scale = (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4)
                    minimum = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
                for k in range(32):
                    q = (int(block[16 + 32 * (j // 2) + k]) >> (4 * (j % 2))) & 15
                    values[32 * j + k] = np.float32(np.float32(d * scale) * q) - np.float32(dm * minimum)
        else:
            d = struct.unpack("<e", block[208:210].tobytes())[0]
            scales = block[192:208].view(np.int8).tolist()
            for n in (0, 128):
                half = n // 128
                for k in range(32):
                    lo0, lo1 = int(block[half * 64 + k]), int(block[half * 64 + 32 + k])
                    hi = int(block[128 + half * 32 + k])
                    quant = [(lo0 & 15) | ((hi & 3) << 4),
                             (lo1 & 15) | (((hi >> 2) & 3) << 4),
                             (lo0 >> 4) | (((hi >> 4) & 3) << 4),
                             (lo1 >> 4) | (((hi >> 6) & 3) << 4)]
                    for quarter, q in enumerate(quant):
                        values[n + 32 * quarter + k] = np.float32(d * scales[half * 8 + quarter * 2 + k // 16]) * (q - 32)
        result.extend(values)
    return np.array(result, dtype=np.float32)


@pytest.mark.parametrize("kind", [GGMLType.Q4_K, GGMLType.Q6_K])
def test_quantization_layout_matches_independent_scalar_c(kind):
    raw = _blocks(kind)
    np.testing.assert_array_equal(dequantize(raw.tobytes(), kind), _scalar_c_reference(raw, kind))


@pytest.mark.parametrize("kind", [GGMLType.Q4_K, GGMLType.Q6_K])
def test_quantization_matches_upstream_gguf_python_when_installed(kind):
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize as upstream
    raw = _blocks(kind, 41)
    expected = upstream(raw, gguf.GGMLQuantizationType(int(kind))).reshape(-1)
    np.testing.assert_array_equal(dequantize(raw.tobytes(), kind), expected)


@pytest.mark.parametrize("kind,dt", [(GGMLType.F32, "<f4"), (GGMLType.F16, "<f2")])
def test_float_rows_are_selected_without_full_read(tmp_path, kind, dt):
    matrix = np.arange(35, dtype=np.float32).reshape(7, 5) / 3
    stored = matrix.astype(dt)
    path = _fixture(tmp_path / "float.gguf", [("w", (5, 7), kind, stored.tobytes())])
    with GGUFReader(path) as reader:
        assert reader.tensors["w"].shape == (7, 5)
        assert reader.stats["tensor_bytes"] == 0
        out = reader.read_rows("w", 2, 3)
        np.testing.assert_array_equal(out, stored[2:5].astype(np.float32))
        assert reader.stats["tensor_bytes"] == 3 * 5 * stored.dtype.itemsize
        packed = reader.read_packed_rows("w", 2, 3)
        assert packed.tobytes() == stored[2:5].tobytes()


def test_expert_selection_and_read_shape(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = _fixture(tmp_path / "expert.gguf", [("expert", (4, 3, 2), 0, data.tobytes())])
    with GGUFReader(path) as r:
        np.testing.assert_array_equal(r.read_rows("expert", 1, 2, expert=1), data[1, 1:3])
        np.testing.assert_array_equal(r.read_rows("expert", 4, 2), data.reshape(-1, 4)[4:6])
        np.testing.assert_array_equal(r.read("expert"), data)
        assert r.read_rows("expert", 6, 0).shape == (0, 4)
        for start, count, expert in [(-1, 1, None), (0, -1, None), (0, 1, 2)]:
            with pytest.raises(ValueError):
                r.read_rows("expert", start, count, expert=expert)
        with pytest.raises(IndexError):
            r.read_rows("expert", 2, 2, expert=1)


def test_metadata_types_tokenizer_optional_and_stop_ids_preserved(tmp_path):
    meta = [("general.architecture", 8, "test"), ("general.alignment", 4, 64),
            ("tokenizer.ggml.tokens", 9, (8, ["é", "hello", "<eos>"])),
            ("tokenizer.ggml.eos_token_id", 4, 2), ("small.array", 9, (5, [-1, 9])),
            ("test.boolean", 7, True), ("test.float64", 12, 1.25)]
    path = _fixture(tmp_path / "meta.gguf", [], meta, alignment=64)
    with GGUFReader(path) as r:
        assert r.metadata["tokenizer.ggml.tokens"] == ["é", "hello", "<eos>"]
        assert r.metadata["small.array"] == [-1, 9]
        assert r.data_offset % 64 == 0
    with GGUFReader(path, include_tokenizer=False) as r:
        assert "tokenizer.ggml.tokens" not in r.metadata
        assert r.metadata["tokenizer.ggml.eos_token_id"] == 2


def test_reject_budget_before_payload_pread_or_empty(tmp_path, monkeypatch):
    path = _fixture(tmp_path / "budget.gguf", [("w", (256, 8), 0, bytes(256 * 8 * 4))])
    with GGUFReader(path, memory_budget_bytes=8192) as r:
        before = dict(r.stats)
        monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("allocation preceded guard"))
        with pytest.raises(GGUFMemoryError):
            r.read_rows("w", 0, 8)
        assert r.stats == before


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), float("-inf"),
                                    True, False, -1, 0, 8192.5, "8192", None])
def test_invalid_dequantization_budget_rejected_before_allocation(monkeypatch, budget):
    monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("allocation preceded budget validation"))
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        dequantize(struct.pack("<f", 1), GGMLType.F32, memory_budget_bytes=budget)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf"),
                                   True, False, -1, 8192.5, "8192"])
def test_invalid_read_limit_rejected_before_open(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("file opened before read-limit validation"))
    with pytest.raises(ValueError, match="max_read_bytes"):
        GGUFReader(tmp_path / "unopened.gguf", max_read_bytes=limit)


def test_zero_read_limit_permits_index_but_no_payload(tmp_path):
    path = _fixture(tmp_path / "index_only.gguf", [("w", (4,), 0, bytes(16))])
    with GGUFReader(path, max_read_bytes=0) as reader:
        assert reader.max_rows("w") == 0
        before = dict(reader.stats)
        with pytest.raises(GGUFMemoryError, match="max_read_bytes"):
            reader.read_rows("w", 0, 1)
        assert reader.stats == before


def test_live_arrays_hold_budget_and_release_on_gc(tmp_path):
    path = _fixture(tmp_path / "live.gguf", [("w", (128, 8), 0, bytes(128 * 8 * 4))])
    with GGUFReader(path, memory_budget_bytes=12288) as r:
        before = r.available_memory_bytes
        result = r.read("w")
        assert r.available_memory_bytes == before - result.nbytes
        del result
        gc.collect()
        assert r.available_memory_bytes == before
        packed = r.read_packed_rows("w", 0, 3)
        assert r.available_memory_bytes == before - packed.nbytes
        view = np.frombuffer(packed, dtype=np.float32)
        del packed
        gc.collect()
        assert r.available_memory_bytes == before - view.nbytes
        del view
        gc.collect()
        assert r.available_memory_bytes == before


def test_concurrent_reservation_rejects_second_before_read(tmp_path, monkeypatch):
    path = _fixture(tmp_path / "concurrent.gguf", [("w", (256, 8), 0, bytes(256 * 8 * 4))])
    entered, release = threading.Event(), threading.Event()
    with GGUFReader(path, memory_budget_bytes=12000) as r:
        original = r._pread
        def blocked(count, offset, *, tensor=False):
            if tensor:
                entered.set()
                assert release.wait(5)
            return original(count, offset, tensor=tensor)
        monkeypatch.setattr(r, "_pread", blocked)
        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(r.read_rows, "w", 0, 4)
            assert entered.wait(5)
            try:
                with pytest.raises(GGUFMemoryError):
                    r.read_rows("w", 4, 4)
            finally:
                release.set()
            assert one.result().shape == (4, 256)
        assert r.stats["tensor_bytes"] == 4 * 256 * 4


@pytest.mark.parametrize("case", ["truncated", "overlap", "unaligned", "duplicate", "unknown", "width", "dimension", "version", "alignment", "duplicate_metadata"])
def test_malformed_indexes_fail_closed(tmp_path, case):
    rows = [("a", (4,), 0, bytes(16)), ("b", (4,), 0, bytes(16))]
    meta, offsets, version = [], None, 3
    if case == "overlap": offsets = [0, 0]
    if case == "unaligned": offsets = [1, 32]
    if case == "duplicate": rows[1] = ("a", (4,), 0, bytes(16))
    if case == "unknown": rows[0] = ("a", (4,), 999, bytes(16))
    if case == "width": rows[0] = ("a", (255,), 12, bytes(144))
    if case == "dimension": rows[0] = ("a", (0,), 0, bytes(16))
    if case == "version": version = 100
    if case == "alignment": meta = [("general.alignment", 4, 3)]
    if case == "duplicate_metadata": meta = [("x", 4, 1), ("x", 4, 2)]
    path = _fixture(tmp_path / (case + ".gguf"), rows, meta, offsets=offsets, version=version)
    if case == "truncated": path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(GGUFError):
        GGUFReader(path)


def test_header_bombs_and_metadata_budget_are_bounded(tmp_path):
    p = tmp_path / "bomb.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 2**63, 0))
    with pytest.raises(GGUFError): GGUFReader(p)
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**63))
    with pytest.raises(GGUFError): GGUFReader(p)
    _fixture(p, [], [("x", 8, "a" * 500)])
    with pytest.raises(GGUFMemoryError): GGUFReader(p, max_metadata_bytes=1024)


def test_truncate_after_open_is_detected(tmp_path):
    p = _fixture(tmp_path / "mutated.gguf", [("w", (8,), 0, bytes(32))])
    with GGUFReader(p) as r:
        os.truncate(p, r.data_offset)
        with pytest.raises(GGUFError, match="Short"):
            r.read_rows("w", 0, 1)


def test_closed_reader_and_truncated_quant_blocks(tmp_path):
    p = _fixture(tmp_path / "closed.gguf", [("w", (8,), 0, bytes(32))])
    r = GGUFReader(p)
    r.close(); r.close()
    with pytest.raises(ValueError, match="closed"): r.read("w")
    with pytest.raises(ValueError, match="closed"): r.fileno()
    with pytest.raises(GGUFError): dequantize(bytes(143), GGMLType.Q4_K)
    with pytest.raises(GGUFMemoryError): dequantize(bytes(144), GGMLType.Q4_K, memory_budget_bytes=256)


@pytest.mark.skipif(not os.environ.get("AIRLLM_TEST_GGUF"), reason="Set AIRLLM_TEST_GGUF to opt into <=16 MiB real-checkpoint reads")
def test_real_checkpoint_sample_upstream_parity_without_mapping():
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize as upstream
    total = 0
    with GGUFReader(os.environ["AIRLLM_TEST_GGUF"], include_tokenizer=False) as r:
        for kind in GGMLType:
            tensor = next((t for t in r.tensors.values() if t.quant_type == kind), None)
            if tensor is None: continue
            count = min(3, tensor.rows)
            packed = r.read_packed_rows(tensor.name, 0, count)
            total += packed.nbytes
            actual = r.read_rows(tensor.name, 0, count)
            expected = upstream(packed, gguf.GGMLQuantizationType(int(kind))).reshape(actual.shape)
            np.testing.assert_array_equal(actual, expected)
        assert total < 16 << 20
        assert r.stats["tensor_bytes"] == total * 2
