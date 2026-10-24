"""Small, synthetic packed-IO lifetime and admission tests; no model is loaded."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import airllm.nextgen.gguf as gguf
from airllm.nextgen.adaptive_cache import AdaptiveWeightCache, retained_size
from airllm.nextgen.gguf import GGMLType, GGUFMemoryError, GGUFReader, PackedBufferBudget, TensorInfo
from airllm.nextgen.quantization import PackedMatrix


@pytest.fixture
def packed_reader(tmp_path):
    raw = bytes(range(144))
    path = tmp_path / "weights.bin"
    path.write_bytes(raw)
    reader = GGUFReader.__new__(GGUFReader)
    reader._fd = os.open(path, os.O_RDONLY)
    reader.file_size = len(raw)
    reader._lock = threading.RLock()
    reader._metadata_bytes = reader._reserved = 0
    reader.memory_budget_bytes = len(raw) * 2
    reader.max_read_bytes = len(raw)
    reader.tensors = {"w": TensorInfo("w", (256, 1), GGMLType.Q4_K, 0, len(raw), len(raw))}
    reader.stats = {key: 0 for key in (
        "read_calls", "bytes_read", "logical_bytes", "tensor_bytes", "index_bytes",
        "peak_operation_bytes", "peak_reserved_bytes")}
    try:
        yield reader
    finally:
        reader.close()


def _root_buffer(array):
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


@pytest.mark.parametrize("size", [-1, True, 1.0, "1"])
def test_packed_budget_rejects_invalid_acquire_without_changing_state(size):
    budget = PackedBufferBudget(144)
    with pytest.raises(ValueError):
        budget._acquire(size)
    assert (budget.live_bytes, budget.peak_bytes) == (0, 0)


@pytest.mark.parametrize("size", [-1, True, 1.0, "1", 145])
def test_packed_budget_rejects_invalid_release_without_changing_state(size):
    budget = PackedBufferBudget(144)
    budget._acquire(144)
    with pytest.raises(ValueError):
        budget._release(size)
    assert (budget.live_bytes, budget.peak_bytes) == (144, 144)


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("external_budget", [False, True])
@pytest.mark.parametrize("escape", ["base", "memoryview_base", "memoryview_array", "numpy_view"])
def test_all_public_buffer_views_keep_the_payload_charged(packed_reader, copy, external_budget, escape):
    reader = packed_reader
    reader.memory_budget_bytes = 288 if copy else 144
    budget = PackedBufferBudget(144) if external_budget else None
    array = reader.read_packed_rows("w", 0, 1, copy=copy, owner_budget=budget)
    if escape == "base":
        retained = _root_buffer(array)
    elif escape == "memoryview_base":
        retained = memoryview(_root_buffer(array))
    elif escape == "memoryview_array":
        retained = memoryview(array)
    else:
        retained = np.frombuffer(array, dtype=np.uint8)
    del array
    gc.collect()
    assert (budget.live_bytes if budget else reader._reserved) == 144
    calls = reader.stats["read_calls"]
    with pytest.raises(GGUFMemoryError):
        reader.read_packed_rows("w", 0, 1, copy=copy, owner_budget=budget)
    assert reader.stats["read_calls"] == calls
    del retained
    gc.collect()
    assert (budget.live_bytes if budget else reader._reserved) == 0


def test_copyfree_is_readonly_and_default_copy_is_writable(packed_reader):
    array = packed_reader.read_packed_rows("w", 0, 1, copy=False)
    assert array.tobytes() == bytes(range(144))
    assert not array.flags.writeable
    assert memoryview(_root_buffer(array)).readonly
    with pytest.raises(ValueError):
        array.setflags(write=True)
    del array
    gc.collect()
    array = packed_reader.read_packed_rows("w", 0, 1)
    array[0, 0] = 255
    assert array[0, 0] == 255
    assert packed_reader.stats["peak_operation_bytes"] == 288


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("external_budget", [False, True])
def test_conversion_failure_keeps_escaped_traceback_buffer_charged(packed_reader, monkeypatch, copy, external_budget):
    budget = PackedBufferBudget(144) if external_budget else None

    def reject(buffer, dtype):
        # Its traceback deliberately retains the actual buffer argument.
        raise RuntimeError("injected NumPy conversion failure")

    monkeypatch.setattr(gguf.np, "frombuffer", reject)
    with pytest.raises(RuntimeError) as captured:
        packed_reader.read_packed_rows("w", 0, 1, copy=copy, owner_budget=budget)
    gc.collect()
    assert (budget.live_bytes if budget else packed_reader._reserved) == 144
    captured = None
    gc.collect()
    assert (budget.live_bytes if budget else packed_reader._reserved) == 0


def test_read_failure_releases_both_reservations(packed_reader, monkeypatch):
    budget = PackedBufferBudget(144)

    def reject(*args, **kwargs):
        raise OSError("injected IO failure")

    monkeypatch.setattr(packed_reader, "_pread", reject)
    with pytest.raises(OSError):
        packed_reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
    assert budget.live_bytes == packed_reader._reserved == 0


def test_short_read_traceback_does_not_retain_unaccounted_payload(packed_reader, monkeypatch):
    budget = PackedBufferBudget(144)
    monkeypatch.setattr(gguf.os, "pread", lambda *args: b"short")
    with pytest.raises(gguf.GGUFError, match="wanted 144, got 5") as captured:
        packed_reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
    traceback = captured.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_pread":
            assert "result" not in traceback.tb_frame.f_locals
        traceback = traceback.tb_next
    assert budget.live_bytes == packed_reader._reserved == 0


@pytest.mark.parametrize("copy", [False, True])
def test_peak_payload_is_reserved_before_the_positional_read(packed_reader, monkeypatch, copy):
    original = gguf.os.pread
    budget = PackedBufferBudget(144)
    observed = []

    def track(*args):
        observed.append((packed_reader._reserved, budget.live_bytes))
        return original(*args)

    monkeypatch.setattr(gguf.os, "pread", track)
    array = packed_reader.read_packed_rows("w", 0, 1, copy=copy, owner_budget=budget)
    assert observed == [(144 * (2 if copy else 1), 144)]
    assert packed_reader._reserved == 0
    assert budget.live_bytes == array.nbytes == 144
    del array
    gc.collect()
    assert budget.live_bytes == 0


@pytest.mark.parametrize("copy", [False, True])
def test_empty_selection_does_not_consume_payload_budget(packed_reader, copy):
    packed_reader.memory_budget_bytes = 1
    budget = PackedBufferBudget(0)
    array = packed_reader.read_packed_rows("w", 1, 0, copy=copy, owner_budget=budget)
    assert array.shape == (0, 144)
    assert packed_reader._reserved == budget.live_bytes == budget.peak_bytes == 0
    assert packed_reader.stats["tensor_bytes"] == 0


def test_concurrent_owner_admission_fails_before_second_read(packed_reader, monkeypatch):
    entered, proceed = threading.Event(), threading.Event()
    budget = PackedBufferBudget(144)
    original = packed_reader._pread

    def blocked(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(packed_reader, "_pread", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(packed_reader.read_packed_rows, "w", 0, 1,
                            copy=False, owner_budget=budget)
        assert entered.wait(5)
        try:
            assert packed_reader._reserved == budget.live_bytes == 144
            with pytest.raises(GGUFMemoryError):
                packed_reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
            assert packed_reader._reserved == budget.live_bytes == 144
            assert packed_reader.stats["read_calls"] == 0
        finally:
            proceed.set()
        assert first.result().nbytes == 144
    del first
    gc.collect()
    assert budget.live_bytes == packed_reader._reserved == 0
    assert budget.peak_bytes == 144


@pytest.mark.parametrize("operation", ["size_only", "rejected", "cleared"])
def test_adaptive_inspection_releases_payload_without_cyclic_gc(packed_reader, operation):
    was_enabled = gc.isenabled()
    gc.disable()
    budget = PackedBufferBudget(144)
    try:
        array = packed_reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
        if operation == "size_only":
            assert retained_size({"nested": [array]}) == 144
        else:
            cache = AdaptiveWeightCache(144 if operation == "cleared" else 1)
            admitted = cache.put("w", {"nested": [array]})
            assert admitted == (operation == "cleared")
            cache.clear()
        del array
        # No gc.collect(): admission must not create a hidden payload-owning cycle.
        assert budget.live_bytes == 0
    finally:
        if was_enabled:
            gc.enable()


def test_iterative_adaptive_inspection_preserves_nested_backing_charges():
    root = np.arange(144, dtype=np.uint8)
    other = np.arange(7, dtype=np.uint8)
    first, second = root[:2], root[20:22]
    packed = PackedMatrix(first, second, first, (1, 2), 4, 32)
    nested = {"packed": packed, "views": [second, (first, other)],
              "backend": SimpleNamespace(nbytes=9)}
    assert retained_size(nested) == 144 + 7 + 9


def _metadata_fixture(path, subtype, values, fmt):
    key = b"values"
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    header += struct.pack("<Q", len(key)) + key + struct.pack("<IIQ", 9, subtype, len(values))
    header += b"".join(struct.pack("<" + fmt, value) for value in values)
    path.write_bytes(header + bytes((-len(header)) % 32))
    return path


@pytest.mark.parametrize("kind,fmt,values", [
    (0, "B", [0, 255]), (1, "b", [-128, 127]),
    (2, "H", [0, 65535]), (3, "h", [-32768, 32767]),
    (4, "I", [0, 2**32 - 1]), (5, "i", [-2**31, 2**31 - 1]),
    (6, "f", [1.25, -2.5]), (7, "?", [False, True]),
    (10, "Q", [0, 2**64 - 1]), (11, "q", [-2**63, 2**63 - 1]),
    (12, "d", [1.25, -2.5]),
])
def test_bulk_metadata_matches_scalar_parser(tmp_path, kind, fmt, values):
    path = _metadata_fixture(tmp_path / "metadata.gguf", kind, values, fmt)
    with GGUFReader(path) as scalar, GGUFReader(path, bulk_metadata=True) as bulk:
        assert bulk.metadata == scalar.metadata == {"values": values}
        assert bulk.metadata_bytes == scalar.metadata_bytes


def test_bulk_metadata_chunks_within_remaining_budget(tmp_path):
    values = list(range(100))
    path = _metadata_fixture(tmp_path / "metadata.gguf", 4, values, "I")
    metadata_charge = 256 + 6 * 5 + 128 + 64 + 48 * len(values)
    with GGUFReader(path, bulk_metadata=True,
                    memory_budget_bytes=metadata_charge + 16) as reader:
        assert reader.metadata["values"] == values
        assert reader.available_memory_bytes == 16
        assert reader.stats["peak_operation_bytes"] == 16
        assert reader.stats["peak_reserved_bytes"] <= reader.memory_budget_bytes


def test_bulk_metadata_rejects_before_large_payload_read(tmp_path, monkeypatch):
    path = _metadata_fixture(tmp_path / "metadata.gguf", 10, range(1000), "Q")
    calls = []
    original = gguf.os.pread

    def track(fd, count, offset):
        calls.append(count)
        return original(fd, count, offset)

    monkeypatch.setattr(gguf.os, "pread", track)
    with pytest.raises(GGUFMemoryError):
        GGUFReader(path, bulk_metadata=True, memory_budget_bytes=4096)
    assert max(calls) <= 8
