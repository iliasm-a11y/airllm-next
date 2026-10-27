"""CPU-only exact-region and ownership tests, without importing MLX or Torch."""
import errno
import gc
from pathlib import Path
import struct
import sys
from threading import Event, Thread
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
from airllm.nextgen.gguf import GGUFError, GGUFMemoryError, GGUFReader, PackedBufferBudget
from airllm.nextgen import gguf_packed_io as direct


def fixture(tmp_path, kind=12, *, max_read_bytes=None):
    width = 512 if kind in (12, 14) else 7
    row_bytes = width // 256 * (144 if kind == 12 else 210) if kind in (12, 14) else width * 4
    raw = np.arange(6 * 9 * row_bytes, dtype=np.uint8).reshape(6, 9, row_bytes)
    name = b"bank"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQQIQ", 3, width, 9, 6, kind, 0)
    header += bytes((-len(header)) % 32)
    path = tmp_path / "grouped.gguf"
    path.write_bytes(header + raw.tobytes())
    return GGUFReader(path, memory_budget_bytes=1 << 20, max_read_bytes=max_read_bytes), raw


@pytest.mark.parametrize("kind", [0, 12, 14])
@pytest.mark.parametrize("experts,rows,regions", [([0, 1, 3, 4], 9, 2),
                                                 ([0, 1, 3, 4], 4, 4),
                                                 ([3, 0, 1, 5], 9, 3),
                                                 ([2, 1, 0], 9, 3)])
def test_exact_gather_preserves_requested_order_without_extra_bytes(tmp_path, monkeypatch, kind, experts, rows, regions):
    reader, raw = fixture(tmp_path, kind)
    with reader:
        before = dict(reader.stats)
        calls = []
        original = direct._read_into_once
        def observe(fd, destination, offset):
            calls.append((destination.nbytes, offset))
            return original(fd, destination, offset)
        monkeypatch.setattr(direct, "_read_into_once", observe)
        plan = direct.plan_packed_expert_reads(reader, "bank", 0, rows, experts)
        assert reader.stats == before  # planning alone performs no payload read
        assert len(plan.regions) == regions
        actual = direct.read_packed_experts_into(reader, "bank", 0, rows, experts)
        np.testing.assert_array_equal(actual, raw[experts, :rows])
        assert actual.flags.c_contiguous and actual.flags.writeable
        assert actual.base.flags.owndata and actual.base.nbytes == actual.nbytes
        assert len(calls) == regions
        assert sum(length for length, _ in calls) == actual.nbytes
        assert reader.stats["read_calls"] - before["read_calls"] == regions
        assert reader.stats["logical_bytes"] - before["logical_bytes"] == actual.nbytes
        assert reader.stats["tensor_bytes"] - before["tensor_bytes"] == actual.nbytes
        assert reader.stats["packed_group_adjacent_reads_saved"] == len(experts) - regions
        assert reader.stats["packed_group_read_seconds"] >= 0
        assert reader._reserved == actual.nbytes
        del actual
        gc.collect()
        assert reader._reserved == 0


def test_coalesced_regions_still_obey_max_read_bytes(tmp_path):
    row_bytes = 512 // 256 * 144
    reader, raw = fixture(tmp_path, max_read_bytes=9 * row_bytes)
    with reader:
        plan = direct.plan_packed_expert_reads(reader, "bank", 0, 9, [0, 1, 2])
        assert len(plan.regions) == 3
        assert all(length <= reader.max_read_bytes for _, _, length in plan.regions)
        actual = direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1, 2])
        np.testing.assert_array_equal(actual, raw[:3])


@pytest.mark.parametrize("view_kind", ["slice", "base", "memoryview", "frombuffer"])
def test_full_group_owner_charge_survives_every_alias(tmp_path, view_kind):
    reader, _ = fixture(tmp_path)
    with reader:
        size = 3 * 9 * reader.tensors["bank"].row_bytes
        budget = PackedBufferBudget(size)
        actual = direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1, 2], owner_budget=budget)
        owner = weakref.ref(actual.base)
        view = {"slice": lambda: actual[:1, :1], "base": lambda: actual.base,
                "memoryview": lambda: memoryview(actual),
                "frombuffer": lambda: np.frombuffer(actual, dtype=np.uint8, count=1)}[view_kind]()
        del actual
        gc.collect()
        assert owner() is not None and budget.live_bytes == size
        assert reader._reserved == 0
        with pytest.raises(GGUFMemoryError):
            direct.read_packed_experts_into(reader, "bank", 0, 1, [0], owner_budget=budget)
        del view
        gc.collect()
        assert owner() is None and budget.live_bytes == 0


@pytest.mark.parametrize("refusal", ["owner", "reader", "selection", "closed"])
def test_group_limits_refuse_before_payload_allocation_or_reads(tmp_path, monkeypatch, refusal):
    reader, _ = fixture(tmp_path, max_read_bytes=600 if refusal == "selection" else None)
    budget = PackedBufferBudget(1 if refusal == "owner" else 1 << 20)
    if refusal == "reader":
        reader.memory_budget_bytes = reader._metadata_bytes + 100
    if refusal == "closed":
        reader.close()
    before = dict(reader.stats)
    monkeypatch.setattr(direct.np, "empty", lambda *a, **kw: pytest.fail("Allocation after refusal"))
    monkeypatch.setattr(direct, "_read_into_once", lambda *a: pytest.fail("Read after refusal"))
    with pytest.raises((GGUFMemoryError, ValueError)):
        direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1, 2], owner_budget=budget)
    for key in ("read_calls", "bytes_read", "logical_bytes", "tensor_bytes"):
        assert reader.stats[key] == before[key]
    assert reader._reserved == 0 and budget.live_bytes == 0
    reader.close()


@pytest.mark.parametrize("kwargs", [{"start": True}, {"count": True}, {"count": 0},
                                     {"experts": []}, {"experts": [0, 0]},
                                     {"experts": [False]}, {"experts": [6]},
                                     {"owner_budget": object()}])
def test_group_invalid_selection_refuses_before_io(tmp_path, kwargs):
    reader, _ = fixture(tmp_path)
    with reader:
        values = dict(start=0, count=1, experts=[0, 1])
        values.update(kwargs)
        before = dict(reader.stats)
        with pytest.raises((ValueError, TypeError)):
            direct.read_packed_experts_into(reader, "bank", **values)
        assert reader.stats == before and reader._reserved == 0


def test_group_short_reads_and_eintr_never_cross_region_boundaries(tmp_path, monkeypatch):
    reader, raw = fixture(tmp_path)
    with reader:
        original = direct._read_into_once
        calls = []
        def small(fd, destination, offset):
            calls.append((destination.nbytes, offset))
            if len(calls) == 1:
                raise OSError(errno.EINTR, "interrupted")
            return original(fd, destination[:127], offset)
        monkeypatch.setattr(direct, "_read_into_once", small)
        before = dict(reader.stats)
        actual = direct.read_packed_experts_into(reader, "bank", 2, 4, [4, 0, 2])
        np.testing.assert_array_equal(actual, raw[[4, 0, 2], 2:6])
        assert reader.stats["read_calls"] - before["read_calls"] == len(calls) - 1
        assert reader.stats["tensor_bytes"] - before["tensor_bytes"] == actual.nbytes
        expected_regions = direct.plan_packed_expert_reads(reader, "bank", 2, 4, [4, 0, 2]).regions
        for length, offset in calls:
            assert any(begin <= offset and offset + length <= begin + size for begin, _, size in expected_regions)


@pytest.mark.parametrize("invalid_size", [0, -1, 10**9, True])
def test_group_invalid_read_lengths_release_allocation(tmp_path, monkeypatch, invalid_size):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20)
        monkeypatch.setattr(direct, "_read_into_once", lambda *a: invalid_size)
        with pytest.raises(GGUFError):
            direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1], owner_budget=budget)
        gc.collect()
        assert reader._reserved == budget.live_bytes == 0


@pytest.mark.parametrize("error", [errno.EIO, errno.EINTR, errno.EAGAIN])
def test_group_syscall_errors_are_bounded_and_release_storage(tmp_path, monkeypatch, error):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20)
        calls = []
        def fail(*args):
            calls.append(1)
            raise OSError(error, "failure")
        monkeypatch.setattr(direct, "_read_into_once", fail)
        with pytest.raises(OSError):
            direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1], owner_budget=budget)
        gc.collect()
        assert len(calls) == (1 if error == errno.EIO else 4)
        assert reader._reserved == budget.live_bytes == 0


def test_group_allocation_failure_releases_preflight_charge(tmp_path, monkeypatch):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20)
        def fail(*args, **kwargs):
            raise MemoryError("allocation refused")
        monkeypatch.setattr(direct.np, "empty", fail)
        with pytest.raises(MemoryError):
            direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1], owner_budget=budget)
        assert reader._reserved == budget.live_bytes == 0


@pytest.mark.parametrize("separate_budget", [False, True])
def test_failed_helper_traceback_retains_the_complete_charge(tmp_path, monkeypatch, separate_budget):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20) if separate_budget else None
        refs = []
        def fail(fd, destination, offset):
            refs.append(weakref.ref(destination.base))
            raise RuntimeError("Retain a view in helper traceback")
        monkeypatch.setattr(direct, "_read_into_once", fail)
        retained = None
        try:
            direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1, 2], owner_budget=budget)
        except RuntimeError as error:
            retained = error
        size = 3 * 9 * reader.tensors["bank"].row_bytes
        assert retained is not None and refs[0]() is not None
        assert (budget.live_bytes if budget else reader._reserved) == size
        retained = None
        gc.collect()
        assert refs[0]() is None and reader._reserved == 0
        if budget:
            assert budget.live_bytes == 0


def test_close_waits_for_complete_group_read(tmp_path, monkeypatch):
    reader, raw = fixture(tmp_path)
    entered, release, closed = Event(), Event(), Event()
    original = direct._read_into_once
    result, failures = [], []
    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    def read():
        try:
            result.append(direct.read_packed_experts_into(reader, "bank", 0, 9, [0, 1, 3]))
        except BaseException as error:
            failures.append(error)
    def close():
        reader.close()
        closed.set()
    monkeypatch.setattr(direct, "_read_into_once", delayed)
    worker, closer = Thread(target=read), Thread(target=close)
    worker.start()
    try:
        assert entered.wait(1)
        closer.start()
        assert not closed.wait(.05)
    finally:
        release.set()
        worker.join(2)
        if closer.ident is not None:
            closer.join(2)
        reader.close()
    assert not failures and closed.is_set()
    np.testing.assert_array_equal(result[0], raw[[0, 1, 3]])
    result.clear()
    gc.collect()
    assert reader._reserved == 0
