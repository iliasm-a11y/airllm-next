"""Small CPU checks for direct packed transport and full-buffer ownership."""
import ctypes
import errno
import gc
from threading import Event, Thread
import weakref

import numpy as np
import pytest

from airllm.nextgen.gguf import GGUFError, GGUFMemoryError, GGUFReader, PackedBufferBudget
from airllm.nextgen.gguf_packed_io import read_packed_rows_into
from airllm.nextgen import gguf_packed_io as direct
from test_gguf import _fixture


def fixture(tmp_path, kind=12, *, budget=1 << 20, max_read_bytes=None):
    width = 512 if kind in (12, 14) else 7
    row_bytes = width // 256 * (144 if kind == 12 else 210) if kind in (12, 14) else width * 4
    raw = np.arange(3 * 9 * row_bytes, dtype=np.uint8).reshape(3, 9, row_bytes)
    path = _fixture(tmp_path / "direct.gguf", [("bank", (width, 9, 3), kind, raw.tobytes())])
    return GGUFReader(path, memory_budget_bytes=budget, max_read_bytes=max_read_bytes), raw


@pytest.mark.parametrize("kind", [0, 12, 14])
def test_direct_is_exact_writable_and_uses_no_bytes_reader(tmp_path, monkeypatch, kind):
    reader, raw = fixture(tmp_path, kind)
    with reader:
        before = dict(reader.stats)
        monkeypatch.setattr(reader, "_pread", lambda *a, **kw: pytest.fail("Intermediate pread bytes"))
        actual = read_packed_rows_into(reader, "bank", 2, 4, expert=1)
        np.testing.assert_array_equal(actual, raw[1, 2:6])
        assert actual.flags.c_contiguous and actual.flags.writeable
        assert actual.base.flags.owndata
        assert actual.base.nbytes == actual.nbytes
        assert reader.stats["read_calls"] - before["read_calls"] == 1
        assert reader.stats["tensor_bytes"] - before["tensor_bytes"] == actual.nbytes
        assert reader.stats["packed_direct_io_bytes"] == actual.nbytes
        actual[0, 0] ^= 1
        assert reader._reserved == actual.nbytes
        del actual
        gc.collect()
        assert reader._reserved == 0


@pytest.mark.parametrize("view_kind", ["slice", "base", "memoryview", "frombuffer"])
def test_owner_budget_retains_full_allocation_through_every_view(tmp_path, view_kind):
    reader, _ = fixture(tmp_path)
    with reader:
        size = 4 * reader.tensors["bank"].row_bytes
        budget = PackedBufferBudget(size)
        actual = read_packed_rows_into(reader, "bank", 2, 4, expert=1, owner_budget=budget)
        owner = weakref.ref(actual.base)
        view = {"slice": lambda: actual[:1], "base": lambda: actual.base,
                "memoryview": lambda: memoryview(actual),
                "frombuffer": lambda: np.frombuffer(actual, dtype=np.uint8, count=1)}[view_kind]()
        del actual
        gc.collect()
        assert owner() is not None and budget.live_bytes == size
        assert reader._reserved == 0
        with pytest.raises(GGUFMemoryError):
            read_packed_rows_into(reader, "bank", 0, 1, expert=0, owner_budget=budget)
        del view
        gc.collect()
        assert owner() is None and budget.live_bytes == 0


@pytest.mark.parametrize("refusal", ["owner", "reader", "selection", "closed"])
def test_limits_and_closed_reader_refuse_before_payload_allocation(tmp_path, monkeypatch, refusal):
    reader, _ = fixture(tmp_path, max_read_bytes=600 if refusal == "selection" else None)
    budget = PackedBufferBudget(1 if refusal == "owner" else 1 << 20)
    if refusal == "reader":
        reader.memory_budget_bytes = reader._metadata_bytes + 1
    if refusal == "closed":
        reader.close()
    before = dict(reader.stats)
    monkeypatch.setattr(direct.np, "empty", lambda *a, **kw: pytest.fail("Allocation after refusal"))
    with pytest.raises((GGUFMemoryError, ValueError)):
        read_packed_rows_into(reader, "bank", 2, 4, expert=1, owner_budget=budget)
    for key in ("read_calls", "bytes_read", "logical_bytes", "tensor_bytes"):
        assert reader.stats[key] == before[key]
    assert reader._reserved == 0 and budget.live_bytes == 0
    reader.close()


def test_short_reads_and_eintr_finish_only_selected_bytes(tmp_path, monkeypatch):
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
        actual = read_packed_rows_into(reader, "bank", 2, 4, expert=1)
        np.testing.assert_array_equal(actual, raw[1, 2:6])
        assert reader.stats["read_calls"] - before["read_calls"] == len(calls) - 1
        assert reader.stats["tensor_bytes"] - before["tensor_bytes"] == actual.nbytes
        assert calls[-1][1] < calls[0][1] + actual.nbytes


@pytest.mark.parametrize("error", [0, errno.EIO, errno.EINTR])
def test_failed_syscall_releases_storage_even_when_traceback_is_retained(tmp_path, monkeypatch, error):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20)
        owners = []
        empty = direct.np.empty
        def allocation(*args, **kwargs):
            value = empty(*args, **kwargs)
            owners.append(weakref.ref(value))
            return value
        def fail(*args):
            ctypes.set_errno(error)
            return -1 if error else 0
        monkeypatch.setattr(direct.np, "empty", allocation)
        monkeypatch.delattr(direct.os, "preadv", raising=False)
        monkeypatch.setattr(direct, "_PREAD", fail)
        retained = None
        try:
            read_packed_rows_into(reader, "bank", 0, 2, expert=0, owner_budget=budget)
        except (GGUFError, OSError) as failure:
            retained = failure
        assert retained is not None
        gc.collect()
        assert owners[0]() is None
        assert reader._reserved == 0 and budget.live_bytes == 0


@pytest.mark.parametrize("kwargs", [{"start": True}, {"count": True}, {"expert": True},
                                     {"owner_budget": object()}])
def test_invalid_types_refuse_before_read(tmp_path, kwargs):
    reader, _ = fixture(tmp_path)
    with reader:
        values = dict(start=0, count=1, expert=0)
        values.update(kwargs)
        before = dict(reader.stats)
        with pytest.raises((ValueError, TypeError)):
            read_packed_rows_into(reader, "bank", **values)
        assert reader.stats == before and reader._reserved == 0


def test_close_waits_for_direct_read_without_racing_descriptor(tmp_path, monkeypatch):
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
            result.append(read_packed_rows_into(reader, "bank", 1, 2, expert=2))
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
    assert not failures and closed.is_set() and reader._fd < 0
    np.testing.assert_array_equal(result[0], raw[2, 1:3])
    result.clear()
    gc.collect()
    assert reader._reserved == 0


@pytest.mark.parametrize("separate_budget", [False, True])
def test_failing_helper_traceback_keeps_retained_allocation_charged(tmp_path, monkeypatch, separate_budget):
    reader, _ = fixture(tmp_path)
    with reader:
        budget = PackedBufferBudget(1 << 20) if separate_budget else None
        refs = []
        def helper(fd, destination, offset):
            refs.append(weakref.ref(destination.base))
            raise RuntimeError("Helper traceback deliberately retains destination")
        monkeypatch.setattr(direct, "_read_into_once", helper)
        retained = None
        try:
            read_packed_rows_into(reader, "bank", 0, 2, expert=0, owner_budget=budget)
        except RuntimeError as failure:
            retained = failure
        assert retained is not None and refs[0]() is not None
        size = 2 * reader.tensors["bank"].row_bytes
        if separate_budget:
            assert budget.live_bytes == size and reader._reserved == 0
        else:
            assert reader._reserved == size
        retained = None
        gc.collect()
        assert refs[0]() is None and reader._reserved == 0
        if separate_budget:
            assert budget.live_bytes == 0
