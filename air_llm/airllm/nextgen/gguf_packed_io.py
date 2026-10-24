"""Bounded positional reads directly into the final packed NumPy allocation.

Unlike ``os.pread`` followed by NumPy copies, this transport produces one owned,
writable buffer. It can therefore be offered to the existing DLPack importer.
The selected bytes, descriptor cache policy, and arithmetic are unchanged.
This saves host copies; it does not establish a reduction in physical SSD I/O.
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys
import weakref

import numpy as np

from .gguf import GGUFError, PackedBufferBudget


_PREAD = None
if sys.platform in ("darwin", "linux") and ctypes.sizeof(ctypes.c_void_p) == 8:
    # Both supported 64-bit platforms use a signed 64-bit off_t. Loading the
    # process libc preserves errno and CDLL releases the GIL during the syscall.
    _PREAD = getattr(ctypes.CDLL(None, use_errno=True), "pread", None)
    if _PREAD is not None:
        _PREAD.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64)
        _PREAD.restype = ctypes.c_ssize_t


def _read_into_once(fd, destination, offset):
    """One positional syscall; never create an intermediate payload buffer."""
    view = None
    try:
        if hasattr(os, "preadv"):
            view = memoryview(destination).cast("B")
            return os.preadv(fd, [view], offset)
        if _PREAD is None:
            raise NotImplementedError("Direct packed I/O requires preadv or macOS/Linux libc.pread")
        size = _PREAD(fd, ctypes.c_void_p(destination.ctypes.data), destination.nbytes, offset)
        if size < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return size
    finally:
        # An exception traceback must not retain a buffer after its reservation
        # is released by the caller. The libc pointer itself owns no payload.
        view = None
        destination = None


def read_packed_rows_into(reader, name, start, count, *, expert=None,
                          owner_budget: PackedBufferBudget | None = None):
    """Read one GGUF selection into one exact-sized, writable owned allocation.

    The complete selection and both allocation budgets are checked before
    allocation or I/O. Without ``owner_budget``, the reader retains the charge
    until the last ndarray/buffer view dies. With an owner budget, that budget
    owns the returned buffer and the reader reserves its transient operation.
    Short reads are continued without exceeding the selected region; EOF fails
    closed. Only EINTR/EAGAIN are retried, at most three times per selection.
    """
    if type(start) is not int or type(count) is not int:
        raise ValueError("Row start and count must be nonnegative integers")
    if expert is not None and type(expert) is not int:
        raise ValueError("expert must be an integer or None")
    if owner_budget is not None and not isinstance(owner_budget, PackedBufferBudget):
        raise TypeError("Invalid owner budget")
    info, offset = reader._selection(name, start, count, expert)
    encoded = count * info.row_bytes
    if offset < 0 or offset > reader.file_size or encoded > reader.file_size - offset:
        raise GGUFError("Packed selection exceeds GGUF file bounds")
    if not hasattr(os, "preadv") and _PREAD is None:
        raise NotImplementedError("Direct packed I/O requires preadv or macOS/Linux libc.pread")
    owner = out = destination = None
    charged = False
    # Hold the same descriptor lock as GGUFReader._pread, including allocation
    # preflight, so close cannot race with the positional read.
    with reader._lock:
        if reader._fd < 0:
            raise ValueError("GGUFReader is closed")
        with reader._reservation(encoded) as retained:
            try:
                if owner_budget is not None:
                    owner_budget._acquire(encoded)
                    charged = True
                owner = np.empty(encoded, dtype=np.uint8)
                release = reader._release if owner_budget is None else owner_budget._release
                # Transfer accounting before any fallible helper can retain
                # this allocation in an exception traceback.
                weakref.finalize(owner, release, encoded)
                if owner_budget is None:
                    retained[0] = encoded
                charged = False
                out = owner.reshape(count, info.row_bytes)
                position = retries = 0
                while position < encoded:
                    destination = owner[position:]
                    try:
                        size = _read_into_once(reader._fd, destination, offset + position)
                    except OSError as error:
                        if error.errno in (errno.EINTR, errno.EAGAIN) and retries < 3:
                            retries += 1
                            continue
                        raise
                    finally:
                        destination = None
                    if type(size) is not int or size < 0 or size > encoded - position:
                        raise GGUFError("Invalid direct packed read length")
                    reader.stats["read_calls"] += 1
                    reader.stats["bytes_read"] += size
                    reader.stats["logical_bytes"] += size
                    reader.stats["tensor_bytes"] += size
                    if size == 0:
                        raise GGUFError(f"Short GGUF read at offset {offset + position}")
                    position += size
                reader.stats["packed_direct_io_reads"] = reader.stats.get("packed_direct_io_reads", 0) + 1
                reader.stats["packed_direct_io_bytes"] = reader.stats.get("packed_direct_io_bytes", 0) + encoded
                return out
            except BaseException:
                # Keep a retained traceback from retaining uncharged storage.
                destination = out = owner = None
                raise
            finally:
                if charged:
                    owner_budget._release(encoded)
