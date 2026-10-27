"""Bounded string-array parsing without tensor read-ahead or model changes."""
import os
import struct
import threading

import pytest

from airllm.nextgen.gguf import GGUFError, GGUFMemoryError, GGUFReader
from test_gguf import _fixture


def _strings(path, values, *, tensors=()):
    return _fixture(path, tensors, metadata=[
        ('tokenizer.ggml.tokens', 9, (8, values)),
        ('general.name', 8, 'after array'),
    ])


@pytest.mark.parametrize('capacity', [1, 8, 31, 4096, 65536])
@pytest.mark.parametrize('retain', [True, False])
def test_string_buffer_matches_scalar_and_stays_before_weights(tmp_path, monkeypatch, capacity, retain):
    values = ['', 'é🙂', 'a' * 70_000] + [f'token-{i}' for i in range(2000)]
    path = _strings(tmp_path / 'strings.gguf', values,
                    tensors=[('w', (1,), 0, struct.pack('<f', 3.25))])
    with GGUFReader(path, include_tokenizer=retain) as reference:
        metadata, tensors, data_offset = reference.metadata, reference.tensors, reference.data_offset
    pread = os.pread
    reads = []
    def observed(fd, count, offset):
        reads.append((offset, count))
        return pread(fd, count, offset)
    monkeypatch.setattr(os, 'pread', observed)
    with GGUFReader(path, include_tokenizer=retain, string_buffer_bytes=capacity) as reader:
        assert reader.metadata == metadata
        assert reader.tensors == tensors
        assert reader.stats['tensor_bytes'] == 0
        assert reader._reserved == 0
        assert reader._string_buffer_reserved == 0
        assert reader._string_buffer == b''
        assert max(offset + count for offset, count in reads) <= data_offset


def test_string_buffer_reduces_syscalls_on_retained_and_skipped_strings(tmp_path):
    path = _strings(tmp_path / 'many.gguf', ['small string'] * 8000)
    for retain in (True, False):
        with GGUFReader(path, include_tokenizer=retain) as baseline:
            calls = baseline.stats['read_calls']
        with GGUFReader(path, include_tokenizer=retain, string_buffer_bytes=65536) as reader:
            assert reader.stats['read_calls'] < calls // 50
            assert reader.stats['peak_operation_bytes'] <= 65536
            assert reader._reserved == 0


@pytest.mark.parametrize('capacity', [True, False, -1, 65537, 1.5, None])
def test_invalid_string_buffer_rejected_before_open(tmp_path, monkeypatch, capacity):
    def forbidden(*args):
        raise AssertionError('Opened an invalid configuration')
    monkeypatch.setattr(os, 'open', forbidden)
    with pytest.raises(ValueError, match='string_buffer_bytes'):
        GGUFReader(tmp_path / 'absent.gguf', string_buffer_bytes=capacity)


def test_buffer_retires_before_retained_metadata_consumes_its_budget(tmp_path):
    path = _strings(tmp_path / 'tight.gguf', ['abcdefghij'] * 300)
    with GGUFReader(path) as reference:
        metadata_bytes = reference.metadata_bytes
        expected = reference.metadata
    with GGUFReader(path, memory_budget_bytes=metadata_bytes,
                    max_metadata_bytes=metadata_bytes, string_buffer_bytes=65536) as reader:
        assert reader.metadata == expected
        assert reader.stats['peak_reserved_bytes'] <= metadata_bytes
        assert reader._reserved == 0


def test_buffer_failure_releases_reservation_and_closes_fd(tmp_path, monkeypatch):
    path = _strings(tmp_path / 'short.gguf', ['x'] * 100)
    pread = os.pread
    def short(fd, count, offset):
        result = pread(fd, count, offset)
        return result[:-1] if count > 32 else result
    monkeypatch.setattr(os, 'pread', short)
    reader = GGUFReader.__new__(GGUFReader)
    with pytest.raises(GGUFError, match='Short GGUF read'):
        reader.__init__(path, string_buffer_bytes=65536)
    assert reader._reserved == 0
    assert reader._string_buffer == b''
    assert reader._fd == -1


def test_buffer_interruption_matches_fail_closed_legacy_policy(tmp_path, monkeypatch):
    path = _strings(tmp_path / 'interrupted.gguf', ['x'] * 100)
    pread = os.pread
    def interrupted(fd, count, offset):
        if count > 32:
            raise InterruptedError('injected interruption')
        return pread(fd, count, offset)
    monkeypatch.setattr(os, 'pread', interrupted)
    reader = GGUFReader.__new__(GGUFReader)
    with pytest.raises(InterruptedError):
        reader.__init__(path, string_buffer_bytes=65536)
    assert reader._reserved == 0
    assert reader._fd == -1


@pytest.mark.parametrize('payload', [
    struct.pack('<Q', 1000) + b'x',
    struct.pack('<Q', 1) + b'\xff',
])
def test_malformed_string_arrays_fail_closed(tmp_path, payload):
    key = b'tokenizer.ggml.tokens'
    record = (struct.pack('<Q', len(key)) + key + struct.pack('<IIQ', 9, 8, 1) + payload)
    path = tmp_path / 'invalid.gguf'
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + record)
    reader = GGUFReader.__new__(GGUFReader)
    with pytest.raises(GGUFError):
        reader.__init__(path, string_buffer_bytes=65536)
    assert reader._reserved == 0
    assert reader._fd == -1


def test_metadata_limit_still_rejects_string_growth(tmp_path):
    path = _strings(tmp_path / 'large.gguf', ['x' * 100] * 100)
    reader = GGUFReader.__new__(GGUFReader)
    with pytest.raises(GGUFMemoryError):
        reader.__init__(path, memory_budget_bytes=4096, string_buffer_bytes=65536)
    assert reader._reserved == 0
    assert reader._fd == -1


def test_empty_string_array_and_following_metadata(tmp_path):
    path = _strings(tmp_path / 'empty.gguf', [])
    with GGUFReader(path, string_buffer_bytes=65536) as reader:
        assert reader.metadata['tokenizer.ggml.tokens'] == []
        assert reader.metadata['general.name'] == 'after array'
        assert reader._reserved == 0


def test_close_partial_reader_without_optional_parser_state_closes_fd(tmp_path):
    path = tmp_path / 'partial.bin'
    path.write_bytes(b'fixture')
    fd = os.open(path, os.O_RDONLY)
    reader = GGUFReader.__new__(GGUFReader)
    reader._fd = fd
    reader._lock = threading.RLock()
    reader.close()
    reader.close()
    assert reader._fd == -1
    with pytest.raises(OSError):
        os.fstat(fd)


def test_index_only_sparse_file_above_one_tib_keeps_64bit_offsets(tmp_path, monkeypatch):
    # This checks indexing, never inference or physical terabyte storage.
    name = b'far.weight'
    relative = 1 << 40
    header = (b'GGUF' + struct.pack('<IQQ', 3, 1, 0) +
              struct.pack('<Q', len(name)) + name + struct.pack('<IQIQ', 1, 1, 0, relative))
    header += b'\0' * (-len(header) % 32)
    path = tmp_path / 'sparse-index-only.gguf'
    with path.open('wb') as stream:
        stream.write(header)
        stream.truncate(len(header) + relative + 4)
    pread = os.pread
    reads = []
    def observed(fd, count, offset):
        reads.append((offset, count))
        return pread(fd, count, offset)
    monkeypatch.setattr(os, 'pread', observed)
    with GGUFReader(path, string_buffer_bytes=65536) as reader:
        assert reader.file_size > 1 << 40
        assert reader.tensors['far.weight'].offset == len(header) + relative
        assert reader.tensors['far.weight'].nbytes == 4
        assert reader.stats['tensor_bytes'] == 0
        assert max(offset + count for offset, count in reads) <= len(header)
    # st_blocks reports allocation separately from the sparse logical size.
    assert path.stat().st_blocks * 512 < 1 << 20
