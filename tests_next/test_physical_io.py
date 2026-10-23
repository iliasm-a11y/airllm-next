"""Cache policy isolation, exact short reads, and honest physical-counter scope."""
import errno
import os
import sys

import numpy as np
import pytest

from airllm.nextgen import physical_io
from airllm.nextgen.config import NextConfig
from airllm.nextgen.storage import OffsetCheckpoint
from test_io_profile import checkpoint


def test_physical_counter_unavailable_is_none_not_zero(monkeypatch):
    monkeypatch.setattr(physical_io.sys, 'platform', 'darwin')
    monkeypatch.setattr(physical_io, '_darwin_reader', lambda: lambda: None)
    assert physical_io.physical_read_bytes() is None
    def unavailable(): raise OSError('libproc unavailable')
    monkeypatch.setattr(physical_io, '_darwin_reader', unavailable)
    assert physical_io.physical_read_bytes() is None
    assert 'not per-file or NAND' in physical_io.physical_read_counter_info()['scope']


def test_linux_storage_counter_preserves_actual_value(monkeypatch):
    monkeypatch.setattr(physical_io.sys, 'platform', 'linux')
    monkeypatch.setattr(physical_io.Path, 'read_text', lambda _: 'rchar: 999\nread_bytes: 123\n')
    assert physical_io.physical_read_bytes() == 123
    monkeypatch.setattr(physical_io.Path, 'read_text', lambda _: 'read_bytes: invalid\n')
    assert physical_io.physical_read_bytes() is None


def test_nocache_unsupported_fails_before_data_or_profile_request(tmp_path, monkeypatch):
    import airllm.nextgen.storage as storage
    cp, key, _ = checkpoint(tmp_path, profile_io=True)
    with cp:
        monkeypatch.setattr(storage.sys, 'platform', 'linux')
        with pytest.raises(NotImplementedError, match='macOS'):
            cp.read_array(key, mode='pread_nocache')
        assert cp.stats['fd_opens'] == 0
        assert cp.io_profile()['request_count'] == 0


def test_fcntl_failure_closes_uncached_fd_and_never_falls_back(tmp_path, monkeypatch):
    import airllm.nextgen.storage as storage
    cp, key, _ = checkpoint(tmp_path)
    opened = []
    def refuse(fd):
        opened.append(fd)
        raise OSError(errno.EINVAL, 'unsupported cache policy')
    monkeypatch.setattr(storage.sys, 'platform', 'darwin')
    monkeypatch.setattr(storage, 'set_fd_nocache', refuse)
    with cp:
        with pytest.raises(OSError, match='unsupported cache policy'):
            cp.read_array(key, mode='pread_nocache')
        assert not cp.fds
        assert not cp.stats['read_calls']
        with pytest.raises(OSError) as error: os.fstat(opened[0])
        assert error.value.errno == errno.EBADF


def test_cache_policies_have_independent_opens_and_eviction_safe_borrows(tmp_path, monkeypatch):
    import airllm.nextgen.storage as storage
    cp, key, values = checkpoint(tmp_path, profile_io=True)
    cp.max_open_files = 1
    configured = []
    monkeypatch.setattr(storage.sys, 'platform', 'darwin')
    monkeypatch.setattr(storage, 'set_fd_nocache', configured.append)
    filename, offset, size, _, _ = cp._region(key)
    with cp:
        with cp.descriptor(filename) as normal:
            normal_stat = os.fstat(normal)
            result = cp.read_array(key, mode='pread_nocache')
            np.testing.assert_array_equal(result, values)
            # Eviction closed the LRU's descriptor, while its borrowed dup lives.
            assert os.fstat(normal).st_ino == normal_stat.st_ino
            assert os.pread(normal, size, offset) == values.tobytes()
        np.testing.assert_array_equal(cp.read_array(key), values)
        assert len(configured) == 1
        assert cp.stats['fd_opens'] == 3
        assert cp.io_profile()['modes']['pread_nocache']['read_bytes'] == size


@pytest.mark.parametrize('mode', ['pread', 'pread_nocache'])
def test_pread_short_read_and_interruption_retry_preserves_exact_bytes(tmp_path, monkeypatch, mode):
    import airllm.nextgen.storage as storage
    cp, key, values = checkpoint(tmp_path)
    real_pread = os.pread
    calls = []
    def short(fd, size, offset):
        calls.append((size, offset))
        if len(calls) == 1: raise InterruptedError(errno.EINTR, 'signal')
        return real_pread(fd, min(17, size), offset)
    monkeypatch.delattr(storage.os, 'preadv', raising=False)
    monkeypatch.setattr(storage.os, 'pread', short)
    monkeypatch.setattr(storage.sys, 'platform', 'darwin')
    monkeypatch.setattr(storage, 'set_fd_nocache', lambda fd: None)
    with cp:
        np.testing.assert_array_equal(cp.read_array(key, mode=mode), values)
        assert len(calls) > 3
        assert calls[0] == calls[1]
        assert cp.stats['logical_bytes'] == values.nbytes


@pytest.mark.skipif(sys.platform != 'darwin', reason='real macOS F_NOCACHE only')
def test_real_macos_nocache_pread_exact_and_default_policy_still_reads(tmp_path):
    cp, key, values = checkpoint(tmp_path, profile_io=True)
    with cp:
        np.testing.assert_array_equal(cp.read_array(key, mode='pread_nocache'), values)
        np.testing.assert_array_equal(cp.read_array(key, mode='mmap'), values)
        report = cp.io_profile()
        assert report['physical_counter_source'].startswith('libproc.')
        counter = report['physical_process_read_bytes']
        assert counter is None or isinstance(counter, int) and counter >= 0
        assert not report['physical_ssd_bandwidth_certified']


def test_lossless_nocache_cannot_silently_claim_uncached_transport():
    assert NextConfig(io_mode='pread_nocache').io_mode == 'pread_nocache'
    with pytest.raises(ValueError, match='compressed lossless'):
        NextConfig(io_mode='pread_nocache', lossless_dir='/pack')


def test_complete_fallback_pread_uses_owned_bytes_without_second_payload_copy(tmp_path, monkeypatch):
    import airllm.nextgen.storage as storage
    cp, key, values = checkpoint(tmp_path)
    payload = values.tobytes()
    monkeypatch.setattr(storage.sys, 'platform', 'darwin')
    monkeypatch.setattr(storage.os, 'preadv', lambda *args: pytest.fail('small macOS read needlessly allocated a vector buffer'), raising=False)
    monkeypatch.setattr(storage.os, 'pread', lambda fd, size, offset: payload)
    filename, offset, size, _, _ = cp._region(key)
    with cp:
        assert cp._bytes(filename, offset, size) is payload
        assert cp.stats['logical_bytes'] == size
