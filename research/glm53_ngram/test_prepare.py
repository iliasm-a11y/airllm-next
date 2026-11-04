import hashlib
from types import SimpleNamespace

import pytest

from research.glm53_ngram import prepare as module


def setup_fixture(monkeypatch, tmp_path):
    path = tmp_path/'fixture.gguf'
    data = bytes(range(256))*16
    path.write_bytes(data)
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module, 'FILES', [('fixture.gguf', len(data), hashlib.sha256(data).hexdigest())])
    monkeypatch.setattr(module, 'system_snapshot', lambda: {'pressure_level': 1, 'available_bytes': 8<<30})
    monkeypatch.setattr(module, 'inspect_native_checkpoint', lambda *args: {'integrity_verified': True, 'architecture_matches': True})
    closed = []
    monkeypatch.setattr(module, 'acquire_uid_lock', lambda: (SimpleNamespace(close=lambda: closed.append(True)), {'acquired': True}))
    return path, data, closed


def fake_clock():
    value = [0.]
    return lambda: value[0], lambda seconds: value.__setitem__(0, value[0]+seconds)


def test_three_field_checkpoint_manifest_and_readonly_file_preservation(monkeypatch, tmp_path):
    path, data, closed = setup_fixture(monkeypatch, tmp_path)
    identity = module._identity(path)
    clock, sleep = fake_clock()
    result = module.prepare(tmp_path, tmp_path/'receipt.json', clock=clock, sleep=sleep)
    assert path.read_bytes() == data
    assert module._identity(path) == identity
    assert result['files'][0]['identity_unchanged']
    assert not result['model_inference_started']
    assert closed == [True]


def test_cache_advice_cannot_replace_the_unchanged_model_admission(monkeypatch, tmp_path):
    _, _, closed = setup_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(module, 'system_snapshot', lambda: {'pressure_level': 1, 'available_bytes': 4<<30})
    clock, sleep = fake_clock()
    with pytest.raises(MemoryError, match='no stable unchanged5.5GiB'):
        module.prepare(tmp_path, tmp_path/'receipt.json', clock=clock, sleep=sleep)
    assert closed == [True]


def test_busy_model_lock_prevents_mapping(monkeypatch, tmp_path):
    setup_fixture(monkeypatch, tmp_path)
    def busy():
        raise RuntimeError('ENGINE_BUSY')
    monkeypatch.setattr(module, 'acquire_uid_lock', busy)
    class UnexpectedMapping:
        madvise = True
        def __new__(cls, *args, **kwargs):
            pytest.fail('must not map while another engine is active')
    monkeypatch.setattr(module.mmap, 'mmap', UnexpectedMapping)
    with pytest.raises(RuntimeError, match='ENGINE_BUSY'):
        module.prepare(tmp_path, tmp_path/'receipt.json')
