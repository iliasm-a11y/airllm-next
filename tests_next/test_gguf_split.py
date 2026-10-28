import gc
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pytest

from airllm.nextgen.gguf import GGUFError, GGUFMemoryError
from airllm.nextgen.gguf_split import SplitGGUFReader
from test_gguf import _fixture


def shards(tmp_path, *, reverse=False, duplicate=False, total=2):
    paths = [tmp_path / f'model-{i:05d}-of-00002.gguf' for i in (1, 2)]
    for i, path in enumerate(paths):
        _fixture(path, [(('a' if duplicate else 'ab'[i]), (1024, 2), 0,
                         np.full((2, 1024), i + 1, dtype='<f4').tobytes())],
                 [('general.architecture', 8, 'glm5-next'),
                  ('split.no', 4, 1-i if reverse else i),
                  ('split.count', 4, 2), ('split.tensors.count', 4, total)])
    return paths


def test_dispatch_and_no_weight_reads_on_open(tmp_path):
    paths = shards(tmp_path)
    with SplitGGUFReader(paths[0]) as r:
        assert set(r.tensors) == {'a', 'b'}
        assert r.stats['tensor_bytes'] == 0
        np.testing.assert_array_equal(r.read_rows('b', 0, 1), 2)
        assert r.stats['tensor_bytes'] == 4096
        assert len(r.filenos()) == 2


def test_budget_shared_across_shards_and_live_views(tmp_path):
    paths = shards(tmp_path)
    with SplitGGUFReader(paths, memory_budget_bytes=15000) as r:
        x = r.read_rows('a', 0, 1)
        view = x[:, :2]
        del x
        # First retained 4096 + metadata; each new F32 read reserves8192.
        with pytest.raises(GGUFMemoryError):
            r.read_rows('b', 0, 1)
        del view
        gc.collect()
        np.testing.assert_array_equal(r.read_rows('b', 0, 1), 2)


@pytest.mark.parametrize('kwargs', [{'reverse': True}, {'duplicate': True}, {'total': 3}])
def test_invalid_shard_sets(tmp_path, kwargs):
    with pytest.raises(GGUFError):
        SplitGGUFReader(shards(tmp_path, **kwargs))


def test_missing_or_repeated_shards(tmp_path):
    paths = shards(tmp_path)
    with pytest.raises(GGUFError):
        SplitGGUFReader([paths[0], paths[0]])
    paths[1].unlink()
    with pytest.raises(FileNotFoundError):
        SplitGGUFReader(paths[0])


def test_metadata_only_first_shard(tmp_path):
    paths = shards(tmp_path)
    _fixture(paths[0], [], [('general.architecture', 8, 'glm5-next'),
        ('split.no', 4, 0), ('split.count', 4, 2), ('split.tensors.count', 4, 1)])
    _fixture(paths[1], [('w', (4,), 0, np.zeros(4, '<f4').tobytes())],
        [('split.no', 4, 1), ('split.count', 4, 2), ('split.tensors.count', 4, 1)])
    with SplitGGUFReader(paths[0]) as r:
        assert r.metadata['general.architecture'] == 'glm5-next'
        assert r.read('w').shape == (4,)


@pytest.mark.parametrize('budget', [True, 0, -1, 8192.5, float('nan'), None])
@pytest.mark.parametrize('key', ['memory_budget_bytes', 'max_metadata_bytes'])
def test_invalid_shared_budgets_fail_before_open(tmp_path, monkeypatch, budget, key):
    monkeypatch.setattr(os, 'open', lambda *a, **k: pytest.fail('opened before budget validation'))
    with pytest.raises(ValueError, match=key):
        SplitGGUFReader([tmp_path / 'missing'], **{key: budget})


def test_metadata_budget_is_total_across_shards(tmp_path):
    paths = shards(tmp_path)
    with SplitGGUFReader(paths) as reader:
        individual = [r.metadata_bytes for r in reader.readers]
    # Each individual metadata block fits; their combined charge does not.
    with pytest.raises(GGUFMemoryError):
        SplitGGUFReader(paths, max_metadata_bytes=max(individual) + 1)


def test_incomplete_or_conflicting_metadata_closes_every_reader(tmp_path, monkeypatch):
    from airllm.nextgen.gguf import GGUFReader
    paths = shards(tmp_path)
    opened = []
    original = GGUFReader.__init__
    def record(reader, *args, **kwargs):
        original(reader, *args, **kwargs)
        opened.append(reader)
    monkeypatch.setattr(GGUFReader, '__init__', record)
    _fixture(paths[1], [('b', (4,), 0, bytes(16))],
             [('split.no', 4, 1), ('split.count', 4, 2),
              ('split.tensors.count', 4, 2), ('general.architecture', 8, 'wrong')])
    with pytest.raises(GGUFError, match='Conflicting'):
        SplitGGUFReader(paths)
    assert len(opened) == 2
    for reader in opened:
        with pytest.raises(ValueError, match='closed'):
            reader.fileno()


def test_partial_initialization_closes_opened_shard(tmp_path, monkeypatch):
    from airllm.nextgen.gguf import GGUFReader
    paths = shards(tmp_path)
    paths[1].unlink()
    opened = []
    original = GGUFReader.__init__
    def record(reader, *args, **kwargs):
        original(reader, *args, **kwargs)
        opened.append(reader)
    monkeypatch.setattr(GGUFReader, '__init__', record)
    with pytest.raises(FileNotFoundError):
        SplitGGUFReader(paths)
    assert len(opened) == 1
    assert opened[0]._fd == -1


def test_close_is_idempotent_retained_arrays_remain_budgeted(tmp_path):
    reader = SplitGGUFReader(shards(tmp_path), memory_budget_bytes=15000)
    value = reader.read_rows('a', 0, 1)
    before = reader.available_memory_bytes
    reader.close()
    reader.close()
    assert reader.available_memory_bytes == before
    np.testing.assert_array_equal(value, 1)
    with pytest.raises(ValueError, match='closed'):
        reader.read_rows('b', 0, 1)
    with pytest.raises(ValueError, match='closed'):
        reader.filenos()
    with pytest.raises(ValueError, match='closed'):
        reader.__enter__()
    del value
    gc.collect()
    assert reader.available_memory_bytes == before + 4096


def test_concurrent_shards_do_not_borrow_each_others_reservations(tmp_path, monkeypatch):
    with SplitGGUFReader(shards(tmp_path), memory_budget_bytes=15000) as reader:
        entered, release = Event(), Event()
        owner = reader.owners['a']
        original = owner._pread
        def blocked(count, offset, *, tensor=False):
            if tensor:
                entered.set()
                assert release.wait(5)
            return original(count, offset, tensor=tensor)
        monkeypatch.setattr(owner, '_pread', blocked)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(reader.read_rows, 'a', 0, 1)
            assert entered.wait(5)
            second = pool.submit(reader.read_rows, 'b', 0, 1)
            release.set()
            value = first.result()
            with pytest.raises(GGUFMemoryError):
                second.result()
            np.testing.assert_array_equal(value, 1)


def test_packed_bytes_share_budget_across_shards(tmp_path):
    with SplitGGUFReader(shards(tmp_path), memory_budget_bytes=15000) as reader:
        value = reader.read_packed_rows('a', 0, 2, copy=False)
        view = memoryview(value)
        del value
        with pytest.raises(GGUFMemoryError):
            reader.read_rows('b', 0, 1)
        del view
        gc.collect()
        np.testing.assert_array_equal(reader.read_rows('b', 0, 1), 2)


@pytest.mark.parametrize('metadata', [[('split.no', 4, 0)],
    [('split.no', 7, False), ('split.count', 4, 1), ('split.tensors.count', 4, 0)]])
def test_partial_or_noninteger_split_fields_rejected(tmp_path, metadata):
    path = _fixture(tmp_path / 'single.gguf', [], metadata)
    with pytest.raises(GGUFError):
        SplitGGUFReader(path)
