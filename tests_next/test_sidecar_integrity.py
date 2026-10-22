import hashlib
import json
import numpy as np
import pytest
from safetensors.numpy import load_file
from airllm.nextgen.sidecar import QuantizedSidecar
from airllm.nextgen.quantization import dequant_affine


def test_streamed_sidecar_matches_safetensors_and_hash(tmp_path):
    rng = np.random.default_rng(140)
    w = rng.normal(size=(101, 128)).astype('f4')
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    assert store.add('expert', (w[i:i+7] for i in range(0, 101, 7)), w.shape)
    entry = store.entries['expert']
    path = tmp_path / entry['file']
    arrays = load_file(path)
    packed = store.load('expert')
    for name in ('weight', 'scales', 'biases'):
        np.testing.assert_array_equal(arrays[name], getattr(packed, name))
    np.testing.assert_array_equal(dequant_affine(store.load('expert', rows=(5, 13))), dequant_affine(packed)[5:13])
    assert entry['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.verify() == ['expert']
    assert store.stats['integrity_bytes'] == path.stat().st_size
    store.verify()
    assert store.stats['integrity_bytes'] == path.stat().st_size
    assert entry['peak_conversion_tile_bytes'] < w.nbytes


def test_resume_detects_corruption_and_keeps_manifest(tmp_path):
    w = np.ones((16, 64), dtype='f4')
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    store.add('expert', [w], w.shape)
    path = tmp_path / store.entries['expert']['file']
    with path.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'\xff')
    resumed = QuantizedSidecar(tmp_path, 'source', create=True)
    with pytest.raises(ValueError, match='checksum mismatch'):
        resumed.add('expert', [w], w.shape)
    assert json.loads((tmp_path/'manifest.json').read_text())['entries'] == store.entries


def test_interrupted_conversion_never_commits_partial_matrix(tmp_path):
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    def interrupted():
        yield np.ones((4, 64), dtype='f4')
        raise OSError('disk disconnected')
    with pytest.raises(OSError):
        store.add('expert', interrupted(), (32, 64))
    assert not store.entries
    assert not list(tmp_path.glob('*.safetensors'))
    assert not list(tmp_path.glob('.convert-*'))


def test_sidecar_rejects_large_request_before_read(tmp_path):
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    w = np.ones((32,64), dtype='f4')
    store.add('expert', [w], w.shape)
    with pytest.raises(MemoryError, match='before allocation'):
        store.load('expert', max_bytes=10)
    assert store.stats['payload_read_bytes'] == 0


def test_sidecar_rejects_forged_shape_metadata(tmp_path):
    import struct
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    w = np.ones((32,64), dtype='f4')
    store.add('expert', [w], w.shape)
    path = tmp_path / store.entries['expert']['file']
    with path.open('r+b') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
        header['weight']['shape'][0] += 1
        raw = json.dumps(header,separators=(',',':')).encode()
        assert len(raw) <= length
        stream.seek(8)
        stream.write(raw+b' '*(length-len(raw)))
    with pytest.raises(ValueError,match='metadata'):
        store.load('expert')


@pytest.mark.parametrize('tile', [np.ones((3, 32)), np.ones((2, 2, 64)), np.ones((0,64))])
def test_bad_tile_is_rejected(tmp_path, tile):
    store = QuantizedSidecar(tmp_path, 'source', create=True)
    with pytest.raises(ValueError):
        store.add('bad', [tile], (4, 64))
    assert not store.entries
