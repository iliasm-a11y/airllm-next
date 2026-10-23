"""Worker decoding must not switch a concurrent forward's default GPU device."""
import json
import struct
from threading import Event, current_thread

import mlx.core as mx
import numpy as np
import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.quantization import FP8_LUT
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights


def checkpoint(root, kind):
    root.mkdir()
    shape = (3, 5)
    if kind == 'BF16':
        expected = np.arange(15, dtype='<f4').reshape(shape)
        raw = (expected.view('<u4') >> 16).astype('<u2')
        tensors = {'w': ('BF16', raw)}
    elif kind == 'FP8':
        raw = np.arange(15, dtype='u1').reshape(shape)
        scales = np.array([[1., 2.], [4., 8.]], dtype='<f4')
        expected = FP8_LUT[raw] * scales[(np.arange(3)//2)[:, None], (np.arange(5)//4)[None, :]]
        tensors = {'w': ('F8_E4M3', raw), 'w_scale_inv': ('F32', scales)}
    else:
        expected = np.arange(15, dtype='<f4').reshape(shape)
        tensors = {'w': ('F32', expected)}
    header, payload = {}, bytearray()
    for key, (dtype, value) in tensors.items():
        start = len(payload)
        payload.extend(value.tobytes())
        header[key] = {'dtype': dtype, 'shape': list(value.shape), 'data_offsets': [start, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (root/'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    return expected


@pytest.mark.skipif(not mx.metal.is_available(), reason='Concurrent CPU decoder and GPU forward require Metal')
@pytest.mark.parametrize('kind', ['F32', 'BF16', 'FP8'])
def test_real_worker_decode_keeps_main_thread_gpu_device_and_stream(tmp_path, monkeypatch, kind):
    expected = checkpoint(tmp_path/'cp', kind)
    ready, release = Event(), Event()
    main = current_thread()
    real_eval = mx.eval

    def paused_eval(*arrays, **kwargs):
        if current_thread() is not main:
            ready.set()
            assert release.wait(3), 'Main thread did not release the real decoder'
        return real_eval(*arrays, **kwargs)

    options = NextConfig(dtype='float32', device='gpu', cache_bytes=0, io_bytes=4096,
        max_tensor_bytes=4096, prefetch=False, load_tokenizer=False)
    with OffsetCheckpoint(tmp_path/'cp', max_read_bytes=4096) as source:
        reader = NativeWeights(source, mx, options, {'quantization_config': {'weight_block_size': [2, 4]}})
        stream = mx.new_stream(mx.gpu)
        try:
            monkeypatch.setattr(mx, 'eval', paused_eval)
            with mx.stream(stream):
                assert reader.io.submit('w', expected.nbytes, lambda: reader._load('w'))
                try:
                    assert ready.wait(3), 'Worker did not reach actual CPU decode evaluation'
                    # With a worker mx.stream(mx.cpu) context, this is CPU even
                    # though the main thread remains inside its GPU context.
                    assert mx.default_device() == mx.gpu
                    assert mx.default_stream(mx.gpu) == stream
                finally:
                    release.set()
                actual = reader.io.take('w')
                assert mx.default_device() == mx.gpu
                np.testing.assert_array_equal(np.asarray(actual), expected)
                assert reader.io.bytes == 0
        finally:
            release.set()
            reader.close()
