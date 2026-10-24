from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.orchestrator_native import NativeModelBackend
from tests_next.test_agent_engine import FakeModel, FakeMX


def model():
    native = FakeModel()
    native.options = SimpleNamespace(**vars(native.options), kv_bytes=4096)
    native.mx = SimpleNamespace(stream=FakeMX.stream, argmax=np.argmax,
                               all=np.all, isfinite=np.isfinite)
    return native


def request(output=(), prompt=(1, 3), stop=None):
    return SimpleNamespace(job_id='job', agent_id='agent', input_ids=prompt,
        output_ids=output, max_tokens=4, eos_token_ids=(), stop_event=stop or Event())


def test_native_replay_preserves_serial_tokens_and_one_model():
    native = model()
    backend = NativeModelBackend(native)
    complete = backend.advance(request(), 4)
    native.calls.clear()
    first = backend.advance(request(), 2)
    second = backend.advance(request(tuple(first.token_ids)), 2)
    assert first.token_ids+second.token_ids == complete.token_ids
    assert native.calls[2:] == [[1, 3], [complete.token_ids[0]],
                               [complete.token_ids[1]], [complete.token_ids[2]]]
    assert backend.model is native and backend.max_concurrency == 1
    assert backend._cache is None
    assert second.metrics['replayed_tokens'] == 3


def test_native_replay_mismatch_fails_closed_and_releases():
    backend = NativeModelBackend(model())
    with pytest.raises(RuntimeError, match='replay_token_mismatch'):
        backend.advance(request((63,)), 1)
    assert backend._cache is None and backend._active is None


def test_native_identity_covers_reader_and_configuration():
    native = model()
    native.reader = SimpleNamespace(kernel='reference', metal_simd_groups=1)
    before = NativeModelBackend(native).identity
    native.reader.kernel = 'metal'
    assert NativeModelBackend(native).identity != before
    native.reader.kernel = 'reference'
    assert NativeModelBackend(native).identity == before


def test_gguf_descriptor_identity_is_stable_and_rejects_mutation(tmp_path):
    import os
    from airllm.nextgen.gguf import GGUFReader
    from tests_next.test_gguf import _fixture
    path = _fixture(tmp_path/'model.gguf', [])
    native = model()
    with GGUFReader(path) as checkpoint:
        native.checkpoint = checkpoint
        backend = NativeModelBackend(native)
        assert backend.identity == NativeModelBackend(native).identity
        backend.validate(request())
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000))
        with pytest.raises(RuntimeError, match='changed'):
            backend.validate(request())
        assert backend.identity != NativeModelBackend(native).identity


def test_native_context_vocab_finite_and_cancellation():
    native = model()
    backend = NativeModelBackend(native)
    with pytest.raises(ValueError):
        backend.advance(request(prompt=(64,)), 1)
    stop = Event()
    stop.set()
    assert backend.advance(request(stop=stop), 1).token_ids == []
    assert native.calls == []
    original = native.forward
    def nonfinite(*args, **kwargs):
        out = original(*args, **kwargs)
        out.logits[0, 0, 0] = np.nan
        return out
    native.forward = nonfinite
    with pytest.raises(FloatingPointError):
        backend.advance(request(), 1)
    assert backend._cache is None
