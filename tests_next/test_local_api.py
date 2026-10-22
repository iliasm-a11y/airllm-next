"""HTTP contract tests; stub cases do not certify model quality or performance."""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from threading import Thread
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.server import make_server


class TokenizerStub:
    chat_template = 'test-only'

    def encode(self, text):
        return [1, 8]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 8, 9]

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(map(str, ids))


class ModelStub:
    tokenizer = TokenizerStub()
    options = SimpleNamespace(max_context=128)
    config = {'max_position_embeddings': 128, 'eos_token_id': 2}

    def __init__(self):
        self.calls = []

    def generate(self, ids, **kwargs):
        self.calls.append((ids, kwargs))
        return np.array([ids[0]+[7, 2]])


@contextmanager
def running(model=None, **kwargs):
    server = make_server(model or ModelStub(), port=0, **kwargs)
    worker = Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def request(server, method, path, payload=None, raw=None, headers=None):
    client = HTTPConnection('127.0.0.1', server.server_address[1], timeout=3)
    try:
        body = json.dumps(payload).encode() if raw is None and payload is not None else raw
        client.request(method, path, body=body,
                       headers=headers or ({'Content-Type': 'application/json'} if body is not None else {}))
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def test_models_and_completions_contract():
    model = ModelStub()
    with running(model) as server:
        status, data = request(server, 'GET', '/v1/models')
        assert status == 200 and data['data'][0]['id'] == 'airllm-next'
        status, data = request(server, 'POST', '/v1/completions',
                               {'model': 'airllm-next', 'prompt': 'hello', 'max_tokens': 4})
        assert status == 200 and data['object'] == 'text_completion'
        assert data['choices'][0]['text'] == '7 2'
        assert data['choices'][0]['finish_reason'] == 'stop'
        assert data['usage'] == {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}
        assert model.calls == [([[1, 8]], {'max_new_tokens': 4, 'temperature': 0.})]


def test_chat_uses_local_template():
    model = ModelStub()
    with running(model) as server:
        status, data = request(server, 'POST', '/v1/chat/completions',
                               {'model': 'airllm-next', 'messages': [{'role': 'user', 'content': 'hello'}]})
        assert status == 200
        assert data['choices'][0]['message'] == {'role': 'assistant', 'content': '7 2'}
        assert model.calls[0][0] == [[1, 8, 9]]


def test_greedy_only_backend_rejects_sampling_as_request_error():
    model = ModelStub()
    model.supports_sampling = False
    with running(model) as server:
        status, data = request(server, 'POST', '/v1/completions',
            {'model':'airllm-next','prompt':'hello','temperature':.7})
        assert status == 400
        assert 'greedy' in data['error']['message']
        assert not model.calls


@pytest.mark.parametrize('extra', [
    {'stream': True}, {'stream': 1}, {'n': 2}, {'n': True}, {'tools': []},
    {'temperature': float('nan')}, {'temperature': -1}, {'temperature': True},
    {'max_tokens': True}, {'max_tokens': 0}, {'max_tokens': 513}, {'model': 'cloud-model'},
])
def test_unsupported_and_invalid_options_fail_before_inference(extra):
    model = ModelStub()
    with running(model) as server:
        payload = {'model': 'airllm-next', 'prompt': 'hello'} | extra
        status, data = request(server, 'POST', '/v1/completions', payload)
        assert status == 400 and 'error' in data
        assert model.calls == []


def test_body_limits_context_and_bad_json():
    model = ModelStub()
    with running(model, max_body_bytes=1024) as server:
        assert request(server, 'POST', '/v1/completions', raw=b'x'*1025)[0] == 413
        assert request(server, 'POST', '/v1/completions', raw=b'{')[0] == 400
        assert request(server, 'POST', '/v1/completions', raw=b'{}',
                       headers={'Content-Type': 'text/plain'})[0] == 415
        assert request(server, 'POST', '/v1/completions',
                       {'model': 'airllm-next', 'prompt': 'hello', 'max_tokens': 128})[0] == 400
        assert model.calls == []


def test_loopback_and_tokenizer_required():
    for host in ('0.0.0.0', '192.168.1.4', 'example.com', '::'):
        with pytest.raises(ValueError, match='loopback'):
            make_server(ModelStub(), host=host)
    model = ModelStub(); model.tokenizer = None
    with pytest.raises(ValueError, match='tokenizer'):
        make_server(model)


def test_internal_failure_does_not_expose_diagnostics():
    model = ModelStub()
    def broken(*args, **kwargs):
        raise RuntimeError('/private/checkpoint/secret-path')
    model.generate = broken
    with running(model) as server:
        status, data = request(server, 'POST', '/v1/completions',
                               {'model': 'airllm-next', 'prompt': 'hello'})
        assert status == 500 and 'secret-path' not in json.dumps(data)


def test_endpoint_matches_actual_tiny_glm_generation(tiny_glm, tmp_path):
    """Real miniature GLM numerics over HTTP; no checkpoint-scale speed claim."""
    from airllm.nextgen import AirLLMNextModel, NextConfig
    root, _ = tiny_glm
    options = NextConfig(cache_bytes=4*1024**2, io_bytes=1024**2, kv_bytes=1024**2,
        max_tensor_bytes=1024**2, reserve_bytes=0, workspace_bytes=1024**2,
        device='cpu', dtype='float32', prefill_chunk=2, max_context=128,
        load_tokenizer=False, cache_dir=str(tmp_path/'metadata'))
    # MLX stream handles are thread-affine. The production CLI constructs the
    # model and serves on one thread; exercise the same lifecycle here.
    results = Queue()
    def worker():
        try:
            with AirLLMNextModel(root, next_config=options) as model:
                model.tokenizer = TokenizerStub()
                expected = model.generate([[1, 8]], max_new_tokens=3).tolist()[0][2:]
                with make_server(model, port=0, max_new_tokens=8) as server:
                    results.put((server, expected))
                    server.serve_forever(poll_interval=.01)
        except BaseException as error:
            results.put(error)
    thread = Thread(target=worker, daemon=True); thread.start()
    value = results.get(timeout=30)
    if isinstance(value, BaseException):
        raise value
    server, expected = value
    try:
        status, data = request(server, 'POST', '/v1/completions',
                               {'model': 'airllm-next', 'prompt': 'test', 'max_tokens': 3})
        assert status == 200
        assert data['choices'][0]['text'] == ' '.join(map(str, expected))
        assert data['usage']['completion_tokens'] == len(expected)
    finally:
        server.shutdown(); thread.join(10)
    assert not thread.is_alive()
