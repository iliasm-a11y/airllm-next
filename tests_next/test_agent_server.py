"""Async local job HTTP lifecycle and thread ownership, using explicit stubs."""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from queue import Queue
from threading import Thread
import pytest

from test_agent_engine import FakeModel
from airllm.nextgen.agent_server import make_agent_server


class Tokenizer:
    def encode(self, text): return [1, 3]
    def decode(self, ids, **kwargs): return ','.join(map(str, ids))


@contextmanager
def running(tmp_path, **kwargs):
    ready = Queue()
    def owner():
        model = FakeModel()
        model.tokenizer = Tokenizer()
        server = None
        try:
            server = make_agent_server(model, state_dir=tmp_path, port=0,
                                       quantum_tokens=1, max_new_tokens=64, **kwargs)
            ready.put(server)
            server.serve_forever(.001)
        except BaseException as error:
            ready.put(error)
        finally:
            if server is not None:
                server.server_close()
    worker = Thread(target=owner, daemon=True)
    worker.start()
    server = ready.get(timeout=5)
    if isinstance(server, Exception):
        raise server
    try:
        yield server
    finally:
        server.shutdown()
        worker.join(5)
        assert not worker.is_alive()
        assert ready.empty()


def request(server, method, path, payload=None, *, raw=None):
    client = HTTPConnection('127.0.0.1', server.server_address[1], timeout=3)
    body = json.dumps(payload).encode() if raw is None and payload is not None else raw
    try:
        client.request(method, path, body=body,
                       headers={'Content-Type':'application/json'} if body is not None else {})
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def test_async_jobs_cancel_resume_completion_and_forget(tmp_path):
    with running(tmp_path) as server:
        assert request(server, 'GET', '/v1/models')[0] == 200
        status, job = request(server, 'POST', '/v1/agent/jobs',
            {'model':'airllm-next', 'input_ids':[1, 3], 'max_tokens':64, 'agent_id':'RONIN-A'})
        assert status == 202 and job['state'] == 'queued'
        job_id = job['id']
        status, cancelled = request(server, 'POST', f'/v1/agent/jobs/{job_id}/cancel', {})
        assert status == 200 and cancelled['state'] == 'cancelled'
        assert request(server, 'GET', '/health')[1]['active_contexts'] == 0
        assert request(server, 'POST', f'/v1/agent/jobs/{job_id}/resume', {})[0] == 200
        for _ in range(100):
            status, job = request(server, 'GET', f'/v1/agent/jobs/{job_id}')
            if job['state'] == 'completed':
                break
        assert status == 200 and job['state'] == 'completed'
        assert len(job['output_ids']) == 64 and job['output_text']
        assert request(server, 'DELETE', f'/v1/agent/jobs/{job_id}')[0] == 200
        assert request(server, 'GET', f'/v1/agent/jobs/{job_id}')[0] == 404
        assert request(server, 'GET', '/health')[1]['requests'] == 0


def test_prompt_and_request_validation_capacity(tmp_path):
    with running(tmp_path, max_requests=1, max_body_bytes=1024) as server:
        for update in ({'model':'remote'}, {'temperature':.1}, {'tools':[]}, {'max_tokens':True},
                       {'eos_token_id':[1]*257}):
            assert request(server, 'POST', '/v1/agent/jobs',
                {'model':'airllm-next', 'prompt':'hello'} | update)[0] == 400
        assert request(server, 'POST', '/v1/agent/jobs', raw=b'{"model":"a","model":"b"}')[0] == 400
        assert request(server, 'POST', '/v1/agent/jobs', raw=b'x'*1025)[0] == 429
        status, job = request(server, 'POST', '/v1/agent/jobs',
                             {'model':'airllm-next', 'prompt':'hello', 'max_tokens':2})
        assert status == 202
        assert request(server, 'POST', '/v1/agent/jobs',
                       {'model':'airllm-next', 'prompt':'more'})[0] == 429
        assert request(server, 'GET', '/v1/agent/jobs')[1]['data'][0]['id'] == job['id']
        assert request(server, 'GET', '/not-a-path')[0] == 404


def test_agent_loopback_and_explicit_journal_directory(tmp_path):
    for host in ('0.0.0.0', '192.168.1.2', 'example.com', '::1'):
        with pytest.raises(ValueError):
            make_agent_server(FakeModel(), state_dir=tmp_path, host=host)
    with pytest.raises(ValueError, match='state_dir'):
        make_agent_server(FakeModel(), state_dir=None)
