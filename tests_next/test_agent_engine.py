"""Agent scheduling contracts, then native numerical integration separately."""
from collections import defaultdict
from contextlib import nullcontext
from hashlib import sha256
import json
from threading import RLock, Thread
from types import SimpleNamespace
import numpy as np
import pytest

from airllm.nextgen.agent_engine import AgentEngine, _canonical


class FakeMX:
    __version__ = 'test-only'
    stream = staticmethod(lambda stream: nullcontext())
    argmax = staticmethod(np.argmax)


class FakeModel:
    """State-dependent deterministic stub; never evidence of trained quality."""
    config = {'vocab_size': 64, 'max_position_embeddings': 128, 'eos_token_id': None}
    options = SimpleNamespace(max_context=128, dtype='float32', prefill_chunk=4, row_chunk=32)
    mx = FakeMX()
    stream = None
    device = 'stub'

    def __init__(self):
        self.lock = RLock()
        self.checkpoint = SimpleNamespace(fingerprint='unit-test-fingerprint', stats=defaultdict(int))
        self.calls = []
        self.fail_at = None
        self.sessions = []

    def forward(self, ids, past_key_values=None, logits_to_keep=1):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            self.fail_at = None
            raise OSError('private-checkpoint-path')
        session = past_key_values
        if session is None:
            session = SimpleNamespace(tokens=[], bytes=128, peak_bytes=128)
            self.sessions.append(session)
        session.tokens.extend(ids[0])
        self.calls.append(list(ids[0]))
        self.checkpoint.stats['logical_bytes'] += 100*len(ids[0])
        token = (sum(session.tokens)+len(session.tokens))%64
        logits = np.full((1, 1, 64), -1.)
        logits[0, 0, token] = 1.
        return SimpleNamespace(logits=logits, past_key_values=session)


def run_to_end(engine):
    while engine.status()['queued']:
        engine.step()


def test_priority_round_robin_isolation_and_call_boundaries(tmp_path):
    model = FakeModel()
    with AgentEngine(model, state_dir=tmp_path, quantum_tokens=1) as engine:
        lower = engine.submit([1, 3], max_tokens=3, priority=-1)
        first = engine.submit([2, 4], max_tokens=3)
        second = engine.submit([5, 7], max_tokens=3)
        assert engine.step()['id'] == first
        assert engine.step()['id'] == second
        assert engine.step()['id'] == first
        # Reactivation replays the original prompt and feeds its single output.
        assert model.calls[:4] == [[2, 4], [5, 7], [2, 4], [8]]
        run_to_end(engine)
        for job_id, prompt in ((lower, [1, 3]), (first, [2, 4]), (second, [5, 7])):
            reference = FakeModel()
            with AgentEngine(reference) as serial:
                expected = serial.submit(prompt, max_tokens=3)
                run_to_end(serial)
                assert engine.snapshot(job_id)['output_ids'] == serial.snapshot(expected)['output_ids']
        assert engine.status()['active_contexts'] == 0
        assert all(snapshot['state'] == 'completed' for snapshot in (engine.snapshot(j) for j in engine.jobs))


def test_durable_restart_requires_resume_and_exact_replay(tmp_path):
    with AgentEngine(FakeModel(), state_dir=tmp_path, quantum_tokens=2) as engine:
        job_id = engine.submit([1, 7, 9], max_tokens=5)
        committed = engine.step()['output_ids']
        assert len(committed) == 2
        identity = engine.identity
    model = FakeModel()
    with AgentEngine(model, state_dir=tmp_path, quantum_tokens=2) as engine:
        assert engine.identity == identity
        assert engine.snapshot(job_id)['state'] == 'paused'
        assert engine.step() is None
        engine.resume(job_id)
        run_to_end(engine)
        assert engine.snapshot(job_id)['output_ids'][:2] == committed
        assert model.calls[:3] == [[1, 7, 9], [committed[0]], [committed[1]]]
        assert engine.snapshot(job_id)['metrics']['replayed_tokens'] == 4


def test_cancel_pause_resume_forget_and_reservations(tmp_path):
    with AgentEngine(FakeModel(), state_dir=tmp_path, quantum_tokens=1, max_requests=1) as engine:
        job_id = engine.submit([1], max_tokens=4)
        engine.step()
        assert engine.status()['active_contexts'] == 1
        engine.pause(job_id)
        assert engine.status()['active_contexts'] == 0
        assert engine.step() is None
        engine.resume(job_id)
        engine.cancel(job_id)
        with pytest.raises(MemoryError):
            engine.submit([1], max_tokens=1)
        engine.resume(job_id)
        run_to_end(engine)
        with pytest.raises(ValueError):
            engine.resume(job_id)
        engine.forget(job_id)
        assert engine.status()['requests'] == 0
        assert engine.status()['transcript_reserved_bytes'] == 0
        assert engine.status()['journal_reserved_bytes'] == 0
        assert not (tmp_path/(job_id+'.json')).exists()


def test_failed_quantum_discards_uncommitted_tokens_and_retries(tmp_path):
    model = FakeModel()
    with AgentEngine(model, state_dir=tmp_path, quantum_tokens=3) as engine:
        job_id = engine.submit([1, 3], max_tokens=6)
        committed = engine.step()['output_ids']
        model.fail_at = len(model.calls)+1
        failed = engine.step()
        assert failed['state'] == 'failed' and failed['error'] == 'OSError'
        assert failed['output_ids'] == committed
        assert failed['metrics']['discarded_tokens'] == 1
        assert failed['metrics']['failed_quanta'] == 1
        assert 'private' not in json.dumps(failed)
        engine.resume(job_id)
        run_to_end(engine)
        actual = engine.snapshot(job_id)['output_ids']
    with AgentEngine(FakeModel()) as serial:
        reference_id = serial.submit([1, 3], max_tokens=6)
        run_to_end(serial)
        assert actual == serial.snapshot(reference_id)['output_ids']


def test_interrupted_journal_write_keeps_last_durable_transcript(tmp_path, monkeypatch):
    import airllm.nextgen.agent_engine as module
    model = FakeModel()
    with AgentEngine(model, state_dir=tmp_path, quantum_tokens=1) as engine:
        job_id = engine.submit([1], max_tokens=3)
        before = (tmp_path/(job_id+'.json')).read_bytes()
        original = module.os.replace
        monkeypatch.setattr(module.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('disk disconnected')))
        result = engine.step()
        assert result['state'] == 'failed' and result['error'] == 'journal_write_failed'
        assert result['output_ids'] == []
        assert (tmp_path/(job_id+'.json')).read_bytes() == before
        assert not list(tmp_path.glob('.job-*.tmp'))
        monkeypatch.setattr(module.os, 'replace', original)
    with AgentEngine(FakeModel(), state_dir=tmp_path) as engine:
        assert engine.snapshot(job_id)['output_ids'] == []
        engine.resume(job_id)
        run_to_end(engine)
        assert engine.snapshot(job_id)['state'] == 'completed'


def test_journal_model_mismatch_corruption_and_replay_mismatch(tmp_path):
    with AgentEngine(FakeModel(), state_dir=tmp_path) as engine:
        job_id = engine.submit([1], max_tokens=3)
        engine.step(max_tokens=1)
    other = FakeModel()
    other.checkpoint.fingerprint = 'different-model'
    with pytest.raises(ValueError, match='identity'):
        AgentEngine(other, state_dir=tmp_path)
    path = tmp_path/(job_id+'.json')
    record = json.loads(path.read_text())
    record['payload']['job']['output_ids'][0] = 61
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='checksum'):
        AgentEngine(FakeModel(), state_dir=tmp_path)
    # Correct checksum cannot make a false greedy history valid.
    record['sha256'] = sha256(_canonical(record['payload'])).hexdigest()
    path.write_text(json.dumps(record))
    with AgentEngine(FakeModel(), state_dir=tmp_path) as engine:
        engine.resume(job_id)
        result = engine.step()
        assert result['state'] == 'failed' and result['error'] == 'replay_token_mismatch'
        assert result['output_ids'] == [61]


def test_process_lock_budget_limits_and_stale_temp_cleanup(tmp_path):
    stale = tmp_path/'.job-interrupted.tmp'
    stale.write_bytes(b'uncommitted')
    with AgentEngine(FakeModel(), state_dir=tmp_path, max_transcript_bytes=5000) as engine:
        assert not stale.exists()
        with pytest.raises(BlockingIOError):
            AgentEngine(FakeModel(), state_dir=tmp_path)
        with pytest.raises(MemoryError, match='transcript'):
            engine.submit([1], max_tokens=20)
    with AgentEngine(FakeModel(), state_dir=tmp_path, max_journal_bytes=10000) as engine:
        with pytest.raises(MemoryError, match='disk reservation'):
            engine.submit([1], max_tokens=2)
    with AgentEngine(FakeModel(), state_dir=tmp_path, max_journal_write_bytes=100) as engine:
        with pytest.raises(MemoryError, match='cumulative'):
            engine.submit([1], max_tokens=2)
        assert engine.status()['requests'] == 0


@pytest.mark.parametrize('update', [{'input_ids':[-1]}, {'input_ids':[True]}, {'input_ids':[64]},
    {'input_ids':[2**32]}, {'max_tokens':True}, {'priority':True}, {'eos_token_id':[True]},
    {'agent_id':''}, {'agent_id':'\ud800'}, {'max_tokens':0}, {'max_tokens':128}, {'eos_token_id':[1]*257}])
def test_invalid_requests_fail_without_model_io(update):
    model = FakeModel()
    with AgentEngine(model) as engine:
        payload = {'input_ids':[1], 'max_tokens':4} | update
        with pytest.raises((ValueError, MemoryError)):
            engine.submit(**payload)
        assert not model.calls


def test_owner_thread_is_explicit():
    with AgentEngine(FakeModel()) as engine:
        errors = []
        def other_thread():
            try:
                engine.submit([1])
            except Exception as error:
                errors.append(error)
        worker = Thread(target=other_thread)
        worker.start(); worker.join()
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert engine.status()['requests'] == 0


@pytest.mark.parametrize('kind', ['glm', 'llama', 'qwen'])
def test_native_architecture_sessions_interleave_and_restart_with_ordinary_tokens(tmp_path, kind):
    from conftest import save_tiny
    from test_native import options
    from airllm.nextgen.runtime import AirLLMNextModel
    from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
    from airllm.nextgen.qwen_runtime import ExperimentalQwenModel
    if kind == 'qwen':
        from test_qwen_runtime import make_qwen
        root, _, config = make_qwen(tmp_path)
        model_class = ExperimentalQwenModel
    else:
        root = tmp_path/'model'
        save_tiny(root, kind)
        config = options(tmp_path, prefill_chunk=2)
        model_class = AirLLMNextModel if kind == 'glm' else AirLLMNextLlamaModel
    prompts = [[1, 7, 8], [1, 9, 11, 6]]
    with model_class(root, next_config=config) as model:
        expected = [model.generate([prompt], max_new_tokens=5, eos_token_id=[]).tolist()[0][len(prompt):]
                    for prompt in prompts]
        with AgentEngine(model, state_dir=tmp_path/'journals', quantum_tokens=1) as engine:
            ids = [engine.submit(prompt, max_tokens=5, eos_token_id=[]) for prompt in prompts]
            engine.step(); engine.step(); engine.step()
            assert engine.status()['max_resident_contexts'] == 1
            original_mlp = model.mlps[-1]
            def interrupted(_):
                raise OSError('simulated native removable-drive interruption')
            model.mlps[-1] = interrupted
            try:
                failed = engine.step()
                assert failed['state'] == 'failed' and failed['error'] == 'OSError'
            finally:
                model.mlps[-1] = original_mlp
        with AgentEngine(model, state_dir=tmp_path/'journals', quantum_tokens=1) as engine:
            for job_id in ids:
                assert engine.snapshot(job_id)['state'] in ('paused', 'failed')
                engine.resume(job_id)
            run_to_end(engine)
            assert [engine.snapshot(job_id)['output_ids'] for job_id in ids] == expected
            assert model.reader.io.peak_bytes <= config.io_bytes
            assert model.reader.cache.peak_bytes <= config.cache_bytes
            assert max(engine.snapshot(job_id)['metrics']['resident_context_peak_bytes'] for job_id in ids) <= config.kv_bytes


def test_three_hundred_logical_jobs_share_one_model_and_no_unstarted_contexts():
    model = FakeModel()
    with AgentEngine(model, max_requests=300) as engine:
        ids = [engine.submit([1], max_tokens=1, agent_id=f'agent-{i}') for i in range(300)]
        assert len(set(ids)) == 300
        assert engine.status()['queued'] == 300
        assert engine.status()['active_contexts'] == 0
        assert engine.status()['max_resident_contexts'] == 1
        assert not model.calls
        with pytest.raises(MemoryError, match='capacity'):
            engine.submit([1], max_tokens=1)
        assert engine.step()['state'] == 'completed'
        assert len(model.sessions) == 1
        assert engine.status()['queued'] == 299
