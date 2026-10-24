"""Scheduler contracts use labelled synthetic inference, never model quality."""
from dataclasses import replace
from hashlib import sha256
import json
from threading import Event, Lock
from time import sleep, time
import pytest

from airllm.nextgen.orchestrator import (
    InferenceResult, LocalOrchestrator, OrchestratorConfig, _json)


HEALTHY = {'available_bytes': 12 << 30, 'swap_used_bytes': 0,
           'pressure_level': 1, 'rss_bytes': 64 << 20}


class Backend:
    identity = 'test-only-state-dependent-backend-v1'
    max_concurrency = 1
    concurrent_safe = False
    kv_bytes_per_session = 1024

    def __init__(self):
        self.calls = []
        self.released = []
        self.on_advance = None

    def validate(self, request):
        if any(token >= 128 for token in request.input_ids + request.output_ids + request.eos_token_ids):
            raise ValueError('Vocabulary overflow')

    def advance(self, request, max_tokens):
        self.calls.append(request.job_id)
        if self.on_advance:
            self.on_advance(request)
        ids = list(request.input_ids + request.output_ids)
        new = []
        for _ in range(max_tokens):
            if request.stop_event.is_set():
                break
            token = (sum(ids)+len(ids)) % 128
            ids.append(token)
            new.append(token)
            if token in request.eos_token_ids:
                break
        return InferenceResult(new, not new, {'test_only': True})

    def release(self, job_id):
        self.released.append(job_id)


def config(**kwargs):
    return replace(OrchestratorConfig(), **{'quantum_tokens': 1, **kwargs})


def scheduler(tmp_path, backend=None, cfg=None, resources=None):
    return LocalOrchestrator(backend or Backend(), tmp_path,
                             config=cfg or config(), snapshot_provider=lambda: dict(resources or HEALTHY))


def drain(engine):
    while engine.status()['jobs_queued']:
        assert engine.step()


@pytest.mark.parametrize('count', [1, 10, 100, 500, 600])
def test_registered_waiting_active_and_completed_are_distinct(tmp_path, count):
    backend = Backend()
    with scheduler(tmp_path, backend) as engine:
        for index in range(count):
            engine.register(f'agent-{index}')
        assert engine.status()['registered_agents'] == count
        assert engine.status()['waiting_agents'] == engine.status()['active_agents'] == 0
        ids = [engine.submit(f'agent-{index}', [index % 100 + 1], max_tokens=1) for index in range(count)]
        before = engine.status()
        assert before['waiting_agents'] == count and before['inferences_in_flight'] == 0
        assert before['kv_reserved_bytes'] == 0
        def inspect(request):
            status = engine.status()
            assert status['active_agents'] == status['inferences_in_flight'] == 1
            assert status['kv_reserved_bytes'] == backend.kv_bytes_per_session
        backend.on_advance = inspect
        drain(engine)
        after = engine.status()
        assert after['jobs_completed'] == count
        assert after['waiting_agents'] == after['inferences_in_flight'] == after['active_agents'] == 0
        assert after['peak_simultaneous_inferences'] == 1
        assert after['model_instances'] == 1
        assert after['kv_reserved_bytes'] == 0
        assert set(ids) == set(backend.released)
        assert after['registry_reserved_bytes'] + after['transcript_reserved_bytes'] <= engine.config.max_transcript_bytes


def test_priority_round_robin_replay_and_isolation(tmp_path):
    with scheduler(tmp_path) as engine:
        engine.register('low')
        engine.register('high')
        low = engine.submit('low', [2], max_tokens=3, priority=-1)
        high = engine.submit('high', [7], max_tokens=3, priority=1)
        assert engine.step()[0]['id'] == high
        assert engine.step()[0]['id'] == high
        drain(engine)
        assert engine.snapshot(low)['output_ids'] == [3, 7, 15]
        assert engine.snapshot(high)['output_ids'] == [8, 17, 35]


def test_aging_eventually_serves_old_low_priority(tmp_path):
    with scheduler(tmp_path, cfg=config(aging_seconds=.01)) as engine:
        engine.register('old')
        engine.register('new')
        old = engine.submit('old', [1], max_tokens=1, priority=-1000)
        # A deterministic timestamp fixture avoids a ten-second sleep.
        engine.jobs[old]['queued_at'] -= 21
        engine.submit('new', [2], max_tokens=1, priority=1000)
        assert engine.step()[0]['id'] == old


def test_quota_memory_and_validation_rejections_are_pre_inference(tmp_path):
    backend = Backend()
    with scheduler(tmp_path, backend, cfg=config(max_agents=2, max_jobs=1)) as engine:
        engine.register('one', token_quota=2, job_quota=1)
        engine.register('two')
        with pytest.raises(MemoryError):
            engine.register('three')
        with pytest.raises(ValueError):
            engine.submit('one', [200], max_tokens=1)
        assert not backend.calls
        job = engine.submit('one', [1], max_tokens=2)
        with pytest.raises(MemoryError):
            engine.submit('two', [1], max_tokens=1)
        engine.cancel(job)
        engine.forget(job)
        assert engine.status()['transcript_reserved_bytes'] == 0
        with pytest.raises(MemoryError):
            engine.submit('one', [1], max_tokens=1)
        assert engine.agent_snapshot('one')['admitted_tokens'] == 2
    with scheduler(tmp_path, Backend(), cfg=config(max_agents=2, max_jobs=1)) as restored:
        assert restored.agent_snapshot('one')['admitted_jobs'] == 1
        with pytest.raises(MemoryError):
            restored.submit('one', [1], max_tokens=1)


def test_transcript_reservation_counts_future_output_and_registry(tmp_path):
    cfg = config(max_transcript_bytes=4096 + 16384 + 96*5)
    with scheduler(tmp_path, cfg=cfg) as engine:
        engine.register('one')
        engine.submit('one', [1, 2], max_tokens=3)
        with pytest.raises(MemoryError):
            engine.register('two')
        with pytest.raises(MemoryError):
            engine.submit('one', [1], max_tokens=1)


def test_restart_pauses_and_exact_output_resumes_without_hidden_generation(tmp_path):
    with scheduler(tmp_path) as engine:
        engine.register('one', tools=('read',))
        job = engine.submit('one', [1, 4], max_tokens=4)
        assert engine.step()[0]['output_ids'] == [7]
        engine.pause(job)
    backend = Backend()
    with scheduler(tmp_path, backend) as engine:
        assert engine.snapshot(job)['state'] == 'paused'
        assert engine.agent_snapshot('one')['tools'] == ['read']
        assert engine.step() == [] and not backend.calls
        engine.resume(job)
        drain(engine)
        assert engine.snapshot(job)['output_ids'] == [7, 15, 31, 63]


def test_directory_lock_identity_and_config_are_fail_closed(tmp_path):
    with scheduler(tmp_path) as engine:
        engine.register('one')
        with pytest.raises(BlockingIOError):
            scheduler(tmp_path)
    wrong = Backend()
    wrong.identity = 'different-weights'
    with pytest.raises(ValueError, match='mismatch'):
        scheduler(tmp_path, wrong)
    with pytest.raises(ValueError, match='mismatch'):
        scheduler(tmp_path, cfg=config(quantum_tokens=1, max_agents=100))


def test_journal_checksum_and_semantic_corruption_rejected(tmp_path):
    with scheduler(tmp_path) as engine:
        engine.register('one')
        engine.submit('one', [1], max_tokens=1)
    state = tmp_path / 'orchestrator.json'
    original = json.loads(state.read_text())
    bad = json.loads(state.read_text())
    bad['payload']['agents']['one']['token_quota'] = 0
    state.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='checksum'):
        scheduler(tmp_path)
    bad['sha256'] = sha256(_json(bad['payload'])).hexdigest()
    state.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        scheduler(tmp_path)
    state.write_text(json.dumps(original))
    with scheduler(tmp_path) as restored:
        assert restored.status()['jobs_paused'] == 1


@pytest.mark.parametrize('field,value,reason', [
    ('pressure_level', 2, 'memory_pressure'),
    ('available_bytes', 1, 'system_reserve'),
    ('swap_used_bytes', 129 << 20, 'swap_growth'),
    ('rss_bytes', (2 << 30)+1, 'rss_limit'),
])
def test_memory_pressure_latches_and_requires_explicit_recovery(tmp_path, field, value, reason):
    resources = dict(HEALTHY)
    backend = Backend()
    with scheduler(tmp_path, backend, resources=resources) as engine:
        engine.register('one')
        job = engine.submit('one', [1], max_tokens=2)
        resources[field] = value
        assert engine.step() == []
        assert not backend.calls
        assert engine.status()['stopped_reason'] == reason
        assert engine.snapshot(job)['state'] == 'paused'
        with pytest.raises(RuntimeError):
            engine.resume(job)
        with pytest.raises(MemoryError):
            engine.clear_pressure()
        resources.update(HEALTHY)
        engine.clear_pressure()
        assert engine.step() == []
        engine.resume_all()
        drain(engine)
        assert engine.snapshot(job)['state'] == 'completed'


def test_sampler_checks_pressure_during_owner_thread_inference(tmp_path):
    resources = dict(HEALTHY)
    backend = Backend()
    with scheduler(tmp_path, backend, cfg=config(sample_interval_seconds=.005), resources=resources) as engine:
        engine.register('one')
        job = engine.submit('one', [1], max_tokens=3)
        def wait_for_stop(request):
            resources['pressure_level'] = 2
            assert request.stop_event.wait(1)
        backend.on_advance = wait_for_stop
        engine.step()
        assert engine.status()['stopped_reason'] == 'memory_pressure'
        assert engine.status()['kv_reserved_bytes'] == 0
        assert engine.snapshot(job)['output_ids'] == []
        assert backend.released == [job]


class ParallelBackend(Backend):
    max_concurrency = 4
    concurrent_safe = True

    def advance(self, request, max_tokens):
        sleep(.02)
        return super().advance(request, max_tokens)


def test_adaptive_real_call_overlap_is_bounded_by_capability_and_global_kv(tmp_path):
    backend = ParallelBackend()
    cfg = config(max_concurrency=4, kv_budget_bytes=2*backend.kv_bytes_per_session, healthy_steps_to_grow=1)
    with scheduler(tmp_path, backend, cfg=cfg) as engine:
        for index in range(8):
            engine.register(f'a{index}')
            engine.submit(f'a{index}', [index+1], max_tokens=1)
        assert engine.status()['backend_capacity'] == 4
        assert engine.status()['effective_capacity'] == 2
        drain(engine)
        assert engine.status()['peak_simultaneous_inferences'] == 2
        assert engine.status()['jobs_completed'] == 8


def test_parallel_requires_explicit_backend_promise_and_kv_admission(tmp_path):
    backend = Backend()
    backend.max_concurrency = 2
    with pytest.raises(ValueError, match='concurrent_safe'):
        scheduler(tmp_path, backend)
    backend.max_concurrency = 1
    with pytest.raises(MemoryError, match='KV session'):
        scheduler(tmp_path, backend, cfg=config(kv_budget_bytes=1))


def test_failures_are_isolated_and_backend_metrics_are_bounded(tmp_path):
    backend = Backend()
    with scheduler(tmp_path, backend) as engine:
        engine.register('one')
        engine.register('two')
        failed = engine.submit('one', [1], max_tokens=1)
        good = engine.submit('two', [2], max_tokens=1)
        def fail_once(request):
            backend.on_advance = None
            raise OSError('private path and prompt must not be persisted')
        backend.on_advance = fail_once
        assert engine.step()[0]['state'] == 'failed'
        drain(engine)
        assert engine.snapshot(failed)['error'] == 'OSError'
        assert engine.snapshot(good)['state'] == 'completed'
        assert 'private path' not in (tmp_path/'orchestrator.json').read_text()
    class HugeMetrics(Backend):
        def advance(self, request, max_tokens):
            return InferenceResult([2], metrics={'huge': 'x'*9000})
    with scheduler(tmp_path/'huge', HugeMetrics()) as engine:
        engine.register('one')
        engine.submit('one', [1], max_tokens=1)
        assert engine.step()[0]['error'] == 'ValueError'


def test_release_failure_retains_kv_lease_and_latches_until_new_owner(tmp_path):
    class BadRelease(Backend):
        def release(self, job_id):
            raise OSError('not freed')
    with scheduler(tmp_path, BadRelease()) as engine:
        engine.register('one')
        job = engine.submit('one', [1], max_tokens=2)
        engine.step()
        status = engine.status()
        assert status['stopped_reason'] == 'backend_release_failed'
        assert status['kv_reserved_bytes'] == 1024
        assert status['inferences_in_flight'] == 0
        assert engine.snapshot(job)['state'] == 'paused'
        assert engine.snapshot(job)['output_ids'] == []
        with pytest.raises(RuntimeError, match='Close and restore'):
            engine.clear_pressure()
    with scheduler(tmp_path) as restored:
        restored.clear_pressure()
        restored.resume(job)
        drain(restored)
        assert restored.snapshot(job)['state'] == 'completed'


def test_journal_commit_failure_blocks_uncommitted_in_memory_continuation(tmp_path, monkeypatch):
    backend = Backend()
    with scheduler(tmp_path, backend) as engine:
        engine.register('one')
        job = engine.submit('one', [1], max_tokens=3)
        engine.step()
        committed = engine.snapshot(job)['output_ids']
        def fail():
            raise OSError('disk unavailable')
        monkeypatch.setattr(engine, '_write_snapshot', fail)
        with pytest.raises(OSError):
            engine.step()
        calls = len(backend.calls)
        assert engine.status()['stopped_reason'] == 'journal_commit_failed'
        assert engine.status()['durability_failed']
        assert engine.step() == []
        assert len(backend.calls) == calls
        with pytest.raises(RuntimeError, match='Close and restore'):
            engine.clear_pressure()
    with scheduler(tmp_path) as restored:
        assert restored.snapshot(job)['output_ids'] == committed
        assert restored.snapshot(job)['state'] == 'paused'
        restored.resume(job)
        drain(restored)
        assert restored.snapshot(job)['output_ids'] == [2, 5, 11]


def test_tools_require_both_registered_grant_and_healthy_resources(tmp_path):
    resources = dict(HEALTHY)
    class Broker:
        calls = []
        def execute(self, agent_id, tool, arguments):
            assert engine.status()['active_agents'] == engine.status()['active_tools'] == 1
            assert engine.status()['inferences_in_flight'] == 0
            self.calls.append((agent_id, tool, arguments))
            return {'ok': True}
    broker = Broker()
    with scheduler(tmp_path, resources=resources) as engine:
        engine.register('one', tools=('read',))
        with pytest.raises(PermissionError):
            engine.execute_tool('one', broker, 'write', {})
        assert not broker.calls
        assert engine.execute_tool('one', broker, 'read', {'name': 'fixture'}) == {'ok': True}
        assert engine.status()['tool_calls'] == 1
        assert engine.status()['peak_simultaneous_inferences'] == 0
        resources['pressure_level'] = 2
        with pytest.raises(RuntimeError):
            engine.execute_tool('one', broker, 'read', {})
        assert len(broker.calls) == 1
        assert engine.status()['stopped_reason'] == 'memory_pressure'


def test_stale_atomic_temporaries_are_removed_only_under_exclusive_lock(tmp_path):
    stale = tmp_path/'.state-abandoned.tmp'
    stale.write_bytes(b'unpublished')
    with scheduler(tmp_path) as engine:
        assert not stale.exists()
        stale.write_bytes(b'active writer')
        with pytest.raises(BlockingIOError):
            scheduler(tmp_path)
        assert stale.exists()
    with scheduler(tmp_path):
        assert not stale.exists()


def test_raw_input_and_eos_lengths_rejected_before_deduplication(tmp_path):
    with scheduler(tmp_path, cfg=config(max_context_tokens=10)) as engine:
        engine.register('one')
        with pytest.raises(MemoryError, match='Input token list'):
            engine.submit('one', [1]*11)
        with pytest.raises(MemoryError, match='EOS token list'):
            engine.submit('one', [1], max_tokens=1, eos_token_ids=[0]*257)
        assert not engine.backend.calls


def test_parallel_ttft_uses_each_invocation_start_not_shared_batch_commit(tmp_path):
    class Asymmetric(ParallelBackend):
        starts = {}
        def advance(self, request, max_tokens):
            self.starts[request.job_id] = time()
            sleep(.01 if request.agent_id == 'fast' else .10)
            return InferenceResult([2], metrics={'first_token_seconds': .001})
    backend = Asymmetric()
    cfg = config(max_concurrency=2, kv_budget_bytes=2048, healthy_steps_to_grow=1)
    with scheduler(tmp_path, backend, cfg=cfg) as engine:
        engine.register('fast')
        engine.register('slow')
        fast = engine.submit('fast', [1], max_tokens=1)
        slow = engine.submit('slow', [1], max_tokens=1)
        engine.step()
        for job_id in (fast, slow):
            job = engine.snapshot(job_id)
            expected = backend.starts[job_id]-job['created_at']+.001
            assert job['metrics']['ttft_seconds'] == pytest.approx(expected, abs=.005)
            assert not job['metrics']['backend_metrics']['ttft_is_quantum_upper_bound']


def test_metric_graph_and_unicode_payloads_cannot_exceed_reserved_python_memory(tmp_path):
    class BadMetrics(Backend):
        def advance(self, request, max_tokens):
            return InferenceResult([2], metrics={'text': '\U0001f40d'*128})
    with scheduler(tmp_path, BadMetrics()) as engine:
        engine.register('one')
        engine.submit('one', [1], max_tokens=1)
        assert engine.step()[0]['error'] == 'ValueError'
