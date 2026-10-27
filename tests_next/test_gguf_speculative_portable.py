"""Portable control-flow tests: no MLX import or trained-model speed claim.

An independently stepped toy recurrent target detects stale DeltaNet, conv and
KV state after every rejection position. Real GGUF numerical parity is a
separate required experiment, not established by this file.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest
import weakref

from airllm.nextgen.gguf_speculative import (SnapshotVerifier,
                                           generate_gguf_speculative)
from airllm.nextgen.prompt_draft import PromptSpanDraft


@dataclass
class Snapshot:
    owner: object
    position: int
    banks: dict
    nbytes: int
    active: bool = True


class HybridSession:
    def __init__(self, owner):
        self.owner = owner
        self.position = 0
        self.limit = 256
        self.banks = {'kv': (), 'conv': (), 'recurrent': (0,)}
        self.retained = 0
        self.peak_bytes = 0

    @property
    def bytes(self):
        return 4*sum(map(len, self.banks.values()))

    def snapshot(self):
        if (self.owner.snapshot_budget_bytes is not None and
                self.bytes+self.retained+self.bytes > self.owner.snapshot_budget_bytes):
            raise MemoryError('Qwen rollback snapshot exceeds kv_bytes')
        snapshot = Snapshot(self, self.position, dict(self.banks), self.bytes)
        self.retained += self.bytes
        self.peak_bytes = max(self.peak_bytes, self.bytes+self.retained)
        return snapshot

    def release(self, snapshot):
        assert snapshot.owner is self and snapshot.active
        self.retained -= snapshot.nbytes
        snapshot.active = False

    def restore(self, snapshot):
        self.banks = dict(snapshot.banks)
        self.position = snapshot.position
        self.release(snapshot)


class NumpyRuntime:
    int32 = np.int32
    array = staticmethod(np.array)
    argmax = staticmethod(np.argmax)
    all = staticmethod(np.all)
    isfinite = staticmethod(np.isfinite)
    stream = staticmethod(lambda stream: nullcontext())


class HybridTarget:
    def __init__(self):
        self.config = {'vocab_size': 29}
        self.options = SimpleNamespace(workspace_bytes=8 << 20)
        self.mx = NumpyRuntime()
        self.stream = None
        self.lock = RLock()
        self.closed = False
        self.stats = {}
        self.reader = SimpleNamespace(stats={}, drain=lambda: None)
        self.checkpoint = SimpleNamespace(stats={'logical_bytes': 0})
        self.sessions = []
        self.fail_next = False
        self.snapshot_budget_bytes = None
        self.memory_failure = None
        self.nonfinite = False

    def new_session(self):
        session = HybridSession(self)
        self.sessions.append(session)
        return session

    def forward(self, input_ids, past_key_values=None, logits_to_keep=0):
        if self.memory_failure is not None:
            raise MemoryError(self.memory_failure)
        cache = self.new_session() if past_key_values is None else past_key_values
        snapshot = cache.snapshot()
        predictions = []
        self.checkpoint.stats['logical_bytes'] += 4096
        try:
            for token in input_ids[0]:
                if cache.position >= cache.limit:
                    raise MemoryError('context limit')
                kv = cache.banks['kv']+(int(token),)
                conv = (cache.banks['conv']+(int(token),))[-3:]
                state = (cache.banks['recurrent'][0]*7+int(token)+cache.position) % 97
                cache.banks = {'kv': kv, 'conv': conv, 'recurrent': (state,)}
                cache.position += 1
                predicted = (state+sum(conv)*3+cache.position) % 29
                logits = np.full(29, -100., np.float32)
                logits[predicted] = 10.
                if self.nonfinite:
                    logits[0] = np.nan
                predictions.append(logits)
                if self.fail_next:
                    self.fail_next = False
                    raise KeyboardInterrupt('injected forward failure')
            cache.release(snapshot)
            values = predictions[-logits_to_keep:] if logits_to_keep else predictions
            return SimpleNamespace(logits=np.asarray([values]), past_key_values=cache)
        except BaseException:
            cache.restore(snapshot)
            raise

    def generate(self, input_ids, max_new_tokens=32, eos_token_id=(), temperature=0.):
        start = perf_counter()
        tokens = list(map(int, input_ids[0]))
        cache = None
        current = input_ids
        first = None
        for _ in range(max_new_tokens):
            result = self.forward(current, past_key_values=cache, logits_to_keep=1)
            cache = result.past_key_values
            token = int(np.argmax(result.logits[0, -1]))
            tokens.append(token)
            if first is None:
                first = perf_counter()
            if token in eos_token_id:
                break
            current = [[token]]
        self.stats.update(tokens_generated=len(tokens)-len(input_ids[0]),
            generation_seconds=perf_counter()-start, ttft_seconds=first-start if first else 0.,
            decode_tokens_per_second=0., context_payload_peak_bytes=cache.peak_bytes if cache else 0)
        return np.asarray([tokens], dtype=np.int32)


def reference_tokens(prefix, count):
    # Independently recompute each prediction from the complete prefix, rather
    # than reading the speculative verifier's banks or proposed tokens.
    values = list(prefix)
    for _ in range(count):
        state = 0
        for position, token in enumerate(values):
            state = (7*state+token+position) % 97
        values.append((state+3*sum(values[-3:])+len(values)) % 29)
    return values


def exact_draft(corrupt=None):
    def propose(prefix, count):
        answer = reference_tokens(prefix, count)[len(prefix):]
        if corrupt is not None and corrupt < len(answer):
            answer[corrupt] = (answer[corrupt]+1) % 29
        return answer
    return propose


@pytest.mark.parametrize('depth', [2, 4, 8, 16])
@pytest.mark.parametrize('corrupt', [None, 0, 1, 3, 7, 15])
def test_every_rejection_depth_restores_all_hybrid_banks(depth, corrupt):
    target = HybridTarget()
    prefix = [1, 8, 3]
    result = generate_gguf_speculative(target, [prefix], exact_draft(corrupt),
        max_new_tokens=40, max_depth=depth, adaptive=False, eos_token_id=[])
    expected = reference_tokens(prefix, 40)
    np.testing.assert_array_equal(result, [expected])
    cache = target.sessions[-1]
    assert cache.position == len(expected)-1
    assert cache.banks['kv'] == tuple(expected[:-1])
    assert cache.banks['conv'] == tuple(expected[-4:-1])
    baseline = HybridTarget().forward([expected[:-1]]).past_key_values
    assert cache.banks == baseline.banks
    assert cache.retained == 0
    metrics = target.stats['speculative']
    assert metrics['actual_target_forward_calls'] >= metrics['target_forward_calls']
    assert metrics['actual_target_forward_positions'] >= metrics['target_forward_positions']
    assert metrics['retained_snapshot_bytes'] == 0
    if corrupt is not None and corrupt < depth:
        assert metrics['recurrent_snapshot_restores'] > 0
        assert metrics['recurrent_replay_calls'] > 0
        assert metrics['logical_bytes'] == metrics['actual_target_forward_calls']*4096
    else:
        assert metrics['recurrent_replay_calls'] == 0


@pytest.mark.parametrize('count', [0, 1, 2])
def test_generation_limits_and_no_unused_snapshot(count):
    target = HybridTarget()
    answer = generate_gguf_speculative(target, [[2, 3]], max_new_tokens=count)
    np.testing.assert_array_equal(answer, [reference_tokens([2, 3], count)])
    assert all(cache.retained == 0 for cache in target.sessions)


@pytest.mark.parametrize('eos_index', [0, 1, 2, 3, 5, 8])
def test_eos_never_commits_the_final_token_to_recurrent_banks(eos_index):
    prefix = [1, 8, 3]
    reference = reference_tokens(prefix, 12)
    eos = reference[len(prefix)+eos_index]
    first = reference.index(eos, len(prefix))
    target = HybridTarget()
    answer = generate_gguf_speculative(target, [prefix], exact_draft(),
        max_new_tokens=12, max_depth=8, adaptive=False, eos_token_id=[eos])
    np.testing.assert_array_equal(answer, [reference[:first+1]])
    assert target.sessions[-1].banks['kv'] == tuple(reference[:first])
    assert target.sessions[-1].retained == 0


def test_forward_exception_restores_all_banks_and_releases_both_snapshots():
    target = HybridTarget()
    verifier = SnapshotVerifier(target)
    verifier.forward([[1, 2]])
    verifier.truncate(2)
    original = dict(verifier.session.banks)
    target.fail_next = True
    with pytest.raises(KeyboardInterrupt, match='injected'):
        verifier.forward([[8, 9]], past_key_values=verifier)
    assert verifier.position == 2
    assert verifier.session.banks == original
    assert verifier.session.retained == 0
    verifier.truncate(2)


def test_replay_failure_returns_to_entry_snapshot():
    target = HybridTarget()
    verifier = SnapshotVerifier(target)
    verifier.forward([[1, 2]])
    verifier.truncate(2)
    original = dict(verifier.session.banks)
    verifier.forward([[3, 4, 5]], past_key_values=verifier)
    target.fail_next = True
    with pytest.raises(KeyboardInterrupt):
        verifier.truncate(4)
    assert verifier.position == 2
    assert verifier.session.banks == original
    assert verifier.session.retained == 0


def test_prompt_lookup_uses_only_existing_prefix_and_known_zero_transport():
    draft = PromptSpanDraft(min_match=2)
    assert draft([1, 2, 3, 4, 1, 2], 4) == [3, 4, 1, 2]
    target = HybridTarget()
    answer = generate_gguf_speculative(target, [[1, 2, 3, 4, 1, 2]], draft,
                                      max_new_tokens=24)
    np.testing.assert_array_equal(answer, [reference_tokens([1, 2, 3, 4, 1, 2], 24)])
    measured = target.stats['speculative']
    assert measured['draft_logical_bytes'] == measured['draft_encoded_file_bytes'] == 0
    assert measured['total_logical_bytes'] == measured['logical_bytes']


def test_sampling_and_self_draft_are_rejected_before_inference():
    target = HybridTarget()
    with pytest.raises(ValueError, match='greedy'):
        generate_gguf_speculative(target, [[1]], temperature=.5)
    with pytest.raises(ValueError, match='distinct'):
        generate_gguf_speculative(target, [[1]], target)
    assert target.checkpoint.stats['logical_bytes'] == 0


def test_snapshot_ownership_and_invalid_truncation_are_rejected():
    verifier = SnapshotVerifier(HybridTarget())
    verifier.forward([[1, 2]])
    with pytest.raises(ValueError, match='another'):
        verifier.forward([[3]], past_key_values=object())
    for bad in (-1, 3, True):
        with pytest.raises(ValueError):
            verifier.truncate(bad)
    verifier.release()


def test_snapshot_budget_fallback_charges_failed_work_and_ordinary_refill(monkeypatch):
    import airllm.nextgen.gguf_speculative as module
    target = HybridTarget()
    target.snapshot_budget_bytes = 70
    ticks = iter(range(100))
    monkeypatch.setattr(module, 'perf_counter', lambda: next(ticks))
    answer = generate_gguf_speculative(target, [[1, 8, 3]], exact_draft(),
        max_new_tokens=3, adaptive=False)
    np.testing.assert_array_equal(answer, [reference_tokens([1, 8, 3], 3)])
    measured = target.stats['speculative']
    assert measured['status'] == 'ordinary_after_snapshot_budget_refusal'
    assert measured['abandoned_speculative_seconds'] == 2
    assert measured['generation_seconds'] == 3
    assert measured['ttft_seconds'] == 2+measured['ordinary_runtime']['ttft_seconds']
    assert measured['logical_bytes'] == target.checkpoint.stats['logical_bytes'] == 4*4096
    assert measured['abandoned_speculative_transport']['logical_bytes'] == 4096
    assert measured['logical_bytes_per_accepted_token'] == 4*4096/3
    assert all(cache.retained == 0 for cache in target.sessions)


def test_unrelated_memory_error_propagates_with_failure_evidence():
    target = HybridTarget()
    target.memory_failure = 'System reserve reached'
    with pytest.raises(MemoryError, match='System reserve'):
        generate_gguf_speculative(target, [[1, 2]], max_new_tokens=3)
    measured = target.stats['speculative']
    assert measured['status'] == 'failed'
    assert measured['failure_message'] == 'System reserve reached'
    assert 'fallback_attempted' not in measured
    assert measured['retained_snapshot_bytes'] == 0


def test_nonfinite_target_logits_are_rejected_with_failure_evidence():
    target = HybridTarget()
    target.nonfinite = True
    with pytest.raises(ValueError, match='non-finite'):
        generate_gguf_speculative(target, [[1, 2]], max_new_tokens=3)
    assert target.stats['speculative']['status'] == 'failed'
    assert target.sessions[-1].retained == 0


def test_fallback_drops_old_recurrent_banks_and_traceback_before_new_cache():
    class LifetimeTarget(HybridTarget):
        def __init__(self):
            super().__init__()
            self.session_refs = []
            self.drained = False
            self.reader.drain = self.drain
        def drain(self):
            self.drained = True
        def new_session(self):
            session = super().new_session()
            self.session_refs.append(weakref.ref(session))
            self.sessions.clear()  # Instrumentation must not retain a cache.
            return session
        def generate(self, *args, **kwargs):
            assert self.drained
            assert len(self.session_refs) == 1
            # No gc.collect: the handled traceback and verifier must release
            # this bank before allocating the ordinary fallback's session.
            assert self.session_refs[0]() is None
            return super().generate(*args, **kwargs)
    target = LifetimeTarget()
    target.snapshot_budget_bytes = 70
    answer = generate_gguf_speculative(target, [[1, 8, 3]], exact_draft(), max_new_tokens=3)
    np.testing.assert_array_equal(answer, [reference_tokens([1, 8, 3], 3)])
    assert target.stats['speculative']['abandoned_speculative_cache_position'] == 3
    assert target.stats['speculative']['abandoned_speculative_peak_bytes'] > 0
    assert 'forward' not in vars(target)


def test_ordinary_fallback_has_finite_guard_and_restores_existing_forward_binding():
    target = HybridTarget()
    target.snapshot_budget_bytes = 70
    original = target.forward
    def hooked(*args, **kwargs):
        return original(*args, **kwargs)
    target.forward = hooked
    ordinary = target.generate
    def nonfinite_ordinary(*args, **kwargs):
        target.nonfinite = True
        return ordinary(*args, **kwargs)
    target.generate = nonfinite_ordinary
    with pytest.raises(ValueError, match='non-finite'):
        generate_gguf_speculative(target, [[1, 8, 3]], exact_draft(), max_new_tokens=3)
    assert target.forward is hooked
    assert target.stats['speculative']['status'] == 'failed'
    assert target.stats['speculative']['fallback_attempted'] is True
