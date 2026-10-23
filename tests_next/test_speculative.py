"""Control-flow proofs and actual miniature GLM speculative regression tests."""
from types import SimpleNamespace
import numpy as np
import pytest
from airllm.nextgen.speculative import (
    AdaptiveSpeculation, LocalModelDraft, greedy_speculative_generate)


class ToyCache:
    def __init__(self):
        self.position = 0
        self.values = []
        self.peak_bytes = 0

    def truncate(self, position):
        if not 0 <= position <= self.position:
            raise ValueError('Invalid truncate')
        self.position = position


class ToyTarget:
    """A deterministic causal oracle, for scheduler proofs only, not performance."""
    def __init__(self, vocabulary=17):
        self.vocabulary = vocabulary
        self.bytes = 0
        self.calls = []

    def next(self, prefix):
        return (sum((i+1)*token for i, token in enumerate(prefix))+len(prefix)) % self.vocabulary

    def forward(self, ids, past_key_values=None, logits_to_keep=0):
        cache = past_key_values or ToyCache()
        logits = []
        for token in ids[0]:
            if len(cache.values) <= cache.position:
                cache.values.append(int(token))
            else:
                cache.values[cache.position] = int(token)
            cache.position += 1
            row = np.full(self.vocabulary, -100.)
            row[self.next(cache.values[:cache.position])] = 100.
            logits.append(row)
        self.bytes += 1024
        self.calls.append((ids.tolist(), cache.position))
        return SimpleNamespace(logits=np.asarray([logits]), past_key_values=cache)


def exact_draft(target, corrupt_at=None):
    def propose(prefix, count):
        values = []
        for i in range(count):
            token = target.next(prefix+values)
            if i == corrupt_at:
                token = (token+1) % target.vocabulary
            values.append(token)
        return values
    return propose


def expected(target, prefix, count, eos=()):
    out = list(prefix)
    for _ in range(count):
        out.append(target.next(out))
        if out[-1] in eos:
            break
    return out


@pytest.mark.parametrize('corrupt_at', [None, 0, 1, 2])
@pytest.mark.parametrize('count', [0, 1, 2, 3, 9, 18])
def test_every_emitted_token_is_target_greedy(corrupt_at, count):
    target = ToyTarget()
    result = greedy_speculative_generate(target, exact_draft(target, corrupt_at),
        [[1, 8, 4]], count, adaptive=False, max_depth=3, bytes_read=lambda:target.bytes)
    assert result.tokens == expected(target, [1, 8, 4], count)
    assert result.metrics['tokens_generated'] == count
    if count:
        assert result.cache.position == len(result.tokens)-1
        assert result.cache.values[:result.cache.position] == result.tokens[:-1]
        assert result.metrics['logical_bytes_per_accepted_token'] == target.bytes/count
    else:
        assert not target.calls


@pytest.mark.parametrize('eos', [0, 3, 5, 16])
def test_eos_on_proposal_or_correction_stops_exactly(eos):
    for corrupt_at in (None, 0, 1):
        target = ToyTarget()
        result = greedy_speculative_generate(target, exact_draft(target, corrupt_at),
            [[1, 8, 4]], 32, eos_token_ids=[eos], adaptive=False, max_depth=4)
        assert result.tokens == expected(target, [1, 8, 4], 32, [eos])


def test_blocks_amortize_target_passes_without_skipping_verification():
    target = ToyTarget()
    result = greedy_speculative_generate(target, exact_draft(target), [[1, 2]], 20,
        adaptive=False, max_depth=4, bytes_read=lambda:target.bytes)
    assert result.tokens == expected(target, [1, 2], 20)
    assert 2 < len(target.calls) < 20
    assert result.metrics['accepted_draft_tokens'] > 0
    assert all(r['logical_bytes_per_accepted_token'] == r['logical_bytes']/r['emitted']
               for r in result.metrics['round_measurements'])


def test_empty_draft_still_generates_target_tokens():
    target = ToyTarget()
    result = greedy_speculative_generate(target, lambda p,n: [], [[2, 4]], 12, adaptive=False)
    assert result.tokens == expected(target, [2, 4], 12)
    assert result.metrics['acceptance_rate'] is None


def test_encoded_transport_bytes_remain_distinct_from_logical_weight_bytes():
    target=ToyTarget()
    result=greedy_speculative_generate(target,exact_draft(target),[[2,4]],12,
        adaptive=False,bytes_read=lambda:target.bytes,
        encoded_bytes_read=lambda:target.bytes//4)
    metrics=result.metrics
    assert metrics['logical_bytes']==target.bytes
    assert metrics['encoded_file_bytes']==target.bytes//4
    assert metrics['encoded_file_bytes_per_accepted_token']==metrics['logical_bytes_per_accepted_token']/4
    assert metrics['decode_encoded_file_bytes_per_accepted_token']==metrics['decode_logical_bytes_per_accepted_token']/4


def test_accepted_eos_preserves_the_lagging_cache_invariant():
    target=ToyTarget()
    prefix=[1,8,4];sequence=expected(target,prefix,8);eos=sequence[len(prefix)+3]
    result=greedy_speculative_generate(target,exact_draft(target),[prefix],8,
        adaptive=False,max_depth=4,eos_token_ids=(eos,))
    assert result.tokens==expected(target,prefix,8,[eos])
    assert result.cache.position==len(result.tokens)-1
    assert result.cache.values[:result.cache.position]==result.tokens[:-1]


@pytest.mark.parametrize('proposal', [[True], [-1], [1.2], [2**40]])
def test_bad_draft_ids_rejected(proposal):
    with pytest.raises(ValueError, match='invalid token'):
        greedy_speculative_generate(ToyTarget(),lambda p,n:proposal,[[1]],6)


def test_adaptive_guard_uses_measured_cost_and_disables():
    policy = AdaptiveSpeculation(max_depth=8)
    for _ in range(5):
        policy.observe(seconds=5.,emitted=1,proposed=2,accepted=1,baseline_seconds=1.)
    assert not policy.active
    assert 'measured' in policy.disabled_reason


def test_zero_acceptance_disables_despite_noisy_optimistic_decode_baseline():
    policy=AdaptiveSpeculation(max_depth=8)
    for _ in range(policy.warmup_rounds+policy.slow_rounds_to_disable):
        policy.observe(seconds=.1,emitted=1,proposed=2,accepted=0,baseline_seconds=1000.)
    assert not policy.active
    assert policy.depth==1
    assert policy.disabled_reason=='no_accepted_draft_tokens'


def test_draft_wrapper_checks_prefix():
    class Broken:
        def generate(self,*args,**kwargs):return np.asarray([[8, 9]])
    with pytest.raises(ValueError,match='preserve'):
        LocalModelDraft(Broken())([1],1)


def test_draft_token_outside_target_vocabulary_is_rejected():
    with pytest.raises(ValueError,match='invalid token'):
        greedy_speculative_generate(ToyTarget(),lambda p,n:[19],[[1]],6,vocab_size=17)


def test_faulty_draft_iterator_cannot_allocate_an_unbounded_candidate_list():
    from itertools import repeat
    with pytest.raises(ValueError,match='more candidates'):
        greedy_speculative_generate(ToyTarget(),lambda p,n:repeat(1),[[1]],6)


def test_speculative_block_is_capped_before_the_context_limit():
    class LimitedTarget(ToyTarget):
        def forward(self,ids,past_key_values=None,logits_to_keep=0):
            cache=past_key_values or ToyCache()
            cache.limit=7
            assert cache.position+len(ids[0])<=cache.limit
            return super().forward(ids,cache,logits_to_keep)
    target=LimitedTarget()
    result=greedy_speculative_generate(target,exact_draft(target),[[1,2,3]],5,
        adaptive=False,max_depth=8)
    assert result.tokens==expected(target,[1,2,3],5)


def test_unsigned_and_signed_out_of_range_prompt_rejected():
    for dtype in (np.int64,np.uint64):
        with pytest.raises(ValueError,match='int32'):
            greedy_speculative_generate(ToyTarget(),lambda p,n:[],np.asarray([[2**40]],dtype=dtype),3)


def test_native_glm_speculative_logits_and_rollback(tiny_glm,tmp_path):
    from airllm.nextgen import AirLLMNextModel
    from test_native import options
    root,_ = tiny_glm
    ids = [[1, 8, 9]]
    with AirLLMNextModel(root,next_config=options(tmp_path,prefill_chunk=16)) as model:
        baseline = np.asarray(model.generate(ids,max_new_tokens=10,eos_token_id=[]))

        # A deliberately wrong local draft exercises correction and rollback.
        result = np.asarray(model.generate(ids,max_new_tokens=10,eos_token_id=[],
            draft_model=lambda p,n:[0]*n,speculative_adaptive=False))
        np.testing.assert_array_equal(result,baseline)
        metrics=model.metrics()['runtime']['speculative']
        assert metrics['rollback_tokens']>0
        assert metrics['tokens_generated']==10
        assert metrics['final_cache_position']==12

        # Use a second independently loaded local target clone as an accurate
        # draft. This verifies block target evaluation and bonus-token handling;
        # it is a correctness fixture, not a useful performance configuration.
        with AirLLMNextModel(root,next_config=options(tmp_path/'draft',prefill_chunk=16)) as draft:
            result=np.asarray(model.generate(ids,max_new_tokens=10,eos_token_id=[],
                draft_model=draft,speculative_adaptive=False))
        np.testing.assert_array_equal(result,baseline)
        metrics=model.metrics()['runtime']['speculative']
        assert metrics['accepted_draft_tokens']>0
        assert metrics['target_forward_calls']<10
        model.generate(ids,max_new_tokens=1)
        assert 'speculative' not in model.metrics()['runtime']


def test_native_glm_truncate_matches_recompute(tiny_glm,tmp_path):
    from airllm.nextgen import AirLLMNextModel
    from test_native import options
    root,_=tiny_glm
    with AirLLMNextModel(root,next_config=options(tmp_path,prefill_chunk=16)) as model:
        cache=model([[1,8,9,10]]).past_key_values
        cache.truncate(2)
        actual=model([[11,12]],past_key_values=cache).logits
        expected=model([[1,8,11,12]]).logits[:,2:]
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),atol=4e-6,rtol=2e-4)
        with pytest.raises(ValueError):cache.truncate(5)


def test_native_speculative_sampling_is_explicitly_unsupported(tiny_glm,tmp_path):
    from airllm.nextgen import AirLLMNextModel
    from test_native import options
    root,_=tiny_glm
    with AirLLMNextModel(root,next_config=options(tmp_path)) as model:
        with pytest.raises(NotImplementedError,match='greedy'):
            model.generate([[1]],draft_model=lambda p,n:[1]*n,temperature=.8)



class RetryManualClock:
    """Inject measured service costs; never claims hardware performance."""
    def __init__(self):
        self.seconds = 0.

    def __call__(self):
        return self.seconds


class RetryTimedTarget(ToyTarget):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def forward(self, *args, **kwargs):
        self.clock.seconds += 1.
        return super().forward(*args, **kwargs)


def test_empty_adaptive_draft_retries_only_after_eight_committed_tokens_and_real_candidates():
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    prefix = [1, 2]
    calls = []
    def late_draft(values, count):
        generated = len(values)-len(prefix)
        calls.append(generated)
        clock.seconds += .001
        return [] if generated < 20 else exact_draft(target)(values, count)
    result = greedy_speculative_generate(target, late_draft, [prefix], 36,
        adaptive=True, retry_empty_drafts=True, max_depth=4, clock=clock, bytes_read=lambda:target.bytes,
        draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    assert result.tokens == expected(target, prefix, 36)
    assert result.cache.position == len(result.tokens)-1
    assert result.cache.values[:result.cache.position] == result.tokens[:-1]
    # Empty proposals at generated positions 2..5 disable the lookup. It must
    # then advance eight actual output tokens before each probe, not eight
    # loop iterations, target positions, candidate tokens, or wall-clock ticks.
    assert calls[:6] == [2, 3, 4, 5, 14, 22]
    metrics = result.metrics
    assert metrics['empty_retry_probes'] == 2
    assert metrics['empty_retry_reactivations'] == 1
    assert metrics['accepted_draft_tokens'] > 0
    assert [row['reactivated'] for row in metrics['empty_retry_probe_measurements']] == [False, True]
    assert all(row['candidates'] <= row['count_limit'] <= 4 for row in metrics['empty_retry_probe_measurements'])
    assert metrics['empty_retry_probe_seconds'] == pytest.approx(.002)
    assert metrics['draft_seconds'] == pytest.approx(metrics['draft_calls']*.001)
    assert metrics['generation_seconds'] == pytest.approx(len(target.calls)+metrics['draft_calls']*.001)


@pytest.mark.parametrize('cost,logical,encoded', [
    (.02, 0, 0), (.001, None, 0), (.001, 0, None), (.001, 1, 0), (.001, 0, 1)])
def test_empty_retry_rejects_expensive_or_unobserved_or_nonzero_draft_transport(cost, logical, encoded):
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    counters = [0, 0]
    def empty(values, count):
        clock.seconds += cost
        counters[0] += logical or 0
        counters[1] += encoded or 0
        return []
    result = greedy_speculative_generate(target, empty, [[1, 2]], 32,
        adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=(lambda:counters[0]) if logical is not None else None,
        draft_encoded_bytes_read=(lambda:counters[1]) if encoded is not None else None)
    assert result.tokens == expected(target, [1, 2], 32)
    assert result.metrics['disabled_reason'] == 'empty_draft_proposals'
    assert result.metrics['draft_calls'] == 4
    assert result.metrics['empty_retry_probes'] == 0


def test_a_newly_expensive_probe_is_counted_and_permanently_stops_retrying():
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    calls = []
    def expensive_later(values, count):
        calls.append(len(values)-2)
        clock.seconds += .001 if len(calls) <= 4 else .02
        return [] if len(calls) <= 4 else exact_draft(target)(values, count)
    result = greedy_speculative_generate(target, expensive_later, [[1, 2]], 40,
        adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    metrics = result.metrics
    assert result.tokens == expected(target, [1, 2], 40)
    assert calls == [2, 3, 4, 5, 14]
    assert metrics['empty_retry_probes'] == 1
    assert metrics['empty_retry_candidate_tokens'] > 0
    assert metrics['empty_retry_abandoned_candidate_tokens'] == metrics['empty_retry_candidate_tokens']
    assert metrics['empty_retry_reactivations'] == 0
    assert metrics['empty_retry_stopped_reason'] == 'probe_cost_or_transport_guard'
    assert metrics['empty_retry_probe_seconds'] == pytest.approx(.02)
    assert metrics['generation_seconds'] == pytest.approx(len(target.calls)+.004+.02)


def test_empty_retry_does_not_reopen_a_zero_acceptance_disable():
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    def wrong(values, count):
        clock.seconds += .001
        return exact_draft(target, corrupt_at=0)(values, count)
    result = greedy_speculative_generate(target, wrong, [[1, 2]], 40,
        adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    assert result.tokens == expected(target, [1, 2], 40)
    assert result.metrics['disabled_reason'] == 'no_accepted_draft_tokens'
    assert result.metrics['empty_retry_probes'] == 0


def test_empty_retry_rejects_new_transport_even_when_actual_candidates_are_found():
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    calls, transport = [0], [0]
    def io_later(values, count):
        calls[0] += 1
        clock.seconds += .001
        if calls[0] <= 4:
            return []
        transport[0] += 1
        return exact_draft(target)(values, count)
    result = greedy_speculative_generate(target, io_later, [[1, 2]], 40,
        adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=lambda:transport[0], draft_encoded_bytes_read=lambda:0)
    assert result.tokens == expected(target, [1, 2], 40)
    assert calls[0] == 5
    assert result.metrics['empty_retry_probes'] == 1
    assert result.metrics['empty_retry_reactivations'] == 0
    assert result.metrics['empty_retry_stopped_reason'] == 'probe_cost_or_transport_guard'
    assert result.metrics['empty_retry_probe_measurements'][0]['draft_logical_bytes'] == 1
    assert result.metrics['empty_retry_abandoned_candidate_tokens'] > 0


def test_accepted_eos_after_empty_retry_still_stops_with_the_lagging_cache_invariant():
    class OrderedTarget(RetryTimedTarget):
        def __init__(self, clock):
            super().__init__(clock)
            self.vocabulary = 256

        def next(self, prefix):
            return len(prefix)
    clock = RetryManualClock()
    target = OrderedTarget(clock)
    def late_draft(values, count):
        clock.seconds += .001
        return [] if len(values)-2 < 10 else exact_draft(target)(values, count)
    result = greedy_speculative_generate(target, late_draft, [[1, 2]], 40,
        eos_token_ids=[17], adaptive=True, retry_empty_drafts=True, clock=clock,
        draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    assert result.tokens == expected(target, [1, 2], 40, [17])
    assert result.tokens[-1] == 17
    assert result.metrics['empty_retry_reactivations'] == 1
    assert result.cache.position == len(result.tokens)-1
    assert result.cache.values[:result.cache.position] == result.tokens[:-1]


def test_probe_proposal_iterator_keeps_the_existing_strict_consumption_bound():
    from itertools import repeat
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    calls = [0]
    def faulty_later(values, count):
        calls[0] += 1
        clock.seconds += .001
        return [] if calls[0] <= 4 else repeat(1)
    with pytest.raises(ValueError, match='more candidates'):
        greedy_speculative_generate(target, faulty_later, [[1, 2]], 40,
            adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    assert calls[0] == 5


def test_repeated_empty_probe_metadata_is_capped_without_allocating_target_cache_snapshots():
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    def empty(values, count):
        clock.seconds += .001
        return []
    result = greedy_speculative_generate(target, empty, [[1, 2]], 120,
        adaptive=True, retry_empty_drafts=True, clock=clock, draft_bytes_read=lambda:0, draft_encoded_bytes_read=lambda:0)
    metrics = result.metrics
    assert result.tokens == expected(target, [1, 2], 120)
    assert metrics['empty_retry_probes'] > metrics['empty_retry_history_capacity']
    assert len(metrics['empty_retry_probe_measurements']) == metrics['empty_retry_history_capacity'] == 8
    assert metrics['empty_retry_reactivations'] == 0
    assert all(not row['reactivated'] for row in metrics['empty_retry_probe_measurements'])



@pytest.mark.parametrize('retry_options', [{}, {'retry_empty_drafts': False}])
def test_e1_default_and_explicit_false_keep_empty_draft_disable_permanent(retry_options):
    clock = RetryManualClock()
    target = RetryTimedTarget(clock)
    calls = []
    def late_draft(values, count):
        calls.append(len(values)-2)
        clock.seconds += .001
        return [] if len(values)-2 < 10 else exact_draft(target)(values, count)
    result = greedy_speculative_generate(target, late_draft, [[1, 2]], 40,
        adaptive=True, clock=clock, draft_bytes_read=lambda:0,
        draft_encoded_bytes_read=lambda:0, **retry_options)
    assert result.tokens == expected(target, [1, 2], 40)
    assert calls == [2, 3, 4, 5]
    assert result.metrics['retry_empty_drafts'] is False
    assert result.metrics['disabled_reason'] == 'empty_draft_proposals'
    assert result.metrics['empty_retry_probes'] == result.metrics['empty_retry_reactivations'] == 0
    assert result.metrics['accepted_draft_tokens'] == 0
    assert result.metrics['verifier_rounds'] == result.metrics['rounds'] == 4
    assert result.metrics['empty_retry_interval_tokens'] == 8
    assert result.metrics['empty_retry_cost_fraction'] == .01


@pytest.mark.parametrize('invalid', [0, 1, None, 'true', np.bool_(True)])
def test_retry_empty_drafts_requires_strict_bool_before_any_forward(invalid):
    from airllm.nextgen.runtime import AirLLMNextModel
    target = ToyTarget()
    with pytest.raises(ValueError, match='retry_empty_drafts must be boolean'):
        greedy_speculative_generate(target, lambda values,count:[], [[1, 2]], 8,
            retry_empty_drafts=invalid)
    assert not target.calls
    with pytest.raises(ValueError, match='retry_empty_drafts must be boolean'):
        AirLLMNextModel.generate_speculative(object(), [[1, 2]], lambda values,count:[],
            retry_empty_drafts=invalid)


def test_native_runtime_explicitly_forwards_retry_opt_in(tiny_glm, tmp_path, monkeypatch):
    from airllm.nextgen import AirLLMNextModel
    import airllm.nextgen.speculative as speculative
    from test_native import options
    root, _ = tiny_glm
    original_generate = speculative.greedy_speculative_generate
    received = []
    def capture(*args, **kwargs):
        received.append(kwargs['retry_empty_drafts'])
        return original_generate(*args, **kwargs)
    monkeypatch.setattr(speculative, 'greedy_speculative_generate', capture)
    with AirLLMNextModel(root, next_config=options(tmp_path)) as model:
        model.generate_speculative([[1, 8]], lambda values,count:[], max_new_tokens=2,
            retry_empty_drafts=True)
        assert model.metrics()['runtime']['speculative']['retry_empty_drafts'] is True
    assert received == [True]
