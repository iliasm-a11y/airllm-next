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
