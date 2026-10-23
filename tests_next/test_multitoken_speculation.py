from types import SimpleNamespace
import numpy as np
import pytest
from airllm.nextgen.speculative import AdaptiveSpeculation, greedy_speculative_generate
from airllm.nextgen.text_draft import ReencodedModelDraft
from test_speculative import ToyTarget, exact_draft, expected


@pytest.mark.parametrize('depth',[2,4,8,16,32])
@pytest.mark.parametrize('corrupt',[None,0,1])
def test_long_blocks_preserve_exact_prefix_and_rollback(depth,corrupt):
    target=ToyTarget()
    result=greedy_speculative_generate(target,exact_draft(target,corrupt),[[1,2]],70,
        max_depth=depth,adaptive=False,bytes_read=lambda:target.bytes,
        forward_metrics=lambda:{'passes':len(target.calls)})
    assert result.tokens==expected(target,[1,2],70)
    assert result.cache.values[:result.cache.position]==result.tokens[:-1]
    assert all(r['expert_batch_delta']=={'passes':1} for r in result.metrics['round_measurements'])


def test_candidate_grid_explores_measured_neighbor_without_intermediate_depths():
    policy=AdaptiveSpeculation(max_depth=32,candidate_depths=(2,4,8,16,32))
    for depth in (2,4,8,16):
        assert policy.depth==depth
        for _ in range(2):
            policy.observe(seconds=.01,emitted=depth+1,proposed=depth,accepted=depth,
                baseline_seconds=1.,depth=depth)
    assert policy.depth==32


def test_verification_limit_bounds_actual_target_input():
    target=ToyTarget()
    result=greedy_speculative_generate(target,exact_draft(target),[[1,2]],30,
        max_depth=32,adaptive=False,max_verify_positions=5)
    assert result.tokens==expected(target,[1,2],30)
    assert max(len(ids[0]) for ids,_ in target.calls)<=5


def test_grid_rejection_and_tail_observation_stay_on_candidate_grid():
    policy=AdaptiveSpeculation(max_depth=32,candidate_depths=(2,4,8,16,32))
    for depth in (2,1):
        policy.observe(seconds=.01,emitted=1,proposed=depth,accepted=0,
            baseline_seconds=1.,depth=depth)
        assert policy.depth in policy.candidate_depths


def test_prefill_keeps_only_final_logits_but_verification_keeps_every_position():
    class LimitedLogitsTarget(ToyTarget):
        def forward(self,ids,past_key_values=None,logits_to_keep=0):
            if past_key_values is None:
                assert logits_to_keep==1
            elif len(ids[0])>1:
                assert logits_to_keep==0
            return super().forward(ids,past_key_values,logits_to_keep)
    target=LimitedLogitsTarget()
    prefix=list(range(700))
    result=greedy_speculative_generate(target,exact_draft(target),[prefix],12,adaptive=False)
    assert result.tokens==expected(target,prefix,12)


class CharacterTokenizer:
    def __init__(self,offset):self.offset=offset
    def encode(self,text,**kwargs):return [ord(c)+self.offset for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i-self.offset) for i in ids)


def test_reencoded_draft_maps_tokens_instead_of_reusing_incompatible_ids():
    target=CharacterTokenizer(0)
    tokenizer=CharacterTokenizer(10)
    class Draft:
        checkpoint=object()
        options=SimpleNamespace(max_context=128)
        config={'max_position_embeddings':128}
        def generate(self,ids,**kwargs):return np.array([ids[0].tolist()+tokenizer.encode('xyz')[:kwargs['max_new_tokens']]])
    model=Draft();model.tokenizer=tokenizer
    draft=ReencodedModelDraft(model,target)
    assert draft(target.encode('abc'),2)==target.encode('xy')
    assert draft.stats['draft_tokens']==2
    assert draft.checkpoint is model.checkpoint
