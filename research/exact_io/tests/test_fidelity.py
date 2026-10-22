from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'air_llm'))
from airllm.nextgen.prompt_draft import PromptSpanDraft
from airllm.nextgen.speculative import greedy_speculative_generate


class Cache:
    def __init__(self):self.tokens=[];self.limit=8192
    @property
    def position(self):return len(self.tokens)
    def truncate(self,n):del self.tokens[n:]


class DeterministicTarget:
    """Stateful test oracle only. Not an inference throughput benchmark."""
    def __init__(self):self.calls=0
    def forward(self,ids,past_key_values=None,logits_to_keep=0):
        self.calls+=1
        c=past_key_values if past_key_values is not None else Cache()
        logits=[]
        for t in ids[0]:
            c.tokens.append(int(t));p=(int(t)+1)%7
            row=np.full(7,-1,dtype=np.float32);row[p]=1;logits.append(row)
        return SimpleNamespace(logits=np.array([logits]),past_key_values=c)


@pytest.mark.parametrize('length',[0,1,2,3,8,33,65])
@pytest.mark.parametrize('kind',['lookup','wrong','empty'])
def test_no_unverified_draft_and_exact_lagging_state(length,kind):
    prefix=list(range(7))*3
    propose=PromptSpanDraft() if kind=='lookup' else (lambda p,n:[(p[-1]+4)%7]*n) if kind=='wrong' else lambda p,n:[]
    model=DeterministicTarget()
    got=greedy_speculative_generate(model,propose,np.array([prefix]),length,
        max_depth=16,adaptive=False)
    expected=prefix+[(prefix[-1]+1+i)%7 for i in range(length)]
    assert got.tokens==expected
    if length:assert got.cache.tokens==expected[:-1]
    else:assert got.cache is None and model.calls==0
    if kind=='wrong':assert got.metrics['accepted_draft_tokens']==0


@pytest.mark.parametrize('eos',[0,2,6])
def test_lookup_eos_rollback(eos):
    prefix=list(range(7))*3
    got=greedy_speculative_generate(DeterministicTarget(),PromptSpanDraft(),np.array([prefix]),50,
        eos_token_ids=[eos],max_depth=16,adaptive=False)
    assert got.tokens==prefix+list(range(eos+1))
    assert got.cache.tokens==got.tokens[:-1]


def test_lookup_is_bounded_and_uses_only_available_tokens():
    lookup=PromptSpanDraft(max_lookback=16,max_match=6,min_match=2)
    assert lookup([1,2,3,4,1,2],5)==[3,4,1,2]
    assert lookup([7,8,9],0)==[]
    assert lookup([7,8,9],4)==[]
    with pytest.raises(ValueError):PromptSpanDraft(max_lookback=0)


@pytest.mark.parametrize('p,q',[
    ([F(3,5),F(2,5)],[F(9,10),F(1,10)]),
    ([F(1),F(0)],[F(0),F(1)]),
    ([F(1,3)]*3,[F(1,2),F(1,4),F(1,4)]),
    ([F(1,3)]*3,[F(1,3)]*3)])
def test_exact_sampling_identity_with_rational_probabilities(p,q):
    # Analytic finite identity; no empirical histogram is substituted for proof.
    accepted=[min(a,b) for a,b in zip(p,q)]
    rejection=1-sum(accepted)
    positive=[max(a-b,F(0)) for a,b in zip(p,q)]
    residual=[r/sum(positive) for r in positive] if rejection else [F(0)]*len(p)
    assert [a+rejection*r for a,r in zip(accepted,residual)]==p
