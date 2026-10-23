"""Measured policy and exact retrieval controls, without invented SSD timings."""
from types import SimpleNamespace
import random
import numpy as np
import pytest
from airllm.nextgen.speculative import AdaptiveSpeculation,greedy_speculative_generate
from airllm.nextgen.prompt_draft import PromptSpanDraft
from test_speculative import ToyTarget,exact_draft,expected


@pytest.mark.parametrize('options', [
    {'enabled':1},{'warmup_rounds':-1},{'history_window':0},
    {'calibration_interval':True},{'slow_rounds_to_disable':0},
    {'slowdown_tolerance':float('nan')},{'slowdown_tolerance':.9},
])
def test_policy_parameters_are_validated(options):
    with pytest.raises(ValueError):AdaptiveSpeculation(**options)


def test_rolling_baseline_rejects_the_cold_outlier_and_tracks_drift():
    policy=AdaptiveSpeculation(history_window=3)
    policy.record_baseline(1000.,100,50)
    policy.record_baseline(1.,100,50)
    assert policy.baseline_seconds==1.
    policy.record_baseline(1.1,100,50)
    assert policy.baseline_seconds==1.1
    for _ in range(3):policy.record_baseline(2.,200,100)
    assert policy.baseline_seconds==2.


def test_transport_savings_do_not_override_a_slower_depth():
    policy=AdaptiveSpeculation(max_depth=4,warmup_rounds=0)
    policy.observe(depth=2,seconds=.2,emitted=2,proposed=2,accepted=1,
                   baseline_seconds=1.,logical_bytes=1000,encoded_bytes=500)
    policy.observe(depth=3,seconds=.45,emitted=3,proposed=3,accepted=2,
                   baseline_seconds=1.,logical_bytes=3,encoded_bytes=3)
    assert policy.depth==2
    assert policy.cost_summary()['3']['encoded_file_bytes_per_token']==1.


def test_bytes_break_a_measured_time_tie():
    policy=AdaptiveSpeculation(max_depth=4,warmup_rounds=0)
    policy.observe(depth=2,seconds=.2,emitted=2,proposed=2,accepted=1,
                   baseline_seconds=1.,logical_bytes=1000,encoded_bytes=500)
    policy.observe(depth=3,seconds=.306,emitted=3,proposed=3,accepted=2,
                   baseline_seconds=1.,logical_bytes=300,encoded_bytes=150)
    assert policy.depth==3


def test_missing_encoded_count_uses_one_common_logical_domain():
    policy=AdaptiveSpeculation(max_depth=4,warmup_rounds=0)
    policy.observe(depth=2,seconds=.2,emitted=2,proposed=2,accepted=1,
                   baseline_seconds=1.,logical_bytes=1000,encoded_bytes=1)
    policy.observe(depth=3,seconds=.3,emitted=3,proposed=3,accepted=2,
                   baseline_seconds=1.,logical_bytes=300,encoded_bytes=None)
    assert policy.depth==3


def test_cost_history_is_bounded_and_weighted_by_actual_emission():
    policy=AdaptiveSpeculation(max_depth=4,enabled=False,history_window=2)
    for seconds,emitted in [(100.,1),(.2,2),(.6,3)]:
        policy.observe(depth=2,seconds=seconds,emitted=emitted,proposed=2,accepted=1,
                       baseline_seconds=1.,logical_bytes=100,encoded_bytes=50)
    observed=policy.cost_summary()['2']
    assert observed['samples']==2
    assert observed['emitted']==5
    assert observed['seconds_per_token']==pytest.approx(.8/5)


def test_byte_savings_cannot_keep_a_slower_draft_enabled():
    policy=AdaptiveSpeculation(warmup_rounds=0,slow_rounds_to_disable=2)
    for _ in range(2):
        policy.observe(depth=2,seconds=10.,emitted=3,proposed=2,accepted=2,
                       baseline_seconds=1.,logical_bytes=1,encoded_bytes=1)
    assert not policy.active
    assert 'measured' in policy.disabled_reason


def test_total_decode_guard_counts_cost_outside_verification_samples():
    policy=AdaptiveSpeculation(warmup_rounds=0,slow_rounds_to_disable=2)
    for _ in range(2):
        policy.observe(depth=2,seconds=.1,emitted=3,proposed=2,accepted=2,
                       baseline_seconds=1.,logical_bytes=1,encoded_bytes=1)
        policy.observe_total_decode(seconds=10.,emitted=3,baseline_seconds=1.)
    assert not policy.active
    assert policy.disabled_reason=='measured_total_decode_cost_exceeds_decode_baseline'


def test_periodic_calibration_is_real_and_keeps_the_committed_prefix():
    tick=[0.]
    class TimedTarget(ToyTarget):
        def forward(self,ids,*args,**kwargs):
            tick[0]+=.1+.01*len(ids[0])
            return super().forward(ids,*args,**kwargs)
    target=TimedTarget()
    def clock():return tick[0]
    result=greedy_speculative_generate(target,exact_draft(target),[[1,2]],50,
        max_depth=4,clock=clock,bytes_read=lambda:target.bytes,
        encoded_bytes_read=lambda:target.bytes//4,
        draft_bytes_read=lambda:0,draft_encoded_bytes_read=lambda:0)
    assert result.tokens==expected(target,[1,2],50)
    assert result.cache.values[:result.cache.position]==result.tokens[:-1]
    assert result.metrics['calibration_forward_calls']>1
    assert len(result.metrics['baseline_measurements'])==result.metrics['ordinary_decode_forward_calls']
    assert sum(m['calibration'] for m in result.metrics['baseline_measurements'])==result.metrics['calibration_forward_calls']
    assert result.metrics['total_logical_bytes']==target.bytes
    assert result.metrics['total_encoded_file_bytes']==target.bytes//4
    assert result.metrics['draft_logical_bytes']==0


def test_draft_io_is_included_in_costs_and_remains_distinct():
    target=ToyTarget();draft=[0]
    exact=exact_draft(target)
    def propose(prefix,count):
        draft[0]+=700
        return exact(prefix,count)
    result=greedy_speculative_generate(target,propose,[[1,2]],20,adaptive=False,
        bytes_read=lambda:target.bytes,encoded_bytes_read=lambda:target.bytes//4,
        draft_bytes_read=lambda:draft[0],draft_encoded_bytes_read=lambda:draft[0]//2)
    m=result.metrics
    assert m['total_logical_bytes']==target.bytes+draft[0]
    assert m['total_encoded_file_bytes']==target.bytes//4+draft[0]//2
    assert m['cost_observation_scope']=='target_and_draft'
    assert all(r['total_logical_bytes']==r['logical_bytes']+700 for r in m['round_measurements'])
    assert all(r['total_encoded_file_bytes']==r['encoded_file_bytes']+350 for r in m['round_measurements'])


def test_unknown_draft_transport_is_not_zero():
    target=ToyTarget()
    result=greedy_speculative_generate(target,exact_draft(target),[[1]],10)
    assert result.metrics['draft_logical_bytes'] is None
    assert result.metrics['total_logical_bytes'] is None
    assert 'unknown' in result.metrics['cost_observation_scope']


def test_empty_prompt_lookup_stops_attempting_after_bounded_failures():
    target=ToyTarget();attempts=[]
    def propose(prefix,count):
        attempts.append(count)
        return []
    result=greedy_speculative_generate(target,propose,[[1]],30)
    assert result.tokens==expected(target,[1],30)
    assert len(attempts)<=4
    assert result.metrics['disabled_reason'] in ('empty_draft_proposals','measured_draft_plus_verify_cost_exceeds_decode_baseline',
                                                'measured_total_decode_cost_exceeds_decode_baseline')


def _old_lookup(prefix,count,window,maximum,minimum):
    if not count:return []
    tokens=prefix[-window:]
    for n in range(min(maximum,len(tokens)-1),minimum-1,-1):
        suffix=tokens[-n:]
        for i in range(len(tokens)-n-1,-1,-1):
            if tokens[i:i+n]==suffix:return tokens[i+n:min(i+n+count,len(tokens))]
    return []


def test_linear_lookup_exactly_preserves_longest_newest_overlap_semantics():
    rng=random.Random(7)
    examples=[[1]*128,list(range(128)),[1,2,3]*64,[],[1]]
    examples += [[rng.randrange(6) for _ in range(rng.randrange(120))] for _ in range(300)]
    for prefix in examples:
        for window,maximum,minimum,count in [(128,64,3,8),(29,12,2,4),(16,4,1,0)]:
            lookup=PromptSpanDraft(max_lookback=window,max_match=maximum,min_match=minimum)
            assert lookup(prefix,count)==_old_lookup(prefix,count,window,maximum,minimum)
            assert lookup.logical_bytes_read()==lookup.encoded_bytes_read()==0
