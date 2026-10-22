from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from routing import bypass_opt,cache_misses
from scenarios import expected_union,progress


def brute_opt(requests,capacity):
    states={frozenset():0}
    for key in requests:
        next_states={}
        for state,cost in states.items():
            miss=int(key not in state)
            choices=[state]
            if capacity and key not in state:
                if len(state)<capacity:choices.append(state|{key})
                else:choices.extend((state-{v})|{key} for v in state)
            for s in choices:next_states[s]=min(next_states.get(s,10**9),cost+miss)
        states=next_states
    return min(states.values())


def test_belady_oracle_matches_exhaustive_cache_schedules():
    rng=np.random.default_rng(61)
    for c in [0,1,2,3]:
        for _ in range(20):
            trace=rng.integers(0,5,size=10).tolist()
            assert bypass_opt(trace,c)==brute_opt(trace,c)


def test_uniform_independent_is_not_worst_case_over_correlated_routes():
    # Two uniform marginal routes, always different: expected union 2 > IID 1.75.
    unions=[len({i,(i+1)%4}) for i in range(4)]
    assert np.mean(unions)==2 and expected_union(4,1,2)==1.75


def test_byte_sensitive_function_does_not_require_all_weights_for_zero_input():
    # Every weight affects some input. The specific input zero still needs none.
    W=np.array([[1.,2.],[3.,4.]])
    assert np.array_equal(W@np.zeros(2),np.zeros(2))
    assert np.any(W@np.ones(2) != (W+1)@np.ones(2))


def test_tenfold_exact_amortization_is_not_unconditionally_forbidden():
    n=16;A=10;R=20
    # Hypothetical perfect drafts and identical expert sets: every target token
    # is still fully evaluated, but each required weight is read once per block.
    ota=(A+R)/progress(1,n-1)
    assert (A+R)/ota==16


def test_cache_static_density_depends_on_decoded_byte_cost():
    always_density=1*1/2 # FP8 stored byte, decoded BF16 2-byte residency.
    expert_density=0.75*2/2 # BF16 byte cost, 75% usage.
    assert expert_density>always_density


def test_scan_pollution_is_real_and_not_model_approximation():
    req=[0,1,2,3]*8
    assert cache_misses(req,2,'lru')==len(req)
    assert bypass_opt(req,2)<len(req)
