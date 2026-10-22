"""The CLI's second local model must not overwrite the pair's allocator cap."""
from types import SimpleNamespace
from dataclasses import replace
import pytest
from airllm.nextgen.cli import combined_allocator_budget,apply_combined_allocator
from airllm.nextgen.config import NextConfig


def test_pair_budget_sums_payloads_and_shares_one_system_reserve():
    target=NextConfig(cache_bytes=100,io_bytes=20,kv_bytes=30,workspace_bytes=40,reserve_bytes=50)
    draft=replace(target,cache_bytes=10,io_bytes=2,kv_bytes=3,workspace_bytes=4)
    budget=combined_allocator_budget((target,draft),available_bytes=259)
    assert budget['memory_limit_bytes']==209
    assert budget['reserve_bytes']==50
    assert budget['allocator_cache_limit_bytes']==11
    assert budget['models']==2
    with pytest.raises(MemoryError,match='Combined target and draft'):
        combined_allocator_budget((target,draft),available_bytes=258)


def test_pair_reapplies_the_aggregate_after_a_smaller_draft_constructor():
    calls=[]
    mx=SimpleNamespace(metal=SimpleNamespace(is_available=lambda:True),
                       set_cache_limit=lambda n:calls.append(('cache',n)),
                       set_memory_limit=lambda n:calls.append(('memory',n)))
    target=SimpleNamespace(mx=mx);draft=SimpleNamespace(mx=mx)
    mx.set_memory_limit(1000);mx.set_memory_limit(100)
    apply_combined_allocator(target,draft,{'allocator_cache_limit_bytes':22,'memory_limit_bytes':1100})
    assert calls[-2:]==[('cache',22),('memory',1100)]


def test_pair_rejects_different_allocators_and_cpu_only_host_avoids_metal_settings():
    calls=[]
    mx=SimpleNamespace(metal=SimpleNamespace(is_available=lambda:False),
                       set_cache_limit=lambda n:calls.append(n),set_memory_limit=lambda n:calls.append(n))
    target=SimpleNamespace(mx=mx)
    apply_combined_allocator(target,SimpleNamespace(mx=mx),{})
    assert not calls
    with pytest.raises(ValueError,match='same MLX allocator'):
        apply_combined_allocator(target,SimpleNamespace(mx=object()),{})
