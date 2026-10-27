"""Planner budgets and scheduling parity; no throughput claims from fixtures."""
from dataclasses import replace

import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.hyperprefill import HyperPrefillPolicy


def dimensions():
    return dict(hidden_size=2048, num_experts_per_tok=8,
                moe_intermediate_size=512, num_attention_heads=16,
                num_key_value_heads=2, head_dim=256,
                linear_num_key_heads=16, linear_num_value_heads=32,
                linear_key_head_dim=128, linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
                layer_types=['linear_attention'] * 30 + ['full_attention'] * 10,
                max_position_embeddings=4096)


def options():
    return NextConfig(cache_bytes=64 << 20, io_bytes=64 << 20,
                      kv_bytes=256 << 20, workspace_bytes=128 << 20,
                      reserve_bytes=3 << 30, max_tensor_bytes=16 << 20,
                      max_context=1280, dtype='float32')


def choose(policy=None, config=None, opt=None, **kwargs):
    return (policy or HyperPrefillPolicy()).select(
        config or dimensions(), opt or options(), cache_position=0,
        remaining_tokens=364, available_bytes=6 << 30, **kwargs)


def test_prefill_reuses_larger_bounded_block_and_caps_are_unchanged():
    opt=options()
    before=opt.as_dict()
    decision=choose(opt=opt)
    assert decision.chunk_tokens==128
    assert decision.estimated_workspace_bytes<=opt.workspace_bytes//2
    assert decision.estimated_context_bytes<=opt.kv_bytes
    assert decision.reason=='request_or_cap'
    assert opt.as_dict()==before


def test_policy_reduces_block_to_fit_workspace_without_relaxing_budget():
    small=replace(options(),workspace_bytes=16 << 20)
    decision=choose(opt=small)
    assert 1<=decision.chunk_tokens<128
    assert decision.estimated_workspace_bytes<=small.workspace_bytes//2
    assert decision.reason=='memory_budget'
    next_estimate=HyperPrefillPolicy.estimates(dimensions(),0,decision.chunk_tokens+1)[0]
    assert next_estimate>small.workspace_bytes//2


def test_policy_fails_before_work_when_reserve_or_kv_cannot_fit():
    with pytest.raises(MemoryError,match='cannot fit one token'):
        choose(opt=replace(options(),kv_bytes=1))
    with pytest.raises(MemoryError,match='cannot fit one token'):
        HyperPrefillPolicy().select(dimensions(),options(),cache_position=0,
            remaining_tokens=364,available_bytes=options().reserve_bytes)


def test_policy_respects_context_positions_snapshot_and_decode():
    policy=HyperPrefillPolicy(256)
    opt=options()
    one=policy.select(dimensions(),opt,cache_position=1200,
        remaining_tokens=1,available_bytes=6 << 30,retained_bytes=80 << 20)
    assert one.chunk_tokens==1 and one.reason=='decode'
    assert one.estimated_context_bytes<=opt.kv_bytes
    limited=policy.select(dimensions(),opt,cache_position=1278,
        remaining_tokens=10,available_bytes=6 << 30)
    assert limited.chunk_tokens==2
    with pytest.raises(MemoryError,match='context limit'):
        policy.select(dimensions(),opt,cache_position=1280,
            remaining_tokens=1,available_bytes=6 << 30)


@pytest.mark.parametrize('bad',[True,False,0,-1,1.5,'128'])
def test_policy_rejects_invalid_cap(bad):
    with pytest.raises(ValueError):HyperPrefillPolicy(bad)


def test_attention_estimate_grows_with_cached_context():
    c=dimensions()
    small=HyperPrefillPolicy.estimates(c,0,32)[0]
    large=HyperPrefillPolicy.estimates(c,30000,32)[0]
    assert large>small


@pytest.mark.parametrize('device',['cpu','gpu'])
def test_explicit_legacy_cap_preserves_full_logits_routes_and_cache(tmp_path,device):
    import mlx.core as mx
    import numpy as np
    from test_gguf_qwen import export_fixture, arr
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    if device=='gpu' and not mx.metal.is_available():pytest.skip('Metal unavailable')
    path,_,opt,_,_=export_fixture(tmp_path)
    opt=replace(opt,device=device)
    ids=[[1,8,12,5,21]]
    with NativeGGUFQwenModel(path,opt) as reference:
        expected=reference.forward(ids)
        expected_logits=arr(expected.logits)
        expected_routes=[dict(m.routes) for m in reference.mlps]
        expected_banks={bank:{i:tuple(arr(a).copy() for a in value) if isinstance(value,tuple)
                             else arr(value).copy() for i,value in layers.items()}
                        for bank,layers in expected.past_key_values.banks.items()}
    with NativeGGUFQwenModel(path,opt) as candidate:
        candidate.prefill_policy=HyperPrefillPolicy(opt.prefill_chunk)
        actual=candidate.forward(ids)
        np.testing.assert_array_equal(arr(actual.logits),expected_logits)
        assert [dict(m.routes) for m in candidate.mlps]==expected_routes
        assert actual.past_key_values.position==len(ids[0])
        for bank,layers in actual.past_key_values.banks.items():
            for i,value in layers.items():
                before=expected_banks[bank][i]
                if isinstance(value,tuple):
                    for a,b in zip(value,before):np.testing.assert_array_equal(arr(a),b)
                else:np.testing.assert_array_equal(arr(value),before)


def test_larger_block_matches_independent_reference_tolerance_and_decode(tmp_path):
    import numpy as np
    import torch
    from test_gguf_qwen import export_fixture, arr
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    path,ref,opt,_,_=export_fixture(tmp_path)
    ids=torch.tensor([[1,8,12,5,21]])
    with torch.inference_mode():expected=ref(ids,use_cache=True)
    with NativeGGUFQwenModel(path,replace(opt,device='cpu')) as candidate:
        candidate.prefill_policy=HyperPrefillPolicy(128)
        actual=candidate.forward(ids.numpy())
        assert candidate.stats['hyperprefill_max_chunk_tokens']==5
        np.testing.assert_allclose(arr(actual.logits),expected.logits.numpy(),rtol=5e-4,atol=2e-5)
        with torch.inference_mode():
            follow=ref(torch.tensor([[4,7]]),past_key_values=expected.past_key_values,use_cache=True)
        actual_follow=candidate.forward([[4,7]],past_key_values=actual.past_key_values)
        np.testing.assert_allclose(arr(actual_follow.logits),follow.logits.numpy(),rtol=5e-4,atol=2e-5)
        assert actual_follow.past_key_values.position==7


def test_policy_refusal_rolls_back_existing_native_session(tmp_path):
    import numpy as np
    from test_gguf_qwen import export_fixture, arr
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,device='cpu')) as model:
        cache=model.forward([[1,2]]).past_key_values
        previous={i:arr(v).copy() for i,v in cache.banks['recurrent'].items()}
        tensor_bytes=model.checkpoint.stats['tensor_bytes']
        model.prefill_policy=HyperPrefillPolicy()
        model.options=replace(model.options,kv_bytes=1)
        with pytest.raises(MemoryError,match='cannot fit one token'):
            model.forward([[3,4]],past_key_values=cache,logits_to_keep=1)
        assert cache.position==2 and cache.retained==0
        assert model.checkpoint.stats['tensor_bytes']==tensor_bytes
        for i,v in cache.banks['recurrent'].items():np.testing.assert_array_equal(arr(v),previous[i])


def test_larger_blocks_preserve_per_token_routes_generation_and_eos(tmp_path,monkeypatch):
    import numpy as np
    import airllm.nextgen.gguf_qwen as module
    from test_gguf_qwen import export_fixture
    path,_,opt,_,_=export_fixture(tmp_path)
    opt=replace(opt,device='cpu')
    ids=[[1,8,12,5,21]]
    captured=[]
    route=module.qwen_route
    def tracked(*args,**kwargs):
        selected,scores=route(*args,**kwargs)
        captured.append((np.asarray(selected).copy(),np.asarray(scores).copy()))
        return selected,scores
    monkeypatch.setattr(module,'qwen_route',tracked)
    with module.NativeGGUFQwenModel(path,opt) as reference:
        reference.forward(ids,logits_to_keep=1)
        layers=len(reference.mlps)
        expected_routes=[np.concatenate([a for a,b in captured[i::layers]],axis=0)
                         for i in range(layers)]
        expected_scores=[np.concatenate([b for a,b in captured[i::layers]],axis=0)
                         for i in range(layers)]
        expected_tokens=np.asarray(reference.generate(ids,max_new_tokens=4,eos_token_id=[])).copy()
    captured.clear()
    with module.NativeGGUFQwenModel(path,opt,hyper_prefill=True,hyper_prefill_max_tokens=128) as candidate:
        candidate.forward(ids,logits_to_keep=1)
        assert len(captured)==layers
        for (actual_ids,actual_scores),before_ids,before_scores in zip(captured,expected_routes,expected_scores):
            np.testing.assert_array_equal(actual_ids,before_ids)
            np.testing.assert_allclose(actual_scores,before_scores,rtol=5e-4,atol=2e-5)
        np.testing.assert_array_equal(np.asarray(candidate.generate(ids,max_new_tokens=4,eos_token_id=[])),expected_tokens)
        eos=int(expected_tokens[0,len(ids[0])])
        stopped=np.asarray(candidate.generate(ids,max_new_tokens=4,eos_token_id=eos))
        assert stopped.tolist()==[ids[0]+[eos]]


@pytest.mark.parametrize('keyword,value,match',[
    ('hyper_prefill',1,'must be boolean'),
    ('hyper_prefill_max_tokens',0,'positive integer'),
    ('hyper_prefill_max_tokens',True,'positive integer'),
])
def test_public_prefill_options_reject_invalid_values_before_open(tmp_path,keyword,value,match):
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    with pytest.raises(ValueError,match=match):
        NativeGGUFQwenModel(tmp_path/'does-not-exist.gguf',**{keyword:value})


def test_new_policy_is_disabled_by_default_and_reference_path_can_opt_in(tmp_path):
    from test_gguf_qwen import export_fixture
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,device='cpu')) as legacy:
        assert legacy.prefill_policy is None
        legacy.forward([[1,2,3]],logits_to_keep=1)
        assert not legacy.stats.get('hyperprefill_blocks')
    # Floating/reference GGUF still has the same runtime tile/IO guards.
    # Planning may overestimate routed retention, but never removes those guards.
    with NativeGGUFQwenModel(path,replace(opt,device='cpu'),hyper_prefill=True) as candidate:
        candidate.forward([[1,2,3]],logits_to_keep=1)
        assert candidate.stats['hyperprefill_max_chunk_tokens']==3


def test_larger_prefill_mid_layer_error_restores_recurrent_kv_and_positions(tmp_path):
    import numpy as np
    from test_gguf_qwen import export_fixture, arr
    from airllm.nextgen.gguf_qwen import NativeGGUFQwenModel
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,device='cpu'),hyper_prefill=True) as model:
        cache=model.forward([[1,2]],logits_to_keep=1).past_key_values
        snapshot={bank:{i:tuple(arr(a).copy() for a in value) if isinstance(value,tuple)
                           else arr(value).copy() for i,value in layers.items()}
                  for bank,layers in cache.banks.items()}
        original=model.mlps[1]
        def failed_mlp(x):raise OSError('injected expert read failure')
        model.mlps[1]=failed_mlp
        with pytest.raises(OSError,match='injected expert read failure'):
            model.forward([[3,4,5]],past_key_values=cache,logits_to_keep=1)
        assert cache.position==2 and cache.retained==0
        for bank,layers in cache.banks.items():
            assert layers.keys()==snapshot[bank].keys()
            for i,value in layers.items():
                if isinstance(value,tuple):
                    for a,b in zip(value,snapshot[bank][i]):np.testing.assert_array_equal(arr(a),b)
                else:np.testing.assert_array_equal(arr(value),snapshot[bank][i])
        model.mlps[1]=original
        retry=arr(model.forward([[3,4,5]],past_key_values=cache,logits_to_keep=1).logits)
        fresh=arr(model.forward([[1,2,3,4,5]],logits_to_keep=1).logits)
        np.testing.assert_allclose(retry,fresh,rtol=5e-4,atol=2e-5)
