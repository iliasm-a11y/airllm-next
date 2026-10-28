"""Resource and selected-expert invariants for the native GLM5 path."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from airllm.nextgen.gguf_glm5 import NativeGGUFGLM5Model, _qualified_topk
from test_glm53_runtime import export_glm53_fixture


def test_recurrent_budget_rejected_before_any_weights(tmp_path):
    path, _, _, options = export_glm53_fixture(tmp_path)
    with NativeGGUFGLM5Model(path, replace(options, kv_bytes=16)) as model:
        cache=model.new_session()
        with pytest.raises(MemoryError, match='recurrent state exceeds kv_bytes'):
            model.forward([[1]], past_key_values=cache)
        assert model.checkpoint.stats['tensor_bytes']==0
        assert cache.position==0 and cache.valid


def test_reads_cover_exact_official_routed_expert_sets(tmp_path, monkeypatch):
    path, ref, head, options = export_glm53_fixture(tmp_path)
    expected={}; handles=[]
    def capture(layer):
        def hook(module, args, output):
            expected[layer]=set(output[2][0].tolist())
        return hook
    for layer, block in enumerate(ref.layers):
        if hasattr(block.mlp,'gate'):
            handles.append(block.mlp.gate.register_forward_hook(capture(layer)))
    with torch.inference_mode():
        logits=head(ref(torch.tensor([[7]]),use_cache=False).last_hidden_state).numpy()
    for handle in handles:handle.remove()
    with NativeGGUFGLM5Model(path, replace(options,cache_bytes=0)) as model:
        seen={};read=model.checkpoint.read_rows
        def tracked(name,start,count,**kwargs):
            if '_exps.weight' in name:
                layer=int(name.split('.')[1]);expert=kwargs.get('expert')
                assert type(expert) is int
                assert count<=options.row_chunk
                seen.setdefault(layer,set()).add(expert)
            return read(name,start,count,**kwargs)
        monkeypatch.setattr(model.checkpoint,'read_rows',tracked)
        output=model.forward([[7]])
        np.testing.assert_allclose(np.asarray(output.logits),logits,rtol=2e-4,atol=2e-5)
        assert seen==expected
        assert {layer:set(ids) for layer,ids in model.last_routes.items()}==expected


@pytest.mark.parametrize('option', ['prefetch','spec_prefetch','output_prefetch','load_tokenizer'])
def test_unqualified_options_fail_explicitly_before_open(tmp_path,option):
    path, _, _, options = export_glm53_fixture(tmp_path)
    with pytest.raises(NotImplementedError):
        NativeGGUFGLM5Model(path,replace(options,**{option:True}))


def test_invalid_convolution_representation_rejected_without_payload(tmp_path):
    path, _, _, options = export_glm53_fixture(tmp_path)
    with NativeGGUFGLM5Model(path,options) as model:
        from dataclasses import replace as replace_tensor
        name='blk.0.ssm_conv1d_q.weight'
        info=model.checkpoint.tensors[name]
        # Same element count, different tensor representation: not equivalent
        # to a canonical channel-by-kernel depthwise convolution.
        model.checkpoint.tensors[name]=replace_tensor(info,dimensions=(8,1,4))
        with pytest.raises(ValueError,match='convolution'):
            model._validate_layout()
        assert model.checkpoint.stats['tensor_bytes']==0


@pytest.mark.parametrize('role',['MoE','KPool'])
def test_exact_topk_boundary_tie_is_diagnosed(role):
    with pytest.raises(NotImplementedError,match=role+' top-k boundary'):
        _qualified_topk([0.9,0.8,0.8,0.1],2,role)
    # Interior ties retain all tied experts; no required winner is omitted.
    assert set(_qualified_topk([0.9,0.9,0.1,0.0],2,role))=={0,1}


def test_tf32_request_is_rejected_before_open(tmp_path,monkeypatch):
    _, _, _, options=export_glm53_fixture(tmp_path)
    monkeypatch.setenv('MLX_ENABLE_TF32','1')
    with pytest.raises(NotImplementedError,match='TF32=0'):
        NativeGGUFGLM5Model(tmp_path/'not-opened.gguf',options)


def test_unconfigured_preimported_metal_is_rejected_before_open(tmp_path,monkeypatch):
    import mlx.core as mx
    _, _, _, options=export_glm53_fixture(tmp_path)
    monkeypatch.delenv('MLX_ENABLE_TF32',raising=False)
    monkeypatch.setattr(mx.metal,'is_available',lambda:True)
    with pytest.raises(NotImplementedError,match='before importing MLX'):
        NativeGGUFGLM5Model(tmp_path/'not-opened.gguf',options)
