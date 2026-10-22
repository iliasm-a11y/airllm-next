import json
import os
import numpy as np
import pytest
import torch
import mlx.core as mx
from airllm.nextgen import NextConfig,AirLLMNextModel
from airllm.nextgen.attention import rope
from airllm.nextgen.moe import route


def options(tmp_path,**kw):
    d=dict(cache_bytes=4*1024**2,io_bytes=1024**2,kv_bytes=1024**2,max_tensor_bytes=1024**2,
           reserve_bytes=0,workspace_bytes=1024**2,device=os.environ.get('AIRLLM_TEST_DEVICE','cpu'),dtype='float32',prefill_chunk=2,
           max_context=128,load_tokenizer=False,cache_dir=str(tmp_path/'metadata'))
    d.update(kw);return NextConfig(**d)


def test_rope_exact_official():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave
    x=torch.randn(1,3,5,8);t=torch.arange(7,12).float();freq=10000**(-torch.arange(0,8,2).float()/8)
    a=t[:,None]*freq[None];cos=torch.cat([a,a],-1).cos()[None];sin=torch.cat([a,a],-1).sin()[None]
    y,_=apply_rotary_pos_emb_interleave(x,x,cos,sin)
    actual=rope(mx,mx.array(x.numpy()),7,10000.,8)
    np.testing.assert_allclose(np.array(actual),y.numpy(),atol=2e-6,rtol=2e-6)


def test_router_exact(tiny_glm):
    path,ref=tiny_glm;mod=ref.model.layers[1].mlp.gate;c=ref.config.to_dict()
    with torch.no_grad():mod.e_score_correction_bias.copy_(torch.linspace(-1,1,8))
    x=torch.randn(9,48)
    with torch.no_grad():_,w,ids=mod(x)
    ai,aw=route(mx,mx.array(x.numpy()),mx.array(mod.weight.detach().numpy()),mx.array(mod.e_score_correction_bias.numpy()),c)
    a=np.array(ai);v=np.array(aw)
    for j in range(len(a)):
        actual={int(k):float(vv) for k,vv in zip(a[j],v[j])};expected={int(k):float(vv) for k,vv in zip(ids[j],w[j])}
        assert actual.keys()==expected.keys()
        for k in actual:assert actual[k]==pytest.approx(expected[k],abs=3e-7)


def test_logits_cache_generation(tiny_glm,tmp_path):
    root,ref=tiny_glm;ids=torch.tensor([[1,8,9,10,7,11]])
    with AirLLMNextModel(root,next_config=options(tmp_path)) as model,torch.inference_mode():
        a=model(ids);b=ref(ids,use_cache=True)
        np.testing.assert_allclose(np.array(a.logits),b.logits.numpy(),atol=4e-6,rtol=2e-4)
        next_id=b.logits[:,-1].argmax(-1)[:,None]
        a2=model(next_id,past_key_values=a.past_key_values)
        b2=ref(next_id,past_key_values=b.past_key_values)
        np.testing.assert_allclose(np.array(a2.logits),b2.logits.numpy(),atol=4e-6,rtol=2e-4)
        generated=np.array(model.generate(ids,max_new_tokens=6))
        expected=ref.generate(ids,max_new_tokens=6,do_sample=False).numpy()
        np.testing.assert_array_equal(generated,expected)
        assert model.reader.cache.peak_bytes<=model.options.cache_bytes
        assert model.reader.io.peak_bytes<=model.options.io_bytes


@pytest.mark.parametrize('cache_bytes,prefetch',[(0,False),(10000,True),(4*1024**2,True)])
def test_saturated_cache_unchanged(tiny_glm,tmp_path,cache_bytes,prefetch):
    path,ref=tiny_glm;ids=torch.tensor([[1,9,14]])
    with AirLLMNextModel(path,next_config=options(tmp_path,cache_bytes=cache_bytes,prefetch=prefetch)) as model,torch.inference_mode():
        np.testing.assert_allclose(np.array(model(ids).logits),ref(ids).logits.numpy(),atol=4e-6,rtol=2e-4)
        assert model.reader.cache.bytes<=cache_bytes


def test_entry_and_safe_context(tiny_glm,tmp_path):
    from airllm import AutoModel
    path,_=tiny_glm
    with AutoModel.from_pretrained(path,backend='mlx-next',next_config=options(tmp_path,max_context=3)) as model:
        with pytest.raises(MemoryError,match='Context'):
            model([[1,8,9,10]])
    with pytest.raises(RuntimeError,match='closed'):model([[1]])
