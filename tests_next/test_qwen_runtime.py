"""Actual miniature hybrid Qwen inference against independent Transformers.

These random tiny checkpoints validate equations, streaming and state behavior.
They establish neither trained model quality nor full Qwen3.8 compatibility.
"""
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
import mlx.core as mx
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from airllm.nextgen.config import NextConfig
from airllm.nextgen.qwen_runtime import ExperimentalQwenModel,normalize_qwen_config


def make_qwen(tmp_path,*,bf16=False):
    torch.manual_seed(138)
    c=Qwen3_5MoeTextConfig(vocab_size=64,hidden_size=32,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        linear_conv_kernel_dim=3,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=4,
        moe_intermediate_size=16,shared_expert_intermediate_size=16,
        num_experts=4,num_experts_per_tok=2,max_position_embeddings=256,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5},
        tie_word_embeddings=False,pad_token_id=None,bos_token_id=1,eos_token_id=None)
    c._attn_implementation='eager'
    ref=Qwen3_5MoeForCausalLM(c).eval()
    if bf16:ref=ref.to(torch.bfloat16)
    root=tmp_path/'qwen';root.mkdir()
    config=c.to_dict();config['mamba_ssm_dtype']='float32';(root/'config.json').write_text(json.dumps(config))
    save_file({k:v.contiguous() for k,v in ref.state_dict().items()},str(root/'model.safetensors'))
    opt=NextConfig(cache_bytes=0,io_bytes=2*1024**2,kv_bytes=2*1024**2,
                   workspace_bytes=2*1024**2,max_tensor_bytes=1024**2,
                   reserve_bytes=0,prefill_chunk=16,max_context=128,
                   device='cpu',dtype='bfloat16' if bf16 else 'float32',load_tokenizer=False,
                   prefetch=False,cache_dir=str(tmp_path/'metadata'))
    return root,ref,opt


def arr(x):return np.asarray(x.astype(mx.float32))


def test_qwen_full_hybrid_prefill_and_cached_decode(tmp_path):
    root,ref,opt=make_qwen(tmp_path);ids=torch.tensor([[1,8,12,5,21,3,11]])
    with torch.inference_mode():expected=ref(ids,use_cache=True)
    with ExperimentalQwenModel(root,opt) as model:
        actual=model.forward(ids.numpy());cache=actual.past_key_values
        np.testing.assert_allclose(arr(actual.logits),expected.logits.numpy(),rtol=4e-4,atol=1e-5)
        with torch.inference_mode():expected_next=ref(torch.tensor([[4,7]]),past_key_values=expected.past_key_values,use_cache=True)
        actual_next=model.forward([[4,7]],past_key_values=cache)
        np.testing.assert_allclose(arr(actual_next.logits),expected_next.logits.numpy(),rtol=4e-4,atol=1e-5)
        assert cache.position==9
        assert len(cache.banks['recurrent'])==3 and len(cache.banks['kv'])==1
        assert cache.peak_bytes<=opt.kv_bytes
        assert model.reader.io.peak_bytes<=opt.io_bytes
        assert model.reader.cache.peak_bytes==0


def test_qwen_greedy_generation_matches_official(tmp_path):
    root,ref,opt=make_qwen(tmp_path);ids=torch.tensor([[1,8,12,5]])
    expected=ids.clone()
    with torch.inference_mode():
        for _ in range(8):expected=torch.cat([expected,ref(expected,use_cache=False).logits[:,-1].argmax(-1)[:,None]],dim=1)
    with ExperimentalQwenModel(root,opt) as model:
        actual=model.generate(ids.numpy(),max_new_tokens=8)
        np.testing.assert_array_equal(np.asarray(actual),expected.numpy())
        assert model.metrics()['runtime']['tokens_generated']==8
        assert model.metrics()['experimental'] and not model.metrics()['full_qwen_checkpoint_validated']


def test_qwen_state_snapshot_rollback_matches_fresh_prefix(tmp_path):
    root,ref,opt=make_qwen(tmp_path)
    with ExperimentalQwenModel(root,opt) as model:
        cache=model.forward([[1,2,3]]).past_key_values;snapshot=cache.snapshot()
        model.forward([[4,5]],past_key_values=cache);cache.restore(snapshot)
        assert cache.position==3
        actual=model.forward([[7,8]],past_key_values=cache).logits
        expected=model.forward([[1,2,3,7,8]]).logits[:,-2:]
        np.testing.assert_allclose(arr(actual),arr(expected),rtol=4e-4,atol=1e-5)
        with pytest.raises(NotImplementedError,match='snapshot'):cache.truncate(2)
        with pytest.raises(ValueError,match='inactive'):cache.restore(snapshot)


def test_qwen_forward_failure_restores_recurrent_and_conv(tmp_path):
    root,ref,opt=make_qwen(tmp_path)
    with ExperimentalQwenModel(root,opt) as model:
        cache=model.forward([[1,2,3]]).past_key_values
        old={k:arr(v).copy() for k,v in cache.banks['recurrent'].items()}
        old_conv={k:arr(v).copy() for k,v in cache.banks['conv'].items()}
        original=model.mlps[1]
        class Failure:
            def __call__(self,x):raise RuntimeError('injected failure')
        model.mlps[1]=Failure()
        with pytest.raises(RuntimeError,match='injected'):model.forward([[4]],past_key_values=cache)
        assert cache.position==3 and cache.retained==0
        for k,v in old.items():np.testing.assert_array_equal(arr(cache.banks['recurrent'][k]),v)
        for k,v in old_conv.items():np.testing.assert_array_equal(arr(cache.banks['conv'][k]),v)
        model.mlps[1]=original
        actual=model.forward([[4]],past_key_values=cache).logits
        expected=model.forward([[1,2,3,4]]).logits[:,-1:]
        np.testing.assert_allclose(arr(actual),arr(expected),rtol=4e-4,atol=1e-5)


def test_qwen_snapshot_budget_and_foreign_session(tmp_path):
    root,ref,opt=make_qwen(tmp_path)
    with ExperimentalQwenModel(root,opt) as model:
        cache=model.forward([[1,2]]).past_key_values
        cache.budget=cache.bytes
        with pytest.raises(MemoryError,match='snapshot'):cache.snapshot()
        assert cache.position==2
        with pytest.raises(ValueError,match='another model'):model.forward([[3]],past_key_values=object())


def test_qwen_unsupported_gate_and_modelopt_fail_before_mlx(tmp_path):
    root,ref,opt=make_qwen(tmp_path);config=json.loads((root/'config.json').read_text())
    config['output_gate_type']='sigmoid'
    with pytest.raises(NotImplementedError,match='DeltaNet'):normalize_qwen_config(config)
    config.pop('output_gate_type');config['quantization_config']={'quant_method':'modelopt'}
    with pytest.raises(NotImplementedError,match='ModelOpt'):normalize_qwen_config(config)


@pytest.mark.parametrize('gate',['swish','silu'])
@pytest.mark.parametrize('device',['cpu','gpu'])
def test_qwen_declared_swish_is_deltanet_silu_and_preserves_official_attention(tmp_path,gate,device):
    if device=='gpu' and not mx.metal.is_available():pytest.skip('Metal hardware unavailable')
    root,ref,opt=make_qwen(tmp_path)
    opt=replace(opt,device=device)
    config=json.loads((root/'config.json').read_text());config['output_gate_type']=gate
    (root/'config.json').write_text(json.dumps(config))
    ids=torch.tensor([[1,8,12,5,21,3,11]])
    with torch.inference_mode():expected=ref(ids,use_cache=True)
    with ExperimentalQwenModel(root,opt) as model:
        actual=model.forward(ids.numpy())
        np.testing.assert_allclose(arr(actual.logits),expected.logits.numpy(),rtol=4e-4,atol=1e-5)
        with torch.inference_mode():expected_next=ref(torch.tensor([[4]]),past_key_values=expected.past_key_values,use_cache=True)
        actual_next=model.forward([[4]],past_key_values=actual.past_key_values)
        np.testing.assert_allclose(arr(actual_next.logits),expected_next.logits.numpy(),rtol=4e-4,atol=1e-5)


def test_qwen_split_prefill_matches_whole_block(tmp_path):
    root,ref,opt=make_qwen(tmp_path)
    with ExperimentalQwenModel(root,opt) as model:expected=arr(model.forward([[1,7,5,9,11]]).logits)
    with ExperimentalQwenModel(root,replace(opt,prefill_chunk=2)) as model:
        actual=arr(model.forward([[1,7,5,9,11]]).logits)
        np.testing.assert_allclose(actual,expected,rtol=4e-4,atol=1e-5)


def test_qwen_explicit_automodel_backend_and_unsupported_speculation(tmp_path):
    from airllm import AutoModel
    root,ref,opt=make_qwen(tmp_path)
    with AutoModel.from_pretrained(root,backend='next-qwen-experimental',next_config=opt) as model:
        assert model.generate([[1,8]],max_new_tokens=1).shape==(1,3)
        with pytest.raises(NotImplementedError,match='snapshot integration'):
            model.generate([[1]],draft_model=model)


def test_qwen_bfloat16_full_miniature_checkpoint(tmp_path):
    root,ref,opt=make_qwen(tmp_path,bf16=True);ids=torch.tensor([[1,8,12,5]])
    with torch.inference_mode():expected=ref(ids,use_cache=False).logits.float().numpy()
    with ExperimentalQwenModel(root,opt) as model:
        actual=arr(model.forward(ids.numpy()).logits)
        # Different BF16 reduction orders are observable. This tolerance is
        # fixture-specific, not a certificate for trained-checkpoint quality.
        np.testing.assert_allclose(actual,expected,rtol=.08,atol=.008)
