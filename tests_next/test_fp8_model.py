import json
import math
import numpy as np
import pytest
import torch
from safetensors.torch import load_file,save_file
from transformers import GlmMoeDsaConfig,GlmMoeDsaForCausalLM
from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.sidecar import convert_experts
from airllm.nextgen.quantization import dequant_affine
from test_native import options


def save_glm64(root,fp8=False):
    torch.manual_seed(17)
    cfg=GlmMoeDsaConfig(vocab_size=96,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=256,n_routed_experts=8,
        num_experts_per_tok=2,n_shared_experts=1,first_k_dense_replace=1,moe_intermediate_size=32,
        kv_lora_rank=16,q_lora_rank=32,qk_rope_head_dim=8,qk_nope_head_dim=8,v_head_dim=8,
        index_head_dim=16,index_n_heads=8,index_topk=4,indexer_types=['full','shared','shared','full'],
        n_group=2,topk_group=1,rope_parameters={'rope_type':'default','rope_theta':10000.},
        pad_token_id=0,bos_token_id=1,eos_token_id=2)
    ref=GlmMoeDsaForCausalLM(cfg).eval()
    with torch.no_grad():
        # Positive multi-head index weights avoid undefined boundary-tie ordering
        # in randomly initialized indexers. Real GLM has 32 index heads.
        for layer in ref.model.layers:
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer.weights_proj.weight.abs_()
        for name,buf in ref.named_buffers():
            if name.endswith('e_score_correction_bias'):buf.copy_(torch.linspace(-.05,.1,buf.numel()))
        if fp8:
            for p in ref.parameters():
                if p.ndim>=2:p.copy_(p.to(torch.float8_e4m3fn).float())
    ref.save_pretrained(root,max_shard_size='100KB')
    if fp8:
        index=json.loads((root/'model.safetensors.index.json').read_text())
        for filename in sorted(set(index['weight_map'].values())):
            state=load_file(root/filename);out={}
            for key,w in state.items():
                if w.ndim==2 and not any(t in key for t in ['.gate.weight','indexer.weights_proj','embed_tokens','lm_head']):
                    out[key]=w.to(torch.float8_e4m3fn);sk=key+'_scale_inv'
                    out[sk]=torch.ones(math.ceil(w.shape[0]/16),math.ceil(w.shape[1]/16));index['weight_map'][sk]=filename
                else:out[key]=w
            save_file(out,root/filename)
        (root/'model.safetensors.index.json').write_text(json.dumps(index))
        c=json.loads((root/'config.json').read_text());c['quantization_config']={'quant_method':'fp8','fmt':'e4m3','weight_block_size':[16,16],'activation_scheme':'dynamic'}
        (root/'config.json').write_text(json.dumps(c))
    return ref


def test_native_fp8_full_prefill_and_decode(tmp_path):
    root=tmp_path/'fp8';ref=save_glm64(root,True);ids=torch.tensor([[1,8,9,10,7,11,12]])
    with AirLLMNextModel(root,next_config=options(tmp_path,prefill_chunk=3)) as model,torch.inference_mode():
        a=model(ids);b=ref(ids)
        np.testing.assert_allclose(np.array(a.logits),b.logits.numpy(),atol=6e-6,rtol=3e-4)
        np.testing.assert_array_equal(np.array(model.generate(ids,max_new_tokens=10)),ref.generate(ids,max_new_tokens=10).numpy())
        assert model.reader.stats['dequant_seconds']>0


@pytest.mark.parametrize('bits',[4,8])
def test_converted_experts_end_to_end(tmp_path,bits):
    root=tmp_path/'ref';ref=save_glm64(root)
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:
        sidecar=convert_experts(cp,ref.config.to_dict(),tmp_path/'quant',bits,32,row_chunk=7)
        assert sidecar.entries
        # Replace only expert matrices in the reference, using exactly the
        # dequantized sidecar representation, to isolate kernel correctness.
        with torch.no_grad():
            for i,layer in enumerate(ref.model.layers):
                if i==0:continue
                bank=layer.mlp.experts
                for e in range(8):
                    base=f'model.layers.{i}.mlp.experts.{e}.'
                    gate=dequant_affine(sidecar.load(base+'gate_proj.weight'))
                    up=dequant_affine(sidecar.load(base+'up_proj.weight'))
                    down=dequant_affine(sidecar.load(base+'down_proj.weight'))
                    bank.gate_up_proj[e].copy_(torch.from_numpy(np.concatenate([gate,up],axis=0)))
                    bank.down_proj[e].copy_(torch.from_numpy(down))
    ids=torch.tensor([[1,8,9,7]])
    with AirLLMNextModel(root,next_config=options(tmp_path,quantized_dir=str(tmp_path/'quant'))) as model,torch.inference_mode():
        actual=np.array(model(ids).logits);expected=ref(ids).logits.numpy()
        np.testing.assert_allclose(actual,expected,atol=6e-6,rtol=3e-4)
        np.testing.assert_array_equal(np.array(model.generate(ids,max_new_tokens=8)),ref.generate(ids,max_new_tokens=8).numpy())


@pytest.mark.parametrize('linear_bits',[4,8])
def test_mixed_quantization_and_tiled_output(tmp_path,linear_bits):
    root=tmp_path/'mixed';ref=save_glm64(root)
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:
        sidecar=convert_experts(cp,ref.config.to_dict(),tmp_path/'quant',4,32,scope='mixed',linear_bits=linear_bits)
        assert sidecar.entries['lm_head.weight']['bits']==linear_bits
        assert not any('kv_b_proj' in k for k in sidecar.entries)
        assert not any('.mlp.gate.' in k for k in sidecar.entries)
        p=sidecar.load('lm_head.weight',rows=(7,19))
        np.testing.assert_array_equal(dequant_affine(p),dequant_affine(sidecar.load('lm_head.weight'))[7:19])
        # Apply non-expert sidecar weights to exact dequantized reference.
        with torch.no_grad():
            for key in sidecar.entries:
                if '.experts.' not in key:
                    ref.get_parameter(key).copy_(torch.from_numpy(dequant_affine(sidecar.load(key))))
            for i,layer in enumerate(ref.model.layers[1:],1):
                bank=layer.mlp.experts
                for e in range(8):
                    base=f'model.layers.{i}.mlp.experts.{e}.'
                    gu=np.concatenate([dequant_affine(sidecar.load(base+n+'.weight')) for n in ('gate_proj','up_proj')],axis=0)
                    bank.gate_up_proj[e].copy_(torch.from_numpy(gu));bank.down_proj[e].copy_(torch.from_numpy(dequant_affine(sidecar.load(base+'down_proj.weight'))))
    with AirLLMNextModel(root,next_config=options(tmp_path,quantized_dir=str(tmp_path/'quant'))) as model,torch.inference_mode():
        actual=np.array(model([[1,8,9]]).logits);expected=ref(torch.tensor([[1,8,9]])).logits.numpy()
        np.testing.assert_allclose(actual,expected,atol=6e-6,rtol=3e-4)
        # Force tiled head without changing other group size guards.
        from dataclasses import replace
        model.reader.cfg=replace(model.reader.cfg,max_tensor_bytes=1000,row_chunk=7)
        x=model.mx.ones((1,1,64))
        actual=model.reader.output(x,'lm_head.weight')
        expected=x@model.mx.array(dequant_affine(sidecar.load('lm_head.weight'))).T
        np.testing.assert_allclose(np.array(actual),np.array(expected),atol=1e-6,rtol=1e-5)


def test_fused_3d_fp8_expert_banks_are_sliced(tmp_path):
    root=tmp_path/'fused';ref=save_glm64(root,True)
    index=json.loads((root/'model.safetensors.index.json').read_text());state={}
    for name in sorted(set(index['weight_map'].values())):state.update(load_file(root/name))
    for i in range(1,4):
        prefix=f'model.layers.{i}.mlp.experts.';gu=[];down=[];gs=[];ds=[]
        for e in range(8):
            keys=[prefix+f'{e}.{name}.weight' for name in ('gate_proj','up_proj','down_proj')]
            gu.append(torch.cat([state.pop(keys[0]).view(torch.uint8),state.pop(keys[1]).view(torch.uint8)],0).view(torch.float8_e4m3fn))
            down.append(state.pop(keys[2]));gs.append(torch.cat([state.pop(keys[0]+'_scale_inv'),state.pop(keys[1]+'_scale_inv')],0));ds.append(state.pop(keys[2]+'_scale_inv'))
        state[prefix+'gate_up_proj']=torch.stack([x.view(torch.uint8) for x in gu]).view(torch.float8_e4m3fn)
        state[prefix+'down_proj']=torch.stack([x.view(torch.uint8) for x in down]).view(torch.float8_e4m3fn)
        state[prefix+'gate_up_proj_scale_inv']=torch.stack(gs);state[prefix+'down_proj_scale_inv']=torch.stack(ds)
    # Isolated fixture rewrite only, never a user's checkpoint.
    for p in root.glob('*.safetensors'):p.unlink()
    (root/'model.safetensors.index.json').unlink();save_file(state,root/'model.safetensors')
    with AirLLMNextModel(root,next_config=options(tmp_path,cache_bytes=0)) as model,torch.inference_mode():
        a=model([[1,8,9]]).logits;b=ref(torch.tensor([[1,8,9]])).logits
        np.testing.assert_allclose(np.array(a),b.numpy(),atol=6e-6,rtol=3e-4)
        assert model.mlps[1].fused


def test_bfloat16_native_path(tmp_path):
    root=tmp_path/'bf16';ref=save_glm64(root).to(torch.bfloat16)
    ref.save_pretrained(root,max_shard_size='100KB')
    with AirLLMNextModel(root,next_config=options(tmp_path,dtype='bfloat16')) as model,torch.inference_mode():
        a=np.array(model([[1,8,9]]).logits.astype(model.mx.float32));b=ref(torch.tensor([[1,8,9]])).logits.float().numpy()
        np.testing.assert_allclose(a,b,atol=.007,rtol=.08)


def test_default_bfloat16_quantized_model_generates(tmp_path):
    root=tmp_path/'bf16quant';ref=save_glm64(root).to(torch.bfloat16)
    ref.save_pretrained(root,max_shard_size='100KB')
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:
        convert_experts(cp,ref.config.to_dict(),tmp_path/'quant',4,32)
    with AirLLMNextModel(root,next_config=options(tmp_path,dtype='bfloat16',quantized_dir=str(tmp_path/'quant'))) as model:
        logits=model([[1,8,9]]).logits
        assert np.isfinite(np.array(logits.astype(model.mx.float32))).all()
        assert model.generate([[1,8,9]],max_new_tokens=8,eos_token_id=[]).shape==(1,11)


def test_resume_refuses_silent_mixed_precision_change(tmp_path):
    root=tmp_path/'resume';ref=save_glm64(root)
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:
        convert_experts(cp,ref.config.to_dict(),tmp_path/'quant',4,32,scope='mixed',linear_bits=8)
        with pytest.raises(ValueError,match='per-matrix precision'):
            convert_experts(cp,ref.config.to_dict(),tmp_path/'quant',4,32,scope='mixed',linear_bits=4)
