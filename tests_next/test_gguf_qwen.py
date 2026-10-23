"""Complete miniature GGUF hybrid inference against independent Transformers.

The fixture exports floating weights into the *converted* GGUF conventions.
Codec parity is separately tested in test_gguf.py; miniature logits do not
certify trained 35B quality or real-checkpoint throughput.
"""
from dataclasses import replace
import json
import struct
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch

from airllm.nextgen.gguf import GGUFReader
from airllm.nextgen.gguf_qwen import (GGUFWeights, NativeGGUFQwenModel,
                                     gguf_qwen_config, v_head_permutation)
from test_qwen_runtime import make_qwen


def write_gguf(path,metadata,tensors):
    """Small independent F32 GGUF v3 fixture writer, with aligned offsets."""
    def string(value):
        raw=value.encode();return struct.pack('<Q',len(raw))+raw
    def value(item):
        if isinstance(item,str):return struct.pack('<I',8)+string(item)
        if type(item) is bool:return struct.pack('<I?',7,item)
        if type(item) is int:return struct.pack('<II',4,item)
        if type(item) is float:return struct.pack('<If',6,item)
        if isinstance(item,list):
            return struct.pack('<IIQ',9,4,len(item))+b''.join(struct.pack('<I',v) for v in item)
        raise TypeError(item)
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata))
    for key,item in metadata.items():header+=string(key)+value(item)
    payload=bytearray()
    for name,array in tensors.items():
        while len(payload)%32:payload.append(0)
        dims=array.shape[::-1]
        header+=string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',0,len(payload))
        payload.extend(np.asarray(array,dtype='<f4').tobytes())
    header+=bytes((-len(header))%32)
    path.write_bytes(header+payload)


def export_fixture(tmp_path,*,reordered=True,noninvolutive=False):
    root,ref,opt=make_qwen(tmp_path)
    c=json.loads((root/'config.json').read_text());tensors={}
    if noninvolutive:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
        c['linear_num_key_heads']=4;c['linear_num_value_heads']=8
        changed=Qwen3_5MoeTextConfig(**c);changed._attn_implementation='eager'
        torch.manual_seed(911);ref=Qwen3_5MoeForCausalLM(changed).eval();c=changed.to_dict()
    # Independent exporter: loop the head groups, rather than using the
    # implementation's permutation helper or weight-name adapter.
    kh=c['linear_num_key_heads'];vh=c['linear_num_value_heads'];hd=c['linear_value_head_dim'];repeats=vh//kh
    def convert_heads(array,dim,head_dim):
        if not reordered:return array
        order=[(group*repeats+slot)*head_dim+j for slot in range(repeats) for group in range(kh) for j in range(head_dim)]
        return np.take(array,order,axis=dim)
    def convert(name,array):
        if name.endswith('A_log'):array=-np.exp(array)
        elif name.endswith('norm.weight') and not name.endswith('linear_attn.norm.weight'):array=array+1
        if '.linear_attn.' in name:
            if name.endswith(('in_proj_qkv.weight','conv1d.weight')):
                if name.endswith('conv1d.weight'):array=array[:,0,:]
                kd=2*kh*c['linear_key_head_dim']
                array=np.concatenate([array[:kd],convert_heads(array[kd:],0,hd)])
            elif name.endswith('in_proj_z.weight'):array=convert_heads(array,0,hd)
            elif name.endswith(('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias')):array=convert_heads(array,0,1)
            elif name.endswith('out_proj.weight'):array=convert_heads(array,1,hd)
        return array
    names={'model.embed_tokens.weight':'token_embd.weight','model.norm.weight':'output_norm.weight','lm_head.weight':'output.weight'}
    suffixes={
        'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'post_attention_norm.weight',
        'linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight',
        'linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight',
        'linear_attn.A_log':'ssm_a','linear_attn.dt_bias':'ssm_dt','linear_attn.conv1d.weight':'ssm_conv1d.weight',
        'linear_attn.norm.weight':'ssm_norm.weight','linear_attn.out_proj.weight':'ssm_out.weight',
        'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight','self_attn.v_proj.weight':'attn_v.weight',
        'self_attn.o_proj.weight':'attn_output.weight','self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
        'mlp.gate.weight':'ffn_gate_inp.weight','mlp.shared_expert_gate.weight':'ffn_gate_inp_shexp.weight',
        'mlp.shared_expert.gate_proj.weight':'ffn_gate_shexp.weight','mlp.shared_expert.up_proj.weight':'ffn_up_shexp.weight',
        'mlp.shared_expert.down_proj.weight':'ffn_down_shexp.weight','mlp.experts.down_proj':'ffn_down_exps.weight'}
    for name,weight in ref.state_dict().items():
        array=weight.detach().float().numpy().copy()
        if name in names:tensors[names[name]]=convert(name,array);continue
        prefix,layer,suffix=name.split('.',3)[1:]
        if suffix=='mlp.experts.gate_up_proj':
            gate,up=np.split(array,2,axis=-2)
            tensors[f'blk.{layer}.ffn_gate_exps.weight']=gate;tensors[f'blk.{layer}.ffn_up_exps.weight']=up
        else:tensors[f'blk.{layer}.'+suffixes[suffix]]=convert(name,array)
    m={'general.architecture':'qwen35moe','general.alignment':32,
        'qwen35moe.block_count':c['num_hidden_layers'],'qwen35moe.embedding_length':c['hidden_size'],
        'qwen35moe.attention.head_count':c['num_attention_heads'],
        'qwen35moe.attention.head_count_kv':[0,0,0,c['num_key_value_heads']],
        'qwen35moe.attention.key_length':c['head_dim'],'qwen35moe.attention.value_length':c['head_dim'],
        'qwen35moe.attention.layer_norm_rms_epsilon':c['rms_norm_eps'],
        'qwen35moe.context_length':c['max_position_embeddings'],
        'qwen35moe.rope.dimension_count':4,'qwen35moe.rope.freq_base':10000.,
        'qwen35moe.ssm.conv_kernel':c['linear_conv_kernel_dim'],'qwen35moe.ssm.state_size':c['linear_key_head_dim'],
        'qwen35moe.ssm.group_count':kh,'qwen35moe.ssm.time_step_rank':vh,'qwen35moe.ssm.inner_size':vh*hd,
        'qwen35moe.ssm.v_head_reordered':reordered,'qwen35moe.expert_count':c['num_experts'],
        'qwen35moe.expert_used_count':c['num_experts_per_tok'],
        'qwen35moe.expert_feed_forward_length':c['moe_intermediate_size'],
        'qwen35moe.expert_shared_feed_forward_length':c['shared_expert_intermediate_size']}
    path=tmp_path/'tiny.gguf';write_gguf(path,m,tensors)
    opt=replace(opt,row_chunk=7,prefill_chunk=2,policy='adaptive',cache_bytes=128<<10)
    return path,ref,opt,m,tensors


def arr(value):return np.asarray(value.astype(mx.float32))


@pytest.mark.parametrize('reordered',[False,True])
@pytest.mark.parametrize('device',['cpu','gpu'])
def test_gguf_hybrid_logits_cached_decode_and_generation(tmp_path,reordered,device):
    if device=='gpu' and not mx.metal.is_available():pytest.skip('Metal device unavailable')
    path,ref,opt,_,_=export_fixture(tmp_path,reordered=reordered)
    opt=replace(opt,device=device)
    ids=torch.tensor([[1,8,12,5,21]])
    with torch.inference_mode():expected=ref(ids,use_cache=True)
    with NativeGGUFQwenModel(path,opt) as model:
        assert model.checkpoint.stats['tensor_bytes']==0
        actual=model.forward(ids.numpy())
        np.testing.assert_allclose(arr(actual.logits),expected.logits.numpy(),rtol=5e-4,atol=2e-5)
        with torch.inference_mode():next_ref=ref(torch.tensor([[4,7]]),past_key_values=expected.past_key_values,use_cache=True)
        follow=model.forward([[4,7]],past_key_values=actual.past_key_values)
        np.testing.assert_allclose(arr(follow.logits),next_ref.logits.numpy(),rtol=5e-4,atol=2e-5)
        tokens=ids.clone()
        with torch.inference_mode():
            for _ in range(4):tokens=torch.cat([tokens,ref(tokens,use_cache=False).logits[:,-1].argmax(-1)[:,None]],dim=1)
        np.testing.assert_array_equal(np.asarray(model.generate(ids.numpy(),max_new_tokens=4)),tokens.numpy())
        metrics=model.metrics();assert metrics['runtime']['tokens_generated']==4
        assert metrics['cache']['peak_bytes']<=opt.cache_bytes
        assert model.checkpoint.stats['peak_reserved_bytes']<=opt.io_bytes


def test_gguf_rollback_and_partial_expert_reads(tmp_path,monkeypatch):
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,cache_bytes=0)) as model:
        reads=[];read=model.checkpoint.read_rows
        def tracked(name,start,count,**kw):
            reads.append((name,start,count,kw.get('expert')));return read(name,start,count,**kw)
        monkeypatch.setattr(model.checkpoint,'read_rows',tracked)
        cache=model.forward([[1,2]]).past_key_values;snapshot=cache.snapshot()
        model.forward([[3,4]],past_key_values=cache);cache.restore(snapshot)
        actual=arr(model.forward([[8]],past_key_values=cache).logits)
        expected=arr(model.forward([[1,2,8]]).logits[:,-1:])
        np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=2e-5)
        experts=[r for r in reads if '_exps.' in r[0]]
        assert experts and all(expert is not None and count<=opt.row_chunk for _,_,count,expert in experts)
        assert all(0<=expert<model.config['num_experts'] for _,_,_,expert in experts)


def test_gguf_rejects_bad_layout_and_unsupported_scaling_before_weights(tmp_path):
    path,_,opt,metadata,tensors=export_fixture(tmp_path)
    metadata['qwen35moe.rope.scaling.type']='yarn';write_gguf(path,metadata,tensors)
    with pytest.raises(NotImplementedError,match='Scaled'):NativeGGUFQwenModel(path,opt)
    del metadata['qwen35moe.rope.scaling.type'];del tensors['blk.1.ssm_a'];write_gguf(path,metadata,tensors)
    with pytest.raises(ValueError,match='ssm_a'):NativeGGUFQwenModel(path,opt)


def test_gguf_guard_rejects_before_payload_read_and_restores_session(tmp_path,monkeypatch):
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,cache_bytes=0)) as model:
        cache=model.forward([[1]]).past_key_values
        old={k:arr(v).copy() for k,v in cache.banks['recurrent'].items()}
        calls=model.checkpoint.stats['tensor_bytes']
        monkeypatch.setattr(model.reader,'cfg',replace(opt,workspace_bytes=0))
        with pytest.raises(MemoryError):model.forward([[2]],past_key_values=cache)
        assert cache.position==1 and cache.retained==0
        assert model.checkpoint.stats['tensor_bytes']==calls
        for k,v in old.items():np.testing.assert_array_equal(arr(cache.banks['recurrent'][k]),v)


def test_v_head_order_has_exact_inverse_and_validates_dimensions():
    assert v_head_permutation(2,4,2).tolist()==[0,1,4,5,2,3,6,7]
    x=np.arange(12);p=v_head_permutation(2,6,2)
    np.testing.assert_array_equal(x[p][np.argsort(p)],x)
    with pytest.raises(ValueError):v_head_permutation(2,3)
    with pytest.raises(ValueError):v_head_permutation(True,4)


def test_gguf_interrupt_restores_all_banks_and_can_retry(tmp_path):
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,opt) as model:
        cache=model.forward([[1,2]]).past_key_values
        original=model.mlps[1]
        class Interrupted:
            def __call__(self,x):raise KeyboardInterrupt('injected interruption')
        model.mlps[1]=Interrupted()
        with pytest.raises(KeyboardInterrupt):model.forward([[4]],past_key_values=cache)
        assert cache.position==2 and cache.retained==0
        model.mlps[1]=original
        actual=arr(model.forward([[4]],past_key_values=cache).logits)
        expected=arr(model.forward([[1,2,4]]).logits[:,-1:])
        np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=2e-5)


@pytest.mark.parametrize('kind',[12,14])
def test_gguf_weight_adapter_metal_tiles_packed_cache_and_expert_selection(tmp_path,kind):
    if not mx.metal.is_available():pytest.skip('Metal device unavailable')
    from airllm.nextgen.config import NextConfig
    from test_gguf_metal import scalar_decode
    rng=np.random.default_rng(612);width=256;rows=5;block_bytes=144 if kind==12 else 210
    packed=rng.integers(0,256,(3,rows,block_bytes),dtype=np.uint8)
    for pos in ([0,2] if kind==12 else [208]):
        packed[:,:,pos:pos+2]=np.frombuffer(np.float16(.001).tobytes(),np.uint8)
    name='blk.0.ffn_gate_exps.weight';calls=[]
    class Checkpoint:
        tensors={name:SimpleNamespace(shape=(3,rows,width),quant_type=kind,row_bytes=block_bytes)}
        available_memory_bytes=2<<20
        def read_packed_rows(self,name,start,count,*,expert=None):
            calls.append((expert,start,count));return packed[expert,start:start+count].copy()
        def read_rows(self,*args,**kwargs):raise AssertionError('Fused Metal must not decode a float matrix')
    opt=NextConfig(cache_bytes=64<<10,io_bytes=2<<20,workspace_bytes=2<<20,kv_bytes=2<<20,
                   reserve_bytes=0,max_tensor_bytes=1<<20,row_chunk=2,policy='adaptive',device='gpu',
                   dtype='float32',load_tokenizer=False,prefetch=False)
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,'gguf_v_head_reordered':False}
    reader=GGUFWeights(Checkpoint(),mx,opt,config,kernel='metal')
    x=rng.normal(size=(1,width)).astype(np.float32)
    with mx.stream(mx.new_stream(mx.gpu)):
        actual=arr(reader.linear(mx.array(x),name,expert=1))
        again=arr(reader.linear(mx.array(x),name,expert=1))
    np.testing.assert_allclose(actual,x@scalar_decode(packed[1],kind,width).T,rtol=3e-5,atol=3e-4)
    np.testing.assert_array_equal(actual,again)
    assert calls==[(1,0,2),(1,2,2),(1,4,1)]
    assert reader.cache.bytes==packed[1].nbytes and reader.cache.stats['hits']==3
    reader.close();assert reader.cache.bytes==0


def test_v_reorder_noninvolutive_full_hybrid_logits(tmp_path):
    path,ref,opt,_,_=export_fixture(tmp_path,noninvolutive=True)
    p=v_head_permutation(4,8,8)
    assert not np.array_equal(p,np.argsort(p))
    with torch.inference_mode():expected=ref(torch.tensor([[1,8,12,5]]),use_cache=False).logits.numpy()
    with NativeGGUFQwenModel(path,opt) as model:
        actual=arr(model.forward([[1,8,12,5]]).logits)
    np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=2e-5)


def test_tied_output_does_not_admit_full_vocabulary_scan(tmp_path):
    path,_,opt,metadata,tensors=export_fixture(tmp_path)
    del tensors['output.weight'];write_gguf(path,metadata,tensors)
    with NativeGGUFQwenModel(path,opt) as model:
        x=mx.zeros((1,1,model.config['hidden_size']))
        model.reader.output(x,'model.embed_tokens.weight')
        assert model.reader.cache.bytes==0


def test_metal_allocator_does_not_double_grant_packed_cache(tmp_path):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    path,_,opt,_,_=export_fixture(tmp_path)
    with NativeGGUFQwenModel(path,replace(opt,device='gpu'),kernel='metal') as model:
        assert model.allocator_limit_bytes==opt.mlx_memory_bytes-opt.cache_bytes
