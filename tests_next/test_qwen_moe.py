"""Independent official Torch parity for streamed Qwen sparse experts."""
import numpy as np
import pytest
import torch
import mlx.core as mx
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from airllm.nextgen.config import NextConfig
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights
from airllm.nextgen.qwen_moe import QwenSparseMLP,qwen_route


def fixture(tmp_path,*,split=False,bf16=False):
    torch.manual_seed(2061)
    cfg=Qwen3_5MoeTextConfig(hidden_size=16,intermediate_size=24,
        moe_intermediate_size=12,shared_expert_intermediate_size=20,
        num_experts=8,num_experts_per_tok=2,num_hidden_layers=1)
    model=Qwen3_5MoeSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():p.normal_(0,.12)
    if bf16:model=model.to(torch.bfloat16)
    state={f'model.layers.0.mlp.{k}':v.contiguous() for k,v in model.state_dict().items()}
    if split:
        gu=state.pop('model.layers.0.mlp.experts.gate_up_proj')
        down=state.pop('model.layers.0.mlp.experts.down_proj')
        for e in range(cfg.num_experts):
            gate,up=gu[e].chunk(2,dim=0)
            for name,value in [('gate_proj',gate),('up_proj',up),('down_proj',down[e])]:
                state[f'model.layers.0.mlp.experts.{e}.{name}.weight']=value.contiguous()
    root=tmp_path/'qwen';root.mkdir();save_file(state,str(root/'model.safetensors'))
    options=NextConfig(cache_bytes=0,io_bytes=1024**2,max_tensor_bytes=1024**2,
        dtype='bfloat16' if bf16 else 'float32',device='cpu',prefetch=False,load_tokenizer=False,
        cache_dir=str(tmp_path/'metadata'),workspace_bytes=1024**2,reserve_bytes=0)
    return cfg,model,root,options


@pytest.mark.parametrize('split',[False,True])
def test_qwen_streamed_moe_matches_official(tmp_path,split):
    cfg,ref,root,opt=fixture(tmp_path,split=split)
    cp=OffsetCheckpoint(root,opt.metadata_dir());reader=NativeWeights(cp,mx,opt,cfg.to_dict())
    block=QwenSparseMLP(mx,reader,cfg.to_dict(),0)
    torch.manual_seed(48);x=torch.randn(2,5,16)
    try:
        with torch.inference_mode():expected=ref(x).numpy()
        actual=np.array(block(mx.array(x.numpy())))
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-7)
        with torch.inference_mode():_,scores,ids=ref.gate(x)
        mi,mw=qwen_route(mx,mx.array(x.numpy()),reader.get('model.layers.0.mlp.gate.weight'),cfg.to_dict())
        np.testing.assert_array_equal(np.array(mi),ids.numpy())
        np.testing.assert_allclose(np.array(mw),scores.numpy(),rtol=2e-6,atol=2e-7)
        assert reader.cache.peak_bytes==0
        assert reader.io.peak_bytes<=opt.io_bytes
    finally:reader.close();cp.close()


def test_qwen_only_selected_experts_are_read(tmp_path):
    cfg,ref,root,opt=fixture(tmp_path)
    cp=OffsetCheckpoint(root,opt.metadata_dir());reader=NativeWeights(cp,mx,opt,cfg.to_dict())
    block=QwenSparseMLP(mx,reader,cfg.to_dict(),0)
    observed=[];original=cp.read_array
    def track(key,**kwargs):
        if key.startswith('model.layers.0.mlp.experts.'):observed.append((key,kwargs.get('expert')))
        return original(key,**kwargs)
    cp.read_array=track
    try:
        x=torch.ones(1,1,16)
        with torch.inference_mode():_,_,ids=ref.gate(x)
        block(mx.array(x.numpy()))
        selected=set(ids.numpy().reshape(-1).tolist())
        assert {e for _,e in observed}==selected
        assert len(observed)==len(selected)*2
        assert len(selected)==2
    finally:reader.close();cp.close()


def test_router_tie_rejected_instead_of_different_expert_choice():
    with pytest.raises(ValueError,match='boundary tie'):
        qwen_route(mx,mx.ones((1,16)),mx.zeros((8,16)),{'num_experts':8,'num_experts_per_tok':2})


def test_router_invalid_count():
    with pytest.raises(ValueError,match='expert count'):
        qwen_route(mx,mx.ones((1,16)),mx.zeros((8,16)),{'num_experts':8,'num_experts_per_tok':9})


def test_qwen_bfloat16_checkpoint_direct_read(tmp_path):
    cfg,ref,root,opt=fixture(tmp_path,bf16=True)
    cp=OffsetCheckpoint(root,opt.metadata_dir());reader=NativeWeights(cp,mx,opt,cfg.to_dict())
    block=QwenSparseMLP(mx,reader,cfg.to_dict(),0)
    torch.manual_seed(48);x=torch.randn(1,4,16).to(torch.bfloat16)
    try:
        with torch.inference_mode():expected=ref(x).float().numpy()
        a=mx.array(x.float().numpy()).astype(mx.bfloat16)
        actual=np.array(block(a).astype(mx.float32))
        # Distinct BF16 matmul/rounding paths have finite quantization error;
        # this fixture's tolerance is not a full-model quality claim.
        np.testing.assert_allclose(actual,expected,rtol=.06,atol=.0015)
        assert cp.tensors['model.layers.0.mlp.experts.gate_up_proj']['dtype']=='BF16'
    finally:reader.close();cp.close()
