"""Exact expert dispatch, bounded workspace and rollback regression coverage."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx

from airllm.nextgen.gguf_metal import packed_matmul,packed_expert_matmul
from airllm.nextgen.gguf_qwen import GGUFWeights,GGUFQwenSparseMLP,NativeGGUFQwenModel
from airllm.nextgen.config import NextConfig


def packed_fixture(kind,experts,rows,width):
    rng=np.random.default_rng(471+kind+width)
    size=144 if kind==12 else 210
    packed=rng.integers(0,256,(experts,rows,width//256*size),dtype=np.uint8)
    for block in range(width//256):
        for pos in ([0,2] if kind==12 else [208]):
            packed[:,:,block*size+pos:block*size+pos+2]=np.frombuffer(np.float16(.001).tobytes(),np.uint8)
    return rng,packed


@pytest.mark.parametrize('kind',[12,14])
@pytest.mark.parametrize('batch,width',[(1,512),(3,2048)])
@pytest.mark.parametrize('simd_groups',[1,4,8])
def test_grouped_kernel_is_bitwise_original_simd_reduction(kind,batch,width,simd_groups):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    rng,packed=packed_fixture(kind,4,9,width)
    x=mx.array(rng.normal(size=(4,batch,width)).astype(np.float32))
    with mx.stream(mx.new_stream(mx.gpu)):
        actual=np.asarray(packed_expert_matmul(packed,x,kind,simd_groups=simd_groups))
        expected=np.stack([np.asarray(packed_matmul(packed[e],x[e],kind)) for e in range(4)])
    np.testing.assert_array_equal(actual,expected)


def test_grouped_kernel_rejects_shape_and_budget_before_allocation(monkeypatch):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    _,packed=packed_fixture(12,2,3,256);x=mx.zeros((2,1,256))
    monkeypatch.setattr(mx,'array',lambda *a,**kw:pytest.fail('No import allowed after budget refusal'))
    with pytest.raises(MemoryError):packed_expert_matmul(packed,x,12,max_workspace_bytes=1)
    with pytest.raises(ValueError):packed_expert_matmul(packed,x[:1],12)
    with pytest.raises(ValueError):packed_expert_matmul(packed,x,12,max_workspace_bytes=True)


@pytest.mark.parametrize('kind',[12,14])
def test_grouped_reader_tiles_selected_experts_preserves_order_and_cache(kind):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    rng,packed=packed_fixture(kind,6,7,512);name='blk.0.ffn_gate_exps.weight';calls=[]
    size=packed.shape[-1]
    class Checkpoint:
        tensors={name:SimpleNamespace(shape=(6,7,512),quant_type=kind,row_bytes=size)}
        available_memory_bytes=1<<20
        def read_packed_rows(self,name,start,count,*,expert=None):
            calls.append((expert,start,count));return packed[expert,start:start+count].copy()
    options=NextConfig(cache_bytes=64<<10,io_bytes=1<<20,workspace_bytes=1<<20,kv_bytes=1<<20,
        reserve_bytes=0,max_tensor_bytes=3*2*size,row_chunk=4,policy='adaptive',device='gpu',dtype='float32',prefetch=False)
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,'gguf_v_head_reordered':False}
    reader=GGUFWeights(Checkpoint(),mx,options,config,kernel='metal',expert_batch=True)
    experts=[4,1,5];x=mx.array(rng.normal(size=(3,1,512)).astype(np.float32))
    with mx.stream(mx.new_stream(mx.gpu)):
        result=np.asarray(reader.linear_experts(x,name,experts))
        again=np.asarray(reader.linear_experts(x,name,experts))
        expected=np.stack([np.asarray(packed_matmul(packed[e],x[i],kind)) for i,e in enumerate(experts)])
    np.testing.assert_array_equal(result,expected);np.testing.assert_array_equal(again,expected)
    assert {e for e,_,_ in calls}==set(experts)
    assert max(count for _,_,count in calls)<=2
    assert reader.cache.bytes<=options.cache_bytes
    before=len(calls);reader.cfg=replace(options,workspace_bytes=0)
    with pytest.raises(MemoryError):reader.linear_experts(x,name,experts)
    assert len(calls)==before
    for bad in ([1,1],[1,-1],[True,2]):
        with pytest.raises(ValueError):reader.linear_experts(x,name,bad)


def test_opt_in_float_fixture_stays_exact_and_rolls_back(tmp_path):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    from test_gguf_qwen import export_fixture
    path,_,opt,_,_=export_fixture(tmp_path)
    opt=replace(opt,device='gpu',dtype='float32')
    with NativeGGUFQwenModel(path,opt,kernel='metal') as baseline:
        first=baseline.forward([[1,4,8]],logits_to_keep=1)
        expected=np.asarray(baseline.forward([[2]],past_key_values=first.past_key_values).logits).copy()
    with NativeGGUFQwenModel(path,opt,kernel='metal',expert_batch=True) as candidate:
        cache=candidate.forward([[1,4,8]],logits_to_keep=1).past_key_values;snapshot=cache.snapshot()
        actual=np.asarray(candidate.forward([[2]],past_key_values=cache).logits).copy()
        cache.restore(snapshot)
        replay=np.asarray(candidate.forward([[2]],past_key_values=cache).logits).copy()
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(replay,expected)


def test_opt_in_requires_boolean_and_metal():
    opt=NextConfig(reserve_bytes=0)
    for flag in (1,'true'):
        with pytest.raises(ValueError,match='boolean'):GGUFWeights(None,mx,opt,{},expert_batch=flag)
    with pytest.raises(ValueError,match='requires'):GGUFWeights(None,mx,opt,{},expert_batch=True)
    with pytest.raises(ValueError,match='requires'):GGUFWeights(None,mx,opt,{},pipeline_tiles=True)


@pytest.mark.parametrize('kind',[12,14])
def test_two_tile_pipeline_is_exact_and_evaluates_before_next_submission(kind):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    rng,packed=packed_fixture(kind,1,19,512);name='test.weight'
    class Checkpoint:
        tensors={name:SimpleNamespace(shape=(19,512),quant_type=kind,row_bytes=packed.shape[-1])}
        available_memory_bytes=1<<20
        def read_packed_rows(self,name,start,count,*,expert=None):return packed[0,start:start+count].copy()
    cfg=NextConfig(cache_bytes=0,workspace_bytes=1<<20,reserve_bytes=0,max_tensor_bytes=1<<20,
        row_chunk=3,device='gpu',dtype='float32',prefetch=False)
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,'gguf_v_head_reordered':False}
    # resolve supports indexed tensor names directly.
    x=mx.array(rng.normal(size=(2,512)).astype(np.float32))
    with mx.stream(mx.new_stream(mx.gpu)):
        original=GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal')
        reader=GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal',concat_tiles=True,pipeline_tiles=True)
        expected=np.asarray(original.linear(x,name));actual=np.asarray(reader.linear(x,name))
    np.testing.assert_array_equal(actual,expected)
    assert reader.stats['tile_pipeline_wait_seconds']>=0
    assert reader.stats['peak_workspace_estimate_bytes']<=cfg.workspace_bytes


def test_zero_copy_packed_lifetime_and_portable_fallback(monkeypatch):
    import gc
    from collections import defaultdict
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    rng,packed=packed_fixture(12,1,5,512);packed=packed[0].copy()
    x=mx.array(rng.normal(size=(1,512)).astype(np.float32));stats=defaultdict(float)
    with mx.stream(mx.new_stream(mx.gpu)):
        expected=np.asarray(packed_matmul(packed,x,12)).copy()
        actual=packed_matmul(packed,x,12,synchronize=False,zero_copy=True,stats=stats)
        del packed;gc.collect()
        np.testing.assert_array_equal(np.asarray(actual),expected)
        _,raw=packed_fixture(12,1,5,512);raw=raw[0].copy()
        def unavailable(*a,**kw):raise BufferError('unsupported exporter')
        monkeypatch.setattr(mx,'from_dlpack',unavailable)
        fallback=np.asarray(packed_matmul(raw,x,12,zero_copy=True,stats=stats))
        normal=np.asarray(packed_matmul(raw,x,12))
        np.testing.assert_array_equal(fallback,normal)
    assert stats['packed_zero_copy_imports']==1 and stats['packed_zero_copy_fallbacks']==1


@pytest.mark.parametrize('flags',[['--pipeline-tiles'],['--expert-batch'],['--cache-policy','lru','--cache-decay','100']])
def test_cli_rejects_incompatible_performance_flags_before_model_read(tmp_path,flags):
    from airllm.nextgen.gguf_cli import main
    with pytest.raises(SystemExit) as error:
        main(['gguf-run',str(tmp_path/'missing.gguf'),'--report',str(tmp_path/'report.json'),*flags])
    assert error.value.code==2 and not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('tokens',[1,7])
@pytest.mark.parametrize('concat',[False,True])
@pytest.mark.parametrize('expert_pipeline',[False,True])
def test_ragged_moe_expert_batch_matches_original_routes_and_outputs(tokens,concat,expert_pipeline):
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    rng,gate=packed_fixture(12,6,256,256);_,up=packed_fixture(12,6,256,256);_,down=packed_fixture(14,6,256,256)
    packed={'blk.0.ffn_gate_exps.weight':gate,'blk.0.ffn_up_exps.weight':up,'blk.0.ffn_down_exps.weight':down}
    floating={'blk.0.ffn_gate_inp.weight':rng.normal(size=(6,256)).astype(np.float32)}
    for part,shape in [('gate',(256,256)),('up',(256,256)),('down',(256,256)),('gate_inp',(1,256))]:
        floating['blk.0.ffn_'+part+'_shexp.weight']=rng.normal(scale=.01,size=shape).astype(np.float32)
    class Checkpoint:
        tensors={n:SimpleNamespace(shape=(6,256,256),quant_type=14 if 'down_exps' in n else 12,row_bytes=v.shape[-1]) for n,v in packed.items()}
        tensors.update({n:SimpleNamespace(shape=v.shape,quant_type=0,row_bytes=v.shape[-1]*4) for n,v in floating.items()})
        available_memory_bytes=16<<20
        def read_packed_rows(self,name,start,count,*,expert=None):return packed[name][expert,start:start+count].copy()
        def read_rows(self,name,start,count,*,expert=None):return floating[name][start:start+count].copy()
        def max_rows(self,name):return self.tensors[name].shape[-2]
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,'gguf_v_head_reordered':False,
        'num_experts':6,'num_experts_per_tok':2,'moe_intermediate_size':256}
    cfg=NextConfig(cache_bytes=0,workspace_bytes=16<<20,io_bytes=16<<20,kv_bytes=1<<20,reserve_bytes=0,
        max_tensor_bytes=4<<20,row_chunk=256 if expert_pipeline else 128,device='gpu',dtype='float32',prefetch=False)
    x=rng.normal(size=(1,tokens,256)).astype(np.float32)
    with mx.stream(mx.new_stream(mx.gpu)):
        base=GGUFQwenSparseMLP(mx,GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal'),config,0)
        expected=np.asarray(base(mx.array(x))).copy()
        reader=GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal',expert_batch=True,
                           expert_pipeline=expert_pipeline,concat_tiles=concat)
        candidate=GGUFQwenSparseMLP(mx,reader,config,0);actual=np.asarray(candidate(mx.array(x)))
    np.testing.assert_array_equal(actual,expected);assert candidate.routes==base.routes
    assert reader.cfg is cfg
    if expert_pipeline:assert reader.stats['expert_pipeline_pairs']>0
