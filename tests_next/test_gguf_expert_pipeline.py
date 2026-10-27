"""CPU-only scheduling/budget tests for the optional grouped gate/up pipeline.

These mocks test submission order and ownership, not Metal numerical fidelity.
The real kernel and ragged MoE parity cases live in test_gguf_throughput.py.
"""
from dataclasses import replace
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.gguf_qwen import GGUFWeights


GATE='blk.0.ffn_gate_exps.weight'
UP='blk.0.ffn_up_exps.weight'


def setup_reader(monkeypatch, *, pipeline=True, rows=5, row_chunk=5, workspace=1<<20):
    events=[]
    class FakeMX:
        float32=np.float32
        metal=SimpleNamespace(is_available=lambda:True)
        def zeros(self,shape,dtype):return np.zeros(shape,dtype=dtype)
        def eval(self,*values):events.append(('eval',))
        def async_eval(self,*values):events.append(('async',))
        def synchronize(self):events.append(('drain',))
    class Checkpoint:
        tensors={name:SimpleNamespace(shape=(3,rows,256),quant_type=12,row_bytes=144) for name in (GATE,UP)}
        available_memory_bytes=1<<20
        fail_up=False
        def read_packed_rows(self,name,start,count,*,expert=None):
            events.append(('read',name,expert,start,count))
            if self.fail_up and name==UP:raise OSError('up read failed')
            return np.full((count,144),(expert+1)*(2 if name==UP else 1),dtype=np.uint8)
    def kernel(packed,x,kind,*,synchronize=True,**kwargs):
        events.append(('kernel',synchronize))
        # A deterministic independent-matrix mock; real packed reduction is
        # validated separately with the actual Metal kernel.
        return (np.sum(packed,axis=-1,dtype=np.float32)[:,None,:]+
                np.sum(x,axis=-1,dtype=np.float32)[:,:,None]).astype(np.float32)
    monkeypatch.setattr('airllm.nextgen.gguf_metal.packed_expert_matmul',kernel)
    monkeypatch.setattr('airllm.nextgen.experiment_guard.pressure_level',lambda:1)
    monkeypatch.setattr('airllm.nextgen.gguf_qwen.psutil.virtual_memory',lambda:SimpleNamespace(available=1<<40))
    cfg=NextConfig(cache_bytes=0,io_bytes=1<<20,workspace_bytes=workspace,kv_bytes=1<<20,
        reserve_bytes=0,max_tensor_bytes=1<<20,row_chunk=row_chunk,policy='lru',
        device='gpu',dtype='float32',prefetch=False)
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,
            'gguf_v_head_reordered':False}
    reader=GGUFWeights(Checkpoint(),FakeMX(),cfg,config,kernel='metal',
                       expert_batch=True,expert_pipeline=pipeline)
    x=np.arange(2*256,dtype=np.float32).reshape(2,1,256)
    return reader,x,events


def test_pipeline_prepares_up_after_async_gate_before_wait_and_keeps_order(monkeypatch):
    reader,x,events=setup_reader(monkeypatch)
    gate,up=reader.linear_expert_pair(x,GATE,UP,[2,0])
    captured=list(events)
    reader.expert_pipeline=False
    expected=reader.linear_expert_pair(x,GATE,UP,[2,0])
    np.testing.assert_array_equal(gate,expected[0]);np.testing.assert_array_equal(up,expected[1])
    first=captured.index(('kernel',False));submitted=captured.index(('async',))
    up_read=next(i for i,event in enumerate(captured) if event[:2]==('read',UP))
    wait=captured.index(('eval',));second=captured.index(('kernel',True))
    assert first<submitted<up_read<wait<second
    assert [(event[1],event[2]) for event in captured if event[0]=='read']==[(GATE,2),(GATE,0),(UP,2),(UP,0)]
    assert reader.stats['expert_pipeline_pairs']==1
    assert reader.stats['expert_pipeline_peak_workspace_bytes']<=reader.cfg.workspace_bytes
    assert reader.stats['expert_pipeline_wait_seconds']>=0


@pytest.mark.parametrize('reason,settings',[
    ('multi_tile',{'row_chunk':2}),('workspace',{'workspace':9000}),
    ('pressure',{}),('reserve',{}),
])
def test_pipeline_falls_back_before_async_without_lowering_sequential_guards(monkeypatch,reason,settings):
    reader,x,events=setup_reader(monkeypatch,**settings)
    if reason=='pressure':
        monkeypatch.setattr('airllm.nextgen.experiment_guard.pressure_level',lambda:2)
    if reason=='reserve':
        # Enough for the ordinary sequential path, too little for the pair.
        monkeypatch.setattr('airllm.nextgen.gguf_qwen.psutil.virtual_memory',lambda:SimpleNamespace(available=10000))
    gate,up=reader.linear_expert_pair(x,GATE,UP,[2,0])
    assert gate.shape==up.shape==(2,1,5)
    assert reader.stats['expert_pipeline_fallback_'+reason]==1
    assert ('async',) not in events
    assert all(event[1] for event in events if event[0]=='kernel')
    reader.cfg=replace(reader.cfg,workspace_bytes=1)
    before=sum(event[0]=='read' for event in events)
    with pytest.raises(MemoryError):reader.linear_expert_pair(x,GATE,UP,[2,0])
    assert sum(event[0]=='read' for event in events)==before


def test_up_read_error_drains_before_releasing_group_owners_and_preserves_error(monkeypatch):
    reader,x,events=setup_reader(monkeypatch)
    reader.cp.fail_up=True
    allocated=[];original=np.empty
    def allocation(*args,**kwargs):
        value=original(*args,**kwargs)
        if value.ndim==3:allocated.append(weakref.ref(value))
        return value
    monkeypatch.setattr('airllm.nextgen.gguf_qwen.np.empty',allocation)
    original_drain=reader.mx.synchronize
    def drain():
        assert allocated[0]() is not None
        original_drain()
    reader.mx.synchronize=drain
    with pytest.raises(OSError,match='up read failed') as caught:
        reader.linear_expert_pair(x,GATE,UP,[2,0])
    assert ('async',) in events and events[-1]==('drain',)
    assert reader.stats['expert_pipeline_error_drains']==1
    # Keep the exception traceback alive: scratch owners must still be freed.
    assert caught.value is not None and all(reference() is None for reference in allocated)


def test_drain_failure_does_not_replace_original_read_error(monkeypatch):
    reader,x,events=setup_reader(monkeypatch);reader.cp.fail_up=True
    def failed_drain():raise RuntimeError('device drain failed')
    reader.mx.synchronize=failed_drain
    with pytest.raises(OSError,match='up read failed') as caught:
        reader.linear_expert_pair(x,GATE,UP,[2,0])
    assert any('device drain failed' in note for note in caught.value.__notes__)


@pytest.mark.parametrize('flag',[1,'yes',None])
def test_expert_pipeline_requires_actual_boolean(monkeypatch,flag):
    reader,_,_=setup_reader(monkeypatch)
    with pytest.raises(ValueError,match='expert_pipeline must be boolean'):
        GGUFWeights(reader.cp,reader.mx,reader.cfg,reader.model_config,kernel='metal',
                    expert_batch=True,expert_pipeline=flag)


def test_expert_pipeline_requires_grouped_metal(monkeypatch):
    reader,_,_=setup_reader(monkeypatch)
    with pytest.raises(ValueError,match='expert_pipeline requires'):
        GGUFWeights(reader.cp,reader.mx,reader.cfg,reader.model_config,kernel='metal',expert_pipeline=True)


@pytest.mark.parametrize('kind',[12,14])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('zero_copy',[False,True])
def test_single_tile_pair_real_metal_matches_sequential(kind,batch,zero_copy):
    import mlx.core as mx
    if not mx.metal.is_available():pytest.skip('Metal unavailable')
    from test_gguf_throughput import packed_fixture
    rng,gate=packed_fixture(kind,3,7,512)
    _,up=packed_fixture(kind,3,7,512);up=up[:,::-1].copy()
    data={GATE:gate,UP:up}
    class Checkpoint:
        tensors={name:SimpleNamespace(shape=(3,7,512),quant_type=kind,row_bytes=value.shape[-1])
                 for name,value in data.items()}
        available_memory_bytes=1<<20
        def read_packed_rows(self,name,start,count,*,expert=None):
            return data[name][expert,start:start+count].copy()
    cfg=NextConfig(cache_bytes=0,io_bytes=1<<20,workspace_bytes=1<<20,kv_bytes=1<<20,
        reserve_bytes=0,max_tensor_bytes=1<<20,row_chunk=7,policy='lru',device='gpu',dtype='float32',
        prefetch=False,zero_copy_weights=zero_copy)
    config={'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_value_head_dim':8,
            'gguf_v_head_reordered':False}
    x=mx.array(rng.normal(size=(2,batch,512)).astype(np.float32));experts=[2,0]
    with mx.stream(mx.new_stream(mx.gpu)):
        baseline=GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal',expert_batch=True)
        expected=baseline.linear_expert_pair(x,GATE,UP,experts)
        expected=tuple(np.asarray(value).copy() for value in expected)
        reader=GGUFWeights(Checkpoint(),mx,cfg,config,kernel='metal',expert_batch=True,expert_pipeline=True)
        actual=reader.linear_expert_pair(x,GATE,UP,experts)
        for value,reference in zip(actual,expected):np.testing.assert_array_equal(np.asarray(value),reference)
    assert reader.stats['expert_pipeline_pairs']==1
