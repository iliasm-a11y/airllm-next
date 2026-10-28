"""CPU controls for canonical routing; no MLX/Torch or checkpoint is loaded."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
sys.path.insert(0,str(ROOT/'benchmarks'))
from airllm.nextgen import qwen_moe,gguf_qwen
from benchmark_research51 import router_reference_options
from campaign_research51 import validate_job_args
from hyperstream2_observer import RuntimeObserver


def route_fixture(monkeypatch):
    calls=[]
    def original(mx,x,weight,config,**options):
        flat=x.reshape(-1,x.shape[-1])
        tokens=flat[:,0].astype(np.int32)
        # Deliberately non-sorted expert IDs. Their score positions identify
        # both the token and rank, exposing accidental sorting or row mixing.
        ids=np.stack((tokens+100,tokens+1),axis=-1)
        scores=np.stack((tokens+.125,tokens+.875),axis=-1).astype(np.float32)
        result=(ids,scores)
        calls.append(dict(x=x,weight=weight,config=config,options=options,result=result))
        return result
    monkeypatch.setattr(qwen_moe,'qwen_route',original)
    monkeypatch.setattr(gguf_qwen,'_original_qwen_route',original)
    return calls


def inputs(rows):
    return np.arange(rows*2,dtype=np.float32).reshape(1,rows,2)


def test_disabled_path_is_one_unchanged_original_call(monkeypatch):
    calls=route_fixture(monkeypatch)
    x=inputs(35);weight=object();config={'num_experts_per_tok':2};stats={}
    result=qwen_moe.qwen_route_reference_chunks(None,x,weight,config,
        reference_chunk=0,stats=stats,workspace_bytes=1234,tie_policy='unchanged')
    assert len(calls)==1 and result is calls[0]['result']
    assert calls[0]['x'] is x and calls[0]['weight'] is weight and calls[0]['config'] is config
    assert calls[0]['options']==dict(stats=stats,workspace_bytes=1234,tie_policy='unchanged')
    assert stats=={}


@pytest.mark.parametrize('rows',[1,16,17,35,128])
def test_canonical_blocks_keep_token_positions_rank_order_and_weights(monkeypatch,rows):
    calls=route_fixture(monkeypatch)
    x=inputs(rows);weight=object();config={'num_experts_per_tok':2};stats={}
    mx=SimpleNamespace(concatenate=np.concatenate)
    ids,scores=qwen_moe.qwen_route_reference_chunks(mx,x,weight,config,
        reference_chunk=16,workspace_bytes=1<<20,stats=stats,reserve_bytes=0)
    expected=x.reshape(rows,2)[:,0].astype(np.int32)
    assert np.array_equal(ids,np.stack((expected+100,expected+1),axis=-1))
    assert np.array_equal(scores,np.stack((expected+.125,expected+.875),axis=-1))
    assert [call['x'].reshape(-1,2).shape[0] for call in calls]==[
        min(16,rows-begin) for begin in range(0,rows,16)]
    assert all(call['weight'] is weight and call['config'] is config for call in calls)
    if rows<=16:
        assert calls[0]['x'] is x
        assert calls[0]['options']['workspace_bytes']==1<<20
    else:
        retained=2*rows*2*(8+8)
        assert stats['router_reference16_retained_peak_bytes']==retained
        assert all(call['options']['workspace_bytes']==(1<<20)-retained for call in calls)
    assert stats['router_reference16_calls']==1
    assert stats['router_reference16_blocks']==(rows+15)//16
    assert stats['router_reference16_seconds']>=0


@pytest.mark.parametrize('bad',[True,False,-1,1,8,32,16.0])
def test_invalid_canonical_mode_is_rejected_before_original_call(monkeypatch,bad):
    calls=route_fixture(monkeypatch)
    with pytest.raises(ValueError,match='0 or 16'):
        qwen_moe.qwen_route_reference_chunks(None,inputs(35),None,{},reference_chunk=bad)
    assert calls==[]


@pytest.mark.parametrize('workspace',[None,0,-1,True])
def test_enabled_router_requires_existing_positive_workspace(monkeypatch,workspace):
    calls=route_fixture(monkeypatch)
    with pytest.raises(ValueError,match='positive workspace_bytes'):
        qwen_moe.qwen_route_reference_chunks(None,inputs(17),None,
            {'num_experts_per_tok':2},reference_chunk=16,workspace_bytes=workspace)
    assert calls==[]


def test_retained_outputs_cannot_borrow_a_new_workspace_allowance(monkeypatch):
    calls=route_fixture(monkeypatch)
    with pytest.raises(MemoryError,match='retention exceeds'):
        qwen_moe.qwen_route_reference_chunks(None,inputs(17),None,
            {'num_experts_per_tok':2},reference_chunk=16,workspace_bytes=1024)
    assert calls==[]


def test_system_reserve_is_checked_before_retention(monkeypatch):
    import psutil
    calls=route_fixture(monkeypatch)
    retained=2*17*2*(8+8)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1000+retained-1))
    with pytest.raises(MemoryError,match='System reserve'):
        qwen_moe.qwen_route_reference_chunks(None,inputs(17),None,
            {'num_experts_per_tok':2},reference_chunk=16,workspace_bytes=1<<20,reserve_bytes=1000)
    assert calls==[]


def test_observer_receives_one_complete_route_bank_with_absolute_positions(monkeypatch):
    calls=route_fixture(monkeypatch)
    observer=RuntimeObserver(SimpleNamespace())
    observer._state().update(mlp_layer=30,positions={30:128})
    route=observer._route_wrapper(gguf_qwen.qwen_route)
    mx=SimpleNamespace(concatenate=np.concatenate)
    ids,scores=route(mx,inputs(128),None,{'num_experts_per_tok':2},
        router_reference_chunk=16,workspace_bytes=1<<20,stats={})
    assert len(calls)==8
    assert len(observer.routing['30'])==128
    assert observer.routing['30']['154']==ids[26].tolist()
    assert observer.routing['30']['255']==ids[-1].tolist()
    assert scores[26].tolist()==[52.125,52.875]


@pytest.mark.parametrize('bad',[True,False,-1,8,128])
def test_constructor_rejects_invalid_mode_before_checkpoint_io(bad):
    with pytest.raises(ValueError,match='router_reference_chunk'):
        gguf_qwen.NativeGGUFQwenModel('/nonexistent/checkpoint.gguf',router_reference_chunk=bad)


def test_worker_keeps_new_keyword_away_from_historical_constructor():
    def historical(*,kernel):
        return kernel
    assert historical(**dict(kernel='metal',**router_reference_options(0)))=='metal'
    assert router_reference_options(16)=={'router_reference_chunk':16}
    with pytest.raises(TypeError,match='router_reference_chunk'):
        historical(**dict(kernel='metal',**router_reference_options(16)))


@pytest.mark.parametrize('value',['0','16'])
def test_campaign_whitelist_accepts_only_explicit_supported_modes(value):
    args=['--router-reference-chunk',value,'--numerical-evidence']
    assert validate_job_args(args)==args


@pytest.mark.parametrize('value',['1','8','32','-1','true'])
def test_campaign_whitelist_rejects_unsupported_modes(value):
    with pytest.raises(ValueError,match='Invalid comparison arguments'):
        validate_job_args(['--router-reference-chunk',value])
