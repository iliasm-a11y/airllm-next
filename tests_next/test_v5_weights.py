"""Exact source residency and output tiles share one explicit weight budget."""
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file as save_torch

from airllm.nextgen.compact_cache import CompactWeightCache, backing_bytes
from airllm.nextgen.config import NextConfig
from airllm.nextgen.quantization import FP8_LUT
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights


def test_compact_accounts_full_shared_backing_without_copy():
    payload=bytearray(128)
    a=np.frombuffer(payload,dtype=np.uint8,count=8,offset=3)
    b=np.frombuffer(payload,dtype=np.uint8,count=16,offset=32)
    assert backing_bytes((a,b))==128
    rejected=CompactWeightCache(127)
    assert not rejected.put('view',(a,b)) and rejected.bytes==0
    assert a.flags.writeable  # rejected admission must not mutate the caller
    cache=CompactWeightCache(128)
    assert cache.put('view',(a,b)) and cache.bytes==128
    retained=cache.get('view')
    assert retained[0] is a and retained[1] is b
    assert not a.flags.writeable and not b.flags.writeable
    with pytest.raises(ValueError):a[0]=7


def test_compact_lru_pressure_and_close_release_references():
    cache=CompactWeightCache(12)
    assert cache.put('a',(np.ones(6,np.uint8),))
    assert cache.put('b',(np.ones(6,np.uint8),))
    assert cache.get('a') is not None
    assert cache.put('c',(np.ones(6,np.uint8),))
    assert cache.get('b') is None and cache.bytes==12
    cache.resize(6)
    assert list(cache.data)==['c'] and cache.bytes==6 and cache.peak_bytes==12
    cache.clear()
    assert not cache.data and not cache.sizes and cache.bytes==0


def test_compact_metadata_entry_limit_bounds_tiny_selection_scans():
    cache=CompactWeightCache(100000,max_entries=2)
    for key in ('a','b','c'):
        assert cache.put(key,(np.ones(1,np.uint8),))
    assert list(cache.data)==['b','c'] and cache.bytes==2
    assert cache.stats['evictions']==1
    with pytest.raises(ValueError,match='entry limit'):CompactWeightCache(100,max_entries=0)


@pytest.mark.parametrize('bad',[True,-1,1.5])
def test_compact_rejects_invalid_capacity(bad):
    with pytest.raises(ValueError):CompactWeightCache(bad)


def _fp8_checkpoint(root):
    root.mkdir()
    # Stored FP8 is deliberately generated from distinct finite values; scales
    # differ by expert/block so retaining payload without its scales would fail.
    raw=(torch.arange(3*32*32).reshape(3,32,32)%13-6).float().to(torch.float8_e4m3fn)
    scales=torch.arange(1,13).reshape(3,2,2).float()/8
    save_torch({'bank':raw,'bank_scale_inv':scales},root/'model.safetensors')
    return raw.view(torch.uint8).numpy(),scales.numpy()


def _options(**updates):
    return NextConfig(dtype='float32',cache_bytes=4000,compact_cache_bytes=4000,
        io_bytes=8192,max_tensor_bytes=8192,prefetch=False,coalesce_reads=True,
        load_tokenizer=False,**updates)


def test_compact_fp8_reuses_selected_experts_and_scales_exactly(tmp_path):
    root=tmp_path/'cp';raw,scales=_fp8_checkpoint(root)
    cfg=_options()
    with OffsetCheckpoint(root,max_read_bytes=8192) as cp:
        reader=NativeWeights(cp,mx,cfg,{'quantization_config':{'weight_block_size':[16,16]}})
        try:
            first=[]
            for expert in (2,0,1):
                actual=np.asarray(reader.get('bank',expert))
                expected=FP8_LUT[raw[expert]]*np.repeat(np.repeat(scales[expert],16,0),16,1)
                np.testing.assert_array_equal(actual,expected)
                first.append(actual)
            calls=cp.stats['read_calls'];nbytes=cp.stats['logical_bytes']
            for index,expert in enumerate((2,0,1)):
                np.testing.assert_array_equal(np.asarray(reader.get('bank',expert)),first[index])
            assert cp.stats['read_calls']==calls and cp.stats['logical_bytes']==nbytes
            assert reader.compact_cache.stats['hits']==3
            assert reader.compact_cache.stats['source_bytes_reused']==3*(1024+16)
            assert reader.cache.bytes==0
            assert reader.compact_cache.bytes<=cfg.cache_bytes
            reader.trim()
            assert reader.compact_cache.bytes<=cfg.cache_bytes//2
            assert reader.cache.capacity+reader.compact_cache.capacity<=cfg.cache_bytes
        finally:reader.close()
        assert reader.compact_cache.bytes==reader.cache.bytes==0


def test_compact_fp8_read_failure_never_caches_partial_selection(tmp_path):
    root=tmp_path/'cp';_fp8_checkpoint(root)
    with OffsetCheckpoint(root,max_read_bytes=8192) as cp:
        reader=NativeWeights(cp,mx,_options(),{'quantization_config':{'weight_block_size':[16,16]}})
        original=cp.read_arrays
        try:
            cp.read_arrays=lambda *args,**kwargs:(_ for _ in ()).throw(EOFError('interrupted'))
            with pytest.raises(EOFError,match='interrupted'):reader.get('bank',0)
            assert reader.compact_cache.bytes==0 and not reader.io.pending
            cp.read_arrays=original
            assert np.isfinite(np.asarray(reader.get('bank',0))).all()
        finally:reader.close()


def test_compact_rows_do_not_alias_different_fp8_selections(tmp_path):
    root=tmp_path/'cp';raw,scales=_fp8_checkpoint(root)
    with OffsetCheckpoint(root,max_read_bytes=8192) as cp:
        reader=NativeWeights(cp,mx,_options(),{'quantization_config':{'weight_block_size':[16,16]}})
        try:
            a=np.asarray(reader._load('bank',expert=1,rows=(7,19)))
            b=np.asarray(reader._load('bank',expert=1,rows=(8,20)))
            count=cp.stats['read_calls']
            expected=FP8_LUT[raw[1]]*np.repeat(np.repeat(scales[1],16,0),16,1)
            np.testing.assert_array_equal(a,expected[7:19])
            np.testing.assert_array_equal(b,expected[8:20])
            np.testing.assert_array_equal(np.asarray(reader._load('bank',expert=1,rows=(7,19))),a)
            assert cp.stats['read_calls']==count
        finally:reader.close()


def test_compact_hits_do_not_bypass_expert_and_row_selector_validation(tmp_path):
    root=tmp_path/'cp';_fp8_checkpoint(root)
    with OffsetCheckpoint(root,max_read_bytes=8192) as cp:
        reader=NativeWeights(cp,mx,_options(),{'quantization_config':{'weight_block_size':[16,16]}})
        try:
            reader.get('bank',1)
            reader._load('bank',expert=1,rows=(0,16))
            for expert in (True,1.0,-1,3):
                with pytest.raises(ValueError,match='expert'):reader.get('bank',expert)
            for rows in ((False,16),(0.,16),(0,33)):
                with pytest.raises(ValueError,match='row range'):reader._load('bank',expert=1,rows=rows)
        finally:reader.close()


def test_compact_bfloat16_retains_original_bits(tmp_path):
    root=tmp_path/'cp';root.mkdir()
    values=torch.linspace(-3,3,32*32).reshape(32,32).to(torch.bfloat16)
    save_torch({'w':values},root/'model.safetensors')
    cfg=NextConfig(dtype='float32',cache_bytes=2048,compact_cache_bytes=2048,
        io_bytes=4096,max_tensor_bytes=4096,load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=4096) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            a=np.asarray(reader.get('w'));count=cp.stats['read_calls']
            b=np.asarray(reader.get('w'))
            np.testing.assert_array_equal(a,values.float().numpy())
            np.testing.assert_array_equal(a,b)
            assert cp.stats['read_calls']==count and reader.compact_cache.bytes==2048
        finally:reader.close()


def test_output_tile_residence_uses_existing_budget_and_preserves_logits(tmp_path):
    root=tmp_path/'cp';root.mkdir()
    values=np.arange(33*64,dtype='f4').reshape(33,64)/256
    save_numpy({'w':values},root/'model.safetensors')
    cfg=NextConfig(dtype='float32',cache_bytes=10000,io_bytes=768,
        max_tensor_bytes=1024,row_chunk=3,output_tile_cache=True,load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=1024) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            x=mx.ones((2,64),dtype=mx.float32)
            a=np.asarray(reader.output(x,'w'));count=cp.stats['read_calls']
            b=np.asarray(reader.output(x,'w'))
            np.testing.assert_array_equal(a,np.ones((2,64),np.float32)@values.T)
            np.testing.assert_array_equal(a,b)
            assert cp.stats['read_calls']==count
            assert reader.stats['output_tile_cache_hits']==11
            assert reader.cache.bytes==values.nbytes<=cfg.cache_bytes
        finally:reader.close()


def test_zero_reuse_compact_trace_respects_budget_without_benefit(tmp_path):
    root=tmp_path/'cp';_fp8_checkpoint(root)
    cfg=replace(_options(),cache_bytes=1040,compact_cache_bytes=1040)
    with OffsetCheckpoint(root,max_read_bytes=8192) as cp:
        reader=NativeWeights(cp,mx,cfg,{'quantization_config':{'weight_block_size':[16,16]}})
        try:
            for expert in (0,1,2,0,1,2):reader.get('bank',expert)
            assert reader.compact_cache.stats['hits']==0
            assert reader.compact_cache.peak_bytes==1040
            assert reader.compact_cache.stats['evictions']==5
            assert cp.stats['logical_bytes']==6*1040
        finally:reader.close()


def test_full_fp8_glm_compact_and_decoded_routes_logits_tokens_match(tmp_path):
    from test_fp8_model import save_glm64
    from test_native import options
    from airllm.nextgen.runtime import AirLLMNextModel
    root=tmp_path/'fp8';save_glm64(root,fp8=True)
    cfg=options(tmp_path,cache_bytes=256*1024,prefetch=False)
    ids=[[1,8,9,7,11]]
    with AirLLMNextModel(root,next_config=cfg) as control:
        logits=np.asarray(control(ids).logits)
        tokens=control.generate(ids,max_new_tokens=8,eos_token_id=[]).tolist()
    with AirLLMNextModel(root,next_config=replace(cfg,compact_cache_bytes=cfg.cache_bytes)) as compact:
        np.testing.assert_array_equal(np.asarray(compact(ids).logits),logits)
        assert compact.generate(ids,max_new_tokens=8,eos_token_id=[]).tolist()==tokens
        assert compact.reader.cache.bytes+compact.reader.compact_cache.bytes<=cfg.cache_bytes
