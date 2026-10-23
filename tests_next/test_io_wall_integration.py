"""Actual native inference and bounded pipeline integration, without downloads."""
from dataclasses import replace
import os

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from airllm.nextgen.config import NextConfig
from airllm.nextgen.runtime import AirLLMNextModel
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights


@pytest.mark.parametrize('changes',[
    {'policy':'adaptive','prefetch':False},
    {'spec_prefetch':True},
    {'policy':'adaptive','spec_prefetch':True,'io_mode':'mmap'},
    {'policy':'adaptive','spec_prefetch':True,'coalesce_reads':True},
])
def test_actual_glm_routes_logits_and_tokens_unchanged(tiny_glm,tmp_path,changes):
    root,reference=tiny_glm
    cfg=NextConfig(cache_bytes=32768,io_bytes=262144,max_tensor_bytes=262144,
        workspace_bytes=1024**2,kv_bytes=1024**2,reserve_bytes=0,dtype='float32',
        device=os.environ.get('AIRLLM_TEST_DEVICE','cpu'),prefill_chunk=2,max_context=128,load_tokenizer=False,
        use_saved_tuning=False,cache_dir=str(tmp_path/'meta'),profile_io=True,**changes)
    inputs=torch.tensor([[1,8,12]])
    with AirLLMNextModel(root,next_config=cfg) as model,torch.inference_mode():
        for _ in range(2):
            actual=np.asarray(model(inputs).logits)
            np.testing.assert_allclose(actual,reference(inputs).logits.numpy(),atol=4e-6,rtol=3e-4)
        output=np.asarray(model.generate(inputs,max_new_tokens=8,eos_token_id=[]))
        expected=reference.generate(inputs,attention_mask=torch.ones_like(inputs),
            max_new_tokens=8,eos_token_id=None,pad_token_id=0,do_sample=False).numpy()
        np.testing.assert_array_equal(output,expected)
        metrics=model.metrics()
        assert metrics['io_profile']['read_bytes']>0
        assert model.reader.cache.peak_bytes<=cfg.cache_bytes
        assert model.reader.io.peak_bytes<=cfg.io_bytes
        if cfg.spec_prefetch:
            assert model.reader.spec.stats['observations']>0
            assert model.reader.spec.peak_bytes<=cfg.spec_prefetch_budget


@pytest.mark.parametrize('cache_tiles',[False,True])
@pytest.mark.parametrize('io_mode',['pread','mmap'])
def test_output_pipeline_same_values_with_two_tile_budget(tmp_path,cache_tiles,io_mode):
    root=tmp_path/'model';root.mkdir()
    value=np.arange(37*64,dtype=np.float32).reshape(37,64)/256
    save_file({'w':value},root/'model.safetensors')
    cfg=NextConfig(cache_bytes=4096,io_bytes=2048,max_tensor_bytes=2048,
        output_prefetch=True,output_tile_cache=cache_tiles,io_mode=io_mode,
        row_chunk=3,dtype='float32',load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=2048) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            x=np.arange(128,dtype=np.float32).reshape(2,64)/64
            for _ in range(2):
                actual=np.asarray(reader.output(mx.array(x),'w'))
                np.testing.assert_array_equal(actual,x@value.T)
            assert reader.stats['output_pipeline_submitted']>0
            assert reader.stats['output_pipeline_peak_payload_bytes']<=cfg.io_bytes
            assert reader.io.bytes==0 and not reader.io.pending
            assert reader.cache.peak_bytes<=cfg.cache_bytes
            if cache_tiles:assert any(isinstance(key,tuple) and key[0]=='output' for key in reader.cache.costs)
        finally:reader.close()


def test_output_pipeline_failure_releases_reservations_and_retry(tmp_path):
    root=tmp_path/'model';root.mkdir()
    save_file({'w':np.ones((9,16),np.float32)},root/'model.safetensors')
    cfg=NextConfig(cache_bytes=0,io_bytes=256,max_tensor_bytes=256,
        output_prefetch=True,row_chunk=2,dtype='float32',load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=256) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        original=reader._load
        def fail_second(key,expert=None,rows=None):
            if rows and rows[0]==2:raise OSError('injected exact tile read failure')
            return original(key,expert,rows)
        try:
            reader._load=fail_second
            with pytest.raises(OSError,match='injected'):
                reader.output(mx.ones((1,16)),'w')
            assert reader.io.bytes==0 and not reader.io.pending
            reader._load=original
            np.testing.assert_array_equal(np.asarray(reader.output(mx.ones((1,16)),'w')),
                                          np.full((1,9),16,np.float32))
        finally:reader.close()


@pytest.mark.parametrize('field,value',[
    ('spec_prefetch',1),('profile_io',1),('output_prefetch',1),('io_mode','direct'),
    ('spec_prefetch_bytes',-1),('spec_prefetch_experts',0),('spec_prefetch_experts',65),
])
def test_invalid_io_wall_settings_rejected(field,value):
    with pytest.raises(ValueError):NextConfig(**{field:value})


def test_speculative_budget_is_partition_not_extra_memory():
    cfg=NextConfig(io_bytes=1024,spec_prefetch=True)
    assert cfg.spec_prefetch_budget==256
    assert cfg.mlx_memory_bytes==replace(cfg,spec_prefetch=False).mlx_memory_bytes
    with pytest.raises(ValueError,match='inside io_bytes'):
        replace(cfg,spec_prefetch_bytes=1025)


def test_lossless_profile_explicitly_excludes_packed_transport(tmp_path):
    from airllm.nextgen.lossless import LosslessOverlay,pack_checkpoint
    root=tmp_path/'model';root.mkdir()
    save_file({'w':np.ones((8,4),np.float32)},root/'model.safetensors')
    with OffsetCheckpoint(root,profile_io=True) as source:
        packed=tmp_path/'packed'
        pack_checkpoint(source,packed,codec='raw',block_bytes=128)
        source.enable_profiler()
        overlay=LosslessOverlay(source,packed)
        try:
            np.testing.assert_array_equal(overlay.read_array('w'),np.ones((8,4),np.float32))
            profile=overlay.io_profile()
            assert profile['coverage']=='original_shard_fallback_only'
            assert profile['packed_reads_profiled'] is False
            assert 'read_bytes' not in profile
            assert profile['original_shard_fallback']['read_bytes']==0
            assert profile['packed_transport_counters']['payload_bytes']>0
        finally:overlay.close()


def test_output_pipeline_preserves_compute_error_when_pending_read_also_fails(tmp_path):
    from threading import Event
    ready=Event()
    root=tmp_path/'model';root.mkdir()
    save_file({'w':np.ones((9,16),np.float32)},root/'model.safetensors')
    cfg=NextConfig(cache_bytes=0,io_bytes=256,max_tensor_bytes=256,
        output_prefetch=True,row_chunk=2,dtype='float32',load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=256) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        original=reader._load
        def failed_next(key,expert=None,rows=None):
            if rows and rows[0]==2:
                ready.set()
                raise OSError('pending failure')
            return original(key,expert,rows)
        def failed_compute(x,w):
            assert ready.wait(2)
            raise ValueError('primary compute failure')
        try:
            reader._load=failed_next
            reader.linear_weight=failed_compute
            with pytest.raises(ValueError,match='primary compute'):
                reader.output(mx.ones((1,16)),'w')
            assert not reader.io.pending and reader.io.bytes==0
            assert reader.stats['output_pipeline_cleanup_failures']==1
        finally:reader.close()


def test_output_pipeline_affine_quantized_rows_preserve_reference(tmp_path):
    from airllm.nextgen.quantization import dequant_affine
    from airllm.nextgen.sidecar import QuantizedSidecar
    root=tmp_path/'model';root.mkdir()
    value=np.arange(33*64,dtype=np.float32).reshape(33,64)/256
    save_file({'w':value},root/'model.safetensors')
    sidecar=QuantizedSidecar(tmp_path/'q','source',group_size=32,create=True)
    sidecar.add('w',[value],value.shape)
    expected=dequant_affine(sidecar.load('w'))
    cfg=NextConfig(cache_bytes=0,io_bytes=200,max_tensor_bytes=240,
        output_prefetch=True,row_chunk=2048,dtype='float32',load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=240) as cp:
        reader=NativeWeights(cp,mx,cfg,{},sidecar)
        try:
            x=np.ones((2,64),np.float32)
            np.testing.assert_allclose(np.asarray(reader.output(mx.array(x),'w')),
                                       x@expected.T,rtol=2e-6,atol=2e-5)
            assert reader.stats['output_pipeline_peak_payload_bytes']<=cfg.io_bytes
            assert cp.stats['read_calls']==0
        finally:reader.close()
