"""Exact, bounded imports retain exporters without retaining oversized views."""
import gc
import weakref

import mlx.core as mx
import numpy as np
import pytest

from airllm.nextgen.config import NextConfig
from airllm.nextgen.weights import NativeWeights


def _reader():
    return NativeWeights(None,mx,NextConfig(zero_copy_weights=True,load_tokenizer=False),{})


def test_zero_copy_retains_readonly_exporter_until_last_mlx_reference_is_released():
    reader=_reader()
    try:
        expected=np.arange(64,dtype=np.float32).reshape(8,8)
        source=np.frombuffer(expected.tobytes(),dtype=np.float32).reshape(8,8)
        ref=weakref.ref(source)
        value=reader._array(source,mx.float32)
        if not reader.stats.get('zero_copy_imports'):
            pytest.skip('MLX/platform cannot adopt this host buffer')
        assert not source.flags.writeable
        assert np.asarray(value).ctypes.data==source.ctypes.data
        del source;gc.collect()
        assert ref() is not None
        np.testing.assert_array_equal(np.asarray(value),expected)
        assert reader.stats['zero_copy_import_bytes']==expected.nbytes
        del value;gc.collect()
        assert ref() is None
    finally:reader.close()


@pytest.mark.parametrize('selection',[lambda a:a[4:12],lambda a:a[::2]])
def test_partial_or_strided_views_copy_to_avoid_hidden_backing_residency(selection):
    reader=_reader()
    try:
        parent=np.arange(32,dtype=np.float32)
        source=selection(parent);expected=source.copy()
        value=reader._array(source,mx.float32)
        assert reader.stats.get('zero_copy_imports',0)==0
        assert reader.stats['zero_copy_fallbacks']==1
        parent.fill(-1)
        np.testing.assert_array_equal(np.asarray(value),expected)
    finally:reader.close()


@pytest.mark.parametrize('error',[BufferError,TypeError,ValueError,RuntimeError])
def test_rejected_host_pointer_has_an_exact_copy_fallback(monkeypatch,error):
    reader=_reader()
    try:
        def reject(*args,**kwargs):raise error('unsupported host import')
        monkeypatch.setattr(mx,'from_dlpack',reject,raising=False)
        source=np.arange(32,dtype=np.float32)
        value=reader._array(source,mx.float32)
        assert reader.stats['zero_copy_fallbacks']==1
        assert reader.stats.get('zero_copy_imports',0)==0
        source.fill(-1)
        np.testing.assert_array_equal(np.asarray(value),np.arange(32,dtype=np.float32))
    finally:reader.close()


def test_missing_dlpack_api_has_an_exact_copy_fallback(monkeypatch):
    reader=_reader()
    try:
        monkeypatch.delattr(mx,'from_dlpack',raising=False)
        source=np.arange(32,dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(reader._array(source,mx.float32)),source)
        assert reader.stats['zero_copy_fallbacks']==1
    finally:reader.close()


@pytest.mark.parametrize('fp8',[False,True])
def test_streamed_model_logits_routes_and_generated_tokens_are_exact(tmp_path,fp8):
    from dataclasses import replace
    from test_fp8_model import save_glm64
    from test_native import options
    from airllm.nextgen.runtime import AirLLMNextModel
    root=tmp_path/'model';save_glm64(root,fp8=fp8)
    cfg=options(tmp_path,cache_bytes=128*1024,prefetch=False,use_saved_tuning=False)
    ids=[[1,8,9,7,11]]
    with AirLLMNextModel(root,next_config=cfg) as control:
        expected_logits=np.asarray(control(ids).logits)
        expected_routes={key:dict(value) for key,value in control.routes.items()}
        expected_tokens=control.generate(ids,max_new_tokens=8,eos_token_id=[]).tolist()
    optimized_cfg=replace(cfg,zero_copy_weights=True,cache_dir=str(tmp_path/'optimized-metadata'))
    with AirLLMNextModel(root,next_config=optimized_cfg) as optimized:
        np.testing.assert_array_equal(np.asarray(optimized(ids).logits),expected_logits)
        assert {key:dict(value) for key,value in optimized.routes.items()}==expected_routes
        assert optimized.generate(ids,max_new_tokens=8,eos_token_id=[]).tolist()==expected_tokens
        assert optimized.reader.cache.bytes<=cfg.cache_bytes
        assert optimized.reader.io.peak_bytes<=cfg.io_bytes
        assert optimized.reader.stats.get('zero_copy_imports',0)+optimized.reader.stats.get('zero_copy_fallbacks',0)>0


@pytest.mark.parametrize('bad',[None,0,1,'yes'])
def test_zero_copy_option_requires_a_boolean(bad):
    with pytest.raises(ValueError,match='zero_copy_weights must be boolean'):
        NextConfig(zero_copy_weights=bad)
