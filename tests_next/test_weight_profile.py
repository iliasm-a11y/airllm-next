"""Load-stage attribution separates storage service from MLX materialization.

These fixtures validate counter scope and unchanged values; they do not measure
throughput, SSD bandwidth, or GPU kernel time.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from airllm.nextgen.config import NextConfig
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights


def _fixture(tmp_path, **updates):
    root=tmp_path/'checkpoint';root.mkdir()
    expected=torch.linspace(-2,2,256).reshape(16,16).to(torch.bfloat16)
    save_file({'weight':expected},root/'model.safetensors')
    cfg=NextConfig(dtype='float32',cache_bytes=1024,io_bytes=4096,
        max_tensor_bytes=4096,profile_io=True,prefetch=False,load_tokenizer=False)
    return root,expected.float().numpy(),replace(cfg,**updates)


def test_profiled_bfloat_materialization_preserves_values_and_skips_cached_load(tmp_path):
    root,expected,cfg=_fixture(tmp_path)
    with OffsetCheckpoint(root,max_read_bytes=cfg.max_tensor_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            np.testing.assert_array_equal(np.asarray(reader.get('weight')),expected)
            before=dict(reader.stats)
            np.testing.assert_array_equal(np.asarray(reader.get('weight')),expected)
            assert reader.stats==before
            assert before['source_read_calls']==before['materialize_calls']==before['loads']==1
            assert 0<=before['source_read_seconds']<=before['load_seconds']
            assert 0<=before['materialize_seconds']<=before['load_seconds']
            assert before['source_read_seconds']+before['materialize_seconds']<=before['load_seconds']
        finally:reader.close()


def test_compact_cache_reuse_still_charges_repeat_materialization(tmp_path):
    root,expected,cfg=_fixture(tmp_path,cache_bytes=512,compact_cache_bytes=512)
    with OffsetCheckpoint(root,max_read_bytes=cfg.max_tensor_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            for _ in range(3):
                np.testing.assert_array_equal(np.asarray(reader.get('weight')),expected)
            assert reader.stats['source_read_calls']==1
            assert reader.stats['materialize_calls']==reader.stats['loads']==3
            assert cp.stats['logical_bytes']==512
        finally:reader.close()


def test_profile_disabled_adds_no_stage_counters_or_profile_dispatch(tmp_path,monkeypatch):
    root,expected,cfg=_fixture(tmp_path,profile_io=False)
    with OffsetCheckpoint(root,max_read_bytes=cfg.max_tensor_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            def unexpected(*args,**kwargs):
                raise AssertionError('disabled profile dispatched stage measurement')
            monkeypatch.setattr(reader,'_profile_call',unexpected)
            np.testing.assert_array_equal(np.asarray(reader.get('weight')),expected)
            assert 'source_read_seconds' not in reader.stats
            assert 'materialize_seconds' not in reader.stats
        finally:reader.close()


def test_failed_source_read_is_attributed_without_successful_materialization(tmp_path,monkeypatch):
    root,_,cfg=_fixture(tmp_path)
    with OffsetCheckpoint(root,max_read_bytes=cfg.max_tensor_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            def fail(*args,**kwargs):raise EOFError('source interrupted')
            monkeypatch.setattr(cp,'read_array',fail)
            with pytest.raises(EOFError,match='source interrupted'):reader.get('weight')
            assert reader.stats['source_read_calls']==reader.stats['source_read_failed_calls']==1
            assert reader.stats['materialize_calls']==reader.stats['loads']==0
            assert reader.io.bytes==0 and not reader.io.pending
        finally:reader.close()


def test_worker_stage_counters_keep_all_concurrent_measurements(tmp_path):
    root,_,cfg=_fixture(tmp_path,cache_bytes=0)
    with OffsetCheckpoint(root,max_read_bytes=cfg.max_tensor_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            # Exercise the counter handoff independently of MLX stream timing.
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda x:reader._profile_call('source_read',int,x),range(128)))
            assert results==list(range(128))
            assert reader.stats['source_read_calls']==128
            assert reader.stats['source_read_seconds']>=0
        finally:reader.close()
