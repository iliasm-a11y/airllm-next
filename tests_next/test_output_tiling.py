"""Output projection tiles obey decoded/packed budgets and release each buffer."""
from types import SimpleNamespace

import numpy as np
import pytest
import mlx.core as mx
from safetensors.numpy import save_file

from airllm.nextgen.cache import size_of
from airllm.nextgen.config import NextConfig
from airllm.nextgen.quantization import dequant_affine
from airllm.nextgen.sidecar import QuantizedSidecar
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.weights import NativeWeights


@pytest.mark.parametrize('max_bytes,io_bytes,row_chunk,expected_rows',[
    (1024,768,2048,3),(768,1024,2048,3),(1024,768,2,2)])
def test_output_tiles_shrink_to_both_budgets_and_drain_lookahead(tmp_path,max_bytes,io_bytes,row_chunk,expected_rows):
    values=np.arange(33*64,dtype='f4').reshape(33,64)/256
    root=tmp_path/'cp';root.mkdir();save_file({'w':values},root/'model.safetensors')
    cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=io_bytes,
                   max_tensor_bytes=max_bytes,row_chunk=row_chunk,load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=max_bytes) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            assert reader.io.submit('unused',128,lambda:np.zeros(128,np.uint8))
            observed=[];load=reader._load
            def checked_load(key,expert=None,rows=None):
                assert not reader.io.pending and reader.io.bytes==0
                observed.append(rows)
                value=load(key,expert,rows)
                assert size_of(value)<=min(max_bytes,io_bytes)
                return value
            reader._load=checked_load
            x=mx.ones((2,64),dtype=mx.float32)
            actual=reader.output(x,'w')
            np.testing.assert_array_equal(np.asarray(actual),np.ones((2,64),np.float32)@values.T)
            assert all(stop-start<=expected_rows for start,stop in observed)
            assert observed[0]==(0,expected_rows) and observed[-1][1]==33
            assert reader.stats['output_tiles']==len(observed)
            assert reader.stats['output_tile_peak_bytes']==expected_rows*64*4
            assert reader.io.stats['discarded']==1
        finally:reader.close()


def test_output_tiles_use_packed_row_size(tmp_path):
    values=np.arange(33*64,dtype='f4').reshape(33,64)/256
    root=tmp_path/'cp';root.mkdir();save_file({'w':values},root/'model.safetensors')
    sidecar=QuantizedSidecar(tmp_path/'q','source',group_size=32,create=True)
    sidecar.add('w',[values],values.shape)
    expected_weights=dequant_affine(sidecar.load('w'))
    sidecar.stats.clear()
    cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=200,
                   max_tensor_bytes=240,row_chunk=2048,load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=240) as cp:
        reader=NativeWeights(cp,mx,cfg,{},sidecar)
        try:
            x=np.ones((2,64),np.float32)
            actual=reader.output(mx.array(x),'w')
            np.testing.assert_allclose(np.asarray(actual),x@expected_weights.T,rtol=2e-6,atol=2e-5)
            # Each row is 32 packed bytes + 8 scale + 8 bias bytes.
            assert reader.stats['output_tile_peak_bytes']==4*48
            assert reader.stats['output_tiles']==9
            assert sidecar.stats['payload_read_bytes']==sidecar.entries['w']['nbytes']
            assert cp.stats['read_calls']==0
        finally:reader.close()


def test_output_rejects_unrepresentable_row_before_io(tmp_path):
    root=tmp_path/'cp';root.mkdir();save_file({'w':np.ones((3,64),np.float32)},root/'model.safetensors')
    cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=128,max_tensor_bytes=1024,load_tokenizer=False)
    with OffsetCheckpoint(root,max_read_bytes=1024) as cp:
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            with pytest.raises(MemoryError,match='One output projection row'):
                reader.output(mx.ones((1,64)),'w')
            assert cp.stats['logical_bytes']==0
        finally:reader.close()


def test_output_releases_previous_evaluated_tile_before_loading_next():
    live=set()
    class Tile:
        def __init__(self,rows):
            self.rows=rows;self.nbytes=(rows[1]-rows[0])*64;live.add(rows)
        def __del__(self):live.remove(self.rows)
    fake_mx=SimpleNamespace(array=np.array,float32=np.float32,eval=lambda value:None,
                            concatenate=lambda values,axis:np.concatenate(values,axis=axis))
    cp=SimpleNamespace(tensors={'w':{'shape':[5,16],'dtype':'F32'}})
    cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=128,max_tensor_bytes=128,row_chunk=2,load_tokenizer=False)
    reader=NativeWeights(cp,fake_mx,cfg,{})
    try:
        def load(key,rows):
            assert not live
            return Tile(rows)
        reader._load=load
        reader.linear_weight=lambda x,w:np.zeros((1,w.rows[1]-w.rows[0]))
        assert reader.output(np.zeros((1,16)),'w').shape==(1,5)
        assert not live
    finally:reader.close()
