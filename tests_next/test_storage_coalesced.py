"""Bounded offset-IO regressions, independent of full checkpoint performance."""
from threading import Event, Thread
import numpy as np
import pytest
import mlx.core as mx
from safetensors.numpy import save_file
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.cache import SingleFlight
from airllm.nextgen.cache import WeightCache
from airllm.nextgen.config import NextConfig
from airllm.nextgen.weights import NativeWeights
from airllm.nextgen.quantization import FP8_LUT


def checkpoint(tmp_path, values, limit=8192):
    root=tmp_path/'cp';root.mkdir()
    save_file(values,root/'model.safetensors')
    return OffsetCheckpoint(root,max_read_bytes=limit)


def test_coalesced_exact_order_and_single_backing_buffer(tmp_path):
    a=np.arange(64,dtype='f4').reshape(4,16)
    with checkpoint(tmp_path,{'a':a,'b':a+3}) as cp:
        arrays=cp.read_arrays(['b','a'])
        np.testing.assert_array_equal(arrays[0],a+3)
        np.testing.assert_array_equal(arrays[1],a)
        assert cp.stats['read_calls']==1
        assert cp.stats['logical_bytes']==a.nbytes*2
        assert cp.stats['coalesced_gap_bytes']==0
        # Numpy references keep one bytearray alive; no per-tensor copy is made.
        assert arrays[0].base.base.obj is arrays[1].base.base.obj
        before=cp.stats['read_calls']
        separate=cp.read_arrays(['b','a'],coalesce=False)
        assert cp.stats['read_calls']-before==2
        np.testing.assert_array_equal(separate[0],arrays[0])


def test_coalesced_rows_experts_gap_and_preflight_limit(tmp_path):
    bank=np.arange(3*6*8,dtype='f4').reshape(3,6,8)
    with checkpoint(tmp_path,{'bank':bank}) as cp:
        request=[{'key':'bank','expert':1,'rows':(0,2)},
                 {'key':'bank','expert':1,'rows':(3,5)}]
        arrays=cp.read_arrays(request,max_gap_bytes=32,max_batch_bytes=160)
        np.testing.assert_array_equal(arrays[0],bank[1,:2])
        np.testing.assert_array_equal(arrays[1],bank[1,3:5])
        assert cp.stats['logical_bytes']==160
        assert cp.stats['requested_bytes']==128
        assert cp.stats['coalesced_gap_bytes']==32
        reads=cp.stats['read_calls']
        with pytest.raises(MemoryError):
            cp.read_arrays(request,max_gap_bytes=32,max_batch_bytes=159)
        assert cp.stats['read_calls']==reads


def test_nested_overlapping_requests_do_not_invent_gaps(tmp_path):
    a=np.arange(80,dtype='f4').reshape(10,8)
    with checkpoint(tmp_path,{'a':a}) as cp:
        arrays=cp.read_arrays([{'key':'a','rows':(0,10)},
                              {'key':'a','rows':(1,2)},
                              {'key':'a','rows':(7,9)}])
        np.testing.assert_array_equal(arrays[2],a[7:9])
        assert cp.stats['coalesced_backing_bytes']==a.nbytes
        assert cp.stats['coalesced_gap_bytes']==0


@pytest.mark.parametrize('selective',[False,True])
def test_fp8_output_tile_reads_only_intersected_scale_rows(tmp_path,selective):
    rng=np.random.default_rng(17)
    raw=rng.integers(0,126,(257,512),dtype=np.uint8)
    scales=rng.uniform(.01,.1,(17,32)).astype('f4')
    with checkpoint(tmp_path,{'w':raw,'w_scale_inv':scales}) as cp:
        # safetensors numpy does not expose the float8 dtype; the on-disk bytes
        # are identical, and the descriptor declares their production meaning.
        cp.tensors['w']['dtype']='F8_E4M3'
        cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=16384,
                       max_tensor_bytes=16384,load_tokenizer=False,selective_fp8_scales=selective)
        reader=NativeWeights(cp,mx,cfg,{'quantization_config':{'weight_block_size':[16,16]}})
        try:
            actual=np.asarray(reader._load('w',rows=(129,133)))
            expected=FP8_LUT[raw[129:133]]*scales[
                (np.arange(129,133)//16)[:,None],(np.arange(512)//16)[None,:]]
            np.testing.assert_array_equal(actual,expected)
            expected_scales=scales[8:9] if selective else scales
            assert cp.stats['requested_bytes']==raw[129:133].nbytes+expected_scales.nbytes
        finally:reader.close()


@pytest.mark.parametrize('coalesce,expected_reads',[(False,4),(True,2)])
def test_embedding_consecutive_rows_is_exact_and_optional(tmp_path,coalesce,expected_reads):
    values=np.arange(32*16,dtype='f4').reshape(32,16)
    ids=np.array([[2,3,4,8,2]])
    with checkpoint(tmp_path,{'w':values}) as cp:
        cfg=NextConfig(dtype='float32',cache_bytes=0,io_bytes=16384,
                       max_tensor_bytes=16384,load_tokenizer=False,coalesce_reads=coalesce)
        reader=NativeWeights(cp,mx,cfg,{})
        try:
            np.testing.assert_array_equal(np.asarray(reader.embedding(ids,'w')),values[ids])
            assert cp.stats['read_calls']==expected_reads
        finally:reader.close()


def test_measured_cost_policy_keeps_expensive_reused_matrix():
    c=WeightCache(16,'cost')
    expensive=np.ones(8,np.uint8);cheap=np.ones(8,np.uint8)
    c.get('expensive',8);c.record_load('expensive',.2);c.put('expensive',expensive)
    c.get('cheap',8);c.record_load('cheap',.001);c.put('cheap',cheap)
    c.get('next',8);c.record_load('next',.001);c.put('next',cheap)
    assert c.contains('expensive');assert not c.contains('cheap')
    assert c.bytes==16
    assert len(c.trace)==3;assert all(len(entry)==2 for entry in c.trace)
    assert len(c.load_trace)==3
    with pytest.raises(ValueError):c.record_load('bad',float('nan'))
    c.resize(8);assert c.bytes<=8


def test_singleflight_close_stops_admission_before_drain():
    started=Event();release=Event();draining=Event()
    q=SingleFlight(64,1)
    def load():
        started.set();release.wait(3);return np.zeros(32,np.uint8)
    original_drain=q.drain
    def observed_drain():
        draining.set();original_drain()
    q.drain=observed_drain
    q.submit('x',32,load);assert started.wait(1)
    closer=Thread(target=q.close);closer.start()
    assert draining.wait(1)
    # close marks the admission gate before it waits for the active worker.
    with q.lock:assert q.closed
    with pytest.raises(RuntimeError):q.submit('late',16,lambda:np.zeros(16,np.uint8))
    release.set();closer.join(3)
    assert not closer.is_alive();assert q.bytes==0
    with pytest.raises(ValueError):q.submit('negative',-1,lambda:None)


def test_singleflight_drain_does_not_release_reused_key():
    q=SingleFlight(64,1)
    try:
        q.submit('x',32,lambda:np.zeros(32,np.uint8))
        old=q.pending['x'][0];old.result()
        result=old.result;interleaved=[False]
        def consume_and_resubmit(*args,**kwargs):
            if not interleaved[0]:
                interleaved[0]=True
                assert q.take('x').nbytes==32
                q.submit('x',32,lambda:np.ones(32,np.uint8))
            return result(*args,**kwargs)
        old.result=consume_and_resubmit
        q.drain()
        assert q.bytes==32;assert 'x' in q.pending
        np.testing.assert_array_equal(q.take('x'),1)
        assert q.bytes==0
    finally:q.close()
