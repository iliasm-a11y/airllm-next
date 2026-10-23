import json
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pytest
import torch
import mlx.core as mx
from safetensors.numpy import save_file
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.quantization import FP8_LUT,dequant_fp8,quantize_affine,dequant_affine
from airllm.nextgen.cache import WeightCache,SingleFlight
from airllm.nextgen.sidecar import QuantizedSidecar,convert_experts


def test_fp8_all_bit_patterns():
    expected=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_allclose(FP8_LUT,expected,rtol=0,atol=0,equal_nan=True)


def test_fp8_nonuniform_tail_blocks():
    rng=np.random.default_rng(6);raw=rng.integers(0,126,(35,70),dtype=np.uint8);s=rng.uniform(.01,.5,(3,5)).astype('f4')
    w=torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float()
    scales=torch.from_numpy(s).repeat_interleave(16,0).repeat_interleave(16,1)[:35,:70]
    np.testing.assert_allclose(dequant_fp8(raw,s,(16,16)),(w*scales).numpy(),atol=0,rtol=0)
    np.testing.assert_array_equal(dequant_fp8(raw[11:25],s,(16,16),row_offset=11),(w*scales).numpy()[11:25])


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_mlx_packed_matmul_true_kernel(bits,dtype):
    rng=np.random.default_rng(72);w=rng.normal(size=(40,128)).astype('f4');x=rng.normal(size=(7,128)).astype('f4')
    p=quantize_affine(w,bits,64)
    dt=getattr(mx,dtype);xx=mx.array(x).astype(dt)
    # Production policy: activation may be BF16, affine weights/scales and
    # accumulation stay FP32 to avoid low-precision cancellation error.
    actual=mx.quantized_matmul(xx.astype(mx.float32),mx.array(p.weight),mx.array(p.scales),mx.array(p.biases),bits=bits,group_size=64)
    expected=np.array(xx.astype(mx.float32))@dequant_affine(p).T
    np.testing.assert_allclose(np.array(actual),expected,atol=2e-5,rtol=2e-5)
    assert p.nbytes<w.nbytes


def test_exact_rows_experts_and_mmap(tmp_path):
    data=np.arange(4*13*16,dtype=np.float32).reshape(4,13,16);root=tmp_path/'cp';root.mkdir()
    save_file({'bank':data,'x':data[0]},root/'model.safetensors')
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:
        before=cp.stats['logical_bytes'];a=cp.read_array('bank',expert=2,rows=(3,8))
        np.testing.assert_array_equal(a,data[2,3:8]);assert cp.stats['logical_bytes']-before==5*16*4
        np.testing.assert_array_equal(cp.read_array('x',rows=(1,4),mode='mmap'),data[0,1:4])
    with OffsetCheckpoint(root,tmp_path/'meta') as cp:assert cp.stats['manifest_hit']==1


@pytest.mark.parametrize('without_preadv',[False,True])
def test_short_reads_eintr_and_disappeared_file(tmp_path,monkeypatch,without_preadv):
    import os,errno
    root=tmp_path/'cp';root.mkdir();save_file({'a':np.ones((4,16),np.float32)},root/'model.safetensors')
    with OffsetCheckpoint(root) as cp:
        real_pread=os.pread;real_preadv=getattr(os,'preadv',None);calls=[0]
        def interrupted_once():
            calls[0]+=1
            if calls[0]==1:raise OSError(errno.EINTR,'interrupted')
        def partial_pread(fd,size,offset):
            interrupted_once()
            return real_pread(fd,min(size,7),offset)
        def partial_preadv(fd,buffers,offset):
            interrupted_once()
            return real_preadv(fd,[buffers[0][:7]],offset)
        # Instrument whichever syscall the platform actually selects. Exercise
        # the fallback explicitly too, including Python builds without preadv.
        monkeypatch.setattr(os,'pread',partial_pread)
        if without_preadv:monkeypatch.delattr(os,'preadv',raising=False)
        elif real_preadv is not None:monkeypatch.setattr(os,'preadv',partial_preadv)
        np.testing.assert_array_equal(cp.read_array('a'),1);assert calls[0]>2
        (root/'model.safetensors').write_bytes(b'')
        with pytest.raises(EOFError):cp.read_array('a')


@pytest.mark.parametrize('policy',['lru','slru','lfu'])
def test_byte_cache_and_resize(policy):
    c=WeightCache(400,policy)
    for i in range(20):
        c.get(i%6);c.put(i%6,np.zeros(50,np.uint8));assert c.bytes<=400
    c.resize(80);assert c.bytes<=80
    assert not c.put('huge',np.zeros(400,np.uint8))
    c.clear();assert c.bytes==0


def test_singleflight_dedup_reserve_and_error():
    event=Event();count=[0]
    def load():count[0]+=1;event.wait(2);return np.zeros(40,np.uint8)
    q=SingleFlight(80,2)
    try:
        with ThreadPoolExecutor(4) as pool:assert all(pool.map(lambda _:q.submit('x',40,load),range(12)))
        assert q.bytes==40;assert not q.submit('too big',90,load)
        event.set();assert q.take('x').nbytes==40;assert count==[1];assert q.bytes==0
        def broken():raise OSError('SSD disconnected')
        q.submit('bad',10,broken)
        with pytest.raises(OSError):q.take('bad')
        assert q.bytes==0
        q.submit('overflow',20,lambda:np.zeros(30,np.uint8))
        with pytest.raises(MemoryError):q.take('overflow')
        assert q.bytes==0
    finally:q.close()


def test_sidecar_interrupt_resume_and_fingerprint(tmp_path):
    root=tmp_path/'q';s=QuantizedSidecar(root,'base',create=True,group_size=32)
    rng=np.random.default_rng(4);w=rng.normal(size=(25,64)).astype('f4')
    def bad():yield w[:8];raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):s.add('key',bad(),w.shape)
    assert not s.entries
    resumed=QuantizedSidecar(root,'base',create=True,group_size=32)
    assert resumed.add('key',[w[:8],w[8:]],w.shape)
    p=resumed.load('key');assert p.shape==w.shape
    assert np.linalg.norm(dequant_affine(p)-w)/np.linalg.norm(w)<.15
    assert not resumed.add('key',[w],w.shape)
    with pytest.raises(ValueError,match='different base'):QuantizedSidecar(root,'other')
