from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'air_llm'))
sys.path.insert(0,str(ROOT))
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.storage import encoded_read_bytes
from airllm.nextgen.config import NextConfig
from airllm.nextgen.lossless import (LosslessOverlay,NativePalette,pack_checkpoint,
    palette_encode,palette_decode,encode_block,decode_block,RECORD)
from exact_rank import bf16_mod,modular_rank


def fixture(directory):
    directory.mkdir()
    rng=np.random.default_rng(82)
    arrays={'experts':np.arange(3*17*33,dtype='<u2').reshape(3,17,33),
            'embedding':rng.integers(0,65536,(19,33),dtype='<u2'),
            'bytes':rng.integers(0,256,(23,35),dtype=np.uint8),
            'float':rng.normal(size=(17,33)).astype('<f4')}
    data=bytearray();header={}
    for k,a in arrays.items():
        start=len(data);data.extend(a.tobytes())
        header[k]={'shape':list(a.shape),'dtype':{'u': 'BF16' if a.itemsize==2 else 'U8','f':'F32'}[a.dtype.kind],
                   'data_offsets':[start,len(data)]}
    h=json.dumps(header).encode();h+=b' '*(-len(h)%8)
    (directory/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+data)
    return arrays,OffsetCheckpoint(directory,max_read_bytes=1024**2)


@pytest.mark.parametrize('codec',['raw','zlib','shuffle-zlib','palette16','auto'])
def test_full_bitwise_and_partial_expert_rows(tmp_path,codec):
    arrays,cp=fixture(tmp_path/'src')
    pack_checkpoint(cp,tmp_path/'packed',codec=codec,block_bytes=128)
    reader=LosslessOverlay(cp,tmp_path/'packed')
    for k,a in arrays.items(): assert reader.read_array(k).tobytes()==a.tobytes()
    for e in range(3):
        for lo,hi in [(0,1),(1,16),(16,17),(3,3)]:
            a=reader.read_array('experts',expert=e,rows=(lo,hi))
            assert a.tobytes()==arrays['experts'][e,lo:hi].tobytes()
    assert reader.transport_stats['decoded_block_bytes']>=reader.transport_stats['requested_raw_bytes']
    reader.close()


def test_incomplete_pack_never_opens(tmp_path):
    _,cp=fixture(tmp_path/'src');(tmp_path/'pack').mkdir()
    with pytest.raises(FileNotFoundError): LosslessOverlay(cp,tmp_path/'pack')
    cp.close()


def test_payload_corruption_fails_closed(tmp_path):
    _,cp=fixture(tmp_path/'src')
    pack_checkpoint(cp,tmp_path/'pack',codec='raw',block_bytes=128)
    p=tmp_path/'pack/weights.bin';b=bytearray(p.read_bytes());b[0]^=1;p.write_bytes(b)
    r=LosslessOverlay(cp,tmp_path/'pack')
    with pytest.raises(ValueError,match='checksum'):r.read_array('bytes')
    assert r.transport_stats['payload_bytes']==128
    assert r.transport_stats['index_bytes']==RECORD.size
    r.close()


def test_bound_and_wrong_source_guard(tmp_path):
    _,cp=fixture(tmp_path/'src')
    with pytest.raises(MemoryError):pack_checkpoint(cp,tmp_path/'never',max_total_bytes=1)
    assert not (tmp_path/'never').exists()
    pack_checkpoint(cp,tmp_path/'pack',block_bytes=128)
    with pytest.raises(ValueError):LosslessOverlay(cp,tmp_path/'pack',max_block_bytes=64)
    meta=json.loads((tmp_path/'pack/manifest.json').read_text());meta['source_fingerprint']='wrong'
    (tmp_path/'pack/manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='identity'):LosslessOverlay(cp,tmp_path/'pack')
    cp.close()


def test_index_truncation_and_bogus_lengths(tmp_path):
    _,cp=fixture(tmp_path/'src');pack_checkpoint(cp,tmp_path/'pack',block_bytes=128)
    p=tmp_path/'pack/blocks.idx';b=p.read_bytes();p.write_bytes(b[:-1])
    with pytest.raises(ValueError):LosslessOverlay(cp,tmp_path/'pack')
    p.write_bytes(b)
    r=LosslessOverlay(cp,tmp_path/'pack')
    entry=r.entries['bytes'];off=entry['index_offset']
    q=list(RECORD.unpack(b[off:off+RECORD.size]));q[1]=2**30
    p.write_bytes(b[:off]+RECORD.pack(*q)+b[off+RECORD.size:])
    with pytest.raises(ValueError,match='index'):r.read_array('bytes')
    r.close()


def test_mixed_packed_fallback_and_concurrent_reads(tmp_path):
    arrays,cp=fixture(tmp_path/'src');pack_checkpoint(cp,tmp_path/'pack',keys=['experts'],block_bytes=128)
    r=LosslessOverlay(cp,tmp_path/'pack')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda i:r.read_array('experts',expert=i%3).tobytes(),range(40)))
    for i,b in enumerate(results):assert b==arrays['experts'][i%3].tobytes()
    got=r.read_arrays([{'key':'experts','expert':1,'rows':(3,7)},'embedding'],max_batch_bytes=10000)
    assert got[0].tobytes()==arrays['experts'][1,3:7].tobytes()
    assert got[1].tobytes()==arrays['embedding'].tobytes()
    with pytest.raises(MemoryError):r.read_arrays(['experts','embedding'],max_batch_bytes=1)
    r.close()
    with pytest.raises(RuntimeError):r.read_array('experts')


@pytest.mark.parametrize('n',[0,1,3,31,32,33,65536])
def test_palette_every_bit_pattern(n):
    a=np.arange(65536,dtype='<u2') if n==65536 else np.random.default_rng(n).integers(0,65536,n,dtype='<u2')
    raw=a.tobytes();payload=palette_encode(raw)
    assert palette_decode(payload,len(raw))==raw
    lib=ROOT/'build/palette.dylib'
    if lib.exists():assert NativePalette(lib)(payload,len(raw))==raw


def test_incompressible_raw_fallback_and_expansion_attack():
    raw=np.random.default_rng(1).integers(0,256,65536,dtype=np.uint8).tobytes()
    for c in ['auto','zlib','shuffle-zlib','palette16']:
        kind,payload=encode_block(raw,c)
        assert len(payload)<=len(raw) and decode_block(kind,payload,len(raw))==raw
    import zlib
    with pytest.raises(ValueError):decode_block(1,zlib.compress(b'x'*100000),16)


def test_modular_rank_certificate():
    # Exact BF16 +1, -2, tiny subnormal, +zero and -zero.
    w=np.array([0x3f80,0xc000,1,0,0x8000],dtype=np.uint16)
    residues=bf16_mod(w)
    assert residues[0]==1 and residues[1]==65519
    assert residues[2]==pow(2,-133,65521) and residues[3]==residues[4]==0
    assert modular_rank([[1,2,3],[2,4,6]])==1
    assert modular_rank([[1,2],[3,4]])==2


def test_encoded_bytes_include_index_and_unpacked_fallback(tmp_path):
    _,cp=fixture(tmp_path/'src')
    pack_checkpoint(cp,tmp_path/'pack',keys=['experts'],block_bytes=128,codec='palette16')
    r=LosslessOverlay(cp,tmp_path/'pack')
    before=encoded_read_bytes(r)
    r.read_array('experts',expert=1,rows=(0,2))
    r.read_array('embedding')
    expected=(r.transport_stats['payload_bytes']+r.transport_stats['index_bytes']+cp.tensors['embedding']['bytes'])
    assert encoded_read_bytes(r)-before==expected
    r.close()


def test_lossless_cannot_mislabel_quantized_sidecar():
    with pytest.raises(ValueError,match='quantized sidecar'):
        NextConfig(lossless_dir='/pack',quantized_dir='/int4')
    with pytest.raises(ValueError,match='boolean'):
        NextConfig(lossless_verify_hash='false')
