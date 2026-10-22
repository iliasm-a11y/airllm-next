"""Measure codecs on byte slices of genuine local Ling BF16 weights.

No remote download. No implication that Ling statistics equal Qwen or GLM.
Results include all repeats and a checksum-inclusive path; decoded-GB/s is not
an SSD rate. C output storage is allocated before the kernel benchmark.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import time
import zlib
from compression import zstd
import numpy as np
from common import ROOT, digest, save
from airllm.nextgen.lossless import palette_encode, palette_decode


def native_decoder():
    lib = ROOT/'build/palette.dylib'
    lib.parent.mkdir(exist_ok=True)
    result = subprocess.run(['clang','-O3','-fno-fast-math','-ffp-contract=off',
        '-shared','-fPIC',str(ROOT/'palette_decode.c'),'-o',str(lib)],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    dll = ctypes.CDLL(str(lib))
    fn = dll.palette_decode
    fn.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    fn.restype = ctypes.c_int
    return dll, fn


def collect(model_dir, output, sample_bytes):
    index = json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    files = {}
    picked = []
    for layer in (1,10,19):
        for expert in (0,127,255):
            for projection in ('gate_proj','up_proj','down_proj'):
                key = f'model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
                filename = index[key]
                if filename not in files:
                    with (model_dir/filename).open('rb') as f:
                        length = struct.unpack('<Q',f.read(8))[0]
                        files[filename] = (8+length,json.loads(f.read(length)))
                base, header = files[filename]
                d = header[key]
                assert d['dtype'] == 'BF16'
                a,b = d['data_offsets']
                count = min(sample_bytes,b-a)
                for loc,start in [('first',0),('middle',((b-a-count)//2)//2*2),('last',b-a-count)]:
                    fd = os.open(model_dir/filename,os.O_RDONLY)
                    try: raw = os.pread(fd,count,base+a+start)
                    finally: os.close(fd)
                    if len(raw) != count: raise EOFError('Truncated original weight sample')
                    picked.append(dict(key=key,part=loc,source_file=filename,source_tensor_shape=d['shape'],
                        source_byte_offset=base+a+start,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw=raw))
    # A standalone safetensors fixture retains exact sampled bits. Its rows are
    # sample words, not whole original expert matrices.
    fixture = output/'model.safetensors'
    if fixture.exists(): return fixture, picked
    payload = bytearray()
    header = {}
    for i,p in enumerate(picked):
        begin = len(payload)
        payload.extend(p['raw'])
        header[f'sample.{i}.weight'] = dict(dtype='BF16',shape=[1,len(p['raw'])//2],data_offsets=[begin,len(payload)])
    rawh = json.dumps(header,separators=(',',':')).encode()
    rawh += b' '*((-len(rawh))%8)
    fixture.write_bytes(struct.pack('<Q',len(rawh))+rawh+payload)
    return fixture,picked


def entropy(a):
    p = np.bincount(a,minlength=256).astype(np.float64)
    p = p[p>0]/len(a)
    return float(-np.sum(p*np.log2(p)))


def run(model_dir, repeats, sample_bytes):
    output = ROOT/'samples'
    output.mkdir(exist_ok=True)
    fixture, picked = collect(model_dir,output,sample_bytes)
    save('sample_provenance.json',dict(model='inclusionAI/Ling-mini-2.0 (local inherited files)',
        source_config_sha256=digest(model_dir/'config.json'),fixture_sha256=digest(fixture),
        selected_weight_bytes=sum(p['bytes'] for p in picked),sampling='3 layers x 3 experts x 3 matrices x first/middle/last',
        samples=[{k:v for k,v in p.items() if k!='raw'} for p in picked],
        limitation='Not Qwen or GLM payload. Finite empirical entropies are not certified lossless lower bounds.'))
    dll, fn = native_decoder()
    # Exhaustive arbitrary bit patterns, odd word counts and escape-heavy data.
    roundtrips = []
    rng = np.random.default_rng(20261022)
    for n in (0,1,3,31,32,33,65536):
        raw = np.arange(65536,dtype='<u2').tobytes() if n==65536 else rng.integers(0,256,n*2,dtype=np.uint8).tobytes()
        packed = palette_encode(raw)
        dest = ctypes.create_string_buffer(max(1,len(raw)))
        rc = fn(packed,len(packed),dest,len(raw)//2)
        assert rc == 0 and dest.raw[:len(raw)] == raw and palette_decode(packed,len(raw)) == raw
        roundtrips.append(dict(words=n,all_bits_equal=True))
    raw = b''.join(p['raw'] for p in picked)
    words = np.frombuffer(raw,np.uint8).reshape(-1,2)
    hi = words[:,1]
    exponents = ((np.frombuffer(raw,dtype='<u2')>>7)&255).astype(np.uint8)
    packed_pal = palette_encode(raw)
    codecs = {
        'raw': (raw,lambda p:p),
        'zlib1': (zlib.compress(raw,1),lambda p:zlib.decompress(p)),
        'zstd1': (zstd.compress(raw,level=1),lambda p:zstd.decompress(p)),
        'shuffle2_zstd1': (zstd.compress(words.T.tobytes(),level=1),
            lambda p:np.frombuffer(zstd.decompress(p),np.uint8).reshape(2,-1).T.tobytes()),
        'palette16_numpy': (packed_pal,lambda p:palette_decode(p,len(raw))),
    }
    rows = []
    expected = hashlib.sha256(raw).digest()
    for name,(payload,decode) in codecs.items():
        assert decode(payload) == raw
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            decoded = decode(payload)
            actual = hashlib.sha256(decoded).digest()
            elapsed = time.perf_counter()-start
            assert actual == expected
            times.append(elapsed)
        med = statistics.median(times)
        rows.append(dict(codec=name,raw_bytes=len(raw),encoded_bytes=len(payload),ratio=len(payload)/len(raw),
            decode_and_hash_seconds=times,median_decoded_GBps=len(raw)/med/1e9,
            includes_allocations=True,all_bits_equal=True))
    outbuf = ctypes.create_string_buffer(len(raw))
    assert fn(packed_pal,len(packed_pal),outbuf,len(raw)//2)==0 and outbuf.raw == raw
    native_nohash, native_hash, native_full = [],[],[]
    for _ in range(repeats):
        t = time.perf_counter()
        assert fn(packed_pal,len(packed_pal),outbuf,len(raw)//2)==0
        native_nohash.append(time.perf_counter()-t)
        t = time.perf_counter()
        assert fn(packed_pal,len(packed_pal),outbuf,len(raw)//2)==0
        assert hashlib.sha256(memoryview(outbuf)).digest()==expected
        native_hash.append(time.perf_counter()-t)
        t = time.perf_counter()
        out = ctypes.create_string_buffer(len(raw))
        assert fn(packed_pal,len(packed_pal),out,len(raw)//2)==0
        assert hashlib.sha256(memoryview(out)).digest()==expected
        native_full.append(time.perf_counter()-t)
    for name,times,alloc in [('palette16_c_kernel',native_nohash,False),
                            ('palette16_c_hash',native_hash,False),('palette16_c_alloc_hash',native_full,True)]:
        rows.append(dict(codec=name,raw_bytes=len(raw),encoded_bytes=len(packed_pal),ratio=len(packed_pal)/len(raw),
            decode_and_hash_seconds=times,median_decoded_GBps=len(raw)/statistics.median(times)/1e9,
            includes_allocations=alloc,includes_hash=name!='palette16_c_kernel',all_bits_equal=True))
    output_data = dict(evidence='local CPU microbenchmark, genuine sampled trained BF16 weights',
        platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
        sample_bytes=len(raw),repeats=repeats,roundtrip_tests=roundtrips,
        empirical_marginal_entropy=dict(low_byte=entropy(words[:,0]),high_byte=entropy(hi),exponent=entropy(exponents)),
        results=rows,
        limitations=['No SSD I/O in decode throughput benchmark.',
        'One run order; no thermal stabilization; full raw timings retained.',
        'Native C is an opt-in prototype, not the production transport default.',
        'Palette compression is familiar dictionary coding; no priority/novelty claim.'])
    save('compression.json',output_data)
    print([(r['codec'],round(r['ratio'],4),round(r['median_decoded_GBps'],3)) for r in rows],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=15)
    ap.add_argument('--sample-bytes',type=int,default=256*1024)
    a=ap.parse_args()
    run(a.model_dir,a.repeats,a.sample_bytes)
