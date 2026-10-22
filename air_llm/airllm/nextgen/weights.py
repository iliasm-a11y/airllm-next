"""One shared cache for decoded reference or MLX affine packed matrices.

Embedding rows are read selectively. The output head is tiled. No full model
skeleton or full expert bank is ever allocated by this backend.
"""
from dataclasses import dataclass
import math
from time import perf_counter
import numpy as np
from .quantization import FP8_LUT, PackedMatrix
from .cache import WeightCache, SingleFlight, size_of


class NativeWeights:
    def __init__(self,checkpoint,mx,config,model_config,sidecar=None):
        self.cp=checkpoint;self.mx=mx;self.cfg=config;self.model_config=model_config;self.sidecar=sidecar;self.prefetch_enabled=config.prefetch
        self.cache=WeightCache(config.cache_bytes,config.policy)
        self.io=SingleFlight(config.io_bytes,config.io_workers)
        self.dtype=getattr(mx,config.dtype);self.stats={'dequant_seconds':0.,'loads':0,'speculative_reads':0}
        self.block=tuple(model_config.get('quantization_config',{}).get('weight_block_size',[128,128]))
        self.lut=mx.array(FP8_LUT)

    @staticmethod
    def identity(key,expert=None):return key if expert is None else f'{key}#{expert}'

    def shape(self,key,expert=None):
        shape=self.cp.tensors[key]['shape'];return tuple(shape[1:] if expert is not None else shape)

    def estimate(self,key,expert=None):
        ident=self.identity(key,expert)
        if self.sidecar and ident in self.sidecar.entries:return self.sidecar.entries[ident]['nbytes']
        d=self.cp.tensors[key];n=math.prod(self.shape(key,expert))
        return n*(4 if self.cfg.dtype=='float32' or d['dtype']=='F32' else 2)

    def _array(self,a,dtype=None):
        # Evaluation on CPU ensures data has been read before the Future resolves;
        # on Apple these evaluated buffers are shared with Metal, not copied to VRAM.
        with self.mx.stream(self.mx.cpu):
            v=self.mx.array(a)
            if dtype is not None:v=v.astype(dtype)
            self.mx.eval(v)
        return v

    def _load(self,key,expert=None,rows=None):
        ident=self.identity(key,expert)
        if self.sidecar and ident in self.sidecar.entries:
            p=self.sidecar.load(ident,rows=rows)
            if p.nbytes>self.cfg.max_tensor_bytes:raise MemoryError('Packed matrix tile exceeds max_tensor_bytes')
            with self.cp.io_lock:
                self.cp.stats['logical_bytes']+=p.nbytes
                self.cp.stats['quantized_logical_bytes']+=p.nbytes
            return PackedMatrix(self._array(p.weight),self._array(p.scales,self.mx.float32),
                                self._array(p.biases,self.mx.float32),p.shape,p.bits,p.group_size)
        d=self.cp.tensors[key];a=self.cp.read_array(key,expert=expert,rows=rows)
        if d['dtype'].startswith('F8_'):
            if d['dtype'] not in ('F8_E4M3','F8_E4M3FN'):raise NotImplementedError('Only FP8 E4M3FN is implemented')
            sk=key+'_scale_inv'
            if sk not in self.cp.tensors:raise ValueError(f'Missing FP8 block scales {sk}')
            scales=self.cp.read_array(sk,expert=expert)
            if self.cp.tensors[sk]['dtype']=='BF16':scales=(scales.astype(np.uint32)<<16).view(np.float32)
            br,bc=self.block
            shape=self.shape(key,expert)
            if tuple(scales.shape)!=(math.ceil(shape[0]/br),math.ceil(shape[1]/bc)):
                raise ValueError(f'Invalid FP8 scales for {key}')
            t=perf_counter();offset=rows[0] if rows else 0;mx=self.mx
            # Decode on CPU stream while GPU executes a previous expert. The raw
            # FP8 buffer dies after evaluation. Cached result is reused directly.
            with mx.stream(mx.cpu):
                raw=mx.array(a);s=mx.array(scales)
                ri=(mx.arange(a.shape[0])+offset)//br;ci=mx.arange(a.shape[1])//bc
                v=(self.lut[raw.astype(mx.int32)]*s[ri[:,None],ci[None,:]]).astype(self.dtype)
                mx.eval(v)
            self.stats['dequant_seconds']+=perf_counter()-t
        elif d['dtype']=='BF16':
            with self.mx.stream(self.mx.cpu):
                v=self.mx.array(a).view(self.mx.bfloat16).astype(self.dtype);self.mx.eval(v)
        else:
            # Router scores, normalization weights and correction bias stay fp32.
            target=self.mx.float32 if d['dtype']=='F32' else self.dtype
            v=self._array(a,target)
        self.stats['loads']+=1;return v

    def get(self,key,expert=None):
        ident=self.identity(key,expert);v=self.cache.get(ident,self.estimate(key,expert))
        if v is not None:return v
        n=self.estimate(key,expert)
        if n>self.cfg.max_tensor_bytes:raise MemoryError(f'{key} must be read as row tiles ({n} bytes)')
        v=self.io.take(ident)
        if v is None:
            # Main thread is serialized by model's lock. submit() deduplicates
            # if another prediction already scheduled precisely this matrix.
            if not self.io.submit(ident,n,lambda:self._load(key,expert)):
                # Wait/discard stale lookahead before synchronous demand to avoid
                # exceeding the IO budget with a hidden extra decoded matrix.
                self.io.drain()
                if not self.io.submit(ident,n,lambda:self._load(key,expert)):
                    if n>self.cfg.io_bytes:raise MemoryError('Increase io_bytes for the largest decoded matrix')
                    raise RuntimeError('Could not reserve demand read')
            v=self.io.take(ident)
        self.cache.put(ident,v);return v

    def prefetch(self,key,expert=None):
        if not self.prefetch_enabled:return False
        ident=self.identity(key,expert)
        if self.cache.contains(ident):return False
        n=self.estimate(key,expert)
        if n>self.cfg.max_tensor_bytes:return False
        ok=self.io.submit(ident,n,lambda:self._load(key,expert))
        return ok

    def linear_weight(self,x,w):
        if isinstance(w,PackedMatrix):
            # Keep affine scale/bias and QMM accumulation in FP32. BF16 CPU QMM
            # showed measurable cancellation error against dequantized weights.
            return self.mx.quantized_matmul(x.astype(self.mx.float32),w.weight,w.scales,w.biases,
                          transpose=True,bits=w.bits,group_size=w.group_size,mode='affine')
        return x.astype(w.dtype)@w.T

    def linear(self,x,key,expert=None):return self.linear_weight(x,self.get(key,expert))

    def embedding(self,ids,key):
        mx=self.mx;host=np.asarray(ids);flat=host.reshape(-1);unique,reverse=np.unique(flat,return_inverse=True)
        shape=self.shape(key)
        if len(shape)!=2 or np.any(unique<0) or np.any(unique>=shape[0]):raise ValueError('Token id out of range')
        # Only selected rows (usually one during decode), never the ~1.9 GB table.
        rows=[self._load(key,rows=(int(i),int(i)+1)) for i in unique]
        values=mx.concatenate(rows,axis=0)[mx.array(reverse)]
        return values.reshape(tuple(host.shape)+(shape[1],))

    def output(self,x,key):
        if self.estimate(key)<=min(self.cfg.max_tensor_bytes,self.cfg.io_bytes):return self.linear(x,key)
        # Keep only tile result vectors; each full matrix tile is evaluated and
        # released before the next is read. No lazy graph retains the whole head.
        out=[];rows=self.shape(key)[0]
        for start in range(0,rows,self.cfg.row_chunk):
            w=self._load(key,rows=(start,min(rows,start+self.cfg.row_chunk)))
            y=self.linear_weight(x,w);self.mx.eval(y);out.append(y)
        return self.mx.concatenate(out,axis=-1)

    def close(self):
        try:self.io.close()
        finally:self.cache.clear()
