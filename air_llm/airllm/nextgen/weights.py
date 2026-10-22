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
        start=perf_counter()
        value=self._load_value(key,expert,rows)
        elapsed=perf_counter()-start
        if rows is None:self.cache.record_load(self.identity(key,expert),elapsed)
        self.stats['loads']+=1
        self.stats.setdefault('load_seconds',0.)
        self.stats['load_seconds']+=elapsed
        return value

    def _load_value(self,key,expert=None,rows=None):
        ident=self.identity(key,expert)
        if self.sidecar and ident in self.sidecar.entries:
            p=self.sidecar.load(ident,rows=rows,max_bytes=self.cfg.max_tensor_bytes)
            if p.nbytes>self.cfg.max_tensor_bytes:raise MemoryError('Packed matrix tile exceeds max_tensor_bytes')
            with self.cp.io_lock:
                self.cp.stats['logical_bytes']+=p.nbytes
                self.cp.stats['quantized_logical_bytes']+=p.nbytes
            return PackedMatrix(self._array(p.weight),self._array(p.scales,self.mx.float32),
                                self._array(p.biases,self.mx.float32),p.shape,p.bits,p.group_size)
        d=self.cp.tensors[key]
        if d['dtype'].startswith('F8_'):
            if d['dtype'] not in ('F8_E4M3','F8_E4M3FN'):raise NotImplementedError('Only FP8 E4M3FN is implemented')
            sk=key+'_scale_inv'
            if sk not in self.cp.tensors:raise ValueError(f'Missing FP8 block scales {sk}')
            br,bc=self.block
            shape=self.shape(key,expert)
            if tuple(self.shape(sk,expert))!=(math.ceil(shape[0]/br),math.ceil(shape[1]/bc)):
                raise ValueError(f'Invalid FP8 scales for {key}')
            # Output-head tiles only need the scale rows they intersect. Reading
            # the full scale matrix for every tile causes O(number_of_tiles)
            # redundant scale IO on a massive FP8 checkpoint.
            offset=rows[0] if rows else 0
            scale_rows=(offset//br,math.ceil(rows[1]/br)) if rows and self.cfg.selective_fp8_scales else None
            requests=[{'key':key,'expert':expert,'rows':rows},
                      {'key':sk,'expert':expert,'rows':scale_rows}]
            # No hidden overread: only adjacent payload/scale ranges combine.
            a,scales=self.cp.read_arrays(requests,max_batch_bytes=self.cp.max_read_bytes,
                                         coalesce=self.cfg.coalesce_reads)
            if self.cp.tensors[sk]['dtype']=='BF16':scales=(scales.astype(np.uint32)<<16).view(np.float32)
            t=perf_counter();offset=rows[0] if rows else 0;mx=self.mx
            # Decode on CPU stream while GPU executes a previous expert. The raw
            # FP8 buffer dies after evaluation. Cached result is reused directly.
            with mx.stream(mx.cpu):
                raw=mx.array(a);s=mx.array(scales)
                ri=(mx.arange(a.shape[0])+offset)//br-(scale_rows[0] if scale_rows else 0);ci=mx.arange(a.shape[1])//bc
                v=(self.lut[raw.astype(mx.int32)]*s[ri[:,None],ci[None,:]]).astype(self.dtype)
                mx.eval(v)
            self.stats['dequant_seconds']+=perf_counter()-t
        elif d['dtype']=='BF16':
            a=self.cp.read_array(key,expert=expert,rows=rows)
            with self.mx.stream(self.mx.cpu):
                v=self.mx.array(a).view(self.mx.bfloat16).astype(self.dtype);self.mx.eval(v)
        else:
            a=self.cp.read_array(key,expert=expert,rows=rows)
            # Router scores, normalization weights and correction bias stay fp32.
            target=self.mx.float32 if d['dtype']=='F32' else self.dtype
            v=self._array(a,target)
        return v

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
        # Consecutive token IDs share one bounded row read rather than one IO
        # operation per token; sorted unique IDs preserve the gather mapping.
        row_bytes=max(1,self.estimate(key)//max(1,shape[0]))
        max_rows=max(1,self.cfg.max_tensor_bytes//row_bytes)
        runs=[]
        for i in unique:
            i=int(i)
            if self.cfg.coalesce_reads and runs and i==runs[-1][1] and runs[-1][1]-runs[-1][0]<max_rows:
                runs[-1]=(runs[-1][0],i+1)
            else:runs.append((i,i+1))
        rows=[self._load(key,rows=run) for run in runs]
        values=mx.concatenate(rows,axis=0)[mx.array(reverse)]
        return values.reshape(tuple(host.shape)+(shape[1],))

    def output(self,x,key):
        budget=min(self.cfg.max_tensor_bytes,self.cfg.io_bytes)
        size=self.estimate(key)
        if size<=budget:return self.linear(x,key)
        shape=self.shape(key)
        if len(shape)!=2 or min(shape)<1:raise ValueError('Output projection requires a nonempty matrix')
        rows=shape[0]
        # The estimate is decoded storage, or the affine packed payload. Respect
        # an explicitly smaller row_chunk while shrinking large tiles to both
        # budgets before loading any bytes.
        row_bytes=max(1,(size+rows-1)//rows)
        tile_rows=min(self.cfg.row_chunk,budget//row_bytes)
        if tile_rows<1:raise MemoryError('One output projection row exceeds max_tensor_bytes or io_bytes')
        # Synchronous output tiles need the same IO headroom as a demand read.
        # Discard unused lookahead before creating another decoded buffer.
        self.io.drain()
        # Keep only tile result vectors; each full matrix tile is evaluated and
        # released before the next is read. No lazy graph retains the whole head.
        out=[]
        for start in range(0,rows,tile_rows):
            w=self._load(key,rows=(start,min(rows,start+tile_rows)))
            n=size_of(w)
            if n>budget:raise MemoryError('Output projection tile exceeded its byte budget')
            self.stats['output_tiles']=self.stats.get('output_tiles',0)+1
            self.stats['output_tile_peak_bytes']=max(self.stats.get('output_tile_peak_bytes',0),n)
            y=self.linear_weight(x,w);self.mx.eval(y);out.append(y)
            del w
        return self.mx.concatenate(out,axis=-1)

    def close(self):
        try:self.io.close()
        finally:self.cache.clear()
