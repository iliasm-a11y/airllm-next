"""One shared cache for decoded reference or MLX affine packed matrices.

Embedding rows are read selectively. The output head is tiled. No full model
skeleton or full expert bank is ever allocated by this backend.
"""
from dataclasses import dataclass
import math
from threading import RLock
from time import perf_counter
import numpy as np
from .quantization import FP8_LUT, PackedMatrix
from .cache import WeightCache, SingleFlight, size_of
from .compact_cache import CompactWeightCache, backing_bytes
from .adaptive_cache import AdaptiveWeightCache
from .residency_cache import ResidencyWeightCache, RESIDENCY_POLICIES
from .specprefetch import SpecPrefetch


class NativeWeights:
    def __init__(self,checkpoint,mx,config,model_config,sidecar=None):
        self.cp=checkpoint;self.mx=mx;self.cfg=config;self.model_config=model_config;self.sidecar=sidecar;self.prefetch_enabled=config.prefetch
        # Both tiers share the original weight budget. Retaining encoded source
        # data is opt-in: it trades decoded residency for more compact reuse and
        # may lose on low-reuse or conversion-bound workloads.
        compact_bytes=config.compact_cache_bytes
        cache_type=(ResidencyWeightCache if config.policy in RESIDENCY_POLICIES else
                    AdaptiveWeightCache if config.policy=='adaptive' else WeightCache)
        self.cache=cache_type(config.cache_bytes-compact_bytes,config.policy)
        self.compact_cache=CompactWeightCache(compact_bytes)
        self.io=SingleFlight(config.io_bytes,config.io_workers)
        self.spec=(SpecPrefetch(self.io,config.spec_prefetch_budget,
                  max_predictions=config.spec_prefetch_experts) if config.spec_prefetch else None)
        self.dtype=getattr(mx,config.dtype);self.stats={'dequant_seconds':0.,'loads':0,'speculative_reads':0}
        self._stats_lock=RLock()
        self.profile_stages=config.profile_io
        if self.profile_stages:
            # Worker wall-time sums can overlap compute and other workers. They
            # attribute load cost; they are not additive generation/GPU timings.
            self.stats.update({'source_read_seconds':0.,'source_read_calls':0,
                               'materialize_seconds':0.,'materialize_calls':0})
        self.block=tuple(model_config.get('quantization_config',{}).get('weight_block_size',[128,128]))
        self.lut=mx.array(FP8_LUT)

    @staticmethod
    def identity(key,expert=None):return key if expert is None else f'{key}#{expert}'

    def shape(self,key,expert=None):
        shape=self.cp.tensors[key]['shape']
        if expert is not None:
            if (not isinstance(expert,(int,np.integer)) or isinstance(expert,(bool,np.bool_)) or
                len(shape)!=3 or not 0<=expert<shape[0]):raise ValueError('Invalid packed expert id')
            return tuple(shape[1:])
        return tuple(shape)

    def estimate(self,key,expert=None):
        ident=self.identity(key,expert)
        if self.sidecar and ident in self.sidecar.entries:return self.sidecar.entries[ident]['nbytes']
        d=self.cp.tensors[key];n=math.prod(self.shape(key,expert))
        return n*(4 if self.cfg.dtype=='float32' or d['dtype']=='F32' else 2)

    def _profile_call(self,stage,function,*args,**kwargs):
        start=perf_counter();failed=False
        try:return function(*args,**kwargs)
        except BaseException:
            failed=True;raise
        finally:
            elapsed=perf_counter()-start
            with self._stats_lock:
                self.stats[stage+'_seconds']+=elapsed
                self.stats[stage+'_calls']+=1
                if failed:
                    name=stage+'_failed_calls'
                    self.stats[name]=self.stats.get(name,0)+1

    def _array(self,a,dtype=None,*,bfloat16_bits=False):
        if self.profile_stages:
            return self._profile_call('materialize',self._array_value,a,dtype,bfloat16_bits=bfloat16_bits)
        return self._array_value(a,dtype,bfloat16_bits=bfloat16_bits)

    def _array_value(self,a,dtype=None,*,bfloat16_bits=False):
        # Evaluation on CPU ensures data has been read before the Future resolves;
        # on Apple these evaluated buffers are shared with Metal, not copied to VRAM.
        # Stream contexts also change MLX's process-global default device.
        # A worker context would move concurrent main-thread GPU operations
        # onto CPU. Use explicit streams without mutating that shared default.
        v=self._import_array(a) if self.cfg.zero_copy_weights else self.mx.array(a)
        if bfloat16_bits:v=v.view(self.mx.bfloat16,stream=self.mx.cpu)
        if dtype is not None:v=v.astype(dtype,stream=self.mx.cpu)
        self.mx.eval(v)
        return v

    def _import_array(self,a):
        """Adopt an exact source allocation, with a portable copy fallback.

        MLX retains the NumPy exporter for the lifetime of the imported buffer.
        Only full contiguous buffers qualify: a small coalesced view must not
        hide a larger retained allocation from the decoded-cache/IO accounting.
        Offset and lossless readers create privately owned, unrecycled buffers.
        No caller-owned tensor or checkpoint mapping is modified.
        """
        eligible=isinstance(a,np.ndarray) and a.flags.c_contiguous and a.nbytes>0
        if eligible:
            try:eligible=backing_bytes((a,))==a.nbytes
            except (TypeError,ValueError):eligible=False
        if eligible:
            importer=getattr(self.mx,'from_dlpack',None)
            if importer is not None:
                try:
                    value=importer(a,copy=False)
                except (BufferError,TypeError,ValueError,RuntimeError):
                    # copy=False may be unavailable, or the platform allocator
                    # may reject the host pointer/alignment. Exact values win.
                    pass
                else:
                    with self._stats_lock:
                        self.stats['zero_copy_imports']=self.stats.get('zero_copy_imports',0)+1
                        self.stats['zero_copy_import_bytes']=self.stats.get('zero_copy_import_bytes',0)+a.nbytes
                    return value
        with self._stats_lock:
            self.stats['zero_copy_fallbacks']=self.stats.get('zero_copy_fallbacks',0)+1
        return self.mx.array(a)

    def _load(self,key,expert=None,rows=None):
        start=perf_counter()
        value=self._load_value(key,expert,rows)
        elapsed=perf_counter()-start
        if rows is None:self.cache.record_load(self.identity(key,expert),elapsed)
        with self._stats_lock:
            self.stats['loads']+=1
            self.stats.setdefault('load_seconds',0.)
            self.stats['load_seconds']+=elapsed
        return value

    def _load_value(self,key,expert=None,rows=None):
        shape=self.shape(key,expert)
        if rows is not None:
            if len(shape)!=2 or len(rows)!=2:raise ValueError('Invalid matrix row range')
            if (any(not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) for value in rows)
                    or not 0<=rows[0]<=rows[1]<=shape[0]):raise ValueError('Invalid matrix row range')
            rows=tuple(int(value) for value in rows)
        ident=self.identity(key,expert)
        if self.sidecar and ident in self.sidecar.entries:
            if self.profile_stages:
                p=self._profile_call('source_read',self.sidecar.load,ident,rows=rows,max_bytes=self.cfg.max_tensor_bytes)
            else:p=self.sidecar.load(ident,rows=rows,max_bytes=self.cfg.max_tensor_bytes)
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
            cache_key=('fp8',key,expert,rows,scale_rows)
            source=self.compact_cache.get(cache_key) if self.compact_cache.capacity else None
            if source is None:
                if self.profile_stages:
                    source=self._profile_call('source_read',self.cp.read_arrays,requests,mode=self.cfg.io_mode,
                        max_batch_bytes=self.cp.max_read_bytes,coalesce=self.cfg.coalesce_reads)
                else:
                    source=self.cp.read_arrays(requests,mode=self.cfg.io_mode,max_batch_bytes=self.cp.max_read_bytes,
                                               coalesce=self.cfg.coalesce_reads)
                self.compact_cache.put(cache_key,source)
            a,scales=source
            t=perf_counter()
            if self.cp.tensors[sk]['dtype']=='BF16':scales=(scales.astype(np.uint32)<<16).view(np.float32)
            offset=rows[0] if rows else 0;mx=self.mx
            # Decode on CPU stream while GPU executes a previous expert. The raw
            # FP8 buffer dies after evaluation. Cached result is reused directly.
            if self.cfg.zero_copy_weights:raw=self._import_array(a);s=self._import_array(scales)
            else:raw=mx.array(a);s=mx.array(scales)
            ri=mx.subtract(mx.floor_divide(mx.add(mx.arange(a.shape[0],stream=mx.cpu),offset,
                stream=mx.cpu),br,stream=mx.cpu),scale_rows[0] if scale_rows else 0,stream=mx.cpu)
            ci=mx.floor_divide(mx.arange(a.shape[1],stream=mx.cpu),bc,stream=mx.cpu)
            # Explicit take calls replace indexing syntax, whose implicit
            # gather operations would otherwise follow the shared device.
            decoded=mx.take(self.lut,raw.astype(mx.int32,stream=mx.cpu),stream=mx.cpu)
            selected_scales=mx.take(mx.take(s,ri,axis=0,stream=mx.cpu),ci,axis=1,stream=mx.cpu)
            v=mx.multiply(decoded,selected_scales,stream=mx.cpu).astype(self.dtype,stream=mx.cpu)
            mx.eval(v)
            elapsed=perf_counter()-t
            with self._stats_lock:
                self.stats['dequant_seconds']+=elapsed
                if self.profile_stages:
                    # FP8 lookup/scale expansion includes import and evaluation.
                    # dequant_seconds is a subset of materialize_seconds.
                    self.stats['materialize_seconds']+=elapsed
                    self.stats['materialize_calls']+=1
        elif d['dtype']=='BF16':
            a=self._source_array(key,expert,rows)
            v=self._array(a,self.dtype,bfloat16_bits=True)
        else:
            a=self._source_array(key,expert,rows)
            # Router scores, normalization weights and correction bias stay fp32.
            target=self.mx.float32 if d['dtype']=='F32' else self.dtype
            v=self._array(a,target)
        return v

    def _source_array(self,key,expert,rows):
        cache_key=('array',key,expert,rows)
        source=self.compact_cache.get(cache_key) if self.compact_cache.capacity else None
        if source is None:
            if self.profile_stages:
                value=self._profile_call('source_read',self.cp.read_array,key,expert=expert,rows=rows,mode=self.cfg.io_mode)
            else:value=self.cp.read_array(key,expert=expert,rows=rows,mode=self.cfg.io_mode)
            source=(value,)
            self.compact_cache.put(cache_key,source)
        return source[0]

    def trim(self):
        """Reduce both tiers under observed system pressure, never grow them."""
        self.cache.resize(self.cache.capacity//2)
        self.compact_cache.resize(self.compact_cache.capacity//2)

    def drain(self):
        # Predictions are optional: discard their failures before exact-reader
        # shutdown/pressure handling can propagate an unrelated speculative error.
        if self.spec:self.spec.discard()
        self.io.drain()

    def begin_layer(self,layer,expert_keys):
        """Read only a predicted first matrix while attention runs, never route."""
        if not self.spec:return
        for expert in self.spec.predict(layer):
            key,packed_expert=expert_keys(expert)[0]
            ident=self.identity(key,packed_expert)
            if self.cache.contains(ident):continue
            size=self.estimate(key,packed_expert)
            if size<=self.cfg.max_tensor_bytes:
                with self.cache.lock:load_seconds=self.cache.costs.get(ident)
                self.spec.submit(layer,expert,ident,size,
                    lambda k=key,e=packed_expert:self._load(k,e),load_seconds=load_seconds,
                    scheduling_seconds=self.stats.get('spec_demand_overhead_seconds'))

    def observe_layer(self,layer,experts):
        if self.spec:self.spec.observe(layer,experts)

    def observe_batch(self,expert_keys):
        """Prioritize an actual routed union without reserving extra weights."""
        if self.cfg.policy=='batch-aware':
            self.cache.set_batch(self.identity(key,expert) for key,expert in expert_keys)

    def clear_batch(self):
        if self.cfg.policy=='batch-aware':self.cache.clear_batch()

    def get(self,key,expert=None):
        ident=self.identity(key,expert);v=self.cache.get(ident,self.estimate(key,expert))
        if v is not None:return v
        n=self.estimate(key,expert)
        if n>self.cfg.max_tensor_bytes:raise MemoryError(f'{key} must be read as row tiles ({n} bytes)')
        v=self.spec.take(ident) if self.spec else None
        if v is None:v=self.io.take(ident)
        if v is None:
            demand_start=perf_counter() if self.spec else None
            # Main thread is serialized by model's lock. submit() deduplicates
            # if another prediction already scheduled precisely this matrix.
            if not self.io.submit(ident,n,lambda:self._load(key,expert)):
                # Wait/discard stale lookahead before synchronous demand to avoid
                # exceeding the IO budget with a hidden extra decoded matrix.
                self.drain()
                if not self.io.submit(ident,n,lambda:self._load(key,expert)):
                    if n>self.cfg.io_bytes:raise MemoryError('Increase io_bytes for the largest decoded matrix')
                    raise RuntimeError('Could not reserve demand read')
            v=self.io.take(ident)
            if demand_start is not None:
                elapsed=perf_counter()-demand_start
                with self.cache.lock:load_cost=self.cache.costs.get(ident)
                if load_cost is not None:
                    # The exact IO roundtrip includes queueing and completion
                    # handoff, unlike worker-dispatch latency alone. Subtract
                    # the same key's observed load EWMA to estimate scheduling
                    # overhead; this is a gate hint, not a hardware measurement.
                    overhead=max(0.,elapsed-load_cost)
                    previous=self.stats.get('spec_demand_overhead_seconds')
                    self.stats['spec_demand_overhead_seconds']=(overhead if previous is None
                                                              else previous*.75+overhead*.25)
                    self.stats['spec_demand_overhead_samples']=self.stats.get('spec_demand_overhead_samples',0)+1
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
        self.drain()
        if self.cfg.output_prefetch and self.cfg.io_bytes//2>=row_bytes:
            # The currently used tile and the next Future together fit in the
            # original I/O allowance. No extra double-buffer memory is granted.
            pipeline_rows=min(self.cfg.row_chunk,self.cfg.max_tensor_bytes//row_bytes,
                              (self.cfg.io_bytes//2)//row_bytes)
            return self._output_pipeline(x,key,rows,pipeline_rows,row_bytes)
        # Keep only tile result vectors; each full matrix tile is evaluated and
        # released before the next is read. No lazy graph retains the whole head.
        out=[]
        for start in range(0,rows,tile_rows):
            selection=(start,min(rows,start+tile_rows))
            tile_key=('output',key,*selection)
            estimate=row_bytes*(selection[1]-selection[0])
            w=self.cache.get(tile_key,estimate) if self.cfg.output_tile_cache else None
            if w is None:
                started=perf_counter()
                w=self._load(key,rows=selection)
                if self.cfg.output_tile_cache:
                    self.cache.record_load(tile_key,perf_counter()-started)
                    self.cache.put(tile_key,w)
            else:self.stats['output_tile_cache_hits']=self.stats.get('output_tile_cache_hits',0)+1
            n=size_of(w)
            if n>budget:raise MemoryError('Output projection tile exceeded its byte budget')
            self.stats['output_tiles']=self.stats.get('output_tiles',0)+1
            self.stats['output_tile_peak_bytes']=max(self.stats.get('output_tile_peak_bytes',0),n)
            y=self.linear_weight(x,w);self.mx.eval(y);out.append(y)
            del w
        return self.mx.concatenate(out,axis=-1)

    def _output_pipeline(self,x,key,rows,tile_rows,row_bytes):
        """Overlap an exact next-tile read/decode with current Metal work."""
        out=[]
        def load_tile(selection):
            begin=perf_counter()
            value=self._load(key,rows=selection)
            if self.cfg.output_tile_cache:
                self.cache.record_load(('output',key,*selection),perf_counter()-begin)
            return value
        try:
            for start in range(0,rows,tile_rows):
                selection=(start,min(rows,start+tile_rows))
                tile_key=('output',key,*selection)
                io_key=('output-io',key,*selection)
                estimate=row_bytes*(selection[1]-selection[0])
                w=self.cache.get(tile_key,estimate) if self.cfg.output_tile_cache else None
                if w is None:
                    w=self.io.take(io_key)
                    if w is None:w=load_tile(selection)
                    if self.cfg.output_tile_cache:self.cache.put(tile_key,w)
                else:self.stats['output_tile_cache_hits']=self.stats.get('output_tile_cache_hits',0)+1
                n=size_of(w)
                if n>min(self.cfg.max_tensor_bytes,self.cfg.io_bytes//2):
                    raise MemoryError('Output pipeline tile exceeded its double-buffer allowance')
                next_start=selection[1]
                if next_start<rows:
                    next_selection=(next_start,min(rows,next_start+tile_rows))
                    next_cache_key=('output',key,*next_selection)
                    if not self.cfg.output_tile_cache or not self.cache.contains(next_cache_key):
                        ok=self.io.submit(('output-io',key,*next_selection),
                            row_bytes*(next_selection[1]-next_selection[0]),
                            lambda r=next_selection:load_tile(r))
                        if ok:self.stats['output_pipeline_submitted']=self.stats.get('output_pipeline_submitted',0)+1
                self.stats['output_pipeline_peak_payload_bytes']=max(
                    self.stats.get('output_pipeline_peak_payload_bytes',0),n+self.io.bytes)
                self.stats['output_tiles']=self.stats.get('output_tiles',0)+1
                self.stats['output_tile_peak_bytes']=max(self.stats.get('output_tile_peak_bytes',0),n)
                y=self.linear_weight(x,w);self.mx.eval(y);out.append(y)
                del w
            return self.mx.concatenate(out,axis=-1)
        except BaseException:
            try:self.drain()
            except BaseException:
                # Cleanup still releases every reservation. Preserve the actual
                # compute/read failure if the next asynchronous tile also fails.
                self.stats['output_pipeline_cleanup_failures']=self.stats.get('output_pipeline_cleanup_failures',0)+1
            raise

    def close(self):
        try:
            try:
                if self.spec:self.spec.close()
            finally:self.io.close()
        finally:
            self.cache.clear()
            self.compact_cache.clear()
