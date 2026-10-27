"""Opt-in packed multi-token kernels; no new checkpoint representation.

TensorOps uses strict FP32 operands/accumulation and a 4 KiB threadgroup
tile. The register-reuse alternative retains the matvec lane reduction.
Neither is a speed or real-checkpoint fidelity claim. A measured dispatcher
retains the old kernel when compilation, numerical or timing checks fail.
"""
from functools import lru_cache
from statistics import median
from time import perf_counter

import numpy as np

from .gguf_metal import _Q4, _Q6


def kernel_source(kind, backend, grouped=False, token_tile=4):
    if kind not in (12, 14) or backend not in ('reuse', 'tensorops'):
        raise ValueError('Expected Q4_K/Q6_K and reuse/tensorops')
    if type(grouped) is not bool or type(token_tile) is not int or token_tile not in (2, 4, 8, 16):
        raise ValueError('Invalid group mode or token tile')
    dims = ("const uint rows=packed_shape[1], stride=packed_shape[2], batches=x_shape[1];"
            "const uint expert=threadgroup_position_in_grid.z;" if grouped else
            "const uint rows=packed_shape[0], stride=packed_shape[1], batches=x_shape[0];"
            "const uint expert=0;")
    common = dims + '\nconst uint width=x_shape[x_ndim-1];\n'
    if backend == 'reuse':
        code = common + r'''
        const uint row=threadgroup_position_in_grid.x;
        const uint lane=thread_index_in_simdgroup;
        const uint base=threadgroup_position_in_grid.y*TOKEN_TILE;
        float sums[TOKEN_TILE]={0};
        for(uint j=lane;j<width;j+=32){
            const uint k=j%256;
            const device uint8_t* p=packed+(expert*rows+row)*stride+(j/256)*BLOCK_BYTES;
            DECODE
            #pragma unroll
            for(uint t=0;t<TOKEN_TILE;++t)
                if(base+t<batches) sums[t]+=weight*float(x[(expert*batches+base+t)*width+j]);
        }
        #pragma unroll
        for(uint t=0;t<TOKEN_TILE;++t){
            const float value=simd_sum(sums[t]);
            if(lane==0 && base+t<batches) out[(expert*batches+base+t)*rows+row]=value;
        }
        '''
        header = '#pragma clang fp contract(off)\n'
    else:
        code = common + r'''
        const uint lane=thread_index_in_simdgroup;
        const uint row0=threadgroup_position_in_grid.x*16;
        const uint batch0=threadgroup_position_in_grid.y*16;
        threadgroup float a[16*32], b[16*32];
        using extents_t=metal::dextents<int32_t,2>;
        using tensor_t=metal::tensor<threadgroup float,extents_t,metal::tensor_inline>;
        tensor_t ta(a,extents_t(32,16),metal::array<int32_t,2>{1,32});
        tensor_t tb(b,extents_t(32,16),metal::array<int32_t,2>{1,32});
        constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,16,32,false,true,false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
        auto acc=op.get_destination_cooperative_tensor<tensor_t,tensor_t,float>();
        #pragma unroll
        for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;
        for(uint kb=0;kb<width;kb+=32){
            for(uint index=lane;index<16*32;index+=32){
                const uint r=index/32, j=kb+index%32;
                a[index]=batch0+r<batches ? float(x[(expert*batches+batch0+r)*width+j]) : 0.0f;
                float value=0.0f;
                if(row0+r<rows){
                    const uint k=j%256;
                    const device uint8_t* p=packed+(expert*rows+row0+r)*stride+(j/256)*BLOCK_BYTES;
                    DECODE
                    value=weight;
                }
                b[index]=value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(ta,tb,acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)){
            const auto ij=acc.get_multidimensional_index(i);
            const uint row=row0+ij[0], batch=batch0+ij[1];
            if(row<rows && batch<batches) out[(expert*batches+batch)*rows+row]=acc[i];
        }
        '''
        header = ('#include <metal_tensor>\n'
                  '#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n'
                  '#pragma clang fp contract(off)\n')
    return header, code.replace('TOKEN_TILE', str(token_tile)).replace('BLOCK_BYTES', str(144 if kind == 12 else 210)).replace('DECODE', _Q4 if kind == 12 else _Q6)


@lru_cache(maxsize=32)
def _kernel(kind, backend, grouped, token_tile):
    import mlx.core as mx
    header, code = kernel_source(kind, backend, grouped, token_tile)
    return mx.fast.metal_kernel(name=f'airllm_m5_{backend}_q{kind}_{int(grouped)}_{token_tile}',
        input_names=['packed', 'x'], output_names=['out'], source=code, header=header,
        compile_options={'math_mode': 'safe'})


def experimental_matmul(packed, x, kind, *, backend, grouped=False, token_tile=4):
    """Internal device-array operation. Public caller validates/budgets inputs."""
    import mlx.core as mx
    rows = packed.shape[-2]
    batch = x.shape[-2]
    experts = packed.shape[0] if grouped else 1
    tile = 16 if backend == 'tensorops' else token_tile
    groups = (rows + 15)//16 if backend == 'tensorops' else rows
    return _kernel(kind, backend, grouped, token_tile)(inputs=[packed, x],
        grid=(groups*32, (batch+tile-1)//tile, experts), threadgroup=(32,1,1),
        output_shapes=[(experts,batch,rows) if grouped else (batch,rows)],
        output_dtypes=[mx.float32])[0]


class M5BatchDispatcher:
    """Calibrate each shape on real operands, preserving fallback by default.

    Compilation/warmup and all comparisons belong to request latency. Only
    warm synchronous paired kernel timings select a backend. Host timing
    includes waits; it is never reported as pure GPU time. An accepted shape
    still requires the independent end-to-end real-checkpoint verifier.
    """
    def __init__(self, backend, stats, *, token_tile=4, minimum_speedup=1.03,
                 workspace_bytes=None, guard=None):
        if backend not in ('reuse','tensorops'):
            raise ValueError('Unknown experimental batch backend')
        if type(token_tile) is not int or token_tile not in (2,4,8,16):
            raise ValueError('Invalid token tile')
        if not np.isfinite(minimum_speedup) or minimum_speedup < 1:
            raise ValueError('Invalid minimum speedup')
        if (workspace_bytes is None) != (guard is None):
            raise ValueError('Workspace bytes and resource guard must be supplied together')
        if workspace_bytes is not None and (type(workspace_bytes) is not int or
                workspace_bytes < 1 or not callable(guard)):
            raise ValueError('Expected a positive workspace byte budget and callable resource guard')
        self.backend=backend; self.stats=stats; self.token_tile=token_tile
        self.minimum_speedup=minimum_speedup; self.decisions={}; self.records=[]
        self.workspace_bytes=workspace_bytes; self.guard=guard

    def __call__(self, packed, x, kind, *, grouped, baseline, spare_workspace):
        import mlx.core as mx
        if x.shape[-2] < 2:
            self.stats['m5_fallback_single_token']+=1
            return baseline()
        # Reference/trial, two possibly copied host comparisons, a live timed
        # output and its replacement are charged before calibration. NumPy
        # conversion need not be zero-copy. Never rely on that for admission.
        extra=6*(x.size//x.shape[-1])*packed.shape[-2]*4+4096
        if spare_workspace < extra:
            self.stats['m5_fallback_workspace']+=1
            return baseline()
        key=(kind,grouped,tuple(packed.shape),tuple(x.shape))
        candidate=lambda: experimental_matmul(packed,x,kind,backend=self.backend,
            grouped=grouped,token_tile=self.token_tile)
        def resources_admitted():
            if self.guard is None:
                return True
            # The caller may cap spare space by the complete pair/pipeline
            # working set. Charge that baseline plus candidate temporaries to
            # both the workspace and the current system reserve before use.
            if not 0 <= spare_workspace <= self.workspace_bytes:
                raise ValueError('Spare workspace is outside the configured budget')
            try:
                self.guard(self.workspace_bytes-spare_workspace+extra)
            except MemoryError:
                self.stats['m5_fallback_resources']+=1
                return False
            return True
        if key not in self.decisions:
            if not resources_admitted():
                return baseline()
            started=perf_counter(); record={'shape':repr(key),'backend':self.backend}
            # Baseline errors are never disguised as candidate unavailability.
            reference=baseline(); mx.eval(reference)
            try:
                trial=candidate(); mx.eval(trial)
            except (RuntimeError,ValueError,TypeError) as error:
                self.decisions[key]=False; record.update(selected=False,reason='unavailable',error=str(error))
            else:
                ref=np.asarray(reference); got=np.asarray(trial)
                valid=(ref.shape==got.shape and np.isfinite(ref).all() and np.isfinite(got).all()
                    and np.allclose(ref,got,atol=1e-4,rtol=1e-4))
                record['numerically_accepted_sample']=bool(valid)
                if valid:
                    times=[[],[]]
                    timing_error=None
                    for iteration in range(4):
                        for index in ((0,1) if iteration%2==0 else (1,0)):
                            before=perf_counter()
                            if index==0:
                                value=baseline(); mx.eval(value)
                            else:
                                try:
                                    value=candidate(); mx.eval(value)
                                except (RuntimeError,ValueError,TypeError) as error:
                                    timing_error=str(error)
                                    value=None
                                    break
                            times[index].append(perf_counter()-before)
                            del value
                        if timing_error is not None:
                            break
                    record.update(baseline_seconds=times[0],candidate_seconds=times[1])
                    selected=(timing_error is None and
                        median(times[1])*self.minimum_speedup < median(times[0]))
                    record.update(selected=selected,reason=('unavailable_during_timing' if timing_error is not None
                        else 'measured_speed' if selected else 'not_faster'))
                    if timing_error is not None:
                        record['error']=timing_error
                    self.decisions[key]=selected
                else:
                    self.decisions[key]=False; record.update(selected=False,reason='numerical_rejection')
                del ref,got,trial
            record['calibration_wall_seconds']=perf_counter()-started
            self.stats['m5_calibration_seconds']+=record['calibration_wall_seconds']
            self.records.append(record)
            del reference
        if self.decisions[key]:
            # MLX compilation/execution failures can be lazy. Synchronize here
            # so candidate failure can still use the original operation. This
            # synchronization is part of the experimental path's real cost.
            if not resources_admitted():
                return baseline()
            try:
                result=candidate(); mx.eval(result)
            except (RuntimeError,ValueError,TypeError) as error:
                result=None
                self.decisions[key]=False
                self.records.append({'shape':repr(key),'backend':self.backend,
                    'selected':False,'reason':'runtime_unavailable','error':str(error)})
            else:
                self.stats['m5_'+self.backend+'_calls']+=1
                return result
        self.stats['m5_fallback_calls']+=1
        return baseline()
