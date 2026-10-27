"""Experimental fused Q4_K/Q6_K matvec: packed GGUF bytes stay packed.

Block layouts follow ggml-quants.c (MIT, ggml authors). One SIMD group sums
one output row in FP32. No full floating matrix is allocated. This is a decode
kernel, not an optimized large-batch GEMM. Numerical reduction order differs
from BLAS; acceptance requires independent tolerance and greedy-token checks.
"""
from functools import lru_cache
import numpy as np


_Q4 = r'''
    const uint group = k / 32;
    const uint s = group < 4 ? (p[4 + group] & 63) :
        ((p[4 + group + 4] & 15) | ((p[4 + group - 4] >> 6) << 4));
    const uint m = group < 4 ? (p[4 + group + 4] & 63) :
        ((p[4 + group + 4] >> 4) | ((p[4 + group] >> 6) << 4));
    const uint q = (p[16 + (k / 64) * 32 + k % 32] >> ((group % 2) * 4)) & 15;
    const float d = float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
    const float dm = float(as_type<half>(ushort(ushort(p[2]) | (ushort(p[3]) << 8))));
    const float weight = (d * float(s)) * float(q) - dm * float(m);
'''
_Q6 = r'''
    const uint h = k / 128, part = (k % 128) / 32, lane = k % 32;
    const uint low = p[h * 64 + (part % 2) * 32 + lane];
    const uint high = p[128 + h * 32 + lane];
    const int q = int(((low >> ((part / 2) * 4)) & 15) |
                      (((high >> (part * 2)) & 3) << 4)) - 32;
    const int s = int(as_type<char>(p[192 + k / 16]));
    const float d = float(as_type<half>(ushort(ushort(p[208]) | (ushort(p[209]) << 8))));
    const float weight = (d * float(s)) * float(q);
'''


def _import_packed(mx,packed,zero_copy,stats):
    if not isinstance(packed,np.ndarray):return packed
    if zero_copy:
        from .compact_cache import backing_bytes
        # Full owner only: a tiny slice must not retain an uncharged large
        # backing allocation. Unsupported/readonly exporters copy exactly.
        if packed.flags.c_contiguous and backing_bytes((packed,))==packed.nbytes:
            importer=getattr(mx,'from_dlpack',None)
            if importer is not None:
                try:value=importer(packed,copy=False)
                except (BufferError,TypeError,ValueError,RuntimeError):pass
                else:
                    if stats is not None:stats['packed_zero_copy_imports']=stats.get('packed_zero_copy_imports',0)+1
                    return value
        if stats is not None:stats['packed_zero_copy_fallbacks']=stats.get('packed_zero_copy_fallbacks',0)+1
    return mx.array(packed)


@lru_cache(maxsize=4)
def _kernel(quant_type, grouped=False):
    import mlx.core as mx
    if quant_type not in (12, 14):
        raise NotImplementedError('Fused Metal supports Q4_K and Q6_K only')
    block_bytes = 144 if quant_type == 12 else 210
    code = r'''
        const uint row = thread_position_in_grid.x / 32;
        const uint lane_id = thread_position_in_grid.x % 32;
        const uint batch = thread_position_in_grid.y;
        const uint width = x_shape[x_ndim - 1];
        const uint rows = packed_shape[0];
        if (row >= rows) return;
        float sum = 0;
        for (uint j = lane_id; j < width; j += 32) {
            const uint k = j % 256;
            const device uint8_t* p = packed + row * packed_shape[1] + (j / 256) * BLOCK_BYTES;
            DECODE
            sum += weight * float(x[batch * width + j]);
        }
        sum = simd_sum(sum);
        if (lane_id == 0) out[batch * rows + row] = sum;
    '''.replace('BLOCK_BYTES', str(block_bytes)).replace('DECODE', _Q4 if quant_type == 12 else _Q6)
    if grouped:
        # Each SIMD group retains the original lane order and FP32 sum. Only
        # independent expert matrices are dispatched together; no expert-axis
        # reduction or different arithmetic is introduced here.
        code=code.replace('const uint rows = packed_shape[0];',
            'const uint rows = packed_shape[1];\n'
            'const uint expert = thread_position_in_grid.z;\n'
            'const uint batches = x_shape[1];')
        code=code.replace('packed + row * packed_shape[1]',
            'packed + (expert * rows + row) * packed_shape[2]')
        code=code.replace('x[batch * width + j]', 'x[(expert * batches + batch) * width + j]')
        code=code.replace('out[batch * rows + row]', 'out[(expert * batches + batch) * rows + row]')
    return mx.fast.metal_kernel(name=f'airllm_gguf_q{quant_type}_matvec'+('_experts' if grouped else ''),
                               input_names=['packed', 'x'], output_names=['out'],
                               source=code, header='#pragma clang fp contract(off)\n')


def packed_matmul(packed, x, quant_type, *, max_workspace_bytes=64 * 1024**2, synchronize=True,zero_copy=False,stats=None,simd_groups=1,batch_dispatch=None,batch_spare_workspace_bytes=None):
    """Multiply [..., width] by packed [rows, row_bytes].T; caller tiles reads."""
    import mlx.core as mx
    if type(synchronize) is not bool:raise ValueError('synchronize must be boolean')
    if type(zero_copy) is not bool:raise ValueError('zero_copy must be boolean')
    if type(simd_groups) is not int or simd_groups not in (1,2,4,8):raise ValueError('simd_groups must be 1, 2, 4 or 8')
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 1:
        raise ValueError('max_workspace_bytes must be a positive integer')
    if batch_spare_workspace_bytes is not None and (type(batch_spare_workspace_bytes) is not int or batch_spare_workspace_bytes<0):
        raise ValueError('Batch spare workspace must be a nonnegative integer or None')
    if quant_type not in (12, 14):
        raise NotImplementedError('Fused Metal supports Q4_K and Q6_K only')
    if not mx.metal.is_available():
        raise RuntimeError('Metal is required for the fused packed kernel')
    if len(packed.shape) != 2 or packed.dtype not in (np.dtype('uint8'), mx.uint8):
        raise ValueError('Packed rows must be uint8 [rows, row_bytes]')
    if x.ndim < 1 or x.size == 0 or x.shape[-1] < 1 or x.shape[-1] % 256:
        raise ValueError('Input width must be a positive multiple of 256')
    width = x.shape[-1]
    block_bytes = 144 if quant_type == 12 else 210
    rows = packed.shape[0]
    if rows < 1 or packed.shape[1] != width // 256 * block_bytes:
        raise ValueError('Packed row size does not match input width')
    batch = x.size // width
    # Host packed buffer + MLX copy + contiguous FP32 input/output. MLX's own
    # limit and the externally sampled process footprint cover runtime overhead.
    required = 2 * packed.nbytes + x.size * 4 + batch * rows * 4
    if required > max_workspace_bytes:
        raise MemoryError('Packed matvec exceeds workspace before device import')
    data = _import_packed(mx,packed,zero_copy,stats)
    source = mx.contiguous(x.astype(mx.float32)).reshape(batch, width)
    def baseline():
        return _kernel(quant_type)(inputs=[data, source], grid=(rows * 32, batch, 1),
                                threadgroup=(32*simd_groups, 1, 1),
                                output_shapes=[(batch, rows)], output_dtypes=[mx.float32])[0]
    spare=max_workspace_bytes-required
    if batch_spare_workspace_bytes is not None:spare=min(spare,batch_spare_workspace_bytes)
    result = (baseline() if batch_dispatch is None else batch_dispatch(data,source,quant_type,
        grouped=False,baseline=baseline,spare_workspace=spare))
    if synchronize:mx.eval(result)
    return result.reshape(*x.shape[:-1], rows)


def packed_expert_matmul(packed, x, quant_type, *, max_workspace_bytes=64 * 1024**2,synchronize=True,zero_copy=False,stats=None,simd_groups=1,batch_dispatch=None,batch_spare_workspace_bytes=None):
    """Independent expert matvecs, preserving each original SIMD reduction.

    ``packed`` is uint8 [experts, rows, row_bytes], ``x`` is FP32
    [experts, batch, width]. Temporary packed host+MLX copies are charged before
    import; expert selection and the weighted output reduction stay in caller.
    """
    import mlx.core as mx
    if type(synchronize) is not bool:raise ValueError('synchronize must be boolean')
    if type(zero_copy) is not bool:raise ValueError('zero_copy must be boolean')
    if type(simd_groups) is not int or simd_groups not in (1,2,4,8):raise ValueError('simd_groups must be 1, 2, 4 or 8')
    if type(max_workspace_bytes) is not int or max_workspace_bytes<1:
        raise ValueError('max_workspace_bytes must be a positive integer')
    if batch_spare_workspace_bytes is not None and (type(batch_spare_workspace_bytes) is not int or batch_spare_workspace_bytes<0):
        raise ValueError('Batch spare workspace must be a nonnegative integer or None')
    if quant_type not in (12,14):raise NotImplementedError('Fused Metal supports Q4_K and Q6_K only')
    if not mx.metal.is_available():raise RuntimeError('Metal is required for the fused packed kernel')
    if len(packed.shape)!=3 or packed.dtype not in (np.dtype('uint8'),mx.uint8):
        raise ValueError('Packed experts must be uint8 [experts, rows, row_bytes]')
    if x.ndim!=3 or x.size==0 or x.shape[-1]<1 or x.shape[-1]%256:
        raise ValueError('Expert input must be [experts, batch, width] with width a positive multiple of 256')
    experts,rows,row_bytes=packed.shape;width=x.shape[-1];batch=x.shape[1]
    if experts<1 or rows<1 or x.shape[0]!=experts or row_bytes!=width//256*(144 if quant_type==12 else 210):
        raise ValueError('Packed expert shape does not match input')
    required=2*packed.nbytes+x.size*4+experts*batch*rows*4
    if required>max_workspace_bytes:raise MemoryError('Packed expert matvec exceeds workspace before device import')
    data=_import_packed(mx,packed,zero_copy,stats)
    source=mx.contiguous(x.astype(mx.float32))
    def baseline():
        return _kernel(quant_type,True)(inputs=[data,source],grid=(rows*32,batch,experts),
        threadgroup=(32*simd_groups,1,1),output_shapes=[(experts,batch,rows)],output_dtypes=[mx.float32])[0]
    spare=max_workspace_bytes-required
    if batch_spare_workspace_bytes is not None:spare=min(spare,batch_spare_workspace_bytes)
    result=(baseline() if batch_dispatch is None else batch_dispatch(data,source,quant_type,
        grouped=True,baseline=baseline,spare_workspace=spare))
    if synchronize:mx.eval(result)
    return result
