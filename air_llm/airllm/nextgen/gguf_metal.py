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


@lru_cache(maxsize=2)
def _kernel(quant_type):
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
    return mx.fast.metal_kernel(name=f'airllm_gguf_q{quant_type}_matvec',
                               input_names=['packed', 'x'], output_names=['out'],
                               source=code, header='#pragma clang fp contract(off)\n')


def packed_matmul(packed, x, quant_type, *, max_workspace_bytes=64 * 1024**2):
    """Multiply [..., width] by packed [rows, row_bytes].T; caller tiles reads."""
    import mlx.core as mx
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 1:
        raise ValueError('max_workspace_bytes must be a positive integer')
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
    data = mx.array(packed) if isinstance(packed, np.ndarray) else packed
    source = mx.contiguous(x.astype(mx.float32)).reshape(batch, width)
    result = _kernel(quant_type)(inputs=[data, source], grid=(rows * 32, batch, 1),
                                threadgroup=(32, 1, 1),
                                output_shapes=[(batch, rows)], output_dtypes=[mx.float32])[0]
    mx.eval(result)
    return result.reshape(*x.shape[:-1], rows)
