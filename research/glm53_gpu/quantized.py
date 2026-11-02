"""Opt-in Q2_K Metal micro-path; no model loader or production backend switch.

The original GGUF weight bytes stay Q2_K throughout. ``cpu_q8k`` duplicates
GGML's temporary Q8_K activation quantization and the ARM NEON block arithmetic
(including explicit fused adds); ``float`` is a diagnostic unquantized activation
path like the existing Metal matvec. Neither mode qualifies whole-model fidelity.
Only 2-D packed matrices and one activation vector are accepted deliberately.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

BLOCK_VALUES = 256
BLOCK_BYTES = 84


def validate_shapes(packed, width, budget_bytes):
    """Conservative payload admission before a device import, not an RSS limit."""
    if type(width) is not int or width < 256 or width % 256 or width > 65536:
        raise ValueError("width must be a multiple of 256 in [256, 65536]")
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or
            packed.ndim != 2 or not packed.flags.c_contiguous):
        raise ValueError("packed must be contiguous uint8 [rows, row_bytes]")
    rows, row_bytes = packed.shape
    if rows < 1 or rows > 8192 or row_bytes != width // 256 * 84:
        raise ValueError("packed shape does not match width or bounded row count")
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError("budget_bytes must be a positive integer")
    # Input owner + device packed copy, input host/device FP32, Q8 bytes/scales,
    # host/device result, including allocation of both diagnostic paths.
    required = 2 * packed.nbytes + width * 8 + width + (width // 256) * 4 + rows * 8
    if required > budget_bytes:
        raise MemoryError(f"payload needs {required} bytes, budget is {budget_bytes}")
    half_bytes = packed.reshape(rows, width // 256, 84)[..., 80:84].copy()
    if not np.isfinite(half_bytes.view('<f2')).all():
        raise ValueError("nonfinite Q2_K scales are not accepted by this experiment")
    return required


def dequantize_reference(packed, width):
    """Literal CPU layout reference for small fixtures; never called by matvec."""
    validate_shapes(packed, width, 2**63 - 1)
    rows = packed.shape[0]
    blocks = packed.reshape(rows, width // 256, 84)
    out = np.empty((rows, width), np.float32)
    for row in range(rows):
        for b in range(width // 256):
            p = blocks[row, b]
            d, dm = np.frombuffer(p[80:84].tobytes(), dtype='<f2').astype(np.float32)
            for k in range(256):
                group = k // 16
                qpos = 16 + (k // 128) * 32 + k % 32
                shift = ((k % 128) // 32) * 2
                q = int(p[qpos]) >> shift & 3
                scale = int(p[group])
                out[row, b * 256 + k] = (d * np.float32(scale & 15) * np.float32(q)
                                               - dm * np.float32(scale >> 4))
    return out


@lru_cache(maxsize=3)
def _kernel(mode, width):
    import mlx.core as mx
    root = Path(__file__).parent / 'kernels'
    if mode == 'activation':
        return mx.fast.metal_kernel(
            name='airllm_glm53_q8k_activation', input_names=['x'],
            output_names=['qs', 'scales'],
            source=(root / 'q8k_activation.metal').read_text(),
            header='#pragma clang fp contract(off)\n',
            compile_options={'math_mode': 'safe'})
    if mode == 'cpu_q8k':
        body = r'''
        int dot = 0, minimum = 0;
        for (uint k = lane; k < 256; k += 32) {
            const uint sc = p[k / 16];
            const uint pos = 16 + (k / 128) * 32 + k % 32;
            const uint shift = ((k % 128) / 32) * 2;
            const int q = int((p[pos] >> shift) & 3);
            const int act = int(qs[b * 256 + k]);
            dot += int(sc & 15) * q * act;
            minimum += int(sc >> 4) * act;
        }
        dot = simd_sum(dot);
        minimum = simd_sum(minimum);
        if (lane == 0) {
            const float dd = scales[b] * d;
            const float md = -scales[b] * dm;
            result = fma(md, float(minimum), result);
            result = fma(dd, float(dot), result);
        }
        '''
        names = ['packed', 'qs', 'scales']
    elif mode == 'float':
        body = r'''
        float partial = 0.0f;
        for (uint k = lane; k < 256; k += 32) {
            const uint sc = p[k / 16];
            const uint pos = 16 + (k / 128) * 32 + k % 32;
            const uint shift = ((k % 128) / 32) * 2;
            const float q = float((p[pos] >> shift) & 3);
            const float weight = (d * float(sc & 15)) * q - dm * float(sc >> 4);
            partial += weight * x[b * 256 + k];
        }
        const float block_sum = simd_sum(partial);
        if (lane == 0) result += block_sum;
        '''
        names = ['packed', 'x']
    else:
        raise ValueError('unknown arithmetic mode')
    source = (root / 'q2k_matvec.metal').read_text().replace('WIDTH', str(width)).replace('DOT_BODY', body)
    return mx.fast.metal_kernel(
        name=f'airllm_glm53_q2k_{mode}_{width}', input_names=names,
        output_names=['out'], source=source,
        header='#pragma clang fp contract(off)\n',
        compile_options={'math_mode': 'safe'})


class Q2KMetalMatvec:
    """Bounded prepared packed matrix. Explicit construction is the opt-in.

    ``budget_bytes`` bounds known payload buffers; MLX compiler/driver and
    Python overhead require independent process monitoring by the harness.
    """
    def __init__(self, packed, width, *, budget_bytes=32 * 1024**2):
        self.payload_bytes = validate_shapes(packed, width, budget_bytes)
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError('this experiment requires Metal')
        self.width, self.rows = width, packed.shape[0]
        self.budget_bytes = budget_bytes
        self.packed = mx.array(packed)
        mx.eval(self.packed)

    def close(self):
        # Calls synchronize before returning; no unbounded in-flight GPU queue.
        self.packed = None

    def __enter__(self):
        if self.packed is None:
            raise RuntimeError('prepared matrix is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def quantize_activations(self, x):
        import mlx.core as mx
        if self.packed is None:
            raise RuntimeError('prepared matrix is closed')
        values = self._quantize_activations(x)
        mx.eval(*values)
        return values

    def _quantize_activations(self, x):
        import mlx.core as mx
        a = np.asarray(x)
        if a.dtype != np.float32 or a.shape != (self.width,) or not a.flags.c_contiguous:
            raise ValueError('activation must be contiguous float32 [width]')
        if not np.isfinite(a).all():
            raise ValueError('nonfinite activations are not accepted')
        maxima = np.abs(a.reshape(-1, 256)).max(axis=1)
        if np.any((maxima != 0) & (maxima < np.float32(127.0 / np.finfo(np.float32).max))):
            raise ValueError('activation block is too small for finite Q8_K reciprocal scale')
        value = mx.array(a)
        qs, scales = _kernel('activation', self.width)(
            inputs=[value], grid=(self.width // 8, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(self.width,), (self.width // 256,)],
            output_dtypes=[mx.int8, mx.float32])
        return qs, scales

    def __call__(self, x, *, mode='cpu_q8k', synchronize=True):
        import mlx.core as mx
        if synchronize is not True:
            raise ValueError('bounded research calls require synchronize=True')
        if self.packed is None:
            raise RuntimeError('prepared matrix is closed')
        a = np.asarray(x)
        if mode == 'cpu_q8k':
            qs, scales = self._quantize_activations(a)
            inputs = [self.packed, qs, scales]
        elif mode == 'float':
            if a.dtype != np.float32 or a.shape != (self.width,) or not a.flags.c_contiguous:
                raise ValueError('activation must be contiguous float32 [width]')
            if not np.isfinite(a).all():
                raise ValueError('nonfinite activations are not accepted')
            inputs = [self.packed, mx.array(a)]
        else:
            raise ValueError("mode must be 'cpu_q8k' or 'float'")
        result = _kernel(mode, self.width)(
            inputs=inputs, grid=(self.rows * 32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(self.rows,)], output_dtypes=[mx.float32])[0]
        if synchronize:
            mx.eval(result)
        return result
