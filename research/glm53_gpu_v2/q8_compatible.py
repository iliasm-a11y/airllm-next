"""Opt-in packed Q8_0 matvec matching pinned ARM NEON nrc=1 arithmetic.

Original GGUF weights remain 34-byte Q8_0 blocks. Only temporary activations
are quantized exactly as the native CPU. This does not install a runtime hook
or qualify CPU nrc=2 prefill, whole-model fidelity, or end-to-end performance.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

BLOCK_VALUES = 32
BLOCK_BYTES = 34


def validate_matrix(packed, width, budget_bytes):
    if type(width) is not int or not 32 <= width <= 65536 or width % 32:
        raise ValueError('width must be a multiple of 32 in [32, 65536]')
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or packed.ndim != 2
            or not packed.flags.c_contiguous or not 1 <= packed.shape[0] <= 154880
            or packed.shape[1] != width // 32 * 34):
        raise ValueError('packed must be contiguous uint8 [rows, width/32*34]')
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError('budget_bytes must be a positive integer')
    rows = packed.shape[0]
    required = 2 * packed.nbytes + width * 8 + width + (width // 32) * 2 + rows * 8
    if required > budget_bytes:
        raise MemoryError(f'known payload requires {required} bytes, budget {budget_bytes}')
    scales = packed.reshape(rows, width // 32, 34)[..., :2].copy().view('<f2')
    if not np.isfinite(scales).all():
        raise ValueError('nonfinite Q8_0 scales are unsupported')
    return required


def validate_activation(value, width):
    a = np.asarray(value)
    if a.dtype != np.float32 or a.shape != (width,) or not a.flags.c_contiguous:
        raise ValueError('activation must be contiguous float32 [width]')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite activation is unsupported')
    maximum = np.abs(a.reshape(-1, 32)).max(axis=1)
    # Reject CPU reciprocal overflow and device-dependent subnormal operations.
    with np.errstate(under='ignore'):
        d = maximum / np.float32(127)
    if np.any((maximum != 0) & (d < np.finfo(np.float32).tiny)):
        raise ValueError('subnormal activation quantization is outside the qualified path')
    if np.any(d > np.finfo(np.float16).max):
        raise ValueError('activation scale would overflow native Q8_0 half storage')
    return a


@lru_cache(maxsize=16)
def _kernel(kind, width):
    import mlx.core as mx
    root = Path(__file__).parent / 'kernels_q8'
    if kind == 'activation':
        name, inputs, outputs = 'q8_activation', ['x'], ['qs', 'scale_bits']
    elif kind == 'matvec':
        name, inputs, outputs = 'q8_matvec', ['packed', 'qs', 'scale_bits'], ['out']
    elif kind == 'fused':
        name, inputs, outputs = 'q8_matvec_inline', ['packed', 'x'], ['out']
    elif kind == 'shared':
        if width != 4096:
            raise ValueError('shared activation kernel requires width 4096')
        name, inputs, outputs = 'q8_matvec_shared', ['packed', 'x'], ['out']
    else:
        raise ValueError('unknown kernel kind')
    source = (root / (name + '.metal')).read_text().replace('WIDTH', str(width))
    header = '#pragma clang fp contract(off)\n'
    if kind == 'fused':
        header += (root / 'q8_inline_helpers.metal').read_text()
    return mx.fast.metal_kernel(name=f'airllm_glm53_{name}_{width}', input_names=inputs,
        output_names=outputs, source=source, header=header,
        compile_options={'math_mode': 'safe'})


class Q8CompatibleMatvec:
    """Synchronous prepared matrix; payload admission is separate from RSS guard.

    Each call uses one vector. Callers must not retain results without charging
    their memory to a separate budget. Explicit construction opts into research.
    """
    def __init__(self, packed, width, *, budget_bytes=32 << 20):
        self.payload_bytes = validate_matrix(packed, width, budget_bytes)
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError('Metal is required')
        self.width, self.rows = width, packed.shape[0]
        self.packed = mx.array(packed)
        mx.eval(self.packed)

    def _activations(self, x):
        import mlx.core as mx
        if self.packed is None:
            raise RuntimeError('matrix is closed')
        x = validate_activation(x, self.width)
        return _kernel('activation', self.width)(inputs=[mx.array(x)],
            grid=(self.width, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(self.width,), (self.width // 32,)],
            output_dtypes=[mx.int8, mx.uint16])

    def quantize_activations(self, x):
        import mlx.core as mx
        values = self._activations(x)
        mx.eval(*values)
        return values

    def __call__(self, x, *, fused=False, shared=False):
        import mlx.core as mx
        if type(fused) is not bool or type(shared) is not bool:
            raise ValueError('fused/shared must be booleans')
        if fused and shared:
            raise ValueError('select either fused-inline or shared-activation')
        if shared and self.width != 4096:
            raise ValueError('shared activation kernel requires width 4096')
        if self.packed is None:
            raise RuntimeError('matrix is closed')
        if fused or shared:
            x = validate_activation(x, self.width)
            kind, inputs = ('shared' if shared else 'fused'), [self.packed, mx.array(x)]
        else:
            qs, scales = self._activations(x)
            kind, inputs = 'matvec', [self.packed, qs, scales]
        # Shared quantization needs all 32 lanes, including the final partial tile.
        grid_x = ((self.rows + 7) // 8 * 32) if shared else self.rows * 4
        result = _kernel(kind, self.width)(inputs=inputs,
            grid=(grid_x, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[(self.rows,)], output_dtypes=[mx.float32])[0]
        mx.eval(result)
        return result

    def close(self):
        self.packed = None

    def __enter__(self):
        if self.packed is None:
            raise RuntimeError('matrix is closed')
        return self

    def __exit__(self, *exc):
        self.close()
