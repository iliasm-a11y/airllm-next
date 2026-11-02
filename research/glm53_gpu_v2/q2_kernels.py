"""CPU-compatible original Q2_K GPU variants. Experimental, explicit opt-in.

No weighted expert aggregation is performed: selected expert order is unchanged.
All variants reduce integers before the same ordered per-block ARM FMA chain.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from research.glm53_gpu.quantized import _kernel as _old_kernel

VARIANTS = ('serial32', 'rowtile2', 'rowtile4', 'blocks4', 'blocks8', 'twopass', 'cachedbytes', 'cachedbytes_tile2', 'cachedbytes_tile4')
KERNELS = Path(__file__).parent / 'kernels_q2'
MAX_PAYLOAD = 64 << 20


def source_for(stage, width, activations, tile=1, groups=1):
    text = (KERNELS / f'{stage}.metal').read_text()
    for key, value in {'WIDTH': width, 'ACTIVATIONS': activations,
                       'TILE': tile, 'GROUPS': groups, 'LANES': 32//groups}.items():
        text = text.replace(key, str(value))
    return text


@lru_cache(maxsize=64)
def kernel_for(stage, width, activations, tile=1, groups=1):
    import mlx.core as mx
    inputs = {'coalesced': ['packed', 'qs', 'scales'], 'serial': ['packed', 'qs', 'scales'], 'subgroups': ['packed', 'qs', 'scales'],
              'partials': ['packed', 'qs'], 'ordered_reduce': ['packed', 'dots', 'mins', 'scales']}[stage]
    outputs = ['dots', 'mins'] if stage == 'partials' else ['out']
    return mx.fast.metal_kernel(
        name=f'airllm_v2_q2_{stage}_{width}_{activations}_{tile}_{groups}',
        input_names=inputs, output_names=outputs,
        source=source_for(stage, width, activations, tile, groups),
        header='#pragma clang fp contract(off)\n', compile_options={'math_mode': 'safe'})


@dataclass
class PreparedActivations:
    qs: object
    scales: object
    width: int
    count: int
    payload_bytes: int
    source_sha256: str

    def close(self):
        self.qs = self.scales = None


def prepare_activations(x, *, budget_bytes=MAX_PAYLOAD):
    import mlx.core as mx
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim not in (1, 2) or not x.flags.c_contiguous:
        raise ValueError('activation must be contiguous float32 [width] or [experts,width]')
    width = x.shape[-1]
    count = 1 if x.ndim == 1 else x.shape[0]
    if not 1 <= count <= 8 or width < 256 or width > 65536 or width % 256:
        raise ValueError('activation dimensions outside bounded experiment')
    need = 2*x.nbytes + x.size + x.size//256*4
    if type(budget_bytes) is not int or budget_bytes < need or budget_bytes > MAX_PAYLOAD:
        raise MemoryError('activation preparation exceeds payload budget')
    if not np.isfinite(x).all():
        raise ValueError('activation contains nonfinite values')
    maxima = np.abs(x.reshape(-1, 256)).max(axis=1)
    if np.any((maxima != 0) & (maxima < np.float32(127.0/np.finfo(np.float32).max))):
        raise ValueError('Q8_K activation reciprocal would overflow')
    values = mx.array(x.reshape(-1))
    qs, scales = _old_kernel('activation', width)(
        inputs=[values], grid=(x.size//8, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(x.size,), (x.size//256,)], output_dtypes=[mx.int8, mx.float32])
    mx.eval(qs, scales)
    return PreparedActivations(qs, scales, width, count, need, hashlib.sha256(x.tobytes()).hexdigest())


def payload_requirement(packed, width, *, activation_count=1, variant='serial32'):
    if variant not in VARIANTS:
        raise ValueError('unknown Q2_K variant')
    if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or packed.ndim != 3
            or not packed.flags.c_contiguous):
        raise ValueError('packed must be contiguous uint8 [selected_experts,rows,row_bytes]')
    experts, rows, row_bytes = packed.shape
    if type(width) is not int or width < 256 or width > 65536 or width % 256:
        raise ValueError('width must be a bounded multiple of 256')
    if not 1 <= experts <= 8 or not 1 <= rows <= 8192 or row_bytes != width//256*84:
        raise ValueError('packed dimensions outside bounded experiment')
    if type(activation_count) is not int or activation_count not in (1, experts):
        raise ValueError('activations must be shared or one per selected expert')
    scratch = 2*experts*rows*(width//256)*4 if variant == 'twopass' else 0
    return 2*packed.nbytes + activation_count*(width*9 + width//256*4) + experts*rows*8 + scratch


class Q2KVariants:
    def __init__(self, packed, width, *, budget_bytes=MAX_PAYLOAD):
        need = payload_requirement(packed, width, variant='twopass', activation_count=packed.shape[0] if isinstance(packed,np.ndarray) and packed.ndim==3 else 1)
        if type(budget_bytes) is not int or budget_bytes < need or budget_bytes > MAX_PAYLOAD:
            raise MemoryError('Q2_K grouped payload exceeds budget before device import')
        half_bytes = packed.reshape(packed.shape[0], packed.shape[1], width//256,84)[...,80:84].copy()
        if not np.isfinite(half_bytes.view('<f2')).all():
            raise ValueError('nonfinite Q2_K scales')
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError('Metal required')
        self.width, self.experts, self.rows = width, packed.shape[0], packed.shape[1]
        self.payload_bytes, self.budget_bytes = need, budget_bytes
        self.packed = mx.array(packed)
        mx.eval(self.packed)

    def close(self):
        self.packed = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def execute(self, activation, variant='serial32', *, simd_groups=1):
        import mlx.core as mx
        if variant not in VARIANTS or type(simd_groups) is not int or simd_groups not in (1,2,4):
            raise ValueError('unknown variant or SIMD grouping')
        if self.packed is None or not isinstance(activation, PreparedActivations) or activation.qs is None:
            raise ValueError('closed or invalid prepared operands')
        if (type(activation.width) is not int or type(activation.count) is not int
                or activation.width != self.width or activation.count not in (1,self.experts)
                or not isinstance(activation.qs,mx.array) or not isinstance(activation.scales,mx.array)
                or activation.qs.dtype != mx.int8 or activation.scales.dtype != mx.float32
                or activation.qs.shape != (activation.count*self.width,)
                or activation.scales.shape != (activation.count*self.width//256,)):
            raise ValueError('prepared activation shape/type does not match selected experts')
        if variant == 'twopass':
            dots, mins = kernel_for('partials',self.width,activation.count)(
                inputs=[self.packed,activation.qs], grid=(self.rows*32,self.width//256,self.experts),
                threadgroup=(32*simd_groups,1,1), output_shapes=[(self.experts,self.rows,self.width//256)]*2,
                output_dtypes=[mx.int32,mx.int32])
            result = kernel_for('ordered_reduce',self.width,activation.count)(
                inputs=[self.packed,dots,mins,activation.scales], grid=(self.rows,self.experts,1),
                threadgroup=(32,1,1),output_shapes=[(self.experts,self.rows)],output_dtypes=[mx.float32])[0]
        else:
            tile = {'rowtile2':2,'rowtile4':4,'cachedbytes_tile2':2,'cachedbytes_tile4':4}.get(variant,1)
            groups = {'blocks4':4,'blocks8':8}.get(variant,1)
            stage = 'coalesced' if variant.startswith('cachedbytes') else ('serial' if groups == 1 else 'subgroups')
            result = kernel_for(stage,self.width,activation.count,tile,groups)(
                inputs=[self.packed,activation.qs,activation.scales],
                grid=((self.rows+tile-1)//tile*32,self.experts,1),threadgroup=(32*simd_groups,1,1),
                output_shapes=[(self.experts,self.rows)],output_dtypes=[mx.float32])[0]
        mx.eval(result)
        return result
