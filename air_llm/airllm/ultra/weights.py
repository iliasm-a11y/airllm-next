"""Load one weight at a time, including block-scaled FP8 without native FP8 kernels."""
import math
import torch


def dequantize_fp8(weight, scales, block_size, dtype):
    if weight.ndim != 2 or len(block_size) != 2:
        raise ValueError('Only two-dimensional block-scaled FP8 weights are supported')
    rows, cols = weight.shape
    br, bc = block_size
    if br <= 0 or bc <= 0:
        raise ValueError('Invalid FP8 block size')
    expected = (math.ceil(rows / br), math.ceil(cols / bc))
    if tuple(scales.shape) != expected:
        raise ValueError(f'FP8 scale shape {tuple(scales.shape)} != {expected}')
    out = torch.empty(weight.shape, dtype=dtype, device='cpu')
    # Avoid a second full-size fp32 tensor and an expanded scale matrix.
    for start in range(0, rows, br):
        stop = min(start + br, rows)
        scale_row = scales[start // br].float().repeat_interleave(bc)[:cols]
        out[start:stop] = weight[start:stop].float() * scale_row
    return out


class WeightReader:
    def __init__(self, checkpoint, dtype, quantization=None):
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.quantization = quantization or {}
        method = self.quantization.get('quant_method')
        if method not in (None, 'fp8'):
            raise NotImplementedError(f'Ultra does not implement {method!r}; use the upstream backend')
        self.block_size = self.quantization.get('weight_block_size', [128, 128])

    @staticmethod
    def is_scale(key):
        return key.endswith(('.weight_scale_inv', '.weight_scale'))

    def dtype_for(self, key):
        # GLM's router correction bias and indexer head projection are explicitly
        # retained in fp32 by its native Transformers implementation.
        return torch.float32 if any(s in key for s in ('e_score_correction_bias',
                                                       'indexer.weights_proj')) else self.dtype

    def load(self, keys):
        out = {}
        for key in sorted(keys, key=lambda k: (self.checkpoint.tensors[k]['file'],
                                               self.checkpoint.tensors[k]['offset'])):
            if self.is_scale(key):
                continue
            raw_dtype = self.checkpoint.tensors[key]['dtype']
            target_dtype = self.dtype_for(key)
            if raw_dtype.startswith('F8_'):
                # GLM/DeepSeek block FP8 stores the dequant multiplier as weight_scale_inv.
                scale = key + '_scale_inv'
                if self.quantization.get('quant_method') != 'fp8' or scale not in self.checkpoint.tensors:
                    raise NotImplementedError(f'FP8 weight has no supported block scales: {key}')
                values = self.checkpoint.read([key, scale], copy=False)
                out[key] = dequantize_fp8(values[key], values[scale], self.block_size, target_dtype)
            else:
                out.update(self.checkpoint.read([key], copy=True, dtype=target_dtype))
        return out

    def nbytes(self, keys):
        return sum(self.checkpoint.nbytes([k], self.dtype_for(k)) for k in keys if not self.is_scale(k))
