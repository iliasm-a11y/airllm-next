"""Experimental bounded streaming of trained MLX affine Qwen3 MoE weights.

This separate, explicit backend supports qwen3_moe, not Qwen3.5/3.8. It uses
the checkpoint's existing 4/6/8-bit affine weights without requantizing them.
Equations follow the Apache-2.0 Qwen/Transformers and MIT MLX-LM references.
No full expert bank or complete quantized model is allocated.
"""
import math
import numpy as np

from .attention import rms
from .expert_union import expert_work_groups, expert_workspace_bytes
from .llama_runtime import AirLLMNextLlamaModel, normalize_llama_config
from .quantization import PackedMatrix
from .weights import NativeWeights


def normalize_qwen3_mlx_config(config):
    c = dict(config)
    if c.get('model_type') != 'qwen3_moe':
        raise NotImplementedError('This backend requires model_type=qwen3_moe')
    q = c.get('quantization') or c.get('quantization_config')
    if not isinstance(q, dict) or q.get('bits') not in (4, 6, 8) or q.get('group_size') not in (32, 64, 128):
        raise NotImplementedError('Only existing MLX affine 4/6/8-bit groups 32/64/128 are supported')
    if q.get('mode', 'affine') != 'affine' or set(q) - {'bits', 'group_size', 'mode'}:
        raise NotImplementedError('Mixed quantization and non-affine formats are unsupported')
    if c.get('quantization') and c.get('quantization_config') and c['quantization'] != c['quantization_config']:
        raise ValueError('Conflicting quantization declarations')
    if c.get('decoder_sparse_step', 1) != 1 or c.get('mlp_only_layers', []):
        raise NotImplementedError('Every layer must be a sparse Qwen3 MoE layer')
    # Reuse only the architecture-neutral Llama dimension/RoPE checks.
    plain = dict(c, model_type='llama')
    plain.pop('quantization', None)
    plain.pop('quantization_config', None)
    c.update(normalize_llama_config(plain))
    c['model_type'] = 'qwen3_moe'
    c['quantization'] = dict(q)
    c['quantization_config'] = dict(q)
    for name in ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size'):
        if type(c.get(name)) is not int or c[name] < 1:
            raise ValueError(f'Invalid Qwen3 dimension {name}')
    if c['num_experts_per_tok'] > c['num_experts'] or c.get('norm_topk_prob', True) is not True:
        raise NotImplementedError('Qwen3 requires valid top-k and normalized routing weights')
    if c['hidden_size'] % q['group_size'] or c['moe_intermediate_size'] % q['group_size']:
        raise ValueError('Qwen3 matrix columns must be divisible by affine group_size')
    c['mlp_layer_types'] = ['sparse'] * c['num_hidden_layers']
    return c


class MLXCheckpointWeights(NativeWeights):
    """Read one affine matrix/expert/tile, retaining packed data in the cache.

    Source BF16 scale/bias values are promoted exactly to FP32. The cache and
    I/O reservations include that promotion. Quantized matmul accumulates FP32
    through the shared native reader; this is not a claim of BF16 bit identity
    with mlx-lm's default computation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        q = self.model_config['quantization']
        self.bits, self.group_size = q['bits'], q['group_size']

    def _packed(self, key):
        return self.cp.tensors[key]['dtype'] == 'U32'

    @staticmethod
    def _affine_keys(key):
        if not key.endswith('.weight'):
            raise ValueError('MLX affine weights must have the .weight suffix')
        stem = key[:-len('.weight')]
        return key, stem+'.scales', stem+'.biases'

    def shape(self, key, expert=None):
        shape = super().shape(key, expert)
        if self._packed(key):
            if len(shape) != 2 or shape[-1]*32 % self.bits:
                raise ValueError(f'Invalid affine packed matrix shape: {key}')
            return shape[:-1] + (shape[-1]*32//self.bits,)
        return shape

    def estimate(self, key, expert=None):
        if not self._packed(key):
            return super().estimate(key, expert)
        # uint32 payload plus two promoted FP32 affine arrays.
        shapes = [super(MLXCheckpointWeights, self).shape(k, expert) for k in self._affine_keys(key)]
        return sum(math.prod(s)*4 for s in shapes)

    def _load_value(self, key, expert=None, rows=None):
        if not self._packed(key):
            return super()._load_value(key, expert, rows)
        shape = self.shape(key, expert)
        if rows is not None:
            if (len(rows) != 2 or any(type(x) is not int for x in rows)
                    or not 0 <= rows[0] < rows[1] <= shape[0]):
                raise ValueError('Invalid affine row selection')
            shape = (rows[1]-rows[0], shape[1])
        count = shape[0]
        required = count*(shape[1]*self.bits//8 + 8*shape[1]//self.group_size)
        if required > min(self.cfg.max_tensor_bytes, self.cfg.io_bytes):
            raise MemoryError('Affine packed matrix exceeds the bounded I/O allocation')
        keys = self._affine_keys(key)
        raw = self.cp.read_arrays([dict(key=k, expert=expert, rows=rows) for k in keys],
            mode=self.cfg.io_mode, max_batch_bytes=self.cp.max_read_bytes,
            coalesce=self.cfg.coalesce_reads)
        mx = self.mx
        arrays = [self._array(raw[0])]
        for k, value in zip(keys[1:], raw[1:]):
            if self.cp.tensors[k]['dtype'] == 'BF16':
                value = (value.astype(np.uint32) << 16).view(np.float32)
            arrays.append(self._array(value, mx.float32))
        with self.cp.io_lock:
            self.cp.stats['quantized_logical_bytes'] += sum(a.nbytes for a in raw)
        return PackedMatrix(*arrays, shape, self.bits, self.group_size)

    def embedding(self, ids, key):
        if not self._packed(key):
            return super().embedding(ids, key)
        host = np.asarray(ids)
        unique, reverse = np.unique(host.reshape(-1), return_inverse=True)
        shape = self.shape(key)
        if np.any(unique < 0) or np.any(unique >= shape[0]):
            raise ValueError('Token id out of range')
        # One selected row at a time: never dequantize the complete embedding.
        out = []
        for index in unique:
            p = self._load(key, rows=(int(index), int(index)+1))
            if p.nbytes + shape[1]*4 > self.cfg.max_tensor_bytes:
                raise MemoryError('Embedding row plus dequantized row exceeds max_tensor_bytes')
            row = self.mx.dequantize(p.weight, p.scales, p.biases,
                bits=p.bits, group_size=p.group_size, mode='affine').astype(self.dtype)
            self.mx.eval(row)
            out.append(row)
        return self.mx.concatenate(out, axis=0)[self.mx.array(reverse)].reshape(tuple(host.shape)+(shape[1],))


class Qwen3Attention:
    def __init__(self, mx, reader, config, layer):
        self.mx, self.r, self.c, self.layer = mx, reader, config, layer
        self.p = f'model.layers.{layer}.self_attn.'

    def __call__(self, x, cache, previous):
        mx, c = self.mx, self.c
        batch, length, _ = x.shape
        heads, kv, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        total = cache.position + length
        scratch = 3*batch*heads*length*total*4 + 3*batch*length*(heads+2*kv)*dim*4
        if scratch > self.r.cfg.workspace_bytes:
            raise MemoryError('Qwen3 attention workspace exceeds workspace_bytes')
        values = []
        for name, count in (('q_proj', heads), ('k_proj', kv), ('v_proj', kv)):
            y = self.r.linear(x, self.p+name+'.weight').astype(x.dtype)
            y = y.reshape(batch, length, count, dim)
            if name != 'v_proj':
                y = rms(mx, y, self.r.get(self.p+name[0]+'_norm.weight'), c['rms_norm_eps'])
            values.append(y.transpose(0, 2, 1, 3))
        q, k, v = values
        kwargs = dict(dims=dim, traditional=False, base=c['rope_parameters']['rope_theta'],
                      scale=1./c['rope_parameters'].get('factor', 1.), offset=cache.position)
        q, k = mx.fast.rope(q, **kwargs), mx.fast.rope(k, **kwargs)
        k, v = cache.append(self.layer, k, v)
        mask = None
        if length > 1:
            mask = mx.arange(k.shape[2])[None, :] <= mx.arange(length)[:, None]+cache.position
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, heads*dim)
        return self.r.linear(y, self.p+'o_proj.weight').astype(x.dtype), None


class Qwen3MLXMoE:
    def __init__(self, mx, reader, config, layer):
        self.mx, self.r, self.c, self.layer = mx, reader, config, layer
        self.p = f'model.layers.{layer}.mlp.'

    def expert_keys(self, expert):
        return [(self.p+'switch_mlp.'+name+'.weight', expert)
                for name in ('gate_proj', 'up_proj', 'down_proj')]

    def __call__(self, x):
        mx, c = self.mx, self.c
        flat = x.reshape(-1, x.shape[-1])
        workspace = expert_workspace_bytes(flat.shape[0], c['hidden_size'],
            c['moe_intermediate_size'], c['num_experts'], c['num_experts_per_tok'])
        if workspace > self.r.cfg.workspace_bytes:
            raise MemoryError('Qwen3 MoE workspace exceeds workspace_bytes')
        logits = self.r.linear(flat, self.p+'gate.weight').astype(mx.float32)
        # Full-softmax then normalized top-k follows Qwen3's original equations.
        probabilities = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probabilities, axis=-1)
        k = c['num_experts_per_tok']
        if k < c['num_experts']:
            boundary = mx.take_along_axis(probabilities, order[:, k-1:k+1], axis=-1)
            if bool(mx.any(boundary[:, 0] == boundary[:, 1]).item()):
                raise ValueError('Qwen3 top-k boundary tie has no certified ordering')
        ids = order[:, :k]
        weights = mx.take_along_axis(probabilities, ids, axis=-1)
        weights = (weights/mx.sum(weights, axis=-1, keepdims=True)).astype(x.dtype)
        host = np.array(ids)
        self.r.observe_layer(self.layer, np.unique(host).tolist())
        out = mx.zeros(flat.shape, dtype=x.dtype)
        self.r.observe_batch(item for expert in np.unique(host) for item in self.expert_keys(int(expert)))
        try:
            for expert, positions, slots, next_expert in expert_work_groups(host):
                keys = self.expert_keys(expert)
                for key, e in keys[1:]:
                    self.r.prefetch(key, e)
                selected = flat[mx.array(positions)]
                gate = self.r.linear(selected, *keys[0])
                up = self.r.linear(selected, *keys[1])
                if next_expert is not None:
                    self.r.prefetch(*self.expert_keys(next_expert)[0])
                y = self.r.linear((gate*mx.sigmoid(gate))*up, *keys[2]).astype(x.dtype)
                out[mx.array(positions)] += y*weights[mx.array(positions), mx.array(slots), None]
                # Bound graph lifetime: previous experts' weights can be evicted.
                mx.eval(out)
        finally:
            self.r.clear_batch()
        return out.reshape(x.shape)


class ExperimentalQwen3MLXModel(AirLLMNextLlamaModel):
    """Explicit local-only affine Qwen3-MoE adapter; ordinary generation only."""
    normalize_config = staticmethod(normalize_qwen3_mlx_config)

    def _build_layers(self):
        self.reader.close()
        self.reader = MLXCheckpointWeights(self.checkpoint, self.mx, self.options, self.config)
        self.attention = [Qwen3Attention(self.mx, self.reader, self.config, i)
                          for i in range(self.config['num_hidden_layers'])]
        self.mlps = [Qwen3MLXMoE(self.mx, self.reader, self.config, i)
                     for i in range(self.config['num_hidden_layers'])]

    def _validate_layout(self):
        c, cp = self.config, self.checkpoint
        h, m, e, d = c['hidden_size'], c['moe_intermediate_size'], c['num_experts'], c['head_dim']
        required = {'model.embed_tokens.weight': (c['vocab_size'], h), 'model.norm.weight': (h,)}
        if not c.get('tie_word_embeddings', False):
            required['lm_head.weight'] = (c['vocab_size'], h)
        for layer in range(c['num_hidden_layers']):
            p = f'model.layers.{layer}.'
            required.update({p+'input_layernorm.weight': (h,), p+'post_attention_layernorm.weight': (h,),
                p+'self_attn.q_norm.weight': (d,), p+'self_attn.k_norm.weight': (d,),
                p+'self_attn.q_proj.weight': (c['num_attention_heads']*d, h),
                p+'self_attn.k_proj.weight': (c['num_key_value_heads']*d, h),
                p+'self_attn.v_proj.weight': (c['num_key_value_heads']*d, h),
                p+'self_attn.o_proj.weight': (h, c['num_attention_heads']*d),
                p+'mlp.gate.weight': (e, h),
                p+'mlp.switch_mlp.gate_proj.weight': (e, m, h),
                p+'mlp.switch_mlp.up_proj.weight': (e, m, h),
                p+'mlp.switch_mlp.down_proj.weight': (e, h, m)})
        for key, expected in required.items():
            meta = cp.tensors.get(key)
            if meta is None:
                raise ValueError(f'Missing Qwen3 MLX weight: {key}')
            shape = tuple(meta['shape'])
            if meta['dtype'] == 'U32':
                shape = shape[:-1] + (shape[-1]*32//self.reader.bits,)
                for affine_key in self.reader._affine_keys(key)[1:]:
                    affine = cp.tensors.get(affine_key)
                    affine_shape = expected[:-1] + (expected[-1]//self.reader.group_size,)
                    if affine is None or tuple(affine['shape']) != affine_shape or affine['dtype'] not in ('BF16', 'F16', 'F32'):
                        raise ValueError(f'Invalid Qwen3 affine scales/biases: {affine_key}')
            elif meta['dtype'] not in ('BF16', 'F16', 'F32'):
                raise ValueError(f'Unsupported Qwen3 dtype: {key}')
            if shape != expected:
                raise ValueError(f'Invalid Qwen3 MLX shape: {key}: {shape} != {expected}')
