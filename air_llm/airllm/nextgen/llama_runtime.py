"""Bias-free, default-RoPE Llama through the bounded Next weight engine.

This is a separate architecture implementation. It shares selective embeddings,
tiled output projection, cache policies, sessions and generation with GLM; it
does not reinterpret Llama checkpoints as DSA or as a hybrid recurrent model.
"""
import math

from .adapters import validated_default_rope
from .runtime import AirLLMNextModel


def normalize_llama_config(config):
    c = dict(config)
    if c.get('model_type') != 'llama':
        raise NotImplementedError('The Next Llama backend requires model_type=llama')
    if c.get('quantization_config') or c.get('quantization'):
        raise NotImplementedError('Next Llama accepts floating checkpoint weights only')
    if c.get('attention_bias') or c.get('mlp_bias') or c.get('hidden_act', 'silu') != 'silu':
        raise NotImplementedError('Next Llama requires bias-free SiLU layers')
    if c.get('partial_rotary_factor', 1.) != 1. or c.get('sliding_window'):
        raise NotImplementedError('Next Llama implements full-head RoPE and full causal attention')
    if c.get('pretraining_tp', 1) != 1:
        raise NotImplementedError('Tensor-parallel checkpoint semantics are not implemented')
    rope = validated_default_rope(c)
    c['rope_parameters'] = dict(rope, rope_theta=rope.get('rope_theta', c.get('rope_theta', 10000.)))
    for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                 'num_attention_heads', 'max_position_embeddings'):
        if type(c.get(name)) is not int or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    heads = c['num_attention_heads']
    c.setdefault('num_key_value_heads', heads)
    c.setdefault('head_dim', c['hidden_size'] // heads)
    for name in ('num_key_value_heads', 'head_dim'):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if heads % c['num_key_value_heads'] or c['head_dim'] % 2:
        raise ValueError('Invalid grouped-query or rotary dimensions')
    theta = c['rope_parameters']['rope_theta']
    eps = c.get('rms_norm_eps', 1e-6)
    if not math.isfinite(theta) or theta <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError('RoPE theta and normalization epsilon must be finite and positive')
    return c


class DenseKV:
    """Append-only preallocated GQA K/V with an explicit payload budget."""
    def __init__(self, mx, budget, limit):
        self.mx = mx
        self.budget = budget
        self.limit = limit
        self.layers = {}
        self.position = self.bytes = self.peak_bytes = 0

    def append(self, layer, key, value):
        stop = self.position + key.shape[2]
        if stop > self.limit:
            raise MemoryError('Context exceeds configured safe limit')
        old = self.layers.get(layer)
        if old is None or old[0].shape[2] < stop:
            capacity = min(self.limit, math.ceil(stop / 128) * 128)
            shape = (1, key.shape[1], capacity, key.shape[3])
            required = 2 * math.prod(shape) * key.itemsize
            old_bytes = sum(x.nbytes for x in old) if old else 0
            # While copying a grown buffer, the previous allocation remains
            # live. Count both instead of only the steady-state replacement.
            if self.bytes + required > self.budget:
                raise MemoryError('Dense KV budget exhausted')
            self.peak_bytes = max(self.peak_bytes, self.bytes + required)
            arrays = [self.mx.zeros(shape, dtype=key.dtype) for _ in range(2)]
            if old:
                for target, source in zip(arrays, old):
                    target[:, :, :self.position] = source[:, :, :self.position]
            self.layers[layer] = arrays
            self.bytes += required - old_bytes
            self.peak_bytes = max(self.bytes, self.peak_bytes)
        arrays = self.layers[layer]
        for target, source in zip(arrays, (key, value)):
            target[:, :, self.position:stop] = source
        return [x[:, :, :stop] for x in arrays]

    def truncate(self, position):
        if type(position) is not int or not 0 <= position <= self.position:
            raise ValueError('KV truncate position must be within the committed prefix')
        self.position = position

    def reset(self):
        self.layers.clear()
        self.position = self.bytes = 0


class LlamaAttention:
    def __init__(self, mx, reader, config, layer):
        self.mx, self.reader, self.config, self.layer = mx, reader, config, layer
        self.prefix = f'model.layers.{layer}.self_attn.'

    def __call__(self, x, cache, previous):
        mx, c = self.mx, self.config
        batch, length, _ = x.shape
        heads, kv_heads, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        total = cache.position + length
        scratch = 3 * batch * heads * length * total * 4
        scratch += batch * length * (heads + 2 * kv_heads) * dim * x.itemsize * 3
        if scratch > self.reader.cfg.workspace_bytes:
            raise MemoryError('Llama attention workspace exceeds workspace_bytes; reduce context/prefill_chunk')
        values = []
        for name, count in (('q_proj', heads), ('k_proj', kv_heads), ('v_proj', kv_heads)):
            y = self.reader.linear(x, self.prefix + name + '.weight')
            values.append(y.reshape(batch, length, count, dim).transpose(0, 2, 1, 3))
        q, k, v = values
        args = dict(dims=dim, traditional=False, base=c['rope_parameters']['rope_theta'],
                    scale=1., offset=cache.position)
        q, k = mx.fast.rope(q, **args), mx.fast.rope(k, **args)
        k, v = cache.append(self.layer, k, v)
        mask = None
        if length > 1:
            mask = mx.arange(k.shape[2])[None, :] <= mx.arange(length)[:, None] + cache.position
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-.5, mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)
        return self.reader.linear(a, self.prefix + 'o_proj.weight').astype(x.dtype), None


class LlamaMLP:
    def __init__(self, mx, reader, layer):
        self.mx, self.reader = mx, reader
        self.prefix = f'model.layers.{layer}.mlp.'

    def __call__(self, x):
        gate = self.reader.linear(x, self.prefix + 'gate_proj.weight')
        up = self.reader.linear(x, self.prefix + 'up_proj.weight')
        return self.reader.linear((gate * self.mx.sigmoid(gate)) * up,
                                  self.prefix + 'down_proj.weight').astype(x.dtype)


class AirLLMNextLlamaModel(AirLLMNextModel):
    normalize_config = staticmethod(normalize_llama_config)
    session_type = DenseKV

    def _build_layers(self):
        self.attention = [LlamaAttention(self.mx, self.reader, self.config, i)
                          for i in range(self.config['num_hidden_layers'])]
        self.mlps = [LlamaMLP(self.mx, self.reader, i)
                     for i in range(self.config['num_hidden_layers'])]

    def _validate_layout(self):
        c = self.config
        hidden, intermediate, dim = c['hidden_size'], c['intermediate_size'], c['head_dim']
        required = {'model.embed_tokens.weight': (c['vocab_size'], hidden),
                    'model.norm.weight': (hidden,)}
        if not c.get('tie_word_embeddings', False):
            required['lm_head.weight'] = (c['vocab_size'], hidden)
        for i in range(c['num_hidden_layers']):
            prefix = f'model.layers.{i}.'
            required.update({prefix + 'input_layernorm.weight': (hidden,),
                             prefix + 'post_attention_layernorm.weight': (hidden,),
                             prefix + 'mlp.gate_proj.weight': (intermediate, hidden),
                             prefix + 'mlp.up_proj.weight': (intermediate, hidden),
                             prefix + 'mlp.down_proj.weight': (hidden, intermediate),
                             prefix + 'self_attn.q_proj.weight': (c['num_attention_heads'] * dim, hidden),
                             prefix + 'self_attn.k_proj.weight': (c['num_key_value_heads'] * dim, hidden),
                             prefix + 'self_attn.v_proj.weight': (c['num_key_value_heads'] * dim, hidden),
                             prefix + 'self_attn.o_proj.weight': (hidden, c['num_attention_heads'] * dim)})
        for key, shape in required.items():
            meta = self.checkpoint.tensors.get(key)
            if meta is None or tuple(meta['shape']) != shape:
                raise ValueError(f'Missing or invalid Llama weight {key}, expected {shape}')
            if meta['dtype'] not in ('F32', 'F16', 'BF16'):
                raise NotImplementedError(f'Unsupported floating Llama weight dtype: {meta["dtype"]}')

    def new_session(self):
        cache = DenseKV(self.mx, self.options.kv_bytes,
                        min(self.options.max_context, self.config['max_position_embeddings']))
        cache.owner = self
        return cache
