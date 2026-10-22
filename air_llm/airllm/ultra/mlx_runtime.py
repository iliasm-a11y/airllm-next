"""Native MLX/Metal Llama inference with selective direct checkpoint reads.

Hardware-gated: the implementation is shipped for Mac validation, not reported
as tested on Linux. Plain Llama, default RoPE, no checkpoint quantization only.
GLM's DSA and MoE are handled by the separate Transformers/MPS backend.
"""
import json
from pathlib import Path
import sys
from threading import RLock

import numpy as np

from .cache import AdaptiveCache, BoundedPrefetch
from .checkpoint import DirectCheckpoint
from .config import UltraConfig


def validate_mlx_config(cfg):
    if cfg.get('model_type') != 'llama':
        raise NotImplementedError('mlx-ultra currently supports plain Llama only; use ultra for GLM/MoE')
    if cfg.get('quantization_config') or cfg.get('quantization'):
        raise NotImplementedError('mlx-ultra does not implement checkpoint quantization')
    rope = cfg.get('rope_parameters') or cfg.get('rope_scaling') or {}
    if rope.get('rope_type', rope.get('type', 'default')) != 'default':
        raise NotImplementedError('mlx-ultra currently implements default RoPE only')
    if cfg.get('attention_bias') or cfg.get('mlp_bias'):
        raise NotImplementedError('mlx-ultra currently implements bias-free Llama only')
    if cfg.get('hidden_act', 'silu') != 'silu':
        raise NotImplementedError('mlx-ultra currently implements SiLU only')


class MlxLlamaUltra:
    def __init__(self, pretrained_model_name_or_repo_id, ultra=None):
        path = Path(pretrained_model_name_or_repo_id).resolve()
        cfg = json.loads((path / 'config.json').read_text())
        validate_mlx_config(cfg)
        if sys.platform != 'darwin':
            raise RuntimeError('mlx-ultra requires macOS/Apple Silicon; not validated on this host')
        import mlx.core as mx
        self.mx = mx
        self.config = cfg
        self.ultra = ultra if isinstance(ultra, UltraConfig) else UltraConfig(**(ultra or {}))
        self.lock = RLock()
        self.closed = False
        self.checkpoint = DirectCheckpoint(path, self.ultra.metadata_dir(), self.ultra.max_open_files)
        self.cache = AdaptiveCache(self.ultra.cache_bytes)
        self.prefetch = BoundedPrefetch(self.ultra.prefetch_bytes if self.ultra.prefetch else 0)
        self.names = ['model.embed_tokens'] + [f'model.layers.{i}' for i in range(cfg['num_hidden_layers'])]
        self.names += ['model.norm', 'lm_head']
        self.keys = {name: self.checkpoint.keys(name) for name in self.names}
        if cfg.get('tie_word_embeddings') and not self.keys['lm_head']:
            self.keys['lm_head'] = self.keys['model.embed_tokens']
        for name, keys in self.keys.items():
            if not keys:
                raise ValueError(f'Missing MLX weights for {name}')
            if self.checkpoint.nbytes(keys) > self.ultra.max_group_bytes:
                raise MemoryError(f'MLX group {name} exceeds max_group_bytes')
            if any(self.checkpoint.tensors[k]['dtype'] not in ('F32', 'F16', 'BF16') for k in keys):
                raise NotImplementedError('Unsupported MLX weight dtype')
        self.tokenizer = None
        if self.ultra.load_tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                          trust_remote_code=False)

    def _read(self, name):
        mx = self.mx
        out = {}
        # A dedicated CPU stream forces the selected pages on the I/O worker;
        # unified memory lets Metal consume those arrays without a CUDA-style copy.
        with mx.stream(mx.cpu):
            for key in self.keys[name]:
                d = self.checkpoint.tensors[key]
                dt = {'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}[d['dtype']]
                data = np.memmap(self.checkpoint.root / d['file'], mode='r', dtype=dt,
                                 offset=d['offset'], shape=tuple(d['shape']))
                value = mx.array(data)
                if d['dtype'] == 'BF16':
                    value = value.view(mx.bfloat16)
                mx.eval(value)
                out[key] = value
                self.checkpoint.stats['logical_bytes'] += d['bytes']
                self.checkpoint.stats['tensor_reads'] += 1
        return out

    def _weights(self, index):
        name = self.names[index]
        state = self.cache.get(name)
        if state is None:
            state = self.prefetch.take(name)
            if state is None:
                state = self._read(name)
            self.cache.put(name, state)
        if self.ultra.prefetch and index + 1 < len(self.names):
            nxt = self.names[index + 1]
            if not self.cache.contains(nxt):
                self.prefetch.submit(nxt, self.checkpoint.nbytes(self.keys[nxt]), lambda: self._read(nxt))
        return state

    def forward(self, ids, cache=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('Model is closed')
            mx = self.mx
            cfg = self.config
            if not isinstance(ids, mx.array):
                ids = mx.array(ids, dtype=mx.int32)
            if ids.ndim != 2:
                raise ValueError('MLX input_ids must have [batch, sequence] shape')
            old_cache = cache or [None] * cfg['num_hidden_layers']
            if len(old_cache) != cfg['num_hidden_layers']:
                raise ValueError('Wrong MLX KV cache layer count')
            weights = self._weights(0)
            x = weights['model.embed_tokens.weight'][ids]
            mx.eval(x)
            del weights
            result_cache = []
            h = cfg['num_attention_heads']
            kvh = cfg.get('num_key_value_heads', h)
            dim = cfg.get('head_dim') or cfg['hidden_size'] // h
            rope = cfg.get('rope_parameters') or {}
            theta = rope.get('rope_theta', cfg.get('rope_theta', 10000.0))
            eps = cfg.get('rms_norm_eps', 1e-5)
            for i in range(cfg['num_hidden_layers']):
                w = self._weights(i + 1)
                p = f'model.layers.{i}.'
                normed = mx.fast.rms_norm(x, w[p + 'input_layernorm.weight'], eps)
                batch, length, _ = x.shape
                q = (normed @ w[p + 'self_attn.q_proj.weight'].T).reshape(batch, length, h, dim).transpose(0, 2, 1, 3)
                k = (normed @ w[p + 'self_attn.k_proj.weight'].T).reshape(batch, length, kvh, dim).transpose(0, 2, 1, 3)
                v = (normed @ w[p + 'self_attn.v_proj.weight'].T).reshape(batch, length, kvh, dim).transpose(0, 2, 1, 3)
                previous = old_cache[i]
                offset = previous[0].shape[2] if previous is not None else 0
                if offset + length > cfg.get('max_position_embeddings', 2048):
                    raise ValueError('Input exceeds configured context length')
                q = mx.fast.rope(q, dims=dim, traditional=False, base=theta, scale=1.0, offset=offset)
                k = mx.fast.rope(k, dims=dim, traditional=False, base=theta, scale=1.0, offset=offset)
                if previous is not None:
                    k = mx.concatenate([previous[0], k], axis=2)
                    v = mx.concatenate([previous[1], v], axis=2)
                mask = None
                if length > 1:
                    mask = mx.arange(k.shape[2])[None, :] <= (mx.arange(length)[:, None] + offset)
                a = mx.fast.scaled_dot_product_attention(q, k, v, scale=dim**-0.5, mask=mask)
                a = a.transpose(0, 2, 1, 3).reshape(batch, length, h * dim)
                x = x + a @ w[p + 'self_attn.o_proj.weight'].T
                n = mx.fast.rms_norm(x, w[p + 'post_attention_layernorm.weight'], eps)
                gate = n @ w[p + 'mlp.gate_proj.weight'].T
                up = n @ w[p + 'mlp.up_proj.weight'].T
                x = x + ((gate * mx.sigmoid(gate)) * up) @ w[p + 'mlp.down_proj.weight'].T
                # Evaluation before dropping references is essential for bounded
                # weights: lazy graphs must not retain all preceding layers.
                mx.eval(x, k, v)
                result_cache.append((k, v))
                del w
            w = self._weights(len(self.names) - 2)
            x = mx.fast.rms_norm(x[:, -1:, :], w['model.norm.weight'], eps)
            mx.eval(x)
            del w
            w = self._weights(len(self.names) - 1)
            head = w.get('lm_head.weight', w.get('model.embed_tokens.weight'))
            logits = x @ head.T
            mx.eval(logits)
            return logits, result_cache

    __call__ = forward

    def generate(self, ids, max_new_tokens=32, eos_token_id=None):
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens must be nonnegative')
        with self.lock:
            mx = self.mx
            tokens = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
            if tokens.shape[0] != 1:
                raise NotImplementedError('MLX generate currently supports batch size one')
            current, cache = tokens, None
            eos = self.config.get('eos_token_id') if eos_token_id is None else eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            for _ in range(max_new_tokens):
                logits, cache = self.forward(current, cache=cache)
                token = mx.argmax(logits[:, -1], axis=-1)[:, None]
                tokens = mx.concatenate([tokens, token], axis=1)
                mx.eval(tokens)
                if token.item() in eos_ids:
                    break
                current = token
            return tokens

    def close(self):
        with self.lock:
            if not self.closed:
                try:
                    self.prefetch.close()
                finally:
                    self.cache.clear()
                    self.checkpoint.close()
                    self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
