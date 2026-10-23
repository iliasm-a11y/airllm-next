"""Experimental floating Granite 3.0 MoE through bounded native expert reads.

Equations and legacy checkpoint aliases follow IBM/Hugging Face Transformers
(Apache-2.0). This explicit class is separate from production adapter dispatch.
Validation of one 1.3B checkpoint does not certify other Granite generations,
quantized checkpoints, giant models, or physical external-SSD throughput.
"""
from collections import Counter
import math
import numpy as np

from .adapters import validated_default_rope
from .llama_runtime import AirLLMNextLlamaModel, LlamaAttention, normalize_llama_config
from .expert_union import (ExpertBatchMetrics, expert_work_groups,
                           expert_workspace_bytes, validate_expert_execution)


def normalize_granite_config(config):
    if config.get('model_type') != 'granitemoe':
        raise NotImplementedError('Experimental Granite requires model_type=granitemoe')
    validated_default_rope(config)
    c = normalize_llama_config(dict(config, model_type='llama'))
    c['model_type'] = 'granitemoe'
    for name in ('num_local_experts', 'num_experts_per_tok'):
        if type(c.get(name)) is not int or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if c['num_experts_per_tok'] > c['num_local_experts']:
        raise ValueError('Granite top-k exceeds expert count')
    if c['hidden_size'] != c['num_attention_heads'] * c['head_dim']:
        raise ValueError('Granite hidden size must equal attention heads times head dimension')
    for name in ('attention_multiplier', 'embedding_multiplier', 'residual_multiplier', 'logits_scaling'):
        value = c.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be explicit, finite and positive')
    c['mlp_layer_types'] = ['sparse'] * c['num_hidden_layers']
    return c


def granite_route(mx, x, weight, config):
    """Project first, select top-k logits, then normalize selected logits only."""
    scores = (x.reshape(-1, x.shape[-1]).astype(weight.dtype) @ weight.T).astype(mx.float32)
    order = mx.argsort(-scores, axis=-1)
    k = config['num_experts_per_tok']
    # PyTorch topk does not promise the same ordering/choice for ties. Do not
    # silently present a different boundary choice as an exact reference match.
    if k < config['num_local_experts']:
        ranked = mx.take_along_axis(scores, order[:, k-1:k+1], axis=-1)
        if bool(mx.any(ranked[:, 0] == ranked[:, 1]).item()):
            raise ValueError('Granite top-k boundary tie has no certified cross-runtime ordering')
    ids = order[:, :k]
    weights = mx.softmax(mx.take_along_axis(scores, ids, axis=-1), axis=-1).astype(x.dtype)
    return ids, weights


class GraniteWeightsView:
    """Architecture scalars around the shared reader; budgets/caches stay shared."""
    def __init__(self, reader, config):
        self.reader, self.config = reader, config

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def embedding(self, ids, key):
        return self.reader.embedding(ids, key) * self.config['embedding_multiplier']

    def output(self, x, key):
        return self.reader.output(x, key) / self.config['logits_scaling']


class GraniteAttention(LlamaAttention):
    def __call__(self, x, cache, previous):
        mx, c = self.mx, self.config
        batch, length, _ = x.shape
        heads, kv_heads, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        total = cache.position + length
        scratch = 3 * batch * heads * length * total * 4
        scratch += batch * length * (heads + 2 * kv_heads) * dim * x.itemsize * 3
        if scratch > self.reader.cfg.workspace_bytes:
            raise MemoryError('Granite attention exceeds workspace_bytes')
        values = []
        for name, count in (('q_proj', heads), ('k_proj', kv_heads), ('v_proj', kv_heads)):
            y = self.reader.linear(x, self.prefix + name + '.weight')
            values.append(y.reshape(batch, length, count, dim).transpose(0, 2, 1, 3))
        q, k, v = values
        rope = dict(dims=dim, traditional=False, base=c['rope_parameters']['rope_theta'], scale=1., offset=cache.position)
        q, k = mx.fast.rope(q, **rope), mx.fast.rope(k, **rope)
        k, v = cache.append(self.layer, k, v)
        mask = None
        if length > 1:
            mask = mx.arange(k.shape[2])[None, :] <= mx.arange(length)[:, None] + cache.position
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=c['attention_multiplier'], mask=mask)
        a = a.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)
        return (self.reader.linear(a, self.prefix + 'o_proj.weight') * c['residual_multiplier']).astype(x.dtype), None


class GraniteMLP:
    def __init__(self, mx, reader, config, layer, routes, *, execution='union', batch_metrics=None):
        self.mx, self.reader, self.config, self.layer, self.routes = mx, reader, config, layer, routes
        self.execution = validate_expert_execution(execution)
        self.batch_metrics = batch_metrics
        self.prefix = f'model.layers.{layer}.block_sparse_moe.'
        def key(current, legacy):
            return self.prefix + (current if self.prefix + current in reader.cp.tensors else legacy)
        self.router = key('router.weight', 'router.layer.weight')
        self.gate_up = key('experts.gate_up_proj', 'input_linear.weight')
        self.down = key('experts.down_proj', 'output_linear.weight')
        self.last_route_ids = None

    def expert_keys(self, expert):
        return [(self.gate_up, expert), (self.down, expert)]

    def __call__(self, x):
        mx, r, c = self.mx, self.reader, self.config
        positions = math.prod(x.shape[:-1])
        workspace = expert_workspace_bytes(positions, c['hidden_size'], c['intermediate_size'],
                                           c['num_local_experts'], c['num_experts_per_tok'])
        if workspace > r.cfg.workspace_bytes:
            raise MemoryError('Granite expert activations exceed workspace_bytes; reduce prefill_chunk')
        loads_before = r.stats.get('loads', 0) if self.batch_metrics is not None else 0
        ids, weights = granite_route(mx, x, r.get(self.router), c)
        host = np.asarray(ids)
        self.last_route_ids = host.copy()
        selected = np.unique(host)
        r.observe_layer(self.layer, selected)
        self.routes.setdefault(str(self.layer), Counter()).update(int(e) for e in host.flat)
        flat, out = x.reshape(-1, x.shape[-1]), mx.zeros_like(x.reshape(-1, x.shape[-1]))
        observe_batch, clear_batch = getattr(r, 'observe_batch', None), getattr(r, 'clear_batch', None)
        groups = 0
        try:
            if observe_batch is not None:
                # Multiplicity describes actual known assignments, not predicted reuse.
                observe_batch(pair for expert in host.flat for pair in self.expert_keys(int(expert)))
            for expert, ti, ki, next_expert in expert_work_groups(host, self.execution):
                gate_up = r.get(self.gate_up, expert)
                r.prefetch(self.down, expert)
                rows, slots = mx.array(ti), mx.array(ki)
                gate, up = mx.split(r.linear_weight(flat[rows], gate_up), 2, axis=-1)
                if next_expert is not None:
                    r.prefetch(self.gate_up, next_expert)
                value = r.linear((gate * mx.sigmoid(gate)) * up, self.down, expert)
                out[rows] = out[rows] + (value * weights[rows, slots, None]).astype(out.dtype)
                mx.eval(out)
                groups += 1
                del gate_up
        finally:
            if clear_batch is not None:
                clear_batch()
        if self.batch_metrics is not None:
            self.batch_metrics.record(layer=self.layer, execution=self.execution, positions=positions,
                assignments=host.size, distinct_experts=len(selected), expert_groups=groups,
                demand_load_calls=2 * groups, reader_loads_delta=r.stats.get('loads', 0) - loads_before,
                workspace_bytes=workspace)
        return out.reshape(x.shape) * c['residual_multiplier']


class AirLLMNextGraniteMoeModel(AirLLMNextLlamaModel):
    """Explicit experimental backend, floating Granite 3.0 MoE only."""
    normalize_config = staticmethod(normalize_granite_config)

    def __init__(self, model_local_path_or_repo_id, next_config=None, *,
                 expert_execution='union', expert_metrics_capacity=0, **kwargs):
        self.expert_execution = validate_expert_execution(expert_execution)
        self._expert_batch_metrics = ExpertBatchMetrics(expert_metrics_capacity)
        super().__init__(model_local_path_or_repo_id, next_config=next_config, **kwargs)

    def expert_batch_metrics(self, *, reset=False):
        """Snapshot lifetime layer-batch totals; history is opt-in and capped.

        ``demand_load_calls`` counts expert matrix requests, including hits.
        ``reader_loads_delta`` includes router loads and asynchronous read
        completions; checkpoint I/O counters remain the source for byte totals.
        Snapshots may be differenced around a forward call for benchmark deltas.
        """
        with self.lock:
            return self._expert_batch_metrics.snapshot(reset=reset)

    def _build_layers(self):
        self.reader = GraniteWeightsView(self.reader, self.config)
        self.attention = [GraniteAttention(self.mx, self.reader, self.config, i)
                          for i in range(self.config['num_hidden_layers'])]
        self.mlps = [GraniteMLP(self.mx, self.reader, self.config, i, self.routes,
                              execution=self.expert_execution, batch_metrics=self._expert_batch_metrics)
                     for i in range(self.config['num_hidden_layers'])]

    def _validate_layout(self):
        c = self.config
        hidden, mid, dim, experts = c['hidden_size'], c['intermediate_size'], c['head_dim'], c['num_local_experts']
        required = {'model.embed_tokens.weight': (c['vocab_size'], hidden), 'model.norm.weight': (hidden,)}
        if not c.get('tie_word_embeddings', False):
            required['lm_head.weight'] = (c['vocab_size'], hidden)
        for i, mlp in enumerate(self.mlps):
            p = f'model.layers.{i}.'
            required.update({p+'input_layernorm.weight': (hidden,), p+'post_attention_layernorm.weight': (hidden,),
                p+'self_attn.q_proj.weight': (c['num_attention_heads']*dim, hidden),
                p+'self_attn.k_proj.weight': (c['num_key_value_heads']*dim, hidden),
                p+'self_attn.v_proj.weight': (c['num_key_value_heads']*dim, hidden),
                p+'self_attn.o_proj.weight': (hidden, c['num_attention_heads']*dim),
                mlp.router: (experts, hidden), mlp.gate_up: (experts, 2*mid, hidden), mlp.down: (experts, hidden, mid)})
        for key, shape in required.items():
            meta = self.checkpoint.tensors.get(key)
            if meta is None or tuple(meta['shape']) != shape:
                raise ValueError(f'Missing or invalid Granite weight {key}, expected {shape}')
            if meta['dtype'] not in ('F32', 'F16', 'BF16'):
                raise NotImplementedError(f'Unsupported floating Granite weight dtype: {meta["dtype"]}')

    def metrics(self):
        result = super().metrics()
        result['architecture'] = {'model_type': 'granitemoe', 'status': 'experimental explicit backend',
            'routing': 'top-k logits then softmax; boundary ties rejected', 'production_auto_dispatch': False}
        result['expert_execution'] = self.expert_execution
        result['expert_batches'] = self.expert_batch_metrics()
        return result
