# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Extracted (AST formatting, top-level decorators removed) from Transformers
# 1b754e9a4636191852a1618fc588ad5c863ab4d5, modeling_glm5_next.py.
# Independent CPU oracle only; does not load model weights or claim inference.
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
ACT2FN = {"silu": F.silu, "sigmoid": torch.sigmoid}

class Glm5NextTextMLP(nn.Module):

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)

class Glm5NextTextTopkRouter(nn.Module):

    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.num_group, self.num_experts // self.num_group).reshape(-1, self.num_experts)
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return (router_logits, topk_weights, topk_indices)

class Glm5NextTextUnweightedRMSNorm(nn.Module):

    def __init__(self, eps: float=1e-06):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)

class Glm5NextTextHyperConnection(nn.Module):
    """
    A module to implement manifold-constrained Hyper-Connections (mHC) (Xie et al., 2026) which strengthens the
    conventional residual connections between adjacent Transformer blocks.

    When using mHC, each token is projected onto `hc_mult` streams, so the shape of decoder layer inputs changes from
    [batch_size, sequence_length, hidden_size] to [batch_size, sequence_length, hc_mult, hidden_size].
    To keep the same input shape for attention or MLP blocks, the streams are collapsed into one upon entering a block,
    and expanded back into `hc_mult` streams upon exiting. There is also a weighted residual connection between the
    input and output streams.
    The weights used for collapsing (pre), expanding (post) and mixing (comb) are computed from the `hc_mult` input
    streams through a learned projection (plus Sinkhorn-Knopp algorithm for the comb weight).

    The diagram below shows the flow of the mHC streams (B = batch_size, S = seq_length, N = hc_mult, D = hidden_size):

                                                  ┌───────────────────┐
                             N input streams  ────│ FLATTEN + PROJECT |────> (pre, post, comb) weights
                              [B, S, N, D]        └───────────────────┘      ([B, S, N],  [B, S, N],  [B, S, N, N])
                                  │ │ │
               ╭──────────────────┴─┼─┼──────────────────╮
               │ ╭──────────────────┴─┼────────────────╮ │
               │ │ ╭──────────────────┴──────────────╮ │ │
               │ │ │                                 │ │ │
        ┌─────────────────┐                          │ │ │
        │ COLLAPSE (pre)  │                          │ │ │
        └─────────────────┘                          │ │ │
                 │                                   │ │ │
            Block input                              │ │ │
             [B, S, D]                               │ │ │
                 │                                   │ │ │
        ┌─────────────────┐                     ┌─────────────┐
        │   ATTN or MLP   │                     │  MIX (comb) │
        └─────────────────┘                     └─────────────┘
                 │                                   │ │ │
           Block output                              │ │ │
             [B, S, D]                               │ │ │
                 │                                   │ │ │
        ┌─────────────────┐                          │ │ │
        │  EXPAND (post)  │                          │ │ │
        └─────────────────┘                          │ │ │
               │ │ │                                 │ │ │
        N expanded streams                    N residual streams
           [B, S, N, D]                          [B, S, N, D]
               │ │ │                                 │ │ │
               │ │ ╰───────────┌─────────┐───────────╯ │ │
               │ ╰─────────────│   ADD   │─────────────╯ │
               ╰───────────────└─────────┘───────────────╯
                                  │ │ │
                                  ▼ ▼ ▼
                            N output streams
                              [B, S, N, D]
    """

    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextTextUnweightedRMSNorm(eps=config.rms_norm_eps)
        concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(concatenated_weights_size))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the weights used to mix in the `hc_mult` streams with the input and output of the next layer, which can
        be an attention or a MLP layer. This is done through three weights:

        - pre: used to collapse the `hc_mult` input streams into one, creating an input tensor for the next layer
        - post: used to expand the output of the next layer back into `hc_mult` streams
        - comb: used to mix the `hc_mult` input streams with the `hc_mult` output streams

        All weights are returned except "pre", which is consumed here.
        """
        batch_size, seq_len = hidden_streams.shape[:2]
        hc = self.hc_mult
        flattened = hidden_streams.view(batch_size, seq_len, -1).float()
        flattened = self.input_norm(flattened)
        flattened = F.linear(flattened, self.fn.float())
        pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
        comb_b = comb_b.view(hc, hc)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return (post, comb, collapsed)

class Glm5NextTextRMSNormGated(nn.Module):

    def __init__(self, hidden_size, eps=1e-06, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = 'sigmoid'

    def forward(self, hidden_states, gate=None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)

def l2norm(x: torch.FloatTensor, dim: int=-1, eps: float=1e-06):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm

def recurrent_kimi_delta_attention(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim, dtype=value.dtype, device=value.device)
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device) if initial_state is None else initial_state.to(value)
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]
        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)
    return (core_attn_out.to(initial_dtype), last_recurrent_state if output_final_state else None)

class Glm5NextTextIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer with k-pool compression for GLM-5.3-Flash.

    The indexer uses lightweight projections (`wq_b`, `wk`) separate from the main MLA
    attention path. It scores compressed k-pool candidates, expands selected pools back
    into raw cache indices, and optionally appends the current incomplete tail pool.

    **Cache strategy**: the indexer state cache lives on the per-layer `DynamicIndexedLayer`
    (or the `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-06)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** (-0.5)
        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, q_resid: torch.Tensor, attention_mask: torch.BoolTensor, past_key_values: Cache | None) -> torch.LongTensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA) based on grouping pools (and tails).

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            attention_mask: Local boolean padding mask of shape `[B, S]`.
            past_key_values: Cache object containing the indexer state cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]` (or `[B, S, 2*topk - 1]` with tail).
            The eager / SDPA paths turn these into an additive sparse mask.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)
        kv_len = seq_len
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()
        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(valid_keys=valid_keys, q_length=seq_len, current_length=current_length)
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * self.n_heads ** (-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1))
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
            output_width += self.index_kpool - 1
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)

    def get_visible_tokens(self, valid_keys: torch.BoolTensor, q_length: int, current_length: int) -> torch.BoolTensor:
        """
        Check whether a token is visible for Q based on
            - Causality
            - Padding status
                - Valid keys which is a (cached) variation of the attention mask
        """
        device = valid_keys.device
        kv_positions = torch.arange(valid_keys.shape[-1], device=device)
        q_positions = current_length - q_length + torch.arange(q_length, device=device)
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]
        return causal & valid_keys[:, None, :]

    def get_pooled_states(self, packed_states: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        (Re-)Build compressed k-pool candidates from the packed state.

        Pooling starts at the first real token, not raw slot 0. This is the part
        that makes:

            [P, P, A, B, C, D, ...]

        behave like:

            [A, B, C, D, ...]

        for k-pool grouping.
        """
        keys, gate_scores, valid_keys = torch.split(packed_states, [self.head_dim, self.head_dim, 1], dim=-1)
        valid_keys = valid_keys.bool().squeeze(-1)
        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device
        first_key = torch.where(valid_keys.any(-1), valid_keys.long().argmax(-1), torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)
        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]
        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)
        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~grouped_valid_keys[..., None], float('-inf'))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
        pool_keys = (probabilities * grouped_keys).sum(dim=2)
        keep = pool_valid.any(0)
        return (pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep])

    def append_visible_tail(self, topk_indices: torch.Tensor, token_visible: torch.BoolTensor, key_valid: torch.BoolTensor) -> torch.Tensor:
        """
        Append the current incomplete pool as raw token indices.
        So we if we have a pool size of 4:
            - [P P A B C D E F]
            - [A B C D] (selected pool)
            - [E F] (appended tail)
        """
        if (max_tail_width := (self.index_kpool - 1)) == 0:
            return topk_indices
        batch_size, _, kv_length = token_visible.shape
        device = token_visible.device
        first_key = torch.where(key_valid.any(-1), key_valid.long().argmax(-1), torch.full((batch_size,), kv_length, dtype=torch.long, device=device))
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(self.index_kpool)
        tail_offsets = torch.arange(max_tail_width, device=device)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(kv_length)
        kv_idx = tail_indices.clamp(0, kv_length - 1)
        tail_visible = token_visible.gather(dim=-1, index=kv_idx)
        tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)
        return torch.cat([topk_indices, tail_indices], dim=-1)
