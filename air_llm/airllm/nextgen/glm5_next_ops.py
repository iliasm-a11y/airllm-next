"""CPU numerical oracle for the GLM-5.3-Flash text path.

This module deliberately has no MLX/Metal or model-loading side effects. It is a
reference for validating an accelerated runtime, not a claim of full inference.
Matrix weights use [output, input]; KDA uses [batch, tokens, heads, channels].
Equations follow Transformers 1b754e9a4636191852a1618fc588ad5c863ab4d5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    # Avoid overflow while retaining the sigmoid of the unchanged input.
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    maximum = np.max(x, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp = np.exp(x - maximum)
        out = exp / exp.sum(axis=axis, keepdims=True)
    # All-invalid pools have no contribution, as in torch.nan_to_num.
    return np.nan_to_num(out, nan=0.0)


def rms_norm(x, weight=None, eps=1e-5):
    dtype = np.asarray(x).dtype
    x32 = np.asarray(x, dtype=np.float32)
    out = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    out = out.astype(dtype)
    return out if weight is None else out * np.asarray(weight)


def clamped_swiglu(x, gate_weight, up_weight, down_weight, limit=10.0):
    """GLM's asymmetric gate clamp and symmetric up clamp, before SiLU."""
    gate = np.asarray(x) @ np.asarray(gate_weight).T
    up = np.asarray(x) @ np.asarray(up_weight).T
    gate = np.minimum(gate, limit)
    up = np.clip(up, -limit, limit)
    return ((gate * sigmoid(gate)) * up) @ np.asarray(down_weight).T


@dataclass(frozen=True)
class Routing:
    logits: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


def route_experts(x, router_weight, correction_bias, *, top_k=8,
                  n_group=1, topk_group=1, normalize=True, scale=2.5):
    """Exact sigmoid selection, correction-only choice bias and raw-score weights.

    Ties are resolved by increasing expert id. Torch/Metal top-k tie ordering is
    backend-specific; equality tests must avoid ties or compare selected sets.
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(router_weight, dtype=np.float32)
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("router weight must have [experts, hidden] shape")
    experts = weight.shape[0]
    bias = np.asarray(correction_bias, dtype=np.float32)
    if bias.shape != (experts,) or experts % n_group or not 1 <= top_k <= experts:
        raise ValueError("invalid expert count, bias shape, groups or top-k")
    if not 1 <= topk_group <= n_group or experts // n_group < 2:
        raise ValueError("router requires at least two experts per valid group")
    if top_k > topk_group * (experts // n_group):
        raise ValueError("top-k exceeds the experts in selected groups")
    logits = x.reshape(-1, x.shape[-1]) @ weight.T
    scores = sigmoid(logits)
    choice = scores + bias
    grouped = choice.reshape(-1, n_group, experts // n_group)
    group_scores = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
    selected_groups = np.argsort(-group_scores, axis=-1, kind="stable")[:, :topk_group]
    group_mask = np.zeros(group_scores.shape, dtype=bool)
    np.put_along_axis(group_mask, selected_groups, True, axis=-1)
    mask = np.repeat(group_mask, experts // n_group, axis=-1)
    choice = np.where(mask, choice, -np.inf)
    indices = np.argsort(-choice, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return Routing(logits, weights * scale, indices)


def mhc_pre(streams, fn, base, scale, *, eps=1e-6, norm_eps=1e-5, iterations=20):
    """Return (post, comb, collapsed) matching the upstream mHC ordering."""
    streams = np.asarray(streams)
    if streams.ndim < 3:
        raise ValueError("mHC requires [..., streams, hidden] dimensions")
    hc, hidden = streams.shape[-2:]
    fn, base, scale = map(np.asarray, (fn, base, scale))
    if fn.shape != ((2 + hc) * hc, hc * hidden):
        raise ValueError("invalid mHC projection shape")
    if base.shape != ((2 + hc) * hc,) or scale.shape != (3,) or iterations < 1:
        raise ValueError("invalid mHC base, scale or Sinkhorn iterations")
    flat = streams.astype(np.float32).reshape(*streams.shape[:-2], hc * hidden)
    mixed = rms_norm(flat, eps=norm_eps) @ fn.astype(np.float32).T
    pre = sigmoid(mixed[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * sigmoid(mixed[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixed[..., 2 * hc:] * scale[2] + base[2 * hc:]).reshape(*streams.shape[:-2], hc, hc)
    comb = softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    collapsed = (pre[..., None] * streams).sum(axis=-2).astype(streams.dtype)
    return post, comb, collapsed


def mhc_post(block_output, residual, post, comb):
    dtype = np.asarray(residual).dtype
    return (np.asarray(post).astype(dtype)[..., None] * np.asarray(block_output)[..., None, :]
            + np.swapaxes(np.asarray(comb).astype(dtype), -1, -2) @ residual)


def recurrent_kda(query, key, value, log_decay, beta, initial_state=None, *, normalize_qk=True):
    """Kimi Delta Attention with per-key-channel decay and FP32 recurrent state.

    This is not Qwen's gated delta recurrence. No weights are approximated, no
    experts are skipped, and the caller's incoming state is never mutated.
    """
    dtype = np.asarray(query).dtype
    q, k, v, g, b = [np.asarray(a, dtype=np.float32) for a in (query, key, value, log_decay, beta)]
    if q.ndim != 4 or q.shape != k.shape or g.shape != k.shape:
        raise ValueError("KDA query, key and log-decay must share [B,T,H,K] shape")
    if v.shape[:3] != q.shape[:3] or b.shape != q.shape[:3]:
        raise ValueError("invalid KDA value or beta shape")
    if normalize_qk:
        q = q / np.sqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    shape = (q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    state = np.zeros(shape, np.float32) if initial_state is None else np.array(initial_state, dtype=np.float32, copy=True)
    if state.shape != shape:
        raise ValueError("invalid KDA recurrent-state shape")
    output = np.zeros(v.shape, dtype=np.float32)
    for t in range(q.shape[1]):
        state = state * np.exp(g[:, t])[..., None]
        remembered = (state * k[:, t, :, :, None]).sum(axis=-2)
        delta = (v[:, t] - remembered) * b[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        output[:, t] = (state * q[:, t, :, :, None]).sum(axis=-2)
    return output.astype(dtype), state


def kda_log_decay(hidden_states, f_a, f_b, dt_bias, a_log, *, heads, lower_bound=-5.0):
    """Forget-gate projections, preserving the official safe lower-bound rule."""
    projected = (np.asarray(hidden_states) @ np.asarray(f_a).T) @ np.asarray(f_b).T
    if projected.shape[-1] % heads:
        raise ValueError("forget-gate channels must be divisible by heads")
    g = (projected.astype(np.float32) + np.asarray(dt_bias, np.float32)).reshape(
        *projected.shape[:-1], heads, projected.shape[-1] // heads)
    rate = np.exp(np.asarray(a_log, np.float32)).reshape(heads, 1)
    if lower_bound is not None:
        return lower_bound * sigmoid(rate * g)
    return -rate * np.logaddexp(0.0, g)


def causal_depthwise_conv1d(x, weight, state=None, *, activation=True):
    """Causal short convolution; state stores the K-1 previous raw projections."""
    x, weight = np.asarray(x), np.asarray(weight)
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError("expected x=[B,T,D] and weight=[D,K]")
    kernel = weight.shape[-1]
    if kernel < 1:
        raise ValueError("convolution kernel must be nonempty")
    expected = (x.shape[0], kernel - 1, x.shape[-1])
    state = np.zeros(expected, dtype=x.dtype) if state is None else np.asarray(state)
    if state.shape != expected:
        raise ValueError("invalid convolution state shape")
    history = np.concatenate([state, x], axis=1)
    out = np.zeros_like(x)
    for offset in range(kernel):
        out += history[:, offset:offset + x.shape[1]] * weight[:, offset]
    if activation:
        out = out * sigmoid(out)
    next_state = history[:, -kernel + 1:].copy() if kernel > 1 else history[:, :0].copy()
    return out, next_state


def kpool_select(query, keys, pool_gate_logits, head_weights, valid_keys, query_positions,
                 ape, *, top_k=2048, pool_size=4, include_tail=True):
    """Select raw cache indices from learned complete KPool candidates.

    Inputs are already projected: Q=[B,T,H,D], K/gates=[B,N,D],
    weights=[B,T,H], validity=[B,N], positions=[B,T], APE=[P,D].
    Pools align to the first valid key. Incomplete query-visible tails are always
    included when requested; -1 slots are invalid. The returned ordering of
    equal-scored pools is deterministic but not a backend top-k tie guarantee.
    """
    q, keys, gates, weights = [np.asarray(a) for a in (query, keys, pool_gate_logits, head_weights)]
    valid = np.asarray(valid_keys, dtype=bool)
    positions = np.asarray(query_positions)
    ape = np.asarray(ape)
    if pool_size < 1 or top_k < 1 or top_k % pool_size:
        raise ValueError("positive top-k must be divisible by pool size")
    if q.ndim != 4 or keys.ndim != 3 or gates.shape != keys.shape:
        raise ValueError("invalid KPool query, key or gate shapes")
    batch, tokens, heads, channels = q.shape
    if keys.shape[0] != batch or keys.shape[-1] != channels or weights.shape != q.shape[:3]:
        raise ValueError("KPool projection shapes do not agree")
    if valid.shape != keys.shape[:2] or positions.shape != (batch, tokens) or ape.shape != (pool_size, channels):
        raise ValueError("invalid KPool validity, position or APE shape")
    width = top_k + (pool_size - 1 if include_tail else 0)
    result = np.full((batch, tokens, width), -1, dtype=np.int32)
    for b in range(batch):
        locations = np.flatnonzero(valid[b])
        if not len(locations):
            continue
        first = int(locations[0])
        pools, pooled = [], []
        for start in range(first, keys.shape[1] - pool_size + 1, pool_size):
            indices = np.arange(start, start + pool_size)
            if not valid[b, indices].all():
                continue
            probabilities = softmax(gates[b, indices].astype(np.float32) + ape.astype(np.float32), axis=0)
            pooled.append((probabilities.astype(keys.dtype) * keys[b, indices]).sum(axis=0))
            pools.append(indices)
        for t in range(tokens):
            position = int(positions[b, t])
            if not 0 <= position < keys.shape[1] or not valid[b, position]:
                continue
            eligible = [i for i, indices in enumerate(pools) if indices[-1] <= position]
            selected = []
            if eligible:
                pool_keys = np.asarray([pooled[i] for i in eligible], dtype=np.float32)
                scores = np.maximum(q[b, t].astype(np.float32) @ pool_keys.T * channels ** -0.5, 0)
                scores = (weights[b, t].astype(np.float32)[:, None] * heads ** -0.5 * scores).sum(axis=0)
                chosen = np.argsort(-scores, kind="stable")[:top_k // pool_size]
                selected = np.concatenate([pools[eligible[i]] for i in chosen]).tolist()
            result[b, t, :len(selected)] = selected
            if include_tail:
                visible_count = int(valid[b, :position + 1].sum())
                tail_count = visible_count % pool_size
                tail_start = first + visible_count - tail_count
                for j in range(tail_count):
                    index = tail_start + j
                    if index <= position and valid[b, index]:
                        result[b, t, top_k + j] = index
    return result
