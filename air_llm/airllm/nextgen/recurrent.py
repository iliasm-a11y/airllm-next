"""Portable Qwen Gated DeltaNet recurrence for numerical validation only.

Equations follow Transformers torch_recurrent_gated_delta_rule, Apache-2.0,
Qwen/HuggingFace, source revision a672eb5f474b8b30173e9f96eab66adf292da635:
https://github.com/huggingface/transformers/blob/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

This primitive is NOT a Qwen model backend. Convolution, learned projections,
output gating, full attention, MoE and MTP still need integration and validation.
"""
from dataclasses import dataclass
import numpy as np


def gated_delta_rule(query, key, value, decay, beta, *, initial_state=None,
                     normalize_qk=False, return_state=True):
    """FP32 state update for [batch, sequence, heads, dimension] inputs.

    Q/K heads may be repeated to match value heads, as the official model caller
    does before its reference recurrence. Inputs and initial_state are unmodified.
    This supports arbitrary chunk boundaries and a caller-owned rollback snapshot.
    """
    source_query = np.asarray(query)
    output_dtype = source_query.dtype
    q, k, v = [np.asarray(x, dtype=np.float32) for x in (query, key, value)]
    g, b = [np.asarray(x, dtype=np.float32) for x in (decay, beta)]
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError('Q/K/V must have compatible [batch, sequence, heads, dimension] shapes')
    batch, length, heads, kd = q.shape
    vh, vd = v.shape[-2:]
    if min(batch, heads, kd, vh, vd) < 1 or vh % heads:
        raise ValueError('Value heads must be a positive multiple of Q/K heads')
    if g.shape != (batch, length, vh) or b.shape != g.shape:
        raise ValueError('decay and beta must be [batch, sequence, value_heads]')
    if not all(np.isfinite(x).all() for x in (q, k, v, g, b)) or (g > 0).any():
        raise ValueError('Inputs must be finite and log-space decay must be nonpositive')
    if normalize_qk:
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
    if vh != heads:
        q = np.repeat(q, vh // heads, axis=2)
        k = np.repeat(k, vh // heads, axis=2)
    q = q / np.float32(kd ** 0.5)
    shape = (batch, vh, kd, vd)
    if initial_state is None:
        state = np.zeros(shape, dtype=np.float32)
    else:
        state = np.array(initial_state, dtype=np.float32, copy=True)
        if state.shape != shape or not np.isfinite(state).all():
            raise ValueError('Invalid recurrent initial_state')
    out = np.empty((batch, length, vh, vd), dtype=np.float32)
    for t in range(length):
        state = state * np.exp(g[:, t])[..., None, None]
        predicted = np.sum(state * k[:, t, :, :, None], axis=-2)
        delta = (v[:, t] - predicted) * b[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        out[:, t] = np.sum(state * q[:, t, :, :, None], axis=-2)
    return out.astype(output_dtype), state if return_state else None


@dataclass
class RecurrentSnapshot:
    position: int
    recurrent: dict
    convolution: dict


class RecurrentState:
    """Explicit snapshots prevent treating overwritten recurrent state as a KV tail.

    Snapshot payload is intentionally accounted. This correctness utility uses
    copies; it is not a bounded-memory production implementation for 92 layers.
    """
    def __init__(self):
        self.position = 0
        self.recurrent = {}
        self.convolution = {}

    @property
    def nbytes(self):
        return sum(a.nbytes for bank in (self.recurrent, self.convolution) for a in bank.values())

    def snapshot(self, max_bytes=None):
        if max_bytes is not None and self.nbytes > max_bytes:
            raise MemoryError('Recurrent rollback snapshot exceeds budget')
        return RecurrentSnapshot(self.position,
                                 {k: np.array(v, copy=True) for k, v in self.recurrent.items()},
                                 {k: np.array(v, copy=True) for k, v in self.convolution.items()})

    def restore(self, snapshot):
        if not isinstance(snapshot, RecurrentSnapshot):
            raise TypeError('Recurrent rollback requires a state snapshot, not a token offset')
        self.position = snapshot.position
        self.recurrent = {k: np.array(v, copy=True) for k, v in snapshot.recurrent.items()}
        self.convolution = {k: np.array(v, copy=True) for k, v in snapshot.convolution.items()}
