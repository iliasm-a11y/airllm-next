"""Opt-in bounded input-block selection for streamed Qwen prefill.

This planner reuses the existing chunked runtime: it does not change layers,
experts, weights or numerical kernels. A larger block reduces model passes but
changes GEMM shapes, so numerical/route acceptance remains a separate test.
Estimates are conservative admission heuristics; the existing allocator, IO,
workspace, KV and external process guards remain authoritative.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillDecision:
    chunk_tokens: int
    estimated_workspace_bytes: int
    estimated_context_bytes: int
    reason: str


@dataclass(frozen=True)
class HyperPrefillPolicy:
    """Choose the largest fitting block without increasing any memory cap.

    Half of workspace is kept for packed-tile imports and matrix scratch. The
    other half covers retained routed outputs, activations and attention. This
    is intentionally a bound on admitted block size, not an allocator override.
    Decode always uses one token. ``max_chunk_tokens`` bounds the opt-in change.
    """
    max_chunk_tokens: int = 128

    def __post_init__(self):
        if type(self.max_chunk_tokens) is not int or self.max_chunk_tokens < 1:
            raise ValueError('HyperPrefill maximum chunk must be a positive integer')

    @staticmethod
    def estimates(config, position, tokens, batch_size=1, dtype_bytes=4):
        c = config
        h = c['hidden_size']; k = c['num_experts_per_tok']
        m = c['moe_intermediate_size']; total = position + tokens
        elements = batch_size * tokens
        # Match the current grouped-MoE retention charge (FP32) and include
        # two residual/normalization activation payloads outside that charge.
        routed = elements * k * (2 * h + 4 * m) * 4
        activations = 2 * elements * h * max(4, dtype_bytes)
        heads = c['num_attention_heads']; dim = c['head_dim']
        attention = (3 * batch_size * heads * tokens * total * 4
                     + 2 * batch_size * heads * total * dim * dtype_bytes)
        workspace = activations + max(routed, attention)
        kinds = c['layer_types']
        full_layers = kinds.count('full_attention')
        linear_layers = kinds.count('linear_attention')
        kh = c['linear_num_key_heads']; vh = c['linear_num_value_heads']
        kd = c['linear_key_head_dim']; vd = c['linear_value_head_dim']
        recurrent = batch_size * vh * kd * vd * 4
        conv = (batch_size * c['linear_conv_kernel_dim']
                * (2 * kh * kd + vh * vd) * dtype_bytes)
        kv = 2 * batch_size * c['num_key_value_heads'] * total * dim * dtype_bytes
        context = linear_layers * (recurrent + conv) + full_layers * kv
        # Cache replacement can coexist with its predecessor and an entry
        # snapshot. The latter is separately supplied to select().
        context_transient = max(recurrent, conv, kv)
        return workspace, context, context_transient

    def select(self, config, options, *, cache_position, remaining_tokens,
               batch_size=1, available_bytes, cache_payload_bytes=0,
               retained_bytes=0, dtype_bytes=4):
        for name, value, minimum in (
                ('cache_position', cache_position, 0),
                ('remaining_tokens', remaining_tokens, 1),
                ('batch_size', batch_size, 1),
                ('available_bytes', available_bytes, 0),
                ('cache_payload_bytes', cache_payload_bytes, 0),
                ('retained_bytes', retained_bytes, 0),
                ('dtype_bytes', dtype_bytes, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(name + ' is invalid for HyperPrefill selection')
        maximum = min(remaining_tokens, self.max_chunk_tokens,
                      options.max_context - cache_position,
                      config['max_position_embeddings'] - cache_position)
        if maximum < 1:
            raise MemoryError('HyperPrefill context limit exceeded')
        # Estimates are monotone in tokens: binary search needs no model/IO work.
        def admitted(tokens):
            workspace, context, transient = self.estimates(
                config, cache_position, tokens, batch_size, dtype_bytes)
            fits = (workspace <= options.workspace_bytes // 2
                    and context + transient + retained_bytes <= options.kv_bytes
                    and available_bytes >= options.reserve_bytes + workspace
                    + max(0, context - cache_payload_bytes))
            return fits, workspace, context + transient + retained_bytes
        low, high, chosen = 1, maximum, None
        while low <= high:
            middle = (low + high) // 2
            fits, workspace, context = admitted(middle)
            if fits:
                chosen = (middle, workspace, context)
                low = middle + 1
            else:
                high = middle - 1
        if chosen is None:
            raise MemoryError('HyperPrefill cannot fit one token within existing budgets')
        tokens, workspace, context = chosen
        reason = ('decode' if remaining_tokens == 1 else
                  'request_or_cap' if tokens == maximum else 'memory_budget')
        return PrefillDecision(tokens, workspace, context, reason)
