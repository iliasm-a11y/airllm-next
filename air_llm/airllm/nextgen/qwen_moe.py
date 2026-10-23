"""Experimental, streamed Qwen3.5/3.8 text MoE with exact score-based routing.

Equations/layout follow Transformers Qwen3_5MoeSparseMoeBlock (Apache-2.0;
Copyright 2025 The Qwen Team and The HuggingFace Inc. team). This numerical
implementation shares the generic offset reader and bounded weight cache.
It does not certify a complete Qwen3.8 checkpoint or a Metal kernel.
"""
import sys
from time import perf_counter
import numpy as np


TORCH_ROUTER_IMPORT_ALLOWANCE_BYTES = 384 << 20


def qwen_route(mx, x, weight, config, *, tie_policy='error',
               workspace_bytes=None, reserve_bytes=0, stats=None):
    """Official softmax top-k + selected-score normalization, without pruning.

    Ties remain errors by default. ``torch_cpu_reference`` explicitly opts into
    the complete installed CPU Torch gate (linear, FP32 softmax, topk, normalize)
    on the complete current input chunk and router weights, injecting only
    affected rows. Keeping the chunk shape also keeps CPU kernel selection.
    Native matmul can itself round
    distinct CPU logits into a tie, so copying probabilities alone is insufficient.
    This is a local reference convention, not stable tie ordering across Torch
    builds or agreement with CUDA. Logits never break rounded probability ties.
    Untied selections and the softmax implementation are unchanged.
    """
    if tie_policy not in ('error', 'torch_cpu_reference'):
        raise ValueError('Unsupported Qwen router tie policy')
    if tie_policy == 'torch_cpu_reference':
        if type(workspace_bytes) is not int or workspace_bytes < 1:
            raise ValueError('Qwen reference tie policy requires a positive workspace_bytes budget')
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError('Qwen router reserve_bytes must be a nonnegative integer')
    stats = stats if stats is not None else {}
    experts = config['num_experts']; top_k = config['num_experts_per_tok']
    if not 0 < top_k <= experts:
        raise ValueError('Invalid Qwen top-k expert count')
    if tuple(weight.shape) != (experts, x.shape[-1]):
        raise ValueError('Invalid Qwen router matrix')
    flat = x.reshape(-1, x.shape[-1])
    routing_input = flat.astype(weight.dtype)
    logits = routing_input @ weight.T
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    order = mx.argsort(-probabilities, axis=-1)
    selected = order[:, :top_k]
    scores = mx.take_along_axis(probabilities, selected, axis=-1)
    reference_scores = None
    if top_k < experts:
        begin = perf_counter()
        next_score = mx.take_along_axis(probabilities, order[:, top_k:top_k+1], axis=-1)
        tied = scores[:, -1:] == next_score
        has_tie = bool(mx.any(tied).item())
        stats['qwen_router_tie_checks'] = stats.get('qwen_router_tie_checks', 0) + 1
        stats['qwen_router_tie_check_seconds'] = stats.get('qwen_router_tie_check_seconds', 0.) + perf_counter() - begin
        if has_tie:
            if tie_policy == 'error':
                raise ValueError('Qwen router top-k boundary tie: reference tie-breaking is not validated')
            begin = perf_counter()
            affected_count = int(mx.sum(tied).item())
            # Router arrays already live plus conservative device gather, host
            # copies, CPU topk scratch, index conversion and scatter allowance.
            # Torch's runtime/import is separate and covered by admission/RSS.
            base = sum(a.nbytes for a in (logits, probabilities, order, selected, scores, next_score))
            # Four FP32-equivalent router/input copies cover host transfer,
            # dtype conversion and conservative CPU linear scratch. This gate
            # is small (2 MiB at 256x2048); no expert bank is copied to Torch.
            extra = (4 * 4 * (experts * weight.shape[-1] + flat.shape[0] * flat.shape[-1])
                     + flat.shape[0] * (32 * experts + 64 * top_k + 8))
            if base + extra > workspace_bytes:
                stats['qwen_router_tie_budget_rejections'] = stats.get('qwen_router_tie_budget_rejections', 0) + 1
                raise MemoryError('Qwen reference tie fallback exceeds workspace_bytes')
            stats['qwen_router_tie_workspace_peak_bytes'] = max(stats.get('qwen_router_tie_workspace_peak_bytes', 0), base + extra)
            import psutil
            available = psutil.virtual_memory().available
            first_import = 'torch' not in sys.modules
            allowance = TORCH_ROUTER_IMPORT_ALLOWANCE_BYTES if first_import else 0
            stats['qwen_router_tie_available_bytes'] = available
            stats['qwen_router_tie_import_allowance_bytes'] = allowance
            if available < reserve_bytes + extra + allowance:
                if first_import:
                    stats['qwen_router_tie_import_admission_rejections'] = stats.get('qwen_router_tie_import_admission_rejections', 0) + 1
                    raise MemoryError('System reserve reached before Qwen Torch reference import')
                stats['qwen_router_tie_system_reserve_rejections'] = stats.get('qwen_router_tie_system_reserve_rejections', 0) + 1
                raise MemoryError('System reserve reached before Qwen reference gate allocation')
            import_start = perf_counter()
            try:
                import torch
            except ImportError as error:
                raise RuntimeError('Qwen reference tie policy requires installed CPU PyTorch') from error
            stats['qwen_router_tie_reference_import_seconds'] = stats.get('qwen_router_tie_reference_import_seconds', 0.) + perf_counter() - import_start
            stats['qwen_router_tie_reference_version'] = str(torch.__version__)
            stats['qwen_router_tie_reference_device'] = 'cpu'
            stats['qwen_router_tie_reference_operation'] = 'linear_softmax_topk_normalize'
            dtype_name = ('float32' if weight.dtype == mx.float32 else
                          'float16' if weight.dtype == mx.float16 else
                          'bfloat16' if weight.dtype == mx.bfloat16 else None)
            if dtype_name is None:
                raise NotImplementedError('Qwen reference gate requires float32/float16/bfloat16 router weights')
            copy_start = perf_counter()
            affected = np.flatnonzero(np.asarray(tied).reshape(-1)).astype(np.int32)
            row_ids = mx.array(affected)
            host_input = np.array(routing_input.astype(mx.float32), dtype=np.float32, copy=True)
            host_weight = np.array(weight.astype(mx.float32), dtype=np.float32, copy=True)
            stats['qwen_router_tie_host_copy_seconds'] = stats.get('qwen_router_tie_host_copy_seconds', 0.) + perf_counter() - copy_start
            router_start = perf_counter()
            with torch.inference_mode():
                linear_start = perf_counter()
                reference_logits = torch.nn.functional.linear(
                    torch.from_numpy(host_input).to(getattr(torch, dtype_name)),
                    torch.from_numpy(host_weight).to(getattr(torch, dtype_name)))
                stats['qwen_router_tie_cpu_linear_seconds'] = stats.get('qwen_router_tie_cpu_linear_seconds', 0.) + perf_counter() - linear_start
                softmax_start = perf_counter()
                reference_probabilities = torch.softmax(reference_logits, dim=-1, dtype=torch.float32)
                stats['qwen_router_tie_cpu_softmax_seconds'] = stats.get('qwen_router_tie_cpu_softmax_seconds', 0.) + perf_counter() - softmax_start
                topk_start = perf_counter()
                values, indices = torch.topk(reference_probabilities, top_k, dim=-1)
                values = values / values.sum(dim=-1, keepdim=True)
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError('Non-finite Qwen CPU reference gate scores')
                indices = indices.numpy()[affected].astype(np.int32)
                reference_scores = np.array(values.numpy()[affected], dtype=np.float32, copy=True)
                stats['qwen_router_tie_cpu_topk_normalize_seconds'] = stats.get('qwen_router_tie_cpu_topk_normalize_seconds', 0.) + perf_counter() - topk_start
            stats['qwen_router_tie_cpu_router_seconds'] = stats.get('qwen_router_tie_cpu_router_seconds', 0.) + perf_counter() - router_start
            selected[row_ids] = mx.array(indices)
            mx.eval(selected)
            scores = mx.take_along_axis(probabilities, selected, axis=-1)
            stats['qwen_router_tie_fallback_calls'] = stats.get('qwen_router_tie_fallback_calls', 0) + 1
            stats['qwen_router_tie_fallback_rows'] = stats.get('qwen_router_tie_fallback_rows', 0) + affected_count
            stats['qwen_router_tie_reference_rows'] = stats.get('qwen_router_tie_reference_rows', 0) + flat.shape[0]
            stats['qwen_router_tie_fallback_seconds'] = stats.get('qwen_router_tie_fallback_seconds', 0.) + perf_counter() - begin
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    if reference_scores is not None:
        scores[row_ids] = mx.array(reference_scores)
    return selected, scores.astype(logits.dtype)


class QwenSparseMLP:
    """Demand-load each actually selected expert once per input block.

    Both official packed expert banks and equivalent split expert files work.
    No predicted expert is substituted for a router-selected expert.
    """
    def __init__(self, mx, reader, config, layer):
        self.mx=mx; self.r=reader; self.c=config; self.layer=layer
        self.p=f'model.layers.{layer}.mlp.'
        if config.get('hidden_act','silu') != 'silu':
            raise NotImplementedError('Qwen MoE currently implements SiLU only')
        self.fused=self.p+'experts.gate_up_proj' in reader.cp.tensors
        self.routes={}

    def dense(self, x, prefix):
        gate=self.r.linear(x,prefix+'gate_proj.weight')
        up=self.r.linear(x,prefix+'up_proj.weight')
        return self.r.linear((gate*self.mx.sigmoid(gate))*up,prefix+'down_proj.weight')

    def expert_keys(self, expert):
        if self.fused:
            return [(self.p+'experts.gate_up_proj',expert),(self.p+'experts.down_proj',expert)]
        return [(self.p+f'experts.{expert}.{name}.weight',None)
                for name in ('gate_proj','up_proj','down_proj')]

    def __call__(self, x):
        mx=self.mx
        if x.ndim!=3 or x.shape[-1]!=self.c['hidden_size']:
            raise ValueError('Qwen MoE expects [batch, sequence, hidden_size]')
        ids,weights=qwen_route(mx,x,self.r.get(self.p+'gate.weight'),self.c)
        host=np.asarray(ids); flat=x.reshape(-1,x.shape[-1]); selected=np.unique(host)
        if hasattr(self.r,'observe_layer'):self.r.observe_layer(self.layer,selected)
        self.routes={int(e):int(np.count_nonzero(host==e)) for e in selected}
        output=mx.zeros_like(flat)
        for j,expert in enumerate(selected):
            expert=int(expert); keys=self.expert_keys(expert)
            w0=self.r.get(*keys[0])
            if len(keys)>1:self.r.prefetch(*keys[1])
            token_index,topk_index=np.where(host==expert)
            ti=mx.array(token_index); ki=mx.array(topk_index); current=flat[ti]
            if self.fused:
                gu=self.r.linear_weight(current,w0); gate,up=mx.split(gu,2,axis=-1)
                down=keys[1]
            else:
                gate=self.r.linear_weight(current,w0); up=self.r.linear(current,*keys[1]);down=keys[2]
                self.r.prefetch(*down)
            if j+1<len(selected):self.r.prefetch(*self.expert_keys(int(selected[j+1]))[0])
            value=self.r.linear((gate*mx.sigmoid(gate))*up,*down)
            value=value*weights[ti,ki,None]
            # Each token appears at most once for a given expert. Realize this
            # update before releasing the expert so lazy graphs stay bounded.
            output[ti]=output[ti]+value.astype(output.dtype);mx.eval(output)
            del w0
        shared=self.dense(flat,self.p+'shared_expert.')
        shared_gate=mx.sigmoid(self.r.linear(flat,self.p+'shared_expert_gate.weight'))
        result=(output+shared*shared_gate).reshape(x.shape)
        mx.eval(result)
        return result
