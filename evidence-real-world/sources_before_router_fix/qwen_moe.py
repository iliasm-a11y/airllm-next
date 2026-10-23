"""Experimental, streamed Qwen3.5/3.8 text MoE with exact score-based routing.

Equations/layout follow Transformers Qwen3_5MoeSparseMoeBlock (Apache-2.0;
Copyright 2025 The Qwen Team and The HuggingFace Inc. team). This numerical
implementation shares the generic offset reader and bounded weight cache.
It does not certify a complete Qwen3.8 checkpoint or a Metal kernel.
"""
import numpy as np


def qwen_route(mx, x, weight, config):
    """Official softmax top-k + selected-score normalization, without pruning.

    Torch and MLX do not promise the same tie-breaking for top-k. A tie across
    the selection boundary is rejected explicitly until a matched backend tie
    contract is validated; an arbitrary different expert would change outputs.
    """
    experts = config['num_experts']; top_k = config['num_experts_per_tok']
    if not 0 < top_k <= experts:
        raise ValueError('Invalid Qwen top-k expert count')
    if tuple(weight.shape) != (experts, x.shape[-1]):
        raise ValueError('Invalid Qwen router matrix')
    flat = x.reshape(-1, x.shape[-1])
    logits = flat.astype(weight.dtype) @ weight.T
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    order = mx.argsort(-probabilities, axis=-1)
    selected = order[:, :top_k]
    scores = mx.take_along_axis(probabilities, selected, axis=-1)
    if top_k < experts:
        next_score = mx.take_along_axis(probabilities, order[:, top_k:top_k+1], axis=-1)
        if bool(mx.any(scores[:, -1:] == next_score).item()):
            raise ValueError('Qwen router top-k boundary tie: reference tie-breaking is not validated')
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
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
