"""Faithful sigmoid/correction-bias grouped GLM routing and sparse expert IO."""
from collections import Counter
import numpy as np


def route(mx,x,weight,correction,c):
    logits=x.reshape(-1,x.shape[-1]).astype(mx.float32)@weight.astype(mx.float32).T
    scores=mx.sigmoid(logits);choice=scores+correction.astype(mx.float32)
    groups=c['n_group'];per=c['n_routed_experts']//groups
    if per<2:raise ValueError('GLM grouped router requires at least 2 experts per group')
    grouped=choice.reshape(-1,groups,per)
    gtop=mx.sort(grouped,axis=-1)[...,-2:];gs=mx.sum(gtop,axis=-1)
    gi=mx.argsort(-gs,axis=-1)[...,:c['topk_group']]
    mask=mx.any(mx.arange(groups)[None,:,None]==gi[:,None,:],axis=-1)
    choice=mx.where(mx.repeat(mask,per,axis=-1),choice,-float('inf'))
    ids=mx.argsort(-choice,axis=-1)[...,:c['num_experts_per_tok']]
    weights=mx.take_along_axis(scores,ids,axis=-1)
    if c['norm_topk_prob']:weights=weights/(mx.sum(weights,axis=-1,keepdims=True)+1e-20)
    return ids,weights*c['routed_scaling_factor']


class SparseMLP:
    def __init__(self,mx,reader,c,layer,routes):
        self.mx=mx;self.r=reader;self.c=c;self.layer=layer;self.p=f'model.layers.{layer}.mlp.';self.routes=routes
        self.fused=self.p+'experts.gate_up_proj' in reader.cp.tensors

    def dense(self,x,p):
        gate=self.r.linear(x,p+'gate_proj.weight');up=self.r.linear(x,p+'up_proj.weight')
        return self.r.linear((gate*self.mx.sigmoid(gate))*up,p+'down_proj.weight').astype(x.dtype)

    def expert_keys(self,e):
        if self.fused:return [(self.p+'experts.gate_up_proj',e),(self.p+'experts.down_proj',e)]
        return [(self.p+f'experts.{e}.{n}.weight',None) for n in ('gate_proj','up_proj','down_proj')]

    def __call__(self,x):
        if self.c['mlp_layer_types'][self.layer]=='dense':return self.dense(x,self.p)
        mx=self.mx;c=self.c
        ids,weights=route(mx,x,self.r.get(self.p+'gate.weight'),self.r.get(self.p+'gate.e_score_correction_bias'),c)
        # Exactly one mandatory route readback per MoE layer. The host must know
        # SSD offsets; predictions never replace router decisions.
        host=np.asarray(ids);flat=x.reshape(-1,x.shape[-1]);selected=np.unique(host)
        self.routes.setdefault(str(self.layer),Counter()).update(int(e) for e in host.reshape(-1))
        out=mx.zeros_like(flat)
        # Load each real expert once for the whole prefill chunk. No approximate
        # expert pruning; group token rows and scatter to unique token indices.
        for j,e in enumerate(selected):
            e=int(e);keys=self.expert_keys(e)
            # Demand-load gate, then overlap the next component/expert IO with
            # current compute; small sequential lookahead avoids speculation.
            w0=self.r.get(*keys[0])
            if len(keys)>1:self.r.prefetch(*keys[1])
            ti,ki=np.where(host==e);ti_m=mx.array(ti);ki_m=mx.array(ki)
            current=flat[ti_m]
            if self.fused:
                gu=self.r.linear_weight(current,w0);gate,up=mx.split(gu,2,axis=-1)
                downkey=keys[1]
            else:
                gate=self.r.linear_weight(current,w0);up=self.r.linear(current,*keys[1]);downkey=keys[2]
                self.r.prefetch(*downkey)
            if j+1<len(selected):self.r.prefetch(*self.expert_keys(int(selected[j+1]))[0])
            value=self.r.linear((gate*mx.sigmoid(gate))*up,*downkey)
            value=value*weights[ti_m,ki_m,None]
            # Each token appears at most once per expert; no duplicate scatter.
            out[ti_m]=out[ti_m]+value.astype(out.dtype)
            # Required weight lifetime boundary: lazy graphs cannot retain an
            # entire MoE bank. Eval per expert bounds live decoded weights.
            mx.eval(out)
            del w0
        shared=self.dense(x,self.p+'shared_experts.')
        return out.reshape(x.shape)+shared
