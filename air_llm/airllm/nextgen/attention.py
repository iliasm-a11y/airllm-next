"""GLM-5.3 DSA with exact shared indexers and algebraically absorbed MLA.

Cache holds latent KV (rank + RoPE), never the expanded heads. Only the keys
selected by DSA are gathered. Absorbing W_k and W_v is an algebraic identity,
not a change to sparse selection, learned weights or the softmax definition.
"""
import math
from dataclasses import dataclass


def rms(mx,x,weight,eps=1e-6):
    # MLX fused RMSNorm follows the same float32 reduction as Transformers.
    return mx.fast.rms_norm(x,weight.astype(x.dtype),eps)


def rope(mx,x,offset,theta,dim):
    """Official GLM: rotate adjacent pairs, output [rotated-even, rotated-odd]."""
    t=mx.arange(offset,offset+x.shape[-2],dtype=mx.float32)
    inv=theta**(-mx.arange(0,dim,2,dtype=mx.float32)/dim)
    a=t[:,None]*inv[None,:];c=mx.cos(a).astype(x.dtype);s=mx.sin(a).astype(x.dtype)
    even=x[...,0::2];odd=x[...,1::2]
    return mx.concatenate([even*c-odd*s,odd*c+even*s],axis=-1)


class LatentKV:
    def __init__(self,mx,config,budget,limit,dtype):
        self.mx=mx;self.cfg=config;self.budget=budget;self.limit=limit;self.dtype=dtype
        self.layers={};self.position=0;self.bytes=0;self.peak_bytes=0

    def append(self,layer,latent,rot,index=None):
        mx=self.mx;n=latent.shape[1];stop=self.position+n
        if stop>self.limit:raise MemoryError(f'Context exceeds configured safe limit {self.limit}')
        widths=[latent.shape[-1],rot.shape[-1]]+([index.shape[-1]] if index is not None else [])
        capacity=min(self.limit,math.ceil(stop/128)*128)
        old=self.layers.get(layer)
        old_bytes=sum(v.nbytes for v in old) if old else 0
        if old is None or old[0].shape[1]<stop:
            predicted=sum(widths)*capacity*latent.itemsize
            if self.bytes-old_bytes+predicted>self.budget:raise MemoryError('Latent KV budget exhausted')
            arrays=[mx.zeros((1,capacity,w),dtype=latent.dtype) for w in widths]
            if old:
                for a,b in zip(arrays,old):a[:,:self.position]=b[:,:self.position]
            self.layers[layer]=arrays;self.bytes+=predicted-old_bytes;self.peak_bytes=max(self.peak_bytes,self.bytes)
        arrays=self.layers[layer]
        for target,value in zip(arrays,[latent,rot]+([index] if index is not None else [])):
            target[:,self.position:stop]=value
        return [a[:,:stop] for a in arrays]

    def reset(self):self.layers.clear();self.position=0;self.bytes=0


class DSAAttention:
    def __init__(self,mx,reader,config,layer):
        self.mx=mx;self.r=reader;self.c=config;self.layer=layer
        self.p=f'model.layers.{layer}.self_attn.'

    def linear(self,x,name):return self.r.linear(x,self.p+name+'.weight')
    def norm(self,x,name,eps=1e-6):return rms(self.mx,x,self.r.get(self.p+name+'.weight'),eps)

    def __call__(self,x,cache,previous):
        mx=self.mx;c=self.c;B,S,_=x.shape;H=c['num_attention_heads']
        N=c['qk_nope_head_dim'];D=c['qk_rope_head_dim'];V=c['v_head_dim'];R=c['kv_lora_rank']
        theta=c['rope_parameters']['rope_theta'];offset=cache.position
        qr=self.norm(self.linear(x,'q_a_proj'),'q_a_layernorm')
        q=self.linear(qr,'q_b_proj').reshape(B,S,H,N+D).transpose(0,2,1,3)
        qp=q[...,:N];qrot=rope(mx,q[...,N:],offset,theta,D)
        kv=self.linear(x,'kv_a_proj_with_mqa')
        latent=self.norm(kv[...,:R],'kv_a_layernorm')
        krot=rope(mx,kv[...,R:][:,None,:,:],offset,theta,D)[:,0]
        index_key=None;iq=None
        if c['indexer_types'][self.layer]=='full':
            IH=c['index_n_heads'];ID=c['index_head_dim']
            iq=self.linear(qr,'indexer.wq_b').reshape(B,S,IH,ID).transpose(0,2,1,3)
            ir=rope(mx,iq[...,:D],offset,theta,D)
            iq=mx.concatenate([ir,iq[...,D:]],axis=-1).transpose(0,2,1,3)
            ik=self.linear(x,'indexer.wk')
            w=self.r.get(self.p+'indexer.k_norm.weight');b=self.r.get(self.p+'indexer.k_norm.bias')
            ik=mx.fast.layer_norm(ik,w.astype(ik.dtype),b.astype(ik.dtype),1e-6)
            ikrot=rope(mx,ik[...,:D][:,None,:,:],offset,theta,D)[:,0]
            index_key=mx.concatenate([ikrot,ik[...,D:]],axis=-1)
        stored=cache.append(self.layer,latent,krot,index_key)
        all_latent,all_rot=stored[:2];T=all_latent.shape[1]
        if iq is not None:
            ik=stored[2]
            score=mx.einsum('bshd,btd->bsht',iq.astype(mx.float32),ik.astype(mx.float32))*c['index_head_dim']**-.5
            score=mx.maximum(score,0)
            head_w=self.linear(x,'indexer.weights_proj').astype(mx.float32)*c['index_n_heads']**-.5
            score=mx.sum(score*head_w[...,None],axis=2)
            causal=mx.arange(T)[None,:] <= (mx.arange(S)[:,None]+offset)
            score=mx.where(causal[None],score,-float('inf'))
            K=min(c['index_topk'],T)
            # Partial selection, not a full sort of an enormous context.
            selected=mx.argpartition(-score,K-1,axis=-1)[...,:K]
        else:
            if previous is None:raise ValueError('Shared DSA layer requires actual preceding indexer selection')
            selected=previous;K=selected.shape[-1]
        # Gather latent representations before expanding heads. B=1 is enforced
        # by the model API; S and K remain device-side dynamic array dimensions.
        cl=all_latent[:,selected[0],:];cr=all_rot[:,selected[0],:]
        kvw=self.r.get(self.p+'kv_b_proj.weight')
        if not hasattr(kvw,'reshape'):raise NotImplementedError('Quantizing kv_b_proj is not supported by absorbed attention')
        kvw=kvw.reshape(H,N+V,R);wk=kvw[:,:N];wv=kvw[:,N:]
        # Full-precision accumulation is intentional in the reference path.
        # In reduced precision all operations use the requested MLX dtype.
        qa=mx.einsum('bhsd,hdr->bhsr',qp.astype(wk.dtype),wk)
        scores=(mx.einsum('bhsr,bskr->bhsk',qa,cl.astype(qa.dtype))+
                mx.einsum('bhsd,bskd->bhsk',qrot,cr.astype(qrot.dtype)))*(N+D)**-.5
        valid=selected <= (mx.arange(S)[None,:,None]+offset)
        scores=mx.where(valid[:,None,:,:],scores,-float('inf'))
        prob=mx.softmax(scores.astype(mx.float32),axis=-1).astype(x.dtype)
        value_latent=mx.einsum('bhsk,bskr->bhsr',prob,cl.astype(prob.dtype))
        values=mx.einsum('bhsr,hvr->bhsv',value_latent.astype(wv.dtype),wv)
        values=values.transpose(0,2,1,3).reshape(B,S,H*V)
        return self.linear(values,'o_proj').astype(x.dtype),selected
