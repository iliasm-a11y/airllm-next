"""GLM-5.3 DSA with exact shared indexers and algebraically absorbed MLA.

Cache holds latent KV (rank + RoPE), never the expanded heads. Only the keys
selected by DSA are gathered. Absorbing W_k and W_v is an algebraic identity,
not a change to sparse selection, learned weights or the softmax definition.
"""
import math
from dataclasses import dataclass
from time import perf_counter
import numpy as np


def dsa_topk(mx, scores, k, *, workspace_bytes, workspace_in_use=0, stats=None):
    """Native selection with bounded CPU-reference handling of boundary ties.

    PyTorch CPU topk does not promise stable tie ordering. Selecting a different
    equal-score key can change DSA attention, so an exact finite tie crossing
    the K boundary uses the installed CPU reference's actual topk operation.
    This is a local CPU-reference convention, not a promise to reproduce CUDA
    topk or a different PyTorch build. Untied selections stay device-native.
    """
    selected = mx.argpartition(-scores, k-1, axis=-1)[..., :k]
    total = scores.shape[-1]
    if total <= k:
        return selected
    stats = stats if stats is not None else {}
    begin = perf_counter()
    values = mx.take_along_axis(scores, selected, axis=-1)
    boundary = mx.min(values, axis=-1)
    tied = mx.isfinite(boundary) & (
        mx.sum(scores == boundary[..., None], axis=-1)
        > mx.sum(values == boundary[..., None], axis=-1))
    # One explicit readback after T>K. This includes dependency evaluation;
    # the counter is elapsed sync/check time, not a pure kernel measurement.
    affected = np.flatnonzero(np.asarray(tied).reshape(-1)).astype(np.int32)
    stats['dsa_tie_checks'] = stats.get('dsa_tie_checks', 0)+1
    stats['dsa_tie_check_seconds'] = stats.get('dsa_tie_check_seconds', 0.)+perf_counter()-begin
    if not len(affected):
        return selected
    begin = perf_counter()
    # Account device gather, writable host scores, conservative CPU topk pair
    # workspace, outputs and scatter indices before allocating those buffers.
    # Import/runtime metadata is separate from this managed scratch bound.
    extra = len(affected)*(32*total+64*k)
    if workspace_in_use+extra > workspace_bytes:
        stats['dsa_tie_budget_rejections'] = stats.get('dsa_tie_budget_rejections', 0)+1
        raise MemoryError('DSA reference tie fallback exceeds workspace_bytes')
    stats['dsa_tie_extra_workspace_peak_bytes'] = max(stats.get('dsa_tie_extra_workspace_peak_bytes', 0), extra)
    copy_start = perf_counter()
    row_ids = mx.array(affected)
    host = np.array(scores.reshape(-1, total)[row_ids], dtype=np.float32, copy=True)
    stats['dsa_tie_host_copy_seconds'] = stats.get('dsa_tie_host_copy_seconds', 0.)+perf_counter()-copy_start
    import_start = perf_counter()
    import torch
    stats['dsa_tie_reference_import_seconds'] = stats.get('dsa_tie_reference_import_seconds', 0.)+perf_counter()-import_start
    topk_start = perf_counter()
    indices = torch.from_numpy(host).topk(k, dim=-1).indices.numpy().astype(np.int32)
    stats['dsa_tie_cpu_topk_seconds'] = stats.get('dsa_tie_cpu_topk_seconds', 0.)+perf_counter()-topk_start
    result = selected.reshape(-1, k)
    result[row_ids] = mx.array(indices)
    result = result.reshape(selected.shape)
    mx.eval(result)
    stats['dsa_tie_fallback_calls'] = stats.get('dsa_tie_fallback_calls', 0)+1
    stats['dsa_tie_fallback_rows'] = stats.get('dsa_tie_fallback_rows', 0)+len(affected)
    stats['dsa_tie_fallback_seconds'] = stats.get('dsa_tie_fallback_seconds', 0.)+perf_counter()-begin
    return result


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
            # Growing copies retain the old bank until the new evaluated bank
            # replaces it. Budget the overlap, not only steady-state storage.
            if self.bytes+predicted>self.budget:raise MemoryError('Latent KV budget exhausted during growth')
            self.peak_bytes=max(self.peak_bytes,self.bytes+predicted)
            arrays=[mx.zeros((1,capacity,w),dtype=latent.dtype) for w in widths]
            if old:
                for a,b in zip(arrays,old):a[:,:self.position]=b[:,:self.position]
            # The caller evaluates attention later. Materialize this copy here
            # so that its graph cannot retain old banks after the reservation.
            mx.eval(*arrays)
            self.layers[layer]=arrays;self.bytes+=predicted-old_bytes;self.peak_bytes=max(self.peak_bytes,self.bytes)
        arrays=self.layers[layer]
        for target,value in zip(arrays,[latent,rot]+([index] if index is not None else [])):
            target[:,self.position:stop]=value
        return [a[:,:stop] for a in arrays]

    def reset(self):self.layers.clear();self.position=0;self.bytes=0

    def truncate(self,position):
        """Discard an uncommitted tail without copying latent KV buffers.

        Attention only reads through position+new_input_length, and append
        overwrites the discarded tail before reading it. This contract applies
        to this append-only GLM cache; recurrent architecture states need their
        own snapshot/recompute mechanism and cannot use position rollback.
        """
        if not isinstance(position,int) or isinstance(position,bool) or not 0<=position<=self.position:
            raise ValueError('KV truncate position must be within the committed prefix')
        self.position=position


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
        T=offset+S;K=min(c['index_topk'],T)
        index_heads=c['index_n_heads'] if c['indexer_types'][self.layer]=='full' else 0
        # Bound the major temporary payloads before building the indexer graph:
        # FP32 index/attention scores and probabilities, selected indices and
        # gathered latent/RoPE keys. This excludes separately budgeted weights
        # and KV banks and is deliberately conservative about live overlap.
        workspace=4*B*S*(T*(index_heads+2)+K*(3*H+1))+B*S*K*(R+D)*x.itemsize
        if workspace>self.r.cfg.workspace_bytes:
            raise MemoryError('GLM attention workspace exceeds workspace_bytes; reduce prefill_chunk/context')
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
            selected=dsa_topk(mx,score,K,workspace_bytes=self.r.cfg.workspace_bytes,
                              workspace_in_use=workspace,stats=self.r.stats)
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
