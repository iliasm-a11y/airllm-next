"""Experimental native GLM5-Next text execution over bounded GGUF row reads.

Equations follow Transformers glm5_next (Apache-2.0) and llama.cpp glm5-next
(MIT). This is a correctness-first, batch-one, unpadded path. Q2_K/Q8_0 stay
unchanged on disk; row tiles are dequantized to FP32 for MLX CPU/Metal matmul.
No packed Q2_K Metal kernel, H2, vision or MTP support is claimed here.
"""
from collections import defaultdict
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
import math
import os
import sys
import numpy as np

from .config import NextConfig
from .gguf_qwen import GGUFWeights
from .generation import (token_batch, eos_ids, generation_options,
                         logits_suffix_start, check_logits_budget)
from .cache import size_of


def _qualified_topk(scores,count,role):
    """Select every required winner; refuse backend-dependent boundary ties."""
    values=np.asarray(scores,dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite GLM5 '+role+' scores')
    order=np.argsort(-values,kind='stable')
    if count<len(order) and values[order[count-1]]==values[order[count]]:
        raise NotImplementedError('Unqualified exact FP32 tie at GLM5 '+role+' top-k boundary')
    return order[:count]


def gguf_glm5_config(checkpoint):
    """Interpret the checkpoint's own architecture metadata, never a GLM alias."""
    m = checkpoint.metadata
    if m.get('general.architecture') != 'glm5-next':
        raise NotImplementedError('GLM5-Next requires general.architecture=glm5-next')
    p = 'glm5-next.'
    def integer(name):
        value = m.get(p + name)
        if type(value) is not int or value < 1:
            raise ValueError('Missing/invalid GLM5-Next dimension ' + name)
        return value
    layers = integer('block_count')
    kv = m.get(p + 'attention.head_count_kv')
    if not isinstance(kv, list) or len(kv) != layers or any(type(v) is not int or v not in (0, 1) for v in kv):
        raise ValueError('GLM5-Next needs explicit per-layer latent KV schedule')
    if m.get(p + 'rope.dimension_count') != 0:
        raise NotImplementedError('GLM5-Next DSA requires NoPE')
    if m.get(p + 'expert_group_count', 1) != 1 or m.get(p + 'expert_group_used_count', 1) != 1:
        raise NotImplementedError('Grouped GLM5-Next routing is not implemented')
    if m.get(p + 'expert_gating_func', 2) != 2:
        raise NotImplementedError('GLM5-Next requires sigmoid routing')
    kinds = ['indexed_attention' if v else 'linear_attention' for v in kv]
    full = m.get(p + 'attention.indexer.types', [True] * layers)
    if len(full) != layers or any(type(v) is not bool for v in full):
        raise ValueError('Invalid indexer sharing schedule')
    if any(not full[i] for i, k in enumerate(kinds) if k == 'indexed_attention'):
        raise NotImplementedError('Cross-layer shared indexers are not implemented')
    emb = checkpoint.tensors.get('token_embd.weight')
    if emb is None or len(emb.shape) != 2:
        raise ValueError('Missing GLM5-Next embeddings')
    c = dict(model_type='glm5_next_text', hidden_size=integer('embedding_length'),
             vocab_size=emb.shape[0], num_hidden_layers=layers,
             num_attention_heads=integer('attention.head_count'), layer_types=kinds,
             intermediate_size=integer('feed_forward_length'),
             q_lora_rank=integer('attention.q_lora_rank'), kv_lora_rank=integer('attention.kv_lora_rank'),
             qk_nope_head_dim=integer('attention.key_length_mla'),
             v_head_dim=integer('attention.value_length_mla'),
             linear_head_dim=integer('kda.head_dim'), linear_conv_kernel_dim=integer('ssm.conv_kernel'),
             linear_lower_bound=m.get(p+'kda.gate_lower_bound'),
             index_head_dim=integer('attention.indexer.key_length'),
             index_n_heads=integer('attention.indexer.head_count'),
             index_topk=integer('attention.indexer.top_k'), index_kpool=integer('attention.indexer.kpool'),
             index_kpool_always_select_tail=m.get(p+'attention.indexer.kpool_select_tail', True),
             hc_mult=integer('hyper_connection.count'),
             hc_sinkhorn_iters=integer('hyper_connection.sinkhorn_iterations'),
             hc_eps=m.get(p+'hyper_connection.epsilon', 1e-6),
             rms_norm_eps=m.get(p+'attention.layer_norm_rms_epsilon', 1e-5),
             layer_norm_eps=m.get(p+'attention.layer_norm_epsilon', 1e-6),
             first_k_dense_replace=m.get(p+'leading_dense_block_count', 3),
             n_routed_experts=integer('expert_count'), num_experts_per_tok=integer('expert_used_count'),
             n_shared_experts=integer('expert_shared_count'),
             moe_intermediate_size=integer('expert_feed_forward_length'),
             routed_scaling_factor=m.get(p+'expert_weights_scale', 2.5),
             norm_topk_prob=m.get(p+'expert_weights_norm', True),
             swiglu_limit=m.get(p+'swiglu_clamp_exp', 10.),
             max_position_embeddings=integer('context_length'),
             eos_token_id=m.get('tokenizer.ggml.eos_token_ids', m.get('tokenizer.ggml.eos_token_id')))
    if c['n_shared_experts'] != 1 or not 1 <= c['num_experts_per_tok'] <= c['n_routed_experts']:
        raise ValueError('Invalid GLM5-Next experts')
    dense=c['first_k_dense_replace']
    if type(dense) is not int or not 0<=dense<=layers:
        raise ValueError('Invalid GLM5-Next leading dense block count')
    if type(c['norm_topk_prob']) is not bool or type(c['index_kpool_always_select_tail']) is not bool:
        raise ValueError('Invalid GLM5-Next routing/pool boolean')
    bound=c['linear_lower_bound']
    if bound is not None and (isinstance(bound,bool) or not isinstance(bound,(int,float)) or not math.isfinite(bound) or bound>=0):
        raise ValueError('GLM5-Next KDA gate lower bound must be negative or absent')
    if c['index_topk'] % c['index_kpool']:
        raise ValueError('DSA top-k must be divisible by KPool size')
    clamp=c['swiglu_limit']
    if isinstance(clamp,list):
        if len(clamp)!=layers or len(set(clamp))!=1:
            raise NotImplementedError('Nonuniform GLM5 SwiGLU clamps are unsupported')
        c['swiglu_limit']=clamp[0]
    shared_clamp=m.get(p+'swiglu_clamp_shexp',c['swiglu_limit'])
    if isinstance(shared_clamp,list):
        if len(shared_clamp)!=layers or any(v!=c['swiglu_limit'] for v in shared_clamp):
            raise NotImplementedError('Different shared-expert clamp is unsupported')
    elif shared_clamp!=c['swiglu_limit']:
        raise NotImplementedError('Different shared-expert clamp is unsupported')
    for name in ('hc_eps', 'rms_norm_eps', 'layer_norm_eps', 'routed_scaling_factor', 'swiglu_limit'):
        if not isinstance(c[name], (int, float)) or isinstance(c[name], bool) or not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError('Invalid GLM5-Next parameter '+name)
    stops=set(eos_ids(c['eos_token_id'],c['vocab_size']))
    for name in ('eot_token_id','eom_token_id'):
        if 'tokenizer.ggml.'+name in m:
            stops.update(eos_ids(m['tokenizer.ggml.'+name],c['vocab_size']))
    c['eos_token_id']=sorted(stops)
    return c


class GLM5Weights(GGUFWeights):
    """Reuse AirLLM's tile scheduler/cache without Qwen norm transformations."""
    def __init__(self, checkpoint, mx, options, config):
        compat = dict(linear_num_key_heads=1, linear_num_value_heads=1,
                      linear_value_head_dim=1, gguf_v_head_reordered=False)
        super().__init__(checkpoint, mx, options, compat, kernel='reference')

    def resolve(self, key):
        if key not in self.cp.tensors:
            raise KeyError('Missing GLM5-Next tensor '+key)
        return key

    def get(self, key, expert=None):
        d = self.cp.tensors[self.resolve(key)]
        shape = tuple(d.shape[1:] if expert is not None else d.shape)
        nbytes = math.prod(shape) * 4
        if nbytes > self.cfg.max_tensor_bytes:
            raise MemoryError('GLM5 small-parameter read exceeds max_tensor_bytes')
        self._guard(3*nbytes)
        out = self._tile(key, 0, math.prod(shape[:-1]) if len(shape)>1 else 1, expert).reshape(shape)
        self.mx.eval(out)
        return out


class GLM5Session:
    """Bounded FP32 KDA/conv, latent KV and KPool states; failed calls poison it."""
    def __init__(self, owner):
        self.owner=owner; self.position=0; self.banks={}; self.peak_bytes=0; self.valid=True

    @property
    def bytes(self):
        return size_of(self.banks)

    def reserve(self, nbytes):
        predicted=self.bytes+nbytes
        if predicted>self.owner.options.kv_bytes:
            raise MemoryError('GLM5 context/recurrent state exceeds kv_bytes')
        self.peak_bytes=max(self.peak_bytes,predicted)

    def store(self, layer, value):
        self.reserve(size_of(value))
        self.banks[layer]=value


class NativeGGUFGLM5Model:
    """Real experimental decoder; numerical/full-checkpoint validation is separate."""
    def __init__(self, path, next_config=None, *, kernel='reference'):
        self.root=Path(path).expanduser().resolve(); self.lock=RLock(); self.closed=False
        self.checkpoint=None; self.reader=None; self.tokenizer=None
        if kernel!='reference':
            raise NotImplementedError('Packed Q2_K Metal kernels are not validated; use MLX tiled matmul')
        if next_config is None:
            next_config=NextConfig(cache_bytes=256<<20,io_bytes=32<<20,kv_bytes=512<<20,
                workspace_bytes=128<<20,reserve_bytes=4<<30,max_tensor_bytes=16<<20,
                row_chunk=128,prefill_chunk=1,max_context=512,policy='adaptive',prefetch=False,
                dtype='float32',load_tokenizer=False)
        self.options=next_config if isinstance(next_config,NextConfig) else NextConfig(**next_config)
        if self.options.dtype!='float32':
            raise NotImplementedError('GLM5 correctness path currently requires compute dtype=float32')
        if self.options.lossless_dir or self.options.quantized_dir or self.options.io_mode=='mmap':
            raise NotImplementedError('GLM5 uses original GGUF positional reads only')
        if self.options.prefetch or self.options.spec_prefetch or self.options.output_prefetch:
            raise NotImplementedError('GLM5 asynchronous prefetch is not qualified; set prefetch=False')
        if self.options.load_tokenizer:
            raise NotImplementedError('Use load_tokenizer=False and the separately budgeted GLM5NextGGUFTokenizer adapter')
        loaded_mx=sys.modules.get('mlx.core')
        if ('MLX_ENABLE_TF32' not in os.environ and loaded_mx is not None
                and loaded_mx.metal.is_available()):
            raise NotImplementedError('Set MLX_ENABLE_TF32=0 before importing MLX on Metal; an existing pipeline is unqualified')
        if os.environ.get('MLX_ENABLE_TF32','0')!='0':
            raise NotImplementedError('GLM5 requires MLX_ENABLE_TF32=0 before MLX initializes Metal matmul')
        os.environ.setdefault('MLX_ENABLE_TF32','0')
        begin=perf_counter()
        try:
            from .gguf_split import SplitGGUFReader
            self.checkpoint=SplitGGUFReader(self.root, memory_budget_bytes=self.options.io_bytes,
                max_read_bytes=self.options.max_tensor_bytes,include_tokenizer_metadata=False)
            self.config=gguf_glm5_config(self.checkpoint)
            self._validate_layout()
            if self.options.io_mode=='pread_nocache':
                from .physical_io import set_fd_nocache
                for fd in self.checkpoint.filenos():set_fd_nocache(fd)
            import mlx.core as mx
            self.mx=mx; self.dtype=mx.float32
            self.device=('gpu' if mx.metal.is_available() else 'cpu') if self.options.device=='auto' else self.options.device
            if self.device=='gpu' and not mx.metal.is_available():
                raise RuntimeError('Metal GPU is unavailable')
            self.stream=mx.new_stream(getattr(mx,self.device))
            if mx.metal.is_available():
                mx.set_memory_limit(self.options.mlx_memory_bytes)
                mx.set_cache_limit(min(32<<20,self.options.workspace_bytes//4))
            self.reader=GLM5Weights(self.checkpoint,mx,self.options,self.config)
            self.stats=defaultdict(float); self.stats['startup_seconds']=perf_counter()-begin
            self.last_routes={}
        except BaseException:
            self.close(); raise

    def _validate_layout(self):
        c=self.config; h=c['hidden_size']; heads=c['num_attention_heads']; d=c['linear_head_dim']; inner=heads*d
        shapes={'token_embd.weight':(c['vocab_size'],h),'output.weight':(c['vocab_size'],h),'output_norm.weight':(h,)}
        for i,kind in enumerate(c['layer_types']):
            p=f'blk.{i}.'; hc=c['hc_mult']; mix=(hc+2)*hc
            local={'attn_norm.weight':(h,),'ffn_norm.weight':(h,)}
            for sub in ('attn','ffn'):
                local.update({f'hc_{sub}_fn.weight':(mix,hc*h),f'hc_{sub}_base.weight':(mix,),f'hc_{sub}_scale.weight':(3,)})
            if kind=='linear_attention':
                for q in ('q','k','v'):
                    local[f'attn_{q}.weight']=(inner,h)
                local.update({'ssm_f_a.weight':(d,h),'ssm_f_b.weight':(inner,d),'ssm_beta.weight':(heads,h),
                    'ssm_g_a.weight':(d,h),'ssm_g_b.weight':(inner,d),'ssm_a':(heads,),
                    'ssm_dt.bias':(inner,),'ssm_norm.weight':(d,),'attn_output.weight':(h,inner)})
            else:
                q=c['q_lora_rank']; l=c['kv_lora_rank']; kd=c['qk_nope_head_dim']; v=c['v_head_dim']; ih=c['index_n_heads']; idim=c['index_head_dim']
                local.update({'attn_q_a.weight':(q,h),'attn_q_a_norm.weight':(q,),
                    'attn_q_b.weight':(heads*kd,q),'attn_kv_a_mqa.weight':(l,h),'attn_kv_a_norm.weight':(l,),
                    'attn_k_b.weight':(heads,l,kd),'attn_v_b.weight':(heads,v,l),'attn_output.weight':(h,heads*v),
                    'indexer.attn_q_b.weight':(ih*idim,q),'indexer.attn_k.weight':(idim,h),
                    'indexer.k_norm.weight':(idim,),'indexer.k_norm.bias':(idim,),
                    'indexer.proj.weight':(ih,h),'indexer_compressor_gate.weight':(idim,h),
                    'indexer_compressor_ape.weight':(c['index_kpool'],idim)})
            if i<c['first_k_dense_replace']:
                m=c['intermediate_size']
                local.update({'ffn_gate.weight':(m,h),'ffn_up.weight':(m,h),'ffn_down.weight':(h,m)})
            else:
                m=c['moe_intermediate_size']; e=c['n_routed_experts']
                local.update({'ffn_gate_inp.weight':(e,h),'exp_probs_b.bias':(e,),
                    'ffn_gate_exps.weight':(e,m,h),'ffn_up_exps.weight':(e,m,h),'ffn_down_exps.weight':(e,h,m),
                    'ffn_gate_shexp.weight':(m,h),'ffn_up_shexp.weight':(m,h),'ffn_down_shexp.weight':(h,m)})
            shapes.update({p+n:s for n,s in local.items()})
        for name,shape in shapes.items():
            actual=self.checkpoint.tensors.get(name)
            if actual is None or tuple(actual.shape)!=shape:
                raise ValueError(f'Missing/invalid GLM5 tensor {name}; expected {shape}, got {None if actual is None else actual.shape}')
        for i,kind in enumerate(c['layer_types']):
            if kind=='linear_attention':
                for q in ('q','k','v'):
                    name=f'blk.{i}.ssm_conv1d_{q}.weight'; desc=self.checkpoint.tensors.get(name)
                    allowed=((inner,1,c['linear_conv_kernel_dim']),
                             (1,inner,1,c['linear_conv_kernel_dim']))
                    if desc is None or tuple(desc.shape) not in allowed:
                        raise ValueError('Missing/invalid GLM5 convolution '+name)

    def _norm(self,x,weight=None,eps=None):
        mx=self.mx; y=x.astype(mx.float32)
        y=y*mx.rsqrt(mx.mean(y*y,axis=-1,keepdims=True)+(eps or self.config['rms_norm_eps']))
        return y if weight is None else y*self.reader.get(weight)

    def _hc(self,streams,p,kind):
        mx=self.mx;c=self.config;n=c['hc_mult'];eps=c['hc_eps']
        flattened=self._norm(streams.reshape(1,-1)); coeff=self.reader.linear(flattened,p+f'hc_{kind}_fn.weight')
        base=self.reader.get(p+f'hc_{kind}_base.weight');scale=self.reader.get(p+f'hc_{kind}_scale.weight')
        pre=mx.sigmoid(coeff[:,:n]*scale[0]+base[:n])+eps
        post=2*mx.sigmoid(coeff[:,n:2*n]*scale[1]+base[n:2*n])
        comb=mx.softmax((coeff[:,2*n:]*scale[2]+base[2*n:]).reshape(1,n,n),axis=-1)+eps
        comb=comb/(mx.sum(comb,axis=-2,keepdims=True)+eps)
        for _ in range(c['hc_sinkhorn_iters']-1):
            comb=comb/(mx.sum(comb,axis=-1,keepdims=True)+eps)
            comb=comb/(mx.sum(comb,axis=-2,keepdims=True)+eps)
        collapsed=mx.sum(pre[...,None]*streams,axis=1)
        return collapsed,post,comb

    def _kda(self,x,i,cache):
        mx=self.mx;c=self.config;r=self.reader;p=f'blk.{i}.';heads=c['num_attention_heads'];d=c['linear_head_dim'];k=c['linear_conv_kernel_dim'];inner=heads*d
        old=cache.banks.get(i,{})
        state_bytes=heads*d*d*4
        # Convolution slices may retain their K-wide backing buffers. Charge
        # the full windows rather than only the visible K-1 state elements.
        cache.reserve(state_bytes+3*inner*k*4)
        r._guard(6*state_bytes+3*inner*k*4)
        conv=[];qkv=[]
        for qi,q in enumerate(('q','k','v')):
            proj=r.linear(x,p+f'attn_{q}.weight').reshape(inner)
            previous=old.get('conv', [None]*3)[qi]
            if previous is None:previous=mx.zeros((inner,k-1),dtype=mx.float32)
            else:previous=previous[:,1:]
            joined=mx.concatenate([previous,proj[:,None]],axis=-1)
            weight=r.get(p+f'ssm_conv1d_{q}.weight').reshape(inner,k)
            out=mx.sum(joined*weight,axis=-1);out=out*mx.sigmoid(out)
            qkv.append(out.reshape(heads,d));conv.append(joined)
        q,key,value=qkv
        q=q/mx.sqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-6)/math.sqrt(d)
        key=key/mx.sqrt(mx.sum(key*key,axis=-1,keepdims=True)+1e-6)
        f=r.linear(r.linear(x,p+'ssm_f_a.weight'),p+'ssm_f_b.weight').reshape(heads,d)+r.get(p+'ssm_dt.bias').reshape(heads,d)
        # GGUF stores -exp(A_log), rather than A_log (llama.cpp line830).
        stored_a=r.get(p+'ssm_a').reshape(heads,1)
        if bool(mx.any(stored_a>=0).item()):raise ValueError('GLM5 ssm_a must be negative')
        rate=-stored_a
        bound=c['linear_lower_bound']
        g=bound*mx.sigmoid(rate*f) if bound is not None else -rate*mx.logaddexp(f,mx.array(0.,dtype=mx.float32))
        beta=mx.sigmoid(r.linear(x,p+'ssm_beta.weight')).reshape(heads,1)
        state=old.get('state')
        if state is None:state=mx.zeros((heads,d,d),dtype=mx.float32)
        state=state*mx.exp(g)[...,None]
        delta=(value-mx.sum(state*key[...,None],axis=-2))*beta
        state=state+key[...,None]*delta[:,None,:]
        out=mx.sum(state*q[...,None],axis=-2)
        gate=r.linear(r.linear(x,p+'ssm_g_a.weight'),p+'ssm_g_b.weight').reshape(heads,d)
        out=self._norm(out,p+'ssm_norm.weight')*mx.sigmoid(gate)
        mx.eval(state,*conv,out);cache.store(i,{'conv':conv,'state':state})
        return r.linear(out.reshape(1,inner),p+'attn_output.weight')

    def _dsa(self,x,i,cache):
        mx=self.mx;c=self.config;r=self.reader;p=f'blk.{i}.';heads=c['num_attention_heads'];kd=c['qk_nope_head_dim'];ih=c['index_n_heads'];idim=c['index_head_dim'];pool=c['index_kpool']
        old=cache.banks.get(i,{})
        new_length=cache.position+1
        cache.reserve(new_length*(c['kv_lora_rank']+2*idim)*4)
        r._guard(new_length*(4*c['kv_lora_rank']+8*idim+4*ih)*4)
        qres=self._norm(r.linear(x,p+'attn_q_a.weight'),p+'attn_q_a_norm.weight')
        query=r.linear(qres,p+'attn_q_b.weight').reshape(heads,kd)
        latent=self._norm(r.linear(x,p+'attn_kv_a_mqa.weight'),p+'attn_kv_a_norm.weight')
        latent=mx.concatenate([old['latent'],latent],axis=0) if 'latent' in old else latent
        iq=r.linear(qres,p+'indexer.attn_q_b.weight').reshape(ih,idim)
        key=r.linear(x,p+'indexer.attn_k.weight');mean=mx.mean(key,axis=-1,keepdims=True)
        key=(key-mean)*mx.rsqrt(mx.mean((key-mean)**2,axis=-1,keepdims=True)+c['layer_norm_eps'])
        key=key*r.get(p+'indexer.k_norm.weight')+r.get(p+'indexer.k_norm.bias')
        gate=r.linear(x,p+'indexer_compressor_gate.weight')
        keys=mx.concatenate([old['keys'],key],axis=0) if 'keys' in old else key
        gates=mx.concatenate([old['gates'],gate],axis=0) if 'gates' in old else gate
        total=latent.shape[0];complete=total//pool;selected=[]
        if complete:
            pk=keys[:complete*pool].reshape(complete,pool,idim)
            pg=gates[:complete*pool].reshape(complete,pool,idim)+r.get(p+'indexer_compressor_ape.weight')[None,:,:]
            pooled=mx.sum(mx.softmax(pg,axis=1)*pk,axis=1)
            scores=mx.maximum((iq@pooled.T)/math.sqrt(idim),0.)
            weights=r.linear(x,p+'indexer.proj.weight').reshape(ih)/math.sqrt(ih)
            scores=mx.sum(scores*weights[:,None],axis=0)
            chosen=_qualified_topk(scores,min(complete,c['index_topk']//pool),'KPool')
            selected=[int(j)*pool+t for j in chosen for t in range(pool)]
        if c['index_kpool_always_select_tail']:
            selected.extend(range(complete*pool,total))
        if not selected:raise ValueError('DSA selected no causal tokens; enable KPool tail for short contexts')
        # Duplicates are not present (complete pools and tail are disjoint).
        bank=mx.take(latent,mx.array(sorted(selected)),axis=0);outputs=[]
        for head in range(heads):
            # Absorbed MLA: keep compressed latents, never allocate H*T*(K+V).
            qlatent=r.linear(query[head:head+1],p+'attn_k_b.weight',expert=head)
            scores=(qlatent@bank.T)/math.sqrt(kd)
            mixed=mx.softmax(scores,axis=-1)@bank
            outputs.append(r.linear(mixed,p+'attn_v_b.weight',expert=head))
        out=mx.concatenate(outputs,axis=-1);mx.eval(out,latent,keys,gates)
        cache.store(i,{'latent':latent,'keys':keys,'gates':gates})
        return r.linear(out,p+'attn_output.weight')

    def _mlp(self,x,i):
        mx=self.mx;c=self.config;r=self.reader;p=f'blk.{i}.';limit=c['swiglu_limit']
        def expert(gate,up,down,e=None):
            g=mx.minimum(r.linear(x,p+gate,expert=e),limit)
            u=mx.clip(r.linear(x,p+up,expert=e),-limit,limit)
            return r.linear((g*mx.sigmoid(g))*u,p+down,expert=e)
        if i<c['first_k_dense_replace']:
            return expert('ffn_gate.weight','ffn_up.weight','ffn_down.weight')
        scores=mx.sigmoid(r.linear(x,p+'ffn_gate_inp.weight').astype(mx.float32)).reshape(-1)
        choice=scores+r.get(p+'exp_probs_b.bias')
        ids=_qualified_topk(choice,c['num_experts_per_tok'],'MoE')
        self.last_routes[i]=ids.tolist()
        weights=mx.take(scores,mx.array(ids))
        if c['norm_topk_prob']:weights=weights/(mx.sum(weights)+1e-20)
        weights=weights*c['routed_scaling_factor']
        out=mx.zeros_like(x)
        for index,e in enumerate(ids.tolist()):
            out=out+weights[index]*expert('ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight',int(e));mx.eval(out)
        self.stats['routed_expert_evaluations']+=len(ids)
        return out+expert('ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight')

    def new_session(self):
        if self.closed:raise RuntimeError('Model is closed')
        return GLM5Session(self)

    def forward(self,input_ids,past_key_values=None,*,logits_to_keep=0,**kwargs):
        if kwargs:raise TypeError('Unsupported GLM5 forward options: '+str(sorted(kwargs)))
        ids=token_batch(input_ids,self.config['vocab_size']);start=logits_suffix_start(ids.shape[1],logits_to_keep)
        check_logits_budget(ids.shape[1]-start,self.config['vocab_size'],self.options.workspace_bytes)
        with self.lock:
            if self.closed:raise RuntimeError('Model is closed')
            cache=self.new_session() if past_key_values is None else past_key_values
            if not isinstance(cache,GLM5Session) or cache.owner is not self or not cache.valid:
                raise ValueError('Invalid GLM5 cache; failed sessions must be rebuilt')
            if cache.position+ids.shape[1]>min(self.options.max_context,self.config['max_position_embeddings']):
                raise MemoryError('GLM5 context limit exceeded')
            c=self.config; target=cache.position+ids.shape[1]
            recurrent=c['num_attention_heads']*(c['linear_head_dim']**2+
                3*c['linear_head_dim']*c['linear_conv_kernel_dim'])*4
            indexed=target*(c['kv_lora_rank']+2*c['index_head_dim'])*4
            banks=[recurrent if kind=='linear_attention' else indexed for kind in c['layer_types']]
            # The replacement bank can coexist with the complete old session.
            # Reject the whole request before any weights or cache are touched.
            if sum(banks)+max(banks)>self.options.kv_bytes:
                raise MemoryError('GLM5 context/recurrent state exceeds kv_bytes')
            mx=self.mx;outputs=[]
            try:
                with mx.stream(self.stream):
                    for index in range(ids.shape[1]):
                        x=self.reader.embedding(ids[:,index:index+1],'token_embd.weight').reshape(1,-1)
                        streams=mx.repeat(x[:,None,:],self.config['hc_mult'],axis=1)
                        for i,kind in enumerate(self.config['layer_types']):
                            p=f'blk.{i}.'
                            collapsed,post,comb=self._hc(streams,p,'attn')
                            normed=self._norm(collapsed,p+'attn_norm.weight')
                            mixed=self._kda(normed,i,cache) if kind=='linear_attention' else self._dsa(normed,i,cache)
                            streams=post[...,None]*mixed[:,None,:]+mx.swapaxes(comb,-1,-2)@streams;mx.eval(streams)
                            collapsed,post,comb=self._hc(streams,p,'ffn')
                            mixed=self._mlp(self._norm(collapsed,p+'ffn_norm.weight'),i)
                            streams=post[...,None]*mixed[:,None,:]+mx.swapaxes(comb,-1,-2)@streams;mx.eval(streams)
                        if index>=start:
                            hidden=self._norm(mx.mean(streams,axis=1),'output_norm.weight')
                            logits=self.reader.output(hidden,'output.weight');mx.eval(logits)
                            if not bool(mx.all(mx.isfinite(logits)).item()):raise FloatingPointError('Non-finite GLM5 logits')
                            outputs.append(logits[:,None,:])
                        cache.position+=1
                    result=mx.concatenate(outputs,axis=1);mx.eval(result)
            except BaseException:
                cache.valid=False;raise
            return SimpleNamespace(logits=result,past_key_values=cache)

    def generate(self,input_ids,*,max_new_tokens=128,temperature=0.,eos_token_id=None,on_token=None,**kwargs):
        if kwargs:raise TypeError('Unsupported GLM5 generation options: '+str(sorted(kwargs)))
        count,temp=generation_options(max_new_tokens,temperature)
        if temp!=0:raise NotImplementedError('GLM5 supports greedy temperature=0 only')
        if on_token is not None and not callable(on_token):raise TypeError('on_token must be callable')
        ids=token_batch(input_ids,self.config['vocab_size']); stop=eos_ids(self.config.get('eos_token_id') if eos_token_id is None else eos_token_id,self.config['vocab_size'])
        if ids.shape[1]+count>min(self.options.max_context,self.config['max_position_embeddings']):raise MemoryError('GLM5 prompt/output exceeds context limit')
        result=ids.tolist()[0];current=ids;cache=None;begin=perf_counter();first=None
        for index in range(count):
            output=self.forward(current,past_key_values=cache,logits_to_keep=1);cache=output.past_key_values
            token=int(self.mx.argmax(output.logits[0,-1]).item());now=perf_counter()
            if first is None:first=now
            result.append(token)
            if on_token is not None:on_token({'token_id':token,'token_index':index,'elapsed_seconds':now-begin})
            if token in stop:break
            current=np.array([[token]],dtype=np.int32)
        if first is not None:self.stats['ttft_seconds']=first-begin
        self.stats['tokens_generated']=len(result)-ids.shape[1]
        self.stats['generation_seconds']=perf_counter()-begin
        if first is not None and len(result)-ids.shape[1]>1:
            self.stats['decode_tokens_per_second']=(len(result)-ids.shape[1]-1)/(perf_counter()-first)
        return np.array([result],dtype=np.int32)

    def metrics(self):
        return {'backend':'next-gguf-glm5-experimental','compute_device':self.device,
            'full_glm53_checkpoint_validated':False,'metal_validated':False,'vision_execution':False,'mtp_execution':False,
            'kernel':'dequantized-fp32-mlx','runtime':dict(self.stats),'io':dict(self.checkpoint.stats),
            'weights':dict(self.reader.stats),'cache':{'bytes':self.reader.cache.bytes,'peak_bytes':self.reader.cache.peak_bytes,
            'capacity':self.reader.cache.capacity},'prefetch':{'enabled':False},
            'last_routed_expert_ids':dict(self.last_routes),
            'numerics':{'compute_dtype':'float32','tf32_requested':False,
                'routing_tie_break':'boundary-ties-rejected; interior-order-stable-low-index',
                'expert_accumulation_order':'descending-corrected-score',
                'mla':'absorbed-latent','bitwise_reference_parity_claimed':False}}

    def close(self):
        with self.lock:
            if self.reader is not None:self.reader.close()
            if self.checkpoint is not None:self.checkpoint.close()
            self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


GGUFGLM5Model=NativeGGUFGLM5Model
