"""Native, bounded GGUF Qwen3.5/3.6 text inference with MLX.

GGUF matrices remain on disk. Only requested row tiles and routed experts are
decoded; there is no llama.cpp/Ollama execution or whole-checkpoint conversion.
The numerical blocks/session are shared with :mod:`qwen_runtime`. Layout
adaptation follows ggml-org/llama.cpp conversion/qwen.py (Apache-2.0): real RMS
weights, negative exponential A, and tiled linear-attention V heads. Text-only
M-RoPE has equal temporal/height/width positions and reduces to ordinary RoPE.
Vision, MTP, sampling and speculative decoding are deliberately unsupported.
"""
from collections import defaultdict
import math
from pathlib import Path
import re
from threading import RLock
from time import perf_counter

import numpy as np
import psutil

from .cache import WeightCache
from .config import NextConfig
from .qwen_moe import qwen_route
from .qwen_runtime import (ExperimentalQwenModel, QwenDeltaNet, QwenFullAttention,
                           normalize_qwen_config)


def v_head_permutation(key_heads, value_heads, head_dim=1):
    """Indices mapping HF grouped V heads to GGUF tiled broadcast order."""
    if any(type(v) is not int or v < 1 for v in (key_heads,value_heads,head_dim)):
        raise ValueError('Head dimensions must be positive integers')
    if value_heads % key_heads:
        raise ValueError('V heads must be divisible by K heads')
    return np.arange(value_heads*head_dim).reshape(key_heads,value_heads//key_heads,head_dim).transpose(1,0,2).reshape(-1)


def gguf_qwen_config(checkpoint):
    """Validate GGUF architecture/dimensions without reading any weight bytes."""
    m=checkpoint.metadata
    if m.get('general.architecture') != 'qwen35moe':
        raise NotImplementedError('Native GGUF Qwen requires qwen35moe')
    p='qwen35moe.'
    def required(name):
        value=m.get(p+name)
        if type(value) is not int or value<1:
            raise ValueError('Missing/invalid GGUF dimension '+p+name)
        return value
    layers=required('block_count');heads=required('attention.head_count')
    kv=m.get(p+'attention.head_count_kv')
    interval=m.get(p+'full_attention_interval',4)
    if isinstance(kv,list):
        if len(kv)!=layers or any(type(v) is not int or v<0 for v in kv):
            raise ValueError('Invalid per-layer GGUF KV head counts')
        counts={v for v in kv if v}
        if len(counts)!=1:raise NotImplementedError('Nonuniform full-attention KV head counts')
        kv_heads=next(iter(counts));kinds=['full_attention' if v else 'linear_attention' for v in kv]
    else:
        if type(kv) is not int or kv<1 or type(interval) is not int or interval<1:
            raise ValueError('Invalid GGUF attention pattern')
        kv_heads=kv;kinds=['full_attention' if (i+1)%interval==0 else 'linear_attention' for i in range(layers)]
    embedding=checkpoint.tensors.get('token_embd.weight')
    if embedding is None or len(embedding.shape)!=2:raise ValueError('Missing GGUF token embeddings')
    h=required('embedding_length');d=required('attention.key_length')
    if m.get(p+'attention.value_length',d)!=d:
        raise NotImplementedError('Different Q/K and V attention head dimensions')
    vd=required('ssm.inner_size');vh=required('ssm.time_step_rank')
    if vd%vh:raise ValueError('Invalid GGUF linear-attention value dimension')
    rotary=required('rope.dimension_count')
    sections=m.get(p+'rope.dimension_sections',m.get(p+'rope.mrope_section'))
    if sections is not None and (not isinstance(sections,list) or any(type(v) is not int or v<0 for v in sections) or sum(sections)*2!=rotary):
        raise ValueError('Invalid GGUF M-RoPE sections')
    scaling=m.get(p+'rope.scaling.type','none')
    if scaling not in ('none','default'):raise NotImplementedError('Scaled GGUF RoPE is unsupported')
    c=dict(model_type='qwen3_5_moe_text',hidden_size=h,vocab_size=embedding.shape[0],
        num_hidden_layers=layers,num_attention_heads=heads,num_key_value_heads=kv_heads,head_dim=d,
        linear_conv_kernel_dim=required('ssm.conv_kernel'),linear_key_head_dim=required('ssm.state_size'),
        linear_value_head_dim=vd//vh,linear_num_key_heads=required('ssm.group_count'),linear_num_value_heads=vh,
        num_experts=required('expert_count'),num_experts_per_tok=required('expert_used_count'),
        moe_intermediate_size=required('expert_feed_forward_length'),
        shared_expert_intermediate_size=required('expert_shared_feed_forward_length'),
        max_position_embeddings=required('context_length'),layer_types=kinds,
        rope_parameters={'rope_type':'default','rope_theta':m.get(p+'rope.freq_base',10000000.),'partial_rotary_factor':rotary/d},
        rms_norm_eps=m.get(p+'attention.layer_norm_rms_epsilon',1e-6),
        tie_word_embeddings='output.weight' not in checkpoint.tensors,
        eos_token_id=m.get('tokenizer.ggml.eos_token_ids',m.get('tokenizer.ggml.eos_token_id')),
        gguf_v_head_reordered=m.get(p+'ssm.v_head_reordered',False))
    if type(c['gguf_v_head_reordered']) is not bool:raise ValueError('Invalid GGUF V-head reorder flag')
    # Qwen35's official generation contract stops on both im_end and
    # endoftext. In this GGUF, endoftext is stored as padding_token_id.
    eos=c['eos_token_id'];eos=[] if eos is None else list(eos) if isinstance(eos,list) else [eos]
    padding=m.get('tokenizer.ggml.padding_token_id')
    if (m.get('tokenizer.ggml.pre')=='qwen35' and padding==248044 and 248046 in eos):
        eos.append(padding)
    c['eos_token_id']=sorted(set(eos)) or None
    return normalize_qwen_config(c)


_NAMES={
    'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'post_attention_norm.weight',
    'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight',
    'self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
    'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
    'linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight',
    'linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight',
    'linear_attn.out_proj.weight':'ssm_out.weight','linear_attn.conv1d.weight':'ssm_conv1d.weight',
    'linear_attn.norm.weight':'ssm_norm.weight','linear_attn.dt_bias':'ssm_dt','linear_attn.A_log':'ssm_a',
    'mlp.gate.weight':'ffn_gate_inp.weight','mlp.shared_expert_gate.weight':'ffn_gate_inp_shexp.weight',
    'mlp.shared_expert.gate_proj.weight':'ffn_gate_shexp.weight',
    'mlp.shared_expert.up_proj.weight':'ffn_up_shexp.weight',
    'mlp.shared_expert.down_proj.weight':'ffn_down_shexp.weight',
}


class GGUFWeights:
    """Budgeted row-tiled matmul, including output head and selected experts.

    Decoded row tiles share the existing AirLLM cache policies and cache_bytes.
    Every tile is evaluated before advancing the disk cursor, bounding MLX's
    lazy graph lifetime. The GGUF reader separately reserves decoder/IO memory.
    """
    def __init__(self,checkpoint,mx,options,config,*,kernel='reference',packed_copy_free=False):
        self.cp=checkpoint;self.mx=mx;self.cfg=options;self.model_config=config
        if kernel not in ('reference','metal'):raise ValueError('GGUF kernel must be reference or metal')
        if kernel=='metal' and not mx.metal.is_available():raise RuntimeError('Metal kernel requires a Metal device')
        self.kernel=kernel;self.dtype=getattr(mx,options.dtype);self.stats=defaultdict(float)
        self.spec=None;self.prefetch_enabled=False
        if options.compact_cache_bytes:raise NotImplementedError('GGUF compact-cache transport is not enabled')
        if options.policy=='batch-aware':raise NotImplementedError('GGUF batch-aware tile admission is not enabled')
        if options.policy=='adaptive':
            from .adaptive_cache import AdaptiveWeightCache
            cls=AdaptiveWeightCache
        elif options.policy in ('shared','expert','hybrid','batch-aware'):
            from .residency_cache import ResidencyWeightCache
            cls=ResidencyWeightCache
        else:cls=WeightCache
        self.cache=cls(options.cache_bytes,options.policy)
        if type(packed_copy_free) is not bool:
            raise ValueError('packed_copy_free must be boolean')
        self.packed_copy_free=packed_copy_free
        self.packed_budget=None
        if packed_copy_free:
            from .gguf import PackedBufferBudget
            transient=min(options.max_tensor_bytes,options.workspace_bytes//2)
            self.packed_budget=PackedBufferBudget(options.cache_bytes+transient)
        kh=config['linear_num_key_heads'];vh=config['linear_num_value_heads'];hd=config['linear_value_head_dim']
        self.perm=v_head_permutation(kh,vh,hd);self.inverse=np.argsort(self.perm)
        self.head_perm=v_head_permutation(kh,vh);self.head_inverse=np.argsort(self.head_perm)
        self.reordered=config['gguf_v_head_reordered']

    def resolve(self,key):
        direct={'model.embed_tokens.weight':'token_embd.weight','model.norm.weight':'output_norm.weight','lm_head.weight':'output.weight'}
        if key in direct:return direct[key]
        if key in self.cp.tensors:return key
        match=re.fullmatch(r'model\.layers\.(\d+)\.(.*)',key)
        if match and match[2] in _NAMES:return f'blk.{match[1]}.'+_NAMES[match[2]]
        raise KeyError('Unsupported GGUF Qwen weight '+key)

    def _guard(self,nbytes):
        if nbytes>self.cfg.workspace_bytes:raise MemoryError('GGUF compute exceeds workspace_bytes')
        if psutil.virtual_memory().available < self.cfg.reserve_bytes+nbytes:
            self.trim()
            if psutil.virtual_memory().available < self.cfg.reserve_bytes+nbytes:
                raise MemoryError('System reserve reached before GGUF allocation')
        self.stats['peak_workspace_estimate_bytes']=max(self.stats['peak_workspace_estimate_bytes'],nbytes)

    @staticmethod
    def _identity(name,start,count,expert=None,*,packed=False):
        tile=f'{"packed:" if packed else ""}{name}@{start}:{count}'
        if expert is not None:return (tile,expert)
        return ('output',tile) if name=='output.weight' else tile

    def _tile(self,name,start,count,expert=None,*,output_role=False):
        desc=self.cp.tensors[name];ident=self._identity(name,start,count,expert)
        cacheable=not (output_role or name=='output.weight') or self.cfg.output_tile_cache
        hint=count*desc.shape[-1]*4
        result=self.cache.get(ident,hint) if cacheable else None
        if result is not None:return result
        load_begin=perf_counter();begin=load_begin;host=self.cp.read_rows(name,start,count,expert=expert)
        self.stats['read_dequant_seconds']+=perf_counter()-begin
        begin=perf_counter();result=self.mx.array(host,dtype=self.mx.float32)
        # FP32 source router/norms retain precision; quantized products use the
        # requested compute dtype, matching the established floating runtime.
        if desc.quant_type!=0:result=result.astype(self.dtype)
        self.mx.eval(result)
        self.stats['materialize_seconds']+=perf_counter()-begin;self.stats['loads']+=1
        if cacheable:
            self.cache.record_load(ident,perf_counter()-load_begin)
            self.cache.put(ident,result)
        return result

    def _packed_tile(self,name,start,count,expert=None,*,output_role=False):
        ident=self._identity(name,start,count,expert,packed=True)
        desc=self.cp.tensors[name];cacheable=not (output_role or name=='output.weight') or self.cfg.output_tile_cache
        result=self.cache.get(ident,count*desc.row_bytes) if cacheable else None
        if result is not None:return result
        begin=perf_counter()
        if self.packed_copy_free:
            result=self.cp.read_packed_rows(name,start,count,expert=expert,
                copy=False,owner_budget=self.packed_budget)
            self.stats['packed_copy_free_loads']+=1
            self.stats['packed_host_copy_bytes_avoided']+=2*result.nbytes
        else:
            raw=self.cp.read_packed_rows(name,start,count,expert=expert)
            result=raw.copy();del raw
        elapsed=perf_counter()-begin;self.stats['packed_read_seconds']+=elapsed;self.stats['loads']+=1
        if cacheable:
            self.cache.record_load(ident,elapsed);self.cache.put(ident,result)
        return result

    def get(self,key,expert=None):
        name=self.resolve(key);desc=self.cp.tensors[name]
        shape=tuple(desc.shape)
        if expert is not None:shape=shape[1:]
        n=math.prod(shape)*4
        if n>self.cfg.max_tensor_bytes:raise MemoryError('GGUF get exceeds max_tensor_bytes; use tiled linear')
        # get is reserved for small parameters/router matrices, never an expert bank.
        self._guard(3*n)
        if len(shape)==1:
            value=self._tile(name,0,1,expert).reshape(shape)
        else:value=self._tile(name,0,math.prod(shape[:-1]),expert).reshape(shape)
        mx=self.mx
        if key.endswith('A_log'):
            if bool(mx.any(value>=0).item()):raise ValueError('GGUF ssm_a must be strictly negative')
            value=mx.log(-value.astype(mx.float32))
        if self.reordered and key.endswith(('linear_attn.A_log','linear_attn.dt_bias')):
            value=mx.take(value,mx.array(self.head_inverse),axis=0)
        if key.endswith('linear_attn.conv1d.weight'):
            if self.reordered:
                kd=2*self.model_config['linear_num_key_heads']*self.model_config['linear_key_head_dim']
                value=mx.concatenate([value[:kd],mx.take(value[kd:],mx.array(self.inverse),axis=0)],axis=0)
            value=value[:,None,:]
        elif key=='model.norm.weight' or key.endswith(('input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight')):
            value=value.astype(mx.float32)-1
        mx.eval(value)
        return value

    def _rows_per_tile(self,name,output_bytes,*,input_bytes=0,batch=1):
        d=self.cp.tensors[name];width=d.shape[-1]
        packed=self.kernel=='metal' and d.quant_type in (12,14)
        max_rows=self.cp.available_memory_bytes//max(1,2*d.row_bytes) if packed else self.cp.max_rows(name)
        max_rows=min(max_rows,self.cfg.max_tensor_bytes//max(1,d.row_bytes if packed else width*4))
        workspace_rows=(self.cfg.workspace_bytes-output_bytes-input_bytes)//max(1,(2*d.row_bytes if packed else width*12)+batch*4)
        rows=min(self.cfg.row_chunk,max_rows,workspace_rows)
        if rows<1:raise MemoryError('GGUF row tile cannot fit IO/workspace budget')
        return rows

    def linear(self,x,key,expert=None,*,output_role=False):
        name=self.resolve(key);desc=self.cp.tensors[name]
        shape=tuple(desc.shape)
        if expert is not None:
            if type(expert) is not int or len(shape)!=3 or not 0<=expert<shape[0]:raise ValueError('Invalid GGUF expert id')
            shape=shape[1:]
        if len(shape)!=2 or x.shape[-1]!=shape[1]:raise ValueError('Invalid GGUF linear matrix shape')
        mx=self.mx
        if self.reordered and key.endswith('linear_attn.out_proj.weight'):
            x=mx.take(x,mx.array(self.perm),axis=-1)
        outshape=(*x.shape[:-1],shape[0]);outbytes=math.prod(outshape)*4
        self._guard(outbytes)
        rows=self._rows_per_tile(name,outbytes,input_bytes=x.size*4,batch=x.size//x.shape[-1]);output=mx.zeros(outshape,dtype=x.dtype)
        for start in range(0,shape[0],rows):
            count=min(rows,shape[0]-start)
            if self.kernel=='metal' and desc.quant_type in (12,14):
                required=outbytes+count*desc.row_bytes*2+x.size*4+count*(x.size//x.shape[-1])*4
                self._guard(required);weight=self._packed_tile(name,start,count,expert,output_role=output_role)
                from .gguf_metal import packed_matmul
                begin=perf_counter();tile=packed_matmul(weight,x,int(desc.quant_type),max_workspace_bytes=self.cfg.workspace_bytes-outbytes)
                self.stats['packed_matmul_calls']+=1
            else:
                self._guard(outbytes+x.size*4+count*shape[1]*12+count*(x.size//x.shape[-1])*4);weight=self._tile(name,start,count,expert,output_role=output_role)
                begin=perf_counter();tile=x.astype(weight.dtype)@weight.T
            output[...,start:start+count]=tile.astype(output.dtype);mx.eval(output)
            self.stats['matmul_seconds']+=perf_counter()-begin
            del tile,weight
        if self.reordered:
            if key.endswith('linear_attn.in_proj_qkv.weight'):
                kd=2*self.model_config['linear_num_key_heads']*self.model_config['linear_key_head_dim']
                output=mx.concatenate([output[...,:kd],mx.take(output[...,kd:],mx.array(self.inverse),axis=-1)],axis=-1)
            elif key.endswith('linear_attn.in_proj_z.weight'):
                output=mx.take(output,mx.array(self.inverse),axis=-1)
            elif key.endswith(('linear_attn.in_proj_a.weight','linear_attn.in_proj_b.weight')):
                output=mx.take(output,mx.array(self.head_inverse),axis=-1)
        mx.eval(output)
        return output

    def output(self,x,key):return self.linear(x,key,output_role=True)

    def embedding(self,ids,key):
        name=self.resolve(key);shape=self.cp.tensors[name].shape
        flat=np.asarray(ids).reshape(-1);self._guard(flat.size*shape[1]*12)
        rows=[self._tile(name,int(token),1) for token in flat]
        result=self.mx.concatenate(rows,axis=0).reshape(*ids.shape,shape[1]);self.mx.eval(result)
        return result

    def trim(self):self.cache.clear()
    def drain(self):pass
    def close(self):self.trim()


class GGUFQwenSparseMLP:
    """Exact softmax/top-k routing; row-tile only the selected expert slices."""
    def __init__(self,mx,reader,config,layer):
        self.mx=mx;self.r=reader;self.c=config;self.layer=layer;self.p=f'blk.{layer}.';self.routes={}

    def __call__(self,x):
        mx=self.mx;p=self.p;flat=x.reshape(-1,x.shape[-1])
        # Explicit local CPU-reference convention only for exact probability
        # boundary ties. The generic Qwen route keeps its fail-closed default.
        ids,scores=qwen_route(mx,x,self.r.get(p+'ffn_gate_inp.weight'),self.c,
            tie_policy='torch_cpu_reference',workspace_bytes=self.r.cfg.workspace_bytes,
            reserve_bytes=self.r.cfg.reserve_bytes,stats=self.r.stats)
        host=np.asarray(ids);selected=np.unique(host);output=mx.zeros_like(flat)
        self.routes={int(e):int(np.count_nonzero(host==e)) for e in selected}
        self.r.stats['routed_expert_uses']+=len(selected)
        for expert in selected:
            expert=int(expert);ti,ki=np.where(host==expert);indices=mx.array(ti);ranks=mx.array(ki)
            current=flat[indices]
            gate=self.r.linear(current,p+'ffn_gate_exps.weight',expert)
            up=self.r.linear(current,p+'ffn_up_exps.weight',expert)
            down=self.r.linear((gate*mx.sigmoid(gate))*up,p+'ffn_down_exps.weight',expert)
            output[indices]=output[indices]+(down*scores[indices,ranks,None]).astype(output.dtype);mx.eval(output)
        gate=self.r.linear(flat,p+'ffn_gate_shexp.weight');up=self.r.linear(flat,p+'ffn_up_shexp.weight')
        shared=self.r.linear((gate*mx.sigmoid(gate))*up,p+'ffn_down_shexp.weight')
        shared_gate=mx.sigmoid(self.r.linear(flat,p+'ffn_gate_inp_shexp.weight'))
        result=(output+shared*shared_gate).reshape(x.shape);mx.eval(result)
        return result


class NativeGGUFQwenModel(ExperimentalQwenModel):
    """Explicit experimental GGUF backend; greedy batch-one text generation.

    Initialization reads metadata/index only. Defaults reserve 4 GiB for the
    system, 256 MiB for decoded cache, 32 MiB for IO, 256 MiB for context and
    128 MiB for compute workspace. Risky real-checkpoint experiments still
    require a supervised child process with independent swap/RSS limits.
    """
    def __init__(self,path,next_config=None,*,kernel='reference',cache_decay_interval=None,packed_copy_free=False,bulk_metadata=False):
        self.root=Path(path).expanduser().resolve();self.lock=RLock();self.closed=False
        self.checkpoint=None;self.reader=None
        if next_config is None:
            next_config=NextConfig(cache_bytes=256<<20,io_bytes=32<<20,kv_bytes=256<<20,
                workspace_bytes=128<<20,reserve_bytes=4<<30,max_tensor_bytes=16<<20,
                row_chunk=256,prefill_chunk=1,max_context=512,policy='adaptive',prefetch=False,
                dtype='float32',load_tokenizer=False)
        self.options=next_config if isinstance(next_config,NextConfig) else NextConfig(**next_config)
        if cache_decay_interval is not None:
            if self.options.policy!='adaptive' or type(cache_decay_interval) is not int or cache_decay_interval<1:
                raise ValueError('cache_decay_interval requires adaptive policy and a positive integer')
        if self.options.quantized_dir or self.options.lossless_dir or self.options.spec_prefetch:
            raise NotImplementedError('GGUF sidecars and speculative prefetch are unsupported')
        if self.options.io_mode=='mmap':raise NotImplementedError('Native GGUF uses bounded pread, not mmap')
        if self.options.io_bytes<1:raise ValueError('Native GGUF requires a positive io_bytes budget')
        begin=perf_counter()
        try:
            from .gguf import GGUFReader
            self.checkpoint=GGUFReader(self.root,memory_budget_bytes=self.options.io_bytes,
                max_read_bytes=self.options.max_tensor_bytes,include_tokenizer_metadata=False,
                bulk_metadata=bulk_metadata)
            self.config=gguf_qwen_config(self.checkpoint);self._validate_layout()
            if self.options.io_mode=='pread_nocache':
                # Apply only to this backend's own descriptor, never a service.
                from .physical_io import set_fd_nocache
                set_fd_nocache(self.checkpoint.fileno())
            import mlx.core as mx
            self.mx=mx;self.dtype=getattr(mx,self.options.dtype)
            self.device='gpu' if self.options.device=='auto' and mx.metal.is_available() else 'cpu' if self.options.device=='auto' else self.options.device
            if self.device=='gpu' and not mx.metal.is_available():raise RuntimeError('Metal GPU is unavailable')
            self.stream=mx.new_stream(getattr(mx,self.device))
            if mx.metal.is_available():
                mx.set_cache_limit(min(32<<20,self.options.workspace_bytes//4))
                # NumPy's packed cache lives outside MLX: never grant that
                # allowance again to the device allocator. Decoded cache items
                # are conservatively charged against both ceilings.
                limit=self.options.mlx_memory_bytes-(self.options.cache_bytes if kernel=='metal' else 0)
                mx.set_memory_limit(limit)
                self.allocator_limit_bytes=limit
            self.stats=defaultdict(float)
            if kernel=='metal' and self.device!='gpu':raise ValueError('Metal packed kernel requires device=gpu')
            self.reader=GGUFWeights(self.checkpoint,mx,self.options,self.config,kernel=kernel,packed_copy_free=packed_copy_free)
            if cache_decay_interval is not None:self.reader.cache.decay_interval=cache_decay_interval
            self.mixers=[(QwenDeltaNet if k=='linear_attention' else QwenFullAttention)(mx,self.reader,self.config,i) for i,k in enumerate(self.config['layer_types'])]
            self.mlps=[GGUFQwenSparseMLP(mx,self.reader,self.config,i) for i in range(self.config['num_hidden_layers'])]
            self.tokenizer=None;self.stats['startup_seconds']=perf_counter()-begin
        except BaseException:
            self.close();raise

    def _validate_layout(self):
        c=self.config;h=c['hidden_size'];d=c['head_dim'];a=c['num_attention_heads'];kh=c['num_key_value_heads']
        shapes={'token_embd.weight':(c['vocab_size'],h),'output_norm.weight':(h,)}
        if not c['tie_word_embeddings']:shapes['output.weight']=(c['vocab_size'],h)
        for i,kind in enumerate(c['layer_types']):
            p=f'blk.{i}.';m=c['moe_intermediate_size'];sm=c['shared_expert_intermediate_size'];e=c['num_experts']
            local={'attn_norm.weight':(h,),'post_attention_norm.weight':(h,),
                'ffn_gate_inp.weight':(e,h),'ffn_gate_inp_shexp.weight':(1,h),
                'ffn_gate_exps.weight':(e,m,h),'ffn_up_exps.weight':(e,m,h),'ffn_down_exps.weight':(e,h,m),
                'ffn_gate_shexp.weight':(sm,h),'ffn_up_shexp.weight':(sm,h),'ffn_down_shexp.weight':(h,sm)}
            if kind=='full_attention':
                local.update({'attn_q.weight':(a*2*d,h),'attn_k.weight':(kh*d,h),'attn_v.weight':(kh*d,h),
                    'attn_output.weight':(h,a*d),'attn_q_norm.weight':(d,),'attn_k_norm.weight':(d,)})
            else:
                kd=c['linear_num_key_heads']*c['linear_key_head_dim'];vd=c['linear_num_value_heads']*c['linear_value_head_dim'];vh=c['linear_num_value_heads']
                local.update({'attn_qkv.weight':(2*kd+vd,h),'attn_gate.weight':(vd,h),
                    'ssm_alpha.weight':(vh,h),'ssm_beta.weight':(vh,h),'ssm_conv1d.weight':(2*kd+vd,c['linear_conv_kernel_dim']),
                    'ssm_a':(vh,),'ssm_dt':(vh,),'ssm_norm.weight':(c['linear_value_head_dim'],),'ssm_out.weight':(h,vd)})
            shapes.update({p+n:s for n,s in local.items()})
        for name,expected in shapes.items():
            actual=self.checkpoint.tensors.get(name)
            if actual is None or tuple(actual.shape)!=expected:
                raise ValueError(f'Missing/invalid GGUF Qwen tensor {name}; expected {expected}')

    def metrics(self):
        return {'backend':'next-gguf-qwen-experimental','compute_device':self.device,'experimental':True,
            'full_qwen_checkpoint_validated':False,'metal_validated':False,'mtp_execution':False,
            'vision_execution':False,'runtime':dict(self.stats),'io':dict(self.checkpoint.stats),
            'allocator':{'limit_bytes':getattr(self,'allocator_limit_bytes',None),
                         'peak_bytes':self.mx.get_peak_memory() if self.mx.metal.is_available() else None,
                         'packed_cache_allowance_is_separate':self.reader.kernel=='metal'},
            'weights':dict(self.reader.stats),'kernel':self.reader.kernel,'cache':dict(self.reader.cache.stats)|{
                'bytes':self.reader.cache.bytes,'peak_bytes':self.reader.cache.peak_bytes,'capacity':self.reader.cache.capacity,
                'decay_interval':getattr(self.reader.cache,'decay_interval',None)},
            'prefetch':{'enabled':False},'numerics':{'compute_dtype':self.options.dtype,'text_mrope':'equal positions on all axes'}}


GGUFQwenModel=NativeGGUFQwenModel
