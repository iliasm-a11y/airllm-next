"""Experimental streamed qwen3_5_moe_text backend; miniature validation only.

Equations/weight conventions follow Qwen/HuggingFace Transformers (Apache-2.0,
Copyright 2025 The Qwen Team and The HuggingFace Inc. team), pinned numerical
reference in evidence-ooc/sources. No NVIDIA kernels or remote inference are
used. Full Qwen3.8 weights, MTP and Metal/M5 are not validated by this module.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import psutil
from .adapters import qwen_layer_types,validated_default_rope
from .cache import size_of
from .config import NextConfig
from .storage import OffsetCheckpoint, physical_read_bytes
from .tuning import auto_config, telemetry
from .weights import NativeWeights


def normalize_qwen_config(config):
    c=dict(config)
    if c.get('model_type')!='qwen3_5_moe_text':
        raise NotImplementedError('Experimental Qwen backend requires qwen3_5_moe_text')
    if c.get('quantization_config'):
        raise NotImplementedError('Experimental Qwen requires unquantized floating checkpoints; ModelOpt/NVFP4 is unsupported')
    if c.get('hidden_act','silu')!='silu' or c.get('attention_bias',False):
        raise NotImplementedError('Experimental Qwen requires bias-free SiLU blocks')
    # Current public Transformers uses sigmoid, without reading output_gate_type.
    # Do not silently ignore the newer Qwen3.8 swish declaration.
    if c.get('output_gate_type','sigmoid')!='sigmoid' or not c.get('attn_output_gate',True):
        raise NotImplementedError('Only independently verified sigmoid attention gates are supported; swish Qwen3.8 requires separate validation')
    rope=validated_default_rope(c)
    c['rope_parameters']=dict(rope)
    c['rope_parameters'].setdefault('rope_theta',c.get('rope_theta',10000000.))
    c['partial_rotary_factor']=rope.get('partial_rotary_factor',c.get('partial_rotary_factor',.25))
    c['layer_types']=qwen_layer_types(c)
    required=('hidden_size','vocab_size','num_attention_heads','num_key_value_heads','head_dim',
              'linear_conv_kernel_dim','linear_key_head_dim','linear_value_head_dim',
              'linear_num_key_heads','linear_num_value_heads','num_experts','num_experts_per_tok',
              'moe_intermediate_size','shared_expert_intermediate_size','max_position_embeddings')
    for name in required:
        if type(c.get(name)) is not int or c[name]<1:raise ValueError(f'Invalid Qwen dimension {name}')
    if c['num_attention_heads']%c['num_key_value_heads'] or c['linear_num_value_heads']%c['linear_num_key_heads']:
        raise ValueError('Invalid Qwen grouped-query head ratios')
    rd=int(c['head_dim']*c['partial_rotary_factor'])
    if rd<2 or rd%2 or rd>c['head_dim']:raise ValueError('Invalid Qwen rotary dimension')
    if not 0<c['num_experts_per_tok']<=c['num_experts']:raise ValueError('Invalid Qwen top-k')
    if c.get('mamba_ssm_dtype','float32')!='float32':raise NotImplementedError('Experimental Qwen state must use FP32')
    c.setdefault('rms_norm_eps',1e-6)
    return c


@dataclass
class QwenSnapshot:
    owner: object
    position: int
    banks: dict
    nbytes: int
    active: bool=True


class QwenSession:
    """Immutable array banks and budgeted snapshots for recurrent rollback.

    Updates replace arrays, never overwrite a snapshot. Retained snapshot bytes
    are charged conservatively even if an array still shares storage. All cache
    payloads including convolution, FP32 recurrence and full K/V share one cap.
    """
    def __init__(self,owner):
        self.owner=owner;self.position=0;self.banks={'kv':{},'conv':{},'recurrent':{}}
        self.budget=owner.options.kv_bytes;self.limit=min(owner.options.max_context,owner.config['max_position_embeddings'])
        self.retained=0;self.peak_bytes=0

    @property
    def bytes(self):return size_of(self.banks)

    def reserve(self,bank,layer,new_bytes):
        old=self.banks[bank].get(layer)
        old_bytes=size_of(old) if old is not None else 0
        predicted=self.bytes+self.retained+max(0,new_bytes-old_bytes)
        if predicted>self.budget:raise MemoryError('Qwen context plus retained rollback states exceeds kv_bytes')
        self.peak_bytes=max(self.peak_bytes,predicted)

    def store(self,bank,layer,value):
        self.reserve(bank,layer,size_of(value));self.banks[bank][layer]=value

    def snapshot(self):
        n=self.bytes
        if self.bytes+self.retained+n>self.budget:raise MemoryError('Qwen rollback snapshot exceeds kv_bytes')
        snapshot=QwenSnapshot(self,self.position,{k:dict(v) for k,v in self.banks.items()},n)
        self.retained+=n;self.peak_bytes=max(self.peak_bytes,self.bytes+self.retained)
        return snapshot

    def _check_snapshot(self,snapshot):
        if not isinstance(snapshot,QwenSnapshot) or snapshot.owner is not self or not snapshot.active:
            raise ValueError('Snapshot is inactive or belongs to a different session')

    def release(self,snapshot):
        self._check_snapshot(snapshot);self.retained-=snapshot.nbytes;snapshot.active=False;snapshot.banks={}

    def restore(self,snapshot):
        self._check_snapshot(snapshot);self.position=snapshot.position
        self.banks={k:dict(v) for k,v in snapshot.banks.items()};self.release(snapshot)

    def truncate(self,position):
        if position!=self.position:
            raise NotImplementedError('Recurrent rollback requires an explicit snapshot; token-offset truncation is invalid')


def zero_centered_rms(mx,x,weight,eps):
    y=x.astype(mx.float32)
    y=y*mx.rsqrt(mx.mean(y*y,axis=-1,keepdims=True)+eps)
    return (y*(1+weight.astype(mx.float32))).astype(x.dtype)


def qwen_rope(mx,x,offset,theta,dimension):
    positions=mx.arange(offset,offset+x.shape[-2],dtype=mx.float32)
    inv=theta**(-mx.arange(0,dimension,2,dtype=mx.float32)/dimension)
    angle=positions[:,None]*inv[None,:];angle=mx.concatenate([angle,angle],axis=-1)
    cos=mx.cos(angle).astype(x.dtype);sin=mx.sin(angle).astype(x.dtype)
    rot=x[...,:dimension];half=dimension//2
    rotated=mx.concatenate([-rot[...,half:],rot[...,:half]],axis=-1)
    return mx.concatenate([rot*cos+rotated*sin,x[...,dimension:]],axis=-1)


class QwenFullAttention:
    def __init__(self,mx,reader,config,layer):
        self.mx=mx;self.r=reader;self.c=config;self.layer=layer;self.p=f'model.layers.{layer}.self_attn.'

    def __call__(self,x,cache):
        mx=self.mx;c=self.c;B,S,_=x.shape;H=c['num_attention_heads'];KH=c['num_key_value_heads'];D=c['head_dim']
        def linear(name):return self.r.linear(x,self.p+name+'.weight').astype(x.dtype)
        qg=linear('q_proj').reshape(B,S,H,2*D);q=qg[...,:D];gate=qg[...,D:].reshape(B,S,H*D)
        q=zero_centered_rms(mx,q,self.r.get(self.p+'q_norm.weight'),c['rms_norm_eps']).transpose(0,2,1,3)
        k=linear('k_proj').reshape(B,S,KH,D)
        k=zero_centered_rms(mx,k,self.r.get(self.p+'k_norm.weight'),c['rms_norm_eps']).transpose(0,2,1,3)
        v=linear('v_proj').reshape(B,S,KH,D).transpose(0,2,1,3)
        dim=int(D*c['partial_rotary_factor']);theta=c['rope_parameters']['rope_theta']
        q=qwen_rope(mx,q,cache.position,theta,dim);k=qwen_rope(mx,k,cache.position,theta,dim)
        previous=cache.banks['kv'].get(self.layer)
        T=cache.position+S;cache.reserve('kv',self.layer,2*B*KH*T*D*x.itemsize)
        if previous is not None:
            k=mx.concatenate([previous[0],k],axis=2);v=mx.concatenate([previous[1],v],axis=2)
        cache.store('kv',self.layer,(k,v))
        groups=H//KH;k=mx.repeat(k,groups,axis=1);v=mx.repeat(v,groups,axis=1)
        workspace=B*H*S*T*4
        if workspace>self.r.cfg.workspace_bytes:raise MemoryError('Qwen attention score workspace exceeds workspace_bytes; reduce prefill_chunk/context')
        score=(q@k.transpose(0,1,3,2))*D**-.5
        mask=mx.arange(T)[None,:]<=mx.arange(cache.position,T)[:,None]
        score=mx.where(mask[None,None],score,-float('inf'))
        prob=mx.softmax(score.astype(mx.float32),axis=-1).astype(x.dtype)
        out=(prob@v).transpose(0,2,1,3).reshape(B,S,H*D)
        out=out*mx.sigmoid(gate)
        return self.r.linear(out,self.p+'o_proj.weight').astype(x.dtype)


class QwenDeltaNet:
    def __init__(self,mx,reader,config,layer):
        self.mx=mx;self.r=reader;self.c=config;self.layer=layer;self.p=f'model.layers.{layer}.linear_attn.'

    def __call__(self,x,cache):
        mx=self.mx;c=self.c;B,S,_=x.shape
        KH=c['linear_num_key_heads'];VH=c['linear_num_value_heads'];KD=c['linear_key_head_dim'];VD=c['linear_value_head_dim']
        keydim=KH*KD;valuedim=VH*VD;C=2*keydim+valuedim;K=c['linear_conv_kernel_dim']
        def linear(name):return self.r.linear(x,self.p+name+'.weight').astype(x.dtype)
        qkv=linear('in_proj_qkv');z=linear('in_proj_z').reshape(B,S,VH,VD)
        b=linear('in_proj_b');a=linear('in_proj_a')
        cache.reserve('conv',self.layer,B*K*C*x.itemsize)
        previous=cache.banks['conv'].get(self.layer)
        history=mx.zeros((B,K-1,C),dtype=x.dtype) if previous is None else previous[:,1:]
        conv_input=mx.concatenate([history,qkv],axis=1)
        new_conv=conv_input[:,-K:]
        if new_conv.shape[1]<K:new_conv=mx.concatenate([mx.zeros((B,K-new_conv.shape[1],C),dtype=x.dtype),new_conv],axis=1)
        cache.store('conv',self.layer,mx.contiguous(new_conv))
        w=self.r.get(self.p+'conv1d.weight')[:,0,:].astype(x.dtype)
        mixed=sum(conv_input[:,j:j+S,:]*w[:,j] for j in range(K))
        mixed=mixed*mx.sigmoid(mixed)
        q=mixed[...,:keydim].reshape(B,S,KH,KD).astype(mx.float32)
        k=mixed[...,keydim:2*keydim].reshape(B,S,KH,KD).astype(mx.float32)
        v=mixed[...,2*keydim:].reshape(B,S,VH,VD).astype(mx.float32)
        q=q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-6)/math.sqrt(KD)
        k=k*mx.rsqrt(mx.sum(k*k,axis=-1,keepdims=True)+1e-6)
        if VH!=KH:q=mx.repeat(q,VH//KH,axis=2);k=mx.repeat(k,VH//KH,axis=2)
        beta=mx.sigmoid(b).astype(mx.float32)
        step=a.astype(mx.float32)+self.r.get(self.p+'dt_bias').astype(mx.float32)
        softplus=mx.maximum(step,0)+mx.log1p(mx.exp(-mx.abs(step)))
        decay=-mx.exp(self.r.get(self.p+'A_log').astype(mx.float32))*softplus
        cache.reserve('recurrent',self.layer,B*VH*KD*VD*4)
        state=cache.banks['recurrent'].get(self.layer)
        if state is None:state=mx.zeros((B,VH,KD,VD),dtype=mx.float32)
        outs=[]
        for i in range(S):
            state=state*mx.exp(decay[:,i])[...,None,None]
            memory=mx.sum(state*k[:,i,:,:,None],axis=-2)
            delta=(v[:,i]-memory)*beta[:,i,:,None]
            state=state+k[:,i,:,:,None]*delta[:,:,None,:]
            one=mx.sum(state*q[:,i,:,:,None],axis=-2)
            # Bound graph lifetime even for a long prefill block. The portable
            # recurrence deliberately realizes each step; native fusion is a
            # later measured optimization, not assumed in this reference.
            mx.eval(state,one);outs.append(one)
        cache.store('recurrent',self.layer,state)
        out=mx.stack(outs,axis=1).astype(x.dtype)
        out32=out.astype(mx.float32)*mx.rsqrt(mx.mean(out.astype(mx.float32)**2,axis=-1,keepdims=True)+c['rms_norm_eps'])
        out=(out32.astype(x.dtype)*self.r.get(self.p+'norm.weight').astype(x.dtype)).astype(x.dtype)
        out=(out*(z.astype(mx.float32)*mx.sigmoid(z.astype(mx.float32)))).astype(x.dtype)
        return self.r.linear(out.reshape(B,S,valuedim),self.p+'out_proj.weight').astype(x.dtype)


class ExperimentalQwenModel:
    """Local-only experimental constructor. No implicit architecture dispatch.

    Supports nonempty unpadded batch one, ordinary greedy generation and the
    validated floating qwen3_5 schema. MTP, ModelOpt and swish-gate variants fail
    explicitly. Production/Qwen3.8 checkpoint compatibility remains unverified.
    """
    supports_sampling=False

    def __init__(self,root,next_config=None):
        self.root=Path(root).resolve();self.lock=RLock();self.closed=False;self.checkpoint=None;self.reader=None
        self.options=next_config if isinstance(next_config,NextConfig) else auto_config() if next_config is None else NextConfig(**next_config)
        self.config=normalize_qwen_config(json.loads((self.root/'config.json').read_text()))
        if self.options.quantized_dir:raise NotImplementedError('Qwen sidecar quantization is not validated')
        import mlx.core as mx
        self.mx=mx;self.dtype=getattr(mx,self.options.dtype)
        self.device='gpu' if self.options.device=='auto' and mx.metal.is_available() else 'cpu' if self.options.device=='auto' else self.options.device
        if self.device=='gpu' and not mx.metal.is_available():raise RuntimeError('Metal GPU is unavailable')
        self.stream=mx.new_stream(getattr(mx,self.device));self.stats=defaultdict(float);start=perf_counter()
        if mx.metal.is_available():
            mx.set_cache_limit(min(256*1024**2,self.options.workspace_bytes//4))
            mx.set_memory_limit(self.options.cache_bytes+self.options.io_bytes+self.options.kv_bytes+self.options.workspace_bytes)
        try:
            self.checkpoint=OffsetCheckpoint(self.root,self.options.metadata_dir(),self.options.max_open_files,self.options.max_tensor_bytes)
            if any(d['dtype'] not in ('F32','F16','BF16') for d in self.checkpoint.tensors.values()):
                raise NotImplementedError('Experimental Qwen accepts F32/F16/BF16 tensors only')
            self.reader=NativeWeights(self.checkpoint,mx,self.options,self.config)
            self._validate_layout()
            from .qwen_moe import QwenSparseMLP
            self.mixers=[(QwenDeltaNet if kind=='linear_attention' else QwenFullAttention)(mx,self.reader,self.config,i) for i,kind in enumerate(self.config['layer_types'])]
            self.mlps=[QwenSparseMLP(mx,self.reader,self.config,i) for i in range(self.config['num_hidden_layers'])]
            self.tokenizer=None
            if self.options.load_tokenizer:
                from transformers import AutoTokenizer
                self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,trust_remote_code=False)
            self.stats['startup_seconds']=perf_counter()-start
        except BaseException:
            self.close();raise

    def _validate_layout(self):
        c=self.config;H=c['hidden_size'];V=c['vocab_size'];D=c['head_dim'];A=c['num_attention_heads'];KH=c['num_key_value_heads']
        required={'model.embed_tokens.weight':(V,H),'model.norm.weight':(H,)}
        if not c.get('tie_word_embeddings',False):required['lm_head.weight']=(V,H)
        for i,kind in enumerate(c['layer_types']):
            p=f'model.layers.{i}.';required[p+'input_layernorm.weight']=(H,);required[p+'post_attention_layernorm.weight']=(H,)
            if kind=='full_attention':
                a=p+'self_attn.'
                required.update({a+'q_proj.weight':(A*2*D,H),a+'k_proj.weight':(KH*D,H),a+'v_proj.weight':(KH*D,H),
                                 a+'o_proj.weight':(H,A*D),a+'q_norm.weight':(D,),a+'k_norm.weight':(D,)})
            else:
                a=p+'linear_attn.';kd=c['linear_num_key_heads']*c['linear_key_head_dim'];vd=c['linear_num_value_heads']*c['linear_value_head_dim'];vh=c['linear_num_value_heads'];C=2*kd+vd
                required.update({a+'in_proj_qkv.weight':(C,H),a+'in_proj_z.weight':(vd,H),a+'in_proj_a.weight':(vh,H),a+'in_proj_b.weight':(vh,H),
                                 a+'conv1d.weight':(C,1,c['linear_conv_kernel_dim']),a+'dt_bias':(vh,),a+'A_log':(vh,),
                                 a+'norm.weight':(c['linear_value_head_dim'],),a+'out_proj.weight':(H,vd)})
            m=p+'mlp.';M=c['moe_intermediate_size'];E=c['num_experts'];SM=c['shared_expert_intermediate_size']
            required[m+'gate.weight']=(E,H);required[m+'shared_expert_gate.weight']=(1,H)
            for n,shape in {'gate_proj.weight':(SM,H),'up_proj.weight':(SM,H),'down_proj.weight':(H,SM)}.items():required[m+'shared_expert.'+n]=shape
            if m+'experts.gate_up_proj' in self.checkpoint.tensors:
                required[m+'experts.gate_up_proj']=(E,2*M,H);required[m+'experts.down_proj']=(E,H,M)
            else:
                for e in range(E):
                    for n,shape in {'gate_proj.weight':(M,H),'up_proj.weight':(M,H),'down_proj.weight':(H,M)}.items():required[m+f'experts.{e}.'+n]=shape
        for key,shape in required.items():
            if key not in self.checkpoint.tensors or tuple(self.checkpoint.tensors[key]['shape'])!=shape:
                raise ValueError(f'Missing or invalid Qwen weight {key}; expected {shape}')

    def new_session(self):return QwenSession(self)

    def forward(self,input_ids,past_key_values=None,use_cache=True,logits_to_keep=0):
        with self.lock,self.mx.stream(self.stream):
            if self.closed:raise RuntimeError('Model is closed')
            if hasattr(input_ids,'detach'):input_ids=input_ids.detach().cpu().numpy()
            ids=np.asarray(input_ids)
            if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1 or ids.dtype.kind not in 'iu':raise ValueError('Expected nonempty unpadded integer batch one')
            cache=past_key_values if past_key_values is not None else self.new_session()
            if not isinstance(cache,QwenSession) or cache.owner is not self:raise ValueError('Cache belongs to another model')
            if cache.position+ids.shape[1]>cache.limit:raise MemoryError('Qwen context limit exceeded')
            snapshot=cache.snapshot();mx=self.mx;outputs=[]
            try:
                for start in range(0,ids.shape[1],self.options.prefill_chunk):
                    if psutil.virtual_memory().available<self.options.reserve_bytes:raise MemoryError('System reserve reached')
                    part=ids[:,start:start+self.options.prefill_chunk]
                    x=self.reader.embedding(part,'model.embed_tokens.weight').astype(self.dtype);mx.eval(x)
                    for i,(mixer,mlp) in enumerate(zip(self.mixers,self.mlps)):
                        p=f'model.layers.{i}.'
                        norm=zero_centered_rms(mx,x,self.reader.get(p+'input_layernorm.weight'),self.config['rms_norm_eps'])
                        x=x+mixer(norm,cache);mx.eval(x)
                        norm=zero_centered_rms(mx,x,self.reader.get(p+'post_attention_layernorm.weight'),self.config['rms_norm_eps'])
                        x=x+mlp(norm).astype(x.dtype);mx.eval(x)
                    x=zero_centered_rms(mx,x,self.reader.get('model.norm.weight'),self.config['rms_norm_eps'])
                    if not logits_to_keep or start+part.shape[1]==ids.shape[1]:
                        head='model.embed_tokens.weight' if self.config.get('tie_word_embeddings',False) else 'lm_head.weight'
                        logits=self.reader.output(x if not logits_to_keep else x[:,-logits_to_keep:],head);mx.eval(logits);outputs.append(logits)
                    cache.position+=part.shape[1]
                    for bank in cache.banks.values():
                        for value in bank.values():mx.eval(value)
                logits=mx.concatenate(outputs,axis=1) if len(outputs)>1 else outputs[0]
                cache.release(snapshot)
                return SimpleNamespace(logits=logits,past_key_values=cache if use_cache else None)
            except BaseException:
                cache.restore(snapshot);self.reader.io.drain();raise

    __call__=forward

    def generate(self,input_ids,max_new_tokens=32,eos_token_id=None,temperature=0.,**kwargs):
        if any(name in kwargs for name in ('draft_model','proposal_depth','speculative_depth')):
            raise NotImplementedError('Experimental Qwen speculative verification requires recurrent snapshot integration and is not enabled')
        if kwargs:raise TypeError(f'Unsupported Qwen generation options: {sorted(kwargs)}')
        if type(max_new_tokens) is not int or max_new_tokens<0 or temperature!=0:raise ValueError('Experimental Qwen supports nonnegative token count and greedy temperature=0 only')
        if hasattr(input_ids,'detach'):input_ids=input_ids.detach().cpu().numpy()
        host=np.asarray(input_ids)
        if host.ndim!=2 or host.shape[0]!=1 or host.shape[1]<1 or host.dtype.kind not in 'iu':raise ValueError('Expected integer batch one')
        with self.lock:
            tokens=host.tolist()[0];current=host;cache=None;begin=perf_counter();first=None
            logical=self.checkpoint.stats['logical_bytes'];physical=physical_read_bytes()
            eos=self.config.get('eos_token_id') if eos_token_id is None else eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            for i in range(max_new_tokens):
                out=self.forward(current,past_key_values=cache,logits_to_keep=1);cache=out.past_key_values
                token=int(self.mx.argmax(out.logits[0,-1]).item());tokens.append(token)
                if first is None:first=perf_counter()
                if token in eos:break
                current=np.array([[token]],dtype=np.int32)
            end=perf_counter();n=len(tokens)-host.shape[1];physical_end=physical_read_bytes()
            self.stats.update({'tokens_generated':n,'generation_seconds':end-begin,'ttft_seconds':first-begin if first else 0.,
                               'decode_tokens_per_second':(n-1)/(end-first) if n>1 and end>first else 0.,
                               'logical_bytes_per_generated_token':(self.checkpoint.stats['logical_bytes']-logical)/n if n else 0.,
                               'physical_read_bytes':physical_end-physical if physical is not None and physical_end is not None else None,
                               'context_payload_peak_bytes':cache.peak_bytes if cache else 0})
            return self.mx.array([tokens],dtype=self.mx.int32)

    def metrics(self):
        return {'compute_device':self.device,'experimental':True,'full_qwen_checkpoint_validated':False,'metal_validated':False,
                'mtp_execution':False,'runtime':dict(self.stats),'io':dict(self.checkpoint.stats),'telemetry':telemetry(),
                'cache':dict(self.reader.cache.stats)|{'bytes':self.reader.cache.bytes,'peak_bytes':self.reader.cache.peak_bytes}}

    def close(self):
        with self.lock:
            if self.closed:return
            try:
                if self.reader:self.reader.close()
            finally:
                if self.checkpoint:self.checkpoint.close()
                self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
