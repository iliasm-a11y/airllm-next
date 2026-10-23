"""Native MLX GLM-5.3 engine integrated into AirLLM.AutoModel.

The CPU MLX path is a real numerical validation target. It is not a Metal
benchmark. This backend deliberately rejects unsupported architecture variants.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
import os
import psutil
import numpy as np
from ..ultra.checkpoint import atomic_json
from .config import NextConfig
from .storage import physical_read_bytes,encoded_read_bytes
from .weights import NativeWeights
from .attention import DSAAttention,LatentKV,rms
from .moe import SparseMLP
from .tuning import process_rss,auto_config,runtime_tuning_key,telemetry
from .generation import token_batch,eos_ids,generation_options,logits_suffix_start,check_logits_budget


def normalize_config(cfg):
    c=dict(cfg)
    from .adapters import get_adapter,validated_default_rope
    get_adapter(c).require_runtime(c)
    if c.get('model_type')!='glm_moe_dsa':raise NotImplementedError('Native Next currently implements glm_moe_dsa; use Ultra/upstream for other models')
    if c.get('hidden_act','silu')!='silu' or c.get('attention_bias',False):raise NotImplementedError('Only official bias-free SiLU GLM is implemented')
    q=c.get('quantization_config',{})
    if q and (q.get('quant_method')!='fp8' or q.get('fmt','e4m3')!='e4m3'):raise NotImplementedError('Only official block E4M3 FP8 is supported')
    rope=validated_default_rope(c)
    if not c.get('rope_interleave',True) or not c.get('indexer_rope_interleave',True):raise NotImplementedError('Only official interleaved GLM RoPE is supported')
    c['rope_parameters']=dict(rope)
    c['rope_parameters'].setdefault('rope_theta',c.get('rope_theta',10000.))
    c['rope_parameters'].setdefault('rope_type','default')
    for name in ('hidden_size','vocab_size','num_hidden_layers','num_attention_heads','intermediate_size','max_position_embeddings',
                 'q_lora_rank','kv_lora_rank','qk_rope_head_dim','qk_nope_head_dim','v_head_dim',
                 'index_head_dim','index_n_heads','index_topk','n_routed_experts','moe_intermediate_size','num_experts_per_tok'):
        if type(c.get(name)) is not int or c[name]<1:raise ValueError(f'Invalid GLM dimension {name}')
    L=c['num_hidden_layers'];dense=c.get('first_k_dense_replace',1)
    if type(dense) is not int or not 0<=dense<=L:raise ValueError('Invalid GLM dense layer count')
    if c.get('mlp_layer_types') is None:c['mlp_layer_types']=['dense']*dense+['sparse']*(L-dense)
    if c.get('indexer_types') is None:
        freq=c.get('index_topk_freq',1);offset=c.get('index_skip_topk_offset',2)
        if type(freq) is not int or freq<1 or type(offset) is not int or offset<0:raise ValueError('Invalid DSA indexer frequency/offset')
        c['indexer_types']=['full' if max(i-offset+1,0)%freq==0 else 'shared' for i in range(L)]
    if len(c['indexer_types'])!=L or c['indexer_types'][0]!='full' or any(x not in ('full','shared') for x in c['indexer_types']):raise ValueError('Invalid DSA indexer schedule')
    if len(c['mlp_layer_types'])!=L or any(x not in ('dense','sparse') for x in c['mlp_layer_types']):raise ValueError('Invalid GLM MLP schedule')
    if c.get('scoring_func','sigmoid')!='sigmoid' or c.get('topk_method','noaux_tc')!='noaux_tc':raise NotImplementedError('Only official noaux_tc sigmoid route is implemented')
    if c.get('n_shared_experts',1)!=1:raise NotImplementedError('Only one shared expert is implemented')
    if c['qk_rope_head_dim']%2 or c['index_head_dim']<c['qk_rope_head_dim']:raise ValueError('Invalid rotary/indexer dimensions')
    c.setdefault('norm_topk_prob',True);c.setdefault('n_group',1);c.setdefault('topk_group',1);c.setdefault('routed_scaling_factor',2.5)
    if type(c['n_group']) is not int or c['n_group']<1 or type(c['topk_group']) is not int or not 1<=c['topk_group']<=c['n_group']:
        raise ValueError('Invalid MoE routing groups')
    if c['n_routed_experts']%c['n_group'] or not 0<c['num_experts_per_tok']<=c['n_routed_experts']:raise ValueError('Invalid MoE routing dimensions')
    per_group=c['n_routed_experts']//c['n_group']
    if per_group<2 or c['num_experts_per_tok']>per_group*c['topk_group']:raise ValueError('GLM routing groups cannot supply the requested top-k')
    for name,default in (('rms_norm_eps',1e-5),('routed_scaling_factor',2.5)):
        value=c.get(name,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be finite and positive')
    return c


class AirLLMNextModel:
    normalize_config=staticmethod(normalize_config)
    session_type=LatentKV

    def __init__(self,model_local_path_or_repo_id,next_config=None,**kwargs):
        if kwargs:raise TypeError(f'Unsupported Next options {sorted(kwargs)}')
        self.options=(next_config if isinstance(next_config,NextConfig) else
                      auto_config() if next_config is None else NextConfig(**next_config))
        self.root=Path(model_local_path_or_repo_id).resolve();self.lock=RLock();self.closed=False
        self.stats=defaultdict(float);self.routes={};self.checkpoint=None;self.reader=None
        start=perf_counter()
        if not (self.root/'config.json').is_file():raise FileNotFoundError('A complete local checkpoint is required; downloads are never implicit')
        self.config=self.normalize_config(json.loads((self.root/'config.json').read_text()))
        import mlx.core as mx
        self.mx=mx;self.dtype=getattr(mx,self.options.dtype)
        available=mx.metal.is_available()
        device='gpu' if self.options.device=='auto' and available else self.options.device
        if device=='auto':device='cpu'
        if device=='gpu' and not available:raise RuntimeError('Metal GPU unavailable; explicit CPU MLX validation remains possible')
        self.device=device;self.stream=mx.new_stream(getattr(mx,device));self.start_swap=psutil.swap_memory().used
        self._initial_rss=process_rss()
        self._profile=self.options.metadata_dir()
        # Do not pin or raise wired-memory limits to defeat macOS safety controls.
        if available:
            mx.set_cache_limit(min(256*1024**2,self.options.workspace_bytes//4))
            mx.set_memory_limit(self.options.mlx_memory_bytes)
        try:
            from .lossless import open_checkpoint
            self.checkpoint=open_checkpoint(self.root,self.options,cache_dir=self._profile)
            tuning_path=self._profile/(self.checkpoint.fingerprint+'.tuning.json')
            if self.options.use_saved_tuning and tuning_path.exists():
                try:
                    tune=json.loads(tuning_path.read_text())
                    if tune.get('source_fingerprint')==self.checkpoint.fingerprint:
                        workers=int(tune['selected_io_workers'])
                        if workers in (1,2,4):self.options=replace(self.options,io_workers=workers)
                except (OSError,ValueError,KeyError,TypeError):pass
            self.route_path=self._profile/(self.checkpoint.fingerprint+'.native-routes.json')
            if self.route_path.exists():
                try:self.routes={k:Counter({int(e):n for e,n in v.items()}) for k,v in json.loads(self.route_path.read_text()).items()}
                except (OSError,ValueError,TypeError):self.routes={}
            sidecar=None
            if self.options.quantized_dir:
                from .sidecar import QuantizedSidecar
                sidecar=QuantizedSidecar(self.options.quantized_dir,self.checkpoint.fingerprint)
            self.reader=NativeWeights(self.checkpoint,mx,self.options,self.config,sidecar)
            rt_path=self._profile/(self.checkpoint.fingerprint+'.runtime-tuning.json')
            if self.options.use_saved_tuning and self.options.prefetch and rt_path.exists():
                try:
                    tune=json.loads(rt_path.read_text())
                    if tune.get('key')==runtime_tuning_key(self):
                        self.reader.prefetch_enabled=bool(tune['prefetch_enabled'])
                        if self.options.policy!='adaptive' and tune.get('cache_policy') in ('slru','lru','lfu'):self.reader.cache.policy=tune['cache_policy']
                except (OSError,ValueError,KeyError,TypeError):pass
            self._build_layers()
            self._validate_layout()
            self.tokenizer=None
            if self.options.load_tokenizer:
                from transformers import AutoTokenizer
                self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,trust_remote_code=False)
            self.stats['startup_seconds']=perf_counter()-start
        except BaseException:
            self.close(save_profile=False);raise

    def _build_layers(self):
        self.attention=[DSAAttention(self.mx,self.reader,self.config,i) for i in range(self.config['num_hidden_layers'])]
        self.mlps=[SparseMLP(self.mx,self.reader,self.config,i,self.routes) for i in range(self.config['num_hidden_layers'])]

    def _validate_layout(self):
        cp=self.checkpoint;c=self.config;H=c['hidden_size'];A=c['num_attention_heads']
        required={'model.embed_tokens.weight':(c['vocab_size'],H),'model.norm.weight':(H,)}
        if not c.get('tie_word_embeddings',False):required['lm_head.weight']=(c['vocab_size'],H)
        for i in range(c['num_hidden_layers']):
            p=f'model.layers.{i}.';a=p+'self_attn.';Q=c['q_lora_rank'];R=c['kv_lora_rank'];D=c['qk_rope_head_dim'];N=c['qk_nope_head_dim'];V=c['v_head_dim']
            required.update({p+'input_layernorm.weight':(H,),p+'post_attention_layernorm.weight':(H,),
              a+'q_a_proj.weight':(Q,H),a+'q_a_layernorm.weight':(Q,),a+'q_b_proj.weight':(A*(N+D),Q),
              a+'kv_a_proj_with_mqa.weight':(R+D,H),a+'kv_a_layernorm.weight':(R,),a+'kv_b_proj.weight':(A*(N+V),R),a+'o_proj.weight':(H,A*V)})
            if c['indexer_types'][i]=='full':
                I=c['index_head_dim'];IH=c['index_n_heads'];ip=a+'indexer.'
                required.update({ip+'wq_b.weight':(I*IH,Q),ip+'wk.weight':(I,H),ip+'weights_proj.weight':(IH,H),ip+'k_norm.weight':(I,),ip+'k_norm.bias':(I,)})
            if c['mlp_layer_types'][i]=='dense':banks=[(p+'mlp.',c['intermediate_size'])]
            else:
                E=c['n_routed_experts'];M=c['moe_intermediate_size'];ep=p+'mlp.experts.'
                required[p+'mlp.gate.weight']=(E,H);required[p+'mlp.gate.e_score_correction_bias']=(E,)
                banks=[(p+'mlp.shared_experts.',M)]
                if ep+'gate_up_proj' in cp.tensors:
                    required[ep+'gate_up_proj']=(E,2*M,H);required[ep+'down_proj']=(E,H,M)
                else:banks += [(ep+str(e)+'.',M) for e in range(E)]
            for b,M in banks:
                required.update({b+'gate_proj.weight':(M,H),b+'up_proj.weight':(M,H),b+'down_proj.weight':(H,M)})
        for k,shape in required.items():
            if k not in cp.tensors or tuple(cp.tensors[k]['shape'])!=shape:raise ValueError(f'Missing or invalid GLM weight {k}, expected {shape}')
        # MTP tensors may exist as layer L; they are intentionally unused for
        # ordinary autoregressive decoding, not silently used as speculative tokens.

    def new_session(self):
        cache=LatentKV(self.mx,self.config,self.options.kv_bytes,min(self.options.max_context,self.config['max_position_embeddings']),self.dtype)
        cache.owner=self
        return cache

    def _pressure(self):
        vm=psutil.virtual_memory();rss=process_rss()
        if rss is not None:self.stats['sampled_rss_peak_bytes']=max(self.stats['sampled_rss_peak_bytes'],rss)
        self.stats['swap_growth_bytes']=max(0,psutil.swap_memory().used-self.start_swap)
        if vm.available < self.options.reserve_bytes+self.options.workspace_bytes:
            self.reader.trim()
            self.reader.drain()
            if vm.available<self.options.reserve_bytes:raise MemoryError('System memory reserve reached; context/weight budgets must be reduced')

    def forward(self,input_ids,past_key_values=None,use_cache=True,logits_to_keep=0):
        with self.lock,self.mx.stream(self.stream):
            if self.closed:raise RuntimeError('Model is closed')
            mx=self.mx
            host=token_batch(input_ids,self.config['vocab_size'])
            keep_from=logits_suffix_start(host.shape[1],logits_to_keep)
            check_logits_budget(host.shape[1]-keep_from,self.config['vocab_size'],self.options.workspace_bytes)
            cache=past_key_values if past_key_values is not None else self.new_session()
            if not isinstance(cache,self.session_type) or cache.mx is not mx or getattr(cache,'owner',None) is not self:raise ValueError('Incompatible KV cache')
            if host.shape[1]+cache.position>cache.limit:raise MemoryError('Context exceeds configured safe limit')
            old=cache.position;all_logits=[]
            try:
                for start in range(0,host.shape[1],self.options.prefill_chunk):
                    self._pressure();part=host[:,start:start+self.options.prefill_chunk];n=part.shape[1]
                    x=self.reader.embedding(part,'model.embed_tokens.weight').astype(self.dtype);mx.eval(x)
                    selected=None
                    for i,(attn,mlp) in enumerate(zip(self.attention,self.mlps)):
                        if (self.reader.spec and hasattr(mlp,'expert_keys') and
                            self.config['mlp_layer_types'][i]=='sparse'):
                            self.reader.begin_layer(i,mlp.expert_keys)
                        p=f'model.layers.{i}.'
                        norm=rms(mx,x,self.reader.get(p+'input_layernorm.weight'),self.config.get('rms_norm_eps',1e-5))
                        a,selected=attn(norm,cache,selected);x=x+a;mx.eval(x,selected)
                        norm=rms(mx,x,self.reader.get(p+'post_attention_layernorm.weight'),self.config.get('rms_norm_eps',1e-5))
                        x=x+mlp(norm);mx.eval(x)
                    x=rms(mx,x,self.reader.get('model.norm.weight'),self.config.get('rms_norm_eps',1e-5))
                    # Only final requested logits are computed in generate/prefill.
                    if start+n>keep_from:
                        keep=x[:,max(0,keep_from-start):]
                        head='model.embed_tokens.weight' if self.config.get('tie_word_embeddings') else 'lm_head.weight'
                        logits=self.reader.output(keep,head);mx.eval(logits);all_logits.append(logits)
                    cache.position+=n
                    # Force buffers now: a cache must not retain all prior lazy
                    # graphs, weights or temporary expert results.
                    for buffers in cache.layers.values():mx.eval(*buffers)
                logits=mx.concatenate(all_logits,axis=1) if len(all_logits)>1 else all_logits[0]
                return SimpleNamespace(logits=logits,past_key_values=cache if use_cache else None)
            except BaseException:
                # Append-only cache: committed prefix remains valid. Uncommitted
                # tail positions will be overwritten on retry.
                cache.position=old
                self.reader.drain()
                raise

    __call__=forward

    def generate(self,input_ids,max_new_tokens=32,eos_token_id=None,temperature=0.,
                 draft_model=None,speculative_depth=4,speculative_adaptive=True,**kwargs):
        if kwargs:raise TypeError(f'Unsupported generate options {sorted(kwargs)}')
        max_new_tokens,temperature=generation_options(max_new_tokens,temperature)
        if draft_model is not None:
            if temperature!=0:raise NotImplementedError('Speculative decoding currently supports exact greedy decoding only')
            return self.generate_speculative(input_ids,draft_model,max_new_tokens=max_new_tokens,
                      eos_token_id=eos_token_id,max_depth=speculative_depth,adaptive=speculative_adaptive)
        with self.lock:
            if self.closed:raise RuntimeError('Model is closed')
            self.stats.pop('speculative',None)
            host=token_batch(input_ids,self.config['vocab_size'])
            tokens=host.tolist()[0];current=host;cache=None;mx=self.mx
            eos=self.config.get('eos_token_id') if eos_token_id is None else eos_token_id
            eos=eos_ids(eos,self.config['vocab_size']);begin=perf_counter();first=None
            logical=self.checkpoint.stats['logical_bytes'];physical=physical_read_bytes();after_prefill_logical=logical
            encoded_start=encoded_read_bytes(self.checkpoint);after_prefill_encoded=encoded_start
            for i in range(max_new_tokens):
                output=self.forward(current,past_key_values=cache,logits_to_keep=1);cache=output.past_key_values
                with mx.stream(self.stream):
                    last=output.logits[:,-1,:]
                    token=mx.argmax(last,axis=-1) if temperature==0 else mx.random.categorical(last/temperature)
                    value=int(token.item())
                tokens.append(value)
                if i==0:
                    first=perf_counter();after_prefill_logical=self.checkpoint.stats['logical_bytes']
                    after_prefill_encoded=encoded_read_bytes(self.checkpoint)
                if value in eos:break
                current=np.array([[value]],dtype=np.int32)
            end=perf_counter();n=len(tokens)-host.shape[1]
            self.stats.update({'ttft_seconds':first-begin if first else 0.,'generation_seconds':end-begin,
               'tokens_generated':n,'prefill_tokens':host.shape[1],
               'prefill_tokens_per_second':host.shape[1]/(first-begin) if first and first>begin else 0.,
               'decode_tokens_per_second':(n-1)/(end-first) if first and n>1 and end>first else 0.,
               'logical_bytes_per_token_including_prefill':(self.checkpoint.stats['logical_bytes']-logical)/n if n else 0.,
               'decode_logical_bytes_per_token':(self.checkpoint.stats['logical_bytes']-after_prefill_logical)/(n-1) if n>1 else 0.,
               'encoded_file_bytes_per_token_including_prefill':(encoded_read_bytes(self.checkpoint)-encoded_start)/n if n else 0.,
               'decode_encoded_file_bytes_per_token':(encoded_read_bytes(self.checkpoint)-after_prefill_encoded)/(n-1) if n>1 else 0.,
               'gpu_time_seconds':None,
               'kv_peak_bytes':cache.peak_bytes if cache else 0})
            physical_end=physical_read_bytes()
            self.stats['physical_read_bytes']=(physical_end-physical) if physical is not None and physical_end is not None else None
            return mx.array([tokens],dtype=mx.int32)

    def generate_speculative(self,input_ids,draft_model,*,max_new_tokens=32,
                             eos_token_id=None,max_depth=4,adaptive=True):
        """Verify local draft proposals with the actual GLM target in blocks.

        An already loaded draft model with generate(), a propose(prefix,count)
        object, or a callable returning candidate token ids is accepted. No
        checkpoint is fetched. The draft's tokenizer must match the target's.
        Stochastic sampling and MTP are not implemented by this greedy path.
        """
        from .speculative import LocalModelDraft,greedy_speculative_generate
        if draft_model is self:raise ValueError('Draft and target must be distinct local models')
        eos=self.config.get('eos_token_id') if eos_token_id is None else eos_token_id
        eos=eos_ids(eos,self.config['vocab_size'])
        if hasattr(draft_model,'generate'):
            draft_config=getattr(draft_model,'config',{})
            draft_vocab=(draft_config.get('vocab_size') if isinstance(draft_config,dict)
                         else getattr(draft_config,'vocab_size',None))
            if draft_vocab is not None and draft_vocab!=self.config['vocab_size']:
                raise ValueError('Draft and target vocabularies differ')
            other_tokenizer=getattr(draft_model,'tokenizer',None)
            if self.tokenizer is not None and other_tokenizer is not None:
                if self.tokenizer.get_vocab()!=other_tokenizer.get_vocab():
                    raise ValueError('Draft and target token mappings differ')
            propose=LocalModelDraft(draft_model,eos)
        elif hasattr(draft_model,'propose'):propose=draft_model.propose
        elif callable(draft_model):propose=draft_model
        else:raise TypeError('Supply a loaded local draft model or a candidate proposal callable')
        # A draft's transport is distinct from the target's. Unknown external
        # proposal costs stay unknown rather than masquerading as zero bytes.
        draft_checkpoint=getattr(draft_model,'checkpoint',None)
        draft_read=getattr(draft_model,'logical_bytes_read',None)
        draft_encoded=getattr(draft_model,'encoded_bytes_read',None)
        if draft_checkpoint is not None:
            draft_read=lambda:draft_checkpoint.stats['logical_bytes']
            draft_encoded=lambda:encoded_read_bytes(draft_checkpoint)
        if not callable(draft_read):draft_read=None
        if not callable(draft_encoded):draft_encoded=None
        host=token_batch(input_ids,self.config['vocab_size'])
        with self.lock:
            if self.closed:raise RuntimeError('Model is closed')
            physical=physical_read_bytes()
            def target_argmax(logits):
                with self.mx.stream(self.stream):return self.mx.argmax(logits,axis=-1).tolist()
            result=greedy_speculative_generate(self,propose,host,max_new_tokens,
                       eos_token_ids=eos,max_depth=max_depth,adaptive=adaptive,
                       argmax_tokens=target_argmax,vocab_size=self.config['vocab_size'],
                       bytes_read=lambda:self.checkpoint.stats['logical_bytes'],
                       encoded_bytes_read=lambda:encoded_read_bytes(self.checkpoint),
                       draft_bytes_read=draft_read,draft_encoded_bytes_read=draft_encoded)
            measured=result.metrics
            self.stats.update({'ttft_seconds':measured['ttft_seconds'],
                'generation_seconds':measured['generation_seconds'],
                'tokens_generated':measured['tokens_generated'],'prefill_tokens':len(host[0]),
                'prefill_tokens_per_second':len(host[0])/measured['ttft_seconds'] if measured['ttft_seconds'] else 0.,
                'decode_tokens_per_second':measured['decode_tokens_per_second'],
                'logical_bytes_per_token_including_prefill':measured['logical_bytes_per_accepted_token'],
                'decode_logical_bytes_per_token':measured['decode_logical_bytes_per_accepted_token'],
                'encoded_file_bytes_per_token_including_prefill':measured['encoded_file_bytes_per_accepted_token'],
                'decode_encoded_file_bytes_per_token':measured['decode_encoded_file_bytes_per_accepted_token'],
                'gpu_time_seconds':None,'kv_peak_bytes':result.cache.peak_bytes if result.cache else 0,
                'speculative':measured})
            end=physical_read_bytes()
            self.stats['physical_read_bytes']=end-physical if physical is not None and end is not None else None
            with self.mx.stream(self.stream):return self.mx.array([result.tokens],dtype=self.mx.int32)

    def metrics(self):
        r=self.reader
        return {'compute_device':self.device,'metal_validated':False,
           'numerics':{'compute_dtype':self.options.dtype,'dynamic_fp8_activation_kernels_reproduced':False,
               'mlx_enable_tf32_environment':os.environ.get('MLX_ENABLE_TF32','1'),
               'tf32_environment_note':'Read at MLX initialization; the current environment alone does not prove the active GPU pipeline precision',
               'full_checkpoint_quality_validated':False},'runtime':dict(self.stats),'telemetry':telemetry(),
           'io':dict(self.checkpoint.stats),
           'io_profile':self.checkpoint.io_profile() if hasattr(self.checkpoint,'io_profile') else None,
           'io_profile_coverage':'original shard offsets; packed/quantized transport has separate counters',
           'sidecar_transport':dict(r.sidecar.stats) if r.sidecar else {},
           'lossless_transport':dict(getattr(self.checkpoint,'transport_stats',{})),
           'weights':dict(r.stats),
           'cache':dict(r.cache.stats)|{'bytes':r.cache.bytes,'peak_bytes':r.cache.peak_bytes,'capacity':r.cache.capacity},
           'compact_cache':dict(r.compact_cache.stats)|{'bytes':r.compact_cache.bytes,
               'peak_bytes':r.compact_cache.peak_bytes,'capacity':r.compact_cache.capacity,
               'budget_scope':'partition of cache_bytes; not additional memory'},
           'prefetch':dict(r.io.stats)|{'peak_bytes':r.io.peak_bytes,'budget':r.io.budget,'enabled':r.prefetch_enabled},
           'spec_prefetch':r.spec.metrics() if r.spec else {'enabled':False}}

    def close(self,save_profile=True):
        with self.lock:
            if self.closed:return
            try:
                if self.reader:self.reader.close()
                if save_profile and self.routes and hasattr(self,'route_path'):
                    atomic_json(self.route_path,{k:dict(v) for k,v in self.routes.items()})
            finally:
                if self.checkpoint:self.checkpoint.close()
                self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
