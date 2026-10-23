"""Independent official Transformers CPU execution of a streamed Qwen35 GGUF.

Constructs the genuine Transformers model on the meta device: no embedding,
output, or 3-D expert bank is allocated. Only disk-backed Linear/Embedding
accessors and indexable expert-bank storage replace weight transport. The
official attention, M-RoPE, convolution, DeltaNet, router, expert forward,
normalization, residual and cache implementations execute unchanged in Torch.

GGUF codecs are shared with the native reader (separately scalar/C validated),
so this checks independent *model equations*, not independent decompression.
Run real checkpoints only inside the experiment supervisor. This benchmark
does not make Torch a production AirLLM backend.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextRotaryEmbedding)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.gguf import GGUFReader
from airllm.nextgen.experiment_guard import validate_output_paths


@dataclass(frozen=True)
class ReferenceBudget:
    io_bytes: int=32<<20
    resident_bytes: int=128<<20
    workspace_bytes: int=128<<20
    kv_bytes: int=256<<20
    reserve_bytes: int=4<<30
    row_chunk: int=256
    max_context: int=512

    def __post_init__(self):
        for name in ('io_bytes','resident_bytes','workspace_bytes','kv_bytes','row_chunk','max_context'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(name+' must be positive')
        if type(self.reserve_bytes) is not int or self.reserve_bytes<0:raise ValueError('reserve_bytes must be nonnegative')
        if self.total_bytes>1<<30:raise MemoryError('Reference total configured allocation budget exceeds 1 GiB')

    @property
    def total_bytes(self):return self.io_bytes+self.resident_bytes+self.workspace_bytes+self.kv_bytes


def official_config(reader):
    """Translate metadata directly to the official HF config, independently."""
    m=reader.metadata;p='qwen35moe.'
    if m.get('general.architecture')!='qwen35moe':raise NotImplementedError('Requires qwen35moe GGUF')
    def integer(key):
        value=m.get(p+key)
        if type(value) is not int or value<1:raise ValueError('Invalid metadata dimension '+key)
        return value
    layers=integer('block_count');counts=m.get(p+'attention.head_count_kv')
    if isinstance(counts,list):
        if len(counts)!=layers or any(type(v) is not int or v<0 for v in counts):raise ValueError('Invalid KV head pattern')
        full={v for v in counts if v}
        if len(full)!=1:raise NotImplementedError('Nonuniform KV heads')
        kv=next(iter(full));kinds=['full_attention' if n else 'linear_attention' for n in counts]
    else:
        kv=integer('attention.head_count_kv');interval=integer('full_attention_interval')
        kinds=['full_attention' if (i+1)%interval==0 else 'linear_attention' for i in range(layers)]
    hd=integer('attention.key_length');rot=integer('rope.dimension_count');vh=integer('ssm.time_step_rank');inner=integer('ssm.inner_size')
    if inner%vh or rot>hd or rot%2:raise ValueError('Invalid head/RoPE dimensions')
    if m.get(p+'rope.scaling.type','none') not in ('none','default'):raise NotImplementedError('Scaled RoPE')
    sections=m.get(p+'rope.dimension_sections',m.get(p+'rope.mrope_section'))
    rope={'rope_type':'default','rope_theta':m.get(p+'rope.freq_base',10000000.),'partial_rotary_factor':rot/hd}
    if sections is not None:rope['mrope_section']=sections
    c=Qwen3_5MoeTextConfig(hidden_size=integer('embedding_length'),vocab_size=reader.tensors['token_embd.weight'].shape[0],
        num_hidden_layers=layers,num_attention_heads=integer('attention.head_count'),num_key_value_heads=kv,head_dim=hd,
        linear_num_key_heads=integer('ssm.group_count'),linear_num_value_heads=vh,
        linear_key_head_dim=integer('ssm.state_size'),linear_value_head_dim=inner//vh,
        linear_conv_kernel_dim=integer('ssm.conv_kernel'),num_experts=integer('expert_count'),
        num_experts_per_tok=integer('expert_used_count'),moe_intermediate_size=integer('expert_feed_forward_length'),
        shared_expert_intermediate_size=integer('expert_shared_feed_forward_length'),
        max_position_embeddings=integer('context_length'),layer_types=kinds,rope_parameters=rope,
        rms_norm_eps=m.get(p+'attention.layer_norm_rms_epsilon',1e-6),
        tie_word_embeddings='output.weight' not in reader.tensors,pad_token_id=None,
        bos_token_id=m.get('tokenizer.ggml.bos_token_id'),eos_token_id=m.get('tokenizer.ggml.eos_token_id'),dtype='float32')
    c._attn_implementation='eager';c._experts_implementation='eager'
    return c


_SUFFIX={
    'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'post_attention_norm.weight',
    'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight',
    'self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight',
    'self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight',
    'linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight',
    'linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight',
    'linear_attn.out_proj.weight':'ssm_out.weight','linear_attn.conv1d.weight':'ssm_conv1d.weight',
    'linear_attn.norm.weight':'ssm_norm.weight','linear_attn.A_log':'ssm_a','linear_attn.dt_bias':'ssm_dt',
    'mlp.gate.weight':'ffn_gate_inp.weight','mlp.shared_expert_gate.weight':'ffn_gate_inp_shexp.weight',
    'mlp.shared_expert.gate_proj.weight':'ffn_gate_shexp.weight','mlp.shared_expert.up_proj.weight':'ffn_up_shexp.weight',
    'mlp.shared_expert.down_proj.weight':'ffn_down_shexp.weight'}


class TorchGGUFSource:
    """Torch-only transport; no native MLX runtime or equations are imported."""
    def __init__(self,reader,config,budget):
        self.reader=reader;self.config=config;self.budget=budget;self.resident=0
        self.stats={'row_reads':0,'expert_fetches':0,'peak_workspace_estimate_bytes':0}
        k=config.linear_num_key_heads;v=config.linear_num_value_heads;r=v//k;d=config.linear_value_head_dim
        if v%k:raise ValueError('Invalid grouped V heads')
        # Build the conversion order explicitly. The inverse is not generally
        # the same permutation (the real checkpoint has 16 K and 32 V heads).
        heads=[group*r+slot for slot in range(r) for group in range(k)]
        rows=[head*d+lane for head in heads for lane in range(d)]
        self.to_tiled=torch.tensor(rows);self.to_grouped=torch.argsort(self.to_tiled)
        self.head_to_grouped=torch.argsort(torch.tensor(heads))
        self.reordered=reader.metadata.get('qwen35moe.ssm.v_head_reordered',False)

    def native_name(self,name):
        global_names={'model.embed_tokens.weight':'token_embd.weight','model.norm.weight':'output_norm.weight',
                      'lm_head.weight':'output.weight' if 'output.weight' in self.reader.tensors else 'token_embd.weight'}
        if name in global_names:return global_names[name]
        match=re.fullmatch(r'model\.layers\.(\d+)\.(.*)',name)
        if not match or match[2] not in _SUFFIX:raise KeyError('Unmapped HF reference weight '+name)
        return f'blk.{match[1]}.'+_SUFFIX[match[2]]

    def guard(self,workspace):
        if workspace>self.budget.workspace_bytes:raise MemoryError('Torch reference workspace budget exceeded before allocation')
        if psutil.virtual_memory().available<self.budget.reserve_bytes+workspace:raise MemoryError('Torch reference system reserve reached')
        self.stats['peak_workspace_estimate_bytes']=max(self.stats['peak_workspace_estimate_bytes'],workspace)

    def rows(self,name,start,count,expert=None):
        self.stats['row_reads']+=1
        return self.reader.read_rows(name,start,count,expert=expert)

    def materialize_parameter(self,name,shape):
        native=self.native_name(name);info=self.reader.tensors[native];n=math.prod(shape)*4
        if self.resident+n>self.budget.resident_bytes:raise MemoryError('Resident reference parameters exceed budget')
        self.guard(n*4)
        array=self.rows(native,0,info.rows)
        value=torch.from_numpy(array).reshape(info.shape)
        if name.endswith('linear_attn.conv1d.weight'):
            if self.reordered:
                kd=2*self.config.linear_num_key_heads*self.config.linear_key_head_dim
                value=torch.cat([value[:kd],value[kd:].index_select(0,self.to_grouped)])
            value=value[:,None,:]
        elif name.endswith(('linear_attn.A_log','linear_attn.dt_bias')):
            if name.endswith('A_log'):
                if torch.any(value>=0):raise ValueError('GGUF ssm_a must be negative')
                value=torch.log(-value)
            if self.reordered:value=value.index_select(0,self.head_to_grouped)
        elif name=='model.norm.weight' or name.endswith(('input_layernorm.weight','post_attention_layernorm.weight','self_attn.q_norm.weight','self_attn.k_norm.weight')):
            value=value-1
        if tuple(value.shape)!=tuple(shape):raise ValueError(f'Invalid reference shape {name}: {value.shape} != {shape}')
        result=value.clone();self.resident+=n
        return nn.Parameter(result,requires_grad=False)

    def linear(self,x,name):
        native=self.native_name(name);info=self.reader.tensors[native]
        rows,width=info.shape
        if x.shape[-1]!=width:raise ValueError('Reference linear input dimension mismatch')
        if self.reordered and name.endswith('linear_attn.out_proj.weight'):x=x.index_select(-1,self.to_tiled)
        count=x.numel()//width;out_bytes=count*rows*4
        tile=min(self.budget.row_chunk,self.reader.max_rows(native),
                 (self.budget.workspace_bytes-out_bytes-x.numel()*4)//max(1,width*8+count*4))
        if tile<1:raise MemoryError('No Torch reference row tile fits')
        self.guard(out_bytes+tile*width*8+x.numel()*4)
        result=torch.empty((*x.shape[:-1],rows),dtype=x.dtype,device='cpu')
        for start in range(0,rows,tile):
            array=self.rows(native,start,min(tile,rows-start));weight=torch.from_numpy(array)
            result[...,start:start+len(array)]=F.linear(x,weight)
            del weight,array
        if self.reordered:
            if name.endswith('linear_attn.in_proj_qkv.weight'):
                kd=2*self.config.linear_num_key_heads*self.config.linear_key_head_dim
                result=torch.cat([result[...,:kd],result[...,kd:].index_select(-1,self.to_grouped)],dim=-1)
            elif name.endswith('linear_attn.in_proj_z.weight'):result=result.index_select(-1,self.to_grouped)
            elif name.endswith(('linear_attn.in_proj_a.weight','linear_attn.in_proj_b.weight')):result=result.index_select(-1,self.head_to_grouped)
        return result

    def expert(self,layer,expert,gate_up):
        c=self.config;m=c.moe_intermediate_size;h=c.hidden_size
        self.guard((2 if gate_up else 1)*m*h*4*4)
        self.stats['expert_fetches']+=1
        if gate_up:
            result=torch.empty((2*m,h),dtype=torch.float32)
            for name,start in (('gate',0),('up',m)):
                array=self.rows(f'blk.{layer}.ffn_{name}_exps.weight',0,m,expert)
                result[start:start+m].copy_(torch.from_numpy(array));del array
            return result
        array=self.rows(f'blk.{layer}.ffn_down_exps.weight',0,h,expert)
        # The CPU tensor owns the NumPy result until F.linear finishes, so the
        # reader's live-result reservation remains effective during the dot.
        return torch.from_numpy(array)


class StreamedLinear(nn.Module):
    def __init__(self,source,name):super().__init__();self.source=source;self.name=name
    def forward(self,x):return self.source.linear(x,self.name)


class StreamedEmbedding(nn.Module):
    def __init__(self,source):super().__init__();self.source=source
    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('Reference accepts batch one only')
        h=self.source.config.hidden_size;self.source.guard(ids.numel()*h*8)
        result=torch.empty((*ids.shape,h),dtype=torch.float32)
        for j,token in enumerate(ids[0].tolist()):
            array=self.source.rows('token_embd.weight',token,1)
            result[0,j].copy_(torch.from_numpy(array[0]));del array
        return result


class LazyExpertBank:
    """Storage-only replacement: official Qwen3_5MoeExperts.forward is retained."""
    def __init__(self,source,layer,gate_up):self.source=source;self.layer=layer;self.gate_up=gate_up
    def __getitem__(self,expert):return self.source.expert(self.layer,int(expert),self.gate_up)


def cache_bytes(cache):
    seen=set()
    def count(value):
        if isinstance(value,torch.Tensor):
            ptr=value.untyped_storage().data_ptr()
            if ptr in seen:return 0
            seen.add(ptr);return value.untyped_storage().nbytes()
        if isinstance(value,(list,tuple)):return sum(count(v) for v in value)
        if isinstance(value,dict):return sum(count(v) for v in value.values())
        return 0
    return sum(count(vars(layer)) for layer in cache.layers) if cache is not None else 0


class TorchGGUFReference:
    def __init__(self,path,budget=None):
        self.budget=budget or ReferenceBudget();self.reader=None;self.closed=False;begin=perf_counter()
        try:
            self.reader=GGUFReader(path,memory_budget_bytes=self.budget.io_bytes,include_tokenizer_metadata=False)
            self.config=official_config(self.reader);self.source=TorchGGUFSource(self.reader,self.config,self.budget)
            with torch.device('meta'):self.model=Qwen3_5MoeForCausalLM(self.config)
            self.initial_meta_parameters=sum(p.numel() for p in self.model.parameters())
            def replace_transport(module,prefix=''):
                for name,child in list(module.named_children()):
                    full=f'{prefix}.{name}' if prefix else name
                    if isinstance(child,Qwen3_5MoeExperts):
                        layer=int(full.split('.')[2])
                        del child._parameters['gate_up_proj'];del child._parameters['down_proj']
                        child.gate_up_proj=LazyExpertBank(self.source,layer,True)
                        child.down_proj=LazyExpertBank(self.source,layer,False)
                    elif isinstance(child,nn.Linear):setattr(module,name,StreamedLinear(self.source,full+'.weight'))
                    elif isinstance(child,nn.Embedding):setattr(module,name,StreamedEmbedding(self.source))
                    else:replace_transport(child,full)
            replace_transport(self.model)
            self.model.model.rotary_emb=Qwen3_5MoeTextRotaryEmbedding(self.config)
            for name,param in list(self.model.named_parameters()):
                parent,leaf=name.rsplit('.',1);module=self.model.get_submodule(parent)
                setattr(module,leaf,self.source.materialize_parameter(name,param.shape))
            if any(p.is_meta for p in self.model.parameters()) or any(b.is_meta for b in self.model.buffers()):
                raise RuntimeError('Unmaterialized reference parameter/buffer remains')
            self.model.eval();self.position=0;self.peak_cache_bytes=0
            self.startup_seconds=perf_counter()-begin
        except BaseException:self.close();raise

    def forward(self,ids,past_key_values=None,*,logits_to_keep=1):
        if self.closed:raise RuntimeError('Reference is closed')
        ids=torch.as_tensor(ids,dtype=torch.long,device='cpu')
        if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1]:raise ValueError('Reference accepts nonempty batch one')
        if torch.any(ids<0) or torch.any(ids>=self.config.vocab_size):raise ValueError('Token ID outside vocabulary')
        position=past_key_values.get_seq_length() if past_key_values is not None else 0
        total=position+ids.shape[1]
        if total>min(self.budget.max_context,self.config.max_position_embeddings):raise MemoryError('Reference context limit')
        c=self.config;nlinear=c.layer_types.count('linear_attention');nfull=len(c.layer_types)-nlinear
        estimate=nlinear*(c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim+
            (2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim)*c.linear_conv_kernel_dim)*4
        estimate+=nfull*2*c.num_key_value_heads*total*c.head_dim*4
        if estimate*2>self.budget.kv_bytes:raise MemoryError('Reference cache transient budget exceeded before forward')
        with torch.inference_mode():out=self.model(ids,past_key_values=past_key_values,use_cache=True,logits_to_keep=logits_to_keep)
        current=cache_bytes(out.past_key_values);self.peak_cache_bytes=max(self.peak_cache_bytes,current)
        if current>self.budget.kv_bytes:raise MemoryError('Official cache exceeded estimated payload')
        return out

    def metrics(self):
        source=Path(inspect.getfile(Qwen3_5MoeForCausalLM))
        return {'reference':'official_transformers_torch_cpu','torch':torch.__version__,'transformers':transformers.__version__,
            'official_expert_forward_unchanged':True,'official_source':str(source),
            'official_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'shared_gguf_codec':True,'initial_meta_parameter_count':self.initial_meta_parameters,
            'resident_parameter_bytes':self.source.resident,'configured_total_bytes':self.budget.total_bytes,
            'startup_seconds':self.startup_seconds,'peak_cache_payload_bytes':self.peak_cache_bytes,
            'source':dict(self.source.stats),'io':dict(self.reader.stats)}

    def close(self):
        if self.closed:return
        self.closed=True
        if self.reader:self.reader.close()
        if hasattr(self,'model'):del self.model

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-file',type=Path,required=True);p.add_argument('--input-ids',required=True,help='Comma-separated token IDs, or a JSON array')
    p.add_argument('--max-new-tokens',type=int,default=1);p.add_argument('--output',type=Path,required=True);p.add_argument('--logits-output',type=Path)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--row-chunk',type=int,default=256)
    p.add_argument('--max-context',type=int,default=512);p.add_argument('--io-mib',type=int,default=32)
    p.add_argument('--resident-mib',type=int,default=128);p.add_argument('--workspace-mib',type=int,default=128)
    p.add_argument('--kv-mib',type=int,default=256);p.add_argument('--reserve-mib',type=int,default=4096)
    args=p.parse_args(argv)
    # np.savez appends .npz when absent; validate the actual destination before
    # opening a checkpoint or starting any model/weight allocation.
    logits_path=(args.logits_output if str(args.logits_output).endswith('.npz') else Path(str(args.logits_output)+'.npz')) if args.logits_output else None
    validate_output_paths(args.output,protected=[args.model_file],extra_outputs=[logits_path] if logits_path else [])
    if args.threads<1 or not 1<=args.max_new_tokens<=4:raise ValueError('Reference requires threads >=1 and 1..4 generated tokens')
    ids=json.loads(args.input_ids) if args.input_ids.lstrip().startswith('[') else list(map(int,args.input_ids.split(',')))
    if ids and isinstance(ids[0],list):
        if len(ids)!=1:raise ValueError('Batch one required')
        ids=ids[0]
    if not ids or any(type(v) is not int for v in ids):raise ValueError('Integer input IDs required')
    torch.set_num_threads(args.threads)
    budget=ReferenceBudget(args.io_mib<<20,args.resident_mib<<20,args.workspace_mib<<20,args.kv_mib<<20,
                           args.reserve_mib<<20,args.row_chunk,args.max_context)
    records=[];logits=[];tokens=list(ids);cache=None;current=[ids];started=perf_counter()
    with TorchGGUFReference(args.model_file,budget) as reference:
        for step in range(args.max_new_tokens):
            begin=perf_counter();out=reference.forward(current,cache);cache=out.past_key_values
            row=out.logits[0,-1].detach().numpy().copy();token=int(row.argmax());tokens.append(token);logits.append(row)
            records.append({'step':step,'token_id':token,'forward_seconds':perf_counter()-begin})
            current=[[token]]
            print(json.dumps({'reference_step':step,'token_id':token,'forward_seconds':records[-1]['forward_seconds']}),flush=True)
        result={'model_file':str(args.model_file.resolve()),'input_ids':ids,'generated_ids':tokens[len(ids):],
            'all_ids':tokens,'records':records,'elapsed_seconds':perf_counter()-started,'metrics':reference.metrics(),
            'limitation':'Independent official Torch equations; GGUF decompression is shared with native reader.'}
    output=args.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    if logits_path:
        logits_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(logits_path,logits=np.stack(logits),generated_ids=np.array(tokens[len(ids):],dtype=np.int64),input_ids=np.array(ids,dtype=np.int64))
    print(json.dumps({'output':str(output),'generated_ids':result['generated_ids']}),flush=True)


if __name__=='__main__':main()
