#!/usr/bin/env python3
"""Real >RAM trained Llama generation with independent streamed Transformers layers.

No implicit downloads. PyTorch reference materializes at most one decoder layer;
original safetensors handles map shards, and layers are released after evaluation.
A successful run is evidence only for this checkpoint/prompt/precision/context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
from time import perf_counter


class StreamedTransformersLlama:
    """Independent HF numerical implementation, one CPU decoder block at a time."""
    def __init__(self, root):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        from transformers.cache_utils import DynamicCache
        from safetensors import safe_open
        self.torch=torch;self.safe_open=safe_open;self.root=Path(root)
        self.config=LlamaConfig.from_dict(json.loads((self.root/'config.json').read_text()))
        self.config._attn_implementation='eager'
        index=self.root/'model.safetensors.index.json'
        if index.exists():self.weight_map=json.loads(index.read_text())['weight_map']
        else:
            with safe_open(self.root/'model.safetensors',framework='pt',device='cpu') as f:
                self.weight_map={key:'model.safetensors' for key in f.keys()}
        self.rope=LlamaRotaryEmbedding(config=self.config).cpu()
        self.cache=DynamicCache(config=self.config);self.position=0

    def tensor(self,key,rows=None):
        torch=self.torch
        with self.safe_open(self.root/self.weight_map[key],framework='pt',device='cpu') as f:
            value=f.get_slice(key)[rows[0]:rows[1]] if rows is not None else f.get_tensor(key)
            # copy=True makes independence of the soon-to-be-closed shard map
            # explicit even for a checkpoint already serialized in float32.
            return value.to(dtype=torch.float32,copy=True)

    def forward(self,values):
        import torch
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm
        values=torch.as_tensor(values,dtype=torch.long)
        if values.ndim==1:values=values.unsqueeze(0)
        if values.ndim!=2 or values.shape[0]!=1:raise ValueError('One unpadded sequence required')
        with torch.inference_mode():
            # Read only actual embedding rows, preserving original input order.
            rows=[self.tensor('model.embed_tokens.weight',(int(token),int(token)+1)) for token in values[0]]
            hidden=torch.cat(rows,dim=0).unsqueeze(0);del rows
            positions=torch.arange(self.position,self.position+values.shape[1]).unsqueeze(0)
            total=self.position+values.shape[1]
            allowed=torch.arange(total)[None,:]<=positions[0,:,None]
            mask=torch.where(allowed,0.,torch.finfo(torch.float32).min)[None,None,:,:]
            rotary=self.rope(hidden,positions)
            for i in range(self.config.num_hidden_layers):
                prefix=f'model.layers.{i}.'
                with torch.device('meta'):layer=LlamaDecoderLayer(self.config,i)
                state={key[len(prefix):]:self.tensor(key) for key in self.weight_map if key.startswith(prefix)}
                layer.load_state_dict(state,strict=True,assign=True);layer.eval();del state
                hidden=layer(hidden,attention_mask=mask,position_ids=positions,
                             past_key_values=self.cache,use_cache=True,position_embeddings=rotary)
                del layer
            norm=LlamaRMSNorm(self.config.hidden_size,eps=self.config.rms_norm_eps)
            norm.load_state_dict({'weight':self.tensor('model.norm.weight')},assign=True)
            last=norm(hidden[:,-1:]);del norm,hidden
            key='model.embed_tokens.weight' if self.config.tie_word_embeddings else 'lm_head.weight'
            parts=[]
            for start in range(0,self.config.vocab_size,2048):
                weight=self.tensor(key,(start,min(self.config.vocab_size,start+2048)))
                parts.append(last@weight.T);del weight
            self.position+=values.shape[1]
            return torch.cat(parts,dim=-1)


def cgroup_snapshot():
    root=Path('/sys/fs/cgroup');result={}
    for name in ('memory.max','memory.current','memory.peak','memory.events','memory.stat'):
        path=root/name
        if path.exists():
            try:result[name]=path.read_text().strip()
            except OSError:result[name]=None
    return result


def worker(args):
    sys.path.insert(0,str(Path(args.source).resolve()/'air_llm'))
    import numpy as np,psutil
    source_root=Path(args.source).resolve()
    source_sha256={str(path.relative_to(source_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((source_root/'air_llm/airllm/nextgen').glob('*.py'))}
    import torch
    from transformers import AutoTokenizer
    from airllm.nextgen.storage import physical_read_bytes
    torch.set_num_threads(1)
    root=Path(args.model_dir).resolve()
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    tokens=tok.encode(args.prompt);prefix=list(tokens);logits=[];runs=[];current=[tokens]
    before_physical=physical_read_bytes();swap_start=psutil.swap_memory().used;cgroup_before=cgroup_snapshot();begin=perf_counter()
    if args.worker=='reference':
        model=StreamedTransformersLlama(root);cfg=None
    else:
        from airllm.nextgen import NextConfig,AirLLMNextLlamaModel
        cfg=NextConfig(device='cpu',dtype='float32',cache_bytes=256*1024**2,
            io_bytes=256*1024**2,workspace_bytes=512*1024**2,kv_bytes=128*1024**2,
            reserve_bytes=512*1024**2,max_tensor_bytes=256*1024**2,row_chunk=2048,
            prefill_chunk=16,max_context=256,load_tokenizer=False,prefetch=False,
            cache_dir=args.metadata_dir,use_saved_tuning=False)
        model=AirLLMNextLlamaModel(root,next_config=cfg)
    startup=perf_counter()-begin;cache=None;generation_start=perf_counter()
    try:
        for i in range(args.tokens):
            start=perf_counter();physical=physical_read_bytes()
            if args.worker=='reference':
                value=model.forward(current);a=value.detach().cpu().numpy();selected=int(value[0,-1].argmax().item())
            else:
                value=model.forward(np.asarray(current,dtype=np.int32),past_key_values=cache,logits_to_keep=1)
                cache=value.past_key_values;a=np.asarray(value.logits);selected=int(np.argmax(a[0,-1]))
            elapsed=perf_counter()-start;logits.append(a);tokens.append(selected);current=[[selected]]
            physical_after=physical_read_bytes()
            runs.append({'step':i,'seconds':elapsed,'selected_token':selected,
                'physical_read_bytes':physical_after-physical if physical is not None and physical_after is not None else None})
            print(args.worker,'step',i,'token',selected,'seconds',round(elapsed,3),flush=True)
        elapsed=perf_counter()-generation_start
        end_physical=physical_read_bytes()
        metadata={'variant':args.worker,'source':str(source_root),'source_files_sha256':source_sha256,'model_dir':str(root),
            'prompt':args.prompt,'input_tokens':prefix,'token_ids':tokens,'new_tokens':args.tokens,
            'output_text':tok.decode(tokens[len(prefix):],skip_special_tokens=True),'startup_seconds':startup,
            'generation_seconds':elapsed,'ttft_seconds':runs[0]['seconds'],'steps':runs,
            'decode_tokens_per_second':(args.tokens-1)/sum(r['seconds'] for r in runs[1:]) if args.tokens>1 else None,
            'physical_read_bytes':end_physical-before_physical if end_physical is not None and before_physical is not None else None,
            'system_swap_start_bytes':swap_start,'system_swap_end_bytes':psutil.swap_memory().used,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'host_memory_bytes':psutil.virtual_memory().total,'cgroup_before':cgroup_before,'cgroup_after':cgroup_snapshot(),
            'cgroup_scope':'Shared experiment cgroup, including filesystem cache and sibling processes; not process RSS',
            'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
            'config':cfg.as_dict() if cfg else None,'metrics':model.metrics() if hasattr(model,'metrics') else None,
            'precision':'float32 compute from unchanged original floating checkpoint','mlx_enable_tf32':os.getenv('MLX_ENABLE_TF32'),
            'os_cache_state':'Uncontrolled. Real checkpoint exceeds physical host RAM and cgroup capacity; no cache purge used.',
            'external_ssd_measured':False,'metal_measured':False,'platform':platform.platform()}
        output=Path(args.output);np.savez_compressed(output.with_suffix('.npz'),logits=np.concatenate(logits,axis=1))
        output.write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        if hasattr(model,'close'):model.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True);parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--candidate-source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--prompt',default='def add(a, b):\n    ');parser.add_argument('--tokens',type=int,default=4)
    parser.add_argument('--variants',nargs='+',default=['baseline','candidate','reference'],choices=['baseline','candidate','reference'])
    parser.add_argument('--worker',choices=['baseline','candidate','reference'],help=argparse.SUPPRESS)
    parser.add_argument('--source',help=argparse.SUPPRESS);parser.add_argument('--metadata-dir',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.tokens<1:parser.error('tokens must be positive')
    if args.worker:return worker(args)
    root=Path(args.model_dir).resolve();files=sorted(root.glob('*.safetensors'));payload=sum(p.stat().st_size for p in files)
    if not files:raise FileNotFoundError('Local safetensors checkpoint required')
    import psutil,numpy as np
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True);reports={};paths={}
    for variant in args.variants:
        source=args.baseline_source if variant=='baseline' else args.candidate_source
        path=output.with_name(output.stem+'-'+variant+'.json');print('running',variant,flush=True)
        with tempfile.TemporaryDirectory(prefix='airllm-v5-large-') as meta:
            command=[sys.executable,str(Path(__file__).resolve()),'--model-dir',str(root),
                '--baseline-source',args.baseline_source,'--candidate-source',args.candidate_source,
                '--worker',variant,'--source',source,'--metadata-dir',meta,'--output',str(path),
                '--tokens',str(args.tokens),'--prompt',args.prompt]
            with path.with_suffix('.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True,env=dict(os.environ,MLX_ENABLE_TF32='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            if result.returncode:raise RuntimeError(f'{variant} failed with {result.returncode}; see {path.with_suffix(".log")}')
        reports[variant]=json.loads(path.read_text());paths[variant]=path
    comparisons={}
    reference_label='reference' if 'reference' in reports else args.variants[0]
    expected=np.load(paths[reference_label].with_suffix('.npz'))['logits'];reference=reports[reference_label]
    for variant,report in reports.items():
        actual=np.load(paths[variant].with_suffix('.npz'))['logits']
        comparisons[variant]={'tokens_match_comparison_reference':report['token_ids']==reference['token_ids'],
            'max_abs_logit_difference':float(np.max(np.abs(actual-expected))),
            'logits_within_fixed_atol_1e4_rtol_5e4':bool(np.allclose(actual,expected,atol=1e-4,rtol=5e-4)),
            'generation_seconds':report['generation_seconds'],'decode_tokens_per_second':report['decode_tokens_per_second'],
            'process_peak_rss_bytes':report['process_peak_rss_bytes'],'physical_read_bytes':report['physical_read_bytes'],
            'raw_report':paths[variant].name}
    metadata={'model_id':'codellama/CodeLlama-7b-hf','revision':'6c284d1468fe6c413cf56183e69b194dcfa27fe6',
        'checkpoint_files_bytes':payload,'host_memory_bytes':psutil.virtual_memory().total,
        'checkpoint_exceeds_host_ram':payload>psutil.virtual_memory().total,
        'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
        'tokens_per_variant':args.tokens,'comparison_reference':reference_label,'comparisons':comparisons,
        'tolerances_fixed_before_measurement':{'atol':1e-4,'rtol':5e-4},'no_metadata_or_weight_conversion':True,
        'metal_measured':False,'external_ssd_measured':False,'full_GLM53_generated':False,'full_Qwen38_generated':False}
    output.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
    if not all(v['tokens_match_comparison_reference'] and v['logits_within_fixed_atol_1e4_rtol_5e4'] for v in comparisons.values()):
        raise SystemExit('Large-checkpoint parity criterion failed; raw evidence retained')


if __name__=='__main__':main()
