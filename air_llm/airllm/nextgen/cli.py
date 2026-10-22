"""Local-only AirLLM Next command line. Heavy model downloads are never implicit."""
import argparse
from dataclasses import replace
from contextlib import ExitStack
import json
import platform
import sys
from pathlib import Path
from .config import NextConfig
from .tuning import auto_config,telemetry,measure_storage
from .storage import OffsetCheckpoint
from .limits import estimate_glm


def config_for(args):
    cfg=auto_config(args.profile,device=args.device,load_tokenizer=not getattr(args,'token_ids',None))
    values={}
    for field in ('cache_bytes','io_bytes','kv_bytes'):
        mib=getattr(args,field.replace('_bytes','_mib'),None)
        if mib is not None:values[field]=int(mib*1024**2)
    if getattr(args,'float32',False):values['dtype']='float32'
    if getattr(args,'quantized_dir',None):values['quantized_dir']=args.quantized_dir
    if getattr(args,'no_prefetch',False):values['prefetch']=False
    if getattr(args,'max_context',None):values['max_context']=args.max_context
    if getattr(args,'cache_dir',None):values['cache_dir']=args.cache_dir
    if getattr(args,'cache_policy',None):
        values['policy']=args.cache_policy;values['use_saved_tuning']=False
    if getattr(args,'coalesce_reads',False):values['coalesce_reads']=True
    if getattr(args,'no_selective_fp8_scales',False):values['selective_fp8_scales']=False
    return replace(cfg,**values)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='airllm-next',description='Native local MLX GLM DSA inference; Metal hardware validation required on Mac')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Check local environment without model weights')
    inspect=sub.add_parser('inspect',help='Offline architecture, quantization and storage feasibility (no inference)')
    inspect.add_argument('model');inspect.add_argument('--tokens',type=int,default=4096)
    inspect.add_argument('--storage-budget-bytes',type=int,default=2_000_000_000_000)
    for name in ('prepare','tune','calibrate','limits','convert','verify','run','serve'):
        p=sub.add_parser(name);p.add_argument('model',help='Local safetensors checkpoint directory')
        p.add_argument('--cache-dir');p.add_argument('--profile',choices=['balanced','performance'],default='balanced')
        p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
        if name in ('run','calibrate','serve'):
            p.add_argument('--prompt',default='Bonjour');p.add_argument('--token-ids',help='Comma-separated ids (no tokenizer required)')
            p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--max-context',type=int,default=4096)
            p.add_argument('--cache-mib',type=int);p.add_argument('--io-mib',type=int);p.add_argument('--kv-mib',type=int)
            p.add_argument('--quantized-dir');p.add_argument('--no-prefetch',action='store_true');p.add_argument('--float32',action='store_true')
            p.add_argument('--json-report')
            p.add_argument('--cache-policy',choices=['lru','slru','lfu','cost'])
            p.add_argument('--coalesce-reads',action='store_true',help='Opt in to workload-dependent read coalescing')
            p.add_argument('--no-selective-fp8-scales',action='store_true')
        if name in ('run','serve'):
            p.add_argument('--backend',choices=['glm','qwen-experimental'],default='glm')
        if name=='run':
            p.add_argument('--draft-model',help='Explicit compatible local GLM draft checkpoint; no download')
            p.add_argument('--speculative-depth',type=int,default=4)
            p.add_argument('--no-adaptive-speculation',action='store_true')
        if name=='serve':
            p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
            p.add_argument('--model-name',default='airllm-next')
        if name=='verify':
            p.add_argument('--quantized-dir',required=True,help='Explicit full content scan of this sidecar')
        if name=='convert':
            p.add_argument('--output',required=True);p.add_argument('--bits',type=int,choices=[4,8],default=4)
            p.add_argument('--group-size',type=int,choices=[32,64,128],default=64)
            p.add_argument('--max-matrices',type=int,help='Bounded conversion/checkpointing')
            p.add_argument('--scope',choices=['experts','mixed'],default='experts')
            p.add_argument('--linear-bits',type=int,choices=[4,8],default=8)
    args=parser.parse_args(argv)
    if args.command=='doctor':
        import mlx.core as mx
        print(json.dumps({'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
           'mlx':mx.__version__,'metal_available':mx.metal.is_available(),'telemetry':telemetry(),
           'validation':'Metal and the full GLM-5.3 checkpoint have not been validated in the delivered Linux build'},indent=2))
        return
    if args.command=='inspect':
        from .adapters import inspect_local_checkpoint
        print(json.dumps(inspect_local_checkpoint(args.model,tokens=args.tokens,storage_budget_bytes=args.storage_budget_bytes),indent=2));return
    cfg=config_for(args)
    if args.command in ('run','calibrate','serve'):
        from .runtime import AirLLMNextModel
        model_class=AirLLMNextModel
        if getattr(args,'backend','glm')=='qwen-experimental':
            if getattr(args,'draft_model',None):parser.error('Qwen speculative verification is not enabled by this experimental backend')
            from .qwen_runtime import ExperimentalQwenModel
            model_class=ExperimentalQwenModel
        with ExitStack() as stack:
            model=stack.enter_context(model_class(args.model,next_config=cfg))
            if args.command=='serve':
                from .server import serve
                serve(model,model_name=args.model_name,host=args.host,port=args.port,max_new_tokens=args.max_new_tokens);return
            draft=None
            if getattr(args,'draft_model',None):
                draft_cfg=replace(cfg,cache_bytes=min(cfg.cache_bytes,256*1024**2),
                    io_bytes=min(cfg.io_bytes,128*1024**2),kv_bytes=min(cfg.kv_bytes,128*1024**2),
                    workspace_bytes=min(cfg.workspace_bytes,256*1024**2),load_tokenizer=cfg.load_tokenizer)
                draft=stack.enter_context(AirLLMNextModel(args.draft_model,next_config=draft_cfg))
            if args.token_ids:ids=[[int(x) for x in args.token_ids.split(',')]]
            else:ids=[model.tokenizer.encode(args.prompt)]
            if args.command=='calibrate':
                from .tuning import calibrate_prefetch
                print(json.dumps(calibrate_prefetch(model,ids,tokens=args.max_new_tokens),indent=2));return
            generation_options={}
            if model_class is AirLLMNextModel:
                generation_options={'draft_model':draft,'speculative_depth':getattr(args,'speculative_depth',4),
                    'speculative_adaptive':not getattr(args,'no_adaptive_speculation',False)}
            tokens=model.generate(ids,max_new_tokens=args.max_new_tokens,**generation_options)
            values=tokens.tolist()[0]
            print(model.tokenizer.decode(values[len(ids[0]):]) if model.tokenizer else values)
            metrics=model.metrics();print(json.dumps(metrics,indent=2),file=sys.stderr)
            if args.json_report:Path(args.json_report).write_text(json.dumps(metrics,indent=2))
        return
    with OffsetCheckpoint(args.model,cfg.metadata_dir(),cfg.max_open_files,cfg.max_tensor_bytes) as cp:
        if args.command=='prepare':result={'prepared':True,'generation_ready':False,'fingerprint':cp.fingerprint,'stats':dict(cp.stats),'tensor_count':len(cp.tensors),'checkpoint_bytes':cp.total_bytes}
        elif args.command=='tune':result=measure_storage(cp,cfg)
        elif args.command=='verify':
            from .sidecar import QuantizedSidecar
            sidecar=QuantizedSidecar(args.quantized_dir,cp.fingerprint)
            verified=sidecar.verify()
            result={'verified_files':len(verified),'integrity_bytes':sidecar.stats['integrity_bytes'],
                'scope':'quantized sidecar content only; base checkpoint fingerprint uses file metadata'}
        else:
            config=json.loads((Path(args.model)/'config.json').read_text())
            if args.command=='limits':result=estimate_glm(config,cfg.cache_bytes)
            else:
                from .sidecar import convert_experts
                result=convert_experts(cp,config,args.output,args.bits,args.group_size,max_matrices=args.max_matrices,scope=args.scope,linear_bits=args.linear_bits)
                result={'converted_matrices':len(result.entries),'format_bits':result.bits,'base_checkpoint_unchanged':True,
                   'quality_certified':False,'scope':args.scope,'linear_bits':args.linear_bits,'note':'Mixed uses selected non-expert linear matrices in the requested linear_bits; route/indexer/kv_b remain original' }
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
