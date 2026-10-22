"""Local-only AirLLM Next command line. Heavy model downloads are never implicit."""
import argparse
from dataclasses import replace
from contextlib import ExitStack
import json
import platform
import shutil
import sys
from pathlib import Path
from .config import NextConfig
from .tuning import auto_config,telemetry,measure_storage
from .limits import estimate_glm


def nonnegative_int(value):
    try:
        result=int(value)
    except (TypeError,ValueError):
        raise argparse.ArgumentTypeError('Expected a nonnegative integer') from None
    if result<0:raise argparse.ArgumentTypeError('Expected a nonnegative integer')
    return result


def positive_int(value):
    result=nonnegative_int(value)
    if result<1:raise argparse.ArgumentTypeError('Expected a positive integer')
    return result


def token_ids(value):
    try:
        result=[int(part) for part in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated nonnegative token IDs') from None
    if not result or any(part<0 or part>2**31-1 for part in result):
        raise argparse.ArgumentTypeError('Token IDs must fit nonnegative int32')
    return result


def doctor_report():
    """Describe this process, without treating device visibility as model validation."""
    report={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
        'mlx':None,'metal_available':False,'mlx_import_error':None,'telemetry':telemetry(),
        'working_directory_free_bytes':shutil.disk_usage(Path.cwd()).free,
        'checkpoint_generation_tested':False,
        'validation':'Hardware visibility only; run the validation scripts and real local checkpoints separately',
        'backends':{
            'glm':{'architecture':'glm_moe_dsa','experimental':False,'sampling':True,
                   'speculation':'exact greedy local GLM draft verification'},
            'qwen-experimental':{'architecture':'qwen3_5_moe_text','experimental':True,
                                 'sampling':False,'speculation':False,'nvfp4':False},
            'llama':{'architecture':'llama','experimental':False,'sampling':True,
                     'scope':'Bias-free SiLU Llama with default RoPE; checkpoint validation is separate'}}}
    try:
        import mlx.core as mx
        report['mlx']=mx.__version__
        report['metal_available']=mx.metal.is_available()
    except (ImportError,OSError,RuntimeError) as error:
        report['mlx_import_error']=f'{type(error).__name__}: {error}'
    return report


def config_for(args):
    values={}
    for field in ('cache_bytes','io_bytes','kv_bytes','workspace_bytes','reserve_bytes','max_tensor_bytes'):
        mib=getattr(args,field.replace('_bytes','_mib'),None)
        if mib is not None:values[field]=int(mib*1024**2)
    if getattr(args,'float32',False):values['dtype']='float32'
    if getattr(args,'quantized_dir',None):values['quantized_dir']=args.quantized_dir
    for field in ('lossless_dir','lossless_native_library','row_chunk','prefill_chunk','io_workers'):
        if getattr(args,field,None) is not None:values[field]=getattr(args,field)
    if getattr(args,'no_prefetch',False):values['prefetch']=False
    if getattr(args,'max_context',None) is not None:values['max_context']=args.max_context
    if getattr(args,'cache_dir',None):values['cache_dir']=args.cache_dir
    if getattr(args,'cache_policy',None):
        values['policy']=args.cache_policy;values['use_saved_tuning']=False
    if getattr(args,'coalesce_reads',False):values['coalesce_reads']=True
    if getattr(args,'no_selective_fp8_scales',False):values['selective_fp8_scales']=False
    values.update(device=args.device,load_tokenizer=not getattr(args,'token_ids',None))
    if getattr(args,'command',None) in ('prepare','tune','limits','convert','verify','pack'):
        # Metadata and bounded conversion do not allocate inference caches and
        # must remain usable when the automatic model budget is unavailable.
        return NextConfig(profile=args.profile,**values)
    return auto_config(args.profile,**values)


def combined_allocator_budget(configs,available_bytes=None):
    """Preflight the CLI-owned model group before loading a local draft.

    MLX allocator limits are process-wide. These disjoint requested model
    payloads must fit together; the system reserve is shared, not multiplied.
    This does not discover or budget independently loaded library models.
    """
    if available_bytes is None:
        import psutil
        available_bytes=psutil.virtual_memory().available
    total=sum(getattr(config,name) for config in configs
              for name in ('cache_bytes','io_bytes','kv_bytes','workspace_bytes'))
    reserve=max(config.reserve_bytes for config in configs)
    if total+reserve>available_bytes:
        raise MemoryError('Combined target and draft budgets exceed available memory after the required reserve; reduce their memory budgets')
    return {'scope':'CLI target and explicit local draft only','models':len(configs),
            'memory_limit_bytes':total,'reserve_bytes':reserve,
            'allocator_cache_limit_bytes':min(256*1024**2,sum(c.workspace_bytes for c in configs)//4),
            'available_bytes_before_draft_load':available_bytes}


def apply_combined_allocator(target,draft,budget):
    """Reapply once after both constructors have set their individual limits."""
    if target.mx is not draft.mx:raise ValueError('Target and draft must use the same MLX allocator')
    if target.mx.metal.is_available():
        target.mx.set_cache_limit(budget['allocator_cache_limit_bytes'])
        target.mx.set_memory_limit(budget['memory_limit_bytes'])


def main(argv=None):
    parser=argparse.ArgumentParser(prog='airllm-next',description='Local streamed MLX GLM/Llama inference and explicit experimental Qwen backend')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Check local environment without model weights')
    inspect=sub.add_parser('inspect',help='Offline architecture, quantization and storage feasibility (no inference)')
    inspect.add_argument('model');inspect.add_argument('--tokens',type=nonnegative_int,default=4096)
    inspect.add_argument('--storage-budget-bytes',type=nonnegative_int,default=2_000_000_000_000)
    for name in ('prepare','tune','calibrate','limits','convert','verify','pack','run','serve'):
        p=sub.add_parser(name);p.add_argument('model',help='Local safetensors checkpoint directory')
        p.add_argument('--cache-dir');p.add_argument('--profile',choices=['balanced','performance'],default='balanced')
        p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
        if name in ('run','calibrate','serve'):
            p.add_argument('--prompt',default='Bonjour');p.add_argument('--token-ids',type=token_ids,help='Comma-separated ids (no tokenizer required)')
            p.add_argument('--max-new-tokens',type=positive_int,default=32);p.add_argument('--max-context',type=positive_int,default=4096)
            for field in ('cache','io','kv','workspace','reserve'):
                p.add_argument(f'--{field}-mib',type=nonnegative_int,help=f'{field} memory budget in MiB')
            p.add_argument('--max-tensor-mib',type=positive_int,help='Largest individual tensor read in MiB')
            p.add_argument('--row-chunk',type=positive_int);p.add_argument('--prefill-chunk',type=positive_int)
            p.add_argument('--io-workers',type=positive_int)
            transport=p.add_mutually_exclusive_group()
            transport.add_argument('--quantized-dir',help='Lossy packed-weight sidecar')
            transport.add_argument('--lossless-dir',help='Explicit original-weight compressed overlay; hash verification remains enabled')
            p.add_argument('--lossless-native-library',help='Explicit native palette decoder library for --lossless-dir')
            p.add_argument('--no-prefetch',action='store_true');p.add_argument('--float32',action='store_true')
            p.add_argument('--json-report')
            p.add_argument('--cache-policy',choices=['lru','slru','lfu','cost'])
            p.add_argument('--coalesce-reads',action='store_true',help='Opt in to workload-dependent read coalescing')
            p.add_argument('--no-selective-fp8-scales',action='store_true')
        if name in ('run','serve','calibrate'):
            p.add_argument('--backend',choices=['glm','qwen-experimental','llama'],default='glm')
        if name=='run':
            drafting=p.add_mutually_exclusive_group()
            drafting.add_argument('--draft-model',help='Explicit compatible local draft with the target architecture/tokenizer; no download')
            drafting.add_argument('--prompt-lookup',action='store_true',help='Propose bounded spans from the known prefix; target verifies every token')
            p.add_argument('--speculative-depth',type=positive_int,default=4)
            p.add_argument('--no-adaptive-speculation',action='store_true')
        if name=='serve':
            p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
            p.add_argument('--model-name',default='airllm-next')
            p.add_argument('--max-body-bytes',type=positive_int,default=1024*1024,help='Maximum JSON request body bytes')
        if name=='verify':
            p.add_argument('--quantized-dir',required=True,help='Explicit full content scan of this sidecar')
        if name=='pack':
            p.add_argument('--output',required=True,help='New directory for lossless original-weight transport')
            p.add_argument('--key',action='append',dest='keys',help='Pack only this tensor (repeatable); default selects all')
            p.add_argument('--max-total-bytes',type=positive_int,default=256*1024**2,help='Explicit allowance for source tensor bytes to pack')
            p.add_argument('--block-bytes',type=positive_int,default=1024**2,help='Even block size for bounded compression')
            p.add_argument('--codec',choices=['auto','raw','zlib','shuffle-zlib','palette16'],default='auto')
            p.add_argument('--standalone',action='store_true',help='Include every tensor plus bounded local config/tokenizer assets; original checkpoint no longer required')
            p.add_argument('--max-asset-bytes',type=nonnegative_int,default=256*1024**2,help='Byte allowance for standalone config/tokenizer assets')
        if name=='convert':
            p.add_argument('--output',required=True);p.add_argument('--bits',type=int,choices=[4,8],default=4)
            p.add_argument('--group-size',type=int,choices=[32,64,128],default=64)
            p.add_argument('--max-matrices',type=positive_int,help='Bounded conversion/checkpointing')
            p.add_argument('--scope',choices=['experts','mixed'],default='experts')
            p.add_argument('--linear-bits',type=int,choices=[4,8],default=8)
    args=parser.parse_args(argv)
    if args.command=='doctor':
        report=doctor_report();print(json.dumps(report,indent=2))
        return 1 if report['mlx_import_error'] else 0
    if args.command=='inspect':
        from .adapters import inspect_local_checkpoint
        print(json.dumps(inspect_local_checkpoint(args.model,tokens=args.tokens,storage_budget_bytes=args.storage_budget_bytes),indent=2));return
    if getattr(args,'lossless_native_library',None) and not args.lossless_dir:
        parser.error('--lossless-native-library requires --lossless-dir')
    if args.command=='serve' and args.token_ids is not None:
        parser.error('serve requires a local tokenizer; --token-ids is only for run/calibrate')
    if args.command=='serve' and not 0<=args.port<=65535:
        parser.error('--port must be from 0 to 65535')
    if args.command=='pack' and args.block_bytes%2:
        parser.error('--block-bytes must be even')
    if (getattr(args,'draft_model',None) or getattr(args,'prompt_lookup',False)) and args.backend=='qwen-experimental':
        parser.error('CLI draft verification currently requires --backend glm or llama')
    cfg=config_for(args)
    if args.command in ('run','calibrate','serve'):
        from .runtime import AirLLMNextModel
        model_class=AirLLMNextModel
        if getattr(args,'backend','glm')=='qwen-experimental':
            from .qwen_runtime import ExperimentalQwenModel
            model_class=ExperimentalQwenModel
        elif getattr(args,'backend','glm')=='llama':
            from .llama_runtime import AirLLMNextLlamaModel
            model_class=AirLLMNextLlamaModel
        with ExitStack() as stack:
            model=stack.enter_context(model_class(args.model,next_config=cfg))
            if args.command=='serve':
                from .server import serve
                serve(model,model_name=args.model_name,host=args.host,port=args.port,
                      max_new_tokens=args.max_new_tokens,max_body_bytes=args.max_body_bytes);return
            draft=None;allocator_report=None
            if getattr(args,'draft_model',None):
                draft_cfg=replace(cfg,cache_bytes=min(cfg.cache_bytes,256*1024**2),
                    io_bytes=min(cfg.io_bytes,128*1024**2),kv_bytes=min(cfg.kv_bytes,128*1024**2),
                    workspace_bytes=min(cfg.workspace_bytes,256*1024**2),load_tokenizer=cfg.load_tokenizer,
                    quantized_dir=None,lossless_dir=None,lossless_native_library=None)
                allocator_report=combined_allocator_budget((cfg,draft_cfg))
                draft=stack.enter_context(model_class(args.draft_model,next_config=draft_cfg))
                apply_combined_allocator(model,draft,allocator_report)
            elif getattr(args,'prompt_lookup',False):
                from .prompt_draft import PromptSpanDraft
                draft=PromptSpanDraft()
            if args.token_ids is not None:ids=[args.token_ids]
            else:ids=[model.tokenizer.encode(args.prompt)]
            if not ids[0] or any(type(value) is not int or not 0<=value<model.config['vocab_size'] for value in ids[0]):
                parser.error('Prompt must encode to nonempty token IDs inside the local vocabulary')
            if len(ids[0])+args.max_new_tokens>min(cfg.max_context,model.config['max_position_embeddings']):
                parser.error('Prompt plus requested completion exceeds the configured context limit')
            if args.command=='calibrate':
                from .tuning import calibrate_prefetch
                print(json.dumps(calibrate_prefetch(model,ids,tokens=args.max_new_tokens),indent=2));return
            generation_options={}
            if getattr(args,'backend','glm')!='qwen-experimental':
                generation_options={'draft_model':draft,'speculative_depth':getattr(args,'speculative_depth',4),
                    'speculative_adaptive':not getattr(args,'no_adaptive_speculation',False)}
            tokens=model.generate(ids,max_new_tokens=args.max_new_tokens,**generation_options)
            values=tokens.tolist()[0]
            print(model.tokenizer.decode(values[len(ids[0]):]) if model.tokenizer else values)
            metrics=model.metrics()
            if allocator_report is not None:metrics['allocator_group']=allocator_report
            print(json.dumps(metrics,indent=2),file=sys.stderr)
            if args.json_report:
                from ..ultra.checkpoint import atomic_json
                atomic_json(args.json_report,metrics)
        return
    from .lossless import open_checkpoint
    with open_checkpoint(args.model,cfg) as cp:
        if args.command=='prepare':result={'prepared':True,'generation_ready':False,'fingerprint':cp.fingerprint,'stats':dict(cp.stats),'tensor_count':len(cp.tensors),'checkpoint_bytes':cp.total_bytes}
        elif args.command=='tune':result=measure_storage(cp,cfg)
        elif args.command=='verify':
            from .sidecar import QuantizedSidecar
            sidecar=QuantizedSidecar(args.quantized_dir,cp.fingerprint)
            verified=sidecar.verify()
            result={'verified_files':len(verified),'integrity_bytes':sidecar.stats['integrity_bytes'],
                'scope':'quantized sidecar content only; base checkpoint fingerprint uses file metadata'}
        elif args.command=='pack':
            from .lossless import pack_checkpoint
            result=pack_checkpoint(cp,args.output,keys=args.keys,block_bytes=args.block_bytes,
                                   codec=args.codec,max_total_bytes=args.max_total_bytes,
                                   standalone=args.standalone,max_asset_bytes=args.max_asset_bytes)
            result={'format':'lossless_original_weight_standalone' if args.standalone else 'lossless_original_weight_overlay',
                    'base_checkpoint_required':not args.standalone,
                    'base_checkpoint_unchanged':True,'performance_gain_certified':False,
                    'manifest':result}
        else:
            config=json.loads((Path(args.model)/'config.json').read_text())
            if args.command=='limits':result=estimate_glm(config,cfg.cache_bytes)
            else:
                from .sidecar import convert_experts
                result=convert_experts(cp,config,args.output,args.bits,args.group_size,max_matrices=args.max_matrices,scope=args.scope,linear_bits=args.linear_bits)
                result={'converted_matrices':len(result.entries),'format_bits':result.bits,'base_checkpoint_unchanged':True,
                   'quality_certified':False,'scope':args.scope,'linear_bits':args.linear_bits,'note':'Mixed uses selected non-expert linear matrices in the requested linear_bits; route/indexer/kv_b remain original' }
        print(json.dumps(result,indent=2))


if __name__=='__main__':raise SystemExit(main())
