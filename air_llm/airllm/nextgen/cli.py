"""Local-only AirLLM Next command line. Heavy model downloads are never implicit."""
import argparse
from dataclasses import replace
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
    return replace(cfg,**values)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='airllm-next',description='Native local MLX GLM DSA inference; Metal hardware validation required on Mac')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Check local environment without model weights')
    for name in ('prepare','tune','calibrate','limits','convert','run'):
        p=sub.add_parser(name);p.add_argument('model',help='Local safetensors checkpoint directory')
        p.add_argument('--cache-dir');p.add_argument('--profile',choices=['balanced','performance'],default='balanced')
        p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
        if name in ('run','calibrate'):
            p.add_argument('--prompt',default='Bonjour');p.add_argument('--token-ids',help='Comma-separated ids (no tokenizer required)')
            p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--max-context',type=int,default=4096)
            p.add_argument('--cache-mib',type=int);p.add_argument('--io-mib',type=int);p.add_argument('--kv-mib',type=int)
            p.add_argument('--quantized-dir');p.add_argument('--no-prefetch',action='store_true');p.add_argument('--float32',action='store_true')
            p.add_argument('--json-report')
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
    cfg=config_for(args)
    if args.command in ('run','calibrate'):
        from .runtime import AirLLMNextModel
        with AirLLMNextModel(args.model,next_config=cfg) as model:
            if args.token_ids:ids=[[int(x) for x in args.token_ids.split(',')]]
            else:ids=[model.tokenizer.encode(args.prompt)]
            if args.command=='calibrate':
                from .tuning import calibrate_prefetch
                print(json.dumps(calibrate_prefetch(model,ids,tokens=args.max_new_tokens),indent=2));return
            tokens=model.generate(ids,max_new_tokens=args.max_new_tokens)
            values=tokens.tolist()[0]
            print(model.tokenizer.decode(values[len(ids[0]):]) if model.tokenizer else values)
            metrics=model.metrics();print(json.dumps(metrics,indent=2),file=sys.stderr)
            if args.json_report:Path(args.json_report).write_text(json.dumps(metrics,indent=2))
        return
    with OffsetCheckpoint(args.model,cfg.metadata_dir(),cfg.max_open_files,cfg.max_tensor_bytes) as cp:
        if args.command=='prepare':result={'prepared':True,'generation_ready':False,'fingerprint':cp.fingerprint,'stats':dict(cp.stats),'tensor_count':len(cp.tensors),'checkpoint_bytes':cp.total_bytes}
        elif args.command=='tune':result=measure_storage(cp,cfg)
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
