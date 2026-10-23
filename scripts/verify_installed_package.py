#!/usr/bin/env python3
"""Verify an installed distribution from outside its source checkout.

This constructs random miniature fixtures to test installation and numerical
contracts. It is not a trained-model quality or throughput benchmark.
"""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('MLX_ENABLE_TF32','0')
    import airllm
    package=Path(airllm.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('airllm was not imported from the active installed environment')
    modules=[]
    for entry in pkgutil.walk_packages(airllm.__path__,airllm.__name__+'.'):
        importlib.import_module(entry.name)
        modules.append(entry.name)
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM,GlmMoeDsaConfig,GlmMoeDsaForCausalLM
    from airllm import AutoModel
    from airllm.nextgen import NextConfig
    torch.set_num_threads(1)
    root=Path(args.work_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    options=dict(device='cpu',dtype='float32',cache_bytes=1024**2,io_bytes=1024**2,
        kv_bytes=1024**2,workspace_bytes=4*1024**2,reserve_bytes=0,max_tensor_bytes=1024**2,
        max_context=64,prefill_chunk=2,row_chunk=16,load_tokenizer=False,use_saved_tuning=False)
    def cli(*arguments):
        result=subprocess.run([sys.executable,'-m','airllm.nextgen.cli',*arguments],
            capture_output=True,text=True,check=True)
        return result.stdout
    help_text=cli('--help')
    assert all(command in help_text for command in ('doctor','inspect','run','pack','serve','preflight','agent-serve'))
    doctor=json.loads(cli('doctor'))
    fixtures=[]
    for architecture in ('llama','glm_moe_dsa'):
        torch.manual_seed(521)
        common=dict(vocab_size=96,hidden_size=48,intermediate_size=96,
            num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2)
        if architecture=='llama':
            reference=LlamaForCausalLM(LlamaConfig(**common,attention_bias=False,mlp_bias=False)).eval()
            backend='llama'
        else:
            common['num_key_value_heads']=4
            config=GlmMoeDsaConfig(**common,n_routed_experts=8,num_experts_per_tok=2,
                n_shared_experts=1,first_k_dense_replace=1,moe_intermediate_size=24,
                kv_lora_rank=8,q_lora_rank=16,qk_rope_head_dim=8,qk_nope_head_dim=8,
                v_head_dim=8,index_head_dim=8,index_n_heads=2,index_topk=4,
                indexer_types=['full','shared','full'],
                rope_parameters={'rope_type':'default','rope_theta':10000.})
            reference=GlmMoeDsaForCausalLM(config).eval();backend='glm'
        checkpoint=root/architecture
        reference.save_pretrained(checkpoint,safe_serialization=True,max_shard_size='100KB')
        ids=torch.tensor([[1,8,12]])
        with torch.inference_mode():expected=reference.generate(ids,max_new_tokens=8,do_sample=False).tolist()
        with AutoModel.from_pretrained(checkpoint,backend='next',
                next_config=NextConfig(**options,compact_cache_bytes=256*1024,output_tile_cache=True,
                    cache_dir=str(root/(architecture+'-metadata')))) as model:
            actual=model.generate(ids,max_new_tokens=8).tolist()
            from airllm.nextgen import AgentEngine
            journal=root/(architecture+'-agent-journals')
            with AgentEngine(model,state_dir=journal,quantum_tokens=2) as engine:
                job=engine.submit(ids.tolist()[0],max_tokens=8)
                first=engine.step()
            with AgentEngine(model,state_dir=journal,quantum_tokens=2) as engine:
                if engine.snapshot(job)['state']!='completed':engine.resume(job)
                while engine.step() is not None:pass
                restored=engine.snapshot(job)
                # Replay must reproduce the ordinary native trajectory. Independent
                # Transformers parity remains a separate mandatory gate below.
                assert ids.tolist()[0]+restored['output_ids']==actual[0],(
                    architecture,'agent replay differs from ordinary generation',restored['output_ids'],actual[0])
        assert actual==expected,(architecture,actual,expected)
        inspected=json.loads(cli('inspect',str(checkpoint),'--tokens','32'))
        assert inspected['runtime_available'] and not inspected['generation_validated']
        import ast
        cli_tokens=ast.literal_eval(cli('run',str(checkpoint),'--backend',backend,
            '--device','cpu','--float32','--token-ids','1,8,12','--max-new-tokens','8',
            '--cache-mib','1','--io-mib','1','--kv-mib','1','--workspace-mib','4',
            '--reserve-mib','0','--max-context','64','--cache-dir',str(root/(architecture+'-cli-metadata'))).strip())
        assert cli_tokens==expected[0]
        fixtures.append({'architecture':architecture,'fixture':'seeded random miniature',
            'compute_device':'cpu','new_tokens':len(actual[0])-3,'tokens':actual[0],
            'matches_independent_transformers_reference':True,'cli_matches_reference':True,
            'installed_agent_durable_replay_matches_ordinary':True})
    report={'version':metadata.version('airllm'),'python':sys.version,
        'package_file':str(package),'source_checkout_required':False,
        'imported_production_modules':modules,'doctor':doctor,'fixtures':fixtures,
        'scope':'Installed package and random miniature generation contracts; no trained-model quality or speed claim'}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'version':report['version'],'imported_modules':len(modules),'fixtures_verified':len(fixtures)}))


if __name__=='__main__':main()
