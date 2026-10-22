"""Reproducible random miniature hybrid model parity; never a full-model claim."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
import numpy as np
import torch
import mlx.core as mx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.qwen_runtime import ExperimentalQwenModel


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='cpu');args=p.parse_args(argv)
    if args.device=='gpu' and not mx.metal.is_available():
        raise RuntimeError('Explicit GPU validation requires real Metal; CPU fallback is forbidden')
    torch.set_num_threads(1);mx.set_default_device(getattr(mx,args.device))
    source=ROOT/'tests_next/test_qwen_runtime.py'
    spec=importlib.util.spec_from_file_location('qwen_miniature_fixture',source)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    cases=[]
    with tempfile.TemporaryDirectory() as temp:
        root,ref,opt=fixture.make_qwen(Path(temp))
        opt=replace(opt,device=args.device)
        checkpoint_sha=hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest()
        with ExperimentalQwenModel(root,opt) as model:
            for ids in ([[1,8,12,5]],[[7,3,9,10,2]],[[13]]):
                tid=torch.tensor(ids)
                with torch.inference_mode():expected=ref(tid,use_cache=False).logits.numpy()
                out=model.forward(ids);actual=np.array(out.logits)
                np.testing.assert_allclose(actual,expected,rtol=4e-4,atol=1e-5)
                tokens=tid.clone()
                with torch.inference_mode():
                    for _ in range(8):tokens=torch.cat([tokens,ref(tokens,use_cache=False).logits[:,-1].argmax(-1)[:,None]],dim=1)
                generated=np.array(model.generate(ids,max_new_tokens=8))
                np.testing.assert_array_equal(generated,tokens.numpy())
                cases.append({'input_ids':ids,'expected_tokens':tokens.tolist(),'actual_tokens':generated.tolist(),
                              'logit_max_abs_error':float(np.max(np.abs(actual-expected))),
                              'greedy_8_tokens_identical':True,'metrics':model.metrics()})
        result={'kind':'random_miniature_qwen_hybrid_generation_validation',
                'platform':platform.platform(),'compute_device':args.device,
                'torch':torch.__version__,'mlx':mx.__version__,'numpy':np.__version__,
                'checkpoint_sha256':checkpoint_sha,'fixture_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'layers':4,'layer_types':['linear_attention']*3+['full_attention'],
                'hidden_size':32,'num_experts':4,'experts_per_token':2,'dtype':'float32',
                'full_qwen3_8_checkpoint_validated':False,
                'miniature_metal_numerics_validated':args.device=='gpu',
                'full_checkpoint_metal_validated':False,'trained_generation_quality_validated':False,
                'cases':cases,'rtol':4e-4,'atol':1e-5}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'generated_tokens':24,'all_greedy_sequences_identical':True,
                      'max_logit_abs_error':max(x['logit_max_abs_error'] for x in cases)}))


if __name__=='__main__':main()
