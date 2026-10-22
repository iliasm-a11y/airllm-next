"""Numerical primitive parity against pinned official CPU Torch; not model inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'air_llm'))
from airllm.nextgen.recurrent import gated_delta_rule


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(argv)
    torch.set_num_threads(1)
    path=ROOT/'evidence-ooc/sources/qwen_recurrent_reference.py'
    spec=importlib.util.spec_from_file_location('official_qwen_recurrence',path)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    rng=np.random.default_rng(20261022);cases=[]
    for length in (1,7,64):
        for normalized in (False,True):
            for has_initial in (False,True):
                q=rng.normal(size=(2,length,2,8)).astype(np.float32)
                k=rng.normal(size=q.shape).astype(np.float32)
                v=rng.normal(size=(2,length,4,5)).astype(np.float32)
                g=-rng.uniform(size=(2,length,4)).astype(np.float32)
                b=rng.uniform(size=g.shape).astype(np.float32)
                initial=rng.normal(size=(2,4,8,5)).astype(np.float32) if has_initial else None
                actual,state=gated_delta_rule(q,k,v,g,b,normalize_qk=normalized,initial_state=initial)
                tq=torch.from_numpy(np.repeat(q,2,axis=2));tk=torch.from_numpy(np.repeat(k,2,axis=2))
                expected,ref_state=ref.torch_recurrent_gated_delta_rule(tq,tk,torch.from_numpy(v),torch.from_numpy(g),torch.from_numpy(b),
                    initial_state=torch.from_numpy(initial) if initial is not None else None,
                    output_final_state=True,use_qk_l2norm_in_kernel=normalized)
                expected=expected.numpy();ref_state=ref_state.numpy()
                case={'sequence_length':length,'batch_size':2,'normalized_qk':normalized,'initial_state':has_initial,
                      'output_max_abs_error':float(np.max(np.abs(actual-expected))),
                      'state_max_abs_error':float(np.max(np.abs(state-ref_state)))}
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(state,ref_state,rtol=2e-5,atol=2e-5)
                case['passed']=True;cases.append(case)
    result={'kind':'numerical_primitive_validation','compute_device':'CPU Linux',
            'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,
            'source_revision':'a672eb5f474b8b30173e9f96eab66adf292da635',
            'source_copy_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'full_qwen_generation_validated':False,'metal_validated':False,
            'cases':cases,'rtol':2e-5,'atol':2e-5}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':len(cases),'output_max_abs_error':max(x['output_max_abs_error'] for x in cases),
                      'state_max_abs_error':max(x['state_max_abs_error'] for x in cases)}))


if __name__=='__main__':main()
