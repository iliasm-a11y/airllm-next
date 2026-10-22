"""Compare real Ling expert outputs through baseline and compressed AirLLM I/O.

This validates actual trained expert blocks, not whole-model logits or quality.
No expert is skipped within a tested block. CPU FP32 arithmetic is identical
on both paths and starts from identical BF16 bit patterns.
"""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
from common import ROOT, save
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.nextgen.lossless import LosslessOverlay, NativePalette, pack_checkpoint


def fp32(a):
    return (a.astype(np.uint32)<<16).view(np.float32)


def evaluate(x,w):
    # Same per-row reduction order independent of the number of rows in x.
    outs=[]
    for row in x:
        g=np.einsum('ij,j->i',w[0],row,optimize=False)
        u=np.einsum('ij,j->i',w[1],row,optimize=False)
        h=(g/(1+np.exp(-g)))*u
        outs.append(np.einsum('ij,j->i',w[2],h,optimize=False))
    return np.stack(outs)


def run(model):
    cp=OffsetCheckpoint(model)
    keys=[]
    groups=[]
    for layer,e in [(1,0),(1,127),(10,0),(19,255)]:
        group=[f'model.layers.{layer}.mlp.experts.{e}.{s}.weight' for s in ['gate_proj','up_proj','down_proj']]
        keys.extend(group);groups.append(group)
    dest=ROOT/'build/trained-expert-pack'
    if not dest.exists():
        meta=pack_checkpoint(cp,dest,keys=keys,codec='palette16',block_bytes=256*1024,
            max_total_bytes=32*1024**2)
    reader=LosslessOverlay(cp,dest,palette_decoder=NativePalette(ROOT/'build/palette.dylib'))
    rng=np.random.default_rng(20261022)
    x=rng.normal(size=(4,2048)).astype(np.float32)
    rows=[]
    outputs={}
    for j,group in enumerate(groups):
        raw_a=[cp.read_array(k) for k in group]
        raw_b=[reader.read_array(k) for k in group]
        assert all(a.tobytes()==b.tobytes() for a,b in zip(raw_a,raw_b))
        a=evaluate(x,[fp32(w) for w in raw_a])
        b=evaluate(x,[fp32(w) for w in raw_b])
        assert a.tobytes()==b.tobytes()
        # Also compare four singleton calls with the grouped input interface.
        singles=np.concatenate([evaluate(t[None,:],[fp32(w) for w in raw_b]) for t in x])
        assert singles.tobytes()==a.tobytes()
        rows.append(dict(expert=group[0].rsplit('.',2)[0],shape=list(a.shape),
            weights_equal_bitwise=True,outputs_equal_bitwise=True,max_abs_difference=float(np.max(abs(a-b))),
            singletons_equal_bitwise=True,output_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        outputs[f'baseline_{j}']=a;outputs[f'packed_{j}']=b
    np.savez_compressed(ROOT/'results/trained_expert_outputs.npz',inputs=x,**outputs)
    save('trained_parity.json',dict(evidence='measured trained BF16 expert blocks, FP32 NumPy reference arithmetic',
        results=rows,transport=dict(reader.transport_stats),
        whole_model_logits_validated=False,mlx_model_runtime_validated=False,
        limitation='These are 4 experts of Ling-mini, not a full Ling/GLM/Qwen inference or a semantic quality evaluation.'))
    print(rows,flush=True)
    reader.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True)
    run(ap.parse_args().model_dir)
