"""Assert source/standalone logits and tokens with the same real Next runtime.

This is a transport integration check, not an independent architecture reference
or a timing benchmark. The source is read and is never renamed or modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('MLX_ENABLE_TF32','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
import numpy as np
from airllm import AutoModel
from airllm.nextgen.config import NextConfig
from airllm.nextgen.storage import OffsetCheckpoint
from airllm.ultra.checkpoint import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original')
    p.add_argument('standalone')
    p.add_argument('--output',required=True)
    p.add_argument('--backend',default='mlx-next',choices=['mlx-next','next-llama','next-qwen-experimental'])
    p.add_argument('--device',default='cpu',choices=['cpu','gpu'])
    p.add_argument('--prompt',default='The capital of France is')
    p.add_argument('--tokens',type=int,default=8)
    a=p.parse_args()
    if a.tokens<1:p.error('tokens must be positive')
    manifest=json.loads((Path(a.standalone)/'manifest.json').read_text())
    if manifest.get('standalone') is not True:p.error('The second directory must be a complete standalone pack')
    with tempfile.TemporaryDirectory(prefix='airllm-parity-') as work:
        cfg=NextConfig(cache_bytes=64*1024**2,io_bytes=16*1024**2,workspace_bytes=32*1024**2,
            kv_bytes=8*1024**2,max_tensor_bytes=8*1024**2,reserve_bytes=0,
            device=a.device,dtype='float32',max_context=256,use_saved_tuning=False,cache_dir=work)
        with AutoModel.from_pretrained(a.original,backend=a.backend,next_config=cfg) as model:
            ids=[model.tokenizer.encode(a.prompt)]
            logits=np.asarray(model(ids).logits.astype(model.mx.float32)).copy()
            tokens=np.asarray(model.generate(ids,max_new_tokens=a.tokens,eos_token_id=[])).copy()
            original_text=model.tokenizer.decode(tokens[0,len(ids[0]):].tolist())
        with patch.object(OffsetCheckpoint,'__init__',side_effect=AssertionError('Original shard reader must not be constructed')):
            with AutoModel.from_pretrained(a.standalone,backend=a.backend,next_config=cfg) as model:
                assert [model.tokenizer.encode(a.prompt)]==ids
                packed_logits=np.asarray(model(ids).logits.astype(model.mx.float32))
                packed_tokens=np.asarray(model.generate(ids,max_new_tokens=a.tokens,eos_token_id=[]))
                np.testing.assert_array_equal(packed_logits,logits)
                np.testing.assert_array_equal(packed_tokens,tokens)
                assert model.reader.io.peak_bytes<=cfg.io_bytes
                assert model.reader.cache.peak_bytes<=cfg.cache_bytes
                report={'scope':'Same-runtime transport parity with complete checkpoint and copied local tokenizer',
                    'device':a.device,'dtype':cfg.dtype,'prompt':a.prompt,'prompt_ids':ids[0],
                    'generated_ids':tokens[0,len(ids[0]):].tolist(),'completion':original_text,
                    'logits_bit_identical':True,'tokens_identical':True,
                    'original_reader_constructor_blocked_during_packed_run':True,
                    'logits_sha256':hashlib.sha256(memoryview(logits).cast('B')).hexdigest(),
                    'standalone_tensor_payload_bytes':manifest['raw_bytes'],
                    'standalone_stored_payload_bytes':manifest['stored_bytes'],
                    'cache_peak_bytes':model.reader.cache.peak_bytes,'cache_budget_bytes':cfg.cache_bytes,
                    'io_peak_bytes':model.reader.io.peak_bytes,'io_budget_bytes':cfg.io_bytes}
                atomic_json(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
