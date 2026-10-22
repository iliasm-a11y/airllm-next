"""Full source-free inference parity for the integrated lossless checkpoint.

These random tiny models verify transport and runtime integration; they do not
establish trained model quality or giant-checkpoint compatibility.
"""
import shutil

import numpy as np
import pytest
import torch

from airllm.nextgen import AirLLMNextModel
from airllm.nextgen.lossless import pack_checkpoint
from airllm.nextgen.qwen_runtime import ExperimentalQwenModel
from airllm.nextgen.storage import OffsetCheckpoint
from test_native import options
from test_qwen_runtime import make_qwen


@pytest.mark.parametrize('codec',['raw','palette16'])
def test_glm_standalone_logits_and_generation_match_original(tiny_glm,tmp_path,codec):
    root,_=tiny_glm;cfg=options(tmp_path,workspace_bytes=2*1024**2)
    ids=[[1,8,12,5]]
    with AirLLMNextModel(root,cfg) as model:
        logits=np.asarray(model(ids).logits).copy()
        tokens=np.asarray(model.generate(ids,max_new_tokens=5,eos_token_id=[])).copy()
    packed=tmp_path/'standalone'
    with OffsetCheckpoint(root) as source:
        pack_checkpoint(source,packed,standalone=True,block_bytes=8192,codec=codec)
    shutil.rmtree(root)
    with AirLLMNextModel(packed,cfg) as model:
        np.testing.assert_array_equal(np.asarray(model(ids).logits),logits)
        np.testing.assert_array_equal(np.asarray(model.generate(ids,max_new_tokens=5,eos_token_id=[])),tokens)
        assert model.checkpoint.stats['standalone_pack']==1
        assert model.checkpoint.transport_stats['payload_bytes']>0
        assert model.reader.io.peak_bytes<=cfg.io_bytes


def test_qwen_standalone_preserves_logits_recurrence_and_generation(tmp_path):
    root,ref,cfg=make_qwen(tmp_path)
    ids=[[1,8,12,5]]
    with ExperimentalQwenModel(root,cfg) as model:
        first=model(ids);logits=np.asarray(first.logits).copy()
        next_logits=np.asarray(model([[7]],past_key_values=first.past_key_values).logits).copy()
        tokens=np.asarray(model.generate(ids,max_new_tokens=5,eos_token_id=[])).copy()
    packed=tmp_path/'standalone'
    with OffsetCheckpoint(root) as source:
        pack_checkpoint(source,packed,standalone=True,block_bytes=8192,codec='auto')
    shutil.rmtree(root)
    with ExperimentalQwenModel(packed,cfg) as model,torch.inference_mode():
        first=model(ids)
        np.testing.assert_array_equal(np.asarray(first.logits),logits)
        np.testing.assert_array_equal(np.asarray(model([[7]],past_key_values=first.past_key_values).logits),next_logits)
        np.testing.assert_array_equal(np.asarray(model.generate(ids,max_new_tokens=5,eos_token_id=[])),tokens)
        np.testing.assert_allclose(np.asarray(first.logits),ref(torch.tensor(ids)).logits.numpy(),atol=1e-5,rtol=4e-4)
        assert model.checkpoint.stats['standalone_pack']==1
