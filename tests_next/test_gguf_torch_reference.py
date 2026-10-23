"""Official Torch meta-model transport against a separately constructed model."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from test_gguf_qwen import export_fixture

_file=Path(__file__).resolve().parents[1]/'benchmarks/reference_gguf_torch.py'
_spec=importlib.util.spec_from_file_location('gguf_torch_reference',_file)
reference=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=reference;_spec.loader.exec_module(reference)


@pytest.mark.parametrize('noninvolutive',[False,True])
def test_official_torch_streamed_gguf_prefill_cache_and_four_tokens(tmp_path,noninvolutive):
    path,official,_,_,_=export_fixture(tmp_path,noninvolutive=noninvolutive)
    ids=torch.tensor([[1,8,12,5]])
    with torch.inference_mode():expected=official(ids,use_cache=True)
    budget=reference.ReferenceBudget(io_bytes=2<<20,resident_bytes=2<<20,workspace_bytes=2<<20,
        kv_bytes=2<<20,reserve_bytes=0,row_chunk=7,max_context=32)
    with reference.TorchGGUFReference(path,budget) as model:
        assert model.initial_meta_parameters>sum(p.numel() for p in model.model.parameters())
        assert model.reader.stats['tensor_bytes']<path.stat().st_size
        actual=model.forward(ids,logits_to_keep=0)
        np.testing.assert_allclose(actual.logits.numpy(),expected.logits.numpy(),rtol=5e-5,atol=2e-6)
        with torch.inference_mode():expected_next=official(torch.tensor([[4]]),past_key_values=expected.past_key_values,use_cache=True)
        actual_next=model.forward([[4]],actual.past_key_values)
        np.testing.assert_allclose(actual_next.logits.numpy(),expected_next.logits.numpy(),rtol=5e-5,atol=2e-6)
        tokens=ids.clone();cache=None;current=ids;actual_tokens=ids.clone()
        with torch.inference_mode():
            for _ in range(4):
                tokens=torch.cat([tokens,official(tokens,use_cache=False).logits[:,-1].argmax(-1)[:,None]],dim=1)
                out=model.forward(current,cache);cache=out.past_key_values;current=out.logits[:,-1].argmax(-1)[:,None]
                actual_tokens=torch.cat([actual_tokens,current],dim=1)
        np.testing.assert_array_equal(actual_tokens.numpy(),tokens.numpy())
        assert model.metrics()['official_expert_forward_unchanged']
        assert model.source.stats['expert_fetches']>0
        assert model.reader.stats['peak_reserved_bytes']<=budget.io_bytes


def test_reference_rejects_large_budget_and_context_before_reads(tmp_path):
    with pytest.raises(MemoryError,match='1 GiB'):reference.ReferenceBudget(resident_bytes=1<<30)
    path,_,_,_,_=export_fixture(tmp_path)
    budget=reference.ReferenceBudget(io_bytes=2<<20,resident_bytes=2<<20,workspace_bytes=2<<20,
        kv_bytes=2<<20,reserve_bytes=0,row_chunk=7,max_context=2)
    with reference.TorchGGUFReference(path,budget) as model:
        before=model.reader.stats['tensor_bytes']
        with pytest.raises(MemoryError,match='context'):model.forward([[1,2,3]])
        assert model.reader.stats['tensor_bytes']==before


def test_reference_cli_output_alias_rejected_before_checkpoint_open(tmp_path,monkeypatch):
    model=tmp_path/'checkpoint.gguf';model.write_bytes(b'preserve these weights')
    def forbidden(*args,**kwargs):raise AssertionError('Checkpoint must not be opened')
    monkeypatch.setattr(reference,'TorchGGUFReference',forbidden)
    with pytest.raises(ValueError,match='protected'):
        reference.main(['--model-file',str(model),'--input-ids','1','--output',str(model)])
    alias=tmp_path/'checkpoint.npz';alias.write_bytes(b'preserve compressed weights')
    with pytest.raises(ValueError,match='protected'):
        reference.main(['--model-file',str(alias),'--input-ids','1','--output',str(tmp_path/'result.json'),
                        '--logits-output',str(tmp_path/'checkpoint')])
    assert model.read_bytes()==b'preserve these weights' and alias.read_bytes()==b'preserve compressed weights'
