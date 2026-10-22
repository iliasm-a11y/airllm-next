"""Public inference contracts must hold before any weight IO or KV mutation."""
import numpy as np
import pytest
from airllm.nextgen.runtime import AirLLMNextModel
from airllm.nextgen.qwen_runtime import ExperimentalQwenModel
from airllm.nextgen.generation import token_batch,eos_ids,generation_options
from test_native import options
from test_qwen_runtime import make_qwen,arr


@pytest.fixture(params=['glm','qwen'])
def guarded_model(request,tiny_glm,tmp_path):
    if request.param=='glm':
        root,_=tiny_glm
        model=AirLLMNextModel(root,options(tmp_path,prefill_chunk=2))
    else:
        root,_,opt=make_qwen(tmp_path)
        from dataclasses import replace
        model=ExperimentalQwenModel(root,replace(opt,prefill_chunk=2))
    with model:yield model


def test_forward_and_generate_reject_bad_ids_without_weight_reads(guarded_model):
    model=guarded_model;before=model.checkpoint.stats['logical_bytes']
    invalid=[[[1.5]],[[True]],[[1,-1]],[[model.config['vocab_size']]],
             np.array([[2**40]],dtype=np.uint64),np.array([[2**40]],dtype=np.int64),
             [[1],[2]],[[]],[1,2]]
    for ids in invalid:
        for operation in (model.forward,model.generate):
            with pytest.raises(ValueError):operation(ids)
    assert model.checkpoint.stats['logical_bytes']==before


def test_invalid_generation_options_do_not_read_weights(guarded_model):
    model=guarded_model;before=model.checkpoint.stats['logical_bytes']
    for value in (True,1.5,-1,float('nan')):
        with pytest.raises(ValueError,match='max_new_tokens'):model.generate([[1]],max_new_tokens=value)
    for value in (True,-1,float('nan'),float('inf'),'1'):
        with pytest.raises(ValueError,match='temperature'):model.generate([[1]],temperature=value)
    for value in (True,-1,1.5,'1',[False],[model.config['vocab_size']]):
        with pytest.raises(ValueError,match='eos_token_id'):model.generate([[1]],eos_token_id=value)
    for value in (True,-1,1.5):
        with pytest.raises(ValueError,match='logits_to_keep'):model.forward([[1]],logits_to_keep=value)
    assert model.checkpoint.stats['logical_bytes']==before


def test_chunked_forward_retains_the_requested_global_logits_suffix(guarded_model):
    model=guarded_model;ids=[[1,8,12,5,21]]
    expected=arr(model.forward(ids).logits)
    for count in (1,2,3,5,8):
        actual=model.forward(ids,logits_to_keep=count)
        np.testing.assert_allclose(arr(actual.logits),expected[:,-count:],rtol=4e-4,atol=1e-5)
        assert actual.past_key_values.position==5


def test_eos_collections_stop_on_first_generated_token(guarded_model):
    model=guarded_model;ids=[[1,8,12]]
    first=int(np.asarray(model.generate(ids,max_new_tokens=1,eos_token_id=[]))[0,-1])
    for eos in (first,[first],(first,),{first},frozenset([first]),np.array([first])):
        actual=np.asarray(model.generate(ids,max_new_tokens=5,eos_token_id=eos))
        assert actual.tolist()==[ids[0]+[first]]
        assert model.stats['tokens_generated']==1


def test_zero_generation_and_closed_model_contract(guarded_model):
    model=guarded_model;before=model.checkpoint.stats['logical_bytes']
    actual=model.generate([[1,8]],max_new_tokens=0)
    np.testing.assert_array_equal(np.asarray(actual),[[1,8]])
    assert model.stats['tokens_generated']==0
    assert model.checkpoint.stats['logical_bytes']==before
    model.close()
    with pytest.raises(RuntimeError,match='closed'):model.generate([[1]],max_new_tokens=0)


def test_shared_validators_accept_numpy_integrals_and_preserve_values():
    assert generation_options(np.int64(3),np.float64(.5))==(3,.5)
    assert token_batch(np.array([[1,2]],np.uint64),3).dtype==np.int32
    assert eos_ids(None)==frozenset()
    assert eos_ids((np.int64(2),2),3)==frozenset([2])


def test_requested_logits_budget_fails_before_weight_reads(guarded_model):
    from dataclasses import replace
    model=guarded_model;before=model.checkpoint.stats['logical_bytes']
    model.options=replace(model.options,workspace_bytes=1)
    with pytest.raises(MemoryError,match='logits'):model.forward([[1]])
    assert model.checkpoint.stats['logical_bytes']==before


def test_attention_workspace_rejection_preserves_the_committed_prefix(guarded_model):
    from dataclasses import replace
    model=guarded_model;cache=model.forward([[1,8,12]]).past_key_values
    # Keep output budget separate; the layer's scratch reader is constrained.
    model.reader.cfg=replace(model.reader.cfg,workspace_bytes=1)
    with pytest.raises(MemoryError,match='attention workspace'):
        model.forward([[5]],past_key_values=cache,logits_to_keep=1)
    assert cache.position==3
