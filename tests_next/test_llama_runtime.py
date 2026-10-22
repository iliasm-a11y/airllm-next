"""Native Next Llama parity, transactional sessions and selective weight IO."""
import numpy as np
import pytest
import torch

from conftest import save_tiny
from test_native import options
from airllm import AutoModel
from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel, normalize_llama_config


@pytest.mark.parametrize('tied', [False, True])
def test_llama_logits_suffix_decode_and_generation(tmp_path, tied):
    root = tmp_path / 'llama'
    ref = save_tiny(root, tied=tied)
    ids = torch.tensor([[1, 7, 8, 9, 10]])
    with AirLLMNextLlamaModel(root, next_config=options(tmp_path)) as model, torch.inference_mode():
        expected = ref(ids, use_cache=True)
        actual = model(ids)
        np.testing.assert_allclose(np.array(actual.logits), expected.logits.numpy(), atol=4e-6, rtol=3e-4)
        for keep in (1, 3, 5, 10):
            np.testing.assert_allclose(np.array(model(ids, logits_to_keep=keep).logits),
                                       expected.logits[:, -keep:].numpy(), atol=4e-6, rtol=3e-4)
        next_id = expected.logits[:, -1].argmax(-1)[:, None]
        got = model(next_id, past_key_values=actual.past_key_values)
        wanted = ref(next_id, past_key_values=expected.past_key_values)
        np.testing.assert_allclose(np.array(got.logits), wanted.logits.numpy(), atol=4e-6, rtol=3e-4)
        assert model.generate(ids, max_new_tokens=8).tolist() == ref.generate(ids, max_new_tokens=8, do_sample=False).tolist()
        assert model.reader.cache.peak_bytes <= model.options.cache_bytes
        assert model.reader.io.peak_bytes <= model.options.io_bytes


def test_llama_selective_embedding_and_tiled_head(tmp_path):
    root = tmp_path / 'llama'
    ref = save_tiny(root, tied=True)
    ref.resize_token_embeddings(192, mean_resizing=False)
    root = tmp_path / 'resized-llama'
    ref.save_pretrained(root)
    cfg = options(tmp_path, cache_bytes=0, io_bytes=20480, max_tensor_bytes=20480, row_chunk=8)
    with AirLLMNextLlamaModel(root, next_config=cfg) as model, torch.inference_mode():
        ids = [[1, 8, 9]]
        np.testing.assert_allclose(np.array(model(ids).logits), ref(torch.tensor(ids)).logits.numpy(),
                                   atol=4e-6, rtol=3e-4)
        assert model.checkpoint.stats['read_calls'] > 30
        assert model.reader.cache.bytes == 0


def test_llama_interruption_rolls_back_prefix_and_can_retry(tmp_path):
    root = tmp_path / 'llama'
    save_tiny(root)
    with AirLLMNextLlamaModel(root, next_config=options(tmp_path)) as model:
        cache = model([[1, 7]]).past_key_values
        original = model.mlps[1]
        def interrupted(x):
            raise OSError('simulated removable-storage interruption')
        model.mlps[1] = interrupted
        with pytest.raises(OSError):
            model([[8]], past_key_values=cache)
        assert cache.position == 2
        model.mlps[1] = original
        recovered = np.array(model([[8]], past_key_values=cache).logits)
        fresh = np.array(model([[1, 7, 8]], logits_to_keep=1).logits)
        np.testing.assert_allclose(recovered, fresh, atol=4e-6, rtol=3e-4)
        cache.truncate(1)
        assert cache.position == 1


def test_llama_budget_and_dispatch(tmp_path):
    root = tmp_path / 'llama'
    save_tiny(root)
    with AutoModel.from_pretrained(root, backend='next', next_config=options(tmp_path, kv_bytes=1)) as model:
        assert isinstance(model, AirLLMNextLlamaModel)
        with pytest.raises(MemoryError, match='KV budget'):
            model([[1]])


def test_llama_prompt_lookup_is_target_verified(tmp_path):
    from airllm.nextgen.prompt_draft import PromptSpanDraft
    root = tmp_path / 'llama'
    save_tiny(root)
    with AirLLMNextLlamaModel(root, next_config=options(tmp_path, prefill_chunk=16)) as model:
        ids = [[1, 7, 8, 9, 1, 7, 8]]
        baseline = model.generate(ids, max_new_tokens=8, eos_token_id=[]).tolist()
        actual = model.generate(ids, max_new_tokens=8, eos_token_id=[], draft_model=PromptSpanDraft(min_match=2)).tolist()
        assert actual == baseline
        assert model.metrics()['runtime']['speculative']['mode'] == 'greedy_block_verification'


@pytest.mark.parametrize('update', [
    {'model_type': 'mistral'}, {'attention_bias': True},
    {'quantization_config': {'quant_method': 'modelopt'}},
    {'rope_parameters': {'rope_type': 'llama3'}},
    {'partial_rotary_factor': .5}, {'pretraining_tp': 2},
    {'num_key_value_heads': 3}, {'rope_parameters': {'rope_theta': float('nan')}},
])
def test_llama_unsupported_semantics_fail_explicitly(tmp_path, update):
    ref = save_tiny(tmp_path / 'llama')
    with pytest.raises((ValueError, NotImplementedError)):
        normalize_llama_config(ref.config.to_dict() | update)
