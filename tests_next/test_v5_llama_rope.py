"""Independent Torch parity for the narrow full-head linear Llama RoPE path."""
import json

import numpy as np
import pytest
import torch

from airllm.nextgen.adapters import validated_default_rope, validated_llama_rope
from airllm.nextgen.llama_runtime import AirLLMNextLlamaModel
from test_native import options


@pytest.mark.parametrize('spelling', ['current', 'legacy', 'both'])
def test_linear_llama_prefill_cached_decode_and_generation(tmp_path, spelling):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(9817)
    params = {'rope_type': 'linear', 'rope_theta': 100000., 'factor': 4.}
    cfg = LlamaConfig(vocab_size=64, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, original_max_position_embeddings=8,
        rope_parameters=params, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_bias=False, mlp_bias=False)
    ref = LlamaForCausalLM(cfg).eval()
    root = tmp_path/'linear'
    ref.save_pretrained(root)
    local = json.loads((root/'config.json').read_text())
    if spelling != 'current':
        local['rope_scaling'] = {'type': 'linear', 'factor': 4.}
        local['rope_theta'] = 100000.
    if spelling == 'legacy':
        local.pop('rope_parameters')
    (root/'config.json').write_text(json.dumps(local))
    # Exercise scaled offsets beyond the declared original 8-position context.
    ids = torch.tensor([[1, *range(4, 35)]])
    assert ids.shape[1] > cfg.original_max_position_embeddings
    native_cfg = options(tmp_path, prefill_chunk=5, max_context=64)
    with AirLLMNextLlamaModel(root, next_config=native_cfg) as model, torch.inference_mode():
        expected = ref(ids, use_cache=True)
        actual = model(ids)
        np.testing.assert_allclose(np.array(actual.logits), expected.logits.numpy(), atol=4e-6, rtol=3e-4)
        for token in (35, 36, 37):
            next_ids = torch.tensor([[token]])
            actual = model(next_ids, past_key_values=actual.past_key_values)
            expected = ref(next_ids, past_key_values=expected.past_key_values)
            np.testing.assert_allclose(np.array(actual.logits), expected.logits.numpy(), atol=4e-6, rtol=3e-4)
        wanted = ref.generate(ids, attention_mask=torch.ones_like(ids),
                              max_new_tokens=8, do_sample=False, eos_token_id=[]).tolist()
        assert model.generate(ids, max_new_tokens=8, eos_token_id=[]).tolist() == wanted


@pytest.mark.parametrize('update', [
    {'rope_parameters': {'rope_type': 'linear'}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 0}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': float('nan')}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': float('inf')}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': True}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 10**400}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 4, 'type': 'default'}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 4},
     'rope_scaling': {'type': 'linear', 'factor': 2}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': True},
     'rope_scaling': {'type': 'linear', 'factor': 1}},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 4, 'rope_theta': 1000}, 'rope_theta': 10000},
    {'rope_parameters': {'rope_type': 'linear', 'factor': 4, 'attention_factor': 2}},
    {'rope_parameters': {'rope_type': 'yarn', 'factor': 4}},
    {'rope_scaling': 'linear'},
])
def test_linear_llama_ambiguous_or_unsupported_rope_fails(update):
    with pytest.raises((ValueError, NotImplementedError)):
        validated_llama_rope(update)


def test_linear_alias_normalization_and_other_architecture_guards():
    legacy = {'rope_scaling': {'type': 'linear', 'factor': 4.}, 'rope_theta': 100000}
    assert validated_llama_rope(legacy) == {
        'rope_type': 'linear', 'rope_theta': 100000., 'factor': 4.}
    for config in (legacy, {'rope_parameters': {'rope_type': 'linear', 'factor': 4}}):
        with pytest.raises(NotImplementedError):
            validated_default_rope(config)
