"""Validate the independently streamed HF reference against ordinary HF execution."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM

_spec=importlib.util.spec_from_file_location('v5_large_reference',Path(__file__).with_name('benchmark_large.py'))
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
StreamedTransformersLlama=_module.StreamedTransformersLlama


def test_streamed_hf_reference_prefill_and_cached_decode(tmp_path):
    torch.manual_seed(12031);torch.set_num_threads(1)
    config=LlamaConfig(vocab_size=48,hidden_size=32,intermediate_size=64,
                       num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                       max_position_embeddings=64,attention_dropout=0.,tie_word_embeddings=False)
    config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval();model.save_pretrained(tmp_path)
    streamed=StreamedTransformersLlama(tmp_path)
    cache=None
    with torch.inference_mode():
        for values in ([[1,7,9]],[[8]],[[3,4]]):
            expected=model(torch.tensor(values),past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=expected.past_key_values
            actual=streamed.forward(values)
            torch.testing.assert_close(actual,expected.logits,atol=2e-6,rtol=2e-5)
            assert streamed.position==cache.get_seq_length()
