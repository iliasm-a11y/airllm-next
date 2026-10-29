"""Captured AJ headers: layout validation, never a fabricated weight file."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airllm.nextgen.gguf_glm5 import NativeGGUFGLM5Model, gguf_glm5_config
from airllm.nextgen.glm5_next_tokenizer import GLM5NextGGUFTokenizer
from test_glm5_next_tokenizer import fixture_metadata


ROOT = Path(__file__).resolve().parents[1] / 'research'


def captured_checkpoint(folder):
    root = ROOT / folder
    metadata = json.loads((root / 'checkpoint-architecture-metadata.json').read_text())
    tokenizer = root / 'checkpoint-tokenizer-metadata.json'
    if tokenizer.exists():
        metadata.update(json.loads(tokenizer.read_text()))
    directory = json.loads((root / 'checkpoint-tensor-directory.json').read_text())
    checkpoint = SimpleNamespace(metadata=metadata, tensors={
        t['name']: SimpleNamespace(shape=tuple(t['dimensions'][::-1]))
        for t in directory['tensors']})
    return checkpoint, directory


def test_captured_aj_layout_and_configuration_match_existing_engine():
    aj, directory = captured_checkpoint('ronin_aj_iq3xxs')
    q2, old_directory = captured_checkpoint('ronin_glm53')
    assert aj.metadata['general.architecture'] == 'glm5next'
    config = gguf_glm5_config(aj)
    assert config == gguf_glm5_config(q2)
    assert {k: v.shape for k, v in aj.tensors.items()} == {
        k: v.shape for k, v in q2.tensors.items()}
    model = NativeGGUFGLM5Model.__new__(NativeGGUFGLM5Model)
    model.config, model.checkpoint = config, aj
    model._validate_layout()
    assert len(directory['tensors']) == len(old_directory['tensors']) == 1383
    assert Counter(t['type_id'] for t in directory['tensors']) == {
        0: 704, 8: 214, 14: 339, 18: 34, 21: 8, 22: 84}
    edge_down = {int(t['name'].split('.')[1]) for t in directory['tensors']
                 if t['type_id'] == 21}
    assert edge_down == {3, 4, 5, 6, 41, 42, 43, 44}
    assert all(t['name'].endswith('ffn_down_exps.weight')
               for t in directory['tensors'] if t['type_id'] in (18, 21))
    assert config['eos_token_id'] == [154820, 154827, 154829]


@pytest.mark.parametrize('patch, error, message', [
    ({'glm5-next.block_count': 45}, ValueError, 'Mixed'),
    ({'glm5next.vocab_size': 32}, ValueError, 'vocabulary'),
    ({'glm5next.expert_shared_feed_forward_length': 4096}, NotImplementedError, 'shared-expert'),
    ({'glm5next.nextn_predict_layers': 1}, NotImplementedError, 'MTP'),
    ({'glm5next.nextn_predict_layers': False}, NotImplementedError, 'MTP'),
    ({'general.architecture': 'glm_moe_dsa'}, NotImplementedError, 'architecture'),
])
def test_aj_metadata_does_not_silently_alias_other_models(patch, error, message):
    checkpoint, _ = captured_checkpoint('ronin_aj_iq3xxs')
    checkpoint.metadata = copy.deepcopy(checkpoint.metadata) | patch
    with pytest.raises(error, match=message):
        gguf_glm5_config(checkpoint)


def test_actual_aj_template_is_used_with_its_native_options():
    captured = json.loads((ROOT / 'ronin_aj_iq3xxs' /
                           'checkpoint-tokenizer-metadata.json').read_text())
    template = captured['tokenizer.chat_template']
    assert hashlib.sha256(template.encode()).hexdigest() == (
        'a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724')
    # A small vocabulary tests rendering only, not real-vocabulary token parity.
    metadata = fixture_metadata() | {'tokenizer.chat_template': template}
    metadata.pop('tokenizer.ggml.add_bos_token')
    metadata.pop('tokenizer.ggml.add_eos_token')
    tokenizer = GLM5NextGGUFTokenizer(metadata)
    messages = [{'role': 'user', 'content': 'Bonjour'}]
    assert tokenizer.apply_chat_template(messages, tokenize=False) == (
        '[gMASK]<sop><|system|>Reasoning Effort: Max'
        '<|user|>Bonjour<|assistant|><think>')
    assert 'Reasoning Effort: Low' in tokenizer.apply_chat_template(
        messages, tokenize=False, reasoning_effort='low')
