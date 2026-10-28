"""Audit captured real GGUF headers, without fabricating a weight file."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from airllm.nextgen.gguf_glm5 import NativeGGUFGLM5Model, gguf_glm5_config


def test_captured_real_checkpoint_layout_requires_no_payload():
    root = Path(__file__).resolve().parents[1] / 'research' / 'ronin_glm53'
    metadata = json.loads((root / 'checkpoint-architecture-metadata.json').read_text())
    directory = json.loads((root / 'checkpoint-tensor-directory.json').read_text())
    cp = SimpleNamespace(metadata=metadata, tensors={
        t['name']: SimpleNamespace(shape=tuple(t['dimensions'][::-1]))
        for t in directory['tensors']})
    config = gguf_glm5_config(cp)
    model = NativeGGUFGLM5Model.__new__(NativeGGUFGLM5Model)
    model.config, model.checkpoint = config, cp
    model._validate_layout()
    assert config['num_hidden_layers'] == 45
    assert config['layer_types'].count('linear_attention') == 34
    assert config['layer_types'].count('indexed_attention') == 11
    assert config['n_routed_experts'] == 288
    assert config['num_experts_per_tok'] == 8
    assert config['index_kpool'] == 4
    assert set(config['eos_token_id']) == {154820, 154827, 154829}
    assert Counter(t['type_id'] for t in directory['tensors']) == {
        0: 726, 8: 306, 10: 126, 30: 225}


def test_old_glm_architecture_cannot_enter_new_backend():
    with pytest.raises(NotImplementedError, match='architecture=glm5-next'):
        gguf_glm5_config(SimpleNamespace(metadata={'general.architecture': 'glm_moe_dsa'}))
