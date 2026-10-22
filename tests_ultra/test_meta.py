import pytest
import torch
from conftest import save_tiny
from test_runtime import options
from airllm.ultra import AirLLMUltraModel


def test_meta_factory_never_allocates_expert_bank_on_cpu(tmp_path, monkeypatch):
    root = tmp_path / 'meta'
    save_tiny(root, 'glm')
    original = torch.empty
    sizes = []
    def checked_empty(*args, **kwargs):
        result = original(*args, **kwargs)
        # Native expert tensors have [experts, out, in] shapes. The actual
        # allocation, before nn.Parameter wrapping, must already be on meta.
        if result.ndim == 3:
            sizes.append(tuple(result.shape))
            assert result.device.type == 'meta'
        return result
    monkeypatch.setattr(torch, 'empty', checked_empty)
    with AirLLMUltraModel(root, ultra=options(tmp_path)):
        assert sizes


def test_learned_glm_routing_buffer_is_loaded(tmp_path):
    root = tmp_path / 'router'
    ref = save_tiny(root, 'glm')
    with torch.no_grad():
        for name, buf in ref.named_buffers():
            if name.endswith('e_score_correction_bias'):
                buf.copy_(torch.linspace(-2, 2, buf.numel()))
    ref.save_pretrained(root, max_shard_size='100KB')
    with AirLLMUltraModel(root, ultra=options(tmp_path)) as model, torch.inference_mode():
        ids = torch.tensor([[1, 7, 8]])
        torch.testing.assert_close(model(ids).logits, ref(ids).logits, atol=3e-6, rtol=3e-5)
        # Persistent routing buffers also leave the active device between layers.
        assert all(buf.device.type == 'meta' for name, buf in model.model.named_buffers()
                   if name.endswith('e_score_correction_bias'))
