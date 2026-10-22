import sys
import pytest
import torch
from conftest import save_tiny
from airllm.ultra import UltraConfig
from airllm.ultra.mlx_runtime import validate_mlx_config, MlxLlamaUltra


@pytest.mark.parametrize('cfg', [
    {'model_type': 'glm_moe_dsa'},
    {'model_type': 'llama', 'quantization_config': {'quant_method': 'fp8'}},
    {'model_type': 'llama', 'rope_scaling': {'rope_type': 'llama3'}},
    {'model_type': 'llama', 'attention_bias': True},
])
def test_unsupported_mlx_configs_are_explicit(cfg):
    with pytest.raises(NotImplementedError):
        validate_mlx_config(cfg)


@pytest.mark.skipif(sys.platform != 'darwin', reason='Apple Silicon/MLX hardware unavailable')
def test_native_mlx_logits_and_decode_match_transformers(tmp_path):
    import mlx.core as mx
    import numpy as np
    root = tmp_path / 'mlx'
    ref = save_tiny(root)
    cfg = UltraConfig(cache_dir=str(tmp_path / 'meta'), load_tokenizer=False, reserve_bytes=0)
    with MlxLlamaUltra(root, ultra=cfg) as model, torch.inference_mode():
        ids = torch.tensor([[1, 8, 9]])
        expected = ref(ids, use_cache=True)
        got, cache = model(mx.array(ids.tolist()))
        np.testing.assert_allclose(np.array(got), expected.logits[:, -1:].numpy(), atol=5e-5, rtol=5e-4)
        token = expected.logits[:, -1].argmax(-1)[:, None]
        expected = ref(token, past_key_values=expected.past_key_values)
        got, _ = model(mx.array(token.tolist()), cache=cache)
        np.testing.assert_allclose(np.array(got), expected.logits.numpy(), atol=5e-5, rtol=5e-4)
