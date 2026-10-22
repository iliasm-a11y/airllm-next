import pytest
import torch
from conftest import save_tiny
from test_runtime import options
from airllm.ultra import AirLLMUltraModel


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS hardware unavailable')
@pytest.mark.parametrize('kind', ['llama', 'mixtral', 'glm'])
def test_mps_logits_match_cpu_reference(tmp_path, kind):
    root = tmp_path / kind
    ref = save_tiny(root, kind)
    with AirLLMUltraModel(root, device='mps', dtype=torch.float32,
                         ultra=options(tmp_path)) as model, torch.inference_mode():
        ids = torch.tensor([[1, 7, 9]])
        torch.testing.assert_close(model(ids.to('mps')).logits.cpu(), ref(ids).logits,
                                   rtol=1e-3, atol=1e-4)
