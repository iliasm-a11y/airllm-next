from concurrent.futures import ThreadPoolExecutor
import json
import pytest
import torch
from safetensors.torch import save_file
from conftest import save_tiny
from test_runtime import options
from airllm.ultra import AirLLMUltraModel


@pytest.mark.parametrize('kind', ['mixtral', 'glm'])
def test_fused_3d_checkpoint_matches_reference(tmp_path, kind):
    root = tmp_path / 'fused'
    ref = save_tiny(root, kind)
    # Save raw native state rather than Transformers' backwards-compatible
    # save_pretrained conversion. This exercises on-disk first-axis slicing.
    for path in root.glob('*.safetensors*'):
        path.unlink()
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, root / 'model.safetensors')
    ids = torch.tensor([[1, 7, 8], [1, 8, 9]])
    with AirLLMUltraModel(root, device='cpu', dtype=torch.float32, ultra=options(tmp_path)) as model, torch.inference_mode():
        assert all(source.fused for source in model.expert_sources.values())
        torch.testing.assert_close(model(ids).logits, ref(ids).logits, rtol=4e-5, atol=4e-6)


def test_forward_failure_releases_materialized_parameters(tmp_path):
    root = tmp_path / 'failure'
    save_tiny(root)
    with AirLLMUltraModel(root, device='cpu', ultra=options(tmp_path, prefetch=False)) as model:
        layer = model.model.model.layers[0]
        original = layer.forward
        def fail(*args, **kwargs):
            raise RuntimeError('injected compute error')
        layer.forward = fail
        with pytest.raises(RuntimeError, match='injected'):
            model(torch.tensor([[1]]))
        assert all(p.device.type == 'meta' for p in model.model.parameters())
        layer.forward = original
        assert model(torch.tensor([[1]])).logits.isfinite().all()


def test_concurrent_requests_are_serialized(tmp_path):
    root = tmp_path / 'concurrency'
    save_tiny(root)
    with AirLLMUltraModel(root, device='cpu', ultra=options(tmp_path)) as model:
        def run(_):
            return model(torch.tensor([[1, 7, 8]])).logits.clone()
        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(run, range(4)))
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0])
