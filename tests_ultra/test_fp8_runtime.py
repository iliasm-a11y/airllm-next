import json
import math
import torch
from safetensors.torch import load_file, save_file
from conftest import save_tiny
from test_runtime import options
from airllm.ultra import AirLLMUltraModel


def test_glm_fp8_unfused_checkpoint_end_to_end(tmp_path):
    root = tmp_path / 'glm-fp8'
    ref = save_tiny(root, 'glm')
    # Build an actual block-FP8 checkpoint, using the same per-expert key layout
    # as the official 5.3 index, at tiny dimensions. Unit scales make an exact
    # dequantized reference possible without unsupported CPU FP8 matmul.
    with torch.no_grad():
        for p in ref.parameters():
            if p.ndim >= 2:
                p.copy_(p.to(torch.float8_e4m3fn).float())
    ref.save_pretrained(root, max_shard_size='100KB')
    index_path = root / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    for file in sorted(set(index['weight_map'].values())):
        state = load_file(root / file)
        out = {}
        for key, value in state.items():
            if value.ndim == 2:
                out[key] = value.to(torch.float8_e4m3fn)
                sk = key + '_scale_inv'
                out[sk] = torch.ones(math.ceil(value.shape[0]/16), math.ceil(value.shape[1]/16))
                index['weight_map'][sk] = file
            else:
                out[key] = value
        save_file(out, root / file)
    index_path.write_text(json.dumps(index))
    config_path = root / 'config.json'
    cfg = json.loads(config_path.read_text())
    cfg['quantization_config'] = {'quant_method': 'fp8', 'weight_block_size': [16, 16],
                                  'fmt': 'e4m3', 'activation_scheme': 'dynamic'}
    config_path.write_text(json.dumps(cfg))
    ids = torch.tensor([[1, 7, 8]])
    with AirLLMUltraModel(root, device='cpu', dtype=torch.float32,
                         ultra=options(tmp_path)) as model, torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, ref(ids).logits, rtol=3e-5, atol=3e-6)
        assert torch.equal(model.generate(ids, max_new_tokens=4), ref.generate(ids, max_new_tokens=4))
        assert model.expert_sources
