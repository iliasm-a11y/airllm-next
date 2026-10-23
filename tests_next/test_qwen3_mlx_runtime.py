"""Independent numerical checks; tiny random fixtures are not trained models."""
import json
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
torch = pytest.importorskip('torch')
from safetensors.numpy import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from airllm.nextgen.config import NextConfig
from airllm.nextgen.qwen3_mlx_runtime import ExperimentalQwen3MLXModel, normalize_qwen3_mlx_config


def fixture_checkpoint(root, bits=6, group_size=32, bf16_scales=False):
    mx.set_default_device(mx.cpu)
    torch.manual_seed(7324)
    c = Qwen3MoeConfig(hidden_size=128, intermediate_size=128, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
        num_local_experts=4, num_experts_per_tok=2, norm_topk_prob=True, vocab_size=128,
        decoder_sparse_step=1, mlp_only_layers=[], tie_word_embeddings=False,
        max_position_embeddings=256)
    reference = Qwen3MoeForCausalLM(c).eval()
    reference.config._attn_implementation = 'eager'
    config = c.to_dict()
    config.update(num_experts=4, quantization={'bits': bits, 'group_size': group_size})
    (root/'config.json').write_text(json.dumps(config))
    arrays, decoded = {}, {}

    def pack(name, value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim >= 2:
            w, s, b = mx.quantize(mx.array(value), bits=bits, group_size=group_size)
            if bf16_scales:
                s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
            mx.eval(w, s, b)
            arrays[name] = np.array(w)
            arrays[name[:-7]+'.scales'] = s if bf16_scales else np.array(s)
            arrays[name[:-7]+'.biases'] = b if bf16_scales else np.array(b)
            unquantized = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32),
                bits=bits, group_size=group_size)
            mx.eval(unquantized)
            return np.array(unquantized)
        if bf16_scales:
            rounded = mx.array(value).astype(mx.bfloat16)
            mx.eval(rounded)
            arrays[name] = rounded
            value = np.array(rounded.astype(mx.float32))
        else:
            arrays[name] = value
        return value

    for key, value in reference.state_dict().items():
        value = value.detach().numpy()
        if key.endswith('.experts.gate_up_proj'):
            prefix = key[:-len('experts.gate_up_proj')]+'switch_mlp.'
            gate, up = np.split(value, 2, axis=1)
            gate = pack(prefix+'gate_proj.weight', gate)
            up = pack(prefix+'up_proj.weight', up)
            decoded[key] = np.concatenate([gate, up], axis=1)
        elif key.endswith('.experts.down_proj'):
            prefix = key[:-len('experts.down_proj')]+'switch_mlp.'
            decoded[key] = pack(prefix+'down_proj.weight', value)
        else:
            decoded[key] = pack(key, value)
    reference.load_state_dict({key: torch.from_numpy(value) for key, value in decoded.items()})
    mx.save_safetensors(str(root/'model.safetensors'), {k: mx.array(v) for k, v in arrays.items()})
    return reference, config


@pytest.mark.parametrize('bits,group_size,bf16_scales', [(4, 32, False), (6, 32, False),
    (8, 32, False), (6, 128, True)])
def test_quantized_qwen3_logits_and_incremental_generation(tmp_path, bits, group_size, bf16_scales):
    reference, _ = fixture_checkpoint(tmp_path, bits, group_size, bf16_scales)
    options = NextConfig(cache_bytes=2*1024**2, io_bytes=1024**2, kv_bytes=4*1024**2,
        workspace_bytes=8*1024**2, reserve_bytes=0, max_tensor_bytes=1024**2,
        device='cpu', dtype='float32', load_tokenizer=False, use_saved_tuning=False,
        cache_dir=str(tmp_path/'metadata'), prefetch=True, prefill_chunk=3, max_context=64,
        row_chunk=16)
    model = ExperimentalQwen3MLXModel(tmp_path, next_config=options)
    requests = []
    original = model.checkpoint.read_arrays

    def observed(items, **kwargs):
        requests.extend(items)
        return original(items, **kwargs)
    model.checkpoint.read_arrays = observed
    try:
        tokens = [11, 3, 29]
        cache = model.new_session()
        for step in range(5):
            selected = tokens if step == 0 else [tokens[-1]]
            actual = model.forward(np.array([selected]), past_key_values=cache).logits
            mx.eval(actual)
            with torch.no_grad():
                expected = reference(torch.tensor([tokens])).logits.numpy()[:, -len(selected):]
            np.testing.assert_allclose(np.array(actual), expected, atol=3e-5, rtol=4e-4)
            greedy = int(np.argmax(np.array(actual)[0, -1]))
            assert greedy == int(expected[0, -1].argmax())
            tokens.append(greedy)
        assert cache.position == 7
        # The complete bank is available on disk but never requested in RAM.
        expert_reads = [item for item in requests if '.switch_mlp.' in item['key']]
        assert expert_reads and all(item['expert'] is not None for item in expert_reads)
        assert model.reader.cache.bytes <= options.cache_bytes
        assert model.checkpoint.stats['quantized_logical_bytes'] > 0
    finally:
        model.close(save_profile=False)


@pytest.mark.parametrize('change', [
    {'model_type': 'qwen3_5_moe_text'},
    {'quantization': {'bits': 3, 'group_size': 32}},
    {'quantization': {'bits': 6, 'group_size': 32, 'mode': 'mxfp4'}},
    {'norm_topk_prob': False},
    {'mlp_only_layers': [0]},
])
def test_reject_unsupported_variants(tmp_path, change):
    _, config = fixture_checkpoint(tmp_path)
    config.update(change)
    with pytest.raises((ValueError, NotImplementedError)):
        normalize_qwen3_mlx_config(config)


def test_affine_layout_rejects_wrong_scale_shape(tmp_path):
    from safetensors.numpy import load_file
    fixture_checkpoint(tmp_path)
    arrays = load_file(tmp_path/'model.safetensors')
    arrays['model.layers.0.mlp.switch_mlp.up_proj.scales'] = np.ones((4, 128, 3), np.float32)
    save_file(arrays, tmp_path/'model.safetensors')
    with pytest.raises(ValueError, match='affine scales'):
        ExperimentalQwen3MLXModel(tmp_path, next_config=NextConfig(load_tokenizer=False,
            device='cpu', reserve_bytes=0, cache_dir=str(tmp_path/'meta')))
