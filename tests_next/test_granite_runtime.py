"""Independent official Granite equations, checkpoint aliases and bounded reads."""
import json
import numpy as np
import pytest
import torch
import mlx.core as mx
from safetensors.torch import save_file
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

from test_native import options
from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel, granite_route, normalize_granite_config


def test_granite_explicit_public_export():
    from airllm.nextgen import AirLLMNextGraniteMoeModel as exported
    assert exported is AirLLMNextGraniteMoeModel


def checkpoint(root, *, legacy=False, tied=True, weight_dtype=torch.float32):
    torch.manual_seed(1947)
    cfg = GraniteMoeConfig(hidden_size=16, intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=6,
        num_experts_per_tok=2, vocab_size=41, max_position_embeddings=128,
        embedding_multiplier=1.7, residual_multiplier=.4, attention_multiplier=.17,
        logits_scaling=2.3, tie_word_embeddings=tied, eos_token_id=None, pad_token_id=0)
    cfg._attn_implementation = 'eager'
    ref = GraniteMoeForCausalLM(cfg).eval().to(dtype=weight_dtype)
    root.mkdir()
    cfg.save_pretrained(root)
    state = {}
    for name, value in ref.state_dict().items():
        if tied and name == 'lm_head.weight':
            continue
        if legacy:
            name = name.replace('experts.gate_up_proj', 'input_linear.weight')
            name = name.replace('experts.down_proj', 'output_linear.weight')
            name = name.replace('block_sparse_moe.router.weight', 'block_sparse_moe.router.layer.weight')
        state[name] = value.contiguous().clone()
    save_file(state, str(root/'model.safetensors'))
    return ref.float()


@pytest.mark.parametrize('legacy,tied', [(False, True), (True, False)])
@pytest.mark.parametrize('prefill_chunk', [1, 4])
def test_granite_logits_routes_cached_decode_and_generation(tmp_path, legacy, tied, prefill_chunk):
    root = tmp_path/'granite'
    ref = checkpoint(root, legacy=legacy, tied=tied)
    ids = torch.tensor([[1, 7, 8, 9]])
    routes = {}
    handles = [layer.block_sparse_moe.router.register_forward_hook(
        lambda module, inputs, output, i=i: routes.__setitem__(i, output[0].detach().numpy()))
        for i, layer in enumerate(ref.model.layers)]
    with torch.inference_mode(), AirLLMNextGraniteMoeModel(root,
            next_config=options(tmp_path, prefill_chunk=prefill_chunk)) as model:
        wanted = ref(ids, use_cache=True)
        expected_routes = {i: v.copy() for i,v in routes.items()}
        actual = model(ids)
        np.testing.assert_allclose(np.asarray(actual.logits), wanted.logits.numpy(), atol=4e-6, rtol=3e-4)
        for i, mlp in enumerate(model.mlps):
            np.testing.assert_array_equal(np.sort(mlp.last_route_ids, axis=-1),
                np.sort(expected_routes[i][-prefill_chunk:], axis=-1))
        token = wanted.logits[:, -1].argmax(-1)[:, None]
        got = model(token, past_key_values=actual.past_key_values)
        want = ref(token, past_key_values=wanted.past_key_values)
        np.testing.assert_allclose(np.asarray(got.logits), want.logits.numpy(), atol=4e-6, rtol=3e-4)
        assert model.generate(ids, max_new_tokens=6, eos_token_id=[]).tolist() == ref.generate(
            ids, max_new_tokens=6, do_sample=False, eos_token_id=[]).tolist()
        assert model.reader.cache.peak_bytes <= model.options.cache_bytes
        assert model.reader.io.peak_bytes <= model.options.io_bytes
    for handle in handles:
        handle.remove()


def test_granite_reads_selected_expert_slices_and_speculation_is_exact(tmp_path):
    root = tmp_path/'granite'
    checkpoint(root)
    configs = [options(tmp_path, cache_bytes=0, prefetch=False),
               options(tmp_path, cache_bytes=0, prefetch=True, spec_prefetch=True,
                       spec_prefetch_bytes=4096, spec_prefetch_experts=1)]
    sequences = []
    for cfg in configs:
        with AirLLMNextGraniteMoeModel(root, next_config=cfg) as model:
            reads = []
            original = model.checkpoint.read_array
            def capture(key, **kwargs):
                reads.append((key, kwargs.get('expert')))
                return original(key, **kwargs)
            model.checkpoint.read_array = capture
            model([[7]], logits_to_keep=1)
            for mlp in model.mlps:
                expert_reads = [(key,e) for key,e in reads if key in (mlp.gate_up, mlp.down)]
                assert {e for key,e in expert_reads} == set(mlp.last_route_ids.flat)
                assert all(e is not None for key,e in expert_reads)
            sequences.append(model.generate([[7]], max_new_tokens=5, eos_token_id=[]).tolist())
            assert model.reader.cache.bytes == 0
            assert model.reader.io.peak_bytes <= cfg.io_bytes
    assert sequences[0] == sequences[1]


def test_granite_refuses_topk_boundary_ties():
    with pytest.raises(ValueError, match='boundary tie'):
        granite_route(mx, mx.ones((1, 4)), mx.zeros((6, 4)),
                      {'num_local_experts': 6, 'num_experts_per_tok': 2})


def test_granite_original_bfloat16_expert_slices_execute_in_fp32(tmp_path):
    root = tmp_path/'granite-bf16'
    ref = checkpoint(root, legacy=True, weight_dtype=torch.bfloat16)
    with torch.inference_mode(), AirLLMNextGraniteMoeModel(root, next_config=options(tmp_path)) as model:
        ids = [[1,7,8]]
        actual = np.asarray(model(ids).logits)
        expected = ref(torch.tensor(ids)).logits.numpy()
        np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=3e-4)
        assert model.checkpoint.tensors[model.mlps[0].gate_up]['dtype'] == 'BF16'


@pytest.mark.parametrize('update', [
    {'model_type': 'granitemoeshared'}, {'num_experts_per_tok': 7},
    {'num_local_experts': 0}, {'logits_scaling': 0}, {'embedding_multiplier': float('nan')},
    {'attention_multiplier': True}, {'residual_multiplier': -1},
    {'quantization_config': {'bits': 4}}, {'rope_scaling': {'rope_type': 'linear', 'factor': 2}},
])
def test_granite_rejects_unimplemented_semantics_before_opening_weights(tmp_path, update):
    root = tmp_path/'granite'
    checkpoint(root)
    cfg = json.loads((root/'config.json').read_text()) | update
    with pytest.raises((ValueError, NotImplementedError)):
        normalize_granite_config(cfg)


def test_granite_layout_and_kv_budget_fail_explicitly(tmp_path):
    root = tmp_path/'granite'
    checkpoint(root)
    with AirLLMNextGraniteMoeModel(root, next_config=options(tmp_path, kv_bytes=1)) as model:
        with pytest.raises(MemoryError, match='KV budget'):
            model([[1]])
    cfg = json.loads((root/'config.json').read_text())
    cfg['num_local_experts'] = 7
    (root/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='Granite weight'):
        AirLLMNextGraniteMoeModel(root, next_config=options(tmp_path))


@pytest.mark.parametrize('legacy', [False, True])
def test_official_streamed_reference_matches_full_official_model_and_releases_layers(tmp_path, legacy):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('benchmark_granite_moe',
        Path(__file__).resolve().parents[1]/'benchmarks/benchmark_granite_moe.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    root = tmp_path/'granite'
    original = checkpoint(root, legacy=legacy)
    streamed, routes, mapping, stats, handles = benchmark.build_official_streamed_reference(root)
    ids = torch.tensor([[1,7,8]])
    with torch.inference_mode():
        expected = original(ids, use_cache=True, logits_to_keep=1)
        actual = streamed(ids, use_cache=True, logits_to_keep=1)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        token = expected.logits[:, -1].argmax(-1)[:, None]
        expected = original(token, past_key_values=expected.past_key_values, logits_to_keep=1)
        actual = streamed(token, past_key_values=actual.past_key_values, logits_to_keep=1)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    assert all(p.device.type == 'meta' for p in streamed.parameters())
    assert len(routes) == 2
    assert stats['groups_loaded'] == 10
    assert mapping['lm_head.weight'] == 'model.embed_tokens.weight'
    for handle in handles:
        handle.remove()
