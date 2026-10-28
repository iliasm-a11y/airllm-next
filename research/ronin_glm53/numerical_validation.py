"""Reproduce synthetic GLM5-Next numerical evidence without real model weights.

Run from the repository root: python research/ronin_glm53/numerical_validation.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MLX_ENABLE_TF32', '0')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'air_llm'), str(ROOT / 'tests_next')]

import mlx.core as mx
import numpy as np
import torch
import transformers
from transformers.models.glm5_next import modeling_glm5_next

from airllm.nextgen.gguf_glm5 import NativeGGUFGLM5Model
from test_glm53_runtime import export_glm53_fixture

torch.set_num_threads(1)
mx.set_default_device(mx.cpu)


def error(got, expected):
    difference = np.asarray(got, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        'max_abs': float(np.max(np.abs(difference))),
        'max_relative_denominator_floor_1e-8': float(np.max(np.abs(difference) / np.maximum(np.abs(expected), 1e-8))),
        'relative_l2': float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-12)),
    }


def run_case(path, name, layers=5, experts=4, top_k=2, quantized=False, sparse_dsa=False):
    path.mkdir()
    checkpoint, ref, head, options = export_glm53_fixture(path, layers=layers, experts=experts,
                                                       top_k=top_k, quantized=quantized, sparse_dsa=sparse_dsa)
    routes = {}
    handles = []
    def capture(number):
        def hook(module, inputs, output):
            routes[number] = output[2].detach().numpy().tolist()
        return hook
    for index, layer in enumerate(ref.layers):
        if hasattr(layer.mlp, 'gate'):
            handles.append(layer.mlp.gate.register_forward_hook(capture(index)))
    prompt = torch.tensor([[1, 8, 12, 5]]) if layers == 45 else torch.tensor([[1, 8, 12, 5, 21, 6, 4]])
    follow_ids = torch.tensor([[3]]) if layers == 45 else torch.tensor([[11, 3]])
    with torch.inference_mode():
        expected = ref(prompt, use_cache=True)
        prefill_logits = head(expected.last_hidden_state).numpy()
    prefill_routes = {number: indices[-1] for number, indices in routes.items()}
    with NativeGGUFGLM5Model(checkpoint, options) as model:
        actual = model.forward(prompt.numpy())
        actual_routes = dict(model.last_routes)
        prefill_error = error(np.asarray(actual.logits), prefill_logits)
        prefill_sets_equal = all(set(prefill_routes[layer]) == set(actual_routes[layer]) for layer in prefill_routes)
        with torch.inference_mode():
            expected_follow = ref(follow_ids, use_cache=True, past_key_values=expected.past_key_values)
            decode_logits = head(expected_follow.last_hidden_state).numpy()
        decode_routes = {number: indices[-1] for number, indices in routes.items()}
        follow = model.forward(follow_ids.numpy(), past_key_values=actual.past_key_values)
        native_decode_routes = dict(model.last_routes)
        decode_error = error(np.asarray(follow.logits), decode_logits)
        decode_sets_equal = all(set(decode_routes[layer]) == set(native_decode_routes[layer]) for layer in decode_routes)
        atol, rtol = (5e-5, 4e-4) if quantized else (4e-5, 4e-4) if layers == 45 else (2e-5, 2e-4)
        result = {
            'name': name, 'kind': 'synthetic_miniature_only', 'device': model.device,
            'layers': layers, 'kda_layers': ref.config.layer_types.count('linear_attention'),
            'dsa_layers': ref.config.layer_types.count('indexed_attention'),
            'hidden_size': ref.config.hidden_size, 'routed_experts': experts, 'top_k': top_k,
            'dsa_index_topk': ref.config.index_topk, 'dsa_kpool_size': ref.config.index_kpool,
            'dsa_last_prefill_complete_pools': prompt.shape[1] // ref.config.index_kpool,
            'dsa_last_decode_complete_pools': (prompt.shape[1] + follow_ids.shape[1]) // ref.config.index_kpool,
            'dsa_selected_complete_pools_max': ref.config.index_topk // ref.config.index_kpool,
            'dsa_complete_pool_pruning_exercised': sparse_dsa,
            'checkpoint_tensor_types': sorted({int(t.quant_type) for t in model.checkpoint.tensors.values()}),
            'prefill_tokens': prompt.shape[1], 'decode_tokens': follow_ids.shape[1],
            'tolerances': {'atol': atol, 'rtol': rtol},
            'prefill_logits_error': prefill_error, 'cached_decode_logits_error': decode_error,
            'prefill_allclose': bool(np.allclose(np.asarray(actual.logits), prefill_logits, atol=atol, rtol=rtol)),
            'decode_allclose': bool(np.allclose(np.asarray(follow.logits), decode_logits, atol=atol, rtol=rtol)),
            'last_prefill_routed_id_sets_equal': prefill_sets_equal,
            'last_decode_routed_id_sets_equal': decode_sets_equal,
            'routing_order_bitwise_equality_claimed': False,
            'last_prefill_reference_routes': prefill_routes,
            'last_prefill_native_routes': actual_routes,
            'last_decode_reference_routes': decode_routes,
            'last_decode_native_routes': native_decode_routes,
            'routed_expert_evaluations': model.stats['routed_expert_evaluations'],
            'expected_routed_expert_evaluations': (layers - 3) * top_k * (prompt.shape[1] + follow_ids.shape[1]),
            'cache_peak_bytes': model.reader.cache.peak_bytes,
            'io_peak_reserved_bytes': model.checkpoint.stats['peak_reserved_bytes'],
            'session_peak_bytes': actual.past_key_values.peak_bytes,
        }
        assert result['prefill_allclose'] and result['decode_allclose']
        assert prefill_sets_equal and decode_sets_equal
        assert result['routed_expert_evaluations'] == result['expected_routed_expert_evaluations']
    for handle in handles:
        handle.remove()
    return result


def main():
    source = Path(inspect.getfile(modeling_glm5_next))
    evidence = {
        'scope': 'Synthetic structure and same-decoded-quantization validation only; no trained GLM-5.3 checkpoint generation or Metal/M5 performance measured.',
        'transformers_version': transformers.__version__,
        'transformers_installed_modeling_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'primitive_oracle_source_revision': 'huggingface/transformers@1b754e9a4636191852a1618fc588ad5c863ab4d5',
        'llama_cpp_layout_source_revision': 'ggml-org/llama.cpp@74a4fb0037243f1b6196d0f990ff0aff031356cf',
        'torch_version': torch.__version__, 'mlx_version': importlib.metadata.version('mlx'),
        'gguf_version': importlib.metadata.version('gguf'), 'cases': [],
    }
    with tempfile.TemporaryDirectory(prefix='ronin-glm53-numerical-') as directory:
        path = Path(directory)
        evidence['cases'].append(run_case(path / 'f32', 'five_layer_f32'))
        evidence['cases'].append(run_case(path / 'quant', 'five_layer_mixed_q2_k_q8_0', quantized=True))
        evidence['cases'].append(run_case(path / '45layers', '45_layer_288_expert_top8_f32', layers=45, experts=288, top_k=8))
        evidence['cases'].append(run_case(path / 'sparse_dsa', 'five_layer_f32_sparse_kpool', sparse_dsa=True))
    destination = Path(__file__).with_name('numerical-validation.json')
    destination.write_text(json.dumps(evidence, indent=2) + '\n')
    print(destination)
    for case in evidence['cases']:
        print(case['name'], 'maxabs prefill', case['prefill_logits_error']['max_abs'],
              'decode', case['cached_decode_logits_error']['max_abs'], 'routes',
              case['last_prefill_routed_id_sets_equal'] and case['last_decode_routed_id_sets_equal'])


if __name__ == '__main__':
    main()
