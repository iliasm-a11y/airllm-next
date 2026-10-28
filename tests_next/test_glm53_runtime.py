"""Independent miniature GLM5-Next HF -> F32 GGUF end-to-end validation.

The miniature includes dense and sparse KDA plus KPool DSA. It does not test
trained 313B weights, Q2_K end-to-end fidelity, or Apple Metal performance.
"""
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import Glm5NextTextConfig, Glm5NextTextModel

from airllm.nextgen.config import NextConfig
from airllm.nextgen.gguf_glm5 import NativeGGUFGLM5Model
from test_gguf import _fixture


def export_glm53_fixture(tmp_path, *, layers=5, quantized=False, experts=4, top_k=2, sparse_dsa=False):
    """Convert an independently initialized official HF miniature to GGUF F32."""
    torch.manual_seed(5353)
    hidden = 256 if quantized else 8
    c = Glm5NextTextConfig(
        vocab_size=32, hidden_size=hidden, intermediate_size=2 * hidden, moe_intermediate_size=hidden,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=experts, num_experts_per_tok=top_k, n_shared_experts=1,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, v_head_dim=4,
        linear_head_dim=4, linear_num_heads=2, linear_conv_kernel_dim=4,
        index_head_dim=4, index_n_heads=2, index_topk=4 if sparse_dsa else 16, index_kpool=2,
        hc_mult=4, hc_sinkhorn_iters=20, max_position_embeddings=32,
        pad_token_id=None, eos_token_id=None)
    c._attn_implementation = 'eager'
    ref = Glm5NextTextModel(c).eval()
    head = torch.nn.Linear(hidden, 32, bias=False).eval()
    with torch.no_grad():
        head.weight.normal_(std=0.1)
        for name, parameter in ref.named_parameters():
            if name.endswith(('.attn_hc.base', '.ffn_hc.base')):
                parameter.normal_(std=0.05)
            elif name.endswith(('.attn_hc.scale', '.ffn_hc.scale')):
                parameter.copy_(torch.tensor([0.4, 0.5, 0.6]))
            elif name.endswith('forget_gate.A_log'):
                parameter.copy_(torch.tensor([-0.3, 0.2]))
            elif name.endswith('forget_gate.dt_bias'):
                parameter.copy_(torch.linspace(-0.8, 0.4, parameter.numel()))
            elif name.endswith(('index_kpool_compress_ape', 'index_kpool_compress_gate')):
                parameter.normal_(std=0.2)
        for layer in ref.layers:
            if hasattr(layer.mlp, 'gate'):
                bias = torch.tensor([-0.025, 0.03, -0.01, 0.015]) if experts == 4 else torch.linspace(-0.025, 0.03, experts)
                layer.mlp.gate.e_score_correction_bias.copy_(bias)
            if sparse_dsa and hasattr(layer.self_attn, 'indexer'):
                # Two opposite Q heads and equal learned head weights produce
                # weight*abs(q@kpool), avoiding artificial all-ReLU-zero ties.
                # These remain ordinary legal HF weights and exact KPool math.
                indexer = layer.self_attn.indexer
                indexer.wq_b.weight[4:].copy_(-indexer.wq_b.weight[:4])
                indexer.weights_proj.weight[1].copy_(indexer.weights_proj.weight[0])
    # Synthetic mixed-codec fixture: random legal GGML blocks are decoded by
    # upstream gguf into the independent HF weights. No production codec is
    # used to build the expected model, and no trained checkpoint is modified.
    encoded = {}
    quant_rng = np.random.default_rng(5353256)
    if quantized:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize as upstream_dequantize
    def synthetic_blocks(shape, kind):
        block, size = (256, 84) if kind == 10 else (32, 34)
        raw = quant_rng.integers(0, 256, (int(np.prod(shape)) // block, size), dtype=np.uint8)
        if kind == 10:
            raw[:, 80:82] = np.frombuffer(np.float16(0.001).tobytes(), dtype=np.uint8)
            raw[:, 82:84] = np.frombuffer(np.float16(0.0015).tobytes(), dtype=np.uint8)
        else:
            raw[:, :2] = np.frombuffer(np.float16(0.0005).tobytes(), dtype=np.uint8)
        decoded = upstream_dequantize(raw, GGMLQuantizationType(kind)).reshape(shape)
        return decoded, raw.tobytes()
    if quantized:
        with torch.no_grad():
            for name, weight in [('token_embd.weight', ref.embed_tokens.weight), ('output.weight', head.weight)]:
                decoded, raw = synthetic_blocks(tuple(weight.shape), 8)
                weight.copy_(torch.tensor(decoded))
                encoded[name] = (8, raw)
    tensors = {'token_embd.weight': ref.embed_tokens.weight.detach().numpy().copy(),
               'output_norm.weight': ref.norm.weight.detach().numpy().copy(),
               'output.weight': head.weight.detach().numpy().copy()}
    names = {
        'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'self_attn.q_proj.weight': 'attn_q.weight',
        'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'self_attn.forget_gate.f_a_proj.weight': 'ssm_f_a.weight',
        'self_attn.forget_gate.f_b_proj.weight': 'ssm_f_b.weight',
        'self_attn.forget_gate.dt_bias': 'ssm_dt.bias',
        'self_attn.b_proj.weight': 'ssm_beta.weight',
        'self_attn.g_a_proj.weight': 'ssm_g_a.weight',
        'self_attn.g_b_proj.weight': 'ssm_g_b.weight',
        'self_attn.o_norm.weight': 'ssm_norm.weight',
        'self_attn.q_a_proj.weight': 'attn_q_a.weight',
        'self_attn.q_a_layernorm.weight': 'attn_q_a_norm.weight',
        'self_attn.q_b_proj.weight': 'attn_q_b.weight',
        'self_attn.kv_a_proj_with_mqa.weight': 'attn_kv_a_mqa.weight',
        'self_attn.kv_a_layernorm.weight': 'attn_kv_a_norm.weight',
        'self_attn.indexer.wq_b.weight': 'indexer.attn_q_b.weight',
        'self_attn.indexer.wk.weight': 'indexer.attn_k.weight',
        'self_attn.indexer.k_norm.weight': 'indexer.k_norm.weight',
        'self_attn.indexer.k_norm.bias': 'indexer.k_norm.bias',
        'self_attn.indexer.weights_proj.weight': 'indexer.proj.weight',
        'self_attn.indexer.index_kpool_compress_gate': 'indexer_compressor_gate.weight',
        'self_attn.indexer.index_kpool_compress_ape': 'indexer_compressor_ape.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight',
        'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight',
        'mlp.gate.weight': 'ffn_gate_inp.weight',
        'mlp.gate.e_score_correction_bias': 'exp_probs_b.bias',
        'mlp.experts.down_proj': 'ffn_down_exps.weight',
        'mlp.shared_experts.gate_proj.weight': 'ffn_gate_shexp.weight',
        'mlp.shared_experts.up_proj.weight': 'ffn_up_shexp.weight',
        'mlp.shared_experts.down_proj.weight': 'ffn_down_shexp.weight',
        'attn_hc.fn': 'hc_attn_fn.weight', 'attn_hc.base': 'hc_attn_base.weight',
        'attn_hc.scale': 'hc_attn_scale.weight',
        'ffn_hc.fn': 'hc_ffn_fn.weight', 'ffn_hc.base': 'hc_ffn_base.weight',
        'ffn_hc.scale': 'hc_ffn_scale.weight',
    }
    for name, parameter in ref.state_dict().items():
        if not name.startswith('layers.'):
            continue
        _, number, suffix = name.split('.', 2)
        p = f'blk.{number}.'
        array = parameter.detach().numpy().copy()
        if quantized and suffix == 'mlp.experts.gate_up_proj':
            parts = []
            for label, part in zip(('ffn_gate_exps.weight', 'ffn_up_exps.weight'), np.split(array, 2, axis=1)):
                decoded, raw = synthetic_blocks(part.shape, 10)
                parts.append(decoded)
                encoded[p + label] = (10, raw)
            array = np.concatenate(parts, axis=1)
            with torch.no_grad():
                parameter.copy_(torch.tensor(array))
        elif quantized and suffix == 'mlp.experts.down_proj':
            array, raw = synthetic_blocks(array.shape, 10)
            encoded[p + names[suffix]] = (10, raw)
            with torch.no_grad():
                parameter.copy_(torch.tensor(array))
        elif quantized and array.ndim == 2 and array.shape[-1] % 32 == 0 and suffix in names:
            array, raw = synthetic_blocks(array.shape, 8)
            encoded[p + names[suffix]] = (8, raw)
            with torch.no_grad():
                parameter.copy_(torch.tensor(array))
        if suffix == 'self_attn.forget_gate.A_log':
            tensors[p + 'ssm_a'] = -np.exp(array)
        elif suffix == 'self_attn.conv1d.weight':
            for q, part in zip(('q', 'k', 'v'), np.split(array, 3, axis=0)):
                tensors[p + f'ssm_conv1d_{q}.weight'] = part
        elif suffix == 'self_attn.kv_b_proj.weight':
            combined = array.reshape(c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
            tensors[p + 'attn_k_b.weight'] = combined[:, :c.qk_nope_head_dim].transpose(0, 2, 1).copy()
            tensors[p + 'attn_v_b.weight'] = combined[:, c.qk_nope_head_dim:].copy()
        elif suffix == 'mlp.experts.gate_up_proj':
            tensors[p + 'ffn_gate_exps.weight'], tensors[p + 'ffn_up_exps.weight'] = np.split(array, 2, axis=1)
        else:
            tensors[p + names[suffix]] = array
    values = {
        'block_count': c.num_hidden_layers, 'embedding_length': c.hidden_size,
        'feed_forward_length': c.intermediate_size, 'context_length': c.max_position_embeddings,
        'attention.head_count': c.num_attention_heads,
        'attention.head_count_kv': [0 if t == 'linear_attention' else 1 for t in c.layer_types],
        'attention.q_lora_rank': c.q_lora_rank, 'attention.kv_lora_rank': c.kv_lora_rank,
        'attention.key_length_mla': c.qk_nope_head_dim, 'attention.value_length_mla': c.v_head_dim,
        'attention.layer_norm_rms_epsilon': c.rms_norm_eps, 'attention.layer_norm_epsilon': 1e-6,
        'rope.dimension_count': 0, 'ssm.conv_kernel': c.linear_conv_kernel_dim,
        'kda.head_dim': c.linear_head_dim, 'kda.gate_lower_bound': c.linear_lower_bound,
        'attention.indexer.head_count': c.index_n_heads, 'attention.indexer.key_length': c.index_head_dim,
        'attention.indexer.top_k': c.index_topk, 'attention.indexer.kpool': c.index_kpool,
        'attention.indexer.kpool_select_tail': c.index_kpool_always_select_tail,
        'hyper_connection.count': c.hc_mult, 'hyper_connection.sinkhorn_iterations': c.hc_sinkhorn_iters,
        'hyper_connection.epsilon': c.hc_eps, 'expert_count': c.n_routed_experts,
        'expert_used_count': c.num_experts_per_tok, 'expert_shared_count': c.n_shared_experts,
        'expert_feed_forward_length': c.moe_intermediate_size, 'expert_weights_scale': c.routed_scaling_factor,
        'expert_weights_norm': c.norm_topk_prob, 'expert_gating_func': 2,
        'leading_dense_block_count': 3, 'swiglu_clamp_exp': c.swiglu_limit,
    }
    metadata = [('general.architecture', 8, 'glm5-next'), ('general.alignment', 4, 32)]
    for name, value in values.items():
        if isinstance(value, list):
            kind, value = 9, (4, value)
        else:
            kind = 7 if isinstance(value, bool) else 4 if isinstance(value, int) else 6
        metadata.append(('glm5-next.' + name, kind, value))
    path = _fixture(tmp_path / 'tiny-glm53.gguf',
        [(name, array.shape[::-1], encoded.get(name, (0, None))[0],
          encoded[name][1] if name in encoded else np.asarray(array, dtype='<f4').tobytes())
         for name, array in tensors.items()], metadata=metadata)
    options = NextConfig(cache_bytes=32 << 10, io_bytes=(4 if layers > 5 else 1) << 20, kv_bytes=1 << 20,
        workspace_bytes=1 << 20, reserve_bytes=0, max_tensor_bytes=64 << 10,
        row_chunk=3, prefill_chunk=1, max_context=32, policy='adaptive',
        prefetch=False, dtype='float32', device='cpu', load_tokenizer=False)
    return path, ref, head, options


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_official_hf_full_logits_cached_decode_and_greedy(tmp_path, device):
    if device == 'gpu' and not mx.metal.is_available():
        pytest.skip('Metal device unavailable; no M5 benchmark is claimed')
    path, ref, head, options = export_glm53_fixture(tmp_path)
    ids = torch.tensor([[1, 8, 12, 5, 21, 6, 4]])
    with torch.inference_mode():
        expected = ref(ids, use_cache=True)
        expected_logits = head(expected.last_hidden_state).numpy()
    with NativeGGUFGLM5Model(path, replace(options, device=device)) as model:
        assert model.checkpoint.stats['tensor_bytes'] == 0
        actual = model.forward(ids.numpy())
        np.testing.assert_allclose(np.asarray(actual.logits), expected_logits, rtol=2e-4, atol=2e-5)
        follow_ids = torch.tensor([[11, 3]])
        with torch.inference_mode():
            follow_ref = ref(follow_ids, use_cache=True, past_key_values=expected.past_key_values)
            follow_logits = head(follow_ref.last_hidden_state).numpy()
        follow = model.forward(follow_ids.numpy(), past_key_values=actual.past_key_values)
        np.testing.assert_allclose(np.asarray(follow.logits), follow_logits, rtol=2e-4, atol=2e-5)
        sequence = ids.clone()
        with torch.inference_mode():
            for _ in range(4):
                token = head(ref(sequence, use_cache=False).last_hidden_state[:, -1]).argmax(-1)
                sequence = torch.cat([sequence, token[:, None]], dim=1)
        np.testing.assert_array_equal(model.generate(ids.numpy(), max_new_tokens=4), sequence.numpy())
        assert actual.past_key_values.position == 9
        assert model.reader.cache.peak_bytes <= options.cache_bytes
        assert model.checkpoint.stats['peak_reserved_bytes'] <= options.io_bytes


def test_failed_io_poisons_existing_cache(tmp_path, monkeypatch):
    path, _, _, options = export_glm53_fixture(tmp_path)
    with NativeGGUFGLM5Model(path, options) as model:
        session = model.forward([[1, 2]]).past_key_values
        def broken(*args, **kwargs):
            raise EOFError('injected truncated GGUF payload')
        monkeypatch.setattr(model.reader, 'linear', broken)
        with pytest.raises(EOFError, match='truncated'):
            model.forward([[3]], past_key_values=session)
        assert not session.valid
        with pytest.raises(ValueError, match='Invalid GLM5 cache'):
            model.forward([[3]], past_key_values=session)


def test_context_budget_rejects_before_weight_reads(tmp_path):
    path, _, _, options = export_glm53_fixture(tmp_path)
    with NativeGGUFGLM5Model(path, replace(options, max_context=2)) as model:
        with pytest.raises(MemoryError, match='context limit'):
            model.forward([[1, 2, 3]])
        assert model.checkpoint.stats['tensor_bytes'] == 0


def test_mixed_q2_k_q8_0_full_logits_and_cached_decode(tmp_path):
    pytest.importorskip('gguf')
    path, ref, head, options = export_glm53_fixture(tmp_path, quantized=True)
    ids = torch.tensor([[1, 8, 12, 5, 21, 6, 4]])
    with torch.inference_mode():
        expected = ref(ids, use_cache=True)
        expected_logits = head(expected.last_hidden_state).numpy()
    with NativeGGUFGLM5Model(path, options) as model:
        kinds = {int(t.quant_type) for t in model.checkpoint.tensors.values()}
        assert 10 in kinds and 8 in kinds
        actual = model.forward(ids.numpy())
        np.testing.assert_allclose(np.asarray(actual.logits), expected_logits, rtol=4e-4, atol=5e-5)
        with torch.inference_mode():
            follow_ref = ref(torch.tensor([[11, 3]]), use_cache=True, past_key_values=expected.past_key_values)
            follow_logits = head(follow_ref.last_hidden_state).numpy()
        follow = model.forward([[11, 3]], past_key_values=actual.past_key_values)
        np.testing.assert_allclose(np.asarray(follow.logits), follow_logits, rtol=4e-4, atol=5e-5)


def test_45_layers_288_experts_top8_schedule_and_logits(tmp_path):
    path, ref, head, options = export_glm53_fixture(tmp_path, layers=45, experts=288, top_k=8)
    assert ref.config.layer_types.count('linear_attention') == 34
    assert ref.config.layer_types.count('indexed_attention') == 11
    ids = torch.tensor([[1, 8, 12, 5]])
    with torch.inference_mode():
        expected = ref(ids, use_cache=True)
        expected_logits = head(expected.last_hidden_state).numpy()
    with NativeGGUFGLM5Model(path, options) as model:
        actual = model.forward(ids.numpy())
        np.testing.assert_allclose(np.asarray(actual.logits), expected_logits, rtol=4e-4, atol=4e-5)
        assert model.stats['routed_expert_evaluations'] == 42 * 8 * 4
        with torch.inference_mode():
            follow_ref = ref(torch.tensor([[3]]), use_cache=True, past_key_values=expected.past_key_values)
            follow_logits = head(follow_ref.last_hidden_state).numpy()
        follow = model.forward([[3]], past_key_values=actual.past_key_values)
        np.testing.assert_allclose(np.asarray(follow.logits), follow_logits, rtol=4e-4, atol=4e-5)
        assert model.stats['routed_expert_evaluations'] == 42 * 8 * 5


def test_runtime_kpool_prunes_complete_pools_against_official_hf(tmp_path):
    path, ref, head, options = export_glm53_fixture(tmp_path, sparse_dsa=True)
    ids = torch.tensor([[1, 8, 12, 5, 21, 6, 4]])
    assert ids.shape[1] // ref.config.index_kpool > ref.config.index_topk // ref.config.index_kpool
    with torch.inference_mode():
        expected = ref(ids, use_cache=True)
        logits = head(expected.last_hidden_state).numpy()
    with NativeGGUFGLM5Model(path, options) as model:
        actual = model.forward(ids.numpy())
        np.testing.assert_allclose(np.asarray(actual.logits), logits, rtol=2e-4, atol=2e-5)
        with torch.inference_mode():
            expected_follow = ref(torch.tensor([[11, 3]]), use_cache=True, past_key_values=expected.past_key_values)
            follow_logits = head(expected_follow.last_hidden_state).numpy()
        follow = model.forward([[11, 3]], past_key_values=actual.past_key_values)
        np.testing.assert_allclose(np.asarray(follow.logits), follow_logits, rtol=2e-4, atol=2e-5)
