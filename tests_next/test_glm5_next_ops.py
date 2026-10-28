"""Synthetic GLM5-Next primitives, against extracted official Torch equations.

These tests neither open the 110 GB checkpoint nor establish real generation.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from airllm.nextgen.glm5_next_ops import (
    causal_depthwise_conv1d, clamped_swiglu, kda_log_decay, kpool_select,
    mhc_post, mhc_pre, recurrent_kda, route_experts,
)


def _torch_reference():
    torch = pytest.importorskip("torch")
    import reference_glm53_torch as ref
    return torch, ref


def _array(rng, *shape):
    return rng.normal(size=shape).astype(np.float32)


def test_kda_split_decode_and_input_state_ownership():
    rng = np.random.default_rng(53)
    q, k = [_array(rng, 2, 9, 3, 5) for _ in range(2)]
    v = _array(rng, 2, 9, 3, 4)
    g = -np.abs(_array(rng, 2, 9, 3, 5))
    beta = 1 / (1 + np.exp(-_array(rng, 2, 9, 3)))
    initial = _array(rng, 2, 3, 5, 4)
    saved = initial.copy()
    full, full_state = recurrent_kda(q, k, v, g, beta, initial)
    first, state = recurrent_kda(q[:, :5], k[:, :5], v[:, :5], g[:, :5], beta[:, :5], initial)
    outputs = [first]
    for i in range(5, 9):
        output, state = recurrent_kda(q[:, i:i+1], k[:, i:i+1], v[:, i:i+1],
                                      g[:, i:i+1], beta[:, i:i+1], state)
        outputs.append(output)
    np.testing.assert_array_equal(initial, saved)
    np.testing.assert_allclose(np.concatenate(outputs, axis=1), full, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(state, full_state, rtol=2e-6, atol=2e-7)


def test_kda_against_official_torch_recurrence():
    torch, ref = _torch_reference()
    rng = np.random.default_rng(531)
    args = [_array(rng, 2, 7, 3, 4), _array(rng, 2, 7, 3, 4),
            _array(rng, 2, 7, 3, 5), -np.abs(_array(rng, 2, 7, 3, 4)),
            1 / (1 + np.exp(-_array(rng, 2, 7, 3)))]
    initial = _array(rng, 2, 3, 4, 5)
    actual, state = recurrent_kda(*args, initial)
    expected, expected_state = ref.recurrent_kimi_delta_attention(
        *[torch.tensor(a) for a in args], torch.tensor(initial), True,
        use_qk_l2norm_in_kernel=True)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(state, expected_state.numpy(), rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("groups,selected_groups", [(1, 1), (4, 2)])
def test_router_against_official_with_choice_bias(groups, selected_groups):
    torch, ref = _torch_reference()
    rng = np.random.default_rng(532)
    config = SimpleNamespace(num_experts_per_tok=3, num_local_experts=16, hidden_size=7,
                             routed_scaling_factor=2.5, n_group=groups,
                             topk_group=selected_groups, norm_topk_prob=True)
    x, weight, bias = _array(rng, 5, 7), _array(rng, 16, 7), _array(rng, 16)
    module = ref.Glm5NextTextTopkRouter(config)
    with torch.no_grad():
        module.weight.copy_(torch.tensor(weight))
        module.e_score_correction_bias.copy_(torch.tensor(bias))
    logits, weights, indices = module(torch.tensor(x))
    actual = route_experts(x, weight, bias, top_k=3, n_group=groups, topk_group=selected_groups)
    np.testing.assert_allclose(actual.logits, logits.detach().numpy(), rtol=3e-6, atol=3e-7)
    for row in range(len(x)):
        got = dict(zip(actual.indices[row], actual.weights[row]))
        expected = dict(zip(indices[row].numpy(), weights[row].detach().numpy()))
        assert got.keys() == expected.keys()
        np.testing.assert_allclose([got[i] for i in sorted(got)], [expected[i] for i in sorted(got)], rtol=3e-6)


def test_router_bias_does_not_enter_mixture_weights():
    x = np.array([[0.3, -0.8]], dtype=np.float32)
    w = np.array([[1, 0], [0, 1], [2, 1], [1, 2]], dtype=np.float32)
    bias = np.array([-10, 10, 0, 0], dtype=np.float32)
    routing = route_experts(x, w, bias, top_k=2)
    assert 1 in routing.indices[0]
    scores = 1 / (1 + np.exp(-(x @ w.T)))
    raw = scores[0, routing.indices[0]]
    np.testing.assert_allclose(routing.weights[0], 2.5 * raw / raw.sum(), rtol=1e-6)


def test_mhc_against_official_projection_sinkhorn_and_residual():
    torch, ref = _torch_reference()
    rng = np.random.default_rng(533)
    x = _array(rng, 2, 5, 4, 6)
    fn, base, scale = _array(rng, 24, 24), _array(rng, 24), _array(rng, 3)
    config = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                             rms_norm_eps=1e-5, hidden_size=6)
    module = ref.Glm5NextTextHyperConnection(config)
    with torch.no_grad():
        for name, value in (("fn", fn), ("base", base), ("scale", scale)):
            getattr(module, name).copy_(torch.tensor(value))
    expected = module(torch.tensor(x))
    actual = mhc_pre(x, fn, base, scale)
    for got, target in zip(actual, expected):
        np.testing.assert_allclose(got, target.detach().numpy(), rtol=6e-6, atol=8e-7)
    block = _array(rng, 2, 5, 6)
    post, comb, _ = expected
    expected_post = post[..., None] * torch.tensor(block)[:, :, None] + comb.transpose(-1, -2) @ torch.tensor(x)
    np.testing.assert_allclose(mhc_post(block, x, *actual[:2]), expected_post.detach().numpy(), rtol=6e-6, atol=1e-6)


def test_short_convolution_causal_chunking():
    rng = np.random.default_rng(534)
    x, w = _array(rng, 2, 11, 7), _array(rng, 7, 4)
    full, final = causal_depthwise_conv1d(x, w)
    first, state = causal_depthwise_conv1d(x[:, :6], w)
    second, state = causal_depthwise_conv1d(x[:, 6:], w, state)
    np.testing.assert_allclose(full, np.concatenate([first, second], axis=1), atol=2e-6)
    np.testing.assert_array_equal(state, final)
    torch, _ = _torch_reference()
    expected = torch.nn.functional.conv1d(torch.tensor(x).transpose(1, 2),
                 torch.tensor(w)[:, None], padding=3, groups=7)[..., :11]
    expected = torch.nn.functional.silu(expected).transpose(1, 2)
    np.testing.assert_allclose(full, expected.numpy(), rtol=5e-6, atol=2e-6)


def test_clamped_swiglu_against_official():
    torch, ref = _torch_reference()
    rng = np.random.default_rng(535)
    x = _array(rng, 2, 3, 5) * 20
    gate, up, down = _array(rng, 9, 5), _array(rng, 9, 5), _array(rng, 5, 9)
    config = SimpleNamespace(hidden_size=5, intermediate_size=9, hidden_act="silu", swiglu_limit=10.0)
    module = ref.Glm5NextTextMLP(config)
    with torch.no_grad():
        for name, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            getattr(module, name).weight.copy_(torch.tensor(value))
    expected = module(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(clamped_swiglu(x, gate, up, down), expected, rtol=4e-6, atol=1e-4)


def test_forget_gate_is_per_channel_and_lower_bounded():
    x = np.array([[[1.0, -2.0]]], dtype=np.float32)
    f_a = np.eye(2, dtype=np.float32)
    f_b = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    gate = kda_log_decay(x, f_a, f_b, np.zeros(4, np.float32), np.zeros(2, np.float32), heads=2)
    assert gate.shape == (1, 1, 2, 2)
    assert np.all((gate > -5) & (gate < 0))
    assert gate[0, 0, 0, 0] != gate[0, 0, 0, 1]


def test_kpool_against_official_with_left_padding_and_future_mask():
    torch, ref = _torch_reference()
    rng = np.random.default_rng(536)
    config = SimpleNamespace(hidden_size=5, index_n_heads=2, index_head_dim=3,
                             qk_rope_head_dim=0, index_topk=8, q_lora_rank=4,
                             index_kpool=4, index_kpool_always_select_tail=True)
    module = ref.Glm5NextTextIndexer(config, layer_idx=3)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.tensor(_array(rng, *parameter.shape)))
    hidden, q_resid = _array(rng, 2, 17, 5), _array(rng, 2, 17, 4)
    valid = np.ones((2, 17), dtype=bool)
    valid[0, :2] = False
    tx, tq = torch.tensor(hidden), torch.tensor(q_resid)
    expected = module(tx, tq, torch.tensor(valid), None).numpy()
    with torch.no_grad():
        q = module.wq_b(tq).reshape(2, 17, 2, 3).numpy()
        k = module.k_norm(module.wk(tx)).numpy()
        gates = torch.nn.functional.linear(tx, module.index_kpool_compress_gate).numpy()
        weights = module.weights_proj(tx).numpy()
        ape = module.index_kpool_compress_ape.numpy()
    actual = kpool_select(q, k, gates, weights, valid, np.broadcast_to(np.arange(17), (2, 17)), ape,
                          top_k=8, pool_size=4)
    # Selection ordering and -1 padding do not change the sparse mask.
    for b in range(2):
        for t in range(17):
            assert set(actual[b, t]) == set(expected[b, t])
            assert all(index == -1 or (index <= t and valid[b, index]) for index in actual[b, t])
    partial = kpool_select(q[:, 11:], k, gates, weights[:, 11:], valid,
                           np.broadcast_to(np.arange(11, 17), (2, 6)), ape, top_k=8, pool_size=4)
    np.testing.assert_array_equal(partial, actual[:, 11:])


def test_invalid_shapes_fail_before_computation():
    with pytest.raises(ValueError, match="top-k"):
        route_experts(np.zeros((1, 2)), np.zeros((4, 2)), np.zeros(4), top_k=5)
    with pytest.raises(ValueError, match="projection"):
        mhc_pre(np.zeros((1, 1, 4, 3)), np.zeros((24, 11)), np.zeros(24), np.zeros(3))
    with pytest.raises(ValueError, match="log-decay"):
        recurrent_kda(np.zeros((1, 2, 1, 3)), np.zeros((1, 2, 1, 3)),
                      np.zeros((1, 2, 1, 3)), np.zeros((1, 2, 1, 1)), np.zeros((1, 2, 1)))
