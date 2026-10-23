"""Tiny CPU-only routing regressions; no checkpoint, experts, or GPU work."""
import builtins
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import psutil
import pytest
import torch
import torch.nn.functional as F

from airllm.nextgen.qwen_moe import qwen_route, TORCH_ROUTER_IMPORT_ALLOWANCE_BYTES


@pytest.fixture(autouse=True)
def cpu_only():
    with mx.stream(mx.cpu):
        yield


def test_default_qwen_tie_policy_still_fails_closed():
    with pytest.raises(ValueError, match='boundary tie'):
        qwen_route(mx, mx.ones((1, 1)), mx.zeros((8, 1)),
                   {'num_experts': 8, 'num_experts_per_tok': 2})


@pytest.mark.parametrize('inputs', [
    [[0., 0., 0., 0.]],
    [[10., 1e-8, 0., -10.]],
    [[4., 3., 2., 1.], [0., 0., 0., 0.], [10., 1e-8, 0., -10.]],
])
def test_opt_in_ties_match_complete_official_cpu_router(inputs):
    x = np.array(inputs, dtype=np.float32)
    weight = np.eye(4, dtype=np.float32)
    native_logits = mx.array(x) @ mx.array(weight).T
    probabilities = np.array(mx.softmax(native_logits, axis=-1))
    order = np.argsort(-probabilities, axis=-1)
    tied = probabilities[np.arange(len(x)), order[:, 1]] == probabilities[np.arange(len(x)), order[:, 2]]
    assert tied.any(), 'Fixture must exercise a real native probability boundary tie'
    stats = {}
    ids, scores = qwen_route(mx, mx.array(x), mx.array(weight),
        {'num_experts': 4, 'num_experts_per_tok': 2},
        tie_policy='torch_cpu_reference', workspace_bytes=1 << 20, stats=stats)
    logits = F.linear(torch.from_numpy(x), torch.from_numpy(weight))
    reference_scores, reference_ids = torch.softmax(logits, dtype=torch.float32, dim=-1).topk(2, dim=-1)
    reference_scores = reference_scores / reference_scores.sum(dim=-1, keepdim=True)
    np.testing.assert_array_equal(np.array(ids), reference_ids.numpy())
    # Only affected rows claim exact CPU scores; untied rows remain native.
    np.testing.assert_array_equal(np.array(scores)[tied], reference_scores.numpy()[tied])
    assert stats['qwen_router_tie_fallback_calls'] == 1
    assert stats['qwen_router_tie_fallback_rows'] == int(tied.sum())
    assert stats['qwen_router_tie_reference_device'] == 'cpu'
    assert stats['qwen_router_tie_reference_version'] == str(torch.__version__)
    assert stats['qwen_router_tie_workspace_peak_bytes'] <= 1 << 20


def test_distinct_logits_can_collapse_to_a_softmax_boundary_tie():
    logits = mx.array([[10., 1e-8, 0., -10.]], dtype=mx.float32)
    probabilities = mx.softmax(logits, axis=-1)
    assert float(logits[0, 1].item()) > float(logits[0, 2].item())
    assert float(probabilities[0, 1].item()) == float(probabilities[0, 2].item())


def test_reference_gate_preserves_complete_chunk_shape(monkeypatch):
    inputs = mx.array([[4., 3., 2., 1.], [0., 0., 0., 0.], [10., 1e-8, 0., -10.]])
    calls = []
    linear = F.linear
    def record_shape(x, weight, *args, **kwargs):
        calls.append((tuple(x.shape), tuple(weight.shape)))
        return linear(x, weight, *args, **kwargs)
    monkeypatch.setattr(F, 'linear', record_shape)
    stats = {}
    qwen_route(mx, inputs, mx.eye(4), {'num_experts': 4, 'num_experts_per_tok': 2},
               tie_policy='torch_cpu_reference', workspace_bytes=1 << 20, stats=stats)
    assert calls == [((3, 4), (4, 4))]
    assert stats['qwen_router_tie_fallback_rows'] == 2
    assert stats['qwen_router_tie_reference_rows'] == 3


def test_untied_opt_in_path_is_bitwise_native_without_torch_import(monkeypatch):
    x = mx.ones((1, 1)); weight = mx.array([[4.], [3.], [2.], [1.]])
    probabilities = mx.softmax(x @ weight.T, axis=-1)
    native_ids = mx.argsort(-probabilities, axis=-1)[:, :2]
    native_scores = mx.take_along_axis(probabilities, native_ids, axis=-1)
    native_scores = native_scores / mx.sum(native_scores, axis=-1, keepdims=True)
    real_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == 'torch':
            raise AssertionError('Untied routing must not import Torch')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    ids, scores = qwen_route(mx, x, weight, {'num_experts': 4, 'num_experts_per_tok': 2},
        tie_policy='torch_cpu_reference', workspace_bytes=1 << 20)
    np.testing.assert_array_equal(np.array(ids), np.array(native_ids))
    np.testing.assert_array_equal(np.array(scores), np.array(native_scores))


def test_tie_workspace_refused_before_reference_import(monkeypatch):
    stats = {}
    real_import = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == 'torch':
            raise AssertionError('Workspace refusal must precede reference import')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_torch)
    with pytest.raises(MemoryError, match='workspace_bytes'):
        qwen_route(mx, mx.ones((1, 1)), mx.zeros((8, 1)),
            {'num_experts': 8, 'num_experts_per_tok': 2}, tie_policy='torch_cpu_reference',
            workspace_bytes=1, stats=stats)
    assert stats['qwen_router_tie_budget_rejections'] == 1


def test_first_reference_import_preserves_system_reserve(monkeypatch):
    reserve = 3 << 30
    monkeypatch.delitem(sys.modules, 'torch')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(
        available=reserve + TORCH_ROUTER_IMPORT_ALLOWANCE_BYTES - 1))
    stats = {}
    with pytest.raises(MemoryError, match='reserve'):
        qwen_route(mx, mx.ones((1, 1)), mx.zeros((8, 1)),
            {'num_experts': 8, 'num_experts_per_tok': 2}, tie_policy='torch_cpu_reference',
            workspace_bytes=1 << 20, reserve_bytes=reserve, stats=stats)
    assert stats['qwen_router_tie_import_admission_rejections'] == 1


@pytest.mark.parametrize('kwargs', [
    {'tie_policy': 'guess'},
    {'tie_policy': 'torch_cpu_reference'},
    {'tie_policy': 'torch_cpu_reference', 'workspace_bytes': 1 << 20, 'reserve_bytes': -1},
])
def test_reference_policy_requires_valid_explicit_budgets(kwargs):
    with pytest.raises(ValueError):
        qwen_route(mx, mx.ones((1, 1)), mx.zeros((8, 1)),
                   {'num_experts': 8, 'num_experts_per_tok': 2}, **kwargs)


def test_real_captured_gpu_tie_matches_complete_cpu_router():
    local_capture = os.environ.get('AIRLLM_TEST_ROUTER_CAPTURE')
    if not local_capture:
        pytest.skip('Set AIRLLM_TEST_ROUTER_CAPTURE to the local, unpublished real router NPZ')
    if not mx.metal.is_available():
        pytest.skip('Captured native tie requires its Metal matmul')
    capture = Path(local_capture).expanduser()
    if not capture.is_file():
        pytest.fail('AIRLLM_TEST_ROUTER_CAPTURE does not name an existing fixture')
    if capture.stat().st_size > 8 << 20:
        pytest.fail('Router capture exceeds the 8 MiB fixture limit')
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == '0b343cc940dfa5099fb87832acc1117998f3c95833658bf8b4a0b8095125cacd'
    arrays = np.load(capture)
    x, weight = arrays['x'], arrays['weight']
    with torch.inference_mode():
        logits = F.linear(torch.from_numpy(x.reshape(-1, x.shape[-1])), torch.from_numpy(weight))
        values, reference_ids = torch.softmax(logits, dim=-1, dtype=torch.float32).topk(8, dim=-1)
        reference_scores = values / values.sum(dim=-1, keepdim=True)
    assert 206 in reference_ids[0].tolist() and 49 not in reference_ids[0].tolist()
    stats = {}
    with mx.stream(mx.gpu):
        native_x, native_weight = mx.array(x), mx.array(weight)
        native_logits = native_x.reshape(-1, x.shape[-1]) @ native_weight.T
        np.testing.assert_array_equal(np.array(native_logits), arrays['logits'])
        with pytest.raises(ValueError, match='boundary tie'):
            qwen_route(mx, native_x, native_weight, {'num_experts': 256, 'num_experts_per_tok': 8})
        ids, scores = qwen_route(mx, native_x, native_weight,
            {'num_experts': 256, 'num_experts_per_tok': 8}, tie_policy='torch_cpu_reference',
            workspace_bytes=32 << 20, stats=stats)
        np.testing.assert_array_equal(np.array(ids), reference_ids.numpy())
        np.testing.assert_array_equal(np.array(scores), reference_scores.numpy())
    assert stats['qwen_router_tie_fallback_rows'] == 1
    assert stats['qwen_router_tie_reference_operation'] == 'linear_softmax_topk_normalize'
    assert stats['qwen_router_tie_workspace_peak_bytes'] <= 32 << 20
