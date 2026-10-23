"""DSA equal-score key choices must match the installed CPU reference."""
import numpy as np
import pytest
import torch
import mlx.core as mx

from airllm.nextgen.attention import dsa_topk
from airllm.nextgen import AirLLMNextModel
from test_native import options


def test_all_zero_dsa_query_uses_exact_cpu_reference_indices():
    scores = np.zeros((1, 2, 7), dtype=np.float32)
    stats = {}
    actual = np.asarray(dsa_topk(mx, mx.array(scores), 4,
                                 workspace_bytes=4096, stats=stats))
    expected = torch.from_numpy(scores).topk(4, dim=-1).indices.numpy()
    np.testing.assert_array_equal(actual, expected)
    assert stats['dsa_tie_fallback_rows'] == 2
    assert stats['dsa_tie_extra_workspace_peak_bytes'] <= 4096


def test_only_ambiguous_rows_use_reference_and_untied_sets_are_unchanged():
    scores = np.array([[[6, 5, 4, 3, 2, 1, 0], [5, 4, 3, 3, 3, 1, -np.inf]]], dtype=np.float32)
    stats = {}
    actual = np.asarray(dsa_topk(mx, mx.array(scores), 3,
                                 workspace_bytes=4096, stats=stats))
    expected = torch.from_numpy(scores).topk(3, dim=-1).indices.numpy()
    assert set(actual[0, 0]) == set(expected[0, 0])
    np.testing.assert_array_equal(actual[0, 1], expected[0, 1])
    assert stats['dsa_tie_fallback_rows'] == 1
    assert 6 not in actual[0, 1]  # The masked future key is never selected here.


def test_masked_negative_infinity_ties_need_no_reference_resolution():
    stats = {}
    selected = np.asarray(dsa_topk(mx, mx.array([[[2., 1., -float('inf'), -float('inf'), -float('inf')]]]),
                                   4, workspace_bytes=4096, stats=stats))
    assert {0, 1}.issubset(set(selected[0, 0]))
    assert stats.get('dsa_tie_fallback_calls', 0) == 0


def test_dsa_tie_scratch_rejected_before_host_score_copy():
    stats = {}
    with pytest.raises(MemoryError, match='tie fallback'):
        dsa_topk(mx, mx.zeros((1, 1, 7)), 4, workspace_bytes=479, stats=stats)
    assert stats['dsa_tie_budget_rejections'] == 1
    assert 'dsa_tie_host_copy_seconds' not in stats


@pytest.mark.parametrize('prompt', [[1, 8, 12], [0, 0, 0, 0, 0]])
def test_glm_eight_tokens_and_cached_logits_match_torch_with_zero_queries(tiny_glm, tmp_path, prompt):
    root, reference = tiny_glm
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        expected_tokens = reference.generate(ids, attention_mask=torch.ones_like(ids),
                                              max_new_tokens=8, do_sample=False).tolist()
    if prompt == [1, 8, 12]:
        # This unchanged release fixture exposed the inherited selection bug.
        assert expected_tokens[0][3:] == [57, 57, 8, 0, 77, 45, 72, 57]
    with AirLLMNextModel(root, next_config=options(tmp_path)) as model, torch.inference_mode():
        assert model.generate([prompt], max_new_tokens=8).tolist() == expected_tokens
        assert model.reader.stats['dsa_tie_fallback_calls'] > 0
        current = ids
        target_cache = native_cache = None
        length = len(prompt)
        for _ in range(8):
            expected = reference(current, past_key_values=target_cache, use_cache=True,
                                 attention_mask=torch.ones((1, length), dtype=torch.long))
            actual = model(current, past_key_values=native_cache, logits_to_keep=1)
            np.testing.assert_allclose(np.asarray(actual.logits), expected.logits[:, -1:].numpy(),
                                       atol=4e-6, rtol=3e-4)
            target_cache, native_cache = expected.past_key_values, actual.past_key_values
            current = expected.logits[:, -1].argmax(-1)[:, None]
            length += 1
