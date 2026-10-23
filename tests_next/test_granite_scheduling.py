"""Queued expert work preserves arithmetic and bounds retained weight graphs."""
import numpy as np
import pytest
from test_granite_runtime import checkpoint
from test_native import options
from airllm.nextgen.granite_runtime import AirLLMNextGraniteMoeModel


@pytest.mark.parametrize('group', [2, 4, 8])
@pytest.mark.parametrize('prefetch', [False, True])
def test_queued_expert_logits_routes_and_tokens(tmp_path, group, prefetch):
    root = tmp_path/'granite'
    checkpoint(root)
    values = []
    for size in [1, group]:
        with AirLLMNextGraniteMoeModel(root, next_config=options(tmp_path, prefetch=prefetch),
                expert_sync_group=size) as model:
            logits = np.asarray(model([[1, 7, 8]]).logits).copy()
            routes = [m.last_route_ids.copy() for m in model.mlps]
            tokens = model.generate([[1, 7, 8]], max_new_tokens=6, eos_token_id=[]).tolist()
            values.append((logits, routes, tokens, model.metrics()['expert_scheduling']))
    np.testing.assert_array_equal(values[0][0], values[1][0])
    for a, b in zip(values[0][1], values[1][1]):
        np.testing.assert_array_equal(a, b)
    assert values[0][2] == values[1][2]
    assert values[1][3]['barriers'] < values[0][3]['barriers']
    assert 0 < values[1][3]['retained_bound_peak_bytes'] <= options(tmp_path).workspace_bytes


def test_queue_flushes_before_retained_workspace_overflow(tmp_path):
    root = tmp_path/'granite'
    checkpoint(root)
    with AirLLMNextGraniteMoeModel(root,
            next_config=options(tmp_path, workspace_bytes=13000, prefill_chunk=1),
            expert_sync_group=8) as model:
        model([[7]])
        stats = model.metrics()['expert_scheduling']
        assert stats['retained_bound_peak_bytes'] <= 13000


def test_queue_rejects_single_expert_that_cannot_fit_before_loading(tmp_path):
    root = tmp_path/'granite'
    checkpoint(root)
    with AirLLMNextGraniteMoeModel(root,
            next_config=options(tmp_path, workspace_bytes=1000), expert_sync_group=8) as model:
        with pytest.raises(MemoryError):
            model([[7]])


@pytest.mark.parametrize('group', [0, -1, 33, True, 1.5])
def test_invalid_sync_group_rejected_before_checkpoint_access(tmp_path, group):
    with pytest.raises(ValueError, match='expert_sync_group'):
        AirLLMNextGraniteMoeModel(tmp_path/'absent', expert_sync_group=group)
