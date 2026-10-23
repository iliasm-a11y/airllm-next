"""Scheduling tests use actual selected IDs, never predicted expert routes."""
import numpy as np
import pytest

from airllm.nextgen.expert_union import (ExpertBatchMetrics, expert_work_groups,
    expert_workspace_bytes, validate_expert_execution)


@pytest.mark.parametrize('positions', [1, 2, 4, 8, 16, 32])
def test_union_reuses_shared_experts_and_preserves_every_assignment(positions):
    routes = np.array([[5, 2]] * positions, dtype=np.int32)
    union = list(expert_work_groups(routes))
    serial = list(expert_work_groups(routes, 'serial_positions'))
    assert len(union) == 2
    assert len(serial) == 2 * positions
    for groups in (union, serial):
        actual = np.zeros_like(routes)
        seen = np.zeros_like(routes)
        per_position_order = [[] for _ in range(positions)]
        for expert, rows, slots, _ in groups:
            actual[rows, slots] = expert
            seen[rows, slots] += 1
            for row in rows:
                per_position_order[row].append(expert)
        np.testing.assert_array_equal(actual, routes)
        np.testing.assert_array_equal(seen, 1)
        assert per_position_order == [[2, 5]] * positions


def test_union_cannot_reduce_disjoint_routes_or_hide_empty_work():
    routes = np.arange(16, dtype=np.int32).reshape(8, 2)
    assert len(list(expert_work_groups(routes))) == len(list(expert_work_groups(routes, 'serial_positions'))) == 16
    for invalid in ([], [1, 2], [[1.5]], [[]]):
        with pytest.raises(ValueError, match='route_ids'):
            list(expert_work_groups(invalid))
    with pytest.raises(ValueError, match='expert_execution'):
        validate_expert_execution('automatic')


def test_batch_metrics_history_is_bounded_totals_survive_and_snapshots_are_independent():
    recorder = ExpertBatchMetrics(capacity=2)
    for i in range(5):
        recorder.record(layer=i, execution='union', positions=4, assignments=8,
            distinct_experts=3, expert_groups=3, demand_load_calls=6,
            reader_loads_delta=2, workspace_bytes=100 + i)
    snapshot = recorder.snapshot()
    assert snapshot['totals'] == dict(batches=5, positions=20, assignments=40,
        distinct_expert_sum=15, expert_groups=15, demand_load_calls=30,
        reader_loads_delta=10, workspace_peak_bytes=104)
    assert [event['layer'] for event in snapshot['history']] == [3, 4]
    assert snapshot['dropped_events'] == 3
    snapshot['history'][0]['layer'] = 99
    snapshot['history'].append({})
    assert len(recorder.snapshot()['history']) == 2
    assert recorder.snapshot()['history'][0]['layer'] == 3
    assert recorder.snapshot(reset=True)['totals']['batches'] == 5
    assert recorder.snapshot()['totals']['batches'] == 0


def test_zero_capacity_keeps_only_scalar_totals_and_capacity_is_validated():
    recorder = ExpertBatchMetrics()
    recorder.record(layer=0, execution='union', positions=1, assignments=2,
        distinct_experts=2, expert_groups=2, demand_load_calls=4,
        reader_loads_delta=4, workspace_bytes=100)
    assert recorder.snapshot()['history'] == []
    assert recorder.snapshot()['totals']['demand_load_calls'] == 4
    for invalid in (-1, 4097, True, 1.5):
        with pytest.raises(ValueError, match='expert_metrics_capacity'):
            ExpertBatchMetrics(invalid)


def test_workspace_reservation_scales_linearly_with_positions():
    one = expert_workspace_bytes(1, hidden=16, intermediate=8, experts=6, top_k=2)
    assert one > 0
    assert expert_workspace_bytes(32, 16, 8, 6, 2) == 32 * one
