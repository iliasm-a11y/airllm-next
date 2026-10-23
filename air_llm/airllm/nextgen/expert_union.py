"""Exact expert work scheduling and bounded research telemetry.

These helpers only regroup already selected routes. They neither predict routes
nor propose tokens. ``serial_positions`` is a position-by-position control for
the existing ``union`` scheduler, with the same ascending expert accumulation
order at each position. A load call is an exact-reader request, not proof of a
cache miss, file read or physical storage transfer.
"""
from collections import Counter, deque
import numpy as np


EXPERT_EXECUTIONS = ('union', 'serial_positions')


def validate_expert_execution(execution):
    if execution not in EXPERT_EXECUTIONS:
        raise ValueError(f'expert_execution must be one of {EXPERT_EXECUTIONS}')
    return execution


def expert_work_groups(route_ids, execution='union'):
    """Yield (expert, position indices, route slots, next expert) lazily.

    The router supplies a two-dimensional integer array with distinct expert
    IDs per row. Only one group's indices are materialized at a time. ``union``
    loads each expert once for all positions; the control repeats that work for
    each position and therefore exposes the possible reuse without faking it.
    """
    validate_expert_execution(execution)
    host = np.asarray(route_ids)
    if host.ndim != 2 or host.dtype.kind not in 'iu' or min(host.shape) < 1:
        raise ValueError('route_ids must be a nonempty two-dimensional integer array')
    blocks = ((0, host),) if execution == 'union' else enumerate(host[:, None, :])
    for offset, block in blocks:
        selected = np.unique(block)
        for index, expert in enumerate(selected):
            positions, slots = np.where(block == expert)
            next_expert = int(selected[index + 1]) if index + 1 < len(selected) else None
            yield int(expert), positions + offset, slots, next_expert


def expert_workspace_bytes(positions, hidden, intermediate, experts, top_k):
    """Conservative activation/index bound; loaded weights have separate budgets.

    FP32 is used as a majorant even with lower precision activations. The bound
    includes router scores/ordering, host route copies, output accumulation and
    one expert's gate/up/SiLU/down intermediates at the worst union occupancy.
    It is a safety reservation, not a measured resident-memory peak.
    """
    return 4 * positions * (4 * experts + 10 * top_k + 8 * intermediate + 8 * hidden)


class ExpertBatchMetrics:
    """Scalar lifetime totals plus an optional capped history of layer batches."""
    def __init__(self, capacity=0):
        if type(capacity) is not int or not 0 <= capacity <= 4096:
            raise ValueError('expert_metrics_capacity must be an integer from 0 to 4096')
        self.capacity = capacity
        self.history = deque(maxlen=capacity)
        self.totals = Counter()

    def record(self, *, layer, execution, positions, assignments, distinct_experts,
               expert_groups, demand_load_calls, reader_loads_delta, workspace_bytes):
        self.totals.update(batches=1, positions=positions, assignments=assignments,
                           distinct_expert_sum=distinct_experts, expert_groups=expert_groups,
                           demand_load_calls=demand_load_calls, reader_loads_delta=reader_loads_delta)
        self.totals['workspace_peak_bytes'] = max(self.totals['workspace_peak_bytes'], workspace_bytes)
        if self.capacity:
            self.history.append(dict(layer=int(layer), execution=execution, positions=int(positions),
                assignments=int(assignments), distinct_experts=int(distinct_experts),
                expert_groups=int(expert_groups), demand_load_calls=int(demand_load_calls),
                reader_loads_delta=int(reader_loads_delta), workspace_bytes=int(workspace_bytes)))

    def snapshot(self, *, reset=False):
        keys = ('batches', 'positions', 'assignments', 'distinct_expert_sum', 'expert_groups',
                'demand_load_calls', 'reader_loads_delta', 'workspace_peak_bytes')
        result = dict(totals={key: self.totals[key] for key in keys},
                      history=[dict(event) for event in self.history], capacity=self.capacity,
                      dropped_events=max(0, self.totals['batches'] - len(self.history)))
        if reset:
            self.totals.clear()
            self.history.clear()
        return result
