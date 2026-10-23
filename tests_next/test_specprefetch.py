"""SpecPrefetch control flow and byte accounting, using fake decoded payloads.

These tests measure neither SSD bandwidth nor model throughput. The caller's
actual route is independent of predictions, including wrong/failed predictions.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from airllm.nextgen.cache import SingleFlight
from airllm.nextgen.specprefetch import SpecPrefetch


@pytest.fixture
def scheduler():
    io = SingleFlight(128, 2)
    predictors = []

    def make(budget=64, **kwargs):
        predictor = SpecPrefetch(io, budget, **kwargs)
        predictors.append(predictor)
        return predictor

    yield io, make
    for predictor in predictors:
        predictor.close()
    io.close()


def prime(predictor, layer=0, experts=(1,)):
    predictor.observe(layer, experts)
    return predictor.predict(layer)


def payload(size=8, value=1):
    return np.full(size, value, np.uint8)


@pytest.mark.parametrize('kwargs', [
    {'budget_bytes': -1}, {'budget_bytes': True}, {'budget_bytes': 129},
    {'max_predictions': 0}, {'max_layers': 0}, {'max_transitions': -1},
    {'max_pending': 0}, {'max_route_experts': 0}, {'max_layers': True},
])
def test_invalid_limits_are_rejected(scheduler, kwargs):
    io, _ = scheduler
    options = {'budget_bytes': 64, **kwargs}
    with pytest.raises(ValueError):
        SpecPrefetch(io, **options)


def test_prediction_learns_same_layer_transitions_and_falls_back_to_reuse(scheduler):
    _, make = scheduler
    predictor = make(max_predictions=2)
    assert predictor.predict(0) == ()
    assert prime(predictor, experts=(3, 1)) == (3, 1)
    predictor.observe(0, [7, 5])
    predictor.observe(0, [3, 1])
    assert predictor.predict(0) == (5, 7)  # tied observed transition counts
    assert prime(predictor, layer=1, experts=(9,)) == (9,)
    assert predictor.predict(0) == (5, 7)
    assert predictor.metrics()['transition_predictions'] == 4


def test_predictor_metadata_is_globally_bounded(scheduler):
    _, make = scheduler
    predictor = make(max_layers=3, max_transitions=5, max_route_experts=4)
    for layer in range(30):
        for step in range(3):
            predictor.observe(layer, range(step, step + 20))
            predictor.predict(layer)
        assert len(predictor.layers) <= 3
        assert len(predictor.transitions) <= 5
        assert all(len(state.previous) <= 4 for state in predictor.layers.values())
        assert all(len(state.predicted) <= 1 for state in predictor.layers.values())
    assert predictor.metrics()['layer_evictions'] == 27
    assert predictor.metrics()['truncated_observations'] == 90


def test_only_predicted_experts_are_admitted_and_zero_budget_disables_reads(scheduler):
    io, make = scheduler
    predictor = make()
    assert not predictor.submit(0, 1, 'cold', 8, payload)
    prime(predictor)
    assert not predictor.submit(0, 2, 'wrong', 8, payload)
    disabled = make(budget=0)
    prime(disabled)
    assert not disabled.submit(0, 1, 'disabled', 8, payload)
    assert io.bytes == 0


def test_completed_predictions_keep_both_byte_reservations_until_consumed(scheduler):
    io, make = scheduler
    predictor = make(budget=12)
    prime(predictor)
    assert predictor.submit(0, 1, 'a', 8, payload)
    io.pending['a'][0].result(timeout=2)
    assert io.bytes == predictor.bytes == 8
    assert not predictor.submit(0, 1, 'b', 8, payload)
    np.testing.assert_array_equal(predictor.take('a'), payload())
    assert io.bytes == predictor.bytes == 0
    assert predictor.metrics()['useful_payload_bytes'] == 8
    assert predictor.metrics()['peak_bytes'] == 8


def test_own_deduplication_does_not_adopt_an_existing_exact_read(scheduler):
    io, make = scheduler
    predictor = make()
    prime(predictor)
    calls = []
    assert predictor.submit(0, 1, 'own', 8, lambda: calls.append(1) or payload())
    assert predictor.submit(0, 1, 'own', 8, lambda: calls.append(2) or payload())
    assert io.submit('exact', 8, payload)
    assert not predictor.submit(0, 1, 'exact', 8, payload)
    predictor.discard()
    assert 'exact' in io.pending
    assert io.take('exact').nbytes == 8
    assert calls == [1]
    assert predictor.metrics()['deduplicated'] == 1


def test_shared_budget_is_enforced_alongside_prediction_budget(scheduler):
    io, make = scheduler
    predictor = make(budget=64)
    prime(predictor)
    assert io.submit('exact', 96, lambda: payload(96))
    assert not predictor.submit(0, 1, 'predicted', 40, lambda: payload(40))
    assert predictor.bytes == 0
    assert io.bytes == 96
    assert predictor.metrics()['shared_budget_rejections'] == 1


def test_max_pending_also_bounds_zero_byte_predictions(scheduler):
    io, make = scheduler
    predictor = make(max_pending=2)
    prime(predictor)
    for key in ('a', 'b'):
        assert predictor.submit(0, 1, key, 0, lambda: payload(0))
    assert not predictor.submit(0, 1, 'c', 0, lambda: payload(0))
    assert len(predictor.pending) == 2
    assert io.bytes == predictor.bytes == 0
    predictor.close()
    assert not io.pending


@pytest.mark.parametrize('failure', [OSError('disconnected'), SystemExit('worker failed')])
def test_prediction_failure_is_removed_then_actual_demand_can_retry(scheduler, failure):
    io, make = scheduler
    predictor = make()
    prime(predictor)
    calls = []

    def read():
        calls.append('read')
        if len(calls) == 1:
            raise failure
        return payload(value=7)

    assert predictor.submit(0, 1, 'actual', 8, read)
    predictor.observe(0, [1])
    assert predictor.take('actual') is None
    assert io.take('actual') is None
    assert io.submit('actual', 8, read)
    np.testing.assert_array_equal(io.take('actual'), payload(value=7))
    assert calls == ['read', 'read']
    assert predictor.metrics()['failed'] == 1
    assert io.bytes == predictor.bytes == 0


def test_oversize_prediction_fails_without_poisoning_demand(scheduler):
    io, make = scheduler
    predictor = make()
    prime(predictor)
    assert predictor.submit(0, 1, 'overflow', 8, lambda: payload(9))
    assert predictor.take('overflow') is None
    assert io.bytes == predictor.bytes == 0
    assert predictor.metrics()['failed_reservation_bytes'] == 8
    assert 'useful_payload_bytes' not in predictor.metrics()


def test_discard_swallows_prediction_errors_before_shared_drain(scheduler):
    io, make = scheduler
    predictor = make()
    prime(predictor)

    def bad():
        raise OSError('failed prediction')

    assert predictor.submit(0, 1, 'bad', 8, bad)
    with pytest.raises(OSError):
        io.pending['bad'][0].result(timeout=2)
    predictor.observe(0, [2])
    io.drain()
    assert io.bytes == predictor.bytes == 0
    assert predictor.metrics()['discarded_failed'] == 1
    assert predictor.metrics()['prediction_misses'] == 1


def test_wrong_queued_prediction_is_cancelled_without_touching_exact_io(scheduler):
    _, make = scheduler
    io = SingleFlight(32, 1)
    predictor = SpecPrefetch(io, 16)
    blocker = Event()
    started = Event()
    calls = []

    def exact():
        started.set()
        blocker.wait(2)
        return payload()

    try:
        assert io.submit('exact', 8, exact)
        assert started.wait(2)
        prime(predictor)
        assert predictor.submit(0, 1, 'wrong', 8, lambda: calls.append(1) or payload())
        predictor.observe(0, [2])
        assert calls == []
        assert predictor.metrics()['cancelled'] == 1
        assert predictor.bytes == 0 and io.bytes == 8
        assert 'exact' in io.pending
    finally:
        blocker.set()
        predictor.close()
        io.close()


def test_running_wrong_prediction_holds_reservation_until_discard_finishes(scheduler, monkeypatch):
    io, make = scheduler
    predictor = make()
    started = Event()
    release = Event()
    cancelling = Event()

    def read():
        started.set()
        release.wait(2)
        return payload()

    prime(predictor)
    assert predictor.submit(0, 1, 'running', 8, read)
    assert started.wait(2)
    future = io.pending['running'][0]
    cancel = future.cancel

    def observed_cancel():
        result = cancel()
        cancelling.set()
        return result

    monkeypatch.setattr(future, 'cancel', observed_cancel)
    with ThreadPoolExecutor(1) as workers:
        discarded = workers.submit(predictor.observe, 0, [2])
        assert cancelling.wait(2)
        assert not discarded.done()
        assert io.bytes == predictor.bytes == 8
        release.set()
        discarded.result(timeout=2)
    assert io.bytes == predictor.bytes == 0
    assert predictor.metrics()['wasted_payload_bytes'] == 8
    assert predictor.metrics()['prediction_misses'] == 1


def test_demand_waits_out_concurrent_discard_then_retries_cleanly(scheduler, monkeypatch):
    io, make = scheduler
    predictor = make()
    started = Event()
    release = Event()
    discarding = Event()

    def bad():
        started.set()
        release.wait(2)
        raise OSError('prediction failed during cancellation')

    prime(predictor)
    assert predictor.submit(0, 1, 'key', 8, bad)
    assert started.wait(2)
    future = io.pending['key'][0]
    cancel = future.cancel

    def observed_cancel():
        result = cancel()
        discarding.set()
        return result

    monkeypatch.setattr(future, 'cancel', observed_cancel)
    with ThreadPoolExecutor(2) as workers:
        discard = workers.submit(predictor.discard)
        assert discarding.wait(2)
        demand = workers.submit(predictor.take, 'key')
        release.set()
        discard.result(timeout=2)
        assert demand.result(timeout=2) is None
    assert io.submit('key', 8, payload)
    assert io.take('key').nbytes == 8
    assert io.bytes == predictor.bytes == 0


def test_external_shared_take_is_reconciled_without_inventing_usefulness(scheduler):
    io, make = scheduler
    predictor = make()
    prime(predictor)
    assert predictor.submit(0, 1, 'externally-taken', 8, payload)
    assert io.take('externally-taken').nbytes == 8
    metrics = predictor.metrics()
    assert metrics['outstanding_bytes'] == 0
    assert metrics['externally_released_payload_bytes'] == 8
    assert 'useful_payload_bytes' not in metrics
    assert 'wasted_payload_bytes' not in metrics


def test_layer_eviction_discards_its_completed_predictions(scheduler):
    io, make = scheduler
    predictor = make(max_layers=1)
    prime(predictor)
    assert predictor.submit(0, 1, 'old-layer', 8, payload)
    io.pending['old-layer'][0].result(timeout=2)
    predictor.predict(1)
    assert not predictor.pending and not io.pending
    assert io.bytes == predictor.bytes == 0
    assert predictor.metrics()['wasted_payload_bytes'] == 8


def test_close_racing_admission_leaks_no_reservations_and_preserves_shared_io(scheduler):
    io, make = scheduler
    predictor = make(max_pending=16)
    prime(predictor)
    start = Event()

    def submit(index):
        start.wait(2)
        return predictor.submit(0, 1, index, 8, payload)

    with ThreadPoolExecutor(8) as workers:
        jobs = [workers.submit(submit, index) for index in range(40)]
        start.set()
        predictor.close()
        for job in jobs:
            job.result(timeout=2)
    predictor.close()
    assert predictor.predict(0) == ()
    assert not predictor.submit(0, 1, 'after-close', 8, payload)
    assert io.bytes == predictor.bytes == 0
    assert not io.pending and not predictor.pending
    assert not predictor.layers and not predictor.transitions
    assert io.submit('still-open', 8, payload)
    assert io.take('still-open').nbytes == 8


def test_wrong_predictions_do_not_choose_or_change_actual_expert_values(scheduler):
    io, make = scheduler
    predictor = make()
    prime(predictor, experts=(1,))
    assert predictor.submit(0, 1, 'expert-1', 8, lambda: payload(value=99))
    # The exact router requests expert 2. Prediction 1 is never substituted.
    actual_route = [2]
    predictor.observe(0, actual_route)
    result = []
    for expert in actual_route:
        key = f'expert-{expert}'
        value = predictor.take(key)
        if value is None:
            value = io.take(key)
        if value is None:
            assert io.submit(key, 8, lambda: payload(value=expert))
            value = io.take(key)
        result.append(value)
    np.testing.assert_array_equal(result[0], payload(value=2))
    assert io.bytes == predictor.bytes == 0


def test_native_weights_retries_a_failed_prediction_using_actual_checkpoint(tmp_path, monkeypatch):
    import mlx.core as mx
    from safetensors.numpy import save_file
    from airllm.nextgen.config import NextConfig
    from airllm.nextgen.storage import OffsetCheckpoint
    from airllm.nextgen.weights import NativeWeights

    root = tmp_path / 'cp'
    root.mkdir()
    bank = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    save_file({'bank': bank}, root / 'model.safetensors')
    config = NextConfig(cache_bytes=0, io_bytes=1024, max_tensor_bytes=1024,
                        dtype='float32', prefetch=False, spec_prefetch=True,
                        spec_prefetch_bytes=256, load_tokenizer=False)
    with OffsetCheckpoint(root) as checkpoint:
        reader = NativeWeights(checkpoint, mx, config, {})
        original = reader._load
        calls = []

        def read(key, expert=None, rows=None):
            calls.append((key, expert))
            if len(calls) == 1:
                raise OSError('transient prediction read failure')
            return original(key, expert, rows)

        monkeypatch.setattr(reader, '_load', read)
        try:
            reader.spec.observe(0, [1])
            reader.begin_layer(0, lambda expert: [('bank', expert)])
            reader.observe_layer(0, [1])
            np.testing.assert_array_equal(np.asarray(reader.get('bank', 1)), bank[1])
            assert calls == [('bank', 1), ('bank', 1)]
            assert reader.spec.metrics()['failed'] == 1
            assert reader.io.bytes == reader.spec.bytes == 0
        finally:
            reader.close()
