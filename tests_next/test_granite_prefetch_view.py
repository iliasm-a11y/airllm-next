"""Calibration must control NativeWeights, not a shadow flag on its view."""
from types import SimpleNamespace

import pytest

from airllm.nextgen.granite_runtime import GraniteWeightsView
from airllm.nextgen.weights import NativeWeights
from airllm.nextgen import tuning


@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('on_seconds,choice', [(.5, True), (2., False)])
def test_granite_prefetch_calibration_controls_native_reader(tmp_path, monkeypatch, initial, on_seconds, choice):
    # Exercise the actual NativeWeights.prefetch gate; submit captures requests
    # but never invokes a loader or creates a backend tensor.
    native = NativeWeights.__new__(NativeWeights)
    native.prefetch_enabled = initial
    native.cache = SimpleNamespace(contains=lambda key: False)
    native.estimate = lambda key, expert=None: 8
    native.cfg = SimpleNamespace(max_tensor_bytes=8)
    submitted = []
    native.io = SimpleNamespace(submit=lambda key, size, loader: submitted.append(key) or True)
    native.drain = lambda: None
    view = GraniteWeightsView(native, {})
    observed = []
    model = SimpleNamespace(reader=view, stats={'decode_tokens_per_second': 1.},
        checkpoint=SimpleNamespace(fingerprint='calibration-probe'),
        options=SimpleNamespace(metadata_dir=lambda: tmp_path))
    def generate(*args, **kwargs):
        observed.append((native.prefetch_enabled, view.prefetch('probe.weight')))
    model.generate = generate
    # Deterministic test clocks only; no timing/performance assertion.
    clock = iter([0., 1., 2., 2.+on_seconds, 5., 5.+on_seconds, 8., 9.])
    monkeypatch.setattr(tuning, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(tuning, 'calibrate_cache', lambda cache: {'selected_policy': 'lru'})
    monkeypatch.setattr(tuning, 'runtime_tuning_key', lambda model: 'test-key')
    report = tuning.calibrate_prefetch(model, [[1]], tokens=1)
    expected = [False, False, True, True, False]  # warmup + measured ABBA
    assert [enabled for enabled, _ in observed] == expected
    assert [scheduled for _, scheduled in observed] == expected
    assert submitted == ['probe.weight', 'probe.weight']
    assert report['prefetch_enabled'] is choice
    assert native.prefetch_enabled is view.prefetch_enabled is choice
    assert 'prefetch_enabled' not in view.__dict__


@pytest.mark.parametrize('initial', [False, True])
def test_granite_prefetch_calibration_error_restores_native_flag(tmp_path, initial):
    native = SimpleNamespace(prefetch_enabled=initial)
    view = GraniteWeightsView(native, {})
    observed = []
    def fail(*args, **kwargs):
        observed.append(native.prefetch_enabled)
        raise OSError('calibration probe failure')
    model = SimpleNamespace(reader=view, generate=fail)
    with pytest.raises(OSError, match='calibration probe failure'):
        tuning.calibrate_prefetch(model, [[1]], tokens=1)
    assert observed == [False]
    assert native.prefetch_enabled is view.prefetch_enabled is initial
    assert 'prefetch_enabled' not in view.__dict__
