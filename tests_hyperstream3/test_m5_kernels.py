"""Portable dispatcher contracts and separately admitted real GPU checks.

No tests_next conftest is loaded: that fixture imports Torch and MLX eagerly.
GPU cases require AIRLLM_M5_GPU_TEST=1 and admission before importing MLX.
The scalar GGUF oracle is loaded by AST from the existing independent test.
These are tensor checks, not full-checkpoint greedy fidelity evidence.
"""
import ast
from collections import Counter
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "air_llm"))
from airllm.nextgen import gguf_m5 as m5


@pytest.fixture
def fake_mx(monkeypatch):
    """A NumPy-only stand-in; never imports or initializes actual MLX."""
    cached_kernel = m5._kernel
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    mx.float32 = np.float32
    mx.evaluated = []
    mx.eval = lambda value: mx.evaluated.append(value)
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    yield mx
    cached_kernel.cache_clear()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now


def dispatcher_fixture(monkeypatch, fake_mx, *, candidate_seconds=1., baseline_seconds=10.,
                       candidate_error_at=None, baseline_error_at=None, candidate_value=1.,
                       error_type=RuntimeError, dispatcher_kwargs=None):
    clock = FakeClock()
    monkeypatch.setattr(m5, "perf_counter", clock.read)
    packed = np.zeros((5, 144), np.uint8)
    x = np.zeros((4, 256), np.float32)
    calls = Counter()

    def baseline():
        calls["baseline"] += 1
        clock.now += baseline_seconds
        if calls["baseline"] == baseline_error_at:
            raise RuntimeError("reference error must propagate")
        return np.ones((4, 5), np.float32)

    def candidate(*args, **kwargs):
        calls["candidate"] += 1
        clock.now += candidate_seconds
        if calls["candidate"] == candidate_error_at:
            raise error_type("candidate unavailable")
        return np.full((4, 5), candidate_value, np.float32)

    monkeypatch.setattr(m5, "experimental_matmul", candidate)
    stats = Counter()
    dispatcher = m5.M5BatchDispatcher("reuse", stats, **(dispatcher_kwargs or {}))
    run = lambda budget=1 << 20: dispatcher(packed, x, 12, grouped=False,
        baseline=baseline, spare_workspace=budget)
    return dispatcher, stats, calls, run


@pytest.mark.parametrize("kind,block_bytes", [(12, 144), (14, 210)])
@pytest.mark.parametrize("backend", ["reuse", "tensorops"])
@pytest.mark.parametrize("grouped", [False, True])
def test_generated_source_keeps_packed_storage_fp32_and_correct_axes(kind, block_bytes, backend, grouped):
    header, code = m5.kernel_source(kind, backend, grouped, token_tile=8)
    assert "DECODE" not in code and "BLOCK_BYTES" not in code and "TOKEN_TILE" not in code
    assert f"(j/256)*{block_bytes}" in code
    assert "float weight" in code and "as_type<half>" in code  # original scale representation only
    assert "half(x[" not in code and "half weight" not in code
    assert "(expert*batches+" in code and ")*rows+row]" in code
    assert ("packed_shape[2]" in code) == grouped
    assert "#pragma clang fp contract(off)" in header
    if backend == "tensorops":
        assert "threadgroup float a[16*32], b[16*32]" in code
        assert "(16,16,32,false,true,false," in code  # sixth bool prohibits relaxed precision
        assert "matmul2d<desc,metal::execution_simdgroup>" in code
        assert "acc.is_valid_element(i)" in code
        assert "row=row0+ij[0], batch=batch0+ij[1]" in code
        assert code.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 2
    else:
        assert "float sums[8]" in code and "simd_sum(sums[t])" in code


@pytest.mark.parametrize("args", [(0, "reuse"), (12, "other"), (14, "tensorops", 1),
                                  (12, "reuse", False, 3), (12, "reuse", False, True)])
def test_invalid_source_options_fail_without_device_import(args):
    with pytest.raises(ValueError):
        m5.kernel_source(*args)


@pytest.mark.parametrize("kwargs", [{"backend": "other"}, {"backend": "reuse", "token_tile": 1},
    {"backend": "reuse", "token_tile": True}, {"backend": "tensorops", "minimum_speedup": .9},
    {"backend": "reuse", "minimum_speedup": np.nan},
    {"backend": "reuse", "workspace_bytes": 100},
    {"backend": "reuse", "guard": lambda _: None},
    {"backend": "reuse", "workspace_bytes": 0, "guard": lambda _: None},
    {"backend": "reuse", "workspace_bytes": True, "guard": lambda _: None},
    {"backend": "reuse", "workspace_bytes": 100, "guard": 1}])
def test_dispatcher_invalid_options(kwargs):
    with pytest.raises(ValueError):
        m5.M5BatchDispatcher(stats=Counter(), **kwargs)


def test_kernel_builder_requests_safe_math(fake_mx):
    captured = {}
    fake_mx.fast = ModuleType("mlx.core.fast")
    fake_mx.fast.metal_kernel = lambda **kwargs: captured.update(kwargs) or object()
    m5._kernel.cache_clear()
    first = m5._kernel(12, "tensorops", False, 4)
    assert first is m5._kernel(12, "tensorops", False, 4)
    assert captured["compile_options"] == {"math_mode": "safe"}
    assert captured["input_names"] == ["packed", "x"]


@pytest.mark.parametrize("backend,tile", [("reuse", 4), ("tensorops", 16)])
@pytest.mark.parametrize("grouped", [False, True])
def test_experimental_launch_covers_partial_token_and_weight_tiles(fake_mx, monkeypatch, backend, tile, grouped):
    captured = {}
    sentinel = object()
    def launch(**kwargs):
        captured.update(kwargs)
        return [sentinel]
    monkeypatch.setattr(m5, "_kernel", lambda *args: launch)
    packed = np.zeros((3, 19, 144) if grouped else (19, 144), np.uint8)
    x = np.zeros((3, 17, 256) if grouped else (17, 256), np.float32)
    assert m5.experimental_matmul(packed, x, 12, backend=backend, grouped=grouped) is sentinel
    groups = 2 if backend == "tensorops" else 19
    assert captured["grid"] == (groups * 32, (17 + tile - 1) // tile, 3 if grouped else 1)
    assert captured["threadgroup"] == (32, 1, 1)
    assert captured["output_shapes"] == [(3, 17, 19) if grouped else (17, 19)]


def test_dispatcher_charges_six_output_payloads_before_calibration(monkeypatch, fake_mx):
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx)
    required = 6 * 4 * 5 * 4 + 4096
    np.testing.assert_array_equal(run(required - 1), np.ones((4, 5)))
    assert stats["m5_fallback_workspace"] == 1
    assert calls == {"baseline": 1} and not dispatcher.records
    run(required)
    assert calls["candidate"] == 6 and dispatcher.records[-1]["selected"]


def test_single_token_never_calibrates(monkeypatch, fake_mx):
    monkeypatch.setattr(m5, "experimental_matmul", lambda *a, **k: pytest.fail("candidate called"))
    stats = Counter()
    dispatcher = m5.M5BatchDispatcher("tensorops", stats)
    sentinel = object()
    assert dispatcher(np.zeros((2, 144)), np.zeros((1, 256)), 12, grouped=False,
        baseline=lambda: sentinel, spare_workspace=10**6) is sentinel
    assert stats["m5_fallback_single_token"] == 1 and not dispatcher.records


@pytest.mark.parametrize("candidate_seconds,accepted,reason", [(1., True, "measured_speed"),
                                                            (10., False, "not_faster"),
                                                            (20., False, "not_faster")])
def test_paired_timing_selects_only_measured_improvement(monkeypatch, fake_mx, candidate_seconds, accepted, reason):
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx, candidate_seconds=candidate_seconds)
    np.testing.assert_array_equal(run(), np.ones((4, 5)))
    record = dispatcher.records[0]
    assert record["selected"] is accepted and record["reason"] == reason
    assert record["baseline_seconds"] == [10.] * 4
    assert record["candidate_seconds"] == [candidate_seconds] * 4
    assert stats["m5_calibration_seconds"] == record["calibration_wall_seconds"]
    before = len(dispatcher.records)
    run()
    assert len(dispatcher.records) == before  # cached decision, still budget checked
    assert calls["candidate"] == (7 if accepted else 5)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.])
def test_numerical_disagreement_never_reaches_timing(monkeypatch, fake_mx, value):
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx, candidate_value=value)
    np.testing.assert_array_equal(run(), np.ones((4, 5)))
    assert calls == {"candidate": 1, "baseline": 2}
    assert dispatcher.records[0]["reason"] == "numerical_rejection"
    assert stats["m5_fallback_calls"] == 1


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
@pytest.mark.parametrize("error_at,reason", [(1, "unavailable"), (2, "unavailable_during_timing"),
                                           (6, "runtime_unavailable"), (7, "runtime_unavailable")])
def test_candidate_failure_at_every_phase_preserves_original_fallback(monkeypatch, fake_mx, error_at, reason, error_type):
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx,
        candidate_error_at=error_at, error_type=error_type)
    run()
    if error_at == 7:
        run()
    assert dispatcher.records[-1]["reason"] == reason
    assert not next(iter(dispatcher.decisions.values()))
    candidate_calls = calls["candidate"]
    run()
    assert calls["candidate"] == candidate_calls  # failure cached for this shape
    assert stats["m5_fallback_calls"] >= 1


def test_extra_resource_refusal_happens_before_calibration_and_remains_retryable(monkeypatch, fake_mx):
    charges = []
    def guard(amount):
        charges.append(amount)
        raise MemoryError("current system reserve")
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx,
        dispatcher_kwargs=dict(workspace_bytes=100_000, guard=guard))
    run(20_000)
    assert charges == [100_000 - 20_000 + 6 * 4 * 5 * 4 + 4096]
    assert calls == {"baseline": 1} and not dispatcher.decisions
    assert stats["m5_fallback_resources"] == 1


def test_resource_pressure_before_selected_runtime_falls_back_without_blacklisting(monkeypatch, fake_mx):
    charges = []
    def guard(amount):
        charges.append(amount)
        if len(charges) == 2:
            raise MemoryError("pressure changed after calibration")
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx,
        dispatcher_kwargs=dict(workspace_bytes=100_000, guard=guard))
    run(20_000)
    assert calls["candidate"] == 5  # warmup and timing only
    assert stats["m5_fallback_resources"] == 1
    assert next(iter(dispatcher.decisions.values())) is True
    run(20_000)
    assert calls["candidate"] == 6 and len(charges) == 3
    assert charges == [84_576] * 3


def test_rejected_cached_shape_does_not_require_extra_resource_admission(monkeypatch, fake_mx):
    charges = []
    dispatcher, _, _, run = dispatcher_fixture(monkeypatch, fake_mx, candidate_seconds=20.,
        dispatcher_kwargs=dict(workspace_bytes=100_000, guard=charges.append))
    run(20_000)
    run(20_000)
    assert len(charges) == 1 and next(iter(dispatcher.decisions.values())) is False


@pytest.mark.parametrize("cap", [None, 0, 123, 99_999])
def test_packed_expert_dispatch_spare_never_exceeds_pair_wide_cap(monkeypatch, fake_mx, cap):
    from airllm.nextgen import gguf_metal
    fake_mx.uint8 = np.uint8
    fake_mx.metal = SimpleNamespace(is_available=lambda: True)
    fake_mx.array = np.array
    fake_mx.contiguous = np.ascontiguousarray
    packed = np.zeros((2, 5, 144), np.uint8)
    x = np.zeros((2, 4, 256), np.float32)
    output = np.zeros((2, 4, 5), np.float32)
    monkeypatch.setattr(gguf_metal, "_kernel", lambda *a: lambda **kw: [output])
    seen = []
    def dispatch(*args, **kwargs):
        seen.append(kwargs["spare_workspace"])
        return kwargs["baseline"]()
    result = gguf_metal.packed_expert_matmul(packed, x, 12, max_workspace_bytes=20_000,
        batch_dispatch=dispatch, batch_spare_workspace_bytes=cap)
    np.testing.assert_array_equal(result, output)
    per_call_spare = 20_000 - (2 * packed.nbytes + x.size * 4 + output.nbytes)
    assert seen == [per_call_spare if cap is None else min(cap, per_call_spare)]


@pytest.mark.parametrize("cap", [True, -1, .5])
def test_invalid_pair_spare_refuses_before_device_check(monkeypatch, fake_mx, cap):
    from airllm.nextgen import gguf_metal
    fake_mx.metal = SimpleNamespace(is_available=lambda: pytest.fail("device check before validation"))
    with pytest.raises(ValueError, match="Batch spare workspace"):
        gguf_metal.packed_expert_matmul(np.zeros((1, 1, 144), np.uint8),
            np.zeros((1, 2, 256), np.float32), 12, batch_spare_workspace_bytes=cap)


def test_engine_pair_passes_complete_pair_remaining_budget_to_both_kernels(monkeypatch):
    import importlib.util
    from airllm.nextgen import gguf_metal
    spec = importlib.util.spec_from_file_location("h3_pair_helper",
        ROOT / "tests_next" / "test_gguf_expert_pipeline.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)  # module imports no MLX; GPU tests are never called
    reader, _, _ = helpers.setup_reader(monkeypatch)
    x = np.zeros((2, 3, 256), np.float32)
    dispatcher = object()
    reader.batch_dispatch = dispatcher
    captures = []
    def multiply(packed, inputs, kind, **kwargs):
        captures.append(kwargs)
        return np.zeros((inputs.shape[0], inputs.shape[1], packed.shape[1]), np.float32)
    monkeypatch.setattr(gguf_metal, "packed_expert_matmul", multiply)
    reader.linear_expert_pair(x, helpers.GATE, helpers.UP, [2, 0])
    required = 3 * (2 * 2 * 5 * 144) + 2 * (2 * 2 * 3 * 5 * 4) + 2 * x.nbytes
    assert len(captures) == 2
    assert all(capture["batch_dispatch"] is dispatcher for capture in captures)
    assert all(capture["batch_spare_workspace_bytes"] == reader.cfg.workspace_bytes - required
        for capture in captures)


@pytest.mark.parametrize("error_at", [1, 2])
def test_baseline_errors_are_not_disguised_as_unavailable_candidate(monkeypatch, fake_mx, error_at):
    _, _, _, run = dispatcher_fixture(monkeypatch, fake_mx, baseline_error_at=error_at)
    with pytest.raises(RuntimeError, match="reference error must propagate"):
        run()


@pytest.mark.parametrize("failure_at,reason", [(1, "unavailable"),
    (2, "unavailable_during_timing"), (6, "runtime_unavailable")])
def test_lazy_candidate_errors_are_evaluated_inside_fallback_boundary(monkeypatch, fake_mx, failure_at, reason):
    dispatcher, stats, calls, run = dispatcher_fixture(monkeypatch, fake_mx)
    candidate_operation = m5.experimental_matmul
    class CandidateArray(np.ndarray):
        pass
    def candidate(*args, **kwargs):
        return candidate_operation(*args, **kwargs).view(CandidateArray)
    def evaluate(value):
        if isinstance(value, CandidateArray) and calls["candidate"] == failure_at:
            raise RuntimeError("lazy GPU error")
    monkeypatch.setattr(m5, "experimental_matmul", candidate)
    monkeypatch.setattr(fake_mx, "eval", evaluate)
    np.testing.assert_array_equal(run(), np.ones((4, 5)))
    assert dispatcher.records[-1]["reason"] == reason
    assert stats["m5_fallback_calls"] == 1


def test_gpu_admission_failure_precedes_any_mlx_import(monkeypatch):
    import builtins
    monkeypatch.setenv("AIRLLM_M5_GPU_TEST", "1")
    # Isolate the fixture's import-order contract from optional psutil. The
    # resource guard itself is exercised by its existing dedicated tests.
    guard = ModuleType("airllm.nextgen.experiment_guard")
    guard.ExperimentLimits = lambda: object()
    def refuse(*args):
        raise MemoryError("pressure 2")
    guard.admission = refuse
    monkeypatch.setitem(sys.modules, "airllm.nextgen.experiment_guard", guard)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            pytest.fail("MLX import occurred before admission")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(pytest.skip.Exception, match="GPU admission refused before MLX import"):
        admitted_gpu.__wrapped__()


def _scalar_oracle():
    source = ROOT / "tests_next" / "test_gguf_metal.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scalar_decode")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["scalar_decode"]


@pytest.mark.parametrize("kind", [12, 14])
def test_independent_scalar_oracle_loads_without_other_test_imports(kind):
    block_bytes = 144 if kind == 12 else 210
    packed = np.zeros((1, block_bytes), dtype=np.uint8)
    if kind == 12:
        packed[0, :2] = np.frombuffer(np.float16(.5).tobytes(), dtype=np.uint8)
        packed[0, 4:16] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        packed[0, 16:] = 0x11
        expected = .5
    else:
        packed[0, 192:208] = 1
        packed[0, 208:210] = np.frombuffer(np.float16(.5).tobytes(), dtype=np.uint8)
        expected = -16.
    np.testing.assert_array_equal(_scalar_oracle()(packed, kind, 256),
        np.full((1, 256), expected, dtype=np.float32))


@pytest.fixture
def admitted_gpu():
    if os.environ.get("AIRLLM_M5_GPU_TEST") != "1":
        pytest.skip("GPU validation deferred: requires explicit opt-in and normal-pressure admission")
    from airllm.nextgen.experiment_guard import admission, ExperimentLimits
    try:
        admission(ExperimentLimits())
    except MemoryError as error:
        pytest.skip(f"GPU admission refused before MLX import: {error}")
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    return mx


@pytest.mark.parametrize("kind", [12, 14])
@pytest.mark.parametrize("backend", ["reuse", "tensorops"])
@pytest.mark.parametrize("batch", [2, 4, 8, 16, 17, 32, 64, 128])
@pytest.mark.parametrize("grouped", [False, True])
def test_real_gpu_against_independent_scalar_oracle(admitted_gpu, kind, backend, batch, grouped):
    mx = admitted_gpu
    rng = np.random.default_rng(20261027)
    experts, rows, width = (2 if grouped else 1), 19, 512
    block_bytes = 144 if kind == 12 else 210
    packed = rng.integers(0, 256, size=(experts, rows, 2 * block_bytes), dtype=np.uint8)
    for block in range(2):
        for pos in ([0, 2] if kind == 12 else [208]):
            scales = np.resize(np.array([0., .001, -.002, .125], dtype="<f2"), experts * rows).reshape(experts, rows)
            packed[:, :, block * block_bytes + pos:block * block_bytes + pos + 2] = scales.view(np.uint8).reshape(experts, rows, 2)
    inputs = rng.normal(size=(experts, batch, width)).astype(np.float32)
    scalar_decode = _scalar_oracle()
    expected = np.stack([inputs[e] @ scalar_decode(packed[e], kind, width).T for e in range(experts)])
    with mx.stream(mx.new_stream(mx.gpu)):
        actual = m5.experimental_matmul(mx.array(packed if grouped else packed[0]),
            mx.array(inputs if grouped else inputs[0]), kind, backend=backend, grouped=grouped)
        mx.eval(actual)
        actual = np.asarray(actual)
    np.testing.assert_allclose(actual, expected if grouped else expected[0], atol=1e-3, rtol=5e-5)
