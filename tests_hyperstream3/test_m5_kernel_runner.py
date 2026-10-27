"""Portable controls; these tests do not initialize MLX or use a GPU."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('m5_kernel_runner', ROOT/'benchmarks/validate_m5_kernels.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_shapes_cover_prefill_sizes_and_multiple_reduction_widths():
    cases = runner.shapes()
    assert {16, 32, 64, 128}.issubset({b for _, _, b in cases})
    assert {256, 512, 1024}.issubset({w for _, w, _ in cases})
    assert any(rows%16 for rows, _, _ in cases)
    assert any(batch%16 for _, _, batch in runner.shapes(True))


@pytest.mark.parametrize('kind', [12, 14])
@pytest.mark.parametrize('grouped', [False, True])
def test_tiny_oracle_is_finite_and_explicit_workspace_fits(kind, grouped):
    packed, inputs, expected = runner.make_operands(kind, 3, 256, 2, grouped)
    assert np.isfinite(expected).all()
    assert expected.shape == ((2, 2, 3) if grouped else (2, 3))
    assert runner.allocation_charge(packed, inputs, expected) < runner.ALLOCATOR_BYTES//2
    assert runner.comparison(expected, expected)['accepted']
    assert not runner.comparison(np.full_like(expected, np.nan), expected)['accepted']


def test_atomic_output_preserves_complete_json_and_no_partial(tmp_path):
    output = tmp_path/'result.json'
    runner.atomic_json(output, {'stage': 1})
    runner.atomic_json(output, {'stage': 2})
    assert json.loads(output.read_text()) == {'stage': 2}
    assert not (tmp_path/'result.json.partial').exists()


def test_refusal_persists_status_and_never_imports_mlx(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise MemoryError('macOS memory pressure is above normal before launch')
    monkeypatch.setattr(runner, 'supervise', refuse)
    original = builtins.__import__
    def no_mlx(name, *args, **kwargs):
        if name == 'mlx' or name.startswith('mlx.'):
            pytest.fail('MLX import before resource admission')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_mlx)
    output = tmp_path/'refusal.json'
    assert runner.main(['--output', str(output), '--quick']) == 75
    report = json.loads(output.read_text())
    assert report['status'] == 'deferred'
    assert report['gpu_worker_launched'] is False
    assert report['cases'] == []
    assert report['limits']['reserve_bytes'] == 3*1024**3
    assert report['limits']['swap_growth_bytes'] == 128*1024**2


def test_worker_entry_admits_before_any_mlx_import(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'admission', lambda *args: (_ for _ in ()).throw(MemoryError('pressure 2')))
    monkeypatch.setitem(sys.modules, 'mlx', None)
    with pytest.raises(MemoryError, match='pressure 2'):
        runner.main(['--output', str(tmp_path/'worker.json'), '--worker'])


def test_supervisor_does_not_overwrite_prior_evidence(tmp_path):
    path = tmp_path/'existing.json'
    path.write_text('prior evidence')
    with pytest.raises(FileExistsError):
        runner.main(['--output', str(path)])
    assert path.read_text() == 'prior evidence'


def test_private_worker_without_supervisor_token_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'admission', lambda _: {})
    monkeypatch.delenv('AIRLLM_M5_KERNEL_SUPERVISOR_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='isolated supervisor'):
        runner.main(['--output', str(tmp_path/'worker.json'), '--worker'])


def test_capture_cannot_alias_kernel_source(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(['--output', str(tmp_path/'report.json'), '--capture', str(ROOT/'benchmarks/validate_m5_kernels.py')])


def test_raw_tensor_roundtrip_is_exact_and_rejects_corruption(tmp_path):
    diagnostic_spec = importlib.util.spec_from_file_location('m5_diagnostic', ROOT/'benchmarks/diagnose_m5_numerics.py')
    diagnostic = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic)
    tensor_folder = tmp_path/'report.tensors'
    tensor_folder.mkdir()
    original = np.array([[1.25, -2., .125]], dtype=np.float32)
    descriptor = runner.save_tensor(tensor_folder, 'values.f32', original, '<f4')
    restored = diagnostic.load_tensor(tmp_path, descriptor)
    np.testing.assert_array_equal(original, restored)
    (tensor_folder/'values.f32').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='digest differs'):
        diagnostic.load_tensor(tmp_path, descriptor)


def test_raw_tensor_diagnostic_rejects_escape(tmp_path):
    diagnostic_spec = importlib.util.spec_from_file_location('m5_diagnostic_escape', ROOT/'benchmarks/diagnose_m5_numerics.py')
    diagnostic = importlib.util.module_from_spec(diagnostic_spec)
    diagnostic_spec.loader.exec_module(diagnostic)
    folder = tmp_path/'evidence'
    folder.mkdir()
    outside = tmp_path/'outside.f32'
    outside.write_bytes(np.array([1.],dtype='<f4').tobytes())
    descriptor = dict(file='../outside.f32',sha256=diagnostic.sha(outside),bytes=4,dtype='<f4',shape=[1])
    with pytest.raises(ValueError, match='path or digest'):
        diagnostic.load_tensor(folder, descriptor)
