"""Test the staged patch on temporary copies; never import model backends."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / 'benchmarks/evaluate_real_world.py'
PATCH = ROOT / 'research/real_world/atomic_json_save.patch'
LEGACY_SAVE = "def save(path, obj):\n    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\\n')\n"


@pytest.fixture
def staged(tmp_path):
    """Exercise the actual patch, or the already-adopted product implementation."""
    target = tmp_path / 'benchmarks/evaluate_real_world.py'
    target.parent.mkdir()
    source = TARGET.read_text()
    target.write_text(source)
    if LEGACY_SAVE in source:
        patched = subprocess.run(['patch', '--batch', '-p1', '-i', str(PATCH)],
                                 cwd=tmp_path, capture_output=True, text=True)
        assert patched.returncode == 0, patched.stdout + patched.stderr
    spec = importlib.util.spec_from_file_location('staged_real_world_evaluation', target)
    module = importlib.util.module_from_spec(spec)
    original_sys_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def test_utf8_json_creation_and_temporary_cleanup(staged, tmp_path):
    output = tmp_path / 'résultat.json'
    value = {'réponse': 'Été, 中文 et 👩🏽\u200d💻', 'entier': 42, 'booléen': False}
    staged.save(output, value)
    assert json.loads(output.read_text(encoding='utf-8')) == value
    raw = output.read_bytes()
    assert 'Été'.encode('utf-8') in raw and raw.endswith(b'\n')
    assert list(tmp_path.glob('.résultat.json.*.tmp')) == []


def test_same_directory_closed_complete_file_before_atomic_replace(staged, tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    old, new = {'version': 1}, {'version': 2, 'réponse': 'complète'}
    output.write_text(json.dumps(old))
    streams = []
    real_create = staged.NamedTemporaryFile
    def create(*args, **kwargs):
        stream = real_create(*args, **kwargs)
        streams.append(stream)
        return stream
    replacements = []
    def replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert streams[-1].closed
        assert source.parent.resolve() == output.parent.resolve()
        assert destination == output
        assert json.loads(output.read_text()) == old
        assert json.loads(source.read_text()) == new
        replacements.append((source, destination))
        os.replace(source, destination)
    monkeypatch.setattr(staged, 'NamedTemporaryFile', create)
    monkeypatch.setattr(staged, 'os', SimpleNamespace(replace=replace))
    staged.save(str(output), new)
    assert len(replacements) == 1
    assert json.loads(output.read_text()) == new
    assert not replacements[0][0].exists()
    assert list(tmp_path.glob('.report.json.*.tmp')) == []


def test_partial_write_exception_preserves_previous_json_and_cleans_temporary(staged, tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    original = '{"ancienne_preuve": "intacte"}\n'
    output.write_text(original)
    def incomplete_dump(value, stream, **kwargs):
        stream.write('{"nouvelle_preuve":')
        stream.flush()
        raise OSError('controlled partial write failure')
    monkeypatch.setattr(staged, 'json', SimpleNamespace(dump=incomplete_dump))
    with pytest.raises(OSError, match='partial write failure'):
        staged.save(output, {'nouvelle_preuve': 'complète'})
    assert output.read_text() == original
    assert json.loads(output.read_text()) == {'ancienne_preuve': 'intacte'}
    assert list(tmp_path.glob('.report.json.*.tmp')) == []


def test_serialization_exception_preserves_previous_json_and_cleans_temporary(staged, tmp_path):
    output = tmp_path / 'report.json'
    original = '{"previous": true}\n'
    output.write_text(original)
    with pytest.raises(TypeError):
        staged.save(output, {'first': 1, 'unsupported': object()})
    assert output.read_text() == original
    assert list(tmp_path.glob('.report.json.*.tmp')) == []


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_cannot_publish_nonstandard_json(staged, tmp_path, value):
    output = tmp_path / 'report.json'
    original = '{"previous": true}\n'
    output.write_text(original)
    with pytest.raises(ValueError):
        staged.save(output, {'invalid': value})
    assert output.read_text() == original
    assert list(tmp_path.glob('.report.json.*.tmp')) == []


def test_failed_replace_preserves_previous_json_and_cleans_temporary(staged, tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    original = '{"previous": true}\n'
    output.write_text(original)
    def fail(source, destination):
        assert Path(source).exists()
        raise OSError('controlled replace failure')
    monkeypatch.setattr(staged, 'os', SimpleNamespace(replace=fail))
    with pytest.raises(OSError, match='replace failure'):
        staged.save(output, {'next': True})
    assert output.read_text() == original
    assert list(tmp_path.glob('.report.json.*.tmp')) == []


def test_close_failure_before_replace_preserves_previous_json(staged, tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    original = '{"previous": true}\n'
    output.write_text(original)
    real_create = staged.NamedTemporaryFile
    @contextmanager
    def fail_on_exit(*args, **kwargs):
        with real_create(*args, **kwargs) as stream:
            yield stream
        raise OSError('controlled close boundary failure')
    monkeypatch.setattr(staged, 'NamedTemporaryFile', fail_on_exit)
    with pytest.raises(OSError, match='close boundary failure'):
        staged.save(output, {'next': True})
    assert output.read_text() == original
    assert list(tmp_path.glob('.report.json.*.tmp')) == []
