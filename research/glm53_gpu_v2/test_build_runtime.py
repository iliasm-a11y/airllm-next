"""Portable source-preservation regression; no compiler, Git or Metal executes."""
import json
from pathlib import Path

import pytest

from research.glm53_gpu_v2 import build_runtime


def test_running_inside_source_keeps_temporary_build_outside_it(tmp_path, monkeypatch):
    source = tmp_path / 'preserved-source'
    source.mkdir()
    (source / 'original.txt').write_text('unchanged\n')
    destination = tmp_path / 'runtime-output'
    probe = tmp_path / 'probe.cpp'
    probe.write_text('// not compiled\n')
    monkeypatch.chdir(source)
    monkeypatch.setattr(build_runtime.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(build_runtime.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(build_runtime, 'PATCHES', ())
    monkeypatch.setattr(build_runtime, 'PROBE', probe)
    blocked = ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CMAKE_ARGS',
               'CMAKE_PREFIX_PATH', 'GGML_NATIVE', 'AIRLLM_METAL_Q8_CPU_ARITH',
               'DYLD_LIBRARY_PATH', 'LLAMA_ARG_MODEL', 'LLAMA_API_KEY')
    injected_value = 'do-not-record-build-env-value'
    for name in blocked:
        monkeypatch.setenv(name, injected_value)
    monkeypatch.setenv('BUILD_REGRESSION_SENTINEL', 'retained')
    before = {str(p.relative_to(source)): p.read_bytes() if p.is_file() else None for p in source.rglob('*')}
    observed = []

    def stop_before_first_external_command(command, **kwargs):
        assert command[:3] == ['git', 'clone', '--no-checkout']
        checkout = Path(command[-1]).resolve()
        observed.append(checkout)
        # Check the real scratch directory while it exists, before any Git call.
        assert checkout.parent.is_dir()
        assert checkout.is_relative_to(destination.resolve())
        assert not checkout.is_relative_to(source.resolve())
        assert not set(blocked).intersection(kwargs['env'])
        assert kwargs['env']['BUILD_REGRESSION_SENTINEL'] == 'retained'
        raise RuntimeError('intentional stop before any external command')

    monkeypatch.setattr(build_runtime.subprocess, 'run', stop_before_first_external_command)
    with pytest.raises(RuntimeError, match='intentional stop'):
        build_runtime.build(source, destination, jobs=1)
    assert len(observed) == 1
    after = {str(p.relative_to(source)): p.read_bytes() if p.is_file() else None for p in source.rglob('*')}
    assert after == before
    assert not observed[0].parent.exists()
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['source_checkout_modified'] is False
    assert set(blocked).issubset(manifest['removed_environment_names'])
    assert injected_value not in json.dumps(manifest)
