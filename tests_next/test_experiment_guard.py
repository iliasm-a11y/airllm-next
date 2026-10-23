import json
import sys
from dataclasses import replace

import pytest
import psutil

from airllm.nextgen.experiment_guard import ExperimentLimits, admission, supervise, violation, validate_output_paths


def snapshot(available=10_000, swap=500, pressure=1):
    return dict(available_bytes=available, swap_used_bytes=swap, pressure_level=pressure)


def test_admission_reserves_before_spawn():
    limits = ExperimentLimits(rss_bytes=2000, reserve_bytes=1000, admitted_payload_bytes=1500)
    admission(limits, snapshot(2500))
    with pytest.raises(MemoryError):
        admission(limits, snapshot(2499))
    with pytest.raises(MemoryError):
        admission(limits, snapshot(10_000, pressure=2))
    with pytest.raises(MemoryError):
        admission(replace(limits, admitted_payload_bytes=2001), snapshot())


@pytest.mark.parametrize('changes,rss,elapsed,expected', [
    ({}, 2001, 0, 'owned_process_tree_rss'),
    ({'available': 999}, 1, 0, 'system_reserve'),
    ({'swap': 601}, 1, 0, 'system_swap_growth'),
    ({'pressure': 4}, 1, 0, 'macos_memory_pressure'),
    ({}, 1, 11, 'timeout'),
    ({}, 1, 1, None),
])
def test_each_guard(changes, rss, elapsed, expected):
    limits = ExperimentLimits(rss_bytes=2000, reserve_bytes=1000, admitted_payload_bytes=100,
                              swap_growth_bytes=100, timeout_seconds=10)
    assert violation(limits, snapshot(), snapshot(**changes), rss, elapsed) == expected


def test_subprocess_failure_and_timeout_are_persisted(tmp_path):
    limits = ExperimentLimits(reserve_bytes=0, admitted_payload_bytes=0, timeout_seconds=.2)
    output = tmp_path / 'failure.json'
    with pytest.raises(RuntimeError):
        supervise([sys.executable, '-c', 'raise SystemExit(7)'], output, limits)
    assert json.loads(output.with_suffix('.guard.json').read_text())['exit_code'] == 7
    with pytest.raises(RuntimeError):
        supervise([sys.executable, '-c', 'import time; time.sleep(30)'], output, limits)
    report = json.loads(output.with_suffix('.guard.json').read_text())
    assert report['violation'] == 'timeout'
    assert report['elapsed_seconds'] < 5


@pytest.mark.parametrize('field,value', [('timeout_seconds', float('inf')), ('interval_seconds', 2),
                                        ('rss_bytes', 0), ('swap_growth_bytes', -1)])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError):
        ExperimentLimits(**{field: value})


def test_detached_observed_child_is_stopped(tmp_path):
    marker = tmp_path / 'child.pid'
    child = "import time; time.sleep(30)"
    code = ('import subprocess,sys,time; from pathlib import Path; '
            f'p=subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); '
            f'Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(30)')
    limits = ExperimentLimits(reserve_bytes=0, admitted_payload_bytes=0, timeout_seconds=.4)
    with pytest.raises(RuntimeError):
        supervise([sys.executable, '-c', code], tmp_path / 'detached.json', limits)
    pid = int(marker.read_text())
    report = json.loads((tmp_path / 'detached.guard.json').read_text())
    assert report['observed_descendant_count'] == 1
    try:
        _, alive = psutil.wait_procs([psutil.Process(pid)], timeout=2)
        assert not alive or all(p.status() == psutil.STATUS_ZOMBIE for p in alive)
    except psutil.NoSuchProcess:
        pass


def test_output_cannot_alias_model_or_its_links(tmp_path):
    model = tmp_path / 'model.gguf';model.write_bytes(b'preserve')
    symlink = tmp_path / 'symlink.json';symlink.symlink_to(model)
    hardlink = tmp_path / 'hardlink.json';hardlink.hardlink_to(model)
    for output in (model, symlink, hardlink, tmp_path / 'model.log'):
        with pytest.raises(ValueError):
            validate_output_paths(output, protected=[model])
    assert model.read_bytes() == b'preserve'
    validate_output_paths(tmp_path / 'valid.json', protected=[model])
