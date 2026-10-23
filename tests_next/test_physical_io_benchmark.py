"""Run the public protocol and independently check measured/warmup boundaries."""
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_physical_io.py'


def test_protocol_counts_only_measured_reads_and_removes_owned_fixture(tmp_path):
    output = tmp_path / 'report'
    done = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output),
                           '--size-mib', '1', '--chunk-kib', '64', '--repeats', '1',
                           '--modes', 'pread'], capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    report = json.loads((output / 'report.json').read_text())
    assert len(report['rows']) == 2
    assert report['fixture_removed_after_run']
    assert not list(output.glob('airllm-physical-*'))
    for row in report['rows']:
        assert row['logical_read_bytes'] == 1024 ** 2
        assert row['read_calls'] == 16
        assert row['exact']
        assert row['ordered_chunk_digest_sha256'] == report['fixture']['ordered_chunk_digest_sha256']
        assert not row['physical_ssd_bandwidth_certified']
        if row['cache_scenario'] == 'explicit_normal_pread_warmup':
            assert row['warmup_logical_read_bytes'] == row['logical_read_bytes']
            assert row['warmup_read_calls'] == row['read_calls']
        else:
            assert row['warmup_logical_read_bytes'] == row['warmup_read_calls'] == 0


def test_protocol_refuses_oversized_fixture_before_creating_output(tmp_path):
    output = tmp_path / 'too-large'
    done = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output), '--size-mib', '257'],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode != 0
    assert not output.exists()
