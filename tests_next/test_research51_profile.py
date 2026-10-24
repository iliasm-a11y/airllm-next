"""Pure CPU instrumentation checks; no engine, MLX or model is imported."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def profile(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'benchmarks'))
    spec = importlib.util.spec_from_file_location('research51_profile_test', root / 'benchmarks/profile_research51.py')
    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    spec.loader.exec_module(module)
    module.imported_for_profile_test = set(sys.modules) - before
    return module


def test_only_selected_expert_helpers_count_hits_and_misses(profile):
    class Reader:
        def __init__(self):
            self.stats, self.cache, self.calls = {}, {}, []
        def tile(self, name, start, count, expert=None, *, output_role=False):
            self.calls.append((name, start, count, expert, output_role))
            if name not in self.cache:
                self.stats['loads'] = self.stats.get('loads', 0) + 1
                self.cache[name] = SimpleNamespace(nbytes=count * 32)
            return self.cache[name]
    reader = Reader()
    measured = profile.instrument_packed_tile(Reader.tile)
    name = 'blk.2.ffn_gate_exps.weight'
    first = measured(reader, name, 3, 8, 2, output_role=True)
    again = measured(reader, name, 3, 8, 2, output_role=True)
    assert first is again and reader.calls == [(name, 3, 8, 2, True)] * 2
    assert reader.stats['expert_packed_miss_load_calls'] == reader.stats['expert_packed_cache_hit_calls'] == 1
    assert reader.stats['expert_packed_miss_load_bytes'] == reader.stats['expert_packed_cache_hit_bytes'] == 256
    assert reader.stats['expert_packed_miss_load_seconds'] >= 0
    assert reader.stats['expert_packed_cache_hit_seconds'] >= 0
    for other in ('blk.2.attn_q.weight', 'output.weight', 'blk.2.ffn_gate_shexp.weight'):
        measured(reader, other, 0, 4)
    assert reader.stats['loads'] == 4
    assert reader.stats['expert_packed_miss_load_calls'] == 1
    for other in ('ffn_up_exps.weight', 'blk.2.ffn_down_exps.weight'):
        measured(reader, other, 0, 2)
    assert reader.stats['expert_packed_miss_load_calls'] == 3
    assert reader.stats['expert_packed_miss_load_bytes'] == 384


def test_failure_is_separate_and_original_exception_propagates(profile):
    reader = SimpleNamespace(stats={})
    failure = RuntimeError('synthetic failure')
    def tile(*args, **kwargs):
        raise failure
    with pytest.raises(RuntimeError) as raised:
        profile.instrument_packed_tile(tile)(reader, 'blk.0.ffn_down_exps.weight', 0, 1)
    assert raised.value is failure
    assert reader.stats['expert_packed_failed_calls'] == 1
    assert reader.stats['expert_packed_failed_seconds'] >= 0
    assert 'expert_packed_miss_load_calls' not in reader.stats
    assert 'expert_packed_cache_hit_calls' not in reader.stats


def test_import_does_not_load_the_engine(profile):
    assert profile.WRAPPER.name == 'profile_research51.py'
    assert 'airllm.nextgen.gguf_qwen' not in profile.imported_for_profile_test


def test_parent_forwards_baseline_guard_limits_flags_and_frozen_protocol(profile, tmp_path, monkeypatch):
    from airllm.nextgen import experiment_guard
    tasks = tmp_path / 'tasks.json'
    tasks.write_text('[{"id":"tiny","prompt":"Test","max_new_tokens":8}]')
    calls = []
    def supervisor(command, output, limits, *, env):
        calls.append((command, output, limits, env))
    monkeypatch.setattr(experiment_guard, 'supervise', supervisor)
    output = tmp_path / 'profile.json'
    monkeypatch.setattr(sys, 'argv', [str(profile.WRAPPER), '--model', str(tmp_path / 'unused.gguf'),
        '--tasks', str(tasks), '--output', str(output), '--rows', '64', '--packed-direct-io', '--zero-copy-weights'])
    profile.main()
    assert experiment_guard.supervise is supervisor and len(calls) == 1
    command, actual_output, limits, env = calls[0]
    assert command[1] == str(profile.WRAPPER) and '--worker' in command
    assert '--packed-direct-io' in command and '--zero-copy-weights' in command
    assert command[command.index('--rows') + 1] == '64' and actual_output == output
    protocol = json.loads(command[command.index('--expected-protocol-json') + 1])
    assert set(protocol) == {str(profile.BASE), str(profile.WRAPPER), str(tasks)}
    assert limits.rss_bytes == 2 << 30 and limits.reserve_bytes == 3 << 30
    assert limits.swap_growth_bytes == 128 << 20 and limits.timeout_seconds == 600
    assert limits.max_processes == 32 and env['HF_HUB_OFFLINE'] == '1'
    assert not output.exists()
