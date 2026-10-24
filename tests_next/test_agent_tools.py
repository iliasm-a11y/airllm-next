"""Real isolated-worker checks for explicit file capabilities and budgets."""
import os
import json
import subprocess
import sys
from threading import Event, Thread

import pytest

from airllm.nextgen.agent_tools import ToolBroker, ToolLimits


def broker_at(tmp_path, limits=None):
    broker = ToolBroker(tmp_path / 'agents', limits)
    broker.register('alice', allowed_tools=('read_text', 'write_text', 'list_files'))
    broker.register('bob', allowed_tools=('read_text', 'write_text', 'list_files'))
    return broker


def test_roundtrip_and_cross_agent_isolation(tmp_path):
    broker = broker_at(tmp_path)
    assert broker.execute('alice', 'write_text', {'path': 'notes/a.txt', 'text': 'été'}) == {
        'ok': True, 'result': {'bytes_written': 5}}
    assert broker.execute('alice', 'read_text', {'path': 'notes/a.txt'})['result']['text'] == 'été'
    assert broker.execute('bob', 'read_text', {'path': 'notes/a.txt'}) == {'ok': False, 'error': 'tool_failed'}
    assert broker.execute('alice', 'list_files', {})['result']['files'] == ['notes']
    assert broker.execute('alice', 'list_files', {'path': 'notes'})['result']['files'] == ['a.txt']
    assert broker.snapshot('alice')['alice']['succeeded'] == 4


@pytest.mark.parametrize('path', ['../bob/secret', '/tmp/secret', 'a/../../secret', 'a//b', './x', 'a\\b', 'a/./b', ''])
def test_path_traversal_rejected_without_leaking_paths(tmp_path, path):
    broker = broker_at(tmp_path)
    result = broker.execute('alice', 'write_text', {'path': path, 'text': 'secret'})
    assert result == {'ok': False, 'error': 'invalid_path'}
    assert list((broker.root / 'alice').iterdir()) == []


def test_symlink_and_hardlink_cannot_cross_agent_workspaces(tmp_path):
    broker = broker_at(tmp_path)
    (broker.root / 'bob' / 'secret').write_text('private')
    (broker.root / 'alice' / 'link').symlink_to(broker.root / 'bob', target_is_directory=True)
    (broker.root / 'alice' / 'filelink').symlink_to(broker.root / 'bob' / 'secret')
    os.link(broker.root / 'bob' / 'secret', broker.root / 'alice' / 'hardlink')
    for name in ('link/secret', 'filelink', 'hardlink'):
        assert not broker.execute('alice', 'read_text', {'path': name})['ok']
        assert not broker.execute('alice', 'write_text', {'path': name, 'text': 'changed'})['ok']
    assert (broker.root / 'bob' / 'secret').read_text() == 'private'


def test_invalid_identifiers_and_workspace_symlink(tmp_path):
    broker = ToolBroker(tmp_path)
    for name in ('../bob', 'a/b', '.', '', 'a.b', 'é'):
        with pytest.raises(ValueError, match='identifier'):
            broker.register(name)
        assert broker.execute(name, 'list_files', {})['error'] == 'invalid_agent'
    (tmp_path / 'alice').symlink_to(tmp_path / 'elsewhere', target_is_directory=True)
    with pytest.raises(ValueError, match='workspace'):
        broker.register('alice')


def test_grants_are_explicit_and_denied_before_launch(tmp_path, monkeypatch):
    broker = ToolBroker(tmp_path)
    broker.register('alice')
    def forbidden(*args, **kwargs):
        raise AssertionError('denied capabilities cannot launch a worker')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert broker.execute('alice', 'list_files', {})['error'] == 'tool_denied'
    assert broker.execute('alice', 'shell', {'command': 'anything'})['error'] == 'tool_denied'
    assert broker.execute('unknown', 'read_text', {})['error'] == 'unregistered_agent'


def test_quota_cannot_be_reset_by_registration(tmp_path):
    broker = ToolBroker(tmp_path, ToolLimits(max_calls=2))
    broker.register('alice', allowed_tools=('list_files',))
    assert broker.execute('alice', 'shell', {})['error'] == 'tool_denied'
    assert broker.execute('alice', 'list_files', {})['ok']
    assert broker.execute('alice', 'list_files', {})['error'] == 'call_quota'
    assert broker.snapshot()['alice'] == dict(allowed_tools=['list_files'], max_calls=2,
        calls=2, remaining_calls=0, succeeded=1, failed=1)
    with pytest.raises(ValueError, match='already'):
        broker.register('alice')


def test_clean_environment_and_isolated_interpreter(tmp_path, monkeypatch):
    broker = broker_at(tmp_path)
    original = subprocess.Popen
    seen = []
    monkeypatch.setenv('SECRET_TOKEN_FOR_TEST', 'not-in-child')
    def observe(command, **kwargs):
        seen.append((command, kwargs))
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', observe)
    assert broker.execute('alice', 'list_files', {})['ok']
    command, kwargs = seen[0]
    assert command[0] == sys.executable and command[1] == '-I'
    assert kwargs['env'] == {'PATH': os.defpath, 'LANG': 'C.UTF-8'}
    assert kwargs['cwd'] == broker.root / 'alice'
    assert kwargs['start_new_session'] is True
    assert 'shell' not in kwargs


def test_input_file_workspace_and_file_count_limits(tmp_path):
    limits = ToolLimits(max_input_bytes=1024, max_file_bytes=10, max_workspace_bytes=15, max_files=2)
    broker = broker_at(tmp_path, limits)
    assert broker.execute('alice', 'write_text', {'path': 'a', 'text': 'x' * 11})['error'] == 'file_limit'
    assert broker.execute('alice', 'write_text', {'path': 'a', 'text': 'x' * 10})['ok']
    assert broker.execute('alice', 'write_text', {'path': 'b', 'text': 'y' * 6})['error'] == 'workspace_quota'
    assert broker.execute('alice', 'write_text', {'path': 'b', 'text': 'y' * 5})['ok']
    assert broker.execute('alice', 'write_text', {'path': 'c', 'text': ''})['error'] == 'workspace_quota'
    assert broker.execute('alice', 'write_text', {'path': 'a', 'text': 'x'})['ok']
    assert broker.execute('alice', 'write_text', {'path': 'a', 'text': 'x' * 2048})['error'] == 'input_limit'
    assert len(list((broker.root / 'alice').iterdir())) == 2


def test_directory_entry_budget_reserved_before_creation(tmp_path):
    broker = broker_at(tmp_path, ToolLimits(max_entries=2))
    assert broker.execute('alice', 'write_text', {'path': 'a/b/c', 'text': ''})['error'] == 'workspace_entries_limit'
    assert list((broker.root / 'alice').iterdir()) == []


def test_read_and_list_output_limits(tmp_path):
    broker = broker_at(tmp_path, ToolLimits(max_output_bytes=80, max_file_bytes=100))
    (broker.root / 'alice' / 'text').write_text('x' * 90)
    assert broker.execute('alice', 'read_text', {'path': 'text'})['error'] == 'output_limit'
    (broker.root / 'alice' / 'text').write_text('x' * 101)
    assert broker.execute('alice', 'read_text', {'path': 'text'})['error'] == 'file_limit'
    for i in range(20):
        (broker.root / 'alice' / ('long_name_' + str(i))).touch()
    assert broker.execute('alice', 'list_files', {})['error'] == 'output_limit'


def test_worker_timeout_terminates_and_reaps_owned_child(tmp_path, monkeypatch):
    broker = broker_at(tmp_path, ToolLimits(timeout_seconds=.01))
    original = subprocess.Popen
    children = []
    def slow_worker(command, **kwargs):
        child = original([sys.executable, '-I', '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', slow_worker)
    assert broker.execute('alice', 'list_files', {})['error'] == 'timeout'
    assert children[0].poll() is not None


def test_parent_output_read_is_bounded_even_for_unexpected_worker(tmp_path, monkeypatch):
    broker = broker_at(tmp_path, ToolLimits(max_output_bytes=80))
    original = subprocess.Popen
    def overproducing_worker(command, **kwargs):
        return original([sys.executable, '-I', '-c', 'import sys; sys.stdout.write("x" * 100000)'], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', overproducing_worker)
    assert broker.execute('alice', 'list_files', {})['error'] == 'output_limit'


def test_global_worker_cap_rejects_without_waiting_or_launching(tmp_path, monkeypatch):
    broker = broker_at(tmp_path, ToolLimits(max_workers=1))
    entered, release = Event(), Event()
    def blocked_run(agent_id, payload):
        entered.set()
        assert release.wait(2)
        return {'ok': True, 'result': {'files': []}}
    monkeypatch.setattr(broker, '_run', blocked_run)
    first = Thread(target=lambda: broker.execute('alice', 'list_files', {}))
    first.start()
    assert entered.wait(2)
    try:
        assert broker.execute('bob', 'list_files', {})['error'] == 'workers_busy'
    finally:
        release.set()
        first.join(2)
    assert not first.is_alive()
    assert broker.execute('bob', 'list_files', {})['ok']


def test_durable_grants_and_consumed_quotas_survive_restart(tmp_path):
    broker = ToolBroker(tmp_path, ToolLimits(max_calls=2))
    broker.register('alice', allowed_tools=('list_files',))
    assert broker.execute('alice', 'list_files', {})['ok']
    broker.close()
    resumed = ToolBroker(tmp_path, ToolLimits(max_calls=2))
    assert resumed.snapshot()['alice']['calls'] == 1
    assert resumed.execute('alice', 'shell', {})['error'] == 'tool_denied'
    assert resumed.execute('alice', 'list_files', {})['error'] == 'call_quota'
    resumed.close()


def test_exclusive_root_lease_and_invalid_state_fail_closed(tmp_path):
    broker = ToolBroker(tmp_path)
    with pytest.raises(ValueError, match='lease'):
        ToolBroker(tmp_path)
    broker.close()
    with pytest.raises(ValueError, match='persisted'):
        ToolBroker(tmp_path, ToolLimits(max_calls=1))
    (tmp_path / '.broker.json').write_text('not json')
    with pytest.raises(ValueError, match='persisted'):
        ToolBroker(tmp_path)


def test_crash_consumption_is_not_replayed_or_reset(tmp_path):
    broker = ToolBroker(tmp_path)
    broker.register('alice', allowed_tools=('write_text',))
    broker.close()
    path = tmp_path / '.broker.json'
    state = json.loads(path.read_text())
    state['agents']['alice']['calls'] = 1
    # Simulate death after durable consumption, before result persistence.
    path.write_text(json.dumps(state))
    resumed = ToolBroker(tmp_path)
    assert resumed.snapshot()['alice']['calls'] == 1
    assert resumed.snapshot()['alice']['failed'] == 1
    assert list((tmp_path / 'alice').iterdir()) == []
    resumed.close()


def test_consumption_storage_failure_prevents_dispatch(tmp_path, monkeypatch):
    broker = broker_at(tmp_path)
    def fail_persistence():
        raise OSError('private path must not leak')
    def forbidden(*args):
        raise AssertionError('nothing dispatched before durable consumption')
    monkeypatch.setattr(broker, '_persist', fail_persistence)
    monkeypatch.setattr(broker, '_run', forbidden)
    assert broker.execute('alice', 'list_files', {}) == {'ok': False, 'error': 'state_storage'}


def test_memory_preflight_refuses_worker_without_launch(tmp_path, monkeypatch):
    from airllm.nextgen import experiment_guard
    broker = broker_at(tmp_path)
    monkeypatch.setattr(experiment_guard, 'system_snapshot', lambda: dict(
        available_bytes=0, physical_bytes=16 << 30, swap_used_bytes=0, pressure_level=1))
    def forbidden(*args, **kwargs):
        raise AssertionError('worker launched under memory pressure')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert broker.execute('alice', 'list_files', {}) == {'ok': False, 'error': 'memory_pressure'}


def test_close_keeps_lease_until_active_worker_and_accounting_finish(tmp_path, monkeypatch):
    broker = broker_at(tmp_path)
    entered, release = Event(), Event()
    def blocked_run(agent_id, payload):
        entered.set()
        assert release.wait(2)
        return {'ok': True, 'result': {'files': []}}
    monkeypatch.setattr(broker, '_run', blocked_run)
    first = Thread(target=lambda: broker.execute('alice', 'list_files', {}))
    first.start()
    assert entered.wait(2)
    closer = Thread(target=broker.close)
    closer.start()
    with broker._idle:
        assert broker._idle.wait_for(lambda: broker._closing, 2)
    try:
        with pytest.raises(ValueError, match='lease'):
            ToolBroker(broker.root)
        assert broker.execute('bob', 'list_files', {})['error'] == 'broker_closed'
    finally:
        release.set()
        first.join(2)
        closer.join(2)
    assert not first.is_alive() and not closer.is_alive()
    with ToolBroker(broker.root) as resumed:
        assert resumed.snapshot()['alice']['succeeded'] == 1


def test_nested_arguments_rejected_before_serialization(tmp_path, monkeypatch):
    broker = broker_at(tmp_path)
    nested = []
    for _ in range(1500):
        nested = [nested]
    assert broker.execute('alice', 'write_text', {'path': 'a', 'text': nested})['error'] == 'invalid_arguments'


def test_stale_atomic_snapshot_cleanup_preserves_published_state(tmp_path):
    with ToolBroker(tmp_path) as broker:
        broker.register('alice', allowed_tools=('list_files',))
    (tmp_path / '.broker-crash').write_text('interrupted snapshot')
    with ToolBroker(tmp_path) as resumed:
        assert not (tmp_path / '.broker-crash').exists()
        assert resumed.snapshot()['alice']['calls'] == 0
        assert (tmp_path / '.broker.json').is_file()


@pytest.mark.parametrize('kwargs', [{'max_calls': 0}, {'max_file_bytes': True}, {'timeout_seconds': float('nan')}, {'timeout_seconds': 0}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ToolLimits(**kwargs)
