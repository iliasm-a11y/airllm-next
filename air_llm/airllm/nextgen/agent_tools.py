"""Explicit, bounded file capabilities in isolated worker processes.

There is no shell, arbitrary Python or network tool. This is capability and
workspace isolation, not an OS security sandbox for untrusted executable code.
Never automatically dispatch model text to this broker.
"""
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore, Condition, Lock


TOOLS = frozenset(('read_text', 'write_text', 'list_files'))


@dataclass(frozen=True)
class ToolLimits:
    max_input_bytes: int = 65536
    max_output_bytes: int = 65536
    max_file_bytes: int = 32768
    max_workspace_bytes: int = 1024 * 1024
    max_files: int = 128
    max_entries: int = 256
    max_calls: int = 100
    max_agents: int = 10000
    max_workers: int = 2
    timeout_seconds: float = 2.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'timeout_seconds':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('timeout_seconds must be finite and positive')
            elif type(value) is not int or value < 1:
                raise ValueError('Tool byte and count limits must be positive integers')


def _error(code):
    return {'ok': False, 'error': code}


def _agent_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Invalid agent identifier')
    return value


class ToolBroker:
    """Grant tools explicitly; failed attempts also consume the call quota.

    A workspace belongs to exactly one registered agent. Calls for each agent
    serialize; different agents may use separate bounded workers. Tool grants
    and consumed quotas persist before dispatch. An exclusive root lease
    prevents independent broker processes from racing the same quotas.
    """
    def __init__(self, root, limits=None):
        self.limits = limits or ToolLimits()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._agents = {}
        self._lock = Lock()
        self._idle = Condition(self._lock)
        self._active_calls = 0
        self._closing = False
        self._workers = BoundedSemaphore(self.limits.max_workers)
        self._lease = None
        self._state_path = self.root / '.broker.json'
        try:
            self._lease = os.open(self.root / '.broker.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._cleanup_stale_snapshots()
            self._restore()
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            self.close()
            raise ValueError('Tool workspace lease or persisted state is invalid') from None

    def close(self):
        with self._idle:
            self._closing = True
            self._idle.notify_all()
            while self._active_calls:
                self._idle.wait()
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, '_lease', None) is not None:
            self.close()

    def _cleanup_stale_snapshots(self):
        # A process death during atomic publication can leave a temp snapshot.
        # Recover under the exclusive root lease, with bounded work and bytes.
        count = total = 0
        with os.scandir(self.root) as items:
            for item in items:
                if not item.name.startswith('.broker-'):
                    continue
                info = item.stat(follow_symlinks=False)
                count += 1
                total += info.st_size
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or count > 32 or
                        total > 32 * (self.limits.max_agents * 1024 + 4096)):
                    raise ValueError('Invalid stale tool state')
                os.unlink(item.path)
        if count:
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def _restore(self):
        if not self._state_path.exists() and not self._state_path.is_symlink():
            self._persist()
            return
        with os.fdopen(os.open(self._state_path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
            raw = stream.read(self.limits.max_agents * 1024 + 4097)
        if len(raw) > self.limits.max_agents * 1024 + 4096:
            raise ValueError('State limit')
        state = json.loads(raw)
        if state['schema'] != 1 or state['limits'] != asdict(self.limits):
            raise ValueError('State configuration mismatch')
        agents = state['agents']
        if not isinstance(agents, dict) or len(agents) > self.limits.max_agents:
            raise ValueError('Invalid agents')
        for agent_id, record in agents.items():
            _agent_id(agent_id)
            allowed = frozenset(record['allowed'])
            if not allowed <= TOOLS:
                raise ValueError('Invalid grants')
            for key in ('max_calls', 'calls', 'succeeded', 'failed'):
                if type(record[key]) is not int or record[key] < 0:
                    raise ValueError('Invalid accounting')
            if not 1 <= record['max_calls'] <= self.limits.max_calls or record['calls'] > record['max_calls']:
                raise ValueError('Invalid quota')
            accounted = record['succeeded'] + record['failed']
            if accounted > record['calls']:
                raise ValueError('Invalid accounting')
            workspace = self.root / agent_id
            if workspace.is_symlink() or not workspace.is_dir() or workspace.resolve().parent != self.root:
                raise ValueError('Invalid workspace')
            self._agents[agent_id] = dict(record, allowed=allowed, lock=Lock())
            # A crash between pre-dispatch consumption and result accounting
            # consumes its call permanently, without replaying any file write.
            self._agents[agent_id]['failed'] += record['calls'] - accounted
        self._persist()

    def _persist(self):
        state = dict(schema=1, limits=asdict(self.limits), agents={
            key: dict(allowed=sorted(record['allowed']), **{
                field: record[field] for field in ('max_calls', 'calls', 'succeeded', 'failed')})
            for key, record in self._agents.items()})
        raw = json.dumps(state, sort_keys=True, allow_nan=False).encode('utf-8')
        if len(raw) > self.limits.max_agents * 1024 + 4096:
            raise ValueError('Tool state limit')
        handle, name = tempfile.mkstemp(prefix='.broker-', dir=self.root)
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self._state_path)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def register(self, agent_id, *, allowed_tools=(), max_calls=None):
        agent_id = _agent_id(agent_id)
        allowed = frozenset(allowed_tools)
        if not allowed <= TOOLS:
            raise ValueError('Unknown tool capability')
        cap = self.limits.max_calls if max_calls is None else max_calls
        if type(cap) is not int or cap < 1 or cap > self.limits.max_calls:
            raise ValueError('Invalid agent call quota')
        with self._lock:
            if self._lease is None or self._closing:
                raise ValueError('Tool broker is closed')
            if agent_id in self._agents:
                raise ValueError('Agent is already registered')
            if len(self._agents) >= self.limits.max_agents:
                raise ValueError('Agent registration quota exhausted')
            workspace = self.root / agent_id
            if workspace.is_symlink():
                raise ValueError('Invalid agent workspace')
            workspace.mkdir(mode=0o700, exist_ok=True)
            if not workspace.is_dir() or workspace.resolve().parent != self.root:
                raise ValueError('Invalid agent workspace')
            self._agents[agent_id] = dict(allowed=allowed, max_calls=cap, calls=0,
                                         succeeded=0, failed=0, lock=Lock())
            try:
                self._persist()
            except (OSError, ValueError):
                del self._agents[agent_id]
                raise ValueError('Tool grant could not be persisted') from None

    def snapshot(self, agent_id=None):
        with self._lock:
            records = self._agents if agent_id is None else {agent_id: self._agents[agent_id]}
            return {key: dict(allowed_tools=sorted(record['allowed']),
                              max_calls=record['max_calls'], calls=record['calls'],
                              remaining_calls=record['max_calls'] - record['calls'],
                              succeeded=record['succeeded'], failed=record['failed'])
                    for key, record in records.items()}

    def execute(self, agent_id, tool, arguments):
        try:
            agent_id = _agent_id(agent_id)
        except ValueError:
            return _error('invalid_agent')
        with self._lock:
            if self._lease is None or self._closing:
                return _error('broker_closed')
            record = self._agents.get(agent_id)
        if record is None:
            return _error('unregistered_agent')
        with record['lock']:
            with self._lock:
                # Recheck after waiting for another call of the same agent.
                if self._lease is None or self._closing:
                    return _error('broker_closed')
                self._active_calls += 1
            try:
                return self._execute_registered(agent_id, tool, arguments, record)
            finally:
                with self._idle:
                    self._active_calls -= 1
                    self._idle.notify_all()

    def _execute_registered(self, agent_id, tool, arguments, record):
        with self._lock:
            if record['calls'] >= record['max_calls']:
                return _error('call_quota')
            record['calls'] += 1
            try:
                self._persist()
            except (OSError, ValueError):
                return _error('state_storage')
        if not isinstance(tool, str) or tool not in record['allowed']:
            result = _error('tool_denied')
        elif not isinstance(arguments, dict):
            result = _error('invalid_arguments')
        elif (len(arguments) > (2 if tool == 'write_text' else 1) or
              any(key not in ({'path', 'text'} if tool == 'write_text' else {'path'})
                  or not isinstance(value, str) for key, value in arguments.items())):
            result = _error('invalid_arguments')
        elif any(len(value) > self.limits.max_input_bytes for value in arguments.values()):
            # Reject huge strings before JSON encoding allocates another
            # complete copy; nested structures are not tool arguments.
            result = _error('input_limit')
        else:
            try:
                payload = json.dumps(dict(tool=tool, arguments=arguments,
                                          limits=asdict(self.limits)),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')
            except (TypeError, ValueError, UnicodeError):
                result = _error('invalid_arguments')
            else:
                if len(payload) > self.limits.max_input_bytes:
                    result = _error('input_limit')
                elif not self._workers.acquire(blocking=False):
                    result = _error('workers_busy')
                else:
                    try:
                        result = self._run(agent_id, payload)
                    finally:
                        self._workers.release()
        with self._lock:
            record['succeeded' if result['ok'] else 'failed'] += 1
            try:
                self._persist()
            except (OSError, ValueError):
                # The consumed call was already durable. Preserve the
                # actual operation result while flagging final accounting.
                result['accounting_error'] = 'state_storage'
        return result

    def _run(self, agent_id, payload):
        from .experiment_guard import ExperimentLimits, admission
        try:
            admission(ExperimentLimits(admitted_payload_bytes=32 << 20))
        except MemoryError:
            return _error('memory_pressure')
        workspace = self.root / agent_id
        if workspace.is_symlink() or workspace.resolve().parent != self.root:
            return _error('invalid_workspace')
        # A file-backed output keeps parent memory bounded even if a worker
        # unexpectedly misbehaves. The trusted worker also bounds its emission.
        with tempfile.TemporaryFile() as stdout:
            try:
                worker = subprocess.Popen(
                    [sys.executable, '-I', str(Path(__file__).resolve()), '--worker'],
                    cwd=workspace, env={'PATH': os.defpath, 'LANG': 'C.UTF-8'},
                    stdin=subprocess.PIPE, stdout=stdout, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                try:
                    worker.communicate(payload, timeout=self.limits.timeout_seconds)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.communicate()
                    return _error('timeout')
                stdout.seek(0)
                output = stdout.read(self.limits.max_output_bytes + 1)
                if len(output) > self.limits.max_output_bytes:
                    return _error('output_limit')
                if worker.returncode != 0:
                    return _error('worker_failed')
                result = json.loads(output)
                if not isinstance(result, dict) or type(result.get('ok')) is not bool:
                    return _error('worker_failed')
                return result
            except (OSError, ValueError, UnicodeError):
                return _error('worker_failed')


class _ToolFailure(Exception):
    pass


def _parts(value, *, root=False):
    if root and value == '.':
        return []
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise _ToolFailure('invalid_path')
    parts = value.split('/')
    if len(parts) > 16 or any(not p or p in ('.', '..') or len(p.encode('utf-8')) > 255 for p in parts):
        raise _ToolFailure('invalid_path')
    return parts


def _directory(parts, *, create=False):
    fd = os.open('.', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _workspace_usage(limits):
    count = total = entries = 0
    for parent, dirs, files in os.walk('.', followlinks=False):
        entries += len(dirs) + len(files)
        if entries > limits.max_entries:
            raise _ToolFailure('workspace_entries_limit')
        for name in files:
            info = os.lstat(os.path.join(parent, name))
            if stat.S_ISREG(info.st_mode):
                count += 1
                total += info.st_size
        if count > limits.max_files or total > limits.max_workspace_bytes:
            raise _ToolFailure('workspace_quota')
    return count, total, entries


def _operation(tool, arguments, limits):
    expected = {'path', 'text'} if tool == 'write_text' else {'path'}
    if tool not in TOOLS or set(arguments) - expected:
        raise _ToolFailure('invalid_arguments')
    path = arguments.get('path', '.' if tool == 'list_files' else None)
    parts = _parts(path, root=tool == 'list_files')
    if tool == 'list_files':
        fd = _directory(parts)
        try:
            # scandir streams, avoiding an unbounded directory-name list.
            names = []
            with os.scandir(fd) as items:
                for item in items:
                    names.append(item.name)
                    if len(names) > limits.max_entries:
                        raise _ToolFailure('entry_limit')
            return {'files': sorted(names)}
        finally:
            os.close(fd)
    if tool == 'write_text':
        if not isinstance(arguments.get('text'), str):
            raise _ToolFailure('invalid_arguments')
        data = arguments['text'].encode('utf-8')
        if len(data) > limits.max_file_bytes:
            raise _ToolFailure('file_limit')
        count, used, entries = _workspace_usage(limits)
        # Reserve for every possible new path component before mkdir touches
        # disk. Conservative overcounting is preferable near the entry cap.
        if entries + len(parts) > limits.max_entries:
            raise _ToolFailure('workspace_entries_limit')
    parent = _directory(parts[:-1], create=tool == 'write_text')
    fd = None
    try:
        if tool == 'read_text':
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        else:
            try:
                fd = os.open(parts[-1], os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            except FileNotFoundError:
                if count >= limits.max_files or entries + len(parts) > limits.max_entries or used + len(data) > limits.max_workspace_bytes:
                    raise _ToolFailure('workspace_quota')
                fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             mode=0o600, dir_fd=parent)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise _ToolFailure('invalid_file')
        if tool == 'read_text':
            if info.st_size > limits.max_file_bytes:
                raise _ToolFailure('file_limit')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                data = stream.read(limits.max_file_bytes + 1)
            if len(data) > limits.max_file_bytes:
                raise _ToolFailure('file_limit')
            return {'text': data.decode('utf-8')}
        if used - info.st_size + len(data) > limits.max_workspace_bytes:
            raise _ToolFailure('workspace_quota')
        os.ftruncate(fd, 0)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
        return {'bytes_written': len(data)}
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _worker():
    # The parent sends only bounded JSON. Read a fixed ceiling even if invoked
    # manually; no pickle, importable task or executable command is accepted.
    try:
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise _ToolFailure('input_limit')
        request = json.loads(raw)
        limits = ToolLimits(**request['limits'])
        if len(raw) > limits.max_input_bytes:
            raise _ToolFailure('input_limit')
        result = {'ok': True, 'result': _operation(request['tool'], request['arguments'], limits)}
    except _ToolFailure as exc:
        result = _error(str(exc))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        result = _error('tool_failed')
    output = json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
    cap = limits.max_output_bytes if 'limits' in locals() else 1024
    if len(output) > cap:
        output = json.dumps(_error('output_limit')).encode('utf-8')
    # Extremely small configured caps cannot even represent a result object.
    sys.stdout.buffer.write(output[:cap])


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        _worker()
    else:
        raise SystemExit('This module only provides the ToolBroker API')
