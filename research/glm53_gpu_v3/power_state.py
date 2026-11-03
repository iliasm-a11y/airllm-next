"""Read-only lid/power admission, clock discontinuity guard and scoped idle assertion.

No persistent power settings, system-sleep override or closed-lid bypass is used.
Wall/perf disagreement rejects timing evidence; it cannot uniquely diagnose sleep.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
import platform
import plistlib
import re
import selectors
import subprocess
import threading
import time

import psutil

SOURCES = ('ac', 'battery')


def _command(argv, limit=128*1024, timeout=3):
    """Bound stdout while reading, with a deadline and cleanup of this child only."""
    env = dict(os.environ, LC_ALL='C', LANG='C')
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env) as child:
        try:
            data = bytearray()
            deadline = time.perf_counter()+timeout
            with selectors.DefaultSelector() as select:
                select.register(child.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0 or not select.select(remaining):
                        raise TimeoutError('power query timeout')
                    chunk = os.read(child.stdout.fileno(), min(16384, limit+1-len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > limit:
                        raise ValueError('power query exceeded output budget')
            if child.wait(timeout=max(.01, deadline-time.perf_counter())) != 0:
                raise ValueError('power query returned nonzero')
            return bytes(data)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)


def parse_lid(data):
    if len(data) > 128*1024:
        raise ValueError('ioreg plist exceeds metadata budget')
    rows = plistlib.loads(data)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError('ambiguous IORegistry root')
    value = rows[0].get('AppleClamshellState')
    if type(value) is not bool:
        raise ValueError('clamshell state is unknown')
    return value


def parse_power(data):
    if len(data) > 128*1024:
        raise ValueError('pmset text exceeds metadata budget')
    matches = re.findall(r"^Now drawing from '(AC Power|Battery Power)'\s*$", data.decode('utf-8'), re.M)
    if len(matches) != 1:
        raise ValueError('power source is unknown or ambiguous')
    return 'ac' if matches[0] == 'AC Power' else 'battery'


class PowerStateObserver:
    def __init__(self, command=_command, system=platform.system):
        self.command, self.system = command, system

    def snapshot(self):
        row = {'available': False, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'wall_time_seconds': time.time(), 'perf_counter_seconds': time.perf_counter(),
               'lid_closed': None, 'power_source': None, 'atomic_snapshot': False}
        try:
            if self.system() != 'Darwin':
                raise ValueError('target Mac power observer requires Darwin')
            row['lid_closed'] = parse_lid(self.command(['/usr/sbin/ioreg', '-a', '-r', '-k', 'AppleClamshellState', '-d', '1']))
            row['power_source'] = parse_power(self.command(['/usr/bin/pmset', '-g', 'batt']))
            row['available'] = True
        except (OSError, ValueError, TypeError, plistlib.InvalidFileException, subprocess.SubprocessError) as error:
            row['error'] = f'{type(error).__name__}: {error}'
        row['perf_counter_end_seconds'] = time.perf_counter()
        return row


def power_preflight(expected_source=None, observer=None):
    snapshot = (observer or PowerStateObserver()).snapshot()
    reasons = []
    if snapshot.get('available') is not True:
        reasons.append('power/lid state unavailable')
    if snapshot.get('lid_closed') is not False:
        reasons.append('target Mac lid is closed or unknown')
    source = snapshot.get('power_source')
    if source not in SOURCES:
        reasons.append('power source unknown')
    if expected_source is not None and (expected_source not in SOURCES or source != expected_source):
        reasons.append('power source changed from campaign admission')
    return {'passed': not reasons, 'reasons': reasons, 'snapshot': snapshot}


class ClockContinuity:
    def __init__(self, threshold_seconds=2, wall=time.time, perf=time.perf_counter):
        if not math.isfinite(threshold_seconds) or not 0 < threshold_seconds <= 2:
            raise ValueError('clock drift threshold must be positive and <=2 seconds')
        self.threshold, self.wall, self.perf = threshold_seconds, wall, perf
        self.initial = self.previous = None
        self.violation = None
        self.samples = 0

    def sample(self):
        wall, perf = self.wall(), self.perf()
        self.samples += 1
        row = {'wall_time_seconds': wall, 'perf_counter_seconds': perf, 'threshold_seconds': self.threshold}
        reason = None
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (wall, perf)):
            reason = 'clock value unavailable/nonfinite'
        else:
            if self.initial is None:
                self.initial = (wall, perf)
            drift = (wall-self.initial[0])-(perf-self.initial[1])
            row['wall_minus_perf_elapsed_seconds'] = drift
            row['timestamp_utc'] = datetime.fromtimestamp(wall, timezone.utc).isoformat()
            if self.previous and (wall < self.previous[0] or perf < self.previous[1]):
                reason = 'clock moved backwards'
            elif abs(drift) > self.threshold:
                reason = 'wall/perf elapsed disagreement: sleep or clock discontinuity'
            self.previous = (wall, perf)
        if reason and self.violation is None:
            self.violation = reason
        return dict(row, passed=self.violation is None, reason=self.violation)


class PowerGate:
    def __init__(self, expected_source=None, observer=None, clock=None):
        self.expected_source = expected_source
        self.observer, self.clock = observer or PowerStateObserver(), clock or ClockContinuity()
        self.first = self.last = self.violation = None
        self.power_samples = 0
        self.finalized = False
        self.lock = threading.Lock()

    def sample(self, *, include_power=True, final=False):
        with self.lock:
            row = {'clock': self.clock.sample()}
            if include_power:
                row['power'] = power_preflight(self.expected_source, self.observer)
                self.power_samples += 1
                if self.expected_source is None and row['power']['passed']:
                    self.expected_source = row['power']['snapshot']['power_source']
            row['passed'] = row['clock']['passed'] and row.get('power', {}).get('passed', True)
            if not row['passed'] and self.violation is None:
                self.violation = row
            if self.first is None:
                self.first = row
            self.last = row
            self.finalized = self.finalized or final
            return row

    def summary(self):
        return {'passed': self.finalized and self.violation is None and self.power_samples >= 2,
                'finalized': self.finalized, 'expected_power_source': self.expected_source,
                'clock_samples': self.clock.samples, 'power_samples': self.power_samples,
                'first': self.first, 'last': self.last, 'violation': self.violation,
                'scope': 'open lid and constant known power at sampled times; wall/perf drift<=2s, not proof of every hardware power event'}


class ScopedAwake:
    """Owned caffeinate -i -w childPID; it never overrides clamshell sleep."""
    def __init__(self, child, *, popen=subprocess.Popen, process=psutil.Process):
        self.child, self.popen, self.process = child, popen, process
        self.helper = self.identity = None
        self.record = {'started': False, 'cleaned_up': False, 'command': ['/usr/bin/caffeinate', '-i', '-w', str(child.pid)]}

    def _identity(self, pid, executable=None):
        value = self.process(pid)
        if not value.is_running() or value.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError('owned awake process is not running')
        if executable and os.path.realpath(value.exe()) != executable:
            raise RuntimeError('awake helper executable identity changed')
        return value.create_time()

    def __enter__(self):
        try:
            if type(self.child.pid) is not int or self.child.pid <= 0 or self.child.poll() is not None:
                raise RuntimeError('awake assertion requires a live owned server')
            target = self._identity(self.child.pid)
            self.helper = self.popen(self.record['command'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.identity = self._identity(self.helper.pid, '/usr/bin/caffeinate')
            self.record.update(started=True, helper_pid=self.helper.pid, helper_create_time=self.identity,
                               target_pid=self.child.pid, target_create_time=target)
            self.check()
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        try:
            if (self.child.poll() is not None or self._identity(self.child.pid) != self.record['target_create_time'] or
                    self.helper.poll() is not None or self._identity(self.helper.pid, '/usr/bin/caffeinate') != self.identity):
                raise RuntimeError('owned server/idle assertion identity or lifetime changed')
        except Exception as error:
            self.record['validation_error'] = f'{type(error).__name__}: {error}'
            raise

    def close(self):
        if self.helper is None or self.record.get('cleaned_up'):
            return
        try:
            self.record['owner_signal'] = None
            self.record['helper_alive_before_owner_signal'] = False
            if self.helper.poll() is None:
                # Popen owns this unreaped child. Verify its start identity before signalling.
                if self.identity is not None and self._identity(self.helper.pid, '/usr/bin/caffeinate') != self.identity:
                    raise RuntimeError('refusing to signal a changed awake-helper identity')
                self.record.update(owner_signal='SIGTERM', helper_alive_before_owner_signal=True)
                self.helper.terminate()
                try:
                    self.helper.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if self.helper.poll() is None:
                        self.record['owner_signal'] = 'SIGKILL'
                        self.helper.kill()
                    self.helper.wait(timeout=3)
            # A reaped helper is cleaned up even when its assertion lifetime was
            # invalid. Keep cleanup success separate from timing qualification.
            code = self.helper.returncode
            target_exited = self.child.poll() is not None
            signal = self.record['owner_signal']
            if signal == 'SIGKILL':
                kind, qualified = 'owner_kill', False
            elif code == 0 and target_exited:
                kind, qualified = 'natural_target_exit', True
            elif code == -15 and signal == 'SIGTERM':
                kind, qualified = 'owner_term', True
            else:
                kind, qualified = 'unexpected_exit', False
            self.record.update(cleaned_up=True, helper_exit_code=code, cleanup_kind=kind,
                               target_exited_at_cleanup=target_exited, exit_qualified=qualified)
            if not qualified:
                raise RuntimeError('idle assertion helper exit did not qualify: '+kind)
        except Exception as error:
            self.record['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise

    def __exit__(self, *unused):
        self.close()


def power_evidence_matches(report, expected_source):
    if not isinstance(report, dict):
        return False
    gate = report.get('power_gate') or {}
    if not isinstance(gate, dict):
        return False
    helper = gate.get('idle_assertion') or {}
    if not isinstance(helper, dict):
        return False
    edges = []
    for key in ('first', 'last'):
        edge = gate.get(key) or {}
        if not isinstance(edge, dict):
            return False
        power, clock = edge.get('power') or {}, edge.get('clock') or {}
        if not isinstance(power, dict) or not isinstance(clock, dict):
            return False
        state = power.get('snapshot') or {}
        if not isinstance(state, dict):
            return False
        edges.append(edge.get('passed') is True and clock.get('passed') is True and power.get('passed') is True and
                     state.get('available') is True and state.get('lid_closed') is False and
                     state.get('power_source') == expected_source)
    target, pid = helper.get('target_pid'), helper.get('helper_pid')
    identities = all(type(helper.get(key)) in (int, float) and math.isfinite(helper[key]) and helper[key] > 0
                     for key in ('target_create_time', 'helper_create_time'))
    code = helper.get('helper_exit_code')
    natural = (type(code) is int and code == 0 and helper.get('cleanup_kind') == 'natural_target_exit' and
               helper.get('target_exited_at_cleanup') is True and helper.get('owner_signal') in (None, 'SIGTERM'))
    owned_term = (type(code) is int and code == -15 and helper.get('cleanup_kind') == 'owner_term' and
                  helper.get('owner_signal') == 'SIGTERM' and helper.get('helper_alive_before_owner_signal') is True)
    return bool(all(edges) and expected_source in SOURCES and gate.get('passed') is True and gate.get('finalized') is True and
                gate.get('expected_power_source') == expected_source and gate.get('violation') is None and
                type(gate.get('clock_samples')) is int and gate['clock_samples'] >= 2 and
                type(gate.get('power_samples')) is int and gate['power_samples'] >= 2 and
                type(target) is int and target > 0 and target == report.get('owned_pid') and
                type(pid) is int and pid > 0 and pid != target and identities and
                helper.get('command') == ['/usr/bin/caffeinate', '-i', '-w', str(target)] and
                (natural or owned_term) and helper.get('exit_qualified') is True and helper.get('started') is True and
                helper.get('cleaned_up') is True and not helper.get('cleanup_error') and not helper.get('validation_error'))


if __name__ == '__main__':
    import json
    print(json.dumps(power_preflight(), indent=2))
