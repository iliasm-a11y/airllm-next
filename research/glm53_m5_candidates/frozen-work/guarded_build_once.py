"""One owned build, under the original system guards and exclusive research locks.

No model is loaded here. Process-tree RSS is a conservative sum (shared pages
may count more than once), separately capped at 2 GiB. Original RSS8.5GiB,
swap32MiB, pressure/reserve and power/continuity protections are retained.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import psutil

WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gpu_guarded_build_import', WORK/'metal-cpu-q8-bridge/native_gpu_run.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def run(command, output):
    if output.exists():
        raise FileExistsError(output)
    limits = native.NativeLimits(**json.loads((WORK/'cpu10-reference.json').read_text())['limits'])
    assert limits.max_rss_bytes <= int(8.5*1024**3) and limits.max_swap_growth_bytes <= 32*1024**2
    guard = native.guard_model(limits)
    gate = native.PowerGate('ac')
    report = {'scope': 'build only; no model or numerical/performance qualification',
              'command': command, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'starting', 'samples': [], 'tree_rss_cap_bytes': 2*1024**3,
              'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'native_limits': vars(limits), 'guard_violation': None, 'signals': []}
    child = awake = holder = None
    go_read = go_write = ready_read = ready_write = None
    identities = {}
    started = time.monotonic()
    def save():
        native.write(output, report)
    def remember(process):
        row = {'pid': process.pid, 'create_time': process.create_time(), 'executable': process.exe()}
        previous = identities.get(process.pid)
        if previous is not None and previous['create_time'] != row['create_time']:
            raise RuntimeError('Owned process lifetime changed')
        if previous is not None and previous['executable'] != row['executable']:
            report.setdefault('owned_exec_transitions', []).append({'before': previous, 'after': row})
        identities[process.pid] = row
        return row
    def still_owned(row):
        try:
            process = psutil.Process(row['pid'])
            return (abs(process.create_time()-row['create_time']) < .01 and
                    process.exe() == row['executable'] and process.is_running())
        except psutil.NoSuchProcess:
            return False
    def stop_owned_tree():
        if child is None:
            return
        for row in list(identities.values()):
            try:
                process = psutil.Process(row['pid'])
                if process.create_time() == row['create_time']:
                    remember(process)  # record a normal owned compiler exec
            except psutil.NoSuchProcess:
                pass
        # Only children actually observed below our retained Popen can enter
        # this registry; birth/executable are rechecked before each signal.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for row in list(identities.values()):
                if still_owned(row):
                    os.kill(row['pid'], sig)
                    report['signals'].append({'pid': row['pid'], 'birth': row['create_time'], 'signal': sig.name})
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            if not any(still_owned(row) for row in identities.values()):
                break
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal '+str(signum))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with native.qa.BENCHMARK_LOCK.open('a') as benchmark:
            fcntl.flock(benchmark, fcntl.LOCK_EX | fcntl.LOCK_NB)
            holder, report['uid_lock'] = native.acquire_uid_model_lock()
            if not gate.sample()['passed']:
                raise RuntimeError('Power admission refused')
            baseline = native.resource_snapshot()
            report['baseline'] = baseline
            if baseline['pressure_level'] >= 4 or baseline['available_bytes'] < 3*1024**3:
                raise RuntimeError('Build reserve admission refused')
            with output.with_suffix('.stdout.log').open('x') as stdout, output.with_suffix('.stderr.log').open('x') as stderr:
                go_read, go_write = os.pipe()
                ready_read, ready_write = os.pipe()
                bootstrap = ('import os,runpy,sys; '
                             'fd=int(sys.argv[1]); ready=int(sys.argv[2]); '
                             'os.write(ready,b"R"); os.close(ready); '
                             'token=os.read(fd,1); os.close(fd); '
                             'assert token==b"G", "build start refused"; '
                             'sys.argv=sys.argv[3:]; '
                             'sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); '
                             'runpy.run_path(sys.argv[0],run_name="__main__")')
                report['startup_gate'] = 'child waits before build script until UID/PID observation'
                child = subprocess.Popen([command[0], '-c', bootstrap, str(go_read), str(ready_write), *command[1:]],
                                         stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                         pass_fds=(holder.fileno(), go_read, ready_write))
                os.close(go_read)
                go_read = None
                os.close(ready_write)
                ready_write = None
                if not select.select([ready_read], [], [], 5)[0] or os.read(ready_read, 1) != b'R':
                    raise RuntimeError('Build interpreter did not reach its blocked startup gate')
                os.close(ready_read)
                ready_read = None
                owner = remember(psutil.Process(child.pid))
                report['owned_process'] = owner
                # PowerGate may attach the supervising framework Python to
                # Python.app. The pure CLI builder keeps its declared binary.
                allowed_python = {str(Path(command[0]).resolve()),
                                  str((Path(sys.base_prefix)/'Resources/Python.app/Contents/MacOS/Python').resolve())}
                report['allowed_interpreters'] = sorted(allowed_python)
                if owner['executable'] not in allowed_python:
                    raise RuntimeError('Build interpreter differs')
                expected_python = owner['executable']
                report['declared_interpreter'] = str(Path(command[0]).resolve())
                report['uid_lock']['pass_fds_requested'] = True
                report['uid_fd_observation'] = native.observe_uid_fd(child, Path(expected_python), owner, report['uid_lock'])
                if not report['uid_fd_observation']['observed_matching_lock_fd']:
                    raise RuntimeError('Build child did not inherit model UID lock')
                awake = native.ScopedAwake(child)
                awake.__enter__()
                report['status'] = 'running'
                save()
                os.write(go_write, b'G')
                os.close(go_write)
                go_write = None
                while child.poll() is None:
                    try:
                        parent = psutil.Process(child.pid)
                        remember(parent)
                    except psutil.NoSuchProcess:
                        if child.poll() is not None:
                            break
                        raise
                    tree_rss = 0
                    for process in [parent, *parent.children(recursive=True)]:
                        try:
                            remember(process)
                            tree_rss += process.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    sample = native.resource_snapshot(child.pid)
                    sample.update(rss_bytes=tree_rss, t=time.monotonic()-started)
                    sample['swap_growth_bytes'] = max(0, sample['swap_used_bytes']-baseline['swap_used_bytes'])
                    report['samples'].append(sample)
                    code = native.NativeGlmModel._guard_resource_sample(guard, sample, baseline, time.monotonic())
                    awake.check()
                    if not gate.sample()['passed']:
                        code = 'POWER_CONTEXT_CHANGED'
                    if tree_rss > report['tree_rss_cap_bytes']:
                        code = code or 'BUILD_TREE_RSS_LIMIT'
                    if sample['t'] > 1200:
                        code = code or 'BUILD_DEADLINE'
                    if code:
                        report['guard_violation'] = {'code': code, 'sample': sample}
                        raise RuntimeError(code)
                    if len(report['samples']) % 8 == 0:
                        save()
                    time.sleep(.5)
                report['returncode'] = child.wait(timeout=5)
                report['status'] = 'completed' if child.returncode == 0 else 'failed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        stop_owned_tree()
        if child is not None:
            report['returncode'] = child.poll()
    finally:
        for fd in (go_read, go_write, ready_read, ready_write):
            if fd is not None:
                os.close(fd)
        if child is not None and child.poll() is None:
            stop_owned_tree()
        if awake is not None:
            awake.close()
        report['power_final'] = gate.sample(final=True)
        report['power'] = gate.summary()
        if not report['power_final']['passed'] or not report['power']['passed']:
            report['status'] = 'failed'
            report['guard_violation'] = report['guard_violation'] or {'code': 'POWER_CONTEXT_CHANGED'}
        if holder is not None:
            holder.close()
        report['uid_closed'] = holder is None or holder.closed
        report['observed_processes'] = list(identities.values())
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        for s, handler in previous.items():
            signal.signal(s, handler)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or command[0] != sys.executable:
        raise ValueError('Explicit same-interpreter build command required')
    result = run(command, args.output)
    print(json.dumps({k: result.get(k) for k in ('status', 'returncode', 'error', 'guard_violation', 'elapsed_seconds')}))
    raise SystemExit(0 if result['status'] == 'completed' else 2)
