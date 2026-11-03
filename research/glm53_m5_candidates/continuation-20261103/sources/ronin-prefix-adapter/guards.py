"""Phase-aware adapter for existing power/native guards; no process is started.

The pre-load probe is released before NativeGlmModel.load acquires its own FD.
Only that retained native FD is accepted once a server exists. No second flock
is retained, no unrelated process is enumerated, and no PID is signalled here.
"""
from __future__ import annotations

import fcntl
import inspect
import os
from pathlib import Path
import stat
import tempfile
import threading
import time

from prefix_adapter import AdapterError, MiB, require, sha

# Additional admission reserve, not a measured HTTP peak: native tensor transfer
# <=16 MiB under the pinned state contract plus the 1 MiB Python hashing chunk.
SNAPSHOT_IO_RESERVE = 17 * MiB


class PhaseAwareGuard:
    def __init__(self, power_gate, native_limits_type, *, process_factory=None,
                 fd_observer=None, clock=time.monotonic, sample_seconds=.5):
        if process_factory is None:
            import psutil
            process_factory = psutil.Process
        require(0 < sample_seconds <= .5, "power/ownership sampling must be <=0.5s")
        require(callable(fd_observer), "existing owned-PID/FD observer required")
        self.power, self.limits_type = power_gate, native_limits_type
        self.process_factory, self.fd_observer = process_factory, fd_observer
        self.clock, self.interval = clock, sample_seconds
        self.lock = threading.RLock()
        self.limits = self.owner = self.last = None
        self.last_power = self.last_fd = float("-inf")
        self.fd_receipt = None
        self.samples = 0
        self.violation = None
        self.source_sha256 = {str(Path(inspect.getfile(value)).resolve()): sha(inspect.getfile(value))
                              for value in (type(power_gate), native_limits_type, fd_observer, type(self))}

    def provenance(self):
        require(all(sha(path) == digest for path, digest in self.source_sha256.items()),
                "external guard policy sources changed", "PREFIX_SOURCE")
        return dict(self.source_sha256)

    def _lock_path(self, model):
        root = Path(tempfile.gettempdir()) / ("airllm-native-locks-" + str(os.getuid()))
        root.mkdir(mode=0o700, exist_ok=True)
        s = root.lstat()
        require(stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and not s.st_mode & 0o077,
                "private native UID lock directory required", "PREFIX_UID")
        return root / (model._model_lock_key() + ".lock")

    def _fd_identity(self, holder, path):
        require(holder is not None and not holder.closed, "native UID descriptor is absent", "PREFIX_UID")
        actual, named = os.fstat(holder.fileno()), path.lstat()
        require(stat.S_ISREG(actual.st_mode) and stat.S_ISREG(named.st_mode) and
                actual.st_uid == named.st_uid == os.getuid() and actual.st_nlink == named.st_nlink == 1 and
                (actual.st_dev, actual.st_ino) == (named.st_dev, named.st_ino),
                "native UID descriptor no longer names the owned ordinary lock", "PREFIX_UID")
        return {"path": str(path), "fd": holder.fileno(), "uid": os.getuid(),
                "device": actual.st_dev, "inode": actual.st_ino}

    def _preload(self, model):
        path = self._lock_path(model)
        retained = getattr(model, "_lock_file", None)
        if retained is not None and not retained.closed:
            # Native load can already hold its own lock just before Popen. Do
            # not open/relock it, and do not call this a verified child owner.
            lock = self._fd_identity(retained, path)
            return {"no_conflicting_uid_engine": True, "uid_exclusive": False,
                    "identity_passed": False, "native_lock_pending_spawn": lock}
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "a") as probe:
                fd = None
                lock = self._fd_identity(probe, path)
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise AdapterError("ENGINE_BUSY", "UID model lock already held; no load admitted") from error
                finally:
                    # The probe is never held across native load. Native load
                    # must still acquire its own lock atomically to win races.
                    fcntl.flock(probe, fcntl.LOCK_UN)
            return {"no_conflicting_uid_engine": True, "uid_exclusive": False,
                    "identity_passed": False, "preload_probe_released": True, "lock": lock}
        finally:
            if fd is not None:
                os.close(fd)

    def _loaded(self, model, now):
        child = model._process
        require(child is not None and child.poll() is None, "retained native child is not live", "PREFIX_IDENTITY")
        lock = self._fd_identity(model._lock_file, self._lock_path(model))
        process = self.process_factory(child.pid)
        executable = str(Path(process.exe()).resolve())
        owner = {"pid": child.pid, "create_time": process.create_time(), "executable": executable,
                 "generation": model._owned_process_generation, "popen_object": id(child),
                 "device": lock["device"], "inode": lock["inode"], "fd": lock["fd"]}
        require(process.is_running() and process.status() != "zombie" and
                process.uids().real == os.getuid() and process.ppid() == os.getpid() and
                executable == str(Path(model.runtime_binary).resolve()),
                "owned child's UID, parent, lifetime or executable differs", "PREFIX_IDENTITY")
        require(self.owner is None or owner == self.owner, "native child/FD generation changed", "PREFIX_IDENTITY")
        self.owner = owner
        if now - self.last_fd >= self.interval:
            self.fd_receipt = self.fd_observer(child.pid, lock["fd"], executable, owner, lock)
            self.last_fd = now
        require(self.fd_receipt is not None and self.fd_receipt.get("available") is True and
                self.fd_receipt.get("observed_matching_lock_fd") is True,
                "owned child's inherited native UID descriptor not observed", "PREFIX_UID")
        sample = getattr(model, "_prefix_admission_sample", None)
        require(isinstance(sample, dict) and type(sample.get("available_bytes")) is int,
                "original native admission sample missing", "PREFIX_ADMISSION")
        # Original _admit has already run, including its contextual pressure and
        # swap rules. This check adds reserve; it does not replace those rules.
        required = (model.limits.reserve_bytes + model.limits.base_workspace_bytes +
                    model.context_tokens * 64 * 1024 + model.batch_size * 4 * MiB + SNAPSHOT_IO_RESERVE)
        require(sample["available_bytes"] >= required,
                "original workspace plus bounded snapshot I/O reserve unavailable", "PREFIX_ADMISSION")
        require(child.poll() is None and self.process_factory(child.pid).create_time() == owner["create_time"],
                "owned child changed during descriptor observation", "PREFIX_IDENTITY")
        return {"uid_exclusive": True, "identity_passed": True, "owner": owner,
                "fd_observation": self.fd_receipt, "snapshot_io_extra_reserve_bytes": SNAPSHOT_IO_RESERVE,
                "required_available_bytes": required}

    def __call__(self, model, *, phase):
        with self.lock:
            try:
                require(self.violation is None, "external guard already rejected", "PREFIX_ADMISSION")
                self.provenance()
                require(type(model.limits) is self.limits_type, "original NativeLimits instance required")
                current = dict(vars(model.limits))
                require(self.limits is None or self.limits == current, "native limits changed", "PREFIX_ADMISSION")
                self.limits = current
                now = self.clock()
                include_power = now - self.last_power >= self.interval
                power = self.power.sample(include_power=include_power)
                if include_power:
                    self.last_power = now
                require(power.get("passed") is True and self.power.violation is None,
                        "AC/lid/clock continuity rejected", "POWER_CONTEXT_CHANGED")
                require(phase == ("preload" if model._process is None else "loaded"), "false admission phase")
                ownership = self._preload(model) if phase == "preload" else self._loaded(model, now)
                self.samples += 1
                self.last = dict(phase=phase, passed=True, power_context_passed=True,
                                 native_limits_unchanged=True, **ownership)
                return self.last
            except BaseException as error:
                self.violation = {"code": getattr(error, "code", type(error).__name__), "error": str(error)}
                raise

    def summary(self):
        return {"passed": self.violation is None and self.samples > 0, "samples": self.samples,
                "last": self.last, "violation": self.violation,
                "scope": "released preload probe; retained native FD and owned child checked after spawn; sampled evidence"}
