#!/usr/bin/env python3
"""Guard an owned native diagnostic process; all measurements are sampled.

Uses only Python's standard library. OS-accounted I/O is not logical GGUF
traffic or physical NAND traffic. Captures are diagnostics, not speed tests.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

GiB, MiB = 1024**3, 1024**2


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_ticks", "system_ticks", "idle_wakeups", "interrupt_wakeups", "pageins",
        "wired", "resident", "footprint", "start", "exit", "child_user", "child_system",
        "child_idle", "child_interrupt", "child_pageins", "child_elapsed",
        "storage_read_bytes", "storage_write_bytes")]


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def command_text(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError("Cannot measure " + command[-1] + ": " + result.stderr.strip())
    return result.stdout.strip()


def system_sample():
    swap = command_text(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
    if not match:
        raise RuntimeError("Cannot parse system swap usage")
    return {"pressure_level": int(command_text(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])),
            "swap_used_bytes": int(float(match[1]) * {"K": 1024, "M": MiB, "G": GiB}[match[2]])}


def proc_sample(libproc, pid, tick_seconds):
    value = Usage()
    if libproc.proc_pid_rusage(pid, 2, ctypes.byref(value)) != 0:
        return None
    result = {name: int(getattr(value, name)) for name in (
        "user_ticks", "system_ticks", "pageins", "resident", "footprint",
        "storage_read_bytes", "storage_write_bytes")}
    result["rss_bytes"] = result["resident"]
    result["cpu_seconds"] = (result["user_ticks"] + result["system_ticks"]) * tick_seconds
    return result


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def stop_owned(child):
    if child.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        try:
            child.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            pass


def run(args):
    if os.uname().sysname != "Darwin":
        raise RuntimeError("This guard uses macOS libproc and sysctl")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("An owned diagnostic command is required after --")
    binary = Path(command[0]).expanduser().resolve()
    if not binary.is_file():
        raise ValueError("Diagnostic binary must be an existing absolute or relative file")
    command[0] = str(binary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing guard report")
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    libproc.proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    libproc.proc_pid_rusage.restype = ctypes.c_int
    base = Timebase()
    if ctypes.CDLL(None).mach_timebase_info(ctypes.byref(base)) or not base.denom:
        raise RuntimeError("Cannot measure Mach CPU timebase")
    tick_seconds = base.numer / base.denom / 1e9
    initial = system_sample()
    if initial["pressure_level"] >= 4:
        raise RuntimeError("Critical memory pressure before diagnostic launch")
    report = {"schema_version": 1, "status": "starting", "command": command,
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "limits": {"rss_bytes": args.max_rss_gib * GiB, "swap_growth_bytes": args.max_swap_mib * MiB,
                         "critical_pressure_level": 4, "timeout_seconds": args.timeout_seconds},
              "sampling_interval_seconds": args.poll_seconds, "mach_timebase": {"numer": base.numer, "denom": base.denom},
              "initial_system_sample": initial, "samples": [], "peak_sampled_rss_bytes": 0,
              "peak_sampled_footprint_bytes": 0, "peak_system_swap_growth_bytes": 0,
              "last_sampled_os_storage_read_bytes": None, "last_sampled_cpu_seconds": None,
              "logical_weight_reads_bytes": None, "physical_nand_reads_bytes": None,
              "guard_violation": None,
              "measurement_scope": "Owned process libproc rusage_info_v2; sampled counters can miss the final interval. Swap and memory pressure are system-wide. OS storage bytes include paging and readahead, not physical NAND.",
              "environment_knobs": {name: os.environ[name] for name in ("AIRLLM_MMAP_DEMAND", "DYLD_INSERT_LIBRARIES") if name in os.environ}}
    stdout_path, stderr_path = args.output.with_suffix(".stdout.log"), args.output.with_suffix(".stderr.log")
    report.update(stdout_file=str(stdout_path), stderr_file=str(stderr_path))
    atomic_save(args.output, report)
    interrupted = []
    previous_handlers = {}
    def interrupt(sig, frame):
        interrupted.append(sig)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupt)
    child = None
    started = time.monotonic()
    try:
        with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
            report.update(status="running", owned_pid=child.pid)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                sample = {"elapsed_seconds": elapsed, **system_sample()}
                process = proc_sample(libproc, child.pid, tick_seconds)
                if process is None:
                    if child.poll() is None:
                        raise RuntimeError("Cannot inspect the owned diagnostic process")
                    break
                sample.update(process)
                sample["swap_growth_bytes"] = max(0, sample["swap_used_bytes"] - initial["swap_used_bytes"])
                report["samples"].append(sample)
                report["peak_sampled_rss_bytes"] = max(report["peak_sampled_rss_bytes"], sample["rss_bytes"])
                report["peak_sampled_footprint_bytes"] = max(report["peak_sampled_footprint_bytes"], sample["footprint"])
                report["peak_system_swap_growth_bytes"] = max(report["peak_system_swap_growth_bytes"], sample["swap_growth_bytes"])
                report["last_sampled_os_storage_read_bytes"] = sample["storage_read_bytes"]
                report["last_sampled_cpu_seconds"] = sample["cpu_seconds"]
                if interrupted:
                    report["guard_violation"] = "interrupt_signal_" + str(interrupted[0])
                elif sample["rss_bytes"] > args.max_rss_gib * GiB:
                    report["guard_violation"] = "rss_limit"
                elif sample["swap_growth_bytes"] > args.max_swap_mib * MiB:
                    report["guard_violation"] = "system_swap_growth_limit"
                elif sample["pressure_level"] >= 4:
                    report["guard_violation"] = "critical_memory_pressure"
                elif elapsed > args.timeout_seconds:
                    report["guard_violation"] = "timeout"
                atomic_save(args.output, report)
                if report["guard_violation"]:
                    stop_owned(child)
                    break
                time.sleep(args.poll_seconds)
            report["returncode"] = child.wait()
            report["status"] = "completed" if report["returncode"] == 0 and not report["guard_violation"] else "stopped_or_failed"
    except BaseException as error:
        report.update(status="supervisor_error", error=type(error).__name__ + ": " + str(error))
        if child is not None:
            stop_owned(child)
            report["returncode"] = child.returncode
    finally:
        if child is not None:
            stop_owned(child)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_save(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-gib", type=float, default=9)
    parser.add_argument("--max-swap-mib", type=float, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--poll-seconds", type=float, default=.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for name in ("max_rss_gib", "max_swap_mib", "timeout_seconds", "poll_seconds"):
        value = getattr(args, name)
        if not 0 < value < float("inf"):
            parser.error("Invalid " + name)
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("status", "returncode", "guard_violation", "elapsed_seconds",
        "peak_sampled_rss_bytes", "peak_system_swap_growth_bytes", "last_sampled_os_storage_read_bytes")}, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
