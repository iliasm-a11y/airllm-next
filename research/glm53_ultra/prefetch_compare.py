"""Prepare or explicitly run a guarded original-Q8 prefetch component comparison.

Default preparation reads only source, metadata and the 16 KiB activation
capture. Compilation, library hashing and GGUF payload access require --execute.
No purge, new model runtime, weight replacement, or throughput claim is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

import psutil
from research.glm53_gpu_v2.benchmark_lifecycle import stop_owned_server
from research.glm53_gpu_v2.storage_counters import block_snapshot, compare_snapshots
from research.glm53_gpu_v3.dense_cold_warm import prepare as prepare_original_component

ROOT = Path(__file__).parent
REVISION = "3f8af15a76ac8675812e856da45454bafc90f12c"
MODEL_REVISION = "119034bc3d9d76eb41f91508e6f9fb19fde42672"
MODEL_SHARD = "GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf"
ABSOLUTE_OFFSET = 1590187712
ACTIVATION_SHA256 = "495aaba3baa5a0580b775c7cc5f899a27f60e0b149caead97a776974796b45fe"
PRIOR_OUTPUT_SHA256 = "4d43f62056d934601a565fda2726017c92f573474cd8cccf9783e318d2dea30d"
MIB = 1024**2
WEIGHT_BYTES = 35651584
SCRATCH_BYTES = 8 * MIB
RANGE_CAP = 64 * MIB
RSS_LIMIT = 256 * MIB
SWAP_LIMIT = 32 * MIB
DEFAULT_MODES = ("none", "willneed", "pread", "pread", "willneed", "none")
HEADERS = ("ggml.h", "ggml-cpu.h", "ggml-backend.h", "ggml-alloc.h")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(path):
    value = Path(path).stat()
    return {name: getattr(value, name) for name in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def validate_options(args):
    if args.mapping not in ("standard", "nocache"):
        raise ValueError("Explicit standard/nocache mapping required")
    if (not isinstance(args.modes, (list, tuple)) or not 1 <= len(args.modes) <= 12
            or any(mode not in ("none", "willneed", "pread") for mode in args.modes)):
        raise ValueError("One to twelve bounded component arms required")
    if type(args.phases) is not int or not 2 <= args.phases <= 4:
        raise ValueError("Each arm needs one observation and one to three warm references")
    if type(args.timeout_seconds) not in (int, float) or not math.isfinite(args.timeout_seconds) or not 1 <= args.timeout_seconds <= 60:
        raise ValueError("Component timeout must be in [1,60] seconds")


def prepare(args):
    """Freeze reproducible build inputs without launching any command/model."""
    validate_options(args)
    if args.directory.exists() or args.output.exists():
        raise ValueError("Refusing existing component evidence")
    runtime_raw = args.runtime_manifest.read_bytes()
    runtime = json.loads(runtime_raw)
    if runtime.get("source_revision") != REVISION or runtime.get("status") != "built":
        raise ValueError("Pinned V4 built-runtime manifest required")
    expected = {Path(row["path"]).name: row["sha256"] for row in runtime["artifacts"]
                if row.get("sha256") and row["path"].endswith(".dylib")}
    if not any(name.startswith("libggml-cpu.") for name in expected) or not any(name.startswith("libggml-base.") for name in expected):
        raise ValueError("V4 CPU/base library hashes missing")
    source = args.source.resolve()
    headers = {name: _sha((source / "ggml/include" / name).read_bytes()) for name in HEADERS}
    metadata = prepare_original_component(args)
    if metadata["model_revision"] != MODEL_REVISION:
        raise ValueError("Original pinned GLM Q2_K_S model revision required")
    if metadata["tensor"]["file"] != MODEL_SHARD or metadata["absolute_offset"] != ABSOLUTE_OFFSET:
        raise ValueError("Original pinned Q8 shard and absolute byte offset required")
    if metadata["activation_sha256"] != ACTIVATION_SHA256:
        raise ValueError("Original captured layer40 activation hash mismatch")
    if metadata["tensor"]["bytes_with_alignment"] != WEIGHT_BYTES or WEIGHT_BYTES > RANGE_CAP:
        raise ValueError("Original matrix exceeds component range budget")
    source_raw = (ROOT / "prefetch_compare.cpp").read_bytes()
    frozen_source = args.directory / "prefetch_compare.cpp"
    frozen_source.write_bytes(source_raw)
    binary = args.directory / "prefetch-compare"
    libraries = args.libraries.resolve()
    command = ["clang++", "-std=c++17", "-O2", "-I" + str(source / "ggml/include"),
               str(frozen_source.resolve()), "-L" + str(libraries), "-lggml-cpu", "-lggml-base",
               "-Wl,-rpath," + str(libraries), "-o", str(binary.resolve())]
    plan = {"schema_version": 1, "source_revision": REVISION, "source": str(source),
            "runner_source_sha256": _sha(Path(__file__).read_bytes()),
            "source_header_sha256": headers, "component_source": str(frozen_source.resolve()),
            "component_source_sha256": _sha(source_raw), "runtime_manifest": str(args.runtime_manifest.resolve()),
            "runtime_manifest_sha256": _sha(runtime_raw), "libraries": str(libraries),
            "expected_library_sha256": expected, "compile_command": command,
            "binary": str(binary.resolve()), "mapping": args.mapping, "modes": list(args.modes),
            "phases_per_arm": args.phases, "scope": "component only, existing V4 CPU libraries; no upstream/native runtime modifications"}
    plan_path = args.directory / "build-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    script = args.directory / "build.sh"
    script.write_text("#!/bin/sh\nset -eu\ncd " + shlex.quote(str(ROOT.resolve().parents[1])) + "\n" +
                      shlex.join([sys.executable, "-m", "research.glm53_ultra.prefetch_compare",
                                  "--verify-plan", str(plan_path.resolve())]) + "\nexec " + shlex.join(command) + "\n")
    script.chmod(0o700)
    metadata.update(known_payload_bound_bytes=metadata["known_payload_bound_bytes"] + SCRATCH_BYTES + 3 * (RANGE_CAP // 4096 + 1),
                    scratch_bytes=SCRATCH_BYTES, maximum_original_range_bytes=RANGE_CAP,
                    build_plan=str(plan_path.resolve()), build_script=str(script.resolve()),
                    build_plan_sha256=_sha(plan_path.read_bytes()), mapping=args.mapping, modes=list(args.modes))
    if metadata["known_payload_bound_bytes"] > RANGE_CAP:
        raise MemoryError("Known component payload exceeds64MiB")
    return metadata, plan


def verify_plan(plan_path):
    """Explicit build-time verification; never opens checkpoint weights."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema_version") != 1 or plan.get("source_revision") != REVISION:
        raise ValueError("Unknown/unpinned build plan")
    if _sha(Path(__file__).read_bytes()) != plan["runner_source_sha256"]:
        raise ValueError("Component runner changed after preparation")
    if _sha(Path(plan["component_source"]).read_bytes()) != plan["component_source_sha256"]:
        raise ValueError("Frozen component source changed")
    raw = Path(plan["runtime_manifest"]).read_bytes()
    if _sha(raw) != plan["runtime_manifest_sha256"]:
        raise ValueError("Pinned runtime manifest changed")
    source = Path(plan["source"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    if head != REVISION:
        raise ValueError("Public-header source checkout is not at pinned revision")
    for name, expected in plan["source_header_sha256"].items():
        path = source / "ggml/include" / name
        original = subprocess.check_output(["git", "-C", str(source), "show", REVISION + ":ggml/include/" + name], timeout=5)
        if _sha(path.read_bytes()) != expected or _sha(original) != expected:
            raise ValueError("Public GGML header changed or differs from pinned revision")
    libraries = Path(plan["libraries"])
    hashes = {name: _sha((libraries / name).read_bytes()) for name in plan["expected_library_sha256"]}
    if hashes != plan["expected_library_sha256"]:
        raise ValueError("Existing V4 library hash mismatch; no qualification transferred")
    for alias, prefix in (("libggml-cpu.dylib", "libggml-cpu."), ("libggml-base.dylib", "libggml-base.")):
        target = (libraries / alias).resolve()
        if target.parent != libraries.resolve() or not target.name.startswith(prefix) or target.name not in hashes:
            raise ValueError("Link alias escapes the qualified V4 libraries")
    return {"verified": True, "component_source_sha256": plan["component_source_sha256"],
            "native_libraries_sha256": hashes, "source_revision": REVISION}


def check_guard(*, rss, swap, baseline_swap, pressure, available, elapsed, timeout):
    values = (rss, swap, baseline_swap, available)
    if (any(type(value) is not int or value < 0 for value in values)
            or type(pressure) is not int or pressure not in (1, 2, 4)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in (elapsed, timeout))):
        raise RuntimeError("Required resource observation unavailable")
    if rss > RSS_LIMIT or max(0, swap-baseline_swap) > SWAP_LIMIT or pressure == 4 or available < RSS_LIMIT or elapsed > timeout:
        raise MemoryError("Component RSS/swap/pressure/available-memory/timeout guard")


def _pressure():
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                       text=True, timeout=3))


def validate_phase(phase, *, index, mode, mapping):
    if (phase.get("phase") != index or phase.get("mode") != mode or phase.get("mapping") != mapping
            or phase.get("logical_weight_bytes") != WEIGHT_BYTES or phase.get("scratch_bytes") != SCRATCH_BYTES
            or type(phase.get("mapped_bytes")) is not int
            or not WEIGHT_BYTES <= phase.get("mapped_bytes", 0) <= RANGE_CAP
            or phase.get("all_outputs_finite") is not True or phase.get("file_identity_unchanged") is not True):
        raise ValueError("Native phase identity/range/finiteness rejected")
    expected_read = WEIGHT_BYTES if mode == "pread" else 0
    if phase.get("pread_completed_bytes") != expected_read:
        raise ValueError("Native pread byte count rejected")
    calls = phase.get("pread_calls")
    if type(calls) is not int or not (1 <= calls <= 4096 if mode == "pread" else calls == 0):
        raise ValueError("Native pread call count rejected")
    for name in ("preparation_wall_seconds", "compute_wall_seconds", "preparation_plus_compute_seconds",
                 "observation_gap_seconds", "total_observed_sequence_seconds"):
        value = phase.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Native phase time missing/nonfinite")
    if (not math.isclose(phase["preparation_plus_compute_seconds"],
                         phase["preparation_wall_seconds"] + phase["compute_wall_seconds"], rel_tol=1e-6, abs_tol=1e-9)
            or not math.isclose(phase["total_observed_sequence_seconds"],
                                phase["preparation_plus_compute_seconds"] + phase["observation_gap_seconds"], rel_tol=1e-6, abs_tol=1e-9)):
        raise ValueError("Native timing scopes do not reconcile")
    for name in ("residency_before", "residency_after_preparation", "residency_after_compute"):
        value = phase.get(name, {})
        if (type(value.get("mapping_pages")) is not int or type(value.get("resident_pages")) is not int
                or not 1 <= value["mapping_pages"] <= RANGE_CAP // 4096 + 1
                or not 0 <= value["resident_pages"] <= value["mapping_pages"]):
            raise ValueError("Native mincore observation rejected")
    page = phase.get("page_size_bytes")
    if type(page) is not int or not 4096 <= page <= 65536 or page & (page-1):
        raise ValueError("Native page size missing/invalid")
    pages = (phase["mapped_bytes"] + page - 1) // page
    if any(phase[name]["mapping_pages"] != pages
           for name in ("residency_before", "residency_after_preparation", "residency_after_compute")):
        raise ValueError("Mapped range and mincore page counts do not reconcile")
    if index and phase["residency_before"]["resident_pages"] != phase["residency_before"]["mapping_pages"]:
        raise ValueError("Warm reference mapping is not resident at its initial snapshot")
    for name in ("preparation_process_delta", "compute_process_delta", "total_process_delta"):
        delta = phase.get(name, {})
        for key in ("os_storage_read_bytes", "pageins", "task_faults", "task_pageins", "resident_bytes_after"):
            if type(delta.get(key)) is not int or delta[key] < 0:
                raise ValueError("Required process counters missing/invalid")
        cpu = delta.get("cpu_core_seconds")
        if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
            raise ValueError("Required CPU counter missing/invalid")
    for key in ("os_storage_read_bytes", "pageins", "task_faults", "task_pageins"):
        if phase["total_process_delta"][key] != phase["preparation_process_delta"][key] + phase["compute_process_delta"][key]:
            raise ValueError("Process counter deltas do not reconcile")


def _run_arm(args, plan, report, *, arm, mode, lock_fd, started):
    directory = args.directory / (f"arm-{arm:02d}-" + mode)
    command = [plan["binary"], str((args.model_directory / report["tensor"]["file"]).resolve()),
               str(report["absolute_offset"]), report["activation_file"], str(directory.resolve()),
               mode, args.mapping, str(args.phases)]
    row = {"arm": arm, "mode": mode, "mapping": args.mapping, "command": command, "phases": []}
    report["arms"].append(row)
    stop = threading.Event(); child = watcher = None
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("AIRLLM_", "GGML_", "DYLD_", "LLAMA_ARG_")) and key != "LLAMA_API_KEY"}

    def monitor():
        tick = 0; level = report["pressure_start"]; last_sample = {}
        try:
            while not stop.wait(.05) and child.poll() is None:
                rss = psutil.Process(child.pid).memory_info().rss
                swap = psutil.swap_memory().used
                report["peak_rss_bytes"] = max(report["peak_rss_bytes"], rss)
                report["peak_swap_growth_bytes"] = max(report["peak_swap_growth_bytes"], swap-report["swap_start_bytes"], 0)
                if tick % 4 == 0:
                    level = _pressure()
                available = psutil.virtual_memory().available
                last_sample = {"rss_bytes": rss, "global_swap_growth_bytes": max(0, swap-report["swap_start_bytes"]),
                               "pressure_level": level, "available_bytes": available,
                               "elapsed_seconds": time.monotonic()-started}
                check_guard(rss=rss, swap=swap, baseline_swap=report["swap_start_bytes"], pressure=level,
                            available=available, elapsed=last_sample["elapsed_seconds"],
                            timeout=args.timeout_seconds)
                tick += 1
        except psutil.NoSuchProcess:
            return
        except Exception as exc:
            report["guard_violation"] = {"arm": arm, "error": f"{type(exc).__name__}: {exc}", **last_sample}
            stop_owned_server(child)

    try:
        with (args.directory / f"arm-{arm:02d}.stderr.log").open("w") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                     text=True, env=environment, start_new_session=True, pass_fds=(lock_fd,))
            row["owned_pid"] = child.pid
            watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
            for index in range(args.phases):
                expected_mode = mode if index == 0 else "none"
                ready = json.loads(child.stdout.readline())
                if ready.get("ready") != index or ready.get("mode") != expected_mode or ready.get("mapping") != args.mapping:
                    raise RuntimeError("Native phase synchronization failed")
                before = block_snapshot(); child.stdin.write("run\n"); child.stdin.flush()
                phase = json.loads(child.stdout.readline()); after = block_snapshot()
                validate_phase(phase, index=index, mode=expected_mode, mapping=args.mapping)
                name = f"outputs-{index}.f32"
                if phase.get("output_file") != name:
                    raise ValueError("Unexpected native output path")
                raw = (directory / name).read_bytes()
                if len(raw) != 8192 * 4 or not all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", raw)):
                    raise ValueError("Missing/nonfinite full8192-output reference")
                phase.update(output_sha256=_sha(raw), block_device_before=before, block_device_after=after,
                             block_device_delta=compare_snapshots(before, after),
                             role="observed_preparation_and_compute" if index == 0 else "warm_exact_reference")
                row["phases"].append(phase)
            summary = json.loads(child.stdout.readline())
            if any(summary.get(key) is not True for key in
                   ("complete", "outputs_equal_warm_reference", "file_identity_unchanged", "all_outputs_finite")):
                raise ValueError("Native warm-reference completion gate failed")
            row["native_summary"] = summary
            row["native_returncode"] = child.wait(timeout=5)
            if row["native_returncode"] or report["guard_violation"]:
                raise RuntimeError("Native component rejected")
            if len({phase["output_sha256"] for phase in row["phases"]}) != 1:
                raise ValueError("Preparation differs from the warm output reference")
            row["status"] = "complete"
    finally:
        try:
            if child is not None:
                stop_owned_server(child); row["native_returncode"] = child.returncode
        finally:
            stop.set()
            if watcher is not None:
                watcher.join(timeout=5)


def summarize(arms):
    grouped = {}
    for arm in arms:
        if arm.get("status") != "complete":
            raise ValueError("Cannot summarize an incomplete arm")
        grouped.setdefault(arm["mode"], []).append(arm["phases"][0])
    keys = ("preparation_wall_seconds", "compute_wall_seconds", "preparation_plus_compute_seconds",
            "observation_gap_seconds", "total_observed_sequence_seconds")
    def medians(rows):
        return {"observations": len(rows), **{key: statistics.median(row[key] for row in rows) for key in keys}}
    resident_groups = {}
    for mode, rows in grouped.items():
        for row in rows:
            before = row["residency_before"]
            label = ("fully_resident_at_snapshot" if before["resident_pages"] == before["mapping_pages"]
                     else "not_fully_resident_at_snapshot_not_proven_cold")
            resident_groups.setdefault(mode, {}).setdefault(label, []).append(row)
    return {"mode_medians": {mode: {"observations": len(rows), **{
                key: statistics.median(row[key] for row in rows) for key in keys}} for mode, rows in grouped.items()},
            "mode_medians_warning": "May mix initial residency states; inspect per-arm snapshots and stratified observations before comparing modes",
            "initial_residency_strata": {mode: {label: medians(rows) for label, rows in groups.items()}
                                         for mode, groups in resident_groups.items()},
            "all_initial_mappings_resident_at_snapshot": all(
                phase["residency_before"]["resident_pages"] == phase["residency_before"]["mapping_pages"]
                for rows in grouped.values() for phase in rows),
            "scope": "uncontrolled-cache component observations; not causal cold-storage or model throughput qualification",
            "physical_cold_cache_proven": False, "optimization_qualified": False, "tokens_per_second": None}


def run(args):
    report = {"schema_version": 1, "status": "preparing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "arms": [], "guard_violation": None, "model_decoder_executed": False, "gpu_initialized": False,
              "scope": "one original mapped Q8 matrix CPU10, explicit prefetch preparation, then warm bit reference",
              "limits": {"rss_bytes": RSS_LIMIT, "global_swap_growth_bytes": SWAP_LIMIT,
                         "original_range_bytes": RANGE_CAP, "scratch_bytes": SCRATCH_BYTES,
                         "component_timeout_seconds": args.timeout_seconds, "critical_pressure_level": 4,
                         "launch_pressure_level": 1, "minimum_available_bytes": RSS_LIMIT,
                         "guard_sampling_seconds": .05, "pressure_sampling_seconds": .2},
              "physical_nand_read_bytes": None, "file_cache_hit_bytes": None, "pure_weight_wait_seconds": None,
              "timing_scope": "total_observed_sequence includes after-preparation counter/mincore gap; preparation_plus_compute excludes that explicit gap; setup/pool/scratch allocation precede phases",
              "storage_scope": "OS-accounted process reads/pageins; global block counters include other processes; no NAND or exact cache-hit attribution"}
    try:
        metadata, plan = prepare(args); report.update(metadata)
        shard = args.model_directory / report["tensor"]["file"]
        report["source_identity_before"] = _identity(shard)
        if report["source_identity_before"]["st_size"] < report["absolute_offset"] + WEIGHT_BYTES:
            raise ValueError("Original selected range exceeds source file size")
        if not args.execute:
            report["status"] = "prepared_no_checkpoint_payload_read"
            return report
        with Path("/tmp/airllm-glm53-benchmark.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report.update(swap_start_bytes=psutil.swap_memory().used, pressure_start=_pressure(),
                          peak_rss_bytes=0, peak_swap_growth_bytes=0)
            if report["pressure_start"] != 1:
                raise MemoryError("Component build/launch requires normal measured pressure")
            check_guard(rss=0, swap=report["swap_start_bytes"], baseline_swap=report["swap_start_bytes"],
                        pressure=report["pressure_start"], available=psutil.virtual_memory().available,
                        elapsed=0., timeout=args.timeout_seconds)
            report["build_input_validation"] = verify_plan(report["build_plan"])
            report["component_compiler_version"] = subprocess.check_output(["clang++", "--version"], text=True, timeout=5)
            build = subprocess.run(plan["compile_command"], check=True, capture_output=True, text=True, timeout=30)
            (args.directory / "build.stdout.log").write_text(build.stdout)
            (args.directory / "build.stderr.log").write_text(build.stderr)
            report["binary_sha256"] = _sha(Path(plan["binary"]).read_bytes())
            started = time.monotonic()
            check_guard(rss=0, swap=psutil.swap_memory().used, baseline_swap=report["swap_start_bytes"],
                        pressure=_pressure(), available=psutil.virtual_memory().available,
                        elapsed=0., timeout=args.timeout_seconds)
            try:
                for arm, mode in enumerate(args.modes):
                    _run_arm(args, plan, report, arm=arm, mode=mode, lock_fd=lock.fileno(), started=started)
                report["summary"] = summarize(report["arms"])
            finally:
                report["swap_final_bytes"] = psutil.swap_memory().used
                report["peak_swap_growth_bytes"] = max(report["peak_swap_growth_bytes"],
                    report["swap_final_bytes"]-report["swap_start_bytes"], 0)
                if report["peak_swap_growth_bytes"] > SWAP_LIMIT:
                    report["guard_violation"] = report["guard_violation"] or {"final_global_swap_growth_bytes": report["peak_swap_growth_bytes"]}
            report["pressure_final"] = _pressure()
            check_guard(rss=0, swap=report["swap_final_bytes"], baseline_swap=report["swap_start_bytes"],
                        pressure=report["pressure_final"], available=psutil.virtual_memory().available,
                        elapsed=time.monotonic()-started, timeout=args.timeout_seconds)
            report["source_identity_after"] = _identity(shard)
            report["source_identity_unchanged"] = report["source_identity_before"] == report["source_identity_after"]
            report["build_inputs_after"] = verify_plan(report["build_plan"])
            report["all_outputs_bit_identical"] = len({phase["output_sha256"] for arm in report["arms"] for phase in arm["phases"]}) == 1
            report["prior_component_output_sha256"] = PRIOR_OUTPUT_SHA256
            report["all_outputs_match_prior_component"] = all(
                phase["output_sha256"] == PRIOR_OUTPUT_SHA256 for arm in report["arms"] for phase in arm["phases"])
            if (not report["source_identity_unchanged"] or not report["all_outputs_bit_identical"]
                    or not report["all_outputs_match_prior_component"] or report["guard_violation"]):
                raise RuntimeError("Component identity/fidelity/resource gate rejected")
            report["status"] = "complete"
    except BaseException as exc:
        report.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-plan", type=Path)
    parser.add_argument("--inventory", type=Path, default=ROOT.parent / "glm53_m5/sources/gguf-headers.json")
    parser.add_argument("--runtime-manifest", type=Path, default=ROOT.parent / "glm53_gpu_v3/evidence/build-runtime-v4.json")
    parser.add_argument("--capture", type=Path); parser.add_argument("--source", type=Path)
    parser.add_argument("--libraries", type=Path); parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--directory", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--mapping", choices=("standard", "nocache"))
    parser.add_argument("--modes", nargs="+", choices=("none", "willneed", "pread"), default=list(DEFAULT_MODES))
    parser.add_argument("--phases", type=int, choices=range(2, 5), default=2)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.verify_plan:
        print(json.dumps(verify_plan(args.verify_plan)))
        return 0
    for name in ("capture", "source", "libraries", "model_directory", "directory", "output", "mapping"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    if args.output.exists() or args.directory.exists():
        parser.error("Refusing existing component evidence")
    def interrupt(*unused):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        report = run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] in ("complete", "prepared_no_checkpoint_payload_read") else 2


if __name__ == "__main__":
    raise SystemExit(main())
