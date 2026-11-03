"""Summarize closed NoCopy component receipts; no native/GPU/model is invoked.

Only bounded JSON/log metadata are read. Tensor payloads are not reread, clocks
are not benchmarked, and these resident synthetic timings never qualify GLM.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

WORK = Path(__file__).resolve().parent
ROOT = WORK / "metal-tensorops-q8-nocopy"
RECEIPT = ROOT / "fixtures-warm-v2/run2/receipt.json"
OUTPUT = WORK / "gpu-nocopy-fixture-summary.json"
EXPECTED_RECEIPT_SHA256 = "fb03227073067a427284f6991e5c6d28174cc1484231db57e126ca9cd40b9989"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, limit=8 << 20):
    path = Path(path)
    state = path.stat()
    require(path.is_file() and 0 < state.st_size <= limit, "bounded existing metadata required: " + str(path))
    data = path.read_bytes()
    after = path.stat()
    require((state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "metadata changed during read")
    return data


def trace_rows(text):
    return [dict(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)", line.split("AIRLLM_CPU_METAL_Q8 ", 1)[1]))
            for line in text.splitlines() if "AIRLLM_CPU_METAL_Q8 " in line]


def summarize():
    raw = read(RECEIPT)
    require(hashlib.sha256(raw).hexdigest() == EXPECTED_RECEIPT_SHA256, "this summary is bound to the audited closed run2")
    d = json.loads(raw)
    require(d["status"] == "passed_component_contract_only" and d["full_model_qualified"] is False,
            "closed component-only receipt required")
    require(d["power_gate"]["passed"] is True and d["source_identity_unchanged"] is True,
            "clear existing power/source receipts required")
    require(len(d["results"]) == 25 and d["repetitions_per_completed_graph"] == 5, "exact warm-v2 case set required")
    normal, numerical, exceptions, cleanup, hashes = [], [], [], [], {}
    commands = completed = publications = alias_commands = explicit_weight_copy_bytes = 0
    max_weight = max_view = max_offset = native_rss = 0
    totals = {name: 0.0 for name in ("gpu_seconds", "copy_weight_us", "view_create_us", "command_and_readback_us")}
    iteration_hashes_stable = True
    for pair in d["results"]:
        require(pair["expectation_met"] is True and len(pair["processes"]) == 2, "complete separate CPU/GPU case required")
        cpu, gpu = pair["processes"]
        require(cpu["bridge"] is False and gpu["bridge"] is True, "ordinary CPU oracle and candidate order required")
        comparison = pair["numerical_comparison"]
        require(comparison["all_finite"] is True and comparison["metal_tolerance_passed"] is True,
                "original finite Metal criterion did not pass")
        numerical.append(comparison)
        if not comparison["cpu_bitexact"]:
            exceptions.append({"case": pair["case"], "raw_comparison": comparison})
        for job in pair["processes"]:
            exit_row, idle = job["cleanup"], job["idle_assertion"]
            require(job["returncode"] == 0 and exit_row == {"signals": [], "returncode": 0, "owned_server_stopped": True},
                    "owned child did not close normally without owner signal")
            require(idle.get("cleaned_up") is True and idle.get("helper_exit_code") == 0 and
                    idle.get("cleanup_kind") == "natural_target_exit" and idle.get("exit_qualified") is True,
                    "scoped idle helper did not close naturally")
            result = job["result"]
            iteration_hashes = result["iteration_sha256"]
            stable = len(iteration_hashes) == 5 and len(set(iteration_hashes)) == 1 and iteration_hashes[-1] == result["output_sha256"]
            iteration_hashes_stable &= stable
            native_rss = max(native_rss, result["peak_rss_bytes"])
            require(job["uid_fd_observation"].get("observed_matching_lock_fd") is True, "per-child inherited FD observation missing")
            cleanup.append({"pid": job["owned_identity"]["pid"], "create_time": job["owned_identity"]["create_time"],
                "retained_child_exit_code": 0, "owner_signals": [], "owned_lifetime_exited": True,
                "idle_helper_pid": idle["helper_pid"], "idle_helper_exit_code": 0,
                "idle_helper_cleanup": "natural_target_exit", "five_iteration_hashes_stable": stable})
        require(cpu["result"]["weight_sha256"] == gpu["result"]["weight_sha256"] and
                cpu["result"]["input_sha256"] == gpu["result"]["input_sha256"], "actual separate-process inputs differ")
        if pair["case"]["expectation"] != "gpu":
            require(comparison["cpu_bitexact"] is True and gpu["gpu_trace"]["submitted_commands"] == 0,
                    "CPU fallback/decline must remain exact and submit zero commands")
            continue
        require(gpu["gpu_trace"]["published_operations"] == 5 and gpu["gpu_trace"]["all_gpu_operations_terminal"] is True,
                "five complete actual GPU operations required")
        a, b = cpu["reuse"], gpu["reuse"]
        require(len(a["warm_seconds"]) == len(b["warm_seconds"]) == 4, "separate first and four warm iterations required")
        normal.append({"case": pair["case"], "cpu_first_s": a["first_seconds"], "gpu_first_s": b["first_seconds"],
            "cpu_warm_s": a["warm_seconds"], "gpu_warm_s": b["warm_seconds"],
            "cpu_warm_median_s": a["warm_median_seconds"], "gpu_warm_median_s": b["warm_median_seconds"],
            "gpu_over_cpu_warm_ratio": b["warm_median_seconds"] / a["warm_median_seconds"],
            "timings_qualified": False, "raw_comparison": comparison, "native_trace": gpu["gpu_trace"],
            "native_cpu_lifetime_peak_rss_bytes": cpu["result"]["peak_rss_bytes"],
            "native_gpu_lifetime_peak_rss_bytes": gpu["result"]["peak_rss_bytes"]})
        log = RECEIPT.parent / pair["case"]["id"] / "bridge-on/stderr.log"
        data = read(log, 32 << 20)
        hashes[str(log)] = hashlib.sha256(data).hexdigest()
        rows = trace_rows(data.decode(errors="replace"))
        submitted = [row for row in rows if row.get("event") == "submitted"]
        terminal = [row for row in rows if row.get("event") == "completed"]
        tiles = [row for row in rows if row.get("event") == "tile_done"]
        published = [row for row in rows if row.get("event") == "published"]
        require(len(submitted) == len(terminal) == len(tiles) == len(published) == 5, "complete recorded GPU trace required")
        commands += len(submitted); completed += len(terminal); publications += len(published)
        for row in tiles:
            require(row.get("alias_identity_verified") == "1" and row.get("copy_weight_bytes") == "0" and
                    row.get("copied_weight_bytes_total") == "0", "actual alias/zero explicit weight-copy counters required")
            alias_commands += 1
            explicit_weight_copy_bytes += int(row["copy_weight_bytes"])
            max_weight = max(max_weight, int(row["borrowed_weight_bytes"]))
            max_view = max(max_view, int(row["borrowed_view_bytes"]))
            max_offset = max(max_offset, int(row["weight_view_offset"]))
            for key in totals:
                totals[key] += float(row[key])
    require(len(cleanup) == 50 and len({row["pid"] for row in cleanup}) == 50, "fifty distinct completed children required")
    require(commands == completed == publications == d["gpu_commands"] == 80, "recorded command totals differ")
    require(iteration_hashes_stable, "within-child five-iteration output hashes changed")
    samples = RECEIPT.parent / "samples.jsonl"
    data = read(samples)
    hashes[str(samples)] = hashlib.sha256(data).hexdigest()
    max_growth = max(json.loads(line)["swap_growth_bytes"] for line in data.splitlines())
    ratios = [row["gpu_over_cpu_warm_ratio"] for row in normal]
    return {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_path": str(RECEIPT), "receipt_sha256": EXPECTED_RECEIPT_SHA256,
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_only_summary": True, "tensor_payloads_reread": False, "new_native_gpu_or_model_calls": 0,
        "status": d["status"], "profile_sha256": d["profile_sha256"], "binary_sha256": d["binary_sha256"],
        "case_count": len(d["results"]), "normal_gpu_pairs": len(normal), "cpu_fallback_or_decline_pairs": 25-len(normal),
        "per_graph_completed_iterations": 5, "first_iteration_separate_from_four_resident_warm_repetitions": True,
        "recorded_final_float32_outputs_compared": sum(row["outputs"] for row in numerical),
        "all_recorded_outputs_finite": all(row["all_finite"] for row in numerical),
        "original_metal_tolerance_passed": all(row["metal_tolerance_passed"] for row in numerical),
        "metal_criteria": d["metal_tolerance"], "cpu_bitexact_contract_met": all(row["cpu_bitexact"] for row in numerical),
        "bitexact_pair_count": sum(row["cpu_bitexact"] for row in numerical),
        "non_bitexact_pair_count": len(exceptions), "non_bitexact_cases": exceptions,
        "different_float32_values_total": sum(row["different_float32_bits_count"] for row in numerical),
        "exception_scope": "one non-bitexact case, not a single-bit error:63 of64 float32 values differ within unchanged Metal tolerance",
        "completed_gpu_commands": completed, "published_gpu_operations": publications,
        "recorded_normal_gpu_trace_total": totals,
        "weight_alias_evidence": {"verified_commands": alias_commands, "explicit_weight_copy_bytes_total": explicit_weight_copy_bytes,
            "max_borrowed_weight_bytes": max_weight, "max_borrowed_view_bytes": max_view, "max_weight_view_offset_bytes": max_offset,
            "weight_mappings_read_only": all(job["result"]["weight_mapping_read_only"] for pair in d["results"] for job in pair["processes"]),
            "scope": "native provider alias identity and zero explicit weight memcpy; input quantization and output transfers remain; no proof of zero driver movement/SSD reads"},
        "fixed_owned_gpu_payload_bytes": d["fixed_gpu_payload_bytes"],
        "metal_device_current_allocated_peak_bytes": d["device_allocated_observed_peak_bytes"],
        "native_getrusage_lifetime_peak_rss_bytes": native_rss,
        "peak_sampled_child_rss_bytes": d["observed_peak_child_rss_bytes"],
        "peak_sampled_controller_rss_bytes": d["observed_peak_controller_rss_bytes"],
        "child_rss_cap_bytes": d["child_rss_ceiling_bytes"],
        "global_swap_growth_sampled_peak_bytes": max_growth,
        "power_passed": d["power_gate"]["passed"], "source_identity_unchanged": d["source_identity_unchanged"],
        "fifty_owned_children_exited0_without_owner_signal": True, "fifty_idle_helpers_exited0_naturally": True,
        "cleanup_scope": "recorded retained-Popen exit and scoped helper receipts; summary does not enumerate processes or recheck current PID state",
        "closed_owned_processes": cleanup, "elapsed_seconds_recorded": d["measured_elapsed_seconds"],
        "gpu_over_cpu_warm_ratio_range_unqualified": [min(ratios), max(ratios)],
        "warm_ratio_median_unqualified": statistics.median(ratios), "timings_qualified": False,
        "excluded_resource_shapes": d["excluded_resource_shapes"],
        "unmeasured_driver_residency_bytes": None, "physical_ssd_reads": None,
        "pending_timeout_injection_tested": d["pending_timeout_injection_tested"],
        "terminal_command_error_injection_tested": d["terminal_command_error_injection_tested"],
        "late_fatal_test_performed": False,
        "late_scale_policy": "entire mapped weight scales scanned; unsupported later scale declines before the single command; not a late-fatal fixture",
        "full_model_qualified": False, "functional_agents_qualified": False, "performance_qualified": False,
        "scope": "NoCopy Metal4 TensorOps synthetic dense projections, read-only offset64 mmap, 5 resident graph repetitions; no global GLM/routes/KDA/logits/cache/EOS/SSD claim",
        "metadata_sha256": hashes, "normal_gpu_results": normal}


if __name__ == "__main__":
    result = summarize()
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "case_count", "normal_gpu_pairs", "original_metal_tolerance_passed",
        "cpu_bitexact_contract_met", "non_bitexact_pair_count", "different_float32_values_total", "completed_gpu_commands",
        "native_getrusage_lifetime_peak_rss_bytes", "timings_qualified")}, allow_nan=False))
