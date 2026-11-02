#!/usr/bin/env python3
"""Summarize GLM hardware measurements without editing their raw evidence.

The legacy baseline's user_ns/system_ns fields are raw Mach ticks, despite
their names. This dataset was measured with a 125/3 nanoseconds/tick timebase.
Operation captures synchronize the backend and cannot establish normal speed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


DEFAULT_TIMEBASE = {"numer": 125, "denom": 3}
SCOPES = {
    "cpu": "Owned-process cumulative user/system Mach ticks, converted with the recorded timebase or this M5 dataset's verified 125/3 ns/tick.",
    "legacy_cpu_keys": "schema-v1 user_ns/system_ns in the original baseline are RAW MACH TICKS, not nanoseconds; raw evidence is unchanged.",
    "storage": "Owned-process libproc OS-accounted bytes, including paging and readahead; neither logical GGUF accesses nor physical NAND traffic.",
    "memory": "Sampled process RSS and physical footprint are different counters; neither is a measurement of all system unified-memory use.",
    "swap_pressure": "System-wide counters; changes cannot be attributed exclusively to the inference process.",
    "gpu": "IORegistry system-wide utilization includes the desktop and unrelated processes; it does not prove inference GPU use.",
    "cold_warm": "Process-cold at launch. OS file-cache state is uncontrolled. Repeated requests share the process and disable prompt-cache reuse.",
    "fidelity": "Benchmark timings alone establish no logit, router, state or token parity. Fidelity captures must be compared separately.",
}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def delta(before, after):
    a, b = number(before), number(after)
    return b - a if a is not None and b is not None and b >= a else None


def finite_max(values):
    values = [n for value in values if (n := number(value)) is not None]
    return max(values) if values else None


def distribution(values):
    values = [n for value in values if (n := number(value)) is not None]
    if not values:
        return {"count": 0}
    return {"count": len(values), "minimum": min(values), "maximum": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None}


def timebase(report):
    recorded = report.get("mach_timebase", {})
    if (isinstance(recorded, dict) and number(recorded.get("numer")) is not None
            and number(recorded.get("denom")) is not None and recorded["numer"] > 0 and recorded["denom"] > 0):
        return {**recorded, "source": "recorded in raw measurement"}
    return {**DEFAULT_TIMEBASE, "source": "verified timebase of the Apple M5 host for this measurement campaign; not inferred from misleading legacy field names"}


def usage(raw, scale):
    if not isinstance(raw, dict):
        return None
    legacy = "user_ticks" not in raw and "user_ns" in raw
    user = number(raw.get("user_ns" if legacy else "user_ticks"))
    system = number(raw.get("system_ns" if legacy else "system_ticks"))
    seconds = (user + system) * scale if user is not None and system is not None else None
    return {"user_ticks": user, "system_ticks": system, "cpu_seconds": seconds,
            "raw_cpu_field_names": ["user_ns", "system_ns"] if legacy else ["user_ticks", "system_ticks"],
            "legacy_names_corrected_in_summary": legacy,
            **{key: number(raw.get(key)) for key in ("pageins", "resident", "footprint", "storage_read_bytes", "storage_write_bytes")}}


def usage_delta(before, after):
    if before is None or after is None:
        return None
    return {key: delta(before.get(key), after.get(key)) for key in (
        "user_ticks", "system_ticks", "cpu_seconds", "pageins", "storage_read_bytes", "storage_write_bytes")}


def command_option(command, *names):
    for index, arg in enumerate(command[:-1]):
        if arg in names:
            value = command[index + 1]
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
    return None


def summarize_request(row, index, scale):
    final = row.get("final") if isinstance(row.get("final"), dict) else {}
    timings = final.get("timings") if isinstance(final.get("timings"), dict) else {}
    events = row.get("events", [])
    token_events = [(number(e.get("t")), e.get("response", {}).get("tokens", [])) for e in events
                    if isinstance(e, dict) and isinstance(e.get("response"), dict)
                    and isinstance(e["response"].get("tokens"), list) and e["response"]["tokens"]]
    token_events = [(t, ids) for t, ids in token_events if t is not None]
    emitted_tokens = sum(len(ids) for _, ids in token_events)
    observed_seconds = token_events[-1][0] - token_events[0][0] if len(token_events) > 1 else None
    after_first_event = emitted_tokens - len(token_events[0][1]) if token_events else 0
    observed_rate = (after_first_event / observed_seconds
                     if observed_seconds is not None and observed_seconds > 0 else None)
    predicted = number(timings.get("predicted_n"))
    decode_seconds = number(timings.get("predicted_ms"))
    decode_seconds = decode_seconds / 1000 if decode_seconds is not None else None
    decode_runs = max(predicted - 1, 0) if predicted is not None else None
    prompt_count, prompt_ms = number(timings.get("prompt_n")), number(timings.get("prompt_ms"))
    before, after = usage(row.get("usage_before"), scale), usage(row.get("usage_after"), scale)
    body = row.get("body", {})
    prompt = body.get("prompt") if isinstance(body, dict) else None
    return {"repetition": row.get("repetition", index), "terminal_event_received": final.get("stop") is True,
            "status": "complete" if final.get("stop") is True else "incomplete_or_failed",
            "error": row.get("error"), "wall_seconds": number(row.get("wall_seconds")),
            "ttft_seconds": number(row.get("ttft_seconds")),
            "prompt_tokens": prompt_count, "requested_new_tokens": body.get("n_predict") if isinstance(body, dict) else None,
            "emitted_token_count": emitted_tokens, "engine_predicted_n": predicted,
            "recorded_token_count": len(row["tokens"]) if isinstance(row.get("tokens"), list) else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt, str) else None,
            "prefill_seconds_engine": prompt_ms / 1000 if prompt_ms is not None else None,
            "prefill_tokens_per_second_engine_reported": number(timings.get("prompt_per_second")),
            "prefill_tokens_per_second_recomputed": prompt_count / (prompt_ms / 1000)
                if prompt_count is not None and prompt_ms is not None and prompt_ms > 0 else None,
            "decode": {"observed_tokens_per_second_reported": number(row.get("observed_decode_tokens_per_second")),
                       "observed_tokens_per_second_recomputed": observed_rate,
                       "observed_interval_seconds": observed_seconds, "observed_numerator_tokens": after_first_event,
                       "event_granularity": "one token per SSE event" if token_events and all(len(ids) == 1 for _, ids in token_events)
                           else "coalesced or absent token events; timestamps do not resolve individual tokens",
                       "engine_seconds": decode_seconds, "engine_runs_n_minus_1": decode_runs,
                       "engine_tokens_per_second_reported": number(timings.get("predicted_per_second")),
                       "engine_tokens_per_second_n_minus_1": decode_runs / decode_seconds
                           if decode_runs is not None and decode_seconds is not None and decode_seconds > 0 else None,
                       "formula": "For N individually emitted tokens, observed decode is (N-1)/(last token time-first token time). The pinned engine's predicted_ms covers predicted_n-1 decode runs; the first output token follows prefill.",
                       "inter_token_seconds": distribution(row.get("inter_token_seconds", []))},
            "usage_before": before, "usage_after": after, "request_usage_delta": usage_delta(before, after),
            "stop_type": final.get("stop_type"), "truncated": final.get("truncated"),
            "token_ids_sha256": hashlib.sha256(json.dumps(row["tokens"], separators=(",", ":")).encode()).hexdigest()
                if isinstance(row.get("tokens"), list) else None}


def summarize_measurement(report, filename):
    base = timebase(report)
    scale = base["numer"] / base["denom"] / 1e9
    command = report.get("command", [])
    samples = report.get("samples", [])
    process_samples = [usage(s.get("process", s), scale) for s in samples if isinstance(s, dict)]
    process_samples = [s for s in process_samples if s is not None]
    final = usage(report.get("usage_final"), scale)
    requests = [summarize_request(r, i, scale) for i, r in enumerate(report.get("requests", [])) if isinstance(r, dict)]
    initial_swap = number(report.get("swap_start_bytes"))
    if initial_swap is None:
        initial_swap = number(report.get("initial_system_sample", {}).get("swap_used_bytes"))
    final_swap = number(report.get("swap_final_bytes"))
    swap_samples = [number(s.get("swap_used", s.get("swap_used_bytes"))) for s in samples if isinstance(s, dict)]
    peak_swap = finite_max(swap_samples + [final_swap])
    if final_swap is None and swap_samples:
        final_swap = next((n for n in reversed(swap_samples) if n is not None), None)
    peak_rss = finite_max([report.get("peak_rss_bytes"), report.get("peak_sampled_rss_bytes")]
                         + [s.get("rss", s.get("rss_bytes")) for s in samples if isinstance(s, dict)])
    peak_footprint = finite_max([report.get("peak_footprint_bytes"), report.get("peak_sampled_footprint_bytes")]
                               + [s.get("footprint") for s in process_samples])
    pressure = [s.get("pressure", s.get("pressure_level")) for s in samples if isinstance(s, dict)]
    gpu = [s.get("gpu_system", {}).get("Device Utilization %") for s in samples
           if isinstance(s, dict) and isinstance(s.get("gpu_system"), dict)]
    before_first = requests[0]["usage_before"] if requests else None
    run_usage = final or (process_samples[-1] if process_samples else None)
    diagnostic = "requests" not in report and "limits" in report
    complete_requests = [r for r in requests if r["terminal_event_received"]]
    return {"file": filename, "kind": "guarded_diagnostic" if diagnostic else "generation_benchmark",
            "status": report.get("status", "unknown"), "error": report.get("error"),
            "error_type": report.get("error_type"), "guard_violation": report.get("guard_violation"),
            "exit_code": report.get("exit_code", report.get("returncode")), "started_at": report.get("started_at"),
            "model_revision": report.get("model_revision"), "binary_sha256": report.get("binary_sha256"),
            "configuration": {"threads": command_option(command, "-t", "--threads"),
                              "context_tokens": command_option(command, "-c", "--ctx-size"),
                              "batch_size": command_option(command, "-b", "--batch-size"),
                              "gpu_layers": command_option(command, "-ngl", "--gpu-layers"),
                              "cpu_moe": "--cpu-moe" in command,
                              "moe_cache_mib": command_option(command, "--moe-cache-mib"),
                              **{key: report.get(key) for key in ("mmap_demand", "expert_prefetch", "dense_prefetch")},
                              "unrecorded_flag_note": "Null experiment flags were not recorded in that raw report; no setting is inferred from a filename."},
            "hardware": report.get("hardware"), "os": report.get("os"), "mach_timebase": base,
            "load_seconds": number(report.get("load_seconds")),
            "full_process_wall_seconds": number(report.get("total_seconds", report.get("elapsed_seconds"))),
            "first_token_latency_including_load_sum_seconds": report["load_seconds"] + requests[0]["ttft_seconds"]
                if number(report.get("load_seconds")) is not None and requests and requests[0]["ttft_seconds"] is not None else None,
            "memory": {"sample_count": len(samples), "peak_sampled_rss_bytes": peak_rss,
                       "peak_sampled_physical_footprint_bytes": peak_footprint,
                       "maximum_pressure_level": finite_max(pressure), "system_swap_start_bytes": initial_swap,
                       "system_swap_final_or_last_sample_bytes": final_swap,
                       "system_swap_end_delta_bytes": final_swap - initial_swap
                           if final_swap is not None and initial_swap is not None else None,
                       "peak_system_swap_growth_bytes": max(0, peak_swap - initial_swap)
                           if peak_swap is not None and initial_swap is not None else number(report.get("peak_system_swap_growth_bytes"))},
            "cpu_process_cumulative": run_usage,
            "cpu_cumulative_endpoint": "explicit final counter before process shutdown" if final is not None else "last sample; final interval may be missing",
            "storage": {"process_cumulative_os_read_bytes": run_usage.get("storage_read_bytes") if run_usage else None,
                        "initialization_and_before_first_request_os_read_bytes": before_first.get("storage_read_bytes") if before_first else None,
                        "initialization_scope": "Cumulative process OS reads at first request's usage_before, includes model load and setup; not a pure weight-loading substage.",
                        "request_os_read_deltas_bytes": [r["request_usage_delta"].get("storage_read_bytes")
                            if r["request_usage_delta"] else None for r in requests],
                        "logical_weight_read_bytes": report.get("logical_weight_reads_bytes"),
                        "physical_nand_read_bytes": report.get("physical_nand_reads_bytes")},
            "gpu_system_utilization_percent": distribution(gpu),
            "thermal_before": report.get("thermal_before"), "thermal_after": report.get("thermal_after"),
            "cache_scope": report.get("cache_scope"), "measurement_scope": report.get("measurement_scope"),
            "component_timings_as_recorded": report.get("component_timings"),
            "requests": requests,
            "repeat_variability": {"scope": "Completed requests only; partial and failed requests remain in the requests array.",
                                   "ttft_seconds": distribution([r["ttft_seconds"] for r in complete_requests]),
                                   "observed_decode_tokens_per_second": distribution([r["decode"]["observed_tokens_per_second_recomputed"] for r in complete_requests]),
                                   "request_wall_seconds": distribution([r["wall_seconds"] for r in complete_requests])},
            "performance_claim_eligible": not diagnostic and report.get("status") == "complete"
                and bool(requests) and all(r["terminal_event_received"] for r in requests),
            "qualification_limit": "Eligibility means completed timing evidence only; it does not qualify broad context coverage, sustained long decoding, fidelity, GPU use or any throughput target."}


def operation_family(name):
    for prefix, family in (("ffn_moe_logits", "moe_router"), ("ffn_moe_probs", "moe_router"),
                           ("ffn_moe_argsort", "moe_router"), ("ffn_moe_topk", "moe_router"),
                           ("ffn_moe_weights", "moe_router"), ("ffn_moe_", "routed_expert_compute"),
                           ("ffn_shexp", "shared_expert"), ("ffn_", "feed_forward_other"),
                           ("kda_", "kda_attention"), ("ssm_", "recurrent_other"),
                           ("indexer_", "dsa_indexer"), ("hc_", "mhc_connections"),
                           ("attn_", "attention_other"), ("result_", "output_and_final_norm"),
                           ("inp_", "input")):
        if name.startswith(prefix):
            return family
    return "unnamed_or_other"


def add_bucket(buckets, key, row):
    bucket = buckets.setdefault(key, {"count": 0, "profiled_wall_seconds": 0.0,
                                      "cpu_seconds": 0.0, "cpu_counter_rows": 0,
                                      "os_read_bytes": 0, "os_read_counter_rows": 0,
                                      "pageins": 0, "pagein_counter_rows": 0})
    bucket["count"] += 1
    bucket["profiled_wall_seconds"] += number(row.get("seconds")) or 0
    for source, count in (("cpu_seconds", "cpu_counter_rows"), ("os_read_bytes", "os_read_counter_rows"), ("pageins", "pagein_counter_rows")):
        value = number(row.get(source))
        if value is not None:
            bucket[source] += value
            bucket[count] += 1


def summarize_profile(root):
    path = root / "operations.jsonl"
    groups = {"all": {"op": {}, "name": {}, "family": {}, "total": {}},
              "step0_prefill": {"op": {}, "name": {}, "family": {}, "total": {}},
              "steps_gt0_decode": {"op": {}, "name": {}, "family": {}, "total": {}}}
    errors, steps = [], Counter()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                if (not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 0
                        or not isinstance(row.get("op"), str) or not isinstance(row.get("name"), str)
                        or number(row.get("seconds")) is None or row["seconds"] < 0):
                    raise ValueError("Invalid operation row")
            except (ValueError, TypeError) as error:
                errors.append({"line": line_number, "error": str(error)})
                continue
            steps[str(row["step"])] += 1
            for group in (groups["all"], groups["step0_prefill" if row["step"] == 0 else "steps_gt0_decode"]):
                add_bucket(group["total"], "total", row)
                add_bucket(group["op"], row["op"], row)
                add_bucket(group["name"], (row["op"], row["name"]), row)
                add_bucket(group["family"], operation_family(row["name"]), row)
    summaries = {}
    for phase, group in groups.items():
        total = group["total"].get("total", {})
        wall = total.get("profiled_wall_seconds", 0)
        def ordered(buckets, kind):
            result = []
            for key, values in sorted(buckets.items(), key=lambda item: item[1]["profiled_wall_seconds"], reverse=True):
                label = {"op": key[0], "name": key[1]} if kind == "name" else {kind: key}
                result.append({**label, **values, "share_of_profiled_wall_percent": 100 * values["profiled_wall_seconds"] / wall if wall else None})
            return result
        summaries[phase] = {"totals": total, "by_op": ordered(group["op"], "op"),
                            "by_op_and_name": ordered(group["name"], "name"),
                            "by_family": ordered(group["family"], "family")}
    complete = None
    if (root / "complete.json").is_file():
        complete = json.loads((root / "complete.json").read_text())
    run = json.loads((root / "run.json").read_text()) if (root / "run.json").is_file() else None
    return {"source_directory": root.name, "run": run, "capture_completion": complete,
            "operation_rows_per_step": dict(steps), "invalid_rows": errors,
            "groups": summaries, "throughput_claim_allowed": False,
            "scope": "Diagnostic cb_eval forces a backend synchronization per operation and can alter fusion/overlap. Operation wall time includes that instrumented dispatch, not isolated GPU kernel time. Capture writes, load and scheduler copies between operations are outside these buckets. Step 0 labels all prefill chunks; greater steps are decode. OS read bytes are process-accounted, not physical NAND.",
            "cpu_scope": "cpu_seconds in operation rows were already converted from raw Mach ticks by native_fidelity; they are not converted again here."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.measurements.is_dir():
        parser.error("--measurements must be an existing directory")
    output = args.output.resolve()
    rows, ignored, errors = [], [], []
    for path in sorted(args.measurements.rglob("*.json")):
        if path.resolve() == output:
            continue
        relative = str(path.relative_to(args.measurements))
        if path.name.endswith(".progress.json"):
            ignored.append({"file": relative, "reason": "monitor progress snapshot; not a separate run or failed result"})
            continue
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict) or "status" not in report or "command" not in report:
                ignored.append({"file": relative, "reason": "not a benchmark or guarded-process report"})
                continue
            rows.append(summarize_measurement(report, relative))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append({"file": relative, "error_type": type(error).__name__, "error": str(error)})
    summary = {"schema_version": 1,
               "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scopes": SCOPES, "variants": [r for r in rows if r["kind"] == "generation_benchmark"],
               "guarded_diagnostics": [r for r in rows if r["kind"] == "guarded_diagnostic"],
               "ignored_files": ignored, "unreadable_reports": errors,
               "failed_data_policy": "Failed, stopped, incomplete and running reports are retained with status/reason and any available partial measurements. They are not promoted to completed performance results."}
    if args.profile:
        summary["instrumented_profile"] = summarize_profile(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"variants": len(summary["variants"]), "guarded_diagnostics": len(summary["guarded_diagnostics"]),
                      "unreadable_reports": len(errors), "output": str(args.output)}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
