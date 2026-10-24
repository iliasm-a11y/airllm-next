#!/usr/bin/env python3
"""Validate and summarize one completed 12-worker research51 campaign.

Standard-library analysis only: no engine import, model access, generation, or
rechecking current checkouts. Frozen before/after manifests certify the recorded
measurement interval; input JSON/guard hashes make this analysis reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

LABEL = re.compile(r"(?P<task>.+)\.r(?P<repeat>\d+)\.(?P<variant>baseline|candidate)\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite number: {label}")
    require(value > 0 if positive else value >= 0, f"Invalid nonnegative/positive number: {label}")
    return value


def count(value, label):
    require(type(value) is int and value >= 0, f"Invalid count: {label}")
    return value


def optional_number(value, label):
    return None if value is None else finite(value, label)


def load_json(path, inputs):
    raw = path.read_bytes()
    inputs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def quantile(values, probability):
    """Linear interpolation between sorted observations (Hyndman-Fan type 7)."""
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def describe(values):
    values = list(values)
    require(bool(values), "Cannot describe an empty sample")
    for value in values:
        finite(value, "summary observation")
    q1, q3 = quantile(values, .25), quantile(values, .75)
    return {"n": len(values), "median": statistics.median(values), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": min(values), "max": max(values)}


def optional_sum(values):
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


def optional_max(values):
    values = list(values)
    return None if any(value is None for value in values) else max(values)


def option_value(command, option):
    positions = [index for index, value in enumerate(command) if value == option]
    require(len(positions) == 1 and positions[0] + 1 < len(command), "Missing/duplicated command option " + option)
    return command[positions[0] + 1]


def validate_guard(guard, run, worker):
    require(type(guard.get("exit_code")) is int and guard["exit_code"] == 0,
            "Worker guard did not exit successfully: " + run["id"])
    require("violation" in guard and guard["violation"] is None
            and not guard.get("exception") and not guard.get("cleanup_error"), "Guard violation/error: " + run["id"])
    require(type(guard.get("sample_count")) is int and guard["sample_count"] > 0, "No independent guard samples")
    require(guard.get("command") == run["command"] + ["--worker"], "Guard command differs from frozen run")
    require(worker.get("command") == guard["command"][1:], "Worker command differs from supervised command")
    limits = guard.get("limits")
    require(isinstance(limits, dict) and limits, "Missing resource limits")
    for name in ("rss_bytes", "reserve_bytes", "swap_growth_bytes", "admitted_payload_bytes"):
        count(limits.get(name), "guard limit " + name)
    require(limits["rss_bytes"] > 0 and limits["reserve_bytes"] >= worker["options"]["reserve_bytes"],
            "Guard limit/reserve is invalid or looser than worker reserve")
    for name in ("timeout_seconds", "interval_seconds"):
        finite(limits.get(name), "guard limit " + name, positive=True)
    require(limits["interval_seconds"] <= 1, "Guard sampling interval exceeds one second")
    expected_timeout = float(option_value(run["command"], "--timeout")) if "--timeout" in run["command"] else 600.
    require(limits["timeout_seconds"] == expected_timeout, "Guard timeout differs from requested worker")
    count(limits.get("max_processes"), "guard process limit")
    require(limits["max_processes"] > 0, "Invalid guard process limit")
    peak = count(guard.get("sampled_peak_rss_bytes"), "sampled peak RSS")
    available = count(guard.get("minimum_available_bytes"), "minimum available")
    swap = count(guard.get("maximum_swap_growth_bytes"), "maximum swap growth")
    require(peak <= limits["rss_bytes"] and available >= limits["reserve_bytes"]
            and swap <= limits["swap_growth_bytes"], "Guard telemetry crossed a recorded resource limit")
    if guard.get("sampled_peak_processes") is not None:
        require(count(guard["sampled_peak_processes"], "sampled processes") <= limits["max_processes"],
                "Sampled process count exceeded guard limit")
    pressure = guard.get("peak_pressure_level")
    require(pressure is None or pressure == 1, "Guard observed elevated memory pressure")
    finite(guard.get("elapsed_seconds"), "guard elapsed", positive=True)
    for side in ("before", "after"):
        snapshot = guard.get(side)
        require(isinstance(snapshot, dict), "Missing guard system snapshot")
        for key in ("available_bytes", "physical_bytes", "swap_used_bytes"):
            count(snapshot.get(key), "guard " + side + " " + key)
    return limits


def validate_worker(worker, guard, job, run, campaign):
    ident = job["id"]
    require(worker.get("status") == "completed" and worker.get("label") == ident,
            "Incomplete/mislabeled worker: " + ident)
    for name in ("source_unchanged", "protocol_unchanged", "model_stat_unchanged", "engine_head_unchanged"):
        require(worker.get(name) is True, "Worker input identity changed: " + ident + " " + name)
    manifest = campaign["engine_manifests"].get(job["engine_root"])
    require(isinstance(manifest, dict) and manifest.get("source_sha256"), "Missing planned engine manifest")
    require(worker.get("source_sha256") == manifest["source_sha256"] and worker.get("git_head") == manifest["git_head"]
            and worker.get("git_head_after") == manifest["git_head"], "Worker engine differs from frozen manifest")
    protocol = campaign["protocol_sha256"]
    require(worker.get("protocol_sha256") == protocol, "Worker protocol differs from campaign")
    require(worker.get("harness_sha256") == protocol.get(run["command"][1]), "Worker harness differs from frozen protocol")
    require(worker.get("tasks_sha256") == protocol.get(job["tasks"]), "Worker task file differs from protocol")
    tasks = campaign["tasks_by_job"].get(ident)
    require(isinstance(tasks, list) and len(tasks) == 1 and worker.get("tasks_snapshot") == tasks,
            "Primary analysis requires one frozen task per fresh worker")
    require(worker.get("engine_root") == job["engine_root"], "Worker engine path differs from plan")
    require(option_value(run["command"], "--engine-root") == job["engine_root"]
            and option_value(run["command"], "--tasks") == job["tasks"]
            and option_value(run["command"], "--label") == ident, "Run command differs from plan")
    require(json.loads(option_value(run["command"], "--expected-protocol-json")) == protocol
            and option_value(run["command"], "--expected-engine-head") == manifest["git_head"]
            and json.loads(option_value(run["command"], "--expected-engine-sources-json")) == manifest["source_sha256"],
            "Run did not request the frozen engine and protocol")
    require(run["command"][-len(job["args"]):] == job["args"] if job["args"] else True, "Run controls differ from plan")
    turns = worker.get("turns")
    require(isinstance(turns, list) and len(turns) == 1, "Missing/extra primary turn")
    turn = turns[0]
    require(all(turn.get(key) == tasks[0].get(key) for key in ("id", "prompt", "max_new_tokens")), "Output task differs from request")
    require(turn.get("status") == "completed" and turn.get("stop_reason") == "eos", "Turn did not complete at EOS")
    tokens, stops = turn.get("generated_ids"), turn.get("eos_token_ids")
    require(isinstance(tokens, list) and len(tokens) >= 2 and len(tokens) <= tasks[0]["max_new_tokens"], "Invalid generated token count")
    require(isinstance(stops, list) and bool(stops), "Missing EOS token identities")
    for token in tokens + stops:
        count(token, "token identity")
    require(tokens[-1] in stops and not any(token in stops for token in tokens[:-1]), "EOS is missing or precedes final token")
    require(turn.get("partial_ids") == tokens, "Completed emitted token stream differs from final IDs")
    inputs = turn.get("full_input_ids")
    require(isinstance(inputs, list) and bool(inputs), "Missing full input IDs")
    for token in inputs:
        count(token, "input token identity")
    require(turn.get("appended_input_ids") == inputs and turn.get("reused_prefix_tokens") == 0,
            "Fresh-worker primary turn reused an unexpected KV prefix")
    require(turn.get("full_output_ids") == inputs + tokens, "Full output IDs differ from input plus generation")
    require(turn.get("finite_logits_checked") is True, "Finite logits were not checked")
    params = turn.get("generation_parameters")
    require(isinstance(params, dict) and params.get("temperature") == 0 and params.get("do_sample") is False
            and params.get("max_new_tokens") == tasks[0]["max_new_tokens"], "Primary generation is not the requested greedy work")
    require(isinstance(turn.get("text"), str) and isinstance(turn.get("raw_text"), str), "Missing complete response text")
    seconds = turn.get("token_seconds")
    require(isinstance(seconds, list) and len(seconds) == len(tokens), "Token timing count differs from token IDs")
    for value in seconds:
        finite(value, "token step seconds", positive=True)
    steps = turn.get("step_checks")
    require(isinstance(steps, list) and len(steps) == len(tokens), "Missing per-token logits/routes checks")
    for index, step in enumerate(steps):
        require(step.get("position") == len(inputs) + index, "Cache position differs from fresh causal sequence")
        for key in ("logits_sha256", "routes_sha256"):
            require(isinstance(step.get(key), str) and SHA256.fullmatch(step[key]), "Invalid per-token identity hash")
        finite(step.get("forward_seconds"), "forward seconds", positive=True)
        count(step.get("logical_bytes"), "forward logical read bytes")
        optional_number(step.get("process_storage_read_bytes"), "forward process storage reads")
    require(turn.get("native_cache_position") == steps[-1]["position"], "Final KV position differs from last forward")
    count(turn.get("logical_read_bytes"), "turn logical bytes")
    require(sum(step["logical_bytes"] for step in steps) == turn["logical_read_bytes"], "Forward logical bytes do not reconcile with turn")
    count(turn.get("read_calls"), "turn read calls")
    optional_number(turn.get("process_storage_read_bytes"), "turn process storage read bytes")
    for name in ("wall_seconds", "total_seconds", "ttft_seconds", "decode_tokens_per_second"):
        finite(turn.get(name), "turn " + name, positive=True)
    finite(worker.get("worker_wall_seconds"), "worker wall seconds", positive=True)
    finite(worker.get("initialization_seconds"), "worker initialization seconds")
    finite(worker.get("tokenizer_seconds"), "tokenizer seconds")
    require(guard["elapsed_seconds"] >= worker["worker_wall_seconds"], "Parent elapsed is shorter than its worker interval")
    metrics = turn.get("metrics")
    require(isinstance(metrics, dict), "Missing recorded model metrics")
    for section in ("allocator", "runtime", "cache", "weights", "io"):
        require(isinstance(metrics.get(section), dict), "Missing metric section " + section)
    return turn


def flat_row(worker, guard, turn, task, repeat, variant):
    metrics, n = turn["metrics"], len(turn["generated_ids"])
    decode_seconds = sum(turn["token_seconds"][1:])
    cache, weights, io = metrics["cache"], metrics["weights"], metrics["io"]
    hits, misses = cache.get("hits"), cache.get("misses")
    for value, label in ((hits, "cache hits"), (misses, "cache misses")):
        optional_number(value, label)
    packed = turn.get("packed_budget") or metrics.get("packed_buffer_budget") or {}
    return {"task": task, "repeat": repeat, "variant": variant, "label": worker["label"], "engine_head": worker["git_head"],
            "input_tokens": len(turn["full_input_ids"]), "generated_tokens_including_eos": n,
            "decode_tokens_including_final_eos": n - 1, "decode_step_seconds": decode_seconds,
            "decode_tok_s": turn["decode_tokens_per_second"], "reported_stream_decode_tok_s": turn["decode_tokens_per_second"],
            "stream_decode_seconds": (n - 1) / turn["decode_tokens_per_second"],
            "forward_only_decode_tps": (n - 1) / decode_seconds,
            "ttft_seconds": turn["ttft_seconds"], "generation_seconds": turn["total_seconds"],
            "generation_tok_s": n / turn["total_seconds"], "guard_elapsed_seconds": guard["elapsed_seconds"],
            "end_to_end_tok_s": n / guard["elapsed_seconds"], "worker_wall_seconds_excluding_imports": worker["worker_wall_seconds"],
            "worker_wall_tok_s_excluding_imports": n / worker["worker_wall_seconds"],
            "parent_minus_worker_seconds": guard["elapsed_seconds"] - worker["worker_wall_seconds"],
            "initialization_seconds": worker["initialization_seconds"], "tokenizer_seconds": worker["tokenizer_seconds"],
            "tokenizer_read_calls": optional_number(worker.get("tokenizer_io", {}).get("read_calls"), "tokenizer read calls"),
            "tokenizer_logical_bytes": optional_number(worker.get("tokenizer_io", {}).get("logical_bytes"), "tokenizer logical bytes"),
            "total_checkpoint_read_calls": optional_number(io.get("read_calls"), "checkpoint read calls"),
            "peak_owned_rss_bytes_sampled": guard["sampled_peak_rss_bytes"],
            "peak_mlx_allocator_bytes": optional_number(metrics["allocator"].get("peak_bytes"), "MLX peak"),
            "peak_context_payload_bytes": optional_number(metrics["runtime"].get("context_payload_peak_bytes"), "KV/context peak"),
            "peak_software_cache_bytes": optional_number(cache.get("peak_bytes"), "software cache peak"),
            "peak_checkpoint_io_reserved_bytes": optional_number(io.get("peak_reserved_bytes"), "checkpoint IO reservation peak"),
            "peak_packed_buffer_bytes": optional_number(packed.get("peak_bytes"), "packed buffer peak"),
            "minimum_available_bytes_sampled": guard["minimum_available_bytes"], "system_reserve_bytes": guard["limits"]["reserve_bytes"],
            "minimum_reserve_headroom_bytes_sampled": guard["minimum_available_bytes"] - guard["limits"]["reserve_bytes"],
            "swap_initial_system_bytes": guard["before"]["swap_used_bytes"], "swap_final_system_bytes": guard["after"]["swap_used_bytes"],
            "swap_delta_system_bytes": guard["after"]["swap_used_bytes"] - guard["before"]["swap_used_bytes"],
            "maximum_system_swap_growth_bytes_sampled": guard["maximum_swap_growth_bytes"],
            "peak_pressure_level_sampled": guard.get("peak_pressure_level"), "guard_sample_count": guard["sample_count"],
            "logical_read_bytes": turn["logical_read_bytes"], "read_calls": turn["read_calls"],
            "process_storage_read_bytes": turn.get("process_storage_read_bytes"),
            "decode_logical_read_bytes": sum(step["logical_bytes"] for step in turn["step_checks"][1:]),
            "decode_process_storage_read_bytes": optional_sum(step.get("process_storage_read_bytes") for step in turn["step_checks"][1:]),
            "cache_hits": hits, "cache_misses": misses,
            "cache_hit_ratio": hits / (hits + misses) if hits is not None and misses is not None and hits + misses else None,
            "cache_load_observations": optional_number(cache.get("load_observations"), "cache load observations"),
            "weight_loads": optional_number(weights.get("loads"), "weight loads"),
            "weight_read_dequant_seconds": optional_number(weights.get("read_dequant_seconds"), "read/dequant seconds"),
            "weight_packed_read_seconds": optional_number(weights.get("packed_read_seconds"), "packed read seconds"),
            "expert_load_seconds_separately_measured": None,
            "expert_routed_uses": optional_number(weights.get("routed_expert_uses"), "routed expert uses")}


def aggregate(rows):
    rows = list(rows)
    tokens = sum(row["generated_tokens_including_eos"] for row in rows)
    decode_tokens = sum(row["decode_tokens_including_final_eos"] for row in rows)
    return {"workers": len(rows), "generated_tokens_including_eos": tokens, "decode_tokens_including_final_eos": decode_tokens,
            "decode_step_seconds": sum(row["decode_step_seconds"] for row in rows),
            "stream_decode_seconds": sum(row["stream_decode_seconds"] for row in rows),
            "decode_tok_s": decode_tokens / sum(row["stream_decode_seconds"] for row in rows),
            "forward_only_decode_tps": decode_tokens / sum(row["decode_step_seconds"] for row in rows),
            "end_to_end_tok_s": tokens / sum(row["guard_elapsed_seconds"] for row in rows),
            "guard_elapsed_seconds": sum(row["guard_elapsed_seconds"] for row in rows),
            "generation_tok_s": tokens / sum(row["generation_seconds"] for row in rows),
            "worker_wall_tok_s_excluding_imports": tokens / sum(row["worker_wall_seconds_excluding_imports"] for row in rows),
            "ttft_seconds": describe(row["ttft_seconds"] for row in rows),
            "peak_owned_rss_bytes_sampled": max(row["peak_owned_rss_bytes_sampled"] for row in rows),
            "peak_mlx_allocator_bytes": optional_max(row["peak_mlx_allocator_bytes"] for row in rows),
            "minimum_available_bytes_sampled": min(row["minimum_available_bytes_sampled"] for row in rows),
            "minimum_reserve_headroom_bytes_sampled": min(row["minimum_reserve_headroom_bytes_sampled"] for row in rows),
            "maximum_system_swap_growth_bytes_sampled": max(row["maximum_system_swap_growth_bytes_sampled"] for row in rows),
            "logical_read_bytes": sum(row["logical_read_bytes"] for row in rows),
            "process_storage_read_bytes": optional_sum(row["process_storage_read_bytes"] for row in rows),
            "decode_logical_read_bytes_per_token": sum(row["decode_logical_read_bytes"] for row in rows) / decode_tokens,
            "decode_process_storage_read_bytes_per_token": (None if optional_sum(row["decode_process_storage_read_bytes"] for row in rows) is None
                else sum(row["decode_process_storage_read_bytes"] for row in rows) / decode_tokens),
            "cache_hits": optional_sum(row["cache_hits"] for row in rows),
            "cache_misses": optional_sum(row["cache_misses"] for row in rows)}


def analyze_campaign(folder):
    folder = Path(folder).resolve()
    inputs = []
    campaign = load_json(folder / "campaign.json", inputs)
    require(campaign.get("status") == "completed", "Only a completed campaign can be analyzed")
    require(campaign.get("protocol_unchanged") is True and campaign.get("engines_unchanged") is True,
            "Campaign engine/protocol identity was not preserved")
    manifests = campaign.get("engine_manifests")
    protocol = campaign.get("protocol_sha256")
    require(isinstance(manifests, dict) and manifests and campaign.get("engine_manifests_after") == manifests,
            "Campaign before/after engine manifests differ")
    require(isinstance(protocol, dict) and protocol and all(isinstance(value, str) and SHA256.fullmatch(value) for value in protocol.values()),
            "Missing frozen protocol hashes")
    plan, runs = campaign.get("plan"), campaign.get("runs")
    require(isinstance(plan, list) and isinstance(runs, list) and len(plan) == len(runs) == 12,
            "Primary campaign must contain exactly twelve workers")
    require([job.get("id") for job in plan] == [run.get("id") for run in runs], "Runs are missing/reordered relative to plan")
    require(len({job["id"] for job in plan}) == 12, "Duplicated job identity")
    raw, rows, groups, role_roots = {}, [], {}, defaultdict_set()
    common = None
    guard_limits = None
    for job, run in zip(plan, runs):
        ident = job["id"]
        match = LABEL.fullmatch(ident)
        require(match is not None and Path(ident).name == ident, "Invalid primary job label")
        task, repeat, variant = match["task"], int(match["repeat"]), match["variant"]
        require(repeat in (0, 1), "Primary repetition must be zero or one")
        require(run.get("status") == "completed" and type(run.get("exit_code")) is int and run["exit_code"] == 0,
                "Campaign contains an unsuccessful or incomplete run")
        require(isinstance(job.get("args"), list), "Missing frozen job controls")
        worker_path = folder / (ident + ".json")
        guard_path = worker_path.with_suffix(".guard.json")
        recorded_output = Path(option_value(run["command"], "--output"))
        recorded_guard = Path(run.get("guard_file", ""))
        require(recorded_output.name == ident + ".json" and recorded_guard == recorded_output.with_suffix(".guard.json"),
                "Recorded worker/guard files differ from the run identity")
        worker, guard = load_json(worker_path, inputs), load_json(guard_path, inputs)
        require(isinstance(worker.get("options"), dict), "Missing worker options")
        limits = validate_guard(guard, run, worker)
        if guard_limits is None:
            guard_limits = limits
        require(limits == guard_limits, "Primary workers used different guard limits")
        turn = validate_worker(worker, guard, job, run, campaign)
        require(turn["id"] == task, "Job task label differs from output task")
        identity = {key: worker.get(key) for key in ("model", "hardware", "versions", "cache_regime")}
        require(isinstance(identity["model"], dict) and identity["model"].get("bytes", 0) > 0, "Missing recorded model identity")
        require(worker.get("physical_ssd_bandwidth_certified") is False, "Unexpected physical-bandwidth certification claim")
        if common is None:
            common = identity
        require(identity == common, "Primary workers differ in model identity, machine, versions or cache regime")
        key = (task, repeat)
        groups.setdefault(key, {})
        require(variant not in groups[key], "Duplicate paired variant")
        row = flat_row(worker, guard, turn, task, repeat, variant)
        rows.append(row)
        groups[key][variant] = row
        raw[ident] = (worker, turn)
        role_roots[variant].add(job["engine_root"])
    require(len(groups) == 6 and len({task for task, _ in groups}) == 3 and all(set(group) == {"baseline", "candidate"} for group in groups.values()),
            "Primary campaign requires six exact baseline/candidate pairs across three prompts")
    require(all(len(roots) == 1 for roots in role_roots.values()), "A variant changed engine checkout during campaign")
    require(role_roots["baseline"] != role_roots["candidate"], "Baseline and candidate unexpectedly use the same engine checkout")
    pairs = []
    for (task, repeat), group in sorted(groups.items()):
        baseline, candidate = group["baseline"], group["candidate"]
        bw, bt = raw[baseline["label"]]
        cw, ct = raw[candidate["label"]]
        for key in ("full_input_ids", "appended_input_ids", "generated_ids", "full_output_ids", "eos_token_ids",
                    "generation_parameters", "native_cache_position", "text", "raw_text"):
            require(bt[key] == ct[key], f"Pair {task}.r{repeat} differs in {key}")
        for key in ("logits_sha256", "routes_sha256", "position"):
            require([step[key] for step in bt["step_checks"]] == [step[key] for step in ct["step_checks"]],
                    f"Pair {task}.r{repeat} differs in per-token {key}")
        require({key: value for key, value in bw["options"].items() if key != "zero_copy_weights"}
                == {key: value for key, value in cw["options"].items() if key != "zero_copy_weights"}, "Uncontrolled model option differs between paired workers")
        require({key: value for key, value in bw["backend_options"].items() if key not in ("packed_copy_free", "packed_direct_io", "bulk_metadata")}
                == {key: value for key, value in cw["backend_options"].items() if key not in ("packed_copy_free", "packed_direct_io", "bulk_metadata")},
                "Uncontrolled backend option differs between paired workers")
        pairs.append({"task": task, "repeat": repeat, "baseline": baseline, "candidate": candidate,
                      "tokens_logits_routes_positions_exact": True, "complete_response_text_exact": True,
                      "decode_speedup": candidate["decode_tok_s"] / baseline["decode_tok_s"],
                      "forward_only_decode_speedup": candidate["forward_only_decode_tps"] / baseline["forward_only_decode_tps"],
                      "end_to_end_speedup": candidate["end_to_end_tok_s"] / baseline["end_to_end_tok_s"],
                      "ttft_speedup": baseline["ttft_seconds"] / candidate["ttft_seconds"],
                      "order": [row["variant"] for row in rows if row["task"] == task and row["repeat"] == repeat]})
    agg = {variant: aggregate(row for row in rows if row["variant"] == variant) for variant in ("baseline", "candidate")}
    return build_summary(campaign, inputs, common, rows, pairs, agg), rows, raw


def defaultdict_set():
    return {"baseline": set(), "candidate": set()}


def build_summary(campaign, inputs, common, rows, pairs, agg):
    return {"kind": "complete_guarded_paired_generation_research51", "fixture": campaign.get("fixture", False),
            "pair_count": len(pairs), "worker_count": len(rows), "prompt_count": 3, "repeats_per_prompt": 2,
            "all_tokens_logits_routes_positions_exact": True, "all_complete_responses_exact": True,
            "engine_manifests": campaign["engine_manifests"], "protocol_sha256": campaign["protocol_sha256"],
            "environment_and_model_stat_identity": common, "inputs": inputs,
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "aggregate": agg,
            "aggregate_decode_speedup": agg["candidate"]["decode_tok_s"] / agg["baseline"]["decode_tok_s"],
            "aggregate_end_to_end_speedup": agg["candidate"]["end_to_end_tok_s"] / agg["baseline"]["end_to_end_tok_s"],
            "paired_decode_speedup": describe(pair["decode_speedup"] for pair in pairs),
            "paired_end_to_end_speedup": describe(pair["end_to_end_speedup"] for pair in pairs),
            "paired_forward_only_decode_speedup": describe(pair["forward_only_decode_speedup"] for pair in pairs),
            "paired_ttft_speedup": describe(pair["ttft_speedup"] for pair in pairs),
            "paired_decode_wins": sum(pair["decode_speedup"] > 1 for pair in pairs), "pairs": pairs,
            "by_prompt": {task: {variant: aggregate(row for row in rows if row["task"] == task and row["variant"] == variant)
                                 for variant in ("baseline", "candidate")} for task in sorted({row["task"] for row in rows})},
            "metric_scope": {
                "decode": "primary delivered streaming metric: sum(n-1 tokens including final EOS) / sum((n-1)/runtime decode_tokens_per_second); includes callback/journal work between first and last token",
                "forward_only_decode_tps": "sum(n-1 tokens) / sum(token_seconds[1:]); excludes first token and between-step callbacks, includes synchronous correctness instrumentation",
                "reported_stream_decode": "original runtime field retained separately; its end-minus-first interval includes callback work",
                "end_to_end": "all generated token IDs including EOS / parent supervisor elapsed; includes process startup/imports and supervision",
                "worker_wall": "worker interval starts after imports; includes tokenizer/model initialization and task execution",
                "parent_minus_worker": "startup/import/supervision/teardown difference; not a pure import timing",
                "ttft": "runtime first synchronized token interval; excludes process imports and model initialization",
                "memory": "owned RSS sampled by parent and MLX allocator peak are distinct overlapping measures, never added",
                "swap": "system-wide observed changes; other applications may contribute",
                "physical_reads": "process OS-accounted storage read bytes; unknown remains null; not certified NAND/SSD bandwidth",
                "expert_load_times": "per-expert load times absent; aggregate packed-read and read/dequant counters retained separately",
                "exactness": "per-token logits/routes hashes, token IDs and positions; full KV/recurrent payload equality was not captured here",
                "provenance": "recorded before/after source/protocol/model identities for measurement interval; current checkouts and weights not read",
                "statistics": "six descriptive paired observations over three prompts; IQR uses linear quantiles; no population/thermal/general quality guarantee"},
            "limitations": ["Model stat identity is frozen; this harness did not recalculate the complete model-file hash.",
                            "Fresh workers do not establish cold physical SSD or controlled OS page-cache state.",
                            "Complete Qwen checkpoint execution on these prompts is not a general quality certification.",
                            "No GLM-5.3, competitor performance, sustained thermal or external SSD promotion follows."]}


def fenced(text, language="text"):
    runs = [len(match[0]) for match in re.finditer(r"`+", text)]
    fence = "`" * max(3, max(runs, default=0) + 1)
    return fence + language + "\n" + text + ("" if text.endswith("\n") else "\n") + fence + "\n"


def write_outputs(output, summary, rows, raw):
    output = Path(output)
    require(not output.exists(), "Refusing to overwrite an existing analysis directory")
    # Detect a changed evidence snapshot before materializing any deliverable.
    for source in summary["inputs"]:
        require(hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"], "Evidence changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with (output / "benchmarks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Réponses intégrales — campagne 5.1\n\n",
            "Campagne complète et appariée. Tokens, logits, routes et positions concordent par paire. Les textes ci-dessous sont les sorties enregistrées, sans réécriture. Le token EOS figure dans les IDs et dans le texte brut.\n\n"]
    if summary["fixture"]:
        text.append("**FIXTURES SYNTHÉTIQUES DE TEST : ces données ne sont pas un benchmark modèle.**\n\n")
    for row in rows:
        worker, turn = raw[row["label"]]
        text.extend(["## " + row["label"] + "\n\n", "Commit moteur : `" + worker["git_head"] + "`. Arrêt : EOS.\n\n",
                     "### Prompt intégral\n\n", fenced(turn["prompt"]), "\n### Réponse affichée intégrale\n\n", fenced(turn["text"]),
                     "\n### Sortie brute intégrale\n\n", fenced(turn["raw_text"]), "\n### IDs générés, EOS inclus\n\n",
                     fenced(json.dumps(turn["generated_ids"]), "json"), "\n"])
    (output / "responses.md").write_text("".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("campaign", type=Path, help="Directory containing campaign.json and guarded worker evidence")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    args = parser.parse_args()
    summary, rows, raw = analyze_campaign(args.campaign)
    write_outputs(args.output, summary, rows, raw)
    print(json.dumps({"pair_count": summary["pair_count"], "all_exact": True,
                      "aggregate_decode_speedup": summary["aggregate_decode_speedup"],
                      "aggregate_end_to_end_speedup": summary["aggregate_end_to_end_speedup"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
