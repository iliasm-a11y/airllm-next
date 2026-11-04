"""Fail-closed CPU snapshot diagnostic comparator, not an EOS qualification.

Requires raw full logits, complete sequence serializations, logical tensor
captures and explicit successful guard receipts from both owned runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from research.glm53_batched_prefill.prefix_boundary import report_boundary
from research.glm53_batched_prefill.snapshot_key import canonical_key

from research.glm53_m5.compare_fidelity import (
    CRITERIA, compare_arrays, family, json_lines, safe_file, tensor_view,
)

GiB, MiB = 1024 ** 3, 1024 ** 2
FAMILIES = ("routes", "router", "kda")
# Original GLM-5.3-Flash checkpoint: three dense-only layers, then 42 MoE
# layers; KDA is used except at attention layers 3,7,...,43.
EXPECTED_NAMES = {
    *(f"ffn_moe_topk-{layer}" for layer in range(3, 45)),
    *(f"ffn_moe_logits-{layer}" for layer in range(3, 45)),
    *(f"kda_scan_out-{layer}" for layer in range(45) if layer % 4 != 3),
}
EXPECTED_STATE_TARGETS = {
    *(f"cache_r_l{layer}" for layer in range(45) if layer % 4 != 3),
    *(f"cache_s_l{layer}" for layer in range(45) if layer % 4 != 3),
    *(f"cache_k_l{layer}" for layer in range(3, 45, 4)),
    *(f"cache_idx_k_l{layer}" for layer in range(3, 45, 4)),
}


def _record(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _guard(receipt):
    """Require a conservative sufficient domain of the existing native policy.

    Warning pressure alone is allowed only with full warning headroom and
    bounded warning swap. Reserve shortfalls are refused without relying on
    grace timestamps. The existing watchdog is replayed as a consistency
    check; no native threshold or numeric-fidelity criterion is changed.
    """
    from air_llm.airllm.nextgen.glm_native import NativeGlmModel, NativeLimits

    def number(value, name, *, minimum=0, maximum=2**53):
        if (type(value) not in (int, float) or not minimum <= value <= maximum or
                not math.isfinite(value)):
            raise ValueError("missing or nonfinite numeric guard metadata: " + name)
        return value

    def byte_count(value, name, *, minimum=0, maximum=2**64 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("missing or invalid integer guard metadata: " + name)
        return value

    row = _record(receipt)
    if not isinstance(row, dict):
        raise ValueError("an explicit guard receipt is required")
    if (row.get("status") != "completed" or type(row.get("returncode")) is not int or
            row["returncode"] != 0 or "guard_violation" not in row or row["guard_violation"] is not None):
        raise ValueError("native run did not exit successfully under clear guards")
    if row.get("power_context_passed") is not True or row.get("integrity_passed") is not True:
        raise ValueError("AC/lid/continuity and integrity receipts are required")
    limits = row.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("complete native resource limits are required")
    native = limits.get("native")
    defaults = vars(NativeLimits())
    if not isinstance(native, dict) or set(native) != set(defaults):
        raise ValueError("complete native watchdog limits are required")
    for name, default in defaults.items():
        value = native[name]
        if type(default) is int:
            byte_count(value, name, minimum=1)
        else:
            number(value, name, minimum=0)
            if value <= 0:
                raise ValueError("positive native watchdog limit required: " + name)
        if name in ("reserve_bytes", "base_workspace_bytes", "warning_headroom_bytes"):
            if value < default:
                raise ValueError("native resource reserve was relaxed: " + name)
        elif name not in ("max_rss_bytes", "max_swap_growth_bytes", "request_timeout_seconds") and value > default:
            raise ValueError("native watchdog limit was relaxed: " + name)
    if not 0 < native["request_timeout_seconds"] <= 2700:
        raise ValueError("native request timeout exceeds the authorized ceiling")
    rss = byte_count(limits.get("rss_bytes"), "limits.rss_bytes", minimum=1, maximum=int(8.5 * GiB))
    swap = byte_count(limits.get("swap_growth_bytes"), "limits.swap_growth_bytes", minimum=1, maximum=32 * MiB)
    if native["max_rss_bytes"] != rss or native["max_swap_growth_bytes"] != swap:
        raise ValueError("native and reported hard resource limits differ")
    selected_limits = NativeLimits(**native)
    interval = number(row.get("sampling_interval_seconds"), "sampling_interval_seconds", maximum=.5)
    if interval <= 0:
        raise ValueError("positive sampling interval required")

    policy_path = Path(NativeGlmModel._guard_resource_sample.__code__.co_filename).resolve()
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    preflight = row.get("preflight")
    metadata = preflight.get("metadata") if isinstance(preflight, dict) else None
    recorded_hashes = metadata.get("sha256") if isinstance(metadata, dict) else None
    if not isinstance(recorded_hashes, dict):
        raise ValueError("original native watchdog source identity is required")
    policies = [value for name, value in recorded_hashes.items()
                if isinstance(name, str) and name.replace("\\", "/").endswith("/airllm/nextgen/glm_native.py")]
    if len(policies) != 1 or policies[0] != policy_sha256:
        raise ValueError("native watchdog source identity missing or changed")

    baseline = row.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("native baseline metadata is required")
    baseline_time = number(baseline.get("monotonic"), "baseline.monotonic")
    byte_count(baseline.get("swap_used_bytes"), "baseline.swap_used_bytes")
    byte_count(baseline.get("available_bytes"), "baseline.available_bytes")
    if type(baseline.get("pressure_level")) is not int or baseline["pressure_level"] != 1:
        raise ValueError("native startup requires normal pressure")
    admission = row.get("admission")
    if not isinstance(admission, dict) or admission.get("passed") is not True:
        raise ValueError("successful native startup admission receipt required")
    admitted_sample = admission.get("sample")
    if not isinstance(admitted_sample, dict):
        raise ValueError("native startup admission sample is required")
    number(admitted_sample.get("monotonic"), "admission.sample.monotonic")
    byte_count(admitted_sample.get("swap_used_bytes"), "admission.sample.swap_used_bytes")
    byte_count(admitted_sample.get("available_bytes"), "admission.sample.available_bytes")
    if (type(admitted_sample.get("pressure_level")) is not int or admitted_sample["pressure_level"] != 1 or
            any(admitted_sample.get(key) != baseline[key] for key in
                ("monotonic", "swap_used_bytes", "available_bytes", "pressure_level"))):
        raise ValueError("native startup admission and baseline differ")
    required = byte_count(admission.get("required_bytes_estimate"), "admission.required_bytes_estimate", minimum=1)
    if required < selected_limits.reserve_bytes + selected_limits.base_workspace_bytes or baseline["available_bytes"] < required:
        raise ValueError("native startup workspace and reserve admission failed")

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("no guard observations")
    guard = object.__new__(NativeGlmModel)
    guard.limits = selected_limits
    guard._warning_started = guard._headroom_low_started = None
    guard._pressure_context = None
    previous_time = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("structured native resource sample required")
        now = number(sample.get("monotonic"), "sample.monotonic")
        if now < baseline_time or previous_time is not None and now <= previous_time:
            raise ValueError("native resource sample clocks are reversed or repeated")
        previous_time = now
        byte_count(sample.get("rss_bytes"), "sample.rss_bytes")
        byte_count(sample.get("available_bytes"), "sample.available_bytes")
        used = byte_count(sample.get("swap_used_bytes"), "sample.swap_used_bytes")
        growth = byte_count(sample.get("swap_growth_bytes"), "sample.swap_growth_bytes")
        if growth != max(0, used - baseline["swap_used_bytes"]):
            raise ValueError("reported swap growth differs from baseline counters")
        pressure = sample.get("pressure_level")
        if type(pressure) is not int or pressure not in (1, 2):
            raise ValueError("only observed normal/contextual-warning pressure is admitted")
        if sample["available_bytes"] < selected_limits.reserve_bytes:
            raise ValueError("reserve shortfall refused without inferring native grace timing")
        if pressure == 2 and (
                sample["available_bytes"] < selected_limits.reserve_bytes + selected_limits.warning_headroom_bytes or
                growth > selected_limits.warning_swap_growth_bytes):
            raise ValueError("warning-pressure sufficient headroom/swap bounds failed")
        code = NativeGlmModel._guard_resource_sample(guard, sample, baseline, now)
        if code is not None:
            raise ValueError(f"existing native resource policy rejected sample{index}: {code}")
    return row

def _payload(root, name, size, expected_hash):
    path = safe_file(root, name)
    if path.stat().st_size != size or size <= 0:
        raise ValueError("payload length mismatch: " + name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("payload hash mismatch: " + name)
    return path


def _read_writes(root, filename="state_hashes.jsonl", *, allow_empty=False):
    path = safe_file(root, filename)
    if path.stat().st_size > 64 * MiB:
        raise ValueError("observed state-write metadata exceeds64MiB bound")
    rows = json_lines(path)
    if (not rows and not allow_empty) or len(rows) > 262144:
        raise ValueError("observed state-write records empty or exceed bound")
    for row in rows:
        if (not isinstance(row, dict) or row.get("status") != "hashed" or
                row.get("finite") is not True or
                not isinstance(row.get("sha256"), str) or
                re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None or
                type(row.get("request")) is not int or row["request"] not in (0, 1, 2) or
                type(row.get("offset")) is not int or row["offset"] < 0 or
                row.get("op") not in ("CPY", "SET_ROWS") or
                type(row.get("type")) is not int or row["type"] not in (0, 1, 30) or
                type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= 512 * MiB):
            raise ValueError("unhashed, nonfinite or invalid observed state-write record")
        shape = row.get("shape")
        if (not isinstance(shape, list) or len(shape) != 4 or
                any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("invalid observed state-write shape")
        logical = (4 if row["type"] == 0 else 2) * math.prod(shape)
        if (type(row.get("logical_bytes")) is not int or row["logical_bytes"] != logical or
                not 0 <= logical <= min(row["bytes"], 512 * MiB)):
            raise ValueError("invalid observed state-write logical byte coverage")
        target = row.get("target")
        canonical = re.match(r"^(cache_(?:idx_k|k|r|s)_l\d+)(?:\s|$)", target) if isinstance(target, str) else None
        if canonical is None or canonical.group(1) not in EXPECTED_STATE_TARGETS:
            raise ValueError("unexpected observed state-write target")
        row["canonical_target"] = canonical.group(1)
    return rows


def _read_continuation(root, report):
    expected = set()
    expected_logits = set()
    for request, row in enumerate(report["requests"]):
        samples = row.get("continuation")
        if (row.get("continuation_complete") is not True or not isinstance(samples, list) or
                not 1 <= len(samples) <= 4):
            raise ValueError("four greedy diagnostic samples or a natural EOS are required")
        for step, sample in enumerate(samples):
            if (not isinstance(sample, dict) or type(sample.get("step")) is not int or sample["step"] != step or
                    type(sample.get("token")) is not int or not 0 <= sample["token"] < report["vocab_size"] or
                    type(sample.get("native_eog")) is not bool or
                    (sample["native_eog"] and step != len(samples) - 1)):
                raise ValueError("invalid greedy continuation token sequence")
            path = _payload(root, f"continuation-logits-{request}-{step}.f32",
                            report["vocab_size"] * 4, sample.get("logits_sha256"))
            expected_logits.add(path.name)
            logits = np.memmap(path, dtype="<f4", mode="r")
            if not np.all(np.isfinite(logits)):
                raise ValueError("nonfinite greedy continuation full logits")
            if int(np.argmax(logits)) != sample["token"]:
                raise ValueError("greedy continuation token differs from full-logit argmax")
            if step == 0 and sample["logits_sha256"] != row["logits_sha256"]:
                raise ValueError("continuation step0 differs from the final prefill logits")
            if step:
                expected.add((request, step))
        eos = samples[-1]["native_eog"]
        if ((eos and row.get("continuation_stop") != "eos") or
                (not eos and (len(samples) != 4 or row.get("continuation_stop") != "diagnostic_budget"))):
            raise ValueError("greedy diagnostic stop is incomplete or misclassified")
        if row.get("continuation_next_position") != row["prompt_tokens"] + len(samples) - 1:
            raise ValueError("continuation state position differs from actually fed diagnostic tokens")
        if type(row.get("continuation_state_bytes")) is not int or not 0 < row["continuation_state_bytes"] <= 256 * MiB:
            raise ValueError("post-continuation sequence size is missing or exceeds256MiB")
        _payload(root, f"continuation-state-{request}.bin", row["continuation_state_bytes"],
                 row.get("continuation_state_sha256"))
    if {p.name for p in root.glob("continuation-logits-*.f32")} != expected_logits:
        raise ValueError("continuation full-logit files are missing or extra")
    tensors, occurrences, coverage = {}, {}, {}
    for row in json_lines(safe_file(root, "continuation_tensors.jsonl")):
        label = family(row["name"])
        request, step = row.get("request"), row.get("offset")
        if (type(request) is not int or type(step) is not int or (request, step) not in expected or row.get("phase") != "continuation" or
                row.get("input_tokens") != 1 or row.get("batch_index") != step or
                row.get("type") != (26 if label == "routes" else 0)):
            raise ValueError("invalid continuation tensor descriptor or step")
        key = request, step, row["name"]
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        tensors[(*key, occurrence)] = row
        _payload(root, row["file"], row["bytes"], row.get("sha256"))
        coverage.setdefault((request, step), set()).add(row["name"])
        if label == "result_output":
            values = tensor_view(root, row)
            if values.shape != (report["vocab_size"], 1, 1, 1):
                raise ValueError("continuation result_output is not a complete vocabulary")
            logits = np.memmap(root / f"continuation-logits-{request}-{step}.f32", dtype="<f4", mode="r")
            if not compare_arrays(values[:, 0, 0, 0], logits, CRITERIA["cpu"])["passed"]:
                raise ValueError("captured continuation output differs from its complete sampled logits")
    if set(coverage) != expected or any(not EXPECTED_NAMES | {"result_output"} <= names for names in coverage.values()):
        raise ValueError("continuation route/router/KDA/output layer coverage is incomplete")
    writes = _read_writes(root, "continuation_states.jsonl", allow_empty=True)
    for row in writes:
        if ((row["request"], row["offset"]) not in expected or row.get("phase") != "continuation" or
                row.get("input_tokens") != 1 or row.get("batch_index") != row["offset"]):
            raise ValueError("invalid continuation observed-write step")
    _compare_writes(writes, writes, sorted(expected), False)
    return {"tensors": tensors, "writes": writes}


def _compare_continuation(a_root, b_root, a_report, b_report, a_capture, b_capture, pairs=((0, 0), (1, 1), (2, 2))):
    result = {"passed": True, "requests": [], "scope":
              "at most4 native greedy diagnostic samples and their finite logits, captured tensors, observed writes and post-state; no useful EOS or functional agent qualification"}
    for a_request, b_request in pairs:
        ar, br = a_report["requests"][a_request], b_report["requests"][b_request]
        a_samples, b_samples = ar["continuation"], br["continuation"]
        tokens_equal = ([(r["token"], r["native_eog"]) for r in a_samples] ==
                        [(r["token"], r["native_eog"]) for r in b_samples] and
                        ar["continuation_stop"] == br["continuation_stop"])
        row = {"reference_request": a_request, "candidate_request": b_request,
               "tokens_equal": tokens_equal, "full_logits": [], "tensors_identical": True,
               "post_state_identical": ar["continuation_state_bytes"] == br["continuation_state_bytes"] and
                                       ar["continuation_state_sha256"] == br["continuation_state_sha256"]}
        if len(a_samples) == len(b_samples):
            for step in range(len(a_samples)):
                row["full_logits"].append(compare_arrays(
                    np.memmap(a_root / f"continuation-logits-{a_request}-{step}.f32", dtype="<f4", mode="r"),
                    np.memmap(b_root / f"continuation-logits-{b_request}-{step}.f32", dtype="<f4", mode="r"), CRITERIA["cpu"]))
        selected = lambda data, request: {key[1:]: value for key, value in data["tensors"].items() if key[0] == request}
        at, bt = selected(a_capture, a_request), selected(b_capture, b_request)
        row["tensors_identical"] = set(at) == set(bt)
        if row["tensors_identical"]:
            for key, av in at.items():
                if not compare_arrays(tensor_view(a_root, av), tensor_view(b_root, bt[key]), CRITERIA["cpu"],
                                      integer=family(av["name"]) == "routes")["passed"]:
                    row["tensors_identical"] = False
                    break
        selected_writes = lambda data, request: [{**r, "request": 0} for r in data["writes"] if r["request"] == request]
        try:
            row["observed_writes"] = _compare_writes(selected_writes(a_capture, a_request), selected_writes(b_capture, b_request),
                                                     [(0, n) for n in range(1, len(a_samples))], False)
        except ValueError as error:
            row["observed_writes"] = {"passed": False, "error": str(error)}
        row["passed"] = bool(tokens_equal and len(row["full_logits"]) == len(a_samples) and
            all(r["passed"] for r in row["full_logits"]) and row["post_state_identical"] and
            row["tensors_identical"] and row["observed_writes"]["passed"])
        result["passed"] &= row["passed"]
        result["requests"].append(row)
    return result


def _direct_restore(reference_root, reference, candidate):
    boundary = report_boundary(reference)
    if boundary != report_boundary(candidate) or boundary % reference["microbatch"]:
        raise ValueError("direct full/restore comparison requires the same aligned boundary")
    path = safe_file(reference_root, "state_checkpoints.jsonl")
    if path.stat().st_size > MiB:
        raise ValueError("direct-restore checkpoint metadata exceeds1MiB")
    rows = [r for r in json_lines(path) if r.get("token_end") == boundary]
    if len(rows) != 3 or {r.get("request") for r in rows} != {0, 1, 2}:
        raise ValueError("three reference prefix checkpoints are required for direct restore")
    for row in rows:
        if (row.get("status") != "hashed" or row.get("state_seq_flags") != 0 or row.get("next_position") != boundary or
                type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 256 * MiB or
                not isinstance(row.get("sha256"), str) or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("invalid full reference prefix checkpoint")
    first = next(row for row in rows if row["request"] == 0)
    if any(row["bytes"] != first["bytes"] or row["sha256"] != first["sha256"] for row in rows):
        raise ValueError("reference prefix states differ between the three complete prompts")
    snapshot = candidate.get("snapshot", {})
    if snapshot.get("bytes") != first["bytes"] or snapshot.get("sha256") != first["sha256"]:
        raise ValueError("immutable snapshot differs from the common full reference prefix state")
    return {"passed": True, "boundary": boundary, "reference_checkpoints": 3,
            "snapshot_bytes": first["bytes"], "snapshot_sha256": first["sha256"],
            "scope": "same physical microbatch boundaries plus exact complete-state serialization at the declared prefix; alignment alone does not qualify restore"}


def _compare_writes(a, b, expected_batches, suffix_only, boundary=208):
    selected = lambda rows: [r for r in rows if not suffix_only or r["offset"] >= boundary]
    a, b = selected(a), selected(b)
    for rows in (a, b):
        groups = {}
        order = []
        previous = None
        for row in rows:
            key = row["request"], row["offset"]
            groups.setdefault(key, set()).add(row["canonical_target"])
            if key != previous:
                order.append(key)
                previous = key
        if order != expected_batches or any(targets != EXPECTED_STATE_TARGETS for targets in groups.values()):
            raise ValueError("observed state-write coverage/order incomplete at a compared microbatch")
    keys = ("request", "offset", "target", "op", "type", "bytes", "shape", "logical_bytes", "sha256")
    mismatches = sum(any(x[k] != y[k] for k in keys) for x, y in zip(a, b)) + abs(len(a) - len(b))
    result = {"passed": mismatches == 0, "reference_events": len(a), "candidate_events": len(b),
              "mismatching_events": mismatches, "scope":
              "ordered observed CPY/SET_ROWS writes to attention, KDA R/S and indexer caches; no claim for unobserved state"}
    for index, (x, y) in enumerate(zip(a, b)):
        if any(x[k] != y[k] for k in keys):
            result["first_divergence"] = {"event": index, "request": x["request"], "offset": x["offset"],
                                          "reference_target": x["target"], "candidate_target": y["target"]}
            break
    return result


def _check_replay(root, report, tensors, batches, writes):
    """Identical native input replay, not a private-history isolation test."""
    boundary = report_boundary(report)
    a, b = report["requests"][0], report["requests"][2]
    suffix_only = report["mode"] == "restore"
    same_prompt = all(a[name] == b[name] for name in ("prompt_tokens", "prompt_ids_sha256", "next_position"))
    logits = compare_arrays(np.memmap(root / "logits-0.f32", dtype="<f4", mode="r"),
                            np.memmap(root / "logits-2.f32", dtype="<f4", mode="r"), CRITERIA["cpu"])
    same_state = a["sequence_bytes"] == b["sequence_bytes"] and a["sequence_sha256"] == b["sequence_sha256"]
    selected = lambda request: {key[1:]: row for key, row in tensors.items()
                                if key[0] == request and (not suffix_only or key[1] >= boundary)}
    at, bt = selected(0), selected(2)
    same_captures = bool(at) and set(at) == set(bt)
    if same_captures:
        for key, ar in at.items():
            br = bt[key]
            if not compare_arrays(tensor_view(root, ar), tensor_view(root, br), CRITERIA["cpu"],
                                  integer=family(ar["name"]) == "routes")["passed"]:
                same_captures = False
                break
    normalized = lambda request: [{**r, "request": 0} for r in writes if r["request"] == request]
    expected = [(0, r["offset"]) for r in batches if r["request"] == 0 and (not suffix_only or r["offset"] >= boundary)]
    write_check = _compare_writes(normalized(0), normalized(2), expected, suffix_only, boundary)
    return {"passed": bool(same_prompt and logits["passed"] and same_state and same_captures and write_check["passed"]),
            "same_native_prompt": same_prompt, "full_logits": logits,
            "serialized_sequence_identical": same_state, "captured_tensors_identical": same_captures,
            "observed_state_writes": write_check,
            "scope": "request0 vs2 replays the same native agent1 input; no private-history or functional EOS qualification"}


def _read(root, guard):
    receipt = _guard(guard)
    report = json.loads(safe_file(root, "report.json").read_text())
    boundary = report_boundary(report)
    if "prefix_boundary" in report:
        manifest = receipt["preflight"]["snapshot_manifest"]
        key, _ = canonical_key(manifest)
        if (key != report.get("snapshot_key_sha256") or
                key != receipt["preflight"].get("snapshot_key_sha256") or
                len(manifest["prefix_tokens"]) != boundary or
                manifest["render_options"].get("prefix_boundary") != boundary or
                report.get("state_parameters") != manifest["state_parameters"]):
            raise ValueError("declared boundary is not bound to native dependencies")
    if report.get("complete") is not True or report.get("diagnostic") is not True:
        raise ValueError("complete diagnostic capture required")
    if report.get("model_loads") != 1 or report.get("physical_slots") != 1:
        raise ValueError("one model and one physical slot required")
    if report.get("gpu_layers") != 0 or report.get("threads") != 10:
        raise ValueError("this comparator requires the CPU10/GPU0 snapshot path")
    if (report.get("generation_performed") is not True or report.get("functional_agents_qualified") is not False or
            report.get("continuation_policy") != "fresh native greedy sampler; max4; diagnostic only"):
        raise ValueError("explicit fresh greedy diagnostic continuation is required, without functional qualification")
    requests = report.get("requests", [])
    if len(requests) != 3 or [r.get("agent_id") for r in requests] != [1, 600, 1]:
        raise ValueError("complete agent 1,600,1 prefill sequence required")
    n_vocab = report.get("vocab_size")
    if type(n_vocab) is not int or n_vocab <= 0:
        raise ValueError("native vocabulary size required for full-logit coverage")
    if report.get("mode") not in ("full", "split", "restore"):
        raise ValueError("unknown diagnostic arm")
    if report["mode"] == "restore" and (
            report.get("snapshot", {}).get("successful_restores") != 3 or
            not 0 < report.get("snapshot", {}).get("bytes", 0) <= 256 * MiB):
        raise ValueError("three complete bounded restores required")
    for index, row in enumerate(requests):
        if row.get("request") != index or row.get("next_position") != row.get("prompt_tokens"):
            raise ValueError("request order or final position mismatch")
        if not isinstance(row.get("prompt_ids_sha256"), str) or len(row["prompt_ids_sha256"]) != 64:
            raise ValueError("complete native prompt ID identity required")
        _payload(root, f"logits-{index}.f32", n_vocab * 4, row.get("logits_sha256"))
        _payload(root, f"final-state-{index}.bin", row.get("sequence_bytes", 0), row.get("sequence_sha256"))
    tensors = {}
    occurrences = {}
    for row in json_lines(safe_file(root, "tensors.jsonl")):
        label = family(row["name"])
        if label not in FAMILIES:
            raise ValueError("unexpected tensor family")
        if row.get("type") != (26 if label == "routes" else 0):
            raise ValueError("tensor family dtype mismatch")
        if type(row.get("request")) is not int or row["request"] not in (0, 1, 2):
            raise ValueError("invalid tensor request")
        if type(row.get("offset")) is not int or row["offset"] < 0:
            raise ValueError("invalid tensor token offset")
        prefix = row["request"], row["offset"], row["name"]
        occurrence = occurrences.get(prefix, 0)
        occurrences[prefix] = occurrence + 1
        tensors[(*prefix, occurrence)] = row
        _payload(root, row["file"], row["bytes"], row.get("sha256"))
    batches = [r for r in json_lines(safe_file(root, "batches.jsonl")) if "offset" in r]
    if not batches or any(r.get("returncode") != 0 for r in batches):
        raise ValueError("successful native microbatch receipts required")
    for request, row in enumerate(requests):
        selected = [r for r in batches if r.get("request") == request]
        start = boundary if report["mode"] == "restore" and request else 0
        for batch in selected:
            tokens = batch.get("tokens")
            if (batch.get("offset") != start or type(tokens) is not int or
                    not 0 < tokens <= report["microbatch"]):
                raise ValueError("incomplete or invalid microbatch token coverage")
            start += tokens
        if start != row["prompt_tokens"]:
            raise ValueError("microbatch receipts do not cover the complete prompt")
    return report, tensors, batches, receipt, _read_writes(root), _read_continuation(root, report)


def compare(reference_root, candidate_root, reference_guard, candidate_guard):
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    result = {"schema_version": 1, "passed": False, "scope":
              "finite CPU prefill continuation: full logits, serialized sequence bytes, captured ordered routes/router/KDA and observed cache writes; no EOS or unobserved-state qualification",
              "criteria": CRITERIA["cpu"], "errors": [], "logits": [],
              "sequence_states": [], "coverage": {k: 0 for k in FAMILIES}}
    try:
        a, at, ab, _, aw, a_cont = _read(reference_root, reference_guard)
        b, bt, bb, _, bw, b_cont = _read(candidate_root, candidate_guard)
        boundary = report_boundary(a)
        if boundary != report_boundary(b) or a.get("boundary_policy") != b.get("boundary_policy"):
            raise ValueError("prefix boundary policies differ")
        result["continuation"] = _compare_continuation(reference_root, candidate_root, a, b, a_cont, b_cont)
        result["continuation_replays"] = {
            "reference": _compare_continuation(reference_root, reference_root, a, a, a_cont, a_cont, ((0, 2),)),
            "candidate": _compare_continuation(candidate_root, candidate_root, b, b, b_cont, b_cont, ((0, 2),)),
        }
        result["same_agent_replays"] = {
            "reference": _check_replay(reference_root, a, at, ab, aw),
            "candidate": _check_replay(candidate_root, b, bt, bb, bw),
        }
        for name in ("microbatch", "logical_batch", "threads", "gpu_layers", "physical_slots", "vocab_size", "snapshot_key_sha256"):
            if name not in a or name not in b or a[name] != b[name]:
                raise ValueError("incompatible dependency/profile: " + name)
        pair = a["mode"], b["mode"]
        if pair not in (("full", "split"), ("split", "restore"), ("full", "full"), ("full", "restore")):
            raise ValueError("unsupported complete prefill/restore comparison")
        if pair == ("full", "restore"):
            result["direct_restore_boundary"] = _direct_restore(reference_root, a, b)
        suffix_only = b["mode"] == "restore"
        def selected_batches(rows):
            return [(r["request"], r["offset"], r["tokens"]) for r in rows
                    if not suffix_only or r["offset"] >= boundary]
        if selected_batches(ab) != selected_batches(bb):
            raise ValueError("microbatch boundaries differ; forms are not comparable")
        result["observed_state_writes"] = _compare_writes(
            aw, bw, [(r, offset) for r, offset, _ in selected_batches(ab)], suffix_only, boundary)
        for index, (ar, br) in enumerate(zip(a["requests"], b["requests"])):
            for name in ("prompt_tokens", "prompt_ids_sha256", "next_position"):
                if ar[name] != br[name]:
                    raise ValueError("different native agent prompt or positions")
            av = np.memmap(reference_root / f"logits-{index}.f32", dtype="<f4", mode="r")
            bv = np.memmap(candidate_root / f"logits-{index}.f32", dtype="<f4", mode="r")
            result["logits"].append({"request": index, **compare_arrays(av, bv, CRITERIA["cpu"])})
            result["sequence_states"].append({"request": index,
                "passed": ar["sequence_bytes"] == br["sequence_bytes"] and ar["sequence_sha256"] == br["sequence_sha256"],
                "reference_bytes": ar["sequence_bytes"], "candidate_bytes": br["sequence_bytes"]})
        selected = lambda rows: {k: v for k, v in rows.items() if not suffix_only or k[1] >= boundary}
        at, bt = selected(at), selected(bt)
        if not at or set(at) != set(bt):
            raise ValueError("missing tensors or capture key sets differ")
        checks = []
        per_batch = {}
        per_batch_names = {}
        for key, ar in at.items():
            br = bt[key]
            label = family(ar["name"])
            check = compare_arrays(tensor_view(reference_root, ar), tensor_view(candidate_root, br),
                                   CRITERIA["cpu"], integer=label == "routes")
            checks.append(check["passed"])
            result["coverage"][label] += check["elements"]
            per_batch.setdefault(key[:2], set()).add(label)
            per_batch_names.setdefault(key[:2], set()).add(key[2])
            if not check["passed"] and "first_tensor_divergence" not in result:
                result["first_tensor_divergence"] = {"request": key[0], "offset": key[1], "name": key[2], **check}
        expected_batches = set((r, offset) for r, offset, _ in selected_batches(ab))
        if set(per_batch) != expected_batches or any(v != set(FAMILIES) for v in per_batch.values()):
            raise ValueError("routes/router/KDA coverage missing at a compared microbatch")
        if any(not EXPECTED_NAMES <= names for names in per_batch_names.values()):
            raise ValueError("one or more original GLM layer captures are missing")
        result["passed"] = bool(result["continuation"]["passed"] and all(r["passed"] for r in result["continuation_replays"].values()) and
                                all(r["passed"] for r in result["same_agent_replays"].values()) and
                                result["observed_state_writes"]["passed"] and all(checks) and all(r["passed"] for r in result["logits"]) and
                                all(r["passed"] for r in result["sequence_states"]))
        result["compared_tensor_captures"] = len(checks)
        result["comparison"] = "split_restore" if suffix_only else "full_split"
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["errors"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-guard", type=Path, required=True)
    parser.add_argument("--candidate-guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.reference_guard, args.candidate_guard)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
