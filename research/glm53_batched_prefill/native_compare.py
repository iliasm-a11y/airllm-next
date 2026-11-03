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
    row = _record(receipt)
    if not isinstance(row, dict):
        raise ValueError("an explicit guard receipt is required")
    if row.get("status") != "completed" or row.get("returncode") != 0 or row.get("guard_violation") is not None:
        raise ValueError("native run did not exit successfully under clear guards")
    if row.get("power_context_passed") is not True or row.get("integrity_passed") is not True:
        raise ValueError("AC/lid/continuity and integrity receipts are required")
    limits = row.get("limits", {})
    rss, swap = limits.get("rss_bytes"), limits.get("swap_growth_bytes")
    interval = row.get("sampling_interval_seconds")
    if not isinstance(rss, (int, float)) or not 0 < rss <= 8.5 * GiB:
        raise ValueError("RSS limit absent or relaxed")
    if not isinstance(swap, (int, float)) or not 0 <= swap <= 32 * MiB:
        raise ValueError("swap growth limit absent or relaxed")
    if not isinstance(interval, (int, float)) or not 0 < interval <= .5:
        raise ValueError("sampling interval absent or relaxed")
    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("no guard observations")
    for sample in samples:
        sampled_rss, sampled_swap = sample.get("rss_bytes"), sample.get("swap_growth_bytes")
        if (type(sampled_rss) not in (int, float) or not math.isfinite(sampled_rss) or
                not 0 <= sampled_rss <= rss or
                type(sampled_swap) not in (int, float) or not math.isfinite(sampled_swap) or
                not 0 <= sampled_swap <= swap or sample.get("pressure_level") != 1):
            raise ValueError("an observed resource guard failed")
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


def _read_writes(root):
    path = safe_file(root, "state_hashes.jsonl")
    if path.stat().st_size > 64 * MiB:
        raise ValueError("observed state-write metadata exceeds64MiB bound")
    rows = json_lines(path)
    if not rows or len(rows) > 262144:
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


def _compare_writes(a, b, expected_batches, suffix_only):
    selected = lambda rows: [r for r in rows if not suffix_only or r["offset"] >= 208]
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
    a, b = report["requests"][0], report["requests"][2]
    suffix_only = report["mode"] == "restore"
    same_prompt = all(a[name] == b[name] for name in ("prompt_tokens", "prompt_ids_sha256", "next_position"))
    logits = compare_arrays(np.memmap(root / "logits-0.f32", dtype="<f4", mode="r"),
                            np.memmap(root / "logits-2.f32", dtype="<f4", mode="r"), CRITERIA["cpu"])
    same_state = a["sequence_bytes"] == b["sequence_bytes"] and a["sequence_sha256"] == b["sequence_sha256"]
    selected = lambda request: {key[1:]: row for key, row in tensors.items()
                                if key[0] == request and (not suffix_only or key[1] >= 208)}
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
    expected = [(0, r["offset"]) for r in batches if r["request"] == 0 and (not suffix_only or r["offset"] >= 208)]
    write_check = _compare_writes(normalized(0), normalized(2), expected, suffix_only)
    return {"passed": bool(same_prompt and logits["passed"] and same_state and same_captures and write_check["passed"]),
            "same_native_prompt": same_prompt, "full_logits": logits,
            "serialized_sequence_identical": same_state, "captured_tensors_identical": same_captures,
            "observed_state_writes": write_check,
            "scope": "request0 vs2 replays the same native agent1 input; no private-history or functional EOS qualification"}


def _read(root, guard):
    receipt = _guard(guard)
    report = json.loads(safe_file(root, "report.json").read_text())
    if report.get("complete") is not True or report.get("diagnostic") is not True:
        raise ValueError("complete diagnostic capture required")
    if report.get("model_loads") != 1 or report.get("physical_slots") != 1:
        raise ValueError("one model and one physical slot required")
    if report.get("gpu_layers") != 0 or report.get("threads") != 10:
        raise ValueError("this comparator requires the CPU10/GPU0 snapshot path")
    if report.get("generation_performed") is not False:
        raise ValueError("this comparator covers prefill continuation only")
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
        start = 208 if report["mode"] == "restore" and request else 0
        for batch in selected:
            tokens = batch.get("tokens")
            if (batch.get("offset") != start or type(tokens) is not int or
                    not 0 < tokens <= report["microbatch"]):
                raise ValueError("incomplete or invalid microbatch token coverage")
            start += tokens
        if start != row["prompt_tokens"]:
            raise ValueError("microbatch receipts do not cover the complete prompt")
    return report, tensors, batches, receipt, _read_writes(root)


def compare(reference_root, candidate_root, reference_guard, candidate_guard):
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    result = {"schema_version": 1, "passed": False, "scope":
              "finite CPU prefill continuation: full logits, serialized sequence bytes, captured ordered routes/router/KDA and observed cache writes; no EOS or unobserved-state qualification",
              "criteria": CRITERIA["cpu"], "errors": [], "logits": [],
              "sequence_states": [], "coverage": {k: 0 for k in FAMILIES}}
    try:
        a, at, ab, _, aw = _read(reference_root, reference_guard)
        b, bt, bb, _, bw = _read(candidate_root, candidate_guard)
        result["same_agent_replays"] = {
            "reference": _check_replay(reference_root, a, at, ab, aw),
            "candidate": _check_replay(candidate_root, b, bt, bb, bw),
        }
        for name in ("microbatch", "logical_batch", "threads", "gpu_layers", "physical_slots", "vocab_size", "snapshot_key_sha256"):
            if name not in a or name not in b or a[name] != b[name]:
                raise ValueError("incompatible dependency/profile: " + name)
        if (a["mode"], b["mode"]) not in (("full", "split"), ("split", "restore"), ("full", "full")):
            raise ValueError("compare full/split before split/restore")
        suffix_only = b["mode"] == "restore"
        def selected_batches(rows):
            return [(r["request"], r["offset"], r["tokens"]) for r in rows
                    if not suffix_only or r["offset"] >= 208]
        if selected_batches(ab) != selected_batches(bb):
            raise ValueError("microbatch boundaries differ; forms are not comparable")
        result["observed_state_writes"] = _compare_writes(
            aw, bw, [(r, offset) for r, offset, _ in selected_batches(ab)], suffix_only)
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
        selected = lambda rows: {k: v for k, v in rows.items() if not suffix_only or k[1] >= 208}
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
        result["passed"] = bool(all(r["passed"] for r in result["same_agent_replays"].values()) and
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
