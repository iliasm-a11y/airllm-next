"""Strict CPU prefill fidelity across physical microbatch sizes.

Token axes come from the pinned GLM graph: MoE route/router axis1 and
pre-reshape KDA output axis2 (head_dim, heads, tokens, sequence). Intermediate
write-event counts are not compared across different chunkings. Full sequence
hashes are compared at common token boundaries and at the final prompt state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from research.glm53_m5.compare_fidelity import CRITERIA, compare_arrays, family, json_lines, safe_file, tensor_view
from research.glm53_batched_prefill.native_compare import (
    EXPECTED_NAMES, MiB, _read, _check_replay, _compare_writes,
)
from research.glm53_batched_prefill.snapshot_key import canonical_key

TOKEN_AXES = {"routes": 1, "router": 1, "kda": 2}
CHECKPOINTS = (128, 256, 384, 512)
BATCHES = (8, 16, 32, 64, 128)


def _dependencies(report, receipt):
    manifest = receipt["preflight"]["snapshot_manifest"]
    key, _ = canonical_key(manifest)
    if key != report.get("snapshot_key_sha256") or key != receipt["preflight"].get("snapshot_key_sha256"):
        raise ValueError("native key does not match its complete dependency manifest")
    if report.get("state_parameters") != manifest["state_parameters"]:
        raise ValueError("native effective state parameters differ from dependency manifest")
    if report["microbatch"] not in BATCHES or report["logical_batch"] < report["microbatch"]:
        raise ValueError("unsupported microbatch profile")
    for name, expected in (("n_batch", report["logical_batch"]), ("n_ubatch", report["microbatch"])):
        if manifest["state_parameters"][name] != expected:
            raise ValueError("reported batch differs from effective native parameters")
    normalized = json.loads(json.dumps(manifest, allow_nan=False))
    for name in ("n_batch", "n_ubatch"):
        del normalized["state_parameters"][name]
    return normalized


def _checkpoints(root, report):
    path = safe_file(root, "state_checkpoints.jsonl")
    if path.stat().st_size > MiB:
        raise ValueError("sequence-checkpoint metadata exceeds1MiB")
    rows = json_lines(path)
    result = {}
    for row in rows:
        request, end = row.get("request"), row.get("token_end")
        if (type(request) is not int or request not in (0, 1, 2) or
                type(end) is not int or end <= 0 or row.get("next_position") != end or
                row.get("status") != "hashed" or row.get("state_seq_flags") != 0 or
                type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 256 * MiB or
                not isinstance(row.get("sha256"), str) or
                re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("invalid complete-sequence checkpoint record")
        key = request, end
        if key in result:
            raise ValueError("duplicate sequence checkpoint")
        result[key] = row
    expected = {(r, end) for r, request in enumerate(report["requests"])
                for end in {*[n for n in CHECKPOINTS if n <= request["prompt_tokens"]], request["prompt_tokens"]}}
    if set(result) != expected:
        raise ValueError("complete-sequence checkpoints missing or extra")
    for request, row in enumerate(report["requests"]):
        final = result[request, row["prompt_tokens"]]
        if final["bytes"] != row["sequence_bytes"] or final["sha256"] != row["sequence_sha256"]:
            raise ValueError("final checkpoint differs from verified raw final-state serialization")
    return result


def _checkpoint_replay(checkpoints):
    a = {end: row for (request, end), row in checkpoints.items() if request == 0}
    b = {end: row for (request, end), row in checkpoints.items() if request == 2}
    return bool(a) and set(a) == set(b) and all(
        a[end]["bytes"] == b[end]["bytes"] and a[end]["sha256"] == b[end]["sha256"] for end in a)


def _series(report, tensors, batches):
    receipts = {(row["request"], row["offset"]): dict(row) for row in batches}
    if len(receipts) != len(batches):
        raise ValueError("duplicate microbatch receipts")
    by_request = {}
    for request in range(3):
        selected = [r for r in batches if r["request"] == request]
        for index, row in enumerate(selected):
            receipts[request, row["offset"]]["batch_index"] = index
    result = {}
    for (request, offset, name, occurrence), row in tensors.items():
        if name not in EXPECTED_NAMES:
            continue  # Only the graph's named pre-reshape KDA output has axis2.
        if occurrence != 0:
            raise ValueError("duplicate named tensor at a physical microbatch")
        receipt = receipts.get((request, offset))
        if receipt is None or row.get("input_tokens") != receipt["tokens"] or row.get("batch_index") != receipt["batch_index"]:
            raise ValueError("tensor does not identify its native physical microbatch")
        axis = TOKEN_AXES[family(name)]
        shape = row["shape"]
        if len(shape) != 4 or shape[axis] != receipt["tokens"] or shape[3] != 1:
            raise ValueError("tensor token axis or physical-sequence shape differs from the pinned GLM contract")
        result.setdefault((request, name), []).append(row)
    expected = {(request, name) for request in range(3) for name in EXPECTED_NAMES}
    if set(result) != expected:
        raise ValueError("missing original GLM route/router/KDA layer coverage")
    for (request, name), rows in result.items():
        rows.sort(key=lambda row: row["offset"])
        ordered = [(r["offset"], r["input_tokens"]) for r in rows]
        wanted = [(r["offset"], r["tokens"]) for r in batches if r["request"] == request]
        if ordered != wanted or sum(r["input_tokens"] for r in rows) != report["requests"][request]["prompt_tokens"]:
            raise ValueError("logical token coverage has a gap, overlap or omitted microbatch")
        by_request[request, name] = rows
    return by_request


def _token_compare(a_root, b_root, a_rows, b_rows, name):
    axis = TOKEN_AXES[family(name)]
    a_index = b_index = a_consumed = b_consumed = 0
    result = {"passed": True, "elements": 0, "bit_mismatches": 0,
              "nonfinite_reference": 0, "nonfinite_candidate": 0, "tokens": 0}
    while a_index < len(a_rows) and b_index < len(b_rows):
        ar, br = a_rows[a_index], b_rows[b_index]
        token = ar["offset"] + a_consumed
        if token != br["offset"] + b_consumed:
            raise ValueError("logical token positions differ during stitching")
        count = min(ar["input_tokens"] - a_consumed, br["input_tokens"] - b_consumed)
        av, bv = tensor_view(a_root, ar), tensor_view(b_root, br)
        if any(av.shape[i] != bv.shape[i] for i in range(4) if i != axis):
            raise ValueError("non-token tensor dimensions differ between microbatches")
        a_slice, b_slice = [slice(None)] * 4, [slice(None)] * 4
        a_slice[axis], b_slice[axis] = slice(a_consumed, a_consumed + count), slice(b_consumed, b_consumed + count)
        av, bv = av[tuple(a_slice)], bv[tuple(b_slice)]
        check = compare_arrays(av, bv, CRITERIA["cpu"], integer=family(name) == "routes")
        result["passed"] &= check["passed"]
        result["elements"] += check["elements"]
        result["bit_mismatches"] += check["bit_mismatch_count"]
        result["nonfinite_reference"] += check["nonfinite_reference"]
        result["nonfinite_candidate"] += check["nonfinite_candidate"]
        result["tokens"] += count
        if not check["passed"] and "first_token_divergence" not in result:
            mask = (av.view(np.uint32) != bv.view(np.uint32)) | ~np.isfinite(av) | ~np.isfinite(bv)
            token_mask = np.any(mask, axis=tuple(i for i in range(4) if i != axis))
            result["first_token_divergence"] = token + int(np.flatnonzero(token_mask)[0])
        a_consumed += count
        b_consumed += count
        if a_consumed == ar["input_tokens"]:
            a_index, a_consumed = a_index + 1, 0
        if b_consumed == br["input_tokens"]:
            b_index, b_consumed = b_index + 1, 0
    if a_index != len(a_rows) or b_index != len(b_rows):
        raise ValueError("one token series ended before the other")
    return result


def compare(reference_root, candidate_root, reference_guard, candidate_guard):
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    result = {"schema_version": 1, "passed": False, "errors": [], "criteria": CRITERIA["cpu"],
              "scope": "CPU10/GPU0 full prefill across batch sizes; per-token ordered routes/router/KDA, finite full final logits, raw final sequence states and complete-state hashes at common boundaries; no EOS or private-history qualification",
              "cross_batch_write_event_parity": "not asserted: physical chunks differ; require per-batch target coverage and complete states at common token boundaries",
              "logits": [], "final_states": [], "checkpoints": [], "tensors": [], "coverage": {k: 0 for k in TOKEN_AXES}}
    try:
        a, at, ab, ag, aw = _read(reference_root, reference_guard)
        b, bt, bb, bg, bw = _read(candidate_root, candidate_guard)
        if a["mode"] != "full" or b["mode"] != "full":
            raise ValueError("cross-batch comparison requires full/full uncached prefill")
        if _dependencies(a, ag) != _dependencies(b, bg):
            raise ValueError("model/runtime/state dependencies differ beyond n_batch/n_ubatch")
        for field in ("vocab_size", "threads", "gpu_layers", "physical_slots"):
            if a[field] != b[field]:
                raise ValueError("incompatible native profile: " + field)
        ac, bc = _checkpoints(reference_root, a), _checkpoints(candidate_root, b)
        result["same_agent_replays"] = {
            "reference": _check_replay(reference_root, a, at, ab, aw),
            "candidate": _check_replay(candidate_root, b, bt, bb, bw),
        }
        result["checkpoint_replays"] = {"reference": _checkpoint_replay(ac), "candidate": _checkpoint_replay(bc)}
        # Self-comparison validates complete target coverage, not cross-batch parity.
        result["observed_write_coverage"] = {
            "reference": _compare_writes(aw, aw, [(r["request"], r["offset"]) for r in ab], False),
            "candidate": _compare_writes(bw, bw, [(r["request"], r["offset"]) for r in bb], False),
        }
        if set(ac) != set(bc):
            raise ValueError("common complete-state checkpoint positions differ")
        for key in sorted(ac):
            result["checkpoints"].append({"request": key[0], "token_end": key[1],
                "passed": ac[key]["bytes"] == bc[key]["bytes"] and ac[key]["sha256"] == bc[key]["sha256"]})
        for request, (ar, br) in enumerate(zip(a["requests"], b["requests"])):
            for field in ("prompt_tokens", "prompt_ids_sha256", "next_position"):
                if ar[field] != br[field]:
                    raise ValueError("full native prompts or positions differ")
            result["logits"].append({"request": request, **compare_arrays(
                np.memmap(reference_root / f"logits-{request}.f32", dtype="<f4", mode="r"),
                np.memmap(candidate_root / f"logits-{request}.f32", dtype="<f4", mode="r"), CRITERIA["cpu"])})
            result["final_states"].append({"request": request,
                "passed": ar["sequence_bytes"] == br["sequence_bytes"] and ar["sequence_sha256"] == br["sequence_sha256"]})
        a_series, b_series = _series(a, at, ab), _series(b, bt, bb)
        for request, name in sorted(a_series):
            check = _token_compare(reference_root, candidate_root, a_series[request, name], b_series[request, name], name)
            result["tensors"].append({"request": request, "name": name, **check})
            result["coverage"][family(name)] += check["elements"]
        result["passed"] = bool(all(r["passed"] for r in result["same_agent_replays"].values()) and
            all(result["checkpoint_replays"].values()) and all(r["passed"] for r in result["observed_write_coverage"].values()) and
            all(r["passed"] for r in result["logits"] + result["final_states"] + result["checkpoints"] + result["tensors"]))
        result["reference_microbatch"], result["candidate_microbatch"] = a["microbatch"], b["microbatch"]
    except (OSError, ValueError, TypeError, KeyError) as error:
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
