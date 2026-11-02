#!/usr/bin/env python3
"""Compare native_fidelity captures with criteria fixed before model results.

Only logical tensor elements are compared. GGML byte strides define each view;
padding bytes and file serials are not semantic tensor values.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

CRITERIA = {
    "io": {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0},
    "cpu": {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0},
    "metal": {"bitexact_f32": False, "atol": 1e-3, "rtol": 1e-4},
}
DTYPES = {0: np.dtype("<f4"), 26: np.dtype("<i4")}
FAMILIES = ("routes", "router", "kda", "result_output")


def family(name):
    for prefix, label in (("ffn_moe_topk-", "routes"), ("ffn_moe_logits-", "router"),
                          ("kda_scan_out-", "kda")):
        if name.startswith(prefix):
            return label
    if name == "result_output":
        return name
    raise ValueError("Unexpected diagnostic tensor name: " + name)


def json_lines(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 2_000_000:
                raise ValueError("Oversize manifest row")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or escaping capture file: " + str(name))
    return path


def tensor_view(root, row):
    dtype = DTYPES.get(row["type"])
    if dtype is None:
        raise ValueError("Unsupported GGML type: " + str(row["type"]))
    shape, strides = row["shape"], row["strides"]
    if len(shape) != 4 or len(strides) != 4:
        raise ValueError("Expected four GGML dimensions and strides")
    if any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("Invalid tensor shape")
    if any(type(x) is not int or x < 0 or x % dtype.itemsize for x in strides):
        raise ValueError("Invalid tensor byte stride")
    needed = dtype.itemsize + sum((n - 1) * stride for n, stride in zip(shape, strides))
    path = safe_file(root, row["file"])
    if type(row["bytes"]) is not int or path.stat().st_size != row["bytes"] or needed > row["bytes"]:
        raise ValueError("Tensor payload length or stride span mismatch")
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    return np.ndarray(tuple(shape), dtype=dtype, buffer=raw, strides=tuple(strides))


def compare_arrays(reference, candidate, criteria, *, integer=False):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Tensor shape or dtype mismatch")
    count = reference.size
    numeric_mismatches = bit_mismatches = tolerance_mismatches = 0
    nonfinite_reference = nonfinite_candidate = finite_pairs = 0
    sum_squared = maximum = 0.0
    it = np.nditer([reference, candidate], flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=[["readonly"], ["readonly"]], order="C", buffersize=262144)
    for x, y in it:
        numeric_mismatches += int(np.count_nonzero(x != y))
        bit_mismatches += int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
        finite_x, finite_y = np.isfinite(x), np.isfinite(y)
        nonfinite_reference += int(x.size - np.count_nonzero(finite_x))
        nonfinite_candidate += int(y.size - np.count_nonzero(finite_y))
        both = finite_x & finite_y
        finite_pairs += int(np.count_nonzero(both))
        difference = y[both].astype(np.float64) - x[both].astype(np.float64)
        if difference.size:
            maximum = max(maximum, float(np.max(np.abs(difference))))
            sum_squared += float(np.dot(difference, difference))
        tolerance_mismatches += int(np.count_nonzero(~np.isclose(
            y, x, rtol=criteria["rtol"], atol=criteria["atol"], equal_nan=False)))
    mismatches = bit_mismatches if integer or criteria["bitexact_f32"] else tolerance_mismatches
    finite = nonfinite_reference == nonfinite_candidate == 0
    return {"passed": finite and mismatches == 0, "elements": count,
            "criterion": "exact_i32" if integer else "bitexact_f32" if criteria["bitexact_f32"] else "allclose_f32",
            "mismatch_count": mismatches, "bit_mismatch_count": bit_mismatches,
            "numeric_mismatch_count": numeric_mismatches,
            "tolerance_mismatch_count": tolerance_mismatches,
            "max_abs_error_finite_pairs": maximum if finite_pairs else None,
            "rmse_finite_pairs": math.sqrt(sum_squared / finite_pairs) if finite_pairs else None,
            "finite": finite, "finite_pairs": finite_pairs,
            "nonfinite_reference": nonfinite_reference, "nonfinite_candidate": nonfinite_candidate}


def read_capture(root):
    run = json.loads(safe_file(root, "run.json").read_text())
    complete = json.loads(safe_file(root, "complete.json").read_text())
    prompt = json.loads(safe_file(root, "prompt_tokens.json").read_text())
    if not isinstance(prompt, list) or not prompt or any(type(x) is not int or x < 0 for x in prompt):
        raise ValueError("Invalid prompt token array")
    tokens = json_lines(safe_file(root, "tokens.jsonl"))
    if not tokens or len(tokens) > run["count"]:
        raise ValueError("Capture has no tokens or exceeds its requested count")
    if complete.get("completed") is not True or complete.get("captured_steps") != len(tokens):
        raise ValueError("Capture has no valid successful-completion marker")
    for step, row in enumerate(tokens):
        if row.get("step") != step:
            raise ValueError("Token steps must be contiguous from zero")
        if type(row.get("vocab")) is not int or row["vocab"] < 1:
            raise ValueError("Invalid vocabulary size")
        for key in ("token", "next_input_token"):
            if type(row.get(key)) is not int or not 0 <= row[key] < row["vocab"]:
                raise ValueError("Invalid token id: " + key)
    tensors, occurrences = {}, Counter()
    for row in json_lines(safe_file(root, "tensors.jsonl")):
        if type(row.get("step")) is not int or not 0 <= row["step"] < len(tokens):
            raise ValueError("Tensor step outside the captured token sequence")
        label = family(row["name"])
        if row.get("type") != (26 if label == "routes" else 0):
            raise ValueError("Diagnostic tensor has the wrong GGML type for its family")
        prefix = (row["step"], row["name"])
        occurrence = occurrences[prefix]
        occurrences[prefix] += 1
        tensors[(*prefix, occurrence)] = {**row, "family": label}
    return {"run": run, "prompt": prompt, "tokens": tokens, "tensors": tensors}


def aggregate(rows):
    selected = [row for row in rows if row.get("comparable") and "elements" in row]
    finite_pairs = sum(row["finite_pairs"] for row in selected)
    return {"compared": len(selected), "elements": sum(row["elements"] for row in selected),
            "mismatch_count": sum(row["mismatch_count"] for row in selected),
            "bit_mismatch_count": sum(row["bit_mismatch_count"] for row in selected),
            "finite": bool(selected) and all(row["finite"] for row in selected),
            "max_abs_error_finite_pairs": max((row["max_abs_error_finite_pairs"] for row in selected if row["max_abs_error_finite_pairs"] is not None), default=None),
            "rmse_finite_pairs": math.sqrt(sum((row["rmse_finite_pairs"] or 0)**2 * row["finite_pairs"] for row in selected) / finite_pairs) if finite_pairs else None,
            "finite_pairs": finite_pairs, "passed": bool(selected) and all(row["passed"] for row in selected)}


def compare(reference_root, candidate_root, mode):
    criteria = CRITERIA[mode]
    reference, candidate = read_capture(reference_root), read_capture(candidate_root)
    a_tokens, b_tokens = reference["tokens"], candidate["tokens"]
    steps = min(len(a_tokens), len(b_tokens))
    prefix_equal = reference["prompt"] == candidate["prompt"]
    first_prefix_divergence = None if prefix_equal else 0
    for step in range(steps - 1):
        if first_prefix_divergence is None and a_tokens[step]["next_input_token"] != b_tokens[step]["next_input_token"]:
            first_prefix_divergence = step + 1
    comparable = lambda step: first_prefix_divergence is None or step < first_prefix_divergence
    token_rows, logits_rows, tensor_rows, errors = [], [], [], []
    for step in range(steps):
        a, b = a_tokens[step], b_tokens[step]
        matched = a["token"] == b["token"] and a["vocab"] == b["vocab"]
        token_rows.append({"step": step, "reference": a["token"], "candidate": b["token"],
                           "comparable": comparable(step), "equal": matched})
        if not comparable(step):
            logits_rows.append({"step": step, "comparable": False})
            continue
        try:
            if a["vocab"] != b["vocab"]:
                raise ValueError("Vocabulary sizes differ")
            views = []
            for root, row in ((reference_root, a), (candidate_root, b)):
                path = safe_file(root, "logits-" + str(step) + ".f32")
                if path.stat().st_size != row["vocab"] * 4:
                    raise ValueError("Full logit vector size differs from vocabulary")
                view = np.memmap(path, dtype="<f4", mode="r")
                if np.all(np.isfinite(view)) and int(np.argmax(view)) != row["token"]:
                    raise ValueError("Recorded greedy token disagrees with the full logit vector")
                views.append(view)
            logits_rows.append({"step": step, "comparable": True, **compare_arrays(*views, criteria)})
        except (OSError, ValueError, KeyError) as exc:
            errors.append("logits step " + str(step) + ": " + str(exc))
    a_tensors, b_tensors = reference["tensors"], candidate["tensors"]
    for key in sorted(set(a_tensors) | set(b_tensors)):
        step, name, occurrence = key
        info = {"step": step, "name": name, "occurrence": occurrence, "family": family(name),
                "comparable": comparable(step) and step < steps}
        if not info["comparable"]:
            tensor_rows.append(info)
            continue
        try:
            if key not in a_tensors or key not in b_tensors:
                raise ValueError("Tensor missing on one side")
            a, b = a_tensors[key], b_tensors[key]
            views = tensor_view(reference_root, a), tensor_view(candidate_root, b)
            tensor_rows.append({**info, "strides_equal": a["strides"] == b["strides"],
                                **compare_arrays(*views, criteria, integer=a["type"] == 26)})
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(key) + ": " + str(exc))
    coverage = {}
    for label in FAMILIES:
        selected = [r for r in tensor_rows if r["family"] == label and r["comparable"]]
        coverage[label] = aggregate(selected)
    complete = len(a_tokens) == len(b_tokens) and first_prefix_divergence is None
    tensor_sets_equal = set(a_tensors) == set(b_tensors)
    token_match = all(r["equal"] for r in token_rows if r["comparable"])
    passed = bool(complete and tensor_sets_equal and token_match and not errors
                  and all(r.get("passed", False) for r in logits_rows)
                  and all(c["passed"] for c in coverage.values()))
    return {"schema_version": 1, "passed": passed, "mode": mode,
            "criteria": {**criteria, "ordered_routes_exact": True, "tokens_exact": True, "all_values_finite": True},
            "reference": str(reference_root.resolve()), "candidate": str(candidate_root.resolve()),
            "reference_run": reference["run"], "candidate_run": candidate["run"],
            "prompt_tokens_identical": prefix_equal, "reference_steps": len(a_tokens),
            "candidate_steps": len(b_tokens), "first_incomparable_step": first_prefix_divergence,
            "complete_same_input_sequence": complete, "tokens_exact_on_comparable_steps": token_match,
            "tensor_manifest_keys_equal": tensor_sets_equal, "coverage": coverage,
            "full_logits_summary": aggregate(logits_rows),
            "tokens": token_rows, "full_logits": logits_rows, "tensors": tensor_rows, "errors": errors,
            "scope": "Diagnostic captured tensors and full logits only; does not qualify throughput or uncaptured operations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", choices=CRITERIA, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate, args.mode)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"schema_version": 1, "passed": False, "mode": args.mode,
                  "criteria": CRITERIA[args.mode], "errors": [str(exc)]}
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
