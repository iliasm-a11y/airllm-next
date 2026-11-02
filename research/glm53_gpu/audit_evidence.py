#!/usr/bin/env python3
"""Reanalyse PR17 evidence; does not run inference or infer physical SSD traffic."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def layer_order(row):
    match = re.search(r"-(\d+)", row["name"])
    return row["step"], int(match[1]) if match else -1, row.get("occurrence", 0)


def compact(row):
    keys = ("step", "name", "occurrence", "elements", "mismatch_count",
            "bit_mismatch_count", "max_abs_error_finite_pairs")
    return {key: row[key] for key in keys if key in row}


def first(rows, predicate):
    return next((compact(row) for row in sorted(rows, key=layer_order) if predicate(row)), None)


def audit(published, captures=None):
    comparison_path = published / "comparisons/fidelity-hybrid6-comparison.json.gz"
    summary_path = published / "summary-finished.json.gz"
    comparison, summary = read_json(comparison_path), read_json(summary_path)
    result = {"schema_version": 1, "scope": "Historical PR17 evidence reanalysis, no new model execution",
              "source_sha256": {str(path.relative_to(published)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (comparison_path, summary_path)},
              "fidelity": {key: comparison[key] for key in
                           ("passed", "criteria", "reference_run", "candidate_run",
                            "prompt_tokens_identical", "complete_same_input_sequence",
                            "tokens_exact_on_comparable_steps", "full_logits_summary")}}
    result["fidelity"]["first_observed"] = {}
    for family in ("kda", "router", "routes"):
        rows = [row for row in comparison["tensors"] if row["family"] == family
                and row.get("comparable") and "reshaped" not in row["name"]]
        result["fidelity"]["first_observed"][family] = {
            "bit_difference": first(rows, lambda row: row.get("bit_mismatch_count", 0) > 0),
            "criterion_failure": first(rows, lambda row: row.get("mismatch_count", 0) > 0)}
    result["fidelity"]["manifest_errors"] = comparison["errors"]
    profile = summary["instrumented_profile"]
    result["instrumented_cpu_profile"] = {"scope": profile["scope"], "groups": {}}
    for name, group in profile["groups"].items():
        result["instrumented_cpu_profile"]["groups"][name] = {
            "totals": group["totals"], "top_operations": group["by_op"][:4]}
    result["timings"] = []
    for variant in summary["variants"]:
        if variant["file"] not in {"final-cpu10-16x2.json", "long-final-cpu10-32x2.json",
                                   "hybrid6-combined-pilot.json", "metal-head-cpu10-pilot.json"}:
            continue
        for request in variant["requests"]:
            result["timings"].append({"file": variant["file"], "threads": variant["configuration"]["threads"],
                "gpu_layers": variant["configuration"]["gpu_layers"], "prompt_tokens": request["prompt_tokens"],
                "emitted_tokens": request["emitted_token_count"], "request_wall_seconds": request["wall_seconds"],
                "ttft_seconds": request["ttft_seconds"],
                "observed_decode_tok_s": request["decode"]["observed_tokens_per_second_recomputed"],
                "request_cpu_seconds": request["request_usage_delta"]["cpu_seconds"],
                "request_os_read_bytes": request["request_usage_delta"]["storage_read_bytes"],
                "physical_nand_bytes": None, "cache_state": variant["cache_scope"]})
    if captures is not None:
        result["raw_route_reanalysis"] = route_audit(captures)
    return result


def route_audit(captures):
    import numpy as np
    comparator_path = Path(__file__).resolve().parents[1] / "glm53_m5/compare_fidelity.py"
    spec = importlib.util.spec_from_file_location("pr17_comparator", comparator_path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    reference, candidate = captures / "stock-cpu4", captures / "hybrid6"
    a, b = comparator.read_capture(reference), comparator.read_capture(candidate)
    if a["prompt"] != b["prompt"] or [x["next_input_token"] for x in a["tokens"]] != [x["next_input_token"] for x in b["tokens"]]:
        raise ValueError("Raw route comparison requires identical forced input trajectory")
    counts = {"vectors": 0, "ordered_different": 0, "membership_different": 0,
              "order_only": 0, "different_slots": 0}
    first_order = first_membership = None
    rows = sorted((row for row in a["tensors"].values() if row["family"] == "routes"
                   and "reshaped" not in row["name"]), key=layer_order)
    for row in rows:
        key = (row["step"], row["name"], 0)
        if key not in b["tensors"]:
            raise ValueError("Missing canonical route tensor: " + repr(key))
        x = comparator.tensor_view(reference, row).reshape(8, -1)
        y = comparator.tensor_view(candidate, b["tensors"][key]).reshape(8, -1)
        if x.shape != y.shape:
            raise ValueError("Canonical route shape differs")
        for token in range(x.shape[1]):
            counts["vectors"] += 1
            mismatches = int(np.count_nonzero(x[:, token] != y[:, token]))
            counts["different_slots"] += mismatches
            if not mismatches:
                continue
            changed = set(x[:, token]) != set(y[:, token])
            counts["ordered_different"] += 1
            counts["membership_different"] += int(changed)
            counts["order_only"] += int(not changed)
            detail = {"step": row["step"], "name": row["name"], "token_column": token,
                      "reference": x[:, token].tolist(), "candidate": y[:, token].tolist()}
            if first_order is None:
                first_order = detail
            if changed and first_membership is None:
                first_membership = detail
    return {**counts, "first_order_difference": first_order,
            "first_membership_difference": first_membership,
            "scope": "Canonical top8 tensors only, aliases omitted; step0 contains six prompt positions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=Path(__file__).resolve().parents[1] / "glm53_m5/evidence/published")
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(audit(args.published, args.captures), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
