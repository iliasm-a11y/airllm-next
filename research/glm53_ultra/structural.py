"""Metadata-only ngram opportunity analysis for the pinned GLM research runtime.

This replays common_ngram_simple_draft at llama.cpp 3f8af15a76ac8675812e856da45454bafc90f12c.
Agreement with an existing sequential trajectory is not speculative acceptance,
batch fidelity, or a throughput measurement. No checkpoint payload is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNTIME_REVISION = "3f8af15a76ac8675812e856da45454bafc90f12c"
DEFAULT_CONFIGS = ((12, 48), (4, 4), (2, 2), (1, 1), (2, 1))


def _tokens(value):
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise ValueError("Expected a list of nonnegative integer token IDs")
    return list(value)


def simple_draft(history, sampled, *, size_n=12, size_m=48):
    """Match native history bounds, reverse search, and minimum-draft rule."""
    history = _tokens(history)
    if type(sampled) is not int or sampled < 0:
        raise ValueError("Invalid sampled token")
    if any(type(v) is not int or not 1 <= v <= 1024 for v in (size_n, size_m)):
        raise ValueError("Native simple-ngram sizes must be integers in [1, 1024]")
    count = len(history)
    if count <= size_n + size_m + 1:
        return []
    pattern = history[count-size_n+1:] + [sampled]
    for position in range(count-size_n-1, 0, -1):
        if history[position:position+size_n] == pattern:
            length = min(size_m, count-position-size_n)
            if length < size_n:
                return []
            return history[position+size_n:position+size_n+length]
    return []


def trajectory_opportunities(prompt, generated, *, size_n=12, size_m=48):
    """Inspect every sequential position; opportunities can overlap."""
    prompt, generated = _tokens(prompt), _tokens(generated)
    # Validate sizes even when the trajectory is empty.
    simple_draft([], 0, size_n=size_n, size_m=size_m)
    offers = []
    for index, sampled in enumerate(generated[:-1]):
        proposal = simple_draft(prompt + generated[:index], sampled,
                                size_n=size_n, size_m=size_m)
        if not proposal:
            continue
        actual = generated[index+1:index+1+len(proposal)]
        agreement = 0
        for proposed, observed in zip(proposal, actual):
            if proposed != observed:
                break
            agreement += 1
        offers.append({"after_output_index": index, "proposal": proposal,
                       "observable_future_tokens": len(actual),
                       "matching_prefix_tokens": agreement,
                       "first_token_agrees": proposal[0] == actual[0],
                       "full_proposal_observable": len(actual) == len(proposal),
                       "full_proposal_agrees": (len(actual) == len(proposal)
                                                and agreement == len(proposal))})
    return {"size_n": size_n, "size_m": size_m,
            "generated_tokens": len(generated),
            "inspected_positions": max(0, len(generated)-1),
            "offers": offers, "offer_count": len(offers),
            "first_token_agreement_count": sum(x["first_token_agrees"] for x in offers),
            "full_proposal_agreement_count": sum(x["full_proposal_agrees"] for x in offers),
            "scope": "overlapping sequential-history opportunities; no speculative runtime execution",
            "throughput_estimate": None, "speculative_fidelity_validated": False}


def capture_trajectories(directory):
    """Read only completed small token metadata, keeping requests isolated."""
    directory = Path(directory)
    prompt = _tokens(json.loads((directory / "prompt_tokens.json").read_text()))
    rows = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines()]
    complete = json.loads((directory / "complete.json").read_text())
    if complete.get("completed") is not True or complete.get("captured_steps") != len(rows):
        raise ValueError("Incomplete or inconsistent capture")
    grouped = {}
    previous_request = -1
    for step, row in enumerate(rows):
        request = row.get("request", 0)
        if (row.get("step") != step or type(request) is not int or request < previous_request
                or (previous_request == -1 and request != 0)
                or request > previous_request + 1):
            raise ValueError("Capture steps/requests are not contiguous")
        token = _tokens([row["token"]])[0]
        if row.get("next_input_token", token) != token:
            raise ValueError("Forced input diverges from captured generated token")
        grouped.setdefault(request, []).append(token)
        previous_request = request
    return [{"request": request, "prompt": prompt, "generated": tokens}
            for request, tokens in grouped.items()]


def recurrent_state_bytes(metadata, *, element_bytes=4, cells=1):
    """Derive logical KDA R+S allocation, excluding metadata/KV/indexer/scratch.

    Formula follows llama-hparams.cpp and llama-memory-recurrent.cpp, with
    original dimensions and caller-specified dtype/cell assumptions.
    """
    if any(type(x) is not int or x <= 0 for x in (element_bytes, cells)):
        raise ValueError("Positive integer element/cell sizes required")
    prefix = "glm5-next."
    layers = metadata[prefix + "block_count"]
    heads = metadata[prefix + "attention.head_count"]
    head_dim = metadata[prefix + "kda.head_dim"]
    conv = metadata[prefix + "ssm.conv_kernel"]
    kv_heads = metadata[prefix + "attention.head_count_kv"]
    if (any(type(x) is not int or x <= 0 for x in (layers, heads, head_dim, conv))
            or not isinstance(kv_heads, list) or len(kv_heads) != layers
            or any(type(x) is not int or x < 0 for x in kv_heads)):
        raise ValueError("Unsupported or inconsistent GLM layer metadata")
    recurrent_layers = sum(x == 0 for x in kv_heads)
    conv_elements = 3 * (conv - 1) * heads * head_dim
    state_elements = head_dim * head_dim * heads
    return {"recurrent_layers": recurrent_layers,
            "logical_r_bytes": recurrent_layers * conv_elements * element_bytes * cells,
            "logical_s_bytes": recurrent_layers * state_elements * element_bytes * cells,
            "logical_r_plus_s_bytes": recurrent_layers * (conv_elements + state_elements) * element_bytes * cells,
            "assumptions": {"element_bytes": element_bytes, "cells": cells},
            "scope": "logical KDA R+S allocation per snapshot; not measured serialized checkpoint size"}


def analyze_captures(directories, configs=DEFAULT_CONFIGS):
    result = {"runtime_revision": RUNTIME_REVISION, "captures": [],
              "scope": "metadata replay only; duplicated trajectories are not independent evidence",
              "throughput_estimate": None, "speculative_fidelity_validated": False}
    for directory in directories:
        directory = Path(directory)
        result["captures"].append({
            "capture_directory": str(directory),
            "metadata_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ("prompt_tokens.json", "tokens.jsonl", "complete.json")},
            "requests": [{"request": row["request"], "prompt_tokens": len(row["prompt"]),
                          "generated_tokens": len(row["generated"]),
                          "configs": [trajectory_opportunities(row["prompt"], row["generated"],
                                                              size_n=n, size_m=m) for n, m in configs]}
                         for row in capture_trajectories(directory)]})
    return result


def analyze_bundle(directory):
    """Replay the small published metadata bundle after checking its manifest.

    Original benchmark JSON is represented by an explicitly derived token
    extract and its original-file hash, never by a fabricated native capture.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unknown structural bundle schema")
    checked = set()
    for row in manifest["files"]:
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Structural input escapes its bundle")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Structural input hash/size mismatch")
        checked.add(row["path"])
    for capture in manifest["native_captures"]:
        if any(str(Path(capture) / name) not in checked
               for name in ("prompt_tokens.json", "tokens.jsonl", "complete.json")):
            raise ValueError("Native capture is not completely manifested")
    for name in ("benchmark-continuations.json", "model-dimensions.json"):
        if name not in checked:
            raise ValueError("Derived structural input is not manifested")
    result = analyze_captures([directory / name for name in manifest["native_captures"]])
    for capture in result["captures"]:
        capture["capture_directory"] = str(Path(capture["capture_directory"]).relative_to(directory))
    result["inputs_manifest_sha256"] = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    result["input_path_scope"] = "all input paths are relative to the supplied structural-inputs bundle"
    result["timed_benchmark_continuations"] = []
    for row in json.loads((directory / "benchmark-continuations.json").read_text()):
        result["timed_benchmark_continuations"].append({
            "source": row["source"], "source_sha256": row["source_sha256"],
            "repetition": row["repetition"], "token_extract_scope": row["scope"],
            "configs": [trajectory_opportunities(row["prompt_tokens"], row["generated_tokens"],
                                                size_n=n, size_m=m) for n, m in DEFAULT_CONFIGS]})
    dimensions = json.loads((directory / "model-dimensions.json").read_text())
    result["kda_snapshot_allocation"] = recurrent_state_bytes(dimensions["metadata"])
    result["checkpoint_weight_inventory"] = dimensions["tensor_inventory_summary"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if bool(args.captures) == bool(args.bundle):
        parser.error("Supply native capture directories or --bundle, exclusively")
    result = analyze_bundle(args.bundle) if args.bundle else analyze_captures(args.captures)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
