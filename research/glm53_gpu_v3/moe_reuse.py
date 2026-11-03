"""Saved-route metadata simulations, without expert payloads or model reads.

Cache misses below are logical packed-byte demands under hypothetical explicit
policies. They are not measured SSD/interface/NAND reads or token-rate gains.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

from research.glm53_gpu_v3.source_union import (EXPERT_NAME, read_closed_capture, validate_inventory)
from research.glm53_gpu_v3.dense_inventory import selected_matrices

GIB = 1024**3
MIB = 1024**2


def _ids(ids):
    if (not isinstance(ids, (list, tuple)) or len(ids) != 8 or
            any(type(e) is not int or not 0 <= e < 288 for e in ids) or len(set(ids)) != 8):
        raise ValueError("canonical ordered top-8 IDs required")
    return tuple(ids)


def _sizes(sizes):
    if (not isinstance(sizes, dict) or not sizes or
            any(type(layer) is not int or layer < 0 or type(size) is not int or size <= 0
                for layer, size in sizes.items())):
        raise ValueError("positive original expert sizes required")


def expert_sizes(inventory):
    tensors = validate_inventory(inventory)
    banks = defaultdict(dict)
    for name, tensor in tensors.items():
        match = EXPERT_NAME.fullmatch(name)
        if match:
            if tensor["type"] != "Q2_K" or tensor["dimensions"][2:] != [288, 1]:
                raise ValueError("unexpected original expert bank")
            banks[int(match[1])][match[2]] = tensor["payload_bytes"]//288
    if not banks or any(set(projections) != {"gate", "up", "down"} for projections in banks.values()):
        raise ValueError("complete gate/up/down banks required for each layer")
    sizes = {layer: sum(projections.values()) for layer, projections in banks.items()}
    dense = sum(tensors[row["name"]]["payload_bytes"] for row in selected_matrices(inventory))
    return sizes, dense


def _totals(per_step):
    hits = sum(row["hits"] for row in per_step)
    misses = sum(row["misses"] for row in per_step)
    miss_bytes = sum(row["logical_miss_bytes"] for row in per_step)
    return {"steps": len(per_step), "hits": hits, "misses": misses,
            "hit_fraction": hits/(hits+misses) if hits+misses else None,
            "logical_miss_bytes": miss_bytes,
            "mean_logical_miss_bytes_per_step": miss_bytes/len(per_step) if per_step else None,
            "per_step": per_step, "payload_allocated": False,
            "physical_storage_reads_measured": False}


def simulate_layer_last_k(records, sizes, *, keep=8, initial=None):
    """Keep first k IDs of each layer's previous route, using no future IDs.

    Incoming miss staging/compute buffers are outside the persistent payload
    model. A production implementation must charge them and release GPU leases.
    """
    if type(keep) is not int or not 0 <= keep <= 8:
        raise ValueError("keep must be 0..8")
    _sizes(sizes)
    resident = {layer: _ids(ids)[:keep] for layer, ids in (initial or {}).items()}
    steps = defaultdict(lambda: {"hits": 0, "misses": 0, "logical_miss_bytes": 0})
    for row in records:
        layer, step = row["layer"], row["step"]
        ids = _ids(row["ids"])
        old = set(resident.get(layer, ()))
        hits = sum(e in old for e in ids)
        steps[step]["hits"] += hits
        steps[step]["misses"] += 8-hits
        steps[step]["logical_miss_bytes"] += (8-hits)*sizes[layer]
        resident[layer] = ids[:keep]
    report = _totals([{"step": step, **row} for step, row in sorted(steps.items())])
    report.update(policy="per-layer previous route, original first-k ranks", keep_per_layer=keep,
                  persistent_raw_payload_capacity_bytes=keep*sum(sizes.values()),
                  transient_staging_memory_accounted=False)
    return report


def simulate_global_lru(records, sizes, *, capacity_bytes):
    """Metadata-only LRU with exact expert sizes and selected-group pinning."""
    if type(capacity_bytes) is not int or capacity_bytes <= 0:
        raise ValueError("positive metadata cache capacity required")
    _sizes(sizes)
    resident = OrderedDict()
    payload = peak = evictions = 0
    steps = defaultdict(lambda: {"hits": 0, "misses": 0, "logical_miss_bytes": 0})
    phases = {}
    for row in records:
        layer, step = row["layer"], row["step"]
        ids = _ids(row["ids"])
        size = sizes[layer]
        if 8*size > capacity_bytes:
            raise ValueError("one actual selected group exceeds LRU capacity")
        keys = [(layer, e) for e in ids]
        selected = set(keys)
        needed = sum(size for key in keys if key not in resident)
        victims = [key for key in resident if key not in selected]
        while payload+needed > capacity_bytes:
            victim = victims.pop(0)
            payload -= resident.pop(victim)
            evictions += 1
        for key in keys:
            if key in resident:
                steps[step]["hits"] += 1
            else:
                steps[step]["misses"] += 1
                steps[step]["logical_miss_bytes"] += size
                resident[key] = size
                payload += size
            resident.move_to_end(key)
        peak = max(peak, payload)
        phase = row.get("phase", "decode")
        if step in phases and phases[step] != phase:
            raise ValueError("a step has ambiguous phases")
        phases[step] = phase
    per_step = [{"step": step, "phase": phases[step], **row} for step, row in sorted(steps.items())]
    report = _totals(per_step)
    report.update(policy="global LRU, selected group pinned during admission", capacity_bytes=capacity_bytes,
                  peak_packed_payload_bytes=peak, evictions=evictions,
                  decode=_totals([row for row in per_step if row["phase"] == "decode"]))
    return report


def adjacent_decode_reuse(route_steps, sizes):
    _sizes(sizes)
    layers = sorted(sizes)
    steps = sorted(route_steps)
    per_layer = {layer: {"hits": 0, "demands": 0, "logical_miss_bytes": 0,
                         "transition_hits": []} for layer in layers}
    transitions = []
    for previous, current in zip(steps, steps[1:]):
        if current != previous+1:
            raise ValueError("adjacent decoded steps required")
        hits = misses = miss_bytes = union = 0
        layer_hits = {}
        for layer in layers:
            a = _ids(route_steps[previous][layer])
            b = _ids(route_steps[current][layer])
            shared = len(set(a)&set(b))
            per_layer[layer]["hits"] += shared
            per_layer[layer]["demands"] += 8
            per_layer[layer]["logical_miss_bytes"] += (8-shared)*sizes[layer]
            per_layer[layer]["transition_hits"].append(shared)
            hits += shared
            misses += 8-shared
            miss_bytes += (8-shared)*sizes[layer]
            union += (16-shared)*sizes[layer]
            layer_hits[layer] = shared
        transitions.append({"previous_step": previous, "current_step": current,
                            "hits": hits, "misses": misses, "logical_miss_bytes": miss_bytes,
                            "two_token_expert_union_bytes": union, "shared_experts_by_layer": layer_hits})
    result = _totals(transitions)
    result["scope"] = "adjacent decoded-token transitions only; first decoded token is excluded"
    result["per_layer"] = [{"layer": layer, **row,
        "mean_shared_experts_out_of_eight": row["hits"]/len(transitions) if transitions else None,
        "hit_fraction": row["hits"]/row["demands"] if row["demands"] else None}
        for layer, row in per_layer.items()]
    unions = [row["two_token_expert_union_bytes"] for row in transitions]
    result["two_token_expert_union_bytes"] = {"minimum": min(unions) if unions else None,
        "mean": sum(unions)/len(unions) if unions else None, "maximum": max(unions) if unions else None}
    return result


def analyze(capture, inventory_path, source_union_path):
    inventory_path, source_union_path = Path(inventory_path), Path(source_union_path)
    inventory = json.loads(inventory_path.read_text())
    sizes, dense = expert_sizes(inventory)
    _, routes, count, provenance = read_closed_capture(capture)
    if provenance["prefill_step_ids"] != [0] or count < 3:
        raise ValueError("this comparison requires one closed request and at least two decoded steps")
    if sorted(sizes) != list(range(3, 45)) or set(routes) != {(step, layer) for step in range(count) for layer in sizes}:
        raise ValueError("all42MoE layers and captured steps required")
    proof = json.loads(source_union_path.read_text())
    if (proof.get("all_steps_closed") is not True or proof.get("decode_steps_closed") is not True or
            proof.get("source", {}).get("metadata_sha256") != provenance["metadata_sha256"] or
            proof.get("source", {}).get("canonical_routes_sha256") != provenance["canonical_routes_sha256"] or
            proof.get("source", {}).get("inventory_sha256") != hashlib.sha256(inventory_path.read_bytes()).hexdigest()):
        raise ValueError("closed operand-union proof does not identify these exact routes/metadata")
    decode = {step: {layer: _ids(routes[(step, layer)][0]) for layer in sorted(sizes)} for step in range(1, count)}
    if any(len(routes[(step, layer)]) != 1 for step in decode for layer in sizes):
        raise ValueError("single-column decode required")
    decode_records = [{"step": step, "layer": layer, "ids": ids, "phase": "decode"}
                      for step, by_layer in decode.items() for layer, ids in by_layer.items()]
    full_records = [{"step": step, "layer": layer, "column": column, "ids": ids,
                     "phase": "prefill" if step == 0 else "decode"}
                    for (step, layer), columns in sorted(routes.items()) for column, ids in enumerate(columns)]
    last_prompt = {layer: _ids(routes[(0, layer)][-1]) for layer in sizes}
    adjacent = adjacent_decode_reuse(decode, sizes)
    global_lru = simulate_global_lru(full_records, sizes, capacity_bytes=128*MIB)
    cold = simulate_layer_last_k(decode_records, sizes, keep=8)
    warm = simulate_layer_last_k(decode_records, sizes, keep=8, initial=last_prompt)
    tradeoffs = [simulate_layer_last_k(decode_records, sizes, keep=k, initial=last_prompt) for k in (0, 1, 2, 4, 8)]
    payload = 8*sum(sizes.values())
    total_raw = dense+payload
    ideal = []
    for label, misses in ((f"{adjacent['steps']} adjacent decode transitions", adjacent["mean_logical_miss_bytes_per_step"]),
                          (f"{count-1} decode steps with last prompt route retained", warm["mean_logical_miss_bytes_per_step"]),
                          ("global 128 MiB LRU decode", global_lru["decode"]["mean_logical_miss_bytes_per_step"])):
        for dense_resident in (False, True):
            assumed = misses+(0 if dense_resident else dense)
            ideal.append({"basis": label, "fixed_dense_fully_resident": dense_resident,
                "logical_MoE_miss_bytes_per_token": misses,
                "conditional_interface_bytes_per_token_if_all_misses_reach_storage": assumed,
                "hypothetical_interface_bytes_per_second": 1_000_000_000,
                "ideal_IO_only_seconds_per_token": assumed/1e9,
                "ideal_IO_only_tokens_per_second": 1e9/assumed,
                "performance_measured": False})
    return {"schema_version": 1, "scope": "metadata-only reuse simulation on the supplied actual closed trajectory",
        "source": {**provenance, "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "source_union_file": source_union_path.name, "source_union_sha256": hashlib.sha256(source_union_path.read_bytes()).hexdigest(),
            "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "decode_tokens": count-1, "expert_payload_bytes_by_layer": sizes,
        "complete_previous_top8_all_layers_raw_bytes": payload, "fixed_dense_raw_bytes": dense,
        "fixed_dense_plus_previous_top8_raw_bytes": total_raw,
        "raw_payload_GiB": total_raw/GIB,
        "16GiB_minus_raw_payload_bytes_before_any_other_memory": 16*GIB-total_raw,
        "fits_8_5GiB_process_RSS_cap_as_fully_resident_explicit_CPU_raw_payload": total_raw <= 8.5*GIB,
        "adjacent_decode_transitions": adjacent, "global_LRU128MiB_prefill_then_decode": global_lru,
        "last8_per_layer_cold_decode_start": cold, "last8_per_layer_initialized_from_last_prompt_column": warm,
        "uniform_previous_first_k_rank_tradeoffs": tradeoffs,
        "two_token_union_plus_fixed_dense_bytes": {key: value+dense for key, value in adjacent["two_token_expert_union_bytes"].items()},
        "conditional_storage_scenarios": ideal,
        "interpretation": [
            "No expert payload is allocated, copied or read. Miss bytes are logical packed demand under simulated policies, never measured interface/NAND reads.",
            "The per-layer last-k policy uses only previous actual ordered route ranks, not the next token's IDs. The replay itself is offline evidence, not a native cache implementation or inference speedup.",
            "Initial last-prompt routes are a separate warm-start assumption, not a measured existing cache state. Cold-start and adjacent-only denominators remain separate.",
            f"Retaining previous 8 requires {payload/1e9:.6f} GB across 42 layers in this checkpoint. Releasing unselected old entries before filling misses can keep the persistent payload fixed; staging, GPU leases and all other memory still need a real implementation/admission proof.",
            "A simultaneous two-token union models retaining both old and new sets. It is not mandatory for an eviction-before-fill design, and routes cannot be assumed known before the router runs.",
            "With all dense operands retained plus previous experts, the measured route reuse permits an optimistic IO-only scenario; this does not prove such memory retention or token rate is achievable.",
            "The raw explicit CPU payload alone exceeds the current 8.5 GiB RSS cap. OS file cache or compression outside that payload complicate a universal feasibility claim; 16 GiB alone does not prove 0.5 tok/s impossible.",
            "Normal-workload feasibility must use separately measured nonoverlapping memory categories, including compressor/system/runtime/staging; do not blindly subtract overlapping RSS/cache counters.",
            "Unchanged dense streaming plus this expert cache still requires dense bytes; the major priority remains measured dense-path reuse.",
        ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-union", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refusing to overwrite prior evidence")
    report = analyze(args.capture, args.inventory, args.source_union)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    adjacent = report["adjacent_decode_transitions"]
    print(json.dumps({"adjacent_hits": adjacent["hits"], "adjacent_hit_fraction": adjacent["hit_fraction"],
        "mean_logical_MoE_miss_bytes": adjacent["mean_logical_miss_bytes_per_step"],
        "global_LRU_decode_hits": report["global_LRU128MiB_prefill_then_decode"]["decode"]["hits"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
