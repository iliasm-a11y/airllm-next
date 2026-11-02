"""Replay native route traces and bounded exact expert IO, without model/GPU execution.

`policy` is explicitly a metadata-only eviction replay. `io` actually reads the
selected packed weights through AdaptiveExpertCache and reports measured waits,
syscalls and logical bytes. Neither mode measures physical SSD traffic or tok/s.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import weakref

from research.glm53_m5.compare_fidelity import json_lines, tensor_view
from research.glm53_gpu.expert_cache import ByteSpan, ExpertRef
from research.glm53_gpu.quantized_probe import read_selected_rows
from research.glm53_gpu.stream_selected import MemoryAdmission
from .cache_adaptive import AdaptiveExpertCache, POLICIES, ReusePolicy

MIB = 1024**2


def read_routes(capture):
    root = Path(capture)
    manifest = root / "tensors.jsonl"
    if manifest.stat().st_size > 16 * MIB:
        raise ValueError("route manifest exceeds parsing budget")
    found = {}
    for row in json_lines(manifest):
        if not row.get("name", "").startswith("ffn_moe_topk-"):
            continue
        layer = int(row["name"].rsplit("-", 1)[1])
        step = row["step"]
        if not 3 <= layer < 45 or not 0 <= step < 4:
            continue
        key = step, layer
        if key in found:
            raise ValueError("ambiguous duplicate selected route tensor")
        if (row["type"] != 26 or row["shape"][0] != 8 or row["shape"][2:] != [1, 1] or
                not 0 < row["bytes"] <= MIB):
            raise ValueError("invalid bounded top-8 route tensor")
        view = tensor_view(root, row)
        columns = []
        for column in range(view.shape[1]):
            ids = [int(v) for v in view[:, column, 0, 0]]
            if any(not 0 <= e < 288 for e in ids):
                raise ValueError("GLM route expert outside [0,288)")
            columns.append(ids)
        found[key] = columns
    expected = {(step, layer) for step in range(4) for layer in range(3, 45)}
    if set(found) != expected:
        raise ValueError("replay requires all 42 MoE layers across four captured steps")
    records = []
    for (step, layer), columns in sorted(found.items()):
        for column, ids in enumerate(columns):
            records.append({"step": step, "layer": layer, "column": column, "ids": ids})
    return records, {"capture_name": root.name, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     "route_ids_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                     "steps": 4, "layers": 42, "selected_groups": len(records),
                     "selected_expert_elements": sum(len(r["ids"]) for r in records),
                     "execution_order": "step, layer, token column, original route rank"}


def bank_metadata(gguf, layer, projection):
    tensor = f"blk.{layer}.ffn_{projection}_exps.weight"
    _, width, meta = read_selected_rows(gguf, tensor, 0, 0, 1)
    width, rows, experts = meta["dims"]
    if experts != 288 or width % 256 or rows < 1:
        raise ValueError("unexpected GLM expert bank")
    return {"tensor": tensor, "offset": meta["absolute_offset"], "width": width,
            "rows": rows, "experts": experts, "expert_bytes": rows * (width // 256 * 84)}


def refs_for_record(record, metadata):
    banks = metadata[record["layer"]]
    return [ExpertRef((record["layer"], e), tuple(
        ByteSpan(bank["offset"] + e * bank["expert_bytes"], bank["expert_bytes"]) for bank in banks))
        for e in record["ids"]]


def replay_policy(records, metadata, *, capacity_bytes, policy):
    """Zero-payload allocation trace replay; no IO/wait/throughput is inferred."""
    if policy == "weighted":
        raise ValueError("metadata replay cannot invent measured expert load costs for weighted eviction")
    preference = ReusePolicy(policy)
    resident = OrderedDict()
    payload = hits = misses = evictions = logical = 0
    peak = 0
    by_phase = {phase: {"hits": 0, "misses": 0, "logical_miss_bytes": 0, "selected_groups": 0}
                for phase in ("prefill", "decode")}
    for record in records:
        phase = by_phase["prefill" if record["step"] == 0 else "decode"]
        phase["selected_groups"] += 1
        refs = refs_for_record(record, metadata)
        selected = {ref.key: ref for ref in refs}
        needed = sum(ref.nbytes for key, ref in selected.items() if key not in resident)
        if sum(ref.nbytes for ref in selected.values()) > capacity_bytes:
            raise ValueError("one selected expert group exceeds the replay budget")
        old = set(resident)
        victims = sorted((key for key in resident if key not in selected),
                         key=lambda key: preference.score(key, resident[key]))
        while payload + needed > capacity_bytes:
            victim = victims.pop(0)
            payload -= resident.pop(victim)
            preference.forget_resident(victim)
            evictions += 1
        for key, ref in selected.items():
            if key in resident:
                hits += 1
                phase["hits"] += 1
            else:
                misses += 1
                logical += ref.nbytes
                phase["misses"] += 1
                phase["logical_miss_bytes"] += ref.nbytes
                resident[key] = ref.nbytes
                payload += ref.nbytes
        preference.observe(refs, old, capacity_bytes)
        peak = max(peak, payload)
    return {"policy": policy, "scope": "metadata-only cache policy replay",
            "capacity_bytes": capacity_bytes, "hits": hits, "misses": misses,
            "evictions": evictions, "logical_miss_bytes": logical, "peak_payload_bytes": peak,
            "by_phase": by_phase,
            "physical_io_measured": False, "wait_time_measured": False}


def replay_io(gguf, records, metadata, *, admission, policies=POLICIES,
              capacity_bytes=16 * MIB, max_read_bytes=128 * MIB, max_demands=12):
    """Actual one-expert pump from a captured route prefix, <=128 MiB total reads.

    Only the first decode layer's original ordered expert prefix is used. Each
    request owns one expert so a small cache can exercise eviction. This is an
    IO/hash consumer, not MoE arithmetic or the full42-layer inference pipeline.
    """
    if not 0 < max_read_bytes <= 128 * MIB or not 0 < capacity_bytes <= 128 * MIB:
        raise ValueError("real IO experiment budgets must be <=128 MiB")
    demands = []
    for record in records:
        if record["layer"] == 3 and record["step"] > 0:
            demands.extend(refs_for_record(record, metadata))
    demands = demands[:max_demands]
    results = []
    total_reads = 0
    for policy in policies:
        begin = time.perf_counter()
        rows = []
        with AdaptiveExpertCache(gguf, capacity_bytes, policy=policy, enabled=True,
                                 admission=admission, max_entries=128) as cache:
            for ordinal, ref in enumerate(demands):
                current = cache._entries.get(ref.key)
                incoming = 0 if current is not None and current.state == "ready" else ref.nbytes
                if total_reads + incoming > max_read_bytes:
                    raise ValueError("requested policy comparison would exceed the shared real-read budget")
                start = time.perf_counter()
                with cache.request_selected([ref]) as request:
                    for lease in request.iter_ready():
                        digest = hashlib.sha256(lease.data).hexdigest()
                total_reads += incoming
                rows.append({"ordinal": ordinal, "key": list(ref.key),
                             "spans": [{"offset": s.offset, "length": s.length} for s in ref.spans],
                             "sha256_original_packed_bytes": digest,
                             "request_consume_seconds": time.perf_counter() - start})
            snapshot = cache.snapshot()
        results.append({"policy": policy, "scope": "actual internal-file IO with exact packed-byte hash consumer",
                        "requests": rows, "cache_before_close": snapshot, "cache_after_close": cache.snapshot(),
                        "elapsed_seconds": time.perf_counter() - begin})
    hashes = [{tuple(row["key"]): row["sha256_original_packed_bytes"] for row in result["requests"]}
              for result in results]
    return {"results": results, "same_original_bytes_across_policies": all(h == hashes[0] for h in hashes),
            "shared_logical_read_bytes": total_reads, "shared_read_budget_bytes": max_read_bytes,
            "physical_ssd_traffic_measured": False, "model_or_gpu_executed": False}


def replay_reuse(gguf, records, metadata, *, admission, layer=40, capacity_bytes=32 * MIB,
                 max_read_bytes=64 * MIB):
    """Actually stage two consecutive recorded tokens, verifying shared payload reuse."""
    selected = [r for r in records if r["layer"] == layer and r["step"] in (1, 2) and r["column"] == 0]
    if len(selected) != 2 or not 0 < capacity_bytes <= 64 * MIB or not 0 < max_read_bytes <= 64 * MIB:
        raise ValueError("reuse probe requires two decode steps and <=64 MiB payload/read budgets")
    owners = {}
    hashes = {}
    reports = []
    logical = 0
    reused = 0
    with AdaptiveExpertCache(gguf, capacity_bytes, enabled=True, admission=admission) as cache:
        for token_index, record in enumerate(selected):
            refs = refs_for_record(record, metadata)
            incoming = sum(ref.nbytes for ref in refs if ref.key not in cache._entries)
            if logical + incoming > max_read_bytes:
                raise ValueError("two-token reuse probe exceeds its explicit read budget")
            logical += incoming
            rows = []
            with cache.request_selected(refs) as request:
                for lease in request.iter_ready():
                    key = lease.key
                    digest = hashlib.sha256(lease.data).hexdigest()
                    same_owner = (key in owners and owners[key][0]() is lease._entry and
                                  owners[key][1] == id(lease._entry.payload))
                    if key in hashes and hashes[key] != digest:
                        raise ValueError("a reused expert returned different original bytes")
                    reused += int(same_owner)
                    # Weak entry references and numeric identities do not retain
                    # payloads after lease release/eviction/shrink.
                    owners[key] = (weakref.ref(lease._entry), id(lease._entry.payload))
                    hashes[key] = digest
                    ref = lease._entry.ref
                    rows.append({"expert_id": key[1], "sha256_original_bytes": digest,
                                 "reused_same_resident_buffer": same_owner,
                                 "spans": [{"offset": s.offset, "length": s.length} for s in ref.spans]})
                snapshot = cache.snapshot()
                adaptation = None
                if token_index == 1:
                    immediately_applied = cache.resize(0)
                    adaptation = {"immediately_applied": immediately_applied,
                                  "while_selected_leases_pinned": cache.snapshot()}
            reports.append({"captured_step": record["step"], "original_ordered_ids": record["ids"],
                            "experts": rows, "cache_while_pinned": snapshot,
                            "shrink_test": adaptation, "cache_after_request": cache.snapshot()})
        owners.clear()
        final = cache.snapshot()
    return {"scope": "actual two consecutive captured token routes, one gate projection, IO/hash consumer",
            "layer": layer, "tokens": reports, "same_resident_buffers_reused": reused,
            "logical_bytes_read": logical, "read_budget_bytes": max_read_bytes,
            "cache_before_close": final, "cache_after_close": cache.snapshot(),
            "physical_ssd_traffic_measured": False, "model_or_gpu_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("policy", "io", "reuse"))
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to overwrite previous evidence")
    if args.mode != "policy" and not args.experimental:
        parser.error("actual selected-weight reads require --experimental")
    if args.mode != "policy" and len(args.capture) != 1:
        parser.error("real-read budgets cover exactly one capture per invocation")
    admission = MemoryAdmission()
    if not admission(0):
        parser.error("current memory/pressure admission failed")
    # Policy replay describes full gate/up/down payloads. The real IO experiment
    # deliberately limits itself to one complete gate projection per expert.
    projections = ("gate", "up", "down") if args.mode == "policy" else ("gate",)
    layers = range(3, 45) if args.mode == "policy" else ((40,) if args.mode == "reuse" else (3,))
    metadata = {layer: [bank_metadata(args.gguf, layer, p) for p in projections] for layer in layers}
    captures = []
    for capture in args.capture:
        records, source = read_routes(capture)
        result = ([replay_policy(records, metadata, capacity_bytes=size * MIB, policy=policy)
                   for size in (64, 128) for policy in ("lru", "slru", "lfu")]
                  if args.mode == "policy" else
                  replay_reuse(args.gguf, records, metadata, admission=admission) if args.mode == "reuse" else
                  replay_io(args.gguf, records, metadata, admission=admission))
        captures.append({"source": source, "results": result})
    source_stat = args.gguf.stat()
    report = {"schema": 1, "mode": args.mode, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_file": {"name": args.gguf.name, "bytes": source_stat.st_size,
                              "mtime_ns": source_stat.st_mtime_ns,
                              "integrity_scope": "stat identity per read plus selected expert SHA256; not a new full model hash"},
              "source_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ("cache_adaptive.py", "cache_replay.py")},
              "captures": captures, "validated_bank_metadata": metadata,
              "limits": ["No tokens/s or whole-model fidelity is measured",
                         "File cache and background workload are uncontrolled",
                         "Logical read bytes are not NAND or physical SSD traffic",
                         "No Fanxiang volume was mounted during inspection; /Volumes/SSD was a root-filesystem directory",
                         "Weighted ranking is tested only during real reads with observed load costs; omitted from metadata replay",
                         "No reusable buffer pool: evicted payloads are freed; an uncharged spare pool would violate the payload budget"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "captures": len(captures), "output": str(args.output)}))


if __name__ == "__main__":
    main()
