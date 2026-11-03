"""Offline proof of distinct declared weight-byte ranges from a closed capture.

Inputs are saved source metadata, canonical same-capture route tensors and GGUF
headers. No GGUF payload is opened. The output concerns the weight operands of
MUL_MAT/MUL_MAT_ID in this implementation, not physical SSD/NAND transfers or a
universal lower bound for every possible inference algorithm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from research.glm53_m5.compare_fidelity import safe_file, tensor_view
from research.glm53_gpu_v3.dense_inventory import selected_matrices

CODECS = {"F32": (0, 1, 4), "BF16": (30, 1, 2), "Q8_0": (8, 32, 34), "Q2_K": (10, 256, 84)}
EXPERT_NAME = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
ROUTE_NAME = re.compile(r"^ffn_moe_topk-(\d+)$")
MAX_INT = 2**63-1


def integer(value, label, *, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_INT:
        raise ValueError(f"invalid {label}")
    return value


def dimensions(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ValueError("expected one to four dimensions")
    return [integer(v, "dimension", minimum=1) for v in values]+[1]*(4-len(values))


def packed_layout(shape, codec):
    _, block, size = CODECS[codec]
    ne = dimensions(shape)
    if ne[0] % block:
        raise ValueError("quantized primary dimension is not a whole block")
    nb = [size, ne[0]//block*size, 0, 0]
    nb[2] = nb[1]*ne[1]
    nb[3] = nb[2]*ne[2]
    payload = nb[3]*ne[3]
    integer(payload, "packed payload", minimum=1)
    return ne, nb, payload


def merge_intervals(intervals):
    """Intervals are file-qualified, half-open and unrounded to VM pages."""
    by_file = defaultdict(list)
    for file, start, end in intervals:
        if not isinstance(file, str) or not file or integer(end, "range end") <= integer(start, "range start"):
            raise ValueError("invalid interval")
        by_file[file].append((start, end))
    result = {}
    for file, spans in sorted(by_file.items()):
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        result[file] = merged
    return result


def union_bytes(intervals):
    return sum(end-start for spans in merge_intervals(intervals).values() for start, end in spans)


def validate_inventory(inventory):
    files = {}
    for row in inventory["files"]:
        name = row["file"]
        if not isinstance(name, str) or name in files or Path(name).name != name:
            raise ValueError("duplicate/invalid GGUF file name")
        size = integer(row["file_bytes"], "GGUF file length", minimum=1)
        offset = integer(row["data_offset"], "GGUF data offset")
        if offset > size:
            raise ValueError("GGUF data offset outside declared file")
        files[name] = (offset, size)
    tensors = {}
    spans = []
    for row in inventory["tensors"]:
        name, codec = row["name"], row["type"]
        if not isinstance(name, str) or name in tensors or codec not in CODECS:
            raise ValueError("duplicate/unsupported GGUF tensor")
        ne, nb, payload = packed_layout(row["dims"], codec)
        relative = integer(row["offset"], "tensor offset")
        padded = integer(row["bytes_with_alignment"], "aligned tensor length", minimum=1)
        data, file_size = files[row["file"]]
        if padded < payload or relative+padded > file_size-data:
            raise ValueError("tensor descriptor outside declared GGUF payload")
        start = data+relative
        tensors[name] = {"name": name, "type": codec, "file": row["file"], "dimensions": ne,
                         "strides": nb, "payload_bytes": payload, "file_start": start,
                         "file_end": start+payload}
        spans.append((row["file"], start, start+payload))
    if union_bytes(spans) != sum(t["payload_bytes"] for t in tensors.values()):
        raise ValueError("GGUF source tensors overlap; ownership is ambiguous")
    return tensors


def validate_source(source, base):
    """Admit an original contiguous operand or a proved contiguous alias range.

    Gapped/broadcast/unresolved views are unknown, never replaced by the full
    base tensor. Nonquantized contiguous permutations are supported explicitly.
    """
    if source.get("alias_resolved") is not True:
        raise ValueError("unresolved source alias")
    depth = integer(source.get("alias_depth"), "alias depth")
    if depth > 16:
        raise ValueError("source alias exceeds native traversal bound")
    if source.get("base_name") != base["name"]:
        raise ValueError("source base name differs from joined GGUF tensor")
    type_id, block, size = CODECS[base["type"]]
    if type(source.get("type")) is not int or source.get("type") != type_id:
        raise ValueError("source codec differs from GGUF")
    ne = dimensions(source.get("dimensions"))
    if len(source["dimensions"]) != 4 or ne[0] % block:
        raise ValueError("invalid four-dimensional source layout")
    nb = source.get("strides")
    if not isinstance(nb, list) or len(nb) != 4:
        raise ValueError("invalid source strides")
    nb = [integer(v, "source stride") for v in nb]
    if block > 1 and nb[0] != size:
        raise ValueError("quantized block stride is unsupported")
    if any(v % size for v in nb):
        raise ValueError("stride is not aligned to an encoded element/block")
    declared = integer(source.get("bytes"), "source byte span", minimum=1)
    calculated = (size+sum((n-1)*stride for n, stride in zip(ne, nb)) if block == 1 else
                  ne[0]*nb[0]//block+sum((ne[i]-1)*nb[i] for i in range(1, 4)))
    if calculated != declared:
        raise ValueError("source byte span disagrees with GGML strides")
    counts = [ne[0]//block, *ne[1:]]
    contiguous = size
    for stride, count in sorted((stride, count) for stride, count in zip(nb, counts) if count > 1):
        if stride != contiguous:
            raise ValueError("gapped, overlapping or broadcast alias is unknown")
        contiguous *= count
    if contiguous != declared:
        raise ValueError("source layout does not exactly cover its declared span")
    offset = integer(source.get("offset_from_base"), "source offset")
    if offset % size or offset+declared > base["payload_bytes"]:
        raise ValueError("source alias outside original tensor payload")
    if depth == 0 and (source.get("name") != base["name"] or offset or ne != base["dimensions"] or nb != base["strides"]):
        raise ValueError("unaliased source does not match full original descriptor")
    usage = source.get("buffer_usage")
    if usage is not None and (type(usage) is not int or usage != 1):
        raise ValueError("joined original weight is not in a WEIGHTS buffer")
    return offset, declared


def prove_rows(inventory, source_rows, routes, *, captured_steps, prefill_steps=(0,)):
    """Return exact known ranges plus explicit unknown/coverage failures."""
    tensors = validate_inventory(inventory)
    expected_dense = {row["name"] for row in selected_matrices(inventory)}
    expected_expert = {name for name in tensors if EXPERT_NAME.fullmatch(name)}
    if not expected_dense or not expected_expert:
        raise ValueError("expected fixed matrices and expert banks are required")
    steps = integer(captured_steps, "captured_steps", minimum=1)
    prefills = {integer(step, "prefill step") for step in prefill_steps}
    if 0 not in prefills or any(step >= steps for step in prefills):
        raise ValueError("invalid prefill-step boundaries")
    groups = defaultdict(list)
    for row in source_rows:
        step = integer(row.get("step"), "source-plan step")
        if step >= steps or row.get("op") not in ("MUL_MAT", "MUL_MAT_ID"):
            raise ValueError("source-plan row outside completed capture or unsupported op")
        groups[step].append(row)
    reports = []
    for step in range(steps):
        intervals = {"dense": [], "expert": []}
        names = {"dense": set(), "expert": set()}
        by_name = defaultdict(list)
        counts = defaultdict(int)
        logical = {"dense": 0, "expert": 0}
        unknown = []
        excluded = []
        source_usage_available = True
        for ordinal, row in enumerate(groups[step]):
            source = row.get("source_weight")
            if not isinstance(source, dict):
                unknown.append({"ordinal": ordinal, "reason": "missing source descriptor"})
                continue
            source_usage_available &= "buffer_usage" in source
            base_name = source.get("base_name")
            base = tensors.get(base_name)
            if base is None:
                usage = source.get("buffer_usage")
                detail = {"ordinal": ordinal, "name": source.get("name"), "base_name": base_name,
                          "buffer_usage": usage}
                # Only explicit COMPUTE buffers establish nonweight operands.
                # ANY/missing metadata cannot prove that an unknown view is safe.
                if type(usage) is int and usage == 2:
                    excluded.append({**detail, "reason": "explicit COMPUTE buffer, not original weights"})
                else:
                    unknown.append({**detail, "reason": "source base absent from GGUF inventory"})
                continue
            try:
                offset, span = validate_source(source, base)
                expert = EXPERT_NAME.fullmatch(base_name)
                if row["op"] == "MUL_MAT_ID":
                    if not expert or base["type"] != "Q2_K":
                        raise ValueError("indexed multiplication is not an original Q2_K expert bank")
                    layer = int(expert[1])
                    ne, nb = base["dimensions"], base["strides"]
                    if (ne[2:] != [288, 1] or offset or span != base["payload_bytes"] or
                            source["dimensions"] != ne or source["strides"] != nb):
                        raise ValueError("expert source does not match the complete canonical bank layout")
                    columns = routes.get((step, layer))
                    if not columns:
                        raise ValueError("same-capture canonical route is missing")
                    if step not in prefills and len(columns) != 1:
                        raise ValueError("multi-column route cannot be classified as single-token decode")
                    selected = set()
                    for ids in columns:
                        if (len(ids) != 8 or any(type(e) is not int or not 0 <= e < ne[2] for e in ids) or
                                len(set(ids)) != 8):
                            raise ValueError("invalid ordered top-8 expert IDs")
                        selected.update(ids)
                    bank = [(base["file"], base["file_start"]+e*nb[2], base["file_start"]+(e+1)*nb[2])
                            for e in selected]
                    category = "expert"
                    count_bytes = len(selected)*nb[2]
                else:
                    if expert:
                        raise ValueError("expert bank used by nonindexed multiplication")
                    category = "dense"
                    bank = [(base["file"], base["file_start"]+offset, base["file_start"]+offset+span)]
                    count_bytes = span
                intervals[category].extend(bank)
                by_name[base_name].extend(bank)
                logical[category] += count_bytes
                names[category].add(base_name)
                counts[category] += 1
            except (KeyError, TypeError, ValueError) as error:
                unknown.append({"ordinal": ordinal, "name": source.get("name"), "base_name": base_name,
                                "reason": str(error)})
        dense_bytes, expert_bytes = (union_bytes(intervals[key]) for key in ("dense", "expert"))
        all_intervals = intervals["dense"]+intervals["expert"]
        unique = union_bytes(all_intervals)
        incomplete_dense = sorted(name for name in expected_dense if name in by_name and
                                  union_bytes(by_name[name]) != tensors[name]["payload_bytes"])
        missing_dense = sorted(expected_dense-names["dense"])
        missing_expert = sorted(expected_expert-names["expert"])
        unexpected = sorted((names["dense"]-expected_dense) | (names["expert"]-expected_expert))
        closed = not (unknown or missing_dense or missing_expert or incomplete_dense or unexpected)
        reports.append({"step": step, "phase": "prefill" if step in prefills else "decode",
                        "source_operations": len(groups[step]),
                        "dense_operations": counts["dense"], "expert_operations": counts["expert"],
                        "distinct_dense_names": len(names["dense"]), "distinct_expert_bank_names": len(names["expert"]),
                        "dense_source_names_sha256": hashlib.sha256("\n".join(sorted(names["dense"])).encode()).hexdigest(),
                        "logical_dense_operand_bytes": logical["dense"], "logical_selected_expert_operand_bytes": logical["expert"],
                        "distinct_dense_operand_bytes": dense_bytes, "distinct_selected_expert_operand_bytes": expert_bytes,
                        "distinct_known_matmul_operand_bytes": unique,
                        "dense_expert_overlap_bytes": dense_bytes+expert_bytes-unique,
                        "repeated_known_operand_bytes": sum(logical.values())-unique,
                        "all_expected_fixed_matrices_and_expert_banks_resolved": closed,
                        "source_buffer_usage_available_for_all_rows": source_usage_available,
                        "unknown_sources": unknown, "explicit_nonweight_sources_excluded": excluded,
                        "missing_dense_names": missing_dense, "missing_expert_banks": missing_expert,
                        "incomplete_dense_payload_names": incomplete_dense, "unexpected_weight_sources": unexpected,
                        "file_intervals": merge_intervals(all_intervals)})
    decode = [row for row in reports if row["phase"] == "decode"]
    return {"schema_version": 1, "scope": "distinct declared original weight operands of this captured MUL_MAT/MUL_MAT_ID graph",
            "model_revision": inventory.get("revision"),
            "prefill_step_ids": sorted(prefills),
            "expected_dense_matrices": len(expected_dense), "expected_expert_banks": len(expected_expert),
            "all_steps_closed": bool(reports) and all(r["all_expected_fixed_matrices_and_expert_banks_resolved"] for r in reports),
            "decode_steps_closed": bool(decode) and all(r["all_expected_fixed_matrices_and_expert_banks_resolved"] for r in decode),
            "per_step": reports,
            "decode_distinct_known_bytes_range": [min(r["distinct_known_matmul_operand_bytes"] for r in decode),
                                                  max(r["distinct_known_matmul_operand_bytes"] for r in decode)] if decode else None,
            "physical_ssd_bytes_measured": False, "physical_nand_bytes_measured": False,
            "file_cache_residency_measured": False, "model_payload_read_by_tool": False,
            "limitations": [
                "Header-relative offsets are converted using the GGUF data offset; padding is excluded; unions are file-qualified.",
                "MUL_MAT_ID intervals use only same-step same-layer canonical ordered route IDs; deduplication is set accounting, not rerouting.",
                "Unknown views remain unknown; known ranges still give a partial known-operand lower bound when coverage is incomplete.",
                "The expected fixed-matrix set comes from the owning graph/inventory reconciliation; complete capture and exact name/layout matches close the previous size-multiset identity gap.",
                "This proves declared operand ranges for the implemented graph, not CPU load instruction traces or a mathematical minimum for every possible algorithm.",
                "Nonmatmul weights, embeddings/norms and other model state are outside this byte total, so it is not a total-memory or all-model-IO measure.",
                "Distinct required operand bytes may be resident or compressed; they are not automatically SSD interface transfers or NAND reads.",
            ]}


def bounded_json_lines(path, *, max_bytes=64*1024**2):
    if path.stat().st_size > max_bytes:
        raise ValueError("metadata exceeds parsing budget")
    rows = []
    with path.open() as stream:
        for line in stream:
            if len(line) > 65536:
                raise ValueError("metadata row exceeds parsing budget")
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) > 200000:
                    raise ValueError("metadata row count exceeds parsing budget")
    return rows


def read_closed_capture(root):
    root = Path(root)
    complete_path = safe_file(root, "complete.json")
    if complete_path.stat().st_size > 65536:
        raise ValueError("capture completion metadata exceeds parsing budget")
    complete = json.loads(complete_path.read_text())
    tokens_path = safe_file(root, "tokens.jsonl")
    tokens = bounded_json_lines(tokens_path, max_bytes=1024**2)
    if (complete.get("completed") is not True or type(complete.get("captured_steps")) is not int or not tokens or
            complete.get("captured_steps") != len(tokens) or
            any(type(row.get("step")) is not int for row in tokens) or
            [row.get("step") for row in tokens] != list(range(len(tokens)))):
        raise ValueError("source union requires a closed successful capture")
    requests_present = ["request" in row for row in tokens]
    if any(requests_present) and not all(requests_present):
        raise ValueError("ambiguous partially recorded request boundaries")
    prefills = [0]
    last_request = 0
    for index, row in enumerate(tokens):
        request = integer(row.get("request", 0), "request index")
        if request not in (last_request, last_request+1) or (index == 0 and request != 0):
            raise ValueError("noncontiguous saved request boundaries")
        if request != last_request:
            prefills.append(index)
            last_request = request
    manifest = safe_file(root, "tensors.jsonl")
    source_path = safe_file(root, "source_plan.jsonl")
    rows = bounded_json_lines(manifest, max_bytes=16*1024**2)
    routes, hashes, route_order = {}, {}, []
    bytes_read = 0
    for row in rows:
        match = ROUTE_NAME.fullmatch(row.get("name", ""))
        if not match:
            continue
        step, layer = row.get("step"), int(match[1])
        if type(step) is not int or not 0 <= step < len(tokens) or not 3 <= layer < 45:
            raise ValueError("canonical route outside capture/model layers")
        key = (step, layer)
        if key in routes:
            raise ValueError("ambiguous duplicate canonical route")
        shape = row.get("shape", [])
        if (row.get("type") != 26 or len(shape) != 4 or shape[0] != 8 or shape[2:] != [1, 1] or
                type(shape[1]) is not int or not 1 <= shape[1] <= 512 or
                type(row.get("bytes")) is not int or not 0 < row["bytes"] <= 65536):
            raise ValueError("invalid bounded canonical top-8 tensor")
        payload = safe_file(root, row["file"])
        bytes_read += payload.stat().st_size
        if bytes_read > 1024**2:
            raise ValueError("saved route payload read ceiling exceeded")
        view = tensor_view(root, row)
        columns = [[int(e) for e in view[:, column, 0, 0]] for column in range(shape[1])]
        routes[key] = columns
        hashes[row["file"]] = hashlib.sha256(payload.read_bytes()).hexdigest()
        route_order.append({"step": step, "layer": layer, "ordered_ids_by_column": columns})
    source_rows = bounded_json_lines(source_path)
    run_path = safe_file(root, "run.json")
    if run_path.stat().st_size > 65536 or complete_path.stat().st_size > 65536:
        raise ValueError("capture completion/run metadata exceeds parsing budget")
    metadata_files = (complete_path, tokens_path, manifest, source_path, run_path)
    provenance = {"capture_name": root.name,
                  "prefill_step_ids": prefills,
                  "phase_basis": "saved token request boundaries" if all(requests_present) else "legacy single-request capture",
                  "metadata_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata_files},
                  "saved_route_payload_sha256": hashes, "saved_route_payload_bytes_read": bytes_read,
                  "canonical_routes_sha256": hashlib.sha256(json.dumps(sorted(route_order, key=lambda x:(x['step'],x['layer'])), sort_keys=True).encode()).hexdigest()}
    return source_rows, routes, len(tokens), provenance


def analyze(capture, inventory_path):
    inventory_path = Path(inventory_path)
    if inventory_path.stat().st_size > 16*1024**2:
        raise ValueError("GGUF header inventory exceeds parsing budget")
    inventory = json.loads(inventory_path.read_text())
    source_rows, routes, steps, provenance = read_closed_capture(capture)
    result = prove_rows(inventory, source_rows, routes, captured_steps=steps,
                        prefill_steps=provenance["prefill_step_ids"])
    result["source"] = {**provenance, "inventory_file": inventory_path.name,
                        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "expected_matrix_selection_code_sha256": hashlib.sha256(Path(__file__).with_name("dense_inventory.py").read_bytes()).hexdigest()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refusing to overwrite previous evidence")
    report = analyze(args.capture, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("all_steps_closed", "decode_steps_closed", "decode_distinct_known_bytes_range")}))
    return 0 if report["all_steps_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
