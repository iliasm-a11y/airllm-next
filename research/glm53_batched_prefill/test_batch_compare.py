import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from research.glm53_batched_prefill.batch_compare import (
    compare, _token_compare, _checkpoints, _checkpoint_replay, _dependencies,
)
from research.glm53_batched_prefill.native_compare import EXPECTED_NAMES, EXPECTED_STATE_TARGETS
from research.glm53_batched_prefill.snapshot_key import canonical_key
from research.glm53_batched_prefill.test_snapshot import manifest
from research.glm53_batched_prefill.test_native_compare import guard, add_continuation


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def tensor(root, label, request, offset, take, batch_index):
    token = np.arange(offset, offset + take)
    if label == "routes":
        data = (np.arange(2)[:, None] + token[None, :] % 11).astype("<i4")[:, :, None, None]
        dtype = 26
    elif label == "router":
        data = (np.arange(3)[:, None] + token[None, :] / 1024).astype("<f4")[:, :, None, None]
        dtype = 0
    else:
        data = (np.arange(4).reshape((2, 2), order="F")[:, :, None] + token[None, None, :] / 1024).astype("<f4")[:, :, :, None]
        dtype = 0
    data = np.asfortranarray(data)
    raw = data.tobytes(order="F")
    filename = f"{request}-{offset}-{label}.bin"
    (root / filename).write_bytes(raw)
    return {"request": request, "offset": offset, "input_tokens": take, "batch_index": batch_index,
            "type": dtype, "shape": list(data.shape), "strides": list(data.strides),
            "bytes": len(raw), "file": filename, "sha256": sha(raw)}


def capture(root, ub):
    root.mkdir()
    model = manifest()
    model["state_parameters"].update(n_batch=ub, n_ubatch=ub)
    key, _ = canonical_key(model)
    receipt = guard()
    receipt["preflight"] = {"snapshot_manifest": model, "snapshot_key_sha256": key}
    logits = struct.pack("<4f", 1, 2, 3, 4)
    requests, tensors, batches, writes, checkpoints = [], [], [], [], []
    for request, agent in enumerate((1, 600, 1)):
        state = f"fixture-final-agent-{agent}".encode()
        (root / f"logits-{request}.f32").write_bytes(logits)
        (root / f"final-state-{request}.bin").write_bytes(state)
        requests.append({"request": request, "agent_id": agent, "prompt_tokens": 216, "next_position": 216,
                         "prompt_ids_sha256": sha(struct.pack("<216i", *(list(range(208)) + [agent] * 8))),
                         "logits_sha256": sha(logits), "sequence_bytes": len(state), "sequence_sha256": sha(state)})
        for index, offset in enumerate(range(0, 216, ub)):
            take = min(ub, 216 - offset)
            batches.append({"request": request, "offset": offset, "tokens": take, "returncode": 0})
            by_family = {label: tensor(root, label, request, offset, take, index) for label in ("routes", "router", "kda")}
            for name in sorted(EXPECTED_NAMES):
                label = "routes" if name.startswith("ffn_moe_topk-") else "router" if name.startswith("ffn_moe_logits-") else "kda"
                tensors.append({**by_family[label], "name": name})
            for target in sorted(EXPECTED_STATE_TARGETS):
                writes.append({"request": request, "offset": offset, "target": target, "op": "CPY", "type": 0,
                               "shape": [1, 1, 1, 1], "bytes": 4, "logical_bytes": 4,
                               "status": "hashed", "finite": True, "sha256": sha(struct.pack("<f", offset / 1024))})
        for end in (128, 216):
            raw = state if end == 216 else f"fixture-checkpoint-agent-{agent}-{end}".encode()
            checkpoints.append({"request": request, "token_end": end, "bytes": len(raw), "sha256": sha(raw),
                                "state_seq_flags": 0, "status": "hashed", "next_position": end})
    report = {"complete": True, "diagnostic": True, "model_loads": 1, "physical_slots": 1, "gpu_layers": 0,
              "threads": 10, "generation_performed": False, "requests": requests, "vocab_size": 4, "mode": "full",
              "microbatch": ub, "logical_batch": ub, "snapshot_key_sha256": key, "state_parameters": model["state_parameters"]}
    add_continuation(root, report)
    (root / "report.json").write_text(json.dumps(report))
    lines(root / "tensors.jsonl", tensors)
    lines(root / "batches.jsonl", batches)
    lines(root / "state_hashes.jsonl", writes)
    lines(root / "state_checkpoints.jsonl", checkpoints)
    return receipt


def test_realistic_cross_batch_fixture_stitches_tokens_and_common_states(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ag, bg = capture(a, 8), capture(b, 16)
    result = compare(a, b, ag, bg)
    assert result["passed"] is True, result["errors"]
    assert result["coverage"]["routes"] > 0 and result["coverage"]["kda"] > 0
    assert result["reference_microbatch"] == 8 and result["candidate_microbatch"] == 16
    assert "not asserted" in result["cross_batch_write_event_parity"]
    assert result["observed_write_coverage"]["reference"]["reference_events"] != result["observed_write_coverage"]["candidate"]["reference_events"]


@pytest.mark.parametrize("label,name", [("routes", "ffn_moe_topk-3"), ("router", "ffn_moe_logits-3"), ("kda", "kda_scan_out-0")])
def test_token_axis_stitching_is_exact_for_different_chunks_and_partial_tail(tmp_path, label, name):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    a_rows = [{**tensor(a, label, 0, offset, min(8, 17 - offset), index), "name": name}
              for index, offset in enumerate(range(0, 17, 8))]
    b_rows = [{**tensor(b, label, 0, offset, min(16, 17 - offset), index), "name": name}
              for index, offset in enumerate(range(0, 17, 16))]
    check = _token_compare(a, b, a_rows, b_rows, name)
    assert check["passed"] is True and check["tokens"] == 17


@pytest.mark.parametrize("fault", ["route_reorder", "router_delta", "kda_delta", "kda_nan", "token_gap"])
def test_stitched_token_comparison_rejects_order_numeric_and_position_faults(tmp_path, fault):
    label, name = ("routes", "ffn_moe_topk-3") if fault == "route_reorder" else ("router", "ffn_moe_logits-3") if fault == "router_delta" else ("kda", "kda_scan_out-0")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    ar = [{**tensor(a, label, 0, n, 8, i), "name": name} for i, n in enumerate((0, 8))]
    br = [{**tensor(b, label, 0, 0, 16, 0), "name": name}]
    if fault == "token_gap":
        ar[1]["offset"] = 9
        with pytest.raises(ValueError):
            _token_compare(a, b, ar, br, name)
        return
    path = b / br[0]["file"]
    dtype = "<i4" if label == "routes" else "<f4"
    values = np.frombuffer(path.read_bytes(), dtype=dtype).copy()
    if fault == "route_reorder":
        values[0], values[1] = values[1], values[0]
    elif fault == "kda_nan":
        values[0] = np.nan
    else:
        values[0] += .01
    path.write_bytes(values.tobytes())
    result = _token_compare(a, b, ar, br, name)
    assert result["passed"] is False and result["first_token_divergence"] == 0


def test_dependency_normalization_excludes_only_batch_parameters():
    def record(ub):
        model = manifest(); model["state_parameters"].update(n_batch=ub, n_ubatch=ub)
        key, _ = canonical_key(model)
        return {"microbatch": ub, "logical_batch": ub, "snapshot_key_sha256": key, "state_parameters": model["state_parameters"]}, {"preflight": {"snapshot_manifest": model, "snapshot_key_sha256": key}}
    a, ag = record(8); b, bg = record(16)
    assert _dependencies(a, ag) == _dependencies(b, bg)
    changed = copy.deepcopy(bg["preflight"]["snapshot_manifest"])
    changed["state_parameters"]["rope_freq_scale"] = 2
    key, _ = canonical_key(changed)
    bg["preflight"].update(snapshot_manifest=changed, snapshot_key_sha256=key)
    b.update(snapshot_key_sha256=key, state_parameters=changed["state_parameters"])
    assert _dependencies(a, ag) != _dependencies(b, bg)


@pytest.mark.parametrize("fault", ["missing_boundary", "partial_flags", "different_final", "duplicate", "different_replay"])
def test_common_checkpoint_coverage_and_same_agent_replay_are_required(tmp_path, fault):
    root = tmp_path / "a"
    capture(root, 16)
    report = json.loads((root / "report.json").read_text())
    rows = [json.loads(r) for r in (root / "state_checkpoints.jsonl").read_text().splitlines()]
    if fault == "missing_boundary":
        rows = [r for r in rows if r["token_end"] != 128]
    elif fault == "partial_flags":
        rows[0]["state_seq_flags"] = 1
    elif fault == "different_final":
        rows[1]["sha256"] = "f" * 64
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "different_replay":
        rows[-2]["sha256"] = "f" * 64
    lines(root / "state_checkpoints.jsonl", rows)
    if fault == "different_replay":
        assert _checkpoint_replay(_checkpoints(root, report)) is False
    else:
        with pytest.raises(ValueError):
            _checkpoints(root, report)


def test_optional208_checkpoint_does_not_replace_required_common_boundaries(tmp_path):
    root = tmp_path / "a"
    capture(root, 16)
    report = json.loads((root / "report.json").read_text())
    rows = [json.loads(r) for r in (root / "state_checkpoints.jsonl").read_text().splitlines()]
    rows += [{"request": r, "token_end": 208, "bytes": 32, "sha256": "a" * 64,
              "state_seq_flags": 0, "status": "hashed", "next_position": 208} for r in range(3)]
    lines(root / "state_checkpoints.jsonl", rows)
    assert len(_checkpoints(root, report)) == 9
    rows = [r for r in rows if r["token_end"] != 128]
    lines(root / "state_checkpoints.jsonl", rows)
    with pytest.raises(ValueError):
        _checkpoints(root, report)
