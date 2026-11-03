import hashlib
import json
import struct
from pathlib import Path

import pytest

from research.glm53_batched_prefill.native_compare import (
    compare, EXPECTED_NAMES, EXPECTED_STATE_TARGETS, _guard, _read_writes, _compare_writes,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def guard():
    return {"status": "completed", "returncode": 0, "guard_violation": None,
            "power_context_passed": True, "integrity_passed": True,
            "limits": {"rss_bytes": 8.5 * 1024 ** 3, "swap_growth_bytes": 32 * 1024 ** 2},
            "sampling_interval_seconds": .5,
            "samples": [{"rss_bytes": 1024, "swap_growth_bytes": 0, "pressure_level": 1}]}


def write_capture(root: Path, mode="full"):
    root.mkdir()
    logits = struct.pack("<4f", 1, 2, 3, 4)
    state = b"native-fixture-sequence-memory"
    router = struct.pack("<f", .125)
    route = struct.pack("<i", 7)
    (root / "float.bin").write_bytes(router)
    (root / "int.bin").write_bytes(route)
    requests = []
    for index, agent in enumerate((1, 600, 1)):
        (root / f"logits-{index}.f32").write_bytes(logits)
        (root / f"final-state-{index}.bin").write_bytes(state)
        requests.append({"request": index, "agent_id": agent, "prompt_tokens": 216,
                         "next_position": 216, "prompt_ids_sha256": "a" * 64,
                         "logits_sha256": sha(logits), "sequence_bytes": len(state),
                         "sequence_sha256": sha(state)})
    report = {"complete": True, "diagnostic": True, "model_loads": 1, "physical_slots": 1,
              "gpu_layers": 0, "threads": 10, "generation_performed": False, "requests": requests,
              "vocab_size": 4, "mode": mode, "microbatch": 16, "logical_batch": 16,
              "snapshot_key_sha256": "e" * 64,
              "snapshot": {"successful_restores": 3, "bytes": 32}}
    (root / "report.json").write_text(json.dumps(report))
    batches, tensors, writes = [], [], []
    for request in range(3):
        first = 208 if mode == "restore" and request else 0
        for offset in range(first, 216, 16):
            batches.append({"request": request, "offset": offset, "tokens": min(16, 216 - offset), "returncode": 0})
            for target in sorted(EXPECTED_STATE_TARGETS):
                writes.append({"request": request, "offset": offset, "target": target,
                               "op": "CPY", "type": 0, "shape": [1, 1, 1, 1],
                               "bytes": 4, "logical_bytes": 4, "status": "hashed",
                               "finite": True, "sha256": sha(router)})
            for name in sorted(EXPECTED_NAMES):
                integer = name.startswith("ffn_moe_topk-")
                tensors.append({"request": request, "offset": offset, "name": name,
                                "type": 26 if integer else 0, "shape": [1, 1, 1, 1],
                                "strides": [4, 4, 4, 4], "bytes": 4,
                                "file": "int.bin" if integer else "float.bin",
                                "sha256": sha(route if integer else router)})
    (root / "batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in batches))
    (root / "tensors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tensors))
    (root / "state_hashes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in writes))
    return root


def edit_report(root, fn):
    path = root / "report.json"
    record = json.loads(path.read_text())
    fn(record)
    path.write_text(json.dumps(record))


@pytest.fixture
def pair(tmp_path):
    return write_capture(tmp_path / "a"), write_capture(tmp_path / "b", "split")


def test_full_split_exact_finite_complete_capture_passes(pair):
    result = compare(*pair, guard(), guard())
    assert result["passed"] is True
    assert result["coverage"]["routes"] > 0
    assert len(result["logits"]) == 3
    assert "no EOS" in result["scope"]


def test_split_restore_compares_suffixes_and_all_final_states(tmp_path):
    a = write_capture(tmp_path / "a", "split")
    b = write_capture(tmp_path / "b", "restore")
    result = compare(a, b, guard(), guard())
    assert result["passed"] is True
    assert result["comparison"] == "split_restore"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(complete=False),
    lambda r: r.update(diagnostic=False),
    lambda r: r.update(model_loads=2),
    lambda r: r.update(physical_slots=2),
    lambda r: r.update(snapshot_key_sha256="f" * 64),
    lambda r: r["requests"][1].update(prompt_ids_sha256="f" * 64),
    lambda r: r.update(vocab_size=3),
])
def test_incomplete_or_different_native_run_refused(pair, mutation):
    edit_report(pair[1], mutation)
    assert compare(*pair, guard(), guard())["passed"] is False


@pytest.mark.parametrize("mutation", [
    lambda g: g.update(returncode=-9),
    lambda g: g.update(guard_violation="system_swap_growth_limit"),
    lambda g: g.update(power_context_passed=False),
    lambda g: g.update(integrity_passed=False),
    lambda g: g.update(samples=[]),
    lambda g: g["limits"].update(swap_growth_bytes=64 * 1024 ** 2),
    lambda g: g["samples"][0].update(pressure_level=2),
])
def test_missing_or_failed_guards_refused(pair, mutation):
    receipt = guard()
    mutation(receipt)
    assert compare(*pair, guard(), receipt)["passed"] is False


def test_identical_nans_do_not_pass(pair):
    for root in pair:
        data = struct.pack("<4f", float("nan"), 2, 3, 4)
        (root / "logits-0.f32").write_bytes(data)
        edit_report(root, lambda r: r["requests"][0].update(logits_sha256=sha(data)))
    result = compare(*pair, guard(), guard())
    assert result["passed"] is False
    assert result["logits"][0]["finite"] is False


def test_snapshot_or_logit_hash_cannot_replace_raw_payload(pair):
    (pair[1] / "final-state-0.bin").unlink()
    assert compare(*pair, guard(), guard())["passed"] is False


def test_both_sides_missing_same_layer_cannot_pass(pair):
    for root in pair:
        path = root / "tensors.jsonl"
        rows = [json.loads(r) for r in path.read_text().splitlines()]
        rows = [r for r in rows if r["name"] != "ffn_moe_topk-44"]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = compare(*pair, guard(), guard())
    assert result["passed"] is False
    assert "layer captures" in result["errors"][0]


def test_missing_microbatch_even_on_both_sides_cannot_pass(pair):
    for root in pair:
        path = root / "batches.jsonl"
        rows = [json.loads(r) for r in path.read_text().splitlines()]
        rows = [r for r in rows if r["offset"] != 16]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert compare(*pair, guard(), guard())["passed"] is False


def test_corrupt_capture_cannot_escape_to_parent_path(pair):
    path = pair[1] / "tensors.jsonl"
    rows = [json.loads(r) for r in path.read_text().splitlines()]
    rows[0]["file"] = "../escape.bin"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert compare(*pair, guard(), guard())["passed"] is False


def test_nonfinite_resource_receipts_are_not_clear_guards():
    receipt = guard()
    receipt["samples"][0]["rss_bytes"] = float("nan")
    with pytest.raises(ValueError):
        _guard(receipt)


def write_events(root, rows):
    root.mkdir(exist_ok=True)
    (root / "state_hashes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def events(request=0, offset=208):
    return [{"request": request, "offset": offset, "target": target, "op": "CPY",
             "type": 0, "shape": [1, 1, 1, 1], "bytes": 4, "logical_bytes": 4,
             "status": "hashed", "finite": True, "sha256": "a" * 64}
            for target in sorted(EXPECTED_STATE_TARGETS)]


@pytest.mark.parametrize("fault", ["missing", "empty", "skipped", "nonfinite", "bad_hash", "bad_length", "invalid_target"])
def test_write_reader_rejects_incomplete_or_invalid_events(tmp_path, fault):
    rows = events()
    if fault == "empty":
        rows = []
    elif fault == "skipped":
        rows[0]["status"] = "skipped"
    elif fault == "nonfinite":
        rows[0]["finite"] = False
    elif fault == "bad_hash":
        rows[0]["sha256"] = "unverified"
    elif fault == "bad_length":
        rows[0]["logical_bytes"] = 8
    elif fault == "invalid_target":
        rows[0]["target"] = "cache_unknown_l0"
    if fault != "missing":
        write_events(tmp_path, rows)
    with pytest.raises(ValueError):
        _read_writes(tmp_path)


@pytest.mark.parametrize("fault", ["hash_changed", "order_changed", "one_missing", "same_missing", "batch_missing"])
def test_observed_write_parity_requires_order_hash_and_all_layer_targets(tmp_path, fault):
    a, b = events(), events()
    if fault == "hash_changed":
        b[0]["sha256"] = "b" * 64
    elif fault == "order_changed":
        b[0], b[1] = b[1], b[0]
    elif fault == "one_missing":
        b.pop()
    elif fault == "same_missing":
        a.pop()
        b.pop()
    elif fault == "batch_missing":
        a = b = []
    write_events(tmp_path / "a", a)
    write_events(tmp_path / "b", b)
    if fault == "batch_missing":
        with pytest.raises(ValueError):
            _compare_writes([], [], [(0, 208)], False)
        return
    a, b = _read_writes(tmp_path / "a"), _read_writes(tmp_path / "b")
    if fault in ("same_missing", "one_missing"):
        with pytest.raises(ValueError):
            _compare_writes(a, b, [(0, 208)], False)
    else:
        assert _compare_writes(a, b, [(0, 208)], False)["passed"] is False


def test_observed_write_restore_ignores_only_unrecomputed_common_prefix(tmp_path):
    a, b = events(offset=0) + events(offset=208), events(offset=208)
    write_events(tmp_path / "a", a)
    write_events(tmp_path / "b", b)
    a, b = _read_writes(tmp_path / "a"), _read_writes(tmp_path / "b")
    result = _compare_writes(a, b, [(0, 208)], True)
    assert result["passed"] is True and result["reference_events"] == 90


@pytest.mark.parametrize("fault", ["logits", "sequence", "tensor", "writes"])
def test_same_cross_arm_reset_defect_cannot_qualify(pair, fault):
    for root in pair:
        if fault == "logits":
            data = struct.pack("<4f", 9, 2, 3, 4)
            (root / "logits-2.f32").write_bytes(data)
            edit_report(root, lambda r: r["requests"][2].update(logits_sha256=sha(data)))
        elif fault == "sequence":
            data = b"fixture-leaked-state-after-agent600"
            (root / "final-state-2.bin").write_bytes(data)
            edit_report(root, lambda r: r["requests"][2].update(sequence_sha256=sha(data), sequence_bytes=len(data)))
        elif fault == "tensor":
            data = struct.pack("<f", .25)
            (root / "changed-float.bin").write_bytes(data)
            path = root / "tensors.jsonl"
            rows = [json.loads(r) for r in path.read_text().splitlines()]
            for row in rows:
                if row["request"] == 2 and row["name"] == "kda_scan_out-0":
                    row.update(file="changed-float.bin", sha256=sha(data))
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        elif fault == "writes":
            path = root / "state_hashes.jsonl"
            rows = [json.loads(r) for r in path.read_text().splitlines()]
            for row in rows:
                if row["request"] == 2 and row["target"] == "cache_s_l0":
                    row["sha256"] = "b" * 64
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = compare(*pair, guard(), guard())
    assert result["passed"] is False
    assert result["same_agent_replays"]["reference"]["passed"] is False
    assert result["same_agent_replays"]["candidate"]["passed"] is False
