import hashlib
import json
import struct
from pathlib import Path

import pytest

from research.glm53_batched_prefill.native_compare import (
    compare, EXPECTED_NAMES, EXPECTED_STATE_TARGETS, _guard, _read_writes, _compare_writes,
    _read_continuation, _compare_continuation,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def guard():
    from air_llm.airllm.nextgen.glm_native import NativeGlmModel, NativeLimits
    native = vars(NativeLimits(max_rss_bytes=int(8.5 * 1024**3),
        max_swap_growth_bytes=32 * 1024**2, warning_swap_growth_bytes=32 * 1024**2,
        request_timeout_seconds=2700))
    policy = Path(NativeGlmModel._guard_resource_sample.__code__.co_filename).resolve()
    baseline = {"monotonic": 0.0, "available_bytes": 4 * 1024**3,
                "swap_used_bytes": 100 * 1024**2, "pressure_level": 1}
    return {"status": "completed", "returncode": 0, "guard_violation": None,
            "power_context_passed": True, "integrity_passed": True,
            "limits": {"rss_bytes": native["max_rss_bytes"],
                       "swap_growth_bytes": native["max_swap_growth_bytes"], "native": native},
            "sampling_interval_seconds": .5, "baseline": baseline,
            "admission": {"passed": True, "sample": dict(baseline), "required_bytes_estimate": 2 * 1024**3},
            "preflight": {"metadata": {"sha256": {str(policy): sha(policy.read_bytes())}}},
            "samples": [{**baseline, "monotonic": .5, "rss_bytes": 1024, "swap_growth_bytes": 0}]}

def add_continuation(root, report, *, steps=4, eos=False):
    logits = struct.pack("<4f", 1, 2, 3, 4)
    scalar, route = struct.pack("<f", .125), struct.pack("<i", 7)
    for label, data in (("float", scalar), ("int", route), ("output", logits)):
        (root / ("continuation-fixture-" + label + ".bin")).write_bytes(data)
    tensors, writes = [], []
    for request, row in enumerate(report["requests"]):
        samples = []
        for step in range(steps):
            (root / f"continuation-logits-{request}-{step}.f32").write_bytes(logits)
            samples.append({"step": step, "token": 3, "native_eog": bool(eos and step == steps - 1), "logits_sha256": sha(logits)})
            if not step:
                continue
            for name in sorted(EXPECTED_NAMES | {"result_output"}):
                integer, output = name.startswith("ffn_moe_topk-"), name == "result_output"
                data = logits if output else route if integer else scalar
                label = "output" if output else "int" if integer else "float"
                tensors.append({"request": request, "offset": step, "phase": "continuation", "input_tokens": 1,
                    "batch_index": step, "name": name, "type": 26 if integer else 0,
                    "shape": [4 if output else 1, 1, 1, 1], "strides": [4, 16, 16, 16] if output else [4, 4, 4, 4],
                    "bytes": len(data), "sha256": sha(data), "file": "continuation-fixture-" + label + ".bin"})
            for target in sorted(EXPECTED_STATE_TARGETS):
                writes.append({"request": request, "offset": step, "phase": "continuation", "input_tokens": 1,
                    "batch_index": step, "target": target, "op": "CPY", "type": 0, "shape": [1, 1, 1, 1],
                    "bytes": 4, "logical_bytes": 4, "status": "hashed", "finite": True, "sha256": sha(scalar)})
        state = f"continuation-fixture-agent-{row['agent_id']}".encode()
        (root / f"continuation-state-{request}.bin").write_bytes(state)
        row.update(continuation=samples, continuation_complete=True, continuation_stop="eos" if eos else "diagnostic_budget",
                   continuation_state_bytes=len(state), continuation_state_sha256=sha(state),
                   continuation_next_position=row["prompt_tokens"] + steps - 1)
    report.update(generation_performed=True, functional_agents_qualified=False,
                  continuation_policy="fresh native greedy sampler; max4; diagnostic only")
    (root / "continuation_tensors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tensors))
    (root / "continuation_states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in writes))


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
    add_continuation(root, report)
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
    lambda g: g["samples"][0].update(pressure_level=4),
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
    assert result["errors"]


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
            (root / "continuation-logits-2-0.f32").write_bytes(data)
            def change(record):
                row = record["requests"][2]
                row["logits_sha256"] = sha(data)
                row["continuation"][0].update(logits_sha256=sha(data), token=0)
            edit_report(root, change)
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


def continuation_capture(root, *, steps=4, eos=False):
    root.mkdir(exist_ok=True)
    logits = struct.pack("<4f", 1, 2, 3, 4)
    report = {"vocab_size": 4, "requests": []}
    for request, agent in enumerate((1, 600, 1)):
        (root / f"logits-{request}.f32").write_bytes(logits)
        report["requests"].append({"agent_id": agent, "logits_sha256": sha(logits), "prompt_tokens": 216})
    add_continuation(root, report, steps=steps, eos=eos)
    return report


@pytest.mark.parametrize("steps,eos", [(4, False), (1, True), (3, True), (4, True)])
def test_native_greedy_diagnostic_accepts_four_or_natural_eos_without_functional_claim(tmp_path, steps, eos):
    report = continuation_capture(tmp_path, steps=steps, eos=eos)
    captured = _read_continuation(tmp_path, report)
    result = _compare_continuation(tmp_path, tmp_path, report, report, captured, captured)
    assert result["passed"] is True
    assert "no useful EOS" in result["scope"]


@pytest.mark.parametrize("fault", ["incomplete", "short_budget", "wrong_stop", "wrong_step", "nongreedy", "early_eos",
                                   "missing_logits", "truncated_logits", "nan_logits", "missing_post_state",
                                   "missing_output", "missing_kda", "missing_state_target", "skipped_state"])
def test_greedy_continuation_failures_never_qualify(tmp_path, fault):
    report = continuation_capture(tmp_path)
    row = report["requests"][0]
    if fault == "incomplete":
        row["continuation_complete"] = False
    elif fault == "short_budget":
        row["continuation"].pop()
    elif fault == "wrong_stop":
        row["continuation_stop"] = "limit"
    elif fault == "wrong_step":
        row["continuation"][1]["step"] = 2
    elif fault == "nongreedy":
        row["continuation"][1]["token"] = 1
    elif fault == "early_eos":
        row["continuation"][1]["native_eog"] = True
    elif fault == "missing_logits":
        (tmp_path / "continuation-logits-0-1.f32").unlink()
    elif fault == "truncated_logits":
        (tmp_path / "continuation-logits-0-1.f32").write_bytes(b"short")
    elif fault == "nan_logits":
        data = struct.pack("<4f", float("nan"), 2, 3, 4)
        (tmp_path / "continuation-logits-0-1.f32").write_bytes(data)
        row["continuation"][1]["logits_sha256"] = sha(data)
    elif fault == "missing_post_state":
        (tmp_path / "continuation-state-0.bin").unlink()
    elif fault in ("missing_output", "missing_kda"):
        path = tmp_path / "continuation_tensors.jsonl"
        rows = [json.loads(r) for r in path.read_text().splitlines()]
        name = "result_output" if fault == "missing_output" else "kda_scan_out-0"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["name"] != name))
    elif fault in ("missing_state_target", "skipped_state"):
        path = tmp_path / "continuation_states.jsonl"
        rows = [json.loads(r) for r in path.read_text().splitlines()]
        if fault == "skipped_state":
            rows[0]["status"] = "skipped"
        else:
            rows = [r for r in rows if r["target"] != "cache_s_l0"]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        _read_continuation(tmp_path, report)


def test_continuation_post_state_parity_and_intra_agent_replay_are_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ar, br = continuation_capture(a), continuation_capture(b)
    for root, report in ((a, ar), (b, br)):
        data = b"same-leaked-continuation-after-agent600"
        (root / "continuation-state-2.bin").write_bytes(data)
        report["requests"][2].update(continuation_state_bytes=len(data), continuation_state_sha256=sha(data))
    ac, bc = _read_continuation(a, ar), _read_continuation(b, br)
    assert _compare_continuation(a, b, ar, br, ac, bc)["passed"] is True
    assert _compare_continuation(a, a, ar, ar, ac, ac, ((0, 2),))["passed"] is False
    assert _compare_continuation(b, b, br, br, bc, bc, ((0, 2),))["passed"] is False


def direct_restore_pair(tmp_path):
    a = write_capture(tmp_path / "a", "full")
    b = write_capture(tmp_path / "b", "restore")
    data = b"*" * 32
    rows = [{"request": r, "token_end": 208, "status": "hashed", "state_seq_flags": 0,
             "next_position": 208, "bytes": len(data), "sha256": sha(data)} for r in range(3)]
    (a / "state_checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    edit_report(b, lambda report: report["snapshot"].update(sha256=sha(data)))
    return a, b


def test_full_restore_requires_exact_common_boundary_and_continuation(tmp_path):
    a, b = direct_restore_pair(tmp_path)
    result = compare(a, b, guard(), guard())
    assert result["passed"] is True, result["errors"]
    assert result["direct_restore_boundary"]["reference_checkpoints"] == 3
    assert result["continuation"]["passed"] is True


@pytest.mark.parametrize("fault", ["unaligned", "missing208", "different_snapshot", "different_common_state"])
def test_full_restore_alignment_alone_does_not_qualify(tmp_path, fault):
    a, b = direct_restore_pair(tmp_path)
    if fault == "unaligned":
        for root in (a, b):
            edit_report(root, lambda report: report.update(microbatch=32, logical_batch=32))
    elif fault == "missing208":
        (a / "state_checkpoints.jsonl").unlink()
    elif fault == "different_snapshot":
        edit_report(b, lambda report: report["snapshot"].update(sha256="f" * 64))
    else:
        path = a / "state_checkpoints.jsonl"
        rows = [json.loads(r) for r in path.read_text().splitlines()]
        rows[2]["sha256"] = "f" * 64
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert compare(a, b, guard(), guard())["passed"] is False


def test_contextual_warning_clear_for_long_interval_can_pass():
    receipt = guard()
    first = {**receipt["samples"][0], "pressure_level": 2, "available_bytes": 3 * 1024**3}
    receipt["samples"] = [{**first, "monotonic": t} for t in (.5, 25.5, 2700.5)]
    assert _guard(receipt) is receipt


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(status="failed", error_code="TIMEOUT"),
    lambda r: r.update(returncode=-9),
    lambda r: r.update(returncode=False),
    lambda r: r.pop("guard_violation"),
    lambda r: r["admission"]["sample"].update(pressure_level=True),
    lambda r: r["admission"]["sample"].update(monotonic=float("nan")),
    lambda r: r["samples"][0].update(available_bytes=768 * 1024**2 - 1),
    lambda r: r["samples"][0].update(pressure_level=2, available_bytes=(768 + 512) * 1024**2 - 1),
    lambda r: r["samples"][0].update(pressure_level=4),
    lambda r: r["samples"][0].update(pressure_level=3),
    lambda r: r["samples"][0].update(pressure_level=0),
    lambda r: r["samples"][0].update(pressure_level=True),
    lambda r: r["samples"][0].update(rss_bytes=int(8.5 * 1024**3) + 1),
    lambda r: r["samples"][0].update(swap_used_bytes=133 * 1024**2, swap_growth_bytes=33 * 1024**2),
    lambda r: r["samples"][0].update(swap_used_bytes=133 * 1024**2, swap_growth_bytes=0),
    lambda r: r["samples"][0].update(rss_bytes=float("nan")),
    lambda r: r["samples"][0].update(available_bytes=float("inf")),
    lambda r: r["samples"][0].update(monotonic=float("nan")),
    lambda r: r["samples"][0].pop("available_bytes"),
    lambda r: r["samples"][0].pop("swap_used_bytes"),
    lambda r: r["samples"][0].pop("monotonic"),
    lambda r: r.pop("baseline"),
    lambda r: r.pop("admission"),
    lambda r: r.pop("preflight"),
    lambda r: r.update(preflight=None),
    lambda r: r["preflight"]["metadata"].update(sha256={}),
    lambda r: r["limits"].pop("native"),
    lambda r: r["limits"]["native"].pop("warning_grace_seconds"),
    lambda r: r["limits"]["native"].update(reserve_bytes=767 * 1024**2),
    lambda r: r["limits"]["native"].update(warning_headroom_bytes=511 * 1024**2),
    lambda r: r["limits"]["native"].update(headroom_grace_seconds=4),
    lambda r: r["limits"]["native"].update(warning_grace_seconds=21),
    lambda r: r["limits"]["native"].update(poll_seconds=2),
    lambda r: r["limits"]["native"].update(warning_swap_growth_bytes=33 * 1024**2),
    lambda r: r["limits"]["native"].update(max_rss_bytes=9 * 1024**3),
    lambda r: r["limits"]["native"].update(max_swap_growth_bytes=64 * 1024**2),
    lambda r: r["baseline"].update(pressure_level=2),
    lambda r: r["admission"].update(passed=False),
])
def test_conservative_native_policy_refuses_invalid_or_insufficient_evidence(mutation):
    receipt = guard()
    mutation(receipt)
    with pytest.raises(ValueError):
        _guard(receipt)


def test_warning_requires_stricter_warning_swap_bound_even_before_grace():
    receipt = guard()
    receipt["limits"]["native"]["warning_swap_growth_bytes"] = 16 * 1024**2
    receipt["samples"][0].update(pressure_level=2, swap_used_bytes=120 * 1024**2,
                                swap_growth_bytes=20 * 1024**2)
    with pytest.raises(ValueError, match="warning-pressure sufficient"):
        _guard(receipt)


def test_normal_pressure_retains_original_hard_swap_limit():
    receipt = guard()
    receipt["limits"]["native"]["warning_swap_growth_bytes"] = 16 * 1024**2
    receipt["samples"][0].update(swap_used_bytes=120 * 1024**2, swap_growth_bytes=20 * 1024**2)
    assert _guard(receipt) is receipt


def test_source_policy_fingerprint_must_match_existing_watchdog():
    receipt = guard()
    hashes = receipt["preflight"]["metadata"]["sha256"]
    hashes[next(iter(hashes))] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        _guard(receipt)


def test_reversed_or_repeated_sample_clock_is_refused():
    for second in (.4, .5):
        receipt = guard()
        receipt["samples"].append({**receipt["samples"][0], "monotonic": second})
        with pytest.raises(ValueError, match="clocks"):
            _guard(receipt)


def test_stronger_reserve_and_shorter_graces_remain_admissible():
    receipt = guard()
    receipt["limits"]["native"].update(reserve_bytes=1024**3,
        headroom_grace_seconds=1, warning_grace_seconds=10)
    assert _guard(receipt) is receipt
