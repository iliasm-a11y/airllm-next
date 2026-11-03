"""Portable control-flow/evidence checks. Never compile or run native code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from research.glm53_ultra import prefetch_compare as component


@pytest.fixture(autouse=True)
def synthetic_capture_identity(monkeypatch):
    # Portable fixtures exercise admission/control flow only, never trained data.
    # Production constants retain the real capture hash and 1.59 GB file offset.
    monkeypatch.setattr(component, "ABSOLUTE_OFFSET", 192)
    monkeypatch.setattr(component, "ACTIVATION_SHA256", hashlib.sha256(bytes(16384)).hexdigest())


def arguments(tmp_path):
    capture = tmp_path / "capture"; capture.mkdir()
    (capture / "activation.f32").write_bytes(bytes(16384))
    (capture / "tensors.jsonl").write_text(json.dumps({"step": 1, "name": "attn_norm-40",
        "shape": [4096, 1, 1, 1], "type": 0, "bytes": 16384,
        "strides": [4, 16384, 16384, 16384], "file": "activation.f32"}) + "\n")
    shard = tmp_path / component.MODEL_SHARD
    with shard.open("wb") as stream:
        stream.truncate(component.RANGE_CAP)  # Sparse fixture, no payload read/write.
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"revision": component.MODEL_REVISION,
        "tensors": [{"name": "blk.0.attn_q.weight", "type": "Q8_0", "dims": [4096, 8192],
                     "file": shard.name, "offset": 64, "bytes_with_alignment": component.WEIGHT_BYTES}],
        "files": [{"file": shard.name, "data_offset": 128}]}))
    source = tmp_path / "header source"; include = source / "ggml/include"; include.mkdir(parents=True)
    for name in component.HEADERS:
        (include / name).write_text("// portable source fixture " + name + "\n")
    libraries = tmp_path / "V4 libraries"; libraries.mkdir()
    artifacts = []
    for stem in ("libggml-cpu", "libggml-base"):
        name = stem + ".0.28.0.dylib"; raw = ("portable fixture " + stem).encode()
        (libraries / name).write_bytes(raw)
        (libraries / (stem + ".dylib")).symlink_to(name)
        artifacts.append({"path": "bin/" + name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    runtime = tmp_path / "V4.json"
    runtime.write_text(json.dumps({"status": "built", "source_revision": component.REVISION, "artifacts": artifacts}))
    return SimpleNamespace(inventory=inventory, capture=capture, source=source, libraries=libraries,
        runtime_manifest=runtime, model_directory=tmp_path, directory=tmp_path / "new component",
        output=tmp_path / "report.json", mapping="standard", modes=list(component.DEFAULT_MODES),
        phases=2, timeout_seconds=60, execute=False)


def test_prepare_freezes_recipe_without_payload_or_native_access(tmp_path, monkeypatch):
    args = arguments(tmp_path); original_open = Path.open
    def deny_payload(path, *positional, **keywords):
        if path.suffix in (".gguf", ".dylib"):
            raise AssertionError("Preparation opened checkpoint/library payload")
        return original_open(path, *positional, **keywords)
    def deny_command(*unused, **keywords):
        raise AssertionError("Preparation launched a command")
    monkeypatch.setattr(Path, "open", deny_payload)
    for name in ("run", "Popen", "check_output"):
        monkeypatch.setattr(component.subprocess, name, deny_command)
    report = component.run(args)
    assert report["status"] == "prepared_no_checkpoint_payload_read"
    assert report["absolute_offset"] == 192 and report["known_payload_bound_bytes"] < component.RANGE_CAP
    assert report["arms"] == [] and report["model_decoder_executed"] is False
    assert report["gpu_initialized"] is False and not (args.directory / "prefetch-compare").exists()
    plan = json.loads(Path(report["build_plan"]).read_text())
    script_lines = Path(report["build_script"]).read_text().splitlines()
    assert shlex.split(script_lines[-1].removeprefix("exec ")) == plan["compile_command"]
    assert plan["source_revision"] == component.REVISION


def test_prepare_rejects_impossible_source_range_without_loading_it(tmp_path):
    args = arguments(tmp_path)
    with (tmp_path / component.MODEL_SHARD).open("r+b") as stream:
        stream.truncate(1)
    report = component.run(args)
    assert report["status"] == "failed" and "range exceeds" in report["error"]
    assert not (args.directory / "prefetch-compare").exists()


@pytest.mark.parametrize("changed", ["activation", "offset"])
def test_prepare_refuses_wrong_captured_activation_or_original_offset(tmp_path, changed):
    args = arguments(tmp_path)
    if changed == "activation":
        (args.capture / "activation.f32").write_bytes(b"\x01" + bytes(16383))
    else:
        inventory = json.loads(args.inventory.read_text())
        inventory["tensors"][0]["offset"] += 1
        args.inventory.write_text(json.dumps(inventory))
    report = component.run(args)
    assert report["status"] == "failed"
    expected = "activation hash mismatch" if changed == "activation" else "absolute byte offset"
    assert expected in report["error"]
    assert not (args.directory / "prefetch-compare").exists()


@pytest.mark.parametrize("field,value", [("mapping", "automatic"), ("modes", []),
    ("modes", ["pread"] * 13), ("phases", 1), ("timeout_seconds", float("nan")), ("timeout_seconds", 61)])
def test_options_refuse_unbounded_or_ambiguous_experiments(tmp_path, field, value):
    args = arguments(tmp_path); setattr(args, field, value)
    with pytest.raises(ValueError):
        component.validate_options(args)


@pytest.mark.parametrize("change", [{"rss": component.RSS_LIMIT + 1},
    {"swap": component.SWAP_LIMIT + 1}, {"pressure": 4}, {"pressure": 0},
    {"pressure": True}, {"available": component.RSS_LIMIT - 1}, {"elapsed": 61},
    {"elapsed": float("nan")}])
def test_resource_guard_fails_closed(change):
    values = dict(rss=0, swap=0, baseline_swap=0, pressure=1,
                  available=2 * 1024**3, elapsed=1, timeout=60)
    values.update(change)
    with pytest.raises((MemoryError, RuntimeError)):
        component.check_guard(**values)


def phase(mode="none", index=0):
    delta = {"os_storage_read_bytes": 0, "pageins": 0, "task_faults": 0,
             "task_pageins": 0, "resident_bytes_after": 48 * component.MIB, "cpu_core_seconds": 0.}
    residency = {"mapping_pages": 2177, "resident_pages": 2177}
    return {"phase": index, "mode": mode, "mapping": "standard",
        "logical_weight_bytes": component.WEIGHT_BYTES, "mapped_bytes": component.WEIGHT_BYTES + 64,
        "page_size_bytes": 16384,
        "scratch_bytes": component.SCRATCH_BYTES, "pread_completed_bytes": component.WEIGHT_BYTES if mode == "pread" else 0,
        "pread_calls": 5 if mode == "pread" else 0, "preparation_wall_seconds": .001,
        "compute_wall_seconds": .0004, "preparation_plus_compute_seconds": .0014,
        "observation_gap_seconds": .0001, "total_observed_sequence_seconds": .0015,
        "residency_before": dict(residency), "residency_after_preparation": dict(residency),
        "residency_after_compute": dict(residency), "all_outputs_finite": True, "file_identity_unchanged": True,
        "preparation_process_delta": dict(delta), "compute_process_delta": dict(delta), "total_process_delta": dict(delta)}


def test_native_phase_requires_complete_bytes_and_reconciled_timing():
    valid = phase("pread")
    component.validate_phase(valid, index=0, mode="pread", mapping="standard")
    for key, value in (("pread_completed_bytes", component.WEIGHT_BYTES - 1),
                       ("pread_calls", 0), ("total_observed_sequence_seconds", .0004),
                       ("all_outputs_finite", False), ("compute_wall_seconds", float("inf")),
                       ("page_size_bytes", 0),
                       ("residency_before", {"mapping_pages":2176,"resident_pages":2176})):
        broken = deepcopy(valid); broken[key] = value
        with pytest.raises(ValueError):
            component.validate_phase(broken, index=0, mode="pread", mapping="standard")


def test_warm_reference_requires_observed_matrix_residency():
    reference = phase(index=1)
    reference["residency_before"]["resident_pages"] -= 1
    with pytest.raises(ValueError, match="Warm reference"):
        component.validate_phase(reference, index=1, mode="none", mapping="standard")


def test_missing_process_observation_rejects_phase():
    observation = phase(); del observation["total_process_delta"]
    with pytest.raises(ValueError, match="process counters"):
        component.validate_phase(observation, index=0, mode="none", mapping="standard")


def test_summary_retains_preparation_cost_and_never_qualifies_speed():
    arms = [{"status": "complete", "mode": "pread", "phases": [phase("pread")]}]
    result = component.summarize(arms)
    assert result["mode_medians"]["pread"]["preparation_plus_compute_seconds"] == .0014
    assert result["all_initial_mappings_resident_at_snapshot"] is True
    assert result["optimization_qualified"] is False and result["tokens_per_second"] is None
    arms[0]["status"] = "failed"
    with pytest.raises(ValueError):
        component.summarize(arms)


def test_build_verification_rejects_library_mutation_without_any_model_read(tmp_path, monkeypatch):
    args = arguments(tmp_path); metadata, plan = component.prepare(args)
    def source_git(command, **kwargs):
        if command[3] == "rev-parse":
            return component.REVISION + "\n"
        name = command[-1].rsplit("/", 1)[-1]
        return (args.source / "ggml/include" / name).read_bytes()
    monkeypatch.setattr(component.subprocess, "check_output", source_git)
    assert component.verify_plan(metadata["build_plan"])["verified"] is True
    (args.libraries / "libggml-cpu.0.28.0.dylib").write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="library hash mismatch"):
        component.verify_plan(metadata["build_plan"])
