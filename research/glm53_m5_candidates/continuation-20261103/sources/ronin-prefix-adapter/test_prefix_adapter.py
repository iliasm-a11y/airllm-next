"""Tiny protocol fixtures only. No native process/model/tokenizer/GPU is run.

Positive protocol fixtures explicitly MOCK the unqualified B8 admission and
native package binding. They are not positive cache/fidelity evidence.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prefix_adapter_fixture", ROOT / "prefix_adapter.py")
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
p = module


@dataclass
class Limits:
    reserve_bytes: int = 768 * p.MiB
    base_workspace_bytes: int = 1024 * p.MiB
    max_rss_bytes: int = 9 * 1024 * p.MiB
    max_swap_growth_bytes: int = 64 * p.MiB
    warning_swap_growth_bytes: int = 32 * p.MiB
    warning_headroom_bytes: int = 512 * p.MiB
    headroom_grace_seconds: float = 3
    load_timeout_seconds: float = 900
    request_timeout_seconds: float = 1800
    warning_grace_seconds: float = 20
    poll_seconds: float = 1
    max_pending: int = 1024


class Clock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, value): self.value += value


class Scheduler:
    def __init__(self): self.lock, self.active = threading.Lock(), 0
    @contextmanager
    def slot(self, cancel):
        with self.lock:
            p.require(not cancel.is_set(), "cancelled", "CANCELLED")
            self.active += 1
            try: yield
            finally: self.active -= 1
    def status(self): return {"physical_limit": 1, "active": self.active, "max_pending": 1024}


PAYLOAD = b"synthetic-native-memory-not-model-state" * 2
TOKENS = tuple(range(208))
TEMPLATE = "synthetic native template; no real tokenizer is invoked"
SPEC = p.SnapshotSpec(TOKENS, len(p.RAW_HEADER) + len(PAYLOAD), hashlib.sha256(p.RAW_HEADER + PAYLOAD).hexdigest())


def slot_bytes(payload=PAYLOAD, ids=TOKENS):
    packed = struct.pack("<iII", -1, p.TOKEN_VERSION, len(ids)) + struct.pack("<" + "i" * len(ids), *ids) + struct.pack("<I", 0)
    return struct.pack("<III", p.FILE_MAGIC, p.FILE_VERSION, len(packed) // 4) + packed + payload


def guard(model, *, phase):
    result = {key: True for key in ("passed", "power_context_passed", "native_limits_unchanged")}
    result["phase"] = phase
    if phase == "preload":
        result.update(no_conflicting_uid_engine=True, uid_exclusive=False, identity_passed=False)
    else:
        result.update(uid_exclusive=True, identity_passed=True)
    return result


class FakeNative:
    instances = []
    def __init__(self, path, **options):
        self.options, self.model_path = options, Path(path)
        for name, value in options.items(): setattr(self, name, value)
        self.limits, self.scheduler = options["limits"], Scheduler()
        self.runtime_manifest_file = Path("/synthetic/no-runtime-manifest")
        self._process, self.loaded = None, False
        self._lock_file = None
        self._runtime_sha256 = "a" * 64
        self.calls, self.active_ids, self.memory = [], [], "empty"
        self.loads = self.requests = self.closes = 0
        self.guard_samples = 0
        self.bad_sample = self.bad_restore = self.bad_cache = self.bad_erase = False
        self.bad_prefix = self.slow_generation = False
        self.clock = self._campaign.clock
        self.instances.append(self)

    def load(self, cancel=None):
        if not self.loaded:
            self.loaded = True
            self._process = type("OwnedFixtureProcess", (), {"poll": lambda self: None})()
            self._lock_file = type("RetainedFixtureNativeLock", (), {"closed": False})()
            self.loads += 1
            self._http_json("GET", "/health", timeout=1)
        return self.status()

    def close(self):
        self.closes += 1
        self.loaded, self._process, self.active_ids, self.memory = False, None, [], "empty"
        if self._lock_file is not None: self._lock_file.closed = True
        self._lock_file = None
        return {"loaded": False}

    def _build_command(self, port):
        return ["/synthetic/server", "--no-repack", "--no-op-offload", "--parallel", "1"]

    def _check_running(self): p.require(self.loaded, "fixture stopped")
    def _check_identity(self): pass
    def _admit(self): self.guard_samples += 1

    def status(self):
        return {"ready": self.loaded, "loaded": self.loaded, "checkpoint": {"errors": []},
                "counters": {"loads": self.loads, "requests": self.requests}, "scheduler": self.scheduler.status()}

    def _http_json(self, method, path, payload=None, timeout=30):
        self.calls.append((method, path, payload, timeout))
        if path == "/health": return {"status": "ok"}
        if path == "/props":
            return {"total_slots": 1, "endpoint_slots": True, "default_generation_settings": {"n_ctx": 2048}, "chat_template": TEMPLATE}
        if path == "/slots/0?action=erase":
            n = len(self.active_ids)
            if not self.bad_erase: self.active_ids, self.memory = [], "empty"
            return {"id_slot": 0, "n_erased": n}
        if path == "/completion":
            self.active_ids, self.memory = list(payload["prompt"]), "prefix"
            count = 2 if self.bad_sample else 1
            return {"tokens": [785] * count, "timings": {"predicted_n": count, "prompt_n": 208, "cache_n": 0},
                    "truncated": False, "stop_type": "limit", "content": "SYNTHETIC discarded sample"}
        if path == "/slots":
            return [{"id": 0, "is_processing": False, "n_prompt_tokens": len(self.active_ids)}]
        if path == "/slots/0?action=save":
            data = slot_bytes(payload=PAYLOAD if self.memory == "prefix" else b"x" * len(PAYLOAD))
            (self._prefix_dir / payload["filename"]).write_bytes(data)
            return {"id_slot": 0, "filename": payload["filename"], "n_saved": len(self.active_ids), "n_written": len(data)}
        if path == "/slots/0?action=restore":
            self.active_ids, self.memory = list(TOKENS), "bad" if self.bad_restore else "prefix"
            return {"id_slot": 0, "filename": payload["filename"], "n_restored": 208,
                    "n_read": (self._prefix_dir / payload["filename"]).stat().st_size}
        if path == "/apply-template":
            return {"prompt": json.dumps(payload["messages"], ensure_ascii=False)}
        if path == "/tokenize":
            ids = list(TOKENS) + [9, 10, 11]
            if self.bad_prefix: ids[0] += 1
            return {"tokens": ids}
        raise AssertionError("unexpected mocked HTTP " + path)

    def _events(self, payload, cancel):
        self.calls.append(("STREAM", "/completion", payload, None))
        self.active_ids, self.memory = payload["prompt"] + [99], "private"
        if self.slow_generation: self.clock.advance(2701)
        yield {"content": "SYNTHETIC answer", "tokens": [99], "stop": False}
        yield {"content": "", "tokens": [], "stop": True, "stop_type": "eos", "truncated": False,
               "tokens_predicted": 1, "tokens_evaluated": len(payload["prompt"]),
               "timings": {"cache_n": 0 if self.bad_cache else 208, "prompt_n": 3, "predicted_n": 1}}
        self.requests += 1


Model = p.make_model_class(FakeNative)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.budget = p.CampaignBudget(clock=self.clock)
        self.gate = p.B8Gate(self.root / "no-repo", self.root / "no-full", self.root / "no-restore",
                            self.root / "no-full-guard", self.root / "no-restore-guard")
        self.evidence = {"spec": SPEC, "manifest": {"render_options": {"native_environment": {"AIRLLM_DENSE_PREFETCH_MAX_COLS": "1", "RONIN_PREFILL_DIAGNOSTIC": "1"}},
            "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest()}, "source_sha256": {}}
        self.gate_mock = mock.patch.object(self.gate, "verify", return_value=self.evidence)
        self.gate_mock.start()
        self.binding_mock = mock.patch.object(Model, "_verify_profile_binding", return_value=None)
        self.binding_mock.start()
        self.models = []

    def tearDown(self):
        for model in self.models: model.close()
        self.binding_mock.stop()
        self.gate_mock.stop()
        self.temp.cleanup()

    def options(self, directory="adapter", **changes):
        result = {"enable_shared_prefix": True, "b8_gate": self.gate, "campaign_budget": self.budget,
            "admission_guard": guard, "evidence_dir": self.root / directory,
            "context_tokens": 2048, "threads": 10, "batch_size": 8, "gpu_layers": 0,
            "cpu_moe": False, "moe_cache_mib": 0, "allow_unqualified_gpu": False,
            "limits": Limits(max_rss_bytes=int(8.5 * 1024**3), max_swap_growth_bytes=32 * p.MiB)}
        result.update(changes)
        return result

    def model(self, **changes):
        value = Model(self.root / "synthetic-model", **self.options(**changes))
        self.models.append(value)
        return value

    def test_disabled_without_explicit_optin_has_no_directory_or_native_instance(self):
        before = len(FakeNative.instances)
        with self.assertRaises(p.AdapterError):
            self.model(enable_shared_prefix=False)
        self.assertFalse((self.root / "adapter").exists())
        self.assertEqual(len(FakeNative.instances), before)

    def test_bad_b8_receipt_refuses_before_server_load(self):
        self.gate_mock.stop()
        for root, mode in ((self.gate.reference, "full"), (self.gate.restored, "restore")):
            root.mkdir()
            (root / "report.json").write_text(json.dumps({"complete": False, "mode": mode}))
        model = self.model()
        with mock.patch.object(p, "strict_b8_comparison") as comparison, self.assertRaises(p.AdapterError):
            model.load()
        comparison.assert_not_called()
        self.assertEqual(model.loads, 0)
        self.assertTrue(model._prefix_poisoned)

    def test_relaxed_limits_or_wrong_physical_profile_refused_before_directory(self):
        bad = Limits(max_rss_bytes=int(8.5 * 1024**3), max_swap_growth_bytes=32 * p.MiB, reserve_bytes=1)
        for delta in ({"threads": 8}, {"gpu_layers": 1}, {"limits": bad}):
            with self.assertRaises(p.AdapterError): self.model(**delta)
        self.assertFalse((self.root / "adapter").exists())

    def test_exact_flags_and_clean_environment_are_research_only(self):
        model = self.model()
        model._prefix_evidence = self.evidence
        command = model._build_command(9)
        self.assertIn("--no-repack", command)
        self.assertIn("--swa-full", command)
        self.assertIn("--no-kv-offload", command)
        self.assertIn("--slot-save-path", command)
        with mock.patch.dict("os.environ", {"GGML_CPU_DISABLE_FUSION": "1", "AIRLLM_CPU_METAL_Q8_BRIDGE": "1", "LLAMA_API_KEY": "synthetic"}):
            env = model._runtime_environment()
        self.assertNotIn("GGML_CPU_DISABLE_FUSION", env)
        self.assertNotIn("AIRLLM_CPU_METAL_Q8_BRIDGE", env)
        self.assertNotIn("LLAMA_API_KEY", env)
        self.assertNotIn("RONIN_PREFILL_DIAGNOSTIC", env)
        self.assertEqual(env["XDG_CONFIG_HOME"], str(model._prefix_config_dir))
        self.assertEqual(env["AIRLLM_DENSE_PREFETCH_MAX_COLS"], "1")

    def test_automatic_private_preset_refused_without_loading_or_reading_it(self):
        model = self.model()
        preset = model._prefix_config_dir / "llama.cpp/config.ini"
        preset.parent.mkdir()
        preset.write_text("untrusted fixture preset; never parse")
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("must not read preset")), self.assertRaises(p.AdapterError):
            model.load()
        self.assertFalse(model.loaded)

    def test_common_sample_is_counted_discarded_and_state_is_immutable(self):
        model = self.model()
        model.load(); model.load()
        self.assertEqual(model.loads, 1)
        self.assertEqual(model._prefix_discarded, 1)
        self.assertEqual(model.memory, "empty")
        self.assertEqual(model.active_ids, [])
        self.assertEqual(Path(model._prefix_snapshot["path"]).stat().st_mode & 0o777, 0o400)
        records = [json.loads(line) for line in model._prefix_journal.read_text().splitlines()]
        self.assertTrue(any(r.get("phase") == "preparation_sample_discarded" and r["tokens"] == [785] for r in records))
        health = [r for r in model.calls if r[1] == "/health"]
        self.assertEqual(health[0][3], 1)

    def test_two_logical_requests_restore_after_dirty_state_and_preserve_full_messages(self):
        model = self.model()
        messages = [{"role": "system", "content": "own-memory marker"}, {"role": "user", "content": "full unsummarized request"}]
        for _ in range(2):
            events = list(model.stream_chat(messages, max_new_tokens=512))
            self.assertEqual(events[0]["tokens"], [99])
            self.assertEqual(events[-1]["stop_type"], "eos")
            self.assertEqual(model.memory, "empty")
        self.assertEqual(model.loads, 1)
        self.assertEqual(model.requests, 2)
        self.assertEqual(model._prefix_discarded, 1)
        bodies = [r[2] for r in model.calls if r[1] == "/apply-template"]
        self.assertEqual(bodies, [{"messages": messages}] * 2)
        streams = [r[2] for r in model.calls if r[0] == "STREAM"]
        self.assertEqual(streams[0]["prompt"], list(TOKENS) + [9, 10, 11])
        self.assertTrue(streams[0]["cache_prompt"])
        self.assertEqual(len([r for r in model.calls if r[1] == "/slots/0?action=restore"]), 2)

    def test_wrong_native_prefix_never_reaches_restore_or_generation(self):
        model = self.model()
        model.load(); model.bad_prefix = True
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertEqual(len([r for r in model.calls if r[1] == "/slots/0?action=restore"]), 0)
        self.assertEqual(model.requests, 0)
        self.assertFalse(model.loaded)

    def test_bad_preparation_sample_closes_before_adopting_snapshot(self):
        model = self.model(); model.bad_sample = True
        with self.assertRaises(p.AdapterError): model.load()
        self.assertFalse(model.loaded)
        self.assertIsNone(model._prefix_snapshot)
        self.assertFalse(list(model._prefix_dir.glob("*.bin")))

    def test_restored_payload_and_cache_counter_are_fail_closed(self):
        for field in ("bad_restore", "bad_cache"):
            model = self.model(directory=field)
            model.load(); setattr(model, field, True)
            with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
            self.assertFalse(model.loaded)
            self.assertTrue(model._prefix_poisoned)
            self.assertFalse(list(model._prefix_dir.glob("*.bin")))
            with self.assertRaises(p.AdapterError): model.load()

    def test_corrupted_immutable_identity_refused_before_http_restore(self):
        model = self.model(); model.load()
        path = Path(model._prefix_snapshot["path"])
        path.chmod(0o600)
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertFalse(any(r[1] == "/slots/0?action=restore" for r in model.calls))

    def test_logical_timeout_includes_private_work_without_resetting_campaign(self):
        model = self.model(); model.load(); model.slow_generation = True
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertFalse(model.loaded)
        self.assertEqual(self.clock.value, 2701)
        self.assertEqual(self.budget.deadline, 7200)

    def test_campaign_deadline_refuses_even_with_fresh_logical_budget(self):
        model = self.model(); model.load(); self.clock.advance(7201)
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertFalse(model.loaded)
        self.assertEqual(model.requests, 0)

    def test_abandoned_generator_closes_and_erases_owned_snapshot(self):
        model = self.model()
        stream = model.stream_chat([{"role": "user", "content": "x"}])
        next(stream); stream.close()
        self.assertFalse(model.loaded)
        self.assertTrue(model._prefix_poisoned)
        self.assertFalse(list(model._prefix_dir.glob("*.bin")))

    def test_logical_scope_counts_host_time_before_native_stream(self):
        model = self.model()
        model.load()
        with self.assertRaises(p.AdapterError):
            with model.logical_scope():
                self.clock.advance(2700.1)
                list(model.stream_chat([{"role": "system", "content": "synthetic"}], 2))
        self.assertIsNone(model._prefix_external_deadline)
        self.assertFalse(model.loaded)
        self.assertFalse(list(model._prefix_dir.glob("*.bin")))

    def test_external_guard_failure_does_not_start_a_model(self):
        model = self.model(admission_guard=lambda m, **kw: {"passed": False})
        with self.assertRaises(p.AdapterError): model.load()
        self.assertEqual(model.loads, 0)

    def test_honest_preload_guard_has_no_native_fd_until_load(self):
        observed = []
        def phased(model, *, phase):
            observed.append((phase, model._lock_file is not None))
            return guard(model, phase=phase)
        model = self.model(admission_guard=phased)
        model.load()
        self.assertIn(("preload", False), observed)
        self.assertIn(("loaded", True), observed)
        self.assertEqual(model.loads, 1)

    def test_preload_cannot_claim_ownership_before_native_lock_exists(self):
        def false_owner(model, *, phase):
            return {"phase": phase, "passed": True, "power_context_passed": True, "native_limits_unchanged": True,
                    "no_conflicting_uid_engine": True, "uid_exclusive": True, "identity_passed": True}
        model = self.model(admission_guard=false_owner)
        with self.assertRaises(p.AdapterError): model.load()
        self.assertEqual(model.loads, 0)

    def test_erase_must_be_observed_empty_not_just_return_success(self):
        model = self.model(); model.load(); model.bad_erase = True
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertFalse(model.loaded)
        rows = [json.loads(line) for line in model._prefix_journal.read_text().splitlines()]
        self.assertTrue(any(r.get("phase") == "logical_request" and r.get("status") == "failed" for r in rows))

    def test_http_profile_drift_is_not_normalized_away(self):
        model = self.model(); model.load()
        self.assertEqual(len(model._prefix_snapshot["http_profile_sha256"]), 64)
        model.limits.request_timeout_seconds = 1700
        with self.assertRaises(p.AdapterError): list(model.stream_chat([{"role": "user", "content": "x"}]))
        self.assertFalse(any(r[1] == "/slots/0?action=restore" for r in model.calls))

    def test_unload_alias_invalidates_the_copy_like_close(self):
        model = self.model(); model.load(); model.unload()
        self.assertIsNone(model._prefix_snapshot)
        self.assertIsNone(model._prefix_evidence)
        self.assertFalse(list(model._prefix_dir.glob("*.bin")))


class FileFormatTests(unittest.TestCase):
    def test_tiny_fixture_streams_canonical_payload_and_rejects_wrong_ids_or_appendix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bin"
            path.write_bytes(slot_bytes())
            check = p.validate_slot_file(path, SPEC)
            self.assertEqual(check["canonical_raw_sha256"], SPEC.raw_sha256)
            self.assertEqual(check["memory_offset"], 860)
            for data in (slot_bytes() + b"annex", slot_bytes(ids=TOKENS[:-1] + (999,)), slot_bytes(payload=b"x" * len(PAYLOAD))):
                path.write_bytes(data)
                with self.assertRaises(p.AdapterError): p.validate_slot_file(path, SPEC)

    def test_symlink_or_hardlink_is_not_an_immutable_owned_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "state.bin"; path.write_bytes(slot_bytes())
            link = root / "alias.bin"; link.symlink_to(path)
            with self.assertRaises(p.AdapterError): p.validate_slot_file(link, SPEC)
            link.unlink(); link.hardlink_to(path)
            with self.assertRaises(p.AdapterError): p.validate_slot_file(path, SPEC)

    def test_file_cap_rejected_from_stat_without_payload_read_or_big_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bin"; path.write_bytes(b"tiny")
            observed = p.identity(path)
            with mock.patch.object(p, "identity", return_value=(observed[0], observed[1], p.MAX_FILE + 1, observed[3], observed[4])), \
                 mock.patch.object(Path, "open", side_effect=AssertionError("must not read oversized payload")), \
                 self.assertRaises(p.AdapterError):
                p.validate_slot_file(path, SPEC)


class NativeGateTests(unittest.TestCase):
    def test_future16_declaration_without_actual_two_build_bridge_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); full = root / "unrelated-folder-a"; restored = root / "unrelated-folder-b"
            full.mkdir(); restored.mkdir()
            parameters = {**p.EXPECTED_PARAMETERS, "n_batch": 16, "n_ubatch": 16, **p.VARIANT_PARAMETERS}
            for path, mode in ((full, "full"), (restored, "restore")):
                (path / "report.json").write_text(json.dumps({"complete": True, "mode": mode, "microbatch": 16, "state_parameters": parameters}))
            gate = p.NativePrefixGate(root / "no-repo", full, restored, root / "no-guard-a", root / "no-guard-b")
            with mock.patch.object(p, "strict_b8_comparison") as comparison, self.assertRaises(p.AdapterError): gate.verify()
            comparison.assert_not_called()

    def test_even_a_passing_native16_restore_needs_the_strict_cpu8_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); full = root / "full"; restored = root / "restore"; full.mkdir(); restored.mkdir()
            parameters = {**p.EXPECTED_PARAMETERS, "n_batch": 16, "n_ubatch": 16, **p.VARIANT_PARAMETERS}
            for path, mode in ((full, "full"), (restored, "restore")):
                (path / "report.json").write_text(json.dumps({"complete": True, "mode": mode, "microbatch": 16, "state_parameters": parameters}))
            manifest = {"state_parameters": parameters, "prefix_tokens": list(TOKENS),
                "render_options": {"reasoning_effort_override": None, "native_add_special": True, "native_parse_special": True,
                                   "native_environment": {"AIRLLM_DENSE_PREFETCH_MAX_COLS": "1", "RONIN_PREFILL_DIAGNOSTIC": "1"}}}
            guards = []
            for n in (0, 1):
                path = root / ("guard-" + str(n) + ".json")
                path.write_text(json.dumps({"preflight": {"snapshot_manifest": manifest, "source_sha256": {"fixture": "not-real"}}}))
                guards.append(path)
            gate = p.NativePrefixGate(root / "no-repo", full, restored, *guards, cpu8_bridge={"fixture_only": True})
            native_result = {"passed": True, "errors": [], "criteria": {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0},
                "direct_restore_boundary": {"passed": True, "boundary": 208, "snapshot_bytes": p.RAW_BYTES, "snapshot_sha256": p.RAW_SHA256}}
            with mock.patch.object(p, "strict_b8_comparison", return_value=native_result), \
                 mock.patch.object(p, "strict_variant_bridge", return_value={"passed": False}) as bridge, \
                 self.assertRaises(p.AdapterError):
                gate.verify()
            bridge.assert_called_once()

    def test_closed_metadata_drift_is_rejected_even_if_mock_comparison_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); full = root / "full"; restored = root / "restore"; full.mkdir(); restored.mkdir()
            for path, mode in ((full, "full"), (restored, "restore")):
                (path / "report.json").write_text(json.dumps({"complete": True, "mode": mode, "microbatch": 8,
                                                            "state_parameters": p.EXPECTED_PARAMETERS}))
            manifest = {"state_parameters": p.EXPECTED_PARAMETERS, "prefix_tokens": list(TOKENS),
                "render_options": {"reasoning_effort_override": None, "native_add_special": True, "native_parse_special": True,
                                   "native_environment": {"AIRLLM_DENSE_PREFETCH_MAX_COLS": "1", "RONIN_PREFILL_DIAGNOSTIC": "1"}}}
            guards = []
            for n in (0, 1):
                path = root / ("guard-" + str(n) + ".json")
                path.write_text(json.dumps({"preflight": {"snapshot_manifest": manifest, "source_sha256": {"fixture": "not-real"}}}))
                guards.append(path)
            gate = p.NativePrefixGate(root / "no-repo", full, restored, *guards)
            def changed(*args):
                path = full / "report.json"
                path.write_text(path.read_text() + " ")
                return {"passed": True, "errors": [], "criteria": {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0},
                    "direct_restore_boundary": {"passed": True, "boundary": 208, "snapshot_bytes": p.RAW_BYTES, "snapshot_sha256": p.RAW_SHA256}}
            with mock.patch.object(p, "strict_b8_comparison", side_effect=changed), self.assertRaises(p.AdapterError): gate.verify()


if __name__ == "__main__": unittest.main()
