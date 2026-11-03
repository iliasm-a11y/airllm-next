"""Small HTTP-control mocks; no model, native/GPU process or real guard query.

Positive fixtures explicitly replace native proof admission and package binding.
They prove only source/protocol behavior, never numerical HTTP fidelity or B/C.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prefix_adapter as p


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


class Scheduler:
    @contextmanager
    def slot(self, cancel):
        p.require(not cancel.is_set(), "cancelled")
        yield
    def status(self): return {"physical_limit": 1}


IDS = tuple(range(208))
TEMPLATE = "synthetic HTTP16 template"
PAYLOAD = b"synthetic-non-model-state" * 2
SPEC = p.SnapshotSpec(IDS, len(p.RAW_HEADER) + len(PAYLOAD), hashlib.sha256(p.RAW_HEADER + PAYLOAD).hexdigest())


def slot_bytes():
    packed = struct.pack("<iII", -1, 1, 208) + struct.pack("<208i", *IDS) + struct.pack("<I", 0)
    return struct.pack("<III", p.FILE_MAGIC, p.FILE_VERSION, len(packed) // 4) + packed + PAYLOAD


def guard(model, *, phase):
    return {"phase": phase, "passed": True, "power_context_passed": True, "native_limits_unchanged": True,
            "no_conflicting_uid_engine": phase == "preload", "uid_exclusive": phase == "loaded", "identity_passed": phase == "loaded"}


class Native:
    def __init__(self, path, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        self.model_path = Path(path)
        self.scheduler = Scheduler()
        self._process = self._lock_file = None
        self._runtime_sha256 = "a" * 64
        self.calls, self.active = [], []
        self.prompt_size, self.loads, self.requests = 521, 0, 0
        self.bad_cache = self.bad_sample = self.slow_private = False
    def load(self, cancel=None):
        if self._process is None:
            self._process = type("OwnedMockChild", (), {"poll": lambda s: None})()
            self._lock_file = type("OwnedMockLock", (), {"closed": False})()
            self.loads += 1
            self._http_json("GET", "/health", timeout=1)
        return self.status()
    def close(self):
        if self._lock_file is not None: self._lock_file.closed = True
        self._process = self._lock_file = None
        self.active = []
    def _check_running(self): p.require(self._process is not None, "mock child stopped")
    def _check_identity(self): pass
    def _admit(self): return {"available_bytes": 4 * 1024 ** 3}
    def status(self):
        return {"ready": self._process is not None, "counters": {"loads": self.loads, "requests": self.requests}, "scheduler": self.scheduler.status()}
    def _build_command(self, port):
        return ["/mock/server", "--parallel", "1", "--no-repack", "--no-op-offload", "--batch-size", str(self.batch_size), "--ubatch-size", str(self.batch_size)]
    def cached(self, payload):
        if not payload["cache_prompt"]: return 0
        count = 0
        for a, b in zip(self.active, payload["prompt"]):
            if a != b: break
            count += 1
        return count
    def _http_json(self, method, path, payload=None, timeout=30):
        self.calls.append((method, path, payload))
        if path == "/health": return {"status": "ok"}
        if path == "/props": return {"total_slots": 1, "endpoint_slots": True, "default_generation_settings": {"n_ctx": 2048}, "chat_template": TEMPLATE}
        if path == "/slots/0?action=erase":
            count = len(self.active); self.active = []
            return {"id_slot": 0, "n_erased": count}
        if path == "/slots": return [{"id": 0, "is_processing": False, "n_prompt_tokens": len(self.active)}]
        if path == "/apply-template": return {"prompt": json.dumps(payload["messages"])}
        if path == "/tokenize": return {"tokens": list(range(self.prompt_size))}
        if path == "/completion":
            cached = self.cached(payload)
            if self.slow_private: self._campaign.clock.value += 2701
            self.active = list(payload["prompt"])
            return {"tokens": [999] * (2 if self.bad_sample else 1), "timings": {"cache_n": cached, "prompt_n": len(self.active)-cached, "predicted_n": 1},
                    "tokens_predicted": 1, "tokens_evaluated": len(self.active), "stop_type": "limit", "truncated": False}
        if path == "/slots/0?action=save":
            p.require(self.active == list(IDS), "only mock common state may be saved")
            data = slot_bytes(); (self._prefix_dir / payload["filename"]).write_bytes(data)
            return {"id_slot": 0, "filename": payload["filename"], "n_saved": 208, "n_written": len(data)}
        if path == "/slots/0?action=restore":
            self.active = list(IDS)
            return {"id_slot": 0, "filename": payload["filename"], "n_restored": 208, "n_read": SPEC.file_bytes}
        raise AssertionError(path)
    def _events(self, payload, cancel):
        self.calls.append(("STREAM", "/completion", payload))
        cached = self.cached(payload)
        self.active = list(payload["prompt"])
        yield {"tokens": [100], "content": "synthetic visible output"}
        yield {"tokens": [], "stop": True, "stop_type": "eos", "truncated": False,
               "tokens_predicted": 1, "tokens_evaluated": len(self.active),
               "timings": {"cache_n": cached+1 if self.bad_cache else cached, "prompt_n": len(self.active)-cached, "predicted_n": 1}}
        self.requests += 1


Model = p.make_model_class(Native)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.models = []
        self.binding = mock.patch.object(Model, "_verify_profile_binding", return_value=None)
        self.binding.start()
    def tearDown(self):
        for model in self.models: model.close()
        self.binding.stop(); self.temp.cleanup()
    def model(self, common=False, **changes):
        gate = p.NativeTail16Gate(self.root, self.root / "full16", self.root / "guard", cpu8_bridge={}, common_cache=common)
        gate.verify = mock.Mock(return_value={"native_microbatch": 16, "common_cache_admitted": common, "spec": SPEC,
            "manifest": {"template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(), "render_options": {"native_environment": {"RONIN_PREFILL_DIAGNOSTIC": "1"}}}})
        kwargs = dict(enable_tail16=True, common_cache=common, native_gate=gate, campaign_budget=p.CampaignBudget(clock=Clock()),
            admission_guard=guard, evidence_dir=self.root / ("evidence-"+str(len(self.models))), context_tokens=2048, threads=10,
            batch_size=16, gpu_layers=0, cpu_moe=False, moe_cache_mib=0, allow_unqualified_gpu=False,
            limits=Limits(max_rss_bytes=int(8.5 * 1024**3), max_swap_growth_bytes=32*p.MiB, request_timeout_seconds=2700))
        kwargs.update(changes)
        model = Model(self.root / "mock-model", **kwargs); self.models.append(model)
        return model
    def stream(self, model): return list(model.stream_chat([{"role": "user", "content": "full private message"}], 512))

    def test_optin_and_exact16_required_before_output_creation(self):
        with self.assertRaises(p.AdapterError): self.model(enable_tail16=False)
        with self.assertRaises(p.AdapterError): self.model(batch_size=8)

    def test_no_common_cache_521_prepares520_then_suffix1_and_never_saves(self):
        model = self.model(); events = self.stream(model)
        prepare = [r[2] for r in model.calls if r[1] == "/completion" and r[0] != "STREAM"]
        generate = [r[2] for r in model.calls if r[0] == "STREAM"]
        self.assertEqual([len(row["prompt"]) for row in prepare], [520])
        self.assertFalse(prepare[0]["cache_prompt"])
        self.assertEqual(len(generate[0]["prompt"]), 521)
        self.assertTrue(generate[0]["cache_prompt"])
        self.assertEqual(events[-1]["timings"]["cache_n"], 520)
        self.assertEqual(model._prefix_discarded, 1)
        self.assertEqual(model.active, [])
        self.assertFalse(any("action=save" in r[1] or "action=restore" in r[1] for r in model.calls))
        self.assertIsNone(model._prefix_snapshot)

    def test_523_suffix3_and_direct_cases_match_native_v1_chunking(self):
        for size in (520, 521, 523, 525, 527, 529, 536, 537, 542):
            model = self.model(); model.prompt_size = size
            events = self.stream(model)
            wanted = p.tail16_plan(size)
            self.assertEqual(events[-1]["timings"]["cache_n"], wanted["generation_cached_tokens"])
            self.assertEqual(events[-1]["timings"]["prompt_n"], wanted["generation_prompt_tokens"])
            self.assertEqual(model._prefix_discarded, wanted["private_preparation_sample_count"])
        self.assertEqual(p.tail16_plan(523)["generation_prompt_tokens"], 3)
        self.assertIsNone(p.tail16_plan(542)["prepare_end"])  # even14 tail remains native direct

    def test_common_mode_requires_mode_gate_and_restores208_before_private520(self):
        model = self.model(common=True); self.stream(model)
        private = [r[2] for r in model.calls if r[1] == "/completion" and r[0] != "STREAM"]
        self.assertEqual([len(r["prompt"]) for r in private], [208, 520])
        self.assertFalse(private[0]["cache_prompt"]); self.assertTrue(private[1]["cache_prompt"])
        self.assertTrue(any("action=restore" in r[1] for r in model.calls))
        self.assertEqual(model._prefix_discarded, 2)

    def test_direct_common_mode_requires208_cache_and_distinct_profile_key(self):
        a, b = self.model(), self.model(common=True)
        a.prompt_size = b.prompt_size = 529
        self.assertEqual(self.stream(a)[-1]["timings"]["cache_n"], 0)
        self.assertEqual(self.stream(b)[-1]["timings"]["cache_n"], 208)
        self.assertNotEqual(a._tail16_bound_key, b._tail16_bound_key)

    def test_repeat_logical_requests_use_same_model_and_erase_private_state(self):
        model = self.model()
        self.stream(model); self.stream(model)
        self.assertEqual(model.loads, 1)
        self.assertEqual(model.requests, 2)
        self.assertEqual(model._prefix_discarded, 2)
        self.assertEqual(model.active, [])

    def test_bad_cache_or_preparation_sample_never_produces_success(self):
        for name in ("bad_cache", "bad_sample"):
            model = self.model(); setattr(model, name, True)
            with self.assertRaises(p.AdapterError): self.stream(model)
            self.assertTrue(model._prefix_poisoned)
            self.assertIsNone(model._process)

    def test_private_preparation_consumes_single_logical_deadline(self):
        model = self.model(); model.slow_private = True
        with self.assertRaises(p.AdapterError):
            with model.logical_scope(): self.stream(model)
        self.assertIsNone(model._process)
        self.assertEqual(model._campaign.clock.value, 2701)

    def test_no_cache_mode_does_not_prime_or_claim_a_snapshot(self):
        model = self.model(); model.load()
        self.assertFalse(any(r[1] == "/completion" for r in model.calls))
        self.assertFalse(model.status()["research_shared_prefix"]["native_prefix_admitted"])
        self.assertFalse(model.status()["research_shared_prefix"]["http_path_numerically_qualified"])


class NativeAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.full = self.root / "full"; self.full.mkdir(); self.guard = self.root / "guard.json"
        params = {**p.EXPECTED_PARAMETERS, "n_batch": 16, "n_ubatch": 16, **p.VARIANT_PARAMETERS}
        self.report = {"complete": True, "mode": "full", "microbatch": 16, "state_parameters": params}
        self.manifest = {"state_parameters": params, "prefix_tokens": list(IDS), "render_options": {
            "reasoning_effort_override": None, "native_add_special": True, "native_parse_special": True,
            "native_environment": {"RONIN_PREFILL_DIAGNOSTIC": "1"}}}
        self.save()
        self.gate = p.NativeTail16Gate(self.root, self.full, self.guard, cpu8_bridge={"synthetic": "not real evidence"})
        self.bridge = {"passed": True, "errors": [], "criteria": {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0},
                       "reference_microbatch": 8, "candidate_microbatch": 16, "checkpoint208_compared": True,
                       "build_binding": {"validated": True}}
    def tearDown(self): self.temp.cleanup()
    def save(self):
        (self.full / "report.json").write_text(json.dumps(self.report))
        self.guard.write_text(json.dumps({"preflight": {"snapshot_manifest": self.manifest, "source_sha256": {"synthetic": "not real evidence"}}}))
    def test_mocked_valid_bridge_only_can_admit_no_common_cache_without_B_files(self):
        with mock.patch.object(p, "strict_variant_bridge", return_value=self.bridge) as bridge, \
             mock.patch.object(p, "strict_b8_comparison", side_effect=AssertionError("must not require restore")):
            result = self.gate.verify()
        bridge.assert_called_once()
        self.assertEqual(result["native_microbatch"], 16)
        self.assertFalse(result["common_cache_admitted"])
        self.assertIsNone(result["comparison"])
    def test_incomplete_full_or_failed_bridge_never_admits_no_cache(self):
        self.report["complete"] = False; self.save()
        with mock.patch.object(p, "strict_variant_bridge") as bridge, self.assertRaises(p.AdapterError): self.gate.verify()
        bridge.assert_not_called()
        self.report["complete"] = True; self.save()
        with mock.patch.object(p, "strict_variant_bridge", return_value={**self.bridge, "passed": False}), self.assertRaises(p.AdapterError): self.gate.verify()
    def test_common_cache_missing_restore_is_refused_despite_passing_bridge(self):
        gate = p.NativeTail16Gate(self.root, self.full, self.guard, cpu8_bridge={}, common_cache=True)
        with mock.patch.object(p, "strict_variant_bridge", return_value=self.bridge) as bridge, self.assertRaises(p.AdapterError): gate.verify()
        bridge.assert_not_called()
    def test_no_cache_cannot_silently_ignore_supplied_restore_receipts(self):
        with self.assertRaises(p.AdapterError):
            p.NativeTail16Gate(self.root, self.full, self.guard, cpu8_bridge={}, restored=self.root / "failed_restore")
    def test_unknown_policy_or_missing_diagnostic_flag_is_refused(self):
        self.manifest["state_parameters"]["prefill_chunk_policy_version"] = 2; self.save()
        with mock.patch.object(p, "strict_variant_bridge") as bridge, self.assertRaises(p.AdapterError): self.gate.verify()
        bridge.assert_not_called()
        self.manifest["state_parameters"]["prefill_chunk_policy_version"] = 1
        self.manifest["render_options"]["native_environment"] = {}; self.save()
        with mock.patch.object(p, "strict_variant_bridge") as bridge, self.assertRaises(p.AdapterError): self.gate.verify()
        bridge.assert_not_called()

    def test_bridge_success_does_not_hide_closed_metadata_mutation(self):
        def changed(*args):
            path = self.full / "report.json"; path.write_text(path.read_text() + " ")
            return self.bridge
        with mock.patch.object(p, "strict_variant_bridge", side_effect=changed), self.assertRaises(p.AdapterError): self.gate.verify()


class ChunkContract(unittest.TestCase):
    def test_HTTP_two_tasks_reproduce_native_v1_chunks_for_all_legal_prompt_lengths(self):
        # Native v1: physical=min(16,remaining), odd>8 is rounded to8 before
        # the remainder. HTTP has ordinary16 chunks; its second task supplies
        # precisely the final1/3/5/7 after a bounded prefill-to8 boundary.
        for total in range(209, 1537):
            for cached in (0, 208):
                native = []; offset = cached
                while offset < total:
                    take = min(16, total-offset)
                    if take > 8 and take % 2: take -= take % 8
                    native.append((offset, take)); offset += take
                plan = p.tail16_plan(total, cached)
                http = []; offset = cached
                ends = [plan["prepare_end"], total] if plan["prepare_end"] is not None else [total]
                for end in ends:
                    while offset < end:
                        take = min(16, end-offset); http.append((offset, take)); offset += take
                self.assertEqual(http, native, (total, cached))


class RunnerContract(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("isolated_tail16_campaign_fixture", HERE / "campaign.py")
        self.c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = self.c; spec.loader.exec_module(self.c)
    def test_default_plan_is16_no_common_cache_and_no_claimed_proof(self):
        report = self.c.plan()
        self.assertFalse(report["executed"])
        self.assertFalse(report["common_cache_mode"])
        self.assertFalse(report["native_full16_bridge_qualified"])
        self.assertFalse(report["native_B_qualified"])
        self.assertEqual(report["HTTP_microbatch"], 16)
    def test_full_B_is_not_required_for_no_common_cache_C_criterion(self):
        qa = type("TasksOnly", (), {"TASKS": ()})
        report = {"requests": [], "native_full16_bridge_qualified": True, "native_B_qualified": False,
                  "common_cache_mode": False}
        self.assertTrue(self.c.functional_checks(report, qa)["actual_selected_native_admission"])
        report["common_cache_mode"] = True
        self.assertFalse(self.c.functional_checks(report, qa)["actual_selected_native_admission"])
    def test_optin_refuses_before_any_repository_import(self):
        with mock.patch.object(self.c, "dependencies") as imported, self.assertRaises(p.AdapterError):
            self.c.run_campaign("none", "none", "none", None)
        imported.assert_not_called()


if __name__ == "__main__": unittest.main()
