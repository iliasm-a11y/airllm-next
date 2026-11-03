"""Control-flow fixtures only: no model, tokenizer or native process launched."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research.glm53_shared_prefill import qualify_agents as qa


def config():
    return {"threads": 10, "gpu_layers": 0, "batch_size": 8, "context_tokens": 1024,
            "model_path": "/fixture/model", "runtime_binary": "/fixture/server",
            "runtime_manifest_file": "/fixture/manifest", "verification_file": "/fixture/receipt"}


class BatteryGate:
    def __init__(self, expected):
        assert expected == "ac"
    def sample(self, **kwargs):
        return {"passed": False, "power": {"snapshot": {"power_source": "battery"}}}
    def summary(self):
        return {"passed": False}


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(config()))

    def tearDown(self):
        self.temp.cleanup()

    def test_battery_blocks_before_runtime_cache_help_or_inference(self):
        factory = mock.Mock(side_effect=AssertionError("must not construct runtime"))
        with mock.patch.object(qa, "BENCHMARK_LOCK", self.root / "lock"), \
             mock.patch.object(qa, "metadata_preflight", return_value={"scope": "fixture"}), \
             mock.patch.object(qa, "verify_cache_flags") as help_call, \
             mock.patch.dict(os.environ, {"RONIN_GLM_CONFIG": "previous"}):
            report = qa.run_campaign(self.path, self.root / "output", runtime_factory=factory,
                                     gate_factory=BatteryGate)
            self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous")
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error_code"], "POWER_CONTEXT_UNAVAILABLE")
        self.assertEqual(report["requests"], [])
        self.assertFalse(report["functional_two_agents_qualified"])
        self.assertFalse(report["full_qualification"])
        factory.assert_not_called()
        help_call.assert_not_called()
        self.assertEqual(report, json.loads((self.root / "output/report.json").read_text()))

    def test_stricter_guards_preserved_and_full_context_required(self):
        raw = config()
        raw["limits"] = {"max_rss_bytes": 4 * 1024**3, "max_swap_growth_bytes": 16 * 1024**2,
                         "warning_swap_growth_bytes": 8 * 1024**2}
        self.path.write_text(json.dumps(raw))
        selected, _ = qa.configuration(self.path, 256)
        self.assertEqual({k: selected["limits"][k] for k in raw["limits"]}, raw["limits"])
        weakened = {**raw, "limits": {"reserve_bytes": 1, "warning_headroom_bytes": 1,
            "poll_seconds": 99, "warning_grace_seconds": 999, "headroom_grace_seconds": 999}}
        self.path.write_text(json.dumps(weakened))
        bounded, _ = qa.configuration(self.path, 256)
        defaults = vars(qa.NativeLimits())
        for key in weakened["limits"]:
            self.assertEqual(bounded["limits"][key], defaults[key])
        for key, value in (("context_tokens", 512), ("threads", True), ("gpu_layers", 1)):
            altered = {**raw, key: value}
            self.path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                qa.configuration(self.path, 256)

    def test_cache_command_only_adds_explicit_research_controls(self):
        model = object.__new__(qa.NoCacheResearchModel)
        with mock.patch.object(qa.NativeGlmModel, "_build_command", return_value=["/fixture/server", "--parallel", "1"]):
            self.assertEqual(model._build_command(123), ["/fixture/server", "--parallel", "1"] + qa.CACHE_FLAGS)
        self.assertEqual(model.research_constructed_command["argv"], ["/fixture/server", "--parallel", "1"] + qa.CACHE_FLAGS)
        self.assertIn("does not attest", model.research_constructed_command["scope"])

    def test_owned_argv_observation_checks_executable_and_pid_lifetime(self):
        popen = SimpleNamespace(pid=71001, poll=lambda: None)
        owned = SimpleNamespace(create_time=lambda: 123, is_running=lambda: True,
                                exe=lambda: "/fixture/server", cmdline=lambda: ["/fixture/server", "--parallel", "1"])
        observed = qa.observe_owned_argv(popen, "/fixture/server", process_factory=lambda pid: owned)
        self.assertTrue(observed["available"])
        self.assertEqual(observed["create_time"], 123)
        bad = qa.observe_owned_argv(popen, "/fixture/other", process_factory=lambda pid: owned)
        self.assertFalse(bad["available"])
        owned.create_time = mock.Mock(side_effect=[123, 456])
        changed = qa.observe_owned_argv(popen, "/fixture/server", process_factory=lambda pid: owned)
        self.assertFalse(changed["available"])

    def test_help_verification_fails_closed_if_control_missing(self):
        with mock.patch.object(qa.subprocess, "run", return_value=SimpleNamespace(stdout="--cache-ram", stderr="")):
            with self.assertRaises(qa.QualificationError) as raised:
                qa.verify_cache_flags("/fixture/server")
        self.assertEqual(raised.exception.code, "CACHE_CONTROLS_UNAVAILABLE")

    def test_partial_delimiter_and_wrong_conclusion_do_not_qualify(self):
        checks = qa.answer_checks(qa.TASKS[0], "</think>")
        self.assertFalse(checks["nonempty_final_answer"])
        self.assertFalse(checks["expected_result_numbers"])
        text = "</think>RONIN-OMEGA-962 : A=131, B=142, écart=11. Je choisis B."
        self.assertFalse(qa.answer_checks(qa.TASKS[1], text)["correct_conclusion"])
        valid = "</think>RONIN-ALPHA-713 : 17×24=408 ; 420-408=12. Le stock suffit."
        self.assertTrue(all(v for k, v in qa.answer_checks(qa.TASKS[0], valid).items() if k != "scope"))
        leaked = valid + " RONIN-OMEGA-962"
        self.assertFalse(qa.answer_checks(qa.TASKS[0], leaked)["peer_marker_absent"])
        self.assertFalse(qa.answer_checks(qa.TASKS[0], valid.replace("</think>", ""))["nonempty_final_answer"])

    def test_prompt_capture_observes_existing_calls_without_retokenizing(self):
        observed = []
        def http(method, path, payload=None, timeout=30):
            observed.append(path)
            return {"prompt": "exact-rendered"} if path == "/apply-template" else {"tokens": [1, 2, 3]}
        def events(payload, cancel):
            yield {"content": "answer", "tokens": [9], "stop": False}
            yield {"content": "", "tokens": [10], "stop": True, "stop_type": "eos"}
        model = SimpleNamespace(_http_json=http, _events=events)
        observer = SimpleNamespace(snapshot=lambda: {"available": False, "reason": "fixture"})
        row = {"agent_id": 1, "events": [], "tokens": [], "text": "", "final": None}
        restore = qa.capture_native(model, row, self.root, observer)
        prompt = model._http_json("POST", "/apply-template", {"messages": []})
        ids = model._http_json("POST", "/tokenize", {"content": prompt["prompt"]})
        list(model._events({"cache_prompt": False}, None))
        restore()
        self.assertEqual(observed, ["/apply-template", "/tokenize"])
        self.assertEqual(row["prompt_token_ids"], ids["tokens"])
        self.assertEqual(row["tokens"], [9, 10])
        self.assertEqual(row["text"], "answer")
        self.assertIs(model._http_json, http)
        self.assertIs(model._events, events)
        self.assertEqual(len((self.root / "agent-1-events.jsonl").read_text().splitlines()), 2)

    def test_capture_failure_preserves_partial_transcript_and_finally_observes_io(self):
        def events(payload, cancel):
            yield {"content": "partial", "tokens": [7]}
            raise qa.QualificationError("CANCELLED", "fixture cancellation")
        model = SimpleNamespace(_http_json=lambda *args: {}, _events=events)
        observer = SimpleNamespace(snapshot=mock.Mock(return_value={"available": False}))
        row = {"agent_id": 600, "events": [], "tokens": [], "text": "", "final": None}
        restore = qa.capture_native(model, row, self.root, observer)
        with self.assertRaises(qa.QualificationError):
            list(model._events({}, None))
        restore()
        self.assertEqual(row["tokens"], [7])
        self.assertEqual(row["text"], "partial")
        self.assertIsNone(row["final"])
        self.assertEqual(observer.snapshot.call_count, 2)

    def test_missing_terminal_safe_in_failure_audit(self):
        rows = [{"agent_id": 1, "final": None}, {"agent_id": 600, "final": None}]
        checks = qa.campaign_checks({"requests": rows})
        self.assertFalse(checks["natural_eos_each"])
        self.assertFalse(checks["full_prompts_no_truncation"])
        self.assertFalse(checks["complete_transcripts"])

    def test_mid_request_failure_with_null_terminal_saves_audit_and_restores_environment(self):
        class Gate(BatteryGate):
            def sample(self, **kwargs):
                return {"passed": True}
            def summary(self):
                return {"passed": False, "scope": "fixture only"}
        class Awake:
            def __init__(self, owner):
                self.record = {}
            def __enter__(self):
                return self
            def check(self):
                pass
            def close(self):
                pass
        class Process:
            pid = 71001
            code = None
            def poll(self):
                return self.code
            def send_signal(self, unused):
                self.code = 0
            def wait(self, **kwargs):
                return self.code
        process = Process()
        def original_http(method, path, payload=None, timeout=30):
            return {"prompt": "fixture"} if path == "/apply-template" else {"tokens": [1]}
        def original_events(payload, cancel):
            yield {"content": "partial", "tokens": [7]}
            error = qa.QualificationError("CANCELLED", "fixture cancellation")
            error.details = {"fixture_detail": 13}
            raise error
        model = SimpleNamespace(_process=process, _port=1, _samples=[], _http_json=original_http,
                                _events=original_events, _build_command=lambda port: ["fixture"])
        engine = SimpleNamespace(_model=model, load=lambda *args: None, status=lambda: {},
                                 close=lambda: setattr(process, "code", 0), request_shutdown=lambda: None)
        database = SimpleNamespace(close=mock.Mock())
        runtime = SimpleNamespace(engine=engine, cancels={}, store=SimpleNamespace(db=database),
                                  create=lambda request: {"id": "mission"})
        def dispatch(method, parameters):
            if method == "agent_identities":
                return list(range(600))
            if method == "generate":
                model._http_json("POST", "/apply-template", {})
                model._http_json("POST", "/tokenize", {})
                list(model._events({"n_predict": 256}, None))
            if method == "shutdown":
                engine.close()
            return {}
        runtime.dispatch = dispatch
        receipt = self.root / "receipt.json"
        receipt.write_text("{}")
        selected = config()
        selected["verification_file"] = str(receipt)
        self.path.write_text(json.dumps(selected))
        observer = SimpleNamespace(snapshot=lambda: {"available": False})
        with mock.patch.object(qa, "BENCHMARK_LOCK", self.root / "lock"), \
             mock.patch.object(qa, "metadata_preflight", return_value={}), \
             mock.patch.object(qa, "verify_cache_flags", return_value={}), \
             mock.patch.object(qa.time, "sleep"), \
             mock.patch.dict(os.environ, {"RONIN_GLM_CONFIG": "previous"}):
            report = qa.run_campaign(self.path, self.root / "failed-run", runtime_factory=lambda home: runtime,
                                     gate_factory=Gate, awake_factory=Awake, io_factory=lambda pid: observer)
            self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_code"], "CANCELLED")
        self.assertEqual(report["error_details"], {"fixture_detail": 13})
        self.assertEqual(report["requests"][0]["error_details"], {"fixture_detail": 13})
        self.assertEqual(report["requests"][0]["native_error_details"], {"fixture_detail": 13})
        self.assertIsNone(report["requests"][0]["final"])
        self.assertEqual(report["requests"][0]["tokens"], [7])
        self.assertIs(model._http_json, original_http)
        self.assertIs(model._events, original_events)
        database.close.assert_called_once()
        self.assertEqual(report, json.loads((self.root / "failed-run/report.json").read_text()))


if __name__ == "__main__":
    unittest.main()
