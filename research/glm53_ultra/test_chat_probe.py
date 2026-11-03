"""Control-flow fixtures only: no GLM, subprocess, GPU or power query is run."""
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from research.glm53_ultra import chat_probe
from research.glm53_gpu_v3.power_state import PowerGate


class FakeProcess:
    pid = 71011

    def __init__(self, exit_code=0):
        self.exit_code, self.returncode, self.signals = exit_code, None, []

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = self.exit_code

    def wait(self, timeout=None):
        return self.returncode


class FakeEngine:
    mode = "eos"
    exit_code = 0
    close_error = False

    def __init__(self, *, config):
        self.config = config
        self._model = None
        self.calls = []
        self.loads = self.requests = self.closes = 0
        self.process = FakeProcess(self.exit_code)

    def load(self, path, cancel=None):
        self.loads += 1
        self._model = SimpleNamespace(_process=self.process, last_owned_process_exit=None)
        return self.status()

    def status(self):
        return {"model": {"id": "fixture-only"}, "native": {
            "runtime_sha256": "a" * 64, "runtime_artifacts_verified": True,
            "failure": {"code": "SWAP_LIMIT"} if ((self.mode == "late_guard" and self.closes)
                or (self.mode == "guard" and self.process.returncode is not None)) else None,
            "checkpoint": {"integrity_verified": True, "files": [{"fixture": True}]},
            "counters": {"loads": self.loads, "requests": self.requests},
            "scheduler": {"physical_limit": 1}}}

    def stream(self, messages, **options):
        self.calls.append((messages, options))
        if self.mode == "power":
            if not options["cancel"].wait(2):
                raise AssertionError("Fixture monitor did not cancel")
            raise chat_probe.ChatProbeError("CANCELLED", "fixture cancellation")
        if self.mode == "signal":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        if self.mode == "guard":
            yield {"tokens": [7], "content": "partial", "stop": False}
            self.process.returncode = -15
            self._model._process = None
            self._model.last_owned_process_exit = {
                "pid": self.process.pid, "observation": "popen_poll_of_owned_process",
                "exit_observed": True, "returncode": -15, "guard_failure_code": "SWAP_LIMIT"}
            raise chat_probe.ChatProbeError("SWAP_LIMIT", "fixture guard")
        budget = options["max_new_tokens"]
        ids = list(range(budget)) if self.mode == "limit" else [7]
        yield {"tokens": ids, "content": "OK", "stop": False}
        last_ids = [] if self.mode in ("limit", "missing") else [8]
        count = budget if self.mode == "limit" else 2
        yield {"tokens": last_ids, "content": "", "stop": True, "tokens_predicted": count,
               "stop_type": "limit" if self.mode == "limit" else "eos",
               "truncated": self.mode == "truncated", "timings": {"predicted_n": count}}
        self.requests += 1

    def close(self):
        self.closes += 1
        if self.close_error:
            raise RuntimeError("fixture close failure")
        if self._model is not None:
            if self.process.returncode is None:
                self.process.returncode = -15
            self._model.last_owned_process_exit = {
                "pid": self.process.pid, "observation": "popen_poll_of_owned_process",
                "exit_observed": True, "returncode": self.process.returncode}
            self._model._process = None


class FakeAwake:
    def __init__(self, owner):
        self.record = {"started": False, "cleaned_up": False,
                       "command": ["/usr/bin/caffeinate", "-i", "-w", str(owner.pid)],
                       "target_pid": owner.pid, "target_create_time": 1,
                       "helper_pid": owner.pid + 1, "helper_create_time": 2}

    def __enter__(self):
        self.record["started"] = True
        return self

    def check(self):
        pass

    def close(self):
        self.record.update(cleaned_up=True, helper_exit_code=-15, cleanup_kind="owner_term",
                           owner_signal="SIGTERM", helper_alive_before_owner_signal=True,
                           target_exited_at_cleanup=False, exit_qualified=True)


class Observer:
    def __init__(self, fail_after=None, initially_closed=False):
        self.calls, self.fail_after, self.initially_closed = 0, fail_after, initially_closed

    def snapshot(self):
        self.calls += 1
        return {"available": True, "lid_closed": self.initially_closed,
                "power_source": "battery" if self.fail_after and self.calls > self.fail_after else "ac"}


class ChatProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.receipt = self.root / "receipt.json"
        self.receipt.write_text('{"fixture":"no model weights"}')
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"model_path": "fixture-model", "runtime_binary": "fixture-server",
            "runtime_manifest_file": "fixture-manifest", "verification_file": "receipt.json", "threads": 10,
            "gpu_layers": 0, "context_tokens": 512, "batch_size": 8}))
        self.lock = mock.patch.object(chat_probe, "BENCHMARK_LOCK", self.root / "fixture.lock")
        self.lock.start()
        self.engines, self.awake = [], []
        self.observer = Observer()

    def tearDown(self):
        self.lock.stop()
        self.temp.cleanup()

    def engine_factory(self, *, config):
        engine = FakeEngine(config=config)
        self.engines.append(engine)
        return engine

    def awake_factory(self, owner):
        awake = FakeAwake(owner)
        self.awake.append(awake)
        return awake

    def run_probe(self, **options):
        return chat_probe.run_probe(self.config, self.root / "output", engine_factory=self.engine_factory,
            gate_factory=lambda source: PowerGate(source, observer=self.observer),
            awake_factory=self.awake_factory, monitor_interval=.001, **options)

    def test_eos_uses_one_short_user_native_template_and_attested_graceful_exit(self):
        original = self.receipt.read_bytes()
        report = self.run_probe()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["eos_qualified"])
        self.assertFalse(report["native_fidelity_qualified"])
        self.assertEqual(report["request"]["tokens"], [7, 8])
        self.assertEqual(len(self.engines), 1)
        messages, options = self.engines[0].calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "Réponds uniquement par OK."}])
        self.assertEqual(options["reasoning_effort"], "low")
        self.assertEqual(options["max_new_tokens"], 128)
        self.assertEqual(self.engines[0].process.signals, [signal.SIGINT])
        self.assertEqual(report["owned_server_returncode"], 0)
        self.assertTrue(report["power_continuity_validated"])
        self.assertEqual(self.receipt.read_bytes(), original)
        self.assertNotEqual(self.engines[0].config["verification_file"], str(self.receipt))
        self.assertEqual(json.loads((self.root / "output/report.json").read_text()), report)
        self.assertEqual(len((self.root / "output/events.jsonl").read_text().splitlines()), 2)

    def test_limit_is_incomplete_and_never_eos(self):
        with mock.patch.object(FakeEngine, "mode", "limit"):
            report = self.run_probe()
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(report["completion_contract"]["passed"])
        self.assertFalse(report["eos_qualified"])

    def test_missing_ids_or_context_truncation_reject_eos(self):
        for mode in ("missing", "truncated"):
            with self.subTest(mode=mode), mock.patch.object(FakeEngine, "mode", mode):
                output = self.root / mode
                report = chat_probe.run_probe(self.config, output, engine_factory=self.engine_factory,
                    gate_factory=lambda source: PowerGate(source, observer=Observer()), awake_factory=self.awake_factory)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error_code"], "INCOMPLETE_STREAM")
                self.assertFalse(report["eos_qualified"])

    def test_power_change_cancels_stream_without_qualifying_partial_output(self):
        self.observer = Observer(fail_after=1)
        with mock.patch.object(FakeEngine, "mode", "power"):
            report = self.run_probe()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["power_continuity_validated"])
        self.assertEqual(report["power_monitor_errors"][0]["code"], "POWER_CONTEXT_CHANGED")
        self.assertTrue(self.awake[0].record["cleaned_up"])

    def test_closed_lid_refuses_before_any_engine_or_awake_helper(self):
        self.observer = Observer(initially_closed=True)
        report = self.run_probe()
        self.assertEqual(report["error_code"], "POWER_CONTEXT_UNAVAILABLE")
        self.assertEqual(self.engines, [])
        self.assertEqual(self.awake, [])

    def test_watchdog_exit_evidence_is_retained_after_handle_was_cleared(self):
        with mock.patch.object(FakeEngine, "mode", "guard"):
            report = self.run_probe()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_code"], "SWAP_LIMIT")
        self.assertEqual(report["request"]["tokens"], [7])
        self.assertEqual(report["owned_server_returncode"], -15)
        self.assertTrue(report["owned_server_stopped"])
        self.assertFalse(report["eos_qualified"])

    def test_swap_failure_preserves_established_integrity_facts_without_qualifying_eos(self):
        for fail_cleanup_status in (False, True):
            with self.subTest(cleanup_status_fails=fail_cleanup_status):
                original_status = FakeEngine.status

                def status(engine):
                    if fail_cleanup_status and engine.closes:
                        raise RuntimeError("fixture cleanup status unavailable")
                    return original_status(engine)

                with mock.patch.object(FakeEngine, "mode", "guard"), \
                     mock.patch.object(FakeEngine, "status", status):
                    report = chat_probe.run_probe(self.config, self.root / str(fail_cleanup_status),
                        engine_factory=self.engine_factory,
                        gate_factory=lambda source: PowerGate(source, observer=Observer()),
                        awake_factory=self.awake_factory)
                expected_source = "engine_after_load.native" if fail_cleanup_status else "engine_after_cleanup.native"
                self.assertEqual(report["native_evidence_source"], expected_source)
                self.assertEqual(report["error_code"], "SWAP_LIMIT")
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["eos_checks"]["model_integrity"])
                self.assertTrue(report["eos_checks"]["runtime_artifacts"])
                self.assertTrue(report["eos_checks"]["one_model_load"])
                self.assertFalse(report["eos_checks"]["complete_transcript"])
                self.assertFalse(report["eos_checks"]["natural_eos"])
                self.assertFalse(report["eos_checks"]["resource_guards_clear"])
                self.assertFalse(report["eos_qualified"])

    def test_nonzero_native_exit_or_close_failure_invalidates_eos(self):
        for setting, value in (("exit_code", -15), ("close_error", True)):
            with self.subTest(setting=setting), mock.patch.object(FakeEngine, setting, value):
                report = chat_probe.run_probe(self.config, self.root / setting, engine_factory=self.engine_factory,
                    gate_factory=lambda source: PowerGate(source, observer=Observer()), awake_factory=self.awake_factory)
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["eos_qualified"])
                self.assertTrue(self.awake[-1].record["cleaned_up"])

    def test_sigterm_cleanup_and_handler_restoration(self):
        previous = signal.getsignal(signal.SIGTERM)
        with mock.patch.object(FakeEngine, "mode", "signal"):
            report = self.run_probe()
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(report["eos_qualified"])
        self.assertTrue(self.awake[0].record["cleaned_up"])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_guard_reported_during_cleanup_still_invalidates_eos(self):
        with mock.patch.object(FakeEngine, "mode", "late_guard"):
            report = self.run_probe()
        self.assertTrue(report["completion_contract"]["passed"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["eos_checks"]["resource_guards_clear"])
        self.assertFalse(report["eos_qualified"])

    def test_strict_guards_and_relative_paths_are_preserved(self):
        config = json.loads(self.config.read_text())
        config["limits"] = {"max_rss_bytes": 4 * 1024**3, "max_swap_growth_bytes": 8 * 1024**2}
        self.config.write_text(json.dumps(config))
        report = self.run_probe(tokens=32)
        self.assertEqual(report["guard_limits"]["max_rss_bytes"], 4 * 1024**3)
        self.assertEqual(report["guard_limits"]["max_swap_growth_bytes"], 8 * 1024**2)
        self.assertEqual(self.engines[0].config["runtime_binary"], str(self.root / "fixture-server"))
        self.assertEqual(self.engines[0].config["state_dir"], str(self.root / "output/engine-state"))
        for invalid in (True, 0, 129):
            with self.assertRaises(ValueError):
                chat_probe._configuration(self.config, invalid)


if __name__ == "__main__":
    unittest.main()
