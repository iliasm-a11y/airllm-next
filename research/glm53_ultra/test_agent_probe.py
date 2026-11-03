"""Probe control-flow fixtures only; no model, native process or GPU is launched."""
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from research.glm53_ultra import agent_probe


class FakeProcess:
    pid = 71001

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


class FakeEngine:
    def __init__(self):
        self._model = None
        self.process = FakeProcess()
        self.requests = 0
        self.close_count = 0

    def status(self):
        return {"native": {"counters": {"loads": int(self.requests > 0), "requests": self.requests},
                           "scheduler": {"physical_limit": 1}}}

    def close(self):
        self.close_count += 1
        self.process.returncode = 0
        if self._model is not None:
            self._model._process = None


class FakeRuntime:
    """Deterministic responses exercise bookkeeping and are not GLM outputs."""
    fail_load = False
    omit_memory = False

    def __init__(self, home):
        self.engine = FakeEngine()
        self.store = SimpleNamespace(db=SimpleNamespace(close=mock.Mock()))
        self.memories = {}
        self.decisions = {}
        self.missions = []
        self.shutdowns = 0
        self.config = json.loads(Path(os.environ["RONIN_GLM_CONFIG"]).read_text())

    def create(self, parameters):
        mission = {"id": str(len(self.missions) + 1)}
        self.missions.append(mission)
        return mission

    def dispatch(self, method, parameters):
        if method == "agent_identities":
            return list(range(600))
        if method == "agent_memory":
            self.memories[parameters["id"]] = parameters["text"]
            return {}
        if method == "generate":
            if self.fail_load:
                raise RuntimeError("fixture load failed before creating a native model")
            self.engine._model = SimpleNamespace(_process=self.engine.process)
            self.engine.requests += 1
            mission = parameters["id"]
            decision_id = "fixture-" + mission
            self.decisions[decision_id] = {
                "context_sha256": mission,
                "messages": [{"role": "system", "content": "" if self.omit_memory else self.memories[mission]}],
            }
            return {"decision_id": decision_id, "tokens": [7]}
        if method == "agent_decision":
            return self.decisions[parameters["decision_id"]]
        if method == "agent_context":
            return {"decisions": []}
        if method == "shutdown":
            self.shutdowns += 1
            self.engine.close()
            return {"closed": True}
        raise AssertionError("Unexpected fixture method: " + method)


class AgentProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"gpu_layers": 0, "threads": 10,
                                          "runtime_binary": "/fixture/llama-server"}))
        self.environment = mock.patch.dict(os.environ, {"RONIN_GLM_CONFIG": "previous-config"})
        self.environment.start()
        self.lock = mock.patch.object(agent_probe, "BENCHMARK_LOCK", self.root / "fixture.lock")
        self.lock.start()
        self.power = mock.patch.object(agent_probe, "observe_power_context", return_value={
            "platform": "darwin", "guard_applicable": True, "clamshell_closed": False,
            "power_source": "Battery Power", "battery_percent": 27,
            "sources": [{"name": "fixture_only", "scope": "no hardware observation"}]})
        self.power_observer = self.power.start()
        self.instances = []

    def tearDown(self):
        self.power.stop()
        self.lock.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def factory(self, home):
        runtime = FakeRuntime(home)
        self.instances.append(runtime)
        return runtime

    def run_probe(self):
        return agent_probe.run_probe(self.config, self.root / "output", runtime_factory=self.factory)

    def saved_report(self):
        return json.loads((self.root / "output/report.json").read_text())

    def test_success_checks_actual_process_exit_and_restores_environment(self):
        report = self.run_probe()
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["owned_server_stopped"])
        self.assertEqual(report["owned_server_returncode"], 0)
        self.assertEqual(report["owned_pid_before_cleanup"], FakeProcess.pid)
        self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous-config")
        self.instances[0].store.db.close.assert_called_once()
        self.assertEqual(self.saved_report(), report)

    def test_load_failure_before_model_still_closes_database_and_saves_error(self):
        with mock.patch.object(FakeRuntime, "fail_load", True):
            report = self.run_probe()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_code"], "RuntimeError")
        self.assertIsNone(report["owned_server_stopped"])
        self.assertEqual(self.instances[0].shutdowns, 1)
        self.instances[0].store.db.close.assert_called_once()
        self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous-config")
        self.assertEqual(self.saved_report(), report)

    def test_constructor_failure_still_restores_environment_and_writes_report(self):
        def fail(home):
            raise RuntimeError("fixture constructor failure")
        report = agent_probe.run_probe(self.config, self.root / "output", runtime_factory=fail)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "fixture constructor failure")
        self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous-config")
        self.assertEqual(self.saved_report(), report)

    def test_status_and_shutdown_errors_do_not_skip_native_close_or_database(self):
        process = FakeProcess()
        engine = SimpleNamespace(_model=SimpleNamespace(_process=process),
                                 status=mock.Mock(side_effect=RuntimeError("status failed")),
                                 close=mock.Mock(side_effect=lambda: setattr(process, "returncode", 0)))
        database = SimpleNamespace(close=mock.Mock())
        runtime = SimpleNamespace(engine=engine, store=SimpleNamespace(db=database),
                                  dispatch=mock.Mock(side_effect=RuntimeError("shutdown failed")))
        report = {"status": "completed"}
        agent_probe._cleanup(runtime, report)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["owned_server_stopped"])
        self.assertEqual({error["step"] for error in report["cleanup_errors"]},
                         {"engine_status", "runtime_shutdown"})
        engine.close.assert_called_once()
        database.close.assert_called_once()

    def test_cleared_process_pointer_does_not_fabricate_process_exit(self):
        process = FakeProcess()
        model = SimpleNamespace(_process=process)
        runtime = SimpleNamespace(
            engine=SimpleNamespace(_model=model, status=lambda: {},
                                   close=lambda: setattr(model, "_process", None)),
            store=SimpleNamespace(db=SimpleNamespace(close=mock.Mock())), dispatch=lambda *args: {})
        report = {"status": "completed"}
        agent_probe._cleanup(runtime, report)
        self.assertFalse(report["owned_server_stopped"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cleanup_errors"][0]["error_code"], "SERVER_STILL_RUNNING")

    def test_isolation_requires_both_private_markers_to_be_present(self):
        with mock.patch.object(FakeRuntime, "omit_memory", True):
            report = self.run_probe()
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["checks"]["host_contexts_isolated"])

    def test_stricter_configured_guards_are_never_relaxed(self):
        config = json.loads(self.config.read_text())
        config["limits"] = {"max_rss_bytes": 4 * 1024**3,
                            "max_swap_growth_bytes": 16 * 1024**2,
                            "warning_swap_growth_bytes": 8 * 1024**2}
        self.config.write_text(json.dumps(config))
        report = self.run_probe()
        self.assertEqual(report["guard_limits"], config["limits"])
        self.assertEqual(self.instances[0].config["limits"], config["limits"])

    def test_sigint_and_sigterm_clean_up_and_restore_handlers_without_real_signals(self):
        original_dispatch = FakeRuntime.dispatch
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                original_handler = signal.getsignal(signum)

                def dispatch(runtime, method, parameters):
                    if method in ("generate", "shutdown"):
                        # Invoke the installed handler directly; never signal a process.
                        signal.getsignal(signum)(signum, None)
                    return original_dispatch(runtime, method, parameters)

                output = self.root / ("interrupted-" + str(signum))
                with mock.patch.object(FakeRuntime, "dispatch", dispatch):
                    report = agent_probe.run_probe(self.config, output, runtime_factory=self.factory)
                self.assertEqual(report["status"], "cancelled")
                self.assertEqual(report["error_code"], "CANCELLED")
                self.assertEqual(report["interrupted_signal"], signum)
                self.assertEqual(self.instances[-1].shutdowns, 1)
                self.instances[-1].store.db.close.assert_called_once()
                self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous-config")
                self.assertIs(signal.getsignal(signum), original_handler)
                self.assertEqual(json.loads((output / "report.json").read_text()), report)

    def test_closed_or_unknown_lid_refuses_before_constructing_runtime(self):
        for state, code in ((True, "POWER_CONTEXT_CLOSED"), (None, "POWER_CONTEXT_UNKNOWN")):
            with self.subTest(clamshell_closed=state):
                observation = {"platform": "darwin", "guard_applicable": True,
                               "clamshell_closed": state, "power_source": "Battery Power",
                               "battery_percent": 27, "sources": [{"name": "fixture_only"}]}
                self.power_observer.return_value = observation
                output = self.root / code
                report = agent_probe.run_probe(self.config, output, runtime_factory=self.factory)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["error_code"], code)
                self.assertEqual(report["power_context_before_runtime"], observation)
                self.assertEqual(self.instances, [])
                self.assertEqual(os.environ["RONIN_GLM_CONFIG"], "previous-config")
                self.assertEqual(json.loads((output / "report.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
