"""Small source/protocol mocks only: no power query, native/model/GPU or build."""
from dataclasses import dataclass
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prefix_adapter as p
import guards as g
spec = importlib.util.spec_from_file_location("prefix_campaign_fixture", HERE / "campaign.py")
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


@dataclass
class Limits:
    reserve_bytes: int = 768 * p.MiB
    base_workspace_bytes: int = 1024 * p.MiB


class Power:
    def __init__(self, *args): self.violation = None; self.calls = []
    def sample(self, *, include_power=True, final=False):
        self.calls.append(include_power)
        return {"passed": self.violation is None}
    def summary(self): return {"passed": self.violation is None}


def observed(*args):
    return {"available": True, "observed_matching_lock_fd": True}


class Process:
    def __init__(self, pid): self.pid = pid
    def exe(self): return "/synthetic/server"
    def create_time(self): return 123.0
    def is_running(self): return True
    def status(self): return "running"
    def ppid(self): return os.getpid()
    def uids(self): return SimpleNamespace(real=os.getuid())


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "fixture-uid.lock"
        self.power = Power()
        self.guard = g.PhaseAwareGuard(self.power, Limits, process_factory=Process, fd_observer=observed, clock=lambda: 10.0)
        self.guard._lock_path = lambda model: self.path
        self.model = SimpleNamespace(_process=None, _lock_file=None, limits=Limits(),
            context_tokens=2048, batch_size=8, runtime_binary=Path("/synthetic/server"),
            _owned_process_generation=1, _prefix_admission_sample={"available_bytes": 8 * 1024 ** 3})

    def tearDown(self):
        if self.model._lock_file is not None: self.model._lock_file.close()
        self.temp.cleanup()

    def loaded(self):
        self.model._lock_file = self.path.open("a")
        fcntl.flock(self.model._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.model._process = SimpleNamespace(pid=123, poll=lambda: None)

    def test_preload_probe_released_before_native_lock_acquisition(self):
        row = self.guard(self.model, phase="preload")
        self.assertTrue(row["preload_probe_released"])
        self.assertFalse(row["uid_exclusive"])
        self.assertFalse(row["identity_passed"])
        self.loaded()  # A competing second flock would fail if probe remained.
        row = self.guard(self.model, phase="loaded")
        self.assertTrue(row["uid_exclusive"])
        self.assertTrue(row["identity_passed"])

    def test_conflicting_preload_lock_is_refused_without_claimed_ownership(self):
        with self.path.open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(p.AdapterError): self.guard(self.model, phase="preload")
        self.assertIsNone(self.guard.owner)

    def test_pending_native_fd_is_not_relocked_or_called_child_ownership(self):
        self.model._lock_file = self.path.open("a")
        fcntl.flock(self.model._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = self.guard(self.model, phase="preload")
        self.assertFalse(row["uid_exclusive"])
        self.assertIn("native_lock_pending_spawn", row)

    def test_closed_descriptor_or_unobserved_child_fd_is_refused(self):
        self.loaded()
        self.guard.fd_observer = lambda *args: {"available": True, "observed_matching_lock_fd": False}
        with self.assertRaises(p.AdapterError): self.guard(self.model, phase="loaded")

    def test_child_generation_change_and_limit_mutation_are_refused(self):
        self.loaded()
        self.guard(self.model, phase="loaded")
        self.model._owned_process_generation += 1
        with self.assertRaises(p.AdapterError): self.guard(self.model, phase="loaded")

    def test_warning_pressure_not_rejected_when_original_native_admission_passed(self):
        self.loaded()
        self.model._prefix_admission_sample["pressure_level"] = 2
        self.assertTrue(self.guard(self.model, phase="loaded")["passed"])

    def test_additional_io_reserve_is_admitted_before_snapshot_io(self):
        self.loaded()
        self.model._prefix_admission_sample["available_bytes"] = (
            self.model.limits.reserve_bytes + self.model.limits.base_workspace_bytes +
            2048 * 64 * 1024 + 8 * 4 * p.MiB + g.SNAPSHOT_IO_RESERVE - 1)
        with self.assertRaises(p.AdapterError): self.guard(self.model, phase="loaded")

    def test_latched_power_failure_remains_fatal_without_next_full_query(self):
        self.guard(self.model, phase="preload")
        self.power.violation = {"fixture": "power changed"}
        with self.assertRaises(p.AdapterError): self.guard(self.model, phase="preload")


class CampaignTests(unittest.TestCase):
    def test_default_plan_is_nonexecuting_and_HTTP16_unimplemented(self):
        report = c.plan()
        self.assertFalse(report["executed"])
        self.assertEqual(report["planned_agent_order"], [1, 600, 1])
        self.assertFalse(report["native_B_qualified"])
        self.assertFalse(report["HTTP16_implemented"])

    def test_execution_requires_optin_before_dependency_import(self):
        with mock.patch.object(c, "dependencies") as imported:
            with self.assertRaises(p.AdapterError): c.run_campaign("missing", "missing", "missing", None)
        imported.assert_not_called()

    def test_mismatched_checkout_refused_before_importing_delegated_modules(self):
        gate = p.NativePrefixGate("/isolated/a", "full", "restore", "a", "b")
        with mock.patch.object(c, "dependencies") as imported:
            with self.assertRaises(p.AdapterError):
                c.run_campaign("/isolated/b", "unused", "unused", gate, execute_research=True)
        imported.assert_not_called()

    def test_uid_observer_is_explicit_pinned_sibling_not_missing_repository_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repository = root / "airllm-next"; repository.mkdir()
            path = root / "uid_supervisor.py"
            path.write_text("from pathlib import Path\nREPO=Path(__file__).resolve().parent/'airllm-next'\ndef observe_lock_fd(*args): return {'fixture': True}\n")
            sys.modules.pop("ronin_prefix_existing_uid_observer", None)
            try:
                with mock.patch.object(c, "UID_OBSERVER_SHA256", p.sha(path)):
                    module = c.load_uid_observer(repository)
                    self.assertEqual(module.REPO, repository.resolve())
                    path.write_text(path.read_text() + "# changed\n")
                    with self.assertRaises(p.AdapterError): c.load_uid_observer(repository)
            finally:
                sys.modules.pop("ronin_prefix_existing_uid_observer", None)

    def test_failed_B_gate_prevents_Runtime_factory_and_retains_blocked_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qa = SimpleNamespace(configuration=lambda *a, **kw: ({"context_tokens": 2048, "limits": {}}, "0" * 64),
                PowerGate=Power, BENCHMARK_LOCK=root / "fixture-benchmark.lock", TASKS=(),
                power_evidence_matches=lambda *a: False)
            gate = p.NativePrefixGate(root, root, root, root, root)
            with mock.patch.object(gate, "verify", side_effect=p.AdapterError("B8_UNQUALIFIED", "incomplete real evidence")), \
                 mock.patch.object(c, "source_hashes", return_value={}), \
                 mock.patch.object(c, "plan", return_value=c.plan()), \
                 mock.patch.object(qa, "Runtime", create=True) as runtime:
                report = c.run_campaign(root, "unused", root / "out", gate, execute_research=True,
                    deps=SimpleNamespace(qa=qa, hashes={}))
            runtime.assert_not_called()
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["executed"])
            self.assertFalse(report["native_B_qualified"])
            self.assertFalse(report["functional_C_qualified"])
            self.assertTrue((root / "out/report.json").is_file())

    def test_physical_journal_extracts_real_ids_and_counts_metadata_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "physical.jsonl"
            rows = [
                {"logical": None, "phase": "prepare_common_prefix", "status": "started", "t_campaign": 1},
                {"logical": 1, "phase": "apply_template", "status": "completed", "response": {"prompt": "full host text"}},
                {"logical": 1, "phase": "native_tokenize", "status": "completed", "response": {"tokens": [1, 2, 3]}},
                {"logical": 1, "phase": "restore_prefix", "status": "started", "t_campaign": 2},
                {"logical": 1, "phase": "generate", "status": "started", "t_campaign": 3, "payload": {"n_predict": 512}},
                {"logical": 1, "phase": "generate_event", "t_campaign": 5, "response": {"tokens": [4], "content": "raw"}},
                {"logical": 1, "phase": "generate_event", "t_campaign": 6, "response": {"tokens": [5], "stop": True, "stop_type": "eos"}},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            row = c.read_physical(path, 1)
            self.assertEqual(row["tokens"], [4, 5])
            self.assertEqual(row["events"][0]["t"], 2)
            self.assertEqual(len(row["physical_requests"]), 2)
            self.assertEqual(row["prompt_token_ids"], [1, 2, 3])

    def test_history_requires_same_mission_exact_stored_output_and_private_markers(self):
        tasks = ({"agent_id": 1, "marker": "ALPHA"}, {"agent_id": 600, "marker": "OMEGA"})
        def row(agent, mission, history=()):
            marker = "ALPHA" if agent == 1 else "OMEGA"
            messages = [{"role": "system", "content": marker}] + list(history) + [{"role": "user", "content": "question"}]
            return {"agent_id": agent, "mission_id": mission, "text": marker,
                    "rendered_prompt": json.dumps(messages), "decision": {"messages": messages, "output_text": "stored sanitized output"}}
        rows = [row(1, "A"), row(600, "B"), row(1, "A", ({"role": "user", "content": "question"},
            {"role": "assistant", "content": "stored sanitized output"}))]
        self.assertTrue(all(c.history_checks(rows, tasks).values()))
        rows[2]["decision"]["messages"][2]["content"] = "raw substituted output"
        self.assertFalse(c.history_checks(rows, tasks)["agent1_full_history_returned"])
        rows[2]["rendered_prompt"] += " OMEGA"
        self.assertFalse(c.history_checks(rows, tasks)["private_markers"])

    def test_duplicate_generation_is_refused_instead_of_picking_favorable_call(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "journal"
            row = {"logical": 1, "phase": "generate", "status": "started", "t_campaign": 1, "payload": {}}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaises(p.AdapterError): c.read_physical(path, 1)

    def test_two_observed_useful_EOS_survive_third_failure_without_global_qualification(self):
        rows = [{"agent_id": agent, "final": {"stop_type": "eos"},
                 "completion_contract": {"passed": True}, "answer_checks": {"useful": True, "scope": "fixture"}}
                for agent in (1, 600)] + [{"agent_id": 1, "error": "guard stopped return"}]
        report = {"status": "failed", "requests": rows, "functional_C_qualified": False}
        self.assertTrue(c.two_agent_observation(report)["observed"])
        self.assertFalse(report["functional_C_qualified"])
        rows[1]["final"]["stop_type"] = "limit"
        self.assertFalse(c.two_agent_observation(report)["observed"])


if __name__ == "__main__": unittest.main()
