"""Campaign planning/cancellation fixtures only; never launches a model."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import json
import hashlib
import signal
import subprocess
import sys
import unittest

from research.glm53_gpu_v2.campaign import cancel_owned, make_plan, summarize


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fixture")
        self.library = self.root / "libggml-fixture.dylib"
        self.library.write_bytes(b"fixture library")
        stat = self.model.stat()
        metrics = {"passed": True, "finite": True, "bit_mismatch_count": 0, "compared": 4}
        self.report = {
            "fidelity_passed": True, "guard_status": "completed", "model_file_stats_unchanged": True,
            "full_logits": metrics, "coverage": {k: metrics for k in ("routes", "router", "kda")},
            "environment": {"AIRLLM_METAL_Q8_CPU_ARITH": "1"}, "command": ["probe", "head-only"],
            "native_libraries_sha256": {self.library.name: hashlib.sha256(self.library.read_bytes()).hexdigest()},
            "model_files": [{"file": self.model.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}
        self.fidelity = self.root / "experiment.json"
        self.fidelity.write_text(json.dumps(self.report))
        self.args = SimpleNamespace(output=self.root / "campaign", candidate="head-inline", tail_layers=None,
                                    tokens=16, cycles=1, candidate_fidelity=self.fidelity, model=self.model,
                                    binary=self.root / "server", prompt="La capitale de la France est")

    def tearDown(self):
        self.temp.cleanup()

    def test_abba_fresh_reports_and_current_python_bounded_commands(self):
        plan = make_plan(self.args)
        self.assertEqual(plan["sequence"], "ABBA")
        self.assertEqual([j["arm"] for j in plan["jobs"]], list("ABBA"))
        self.assertEqual(len({j["report"] for j in plan["jobs"]}), 4)
        for job in plan["jobs"]:
            command = job["command"]
            self.assertEqual(command[0], sys.executable)
            for option, value in (("--threads", "10"), ("--context", "512"), ("--batch", "8"),
                                  ("--max-rss-gib", "8.5"), ("--max-swap-mib", "32"), ("--timeout", "900")):
                self.assertEqual(command[command.index(option) + 1], value)
            self.assertEqual("--head-only" in command, job["arm"] == "B")
        self.assertEqual(plan["fidelity"]["captured_steps"], 4)

    def test_unpassed_or_wrong_candidate_fidelity_and_reused_output_are_refused(self):
        for change in ({"fidelity_passed": False}, {"coverage": {}},
                       {"environment": {"AIRLLM_METAL_Q8_CPU_ARITH": "2"}}):
            report = dict(self.report, **change)
            self.fidelity.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                make_plan(self.args)
        self.fidelity.write_text(json.dumps(self.report))
        self.args.output.mkdir()
        with self.assertRaises(ValueError):
            make_plan(self.args)

    def test_model_stat_drift_rejects_a_previously_passed_gate(self):
        self.library.write_bytes(b"changed library")
        with self.assertRaisesRegex(ValueError, "native library hashes"):
            make_plan(self.args)
        self.library.write_bytes(b"fixture library")
        self.model.write_bytes(b"different fixture")
        with self.assertRaisesRegex(ValueError, "model file stats differ"):
            make_plan(self.args)

    def test_cancel_signals_only_verified_owned_benchmark_group_and_waits(self):
        child = Mock(pid=4321)
        child.poll.return_value = None
        process = Mock()
        process.create_time.return_value = 123.25
        kill = Mock()
        result = cancel_owned(child, {"pid": 4321, "create_time": 123.25},
                              get_process=lambda pid: process, get_group=lambda pid: pid, kill_group=kill)
        self.assertEqual(result["status"], "interrupted_and_exited")
        kill.assert_called_once_with(4321, signal.SIGINT)
        child.wait.assert_called_once_with(timeout=20)
        kill.reset_mock()
        result = cancel_owned(child, {"pid": 4321, "create_time": 100},
                              get_process=lambda pid: process, get_group=lambda pid: pid, kill_group=kill)
        self.assertEqual(result["status"], "ownership_mismatch_no_signal")
        kill.assert_not_called()
        child.wait.side_effect = subprocess.TimeoutExpired("fixture", 20)
        result = cancel_owned(child, {"pid": 4321, "create_time": 123.25},
                              get_process=lambda pid: process, get_group=lambda pid: pid, kill_group=kill)
        self.assertEqual(result["status"], "cleanup_pending_no_unverified_forced_kill")
        self.assertEqual(kill.call_count, 1)  # SIGINT only; no unverified SIGKILL.

    def test_summary_keeps_raw_samples_and_rejects_token_divergence(self):
        plan = make_plan(self.args)
        stages = []
        for rate in (1., 2., 2., 1.):
            stages.append({"status": "complete", "report_data": {"status": "complete", "guard_violation": None, "peak_swap_growth_bytes": 0,
                "requests": [{"tokens": [7, 8], "ttft_seconds": 5 / rate,
                              "observed_decode_tokens_per_second": rate}]}})
        result = summarize(plan, stages)
        self.assertTrue(result["comparison_valid_on_stated_scope"])
        self.assertFalse(result["optimization_qualified"])
        self.assertTrue(result["memory_stability_qualified"])
        self.assertEqual(result["arms"]["A"]["decode_tokens_per_second"]["values"], [1., 1.])
        self.assertEqual(result["median_differences"]["decode_tokens_per_second"]["B_minus_A"], 1.)
        stages[1]["report_data"]["requests"][0]["tokens"] = [7, 9]
        stages[1]["report_data"]["peak_swap_growth_bytes"] = 16 * 1024**2
        result = summarize(plan, stages)
        self.assertFalse(result["comparison_valid_on_stated_scope"])
        self.assertFalse(result["memory_stability_qualified"])

    def test_summary_excludes_rejected_stages_and_requires_completed_guard_clean_memory(self):
        plan = make_plan(self.args)
        stages = [{"status": "complete", "report_data": {
            "status": "complete", "guard_violation": None, "peak_swap_growth_bytes": 0,
            "requests": [{"tokens": [7, 8], "ttft_seconds": 5., "observed_decode_tokens_per_second": 1.}]}}
            for _ in range(4)]
        # A completed raw server report is insufficient when the campaign
        # rejected its required GPU instrumentation.
        stages[1]["status"] = "failed"
        result = summarize(plan, stages)
        self.assertEqual([r["index"] for r in result["arms"]["B"]["runs"]], [2])
        self.assertFalse(result["comparison_valid_on_stated_scope"])
        self.assertFalse(result["memory_stability_qualified"])
        # Even an inconsistent stage label cannot qualify a raw guard failure.
        stages[1]["status"] = "complete"
        stages[1]["report_data"]["guard_violation"] = {"rss": 10 * 1024**3}
        result = summarize(plan, stages)
        self.assertEqual([r["index"] for r in result["arms"]["B"]["runs"]], [2])
        self.assertFalse(result["memory_stability_qualified"])
        # Zero swap samples from a load-only failure do not prove stability.
        stages[1]["report_data"]["guard_violation"] = None
        stages[3]["status"] = "failed"
        self.assertFalse(summarize(plan, stages)["memory_stability_qualified"])


if __name__ == "__main__":
    unittest.main()
