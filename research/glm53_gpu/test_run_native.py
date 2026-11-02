"""Portable orchestration regression tests; no native process or GPU executes."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.glm53_gpu import run_native


class NativeOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fixture, not a usable model")
        self.reference = self.root / "reference"
        self.reference.mkdir()
        (self.reference / "tokens.jsonl").write_text('{"token": 11}\n')
        self.args = SimpleNamespace(
            policy="head-only", gpu_layers=0, cache_mib=0,
            output=self.root / "experiment", reference=self.reference,
            tokens=1, layer_local_hc=False, disable_hc_post=False,
            cache_lookup_split=False, probe=self.root / "unused-probe",
            model=self.model, threads=10, prompt="fixture")
        self.completed = {"status": "completed", "guard_violation": None}

    def tearDown(self):
        self.temp.cleanup()

    def comparison(self, passed):
        return {"passed": passed, "full_logits_summary": {"passed": passed}, "coverage": {}}

    def run_quietly(self):
        with redirect_stdout(StringIO()):
            return run_native.run(self.args)

    def test_completed_guard_still_exits_three_on_rejected_fidelity(self):
        with patch.object(run_native, "guard", return_value=self.completed) as guard, \
             patch("research.glm53_m5.compare_fidelity.compare",
                   return_value=self.comparison(False)) as compare:
            self.assertEqual(self.run_quietly(), 3)
        guard.assert_called_once()
        compare.assert_called_once_with(self.reference, self.args.output / "capture", "metal")
        manifest = json.loads((self.args.output / "experiment.json").read_text())
        self.assertEqual(manifest["guard_status"], "completed")
        self.assertFalse(manifest["fidelity_passed"])
        self.assertFalse(json.loads((self.args.output / "comparison.json").read_text())["passed"])

    def test_forcing_uses_actual_reference_inputs_instead_of_argmax(self):
        (self.reference / "tokens.jsonl").write_text(
            '{"token": 99, "next_input_token": 7}\n'
            '{"token": 98, "next_input_token": 8}\n'
            '{"token": 9}\n')
        self.args.tokens = 3
        observed = []

        def guard(args):
            forced = Path(args.command[7])
            observed.extend(json.loads(line)["token"] for line in forced.read_text().splitlines())
            return self.completed

        with patch.object(run_native, "guard", side_effect=guard), \
             patch("research.glm53_m5.compare_fidelity.compare", return_value=self.comparison(True)):
            self.assertEqual(self.run_quietly(), 0)
        self.assertEqual(observed, [7, 8, 9])

    def test_experiment_environment_is_clean_and_restored_after_success(self):
        self.args.reference = None
        inherited = {
            "AIRLLM_UNRELATED": "leaked", "AIRLLM_GLM_HC_POST": "0",
            "GGML_METAL_LEAK": "leaked", "LLAMA_ARG_THREADS": "1",
            "DYLD_INSERT_LIBRARIES": "/unused/injection", "UNCHANGED_FIXTURE": "keep",
        }

        def guard(args):
            for key in ("AIRLLM_UNRELATED", "AIRLLM_GLM_HC_POST", "GGML_METAL_LEAK",
                        "LLAMA_ARG_THREADS", "DYLD_INSERT_LIBRARIES"):
                self.assertNotIn(key, os.environ)
            self.assertEqual(os.environ["UNCHANGED_FIXTURE"], "keep")
            self.assertEqual(os.environ["AIRLLM_MMAP_DEMAND"], "1")
            self.assertEqual(os.environ["AIRLLM_DENSE_PREFETCH_MAX_COLS"], "1")
            self.assertEqual(os.environ["AIRLLM_GPU_CACHE_MIB"], "0")
            os.environ["GUARD_ADDED_FIXTURE"] = "must not leak"
            return self.completed

        with patch.dict(os.environ, inherited, clear=True):
            before = dict(os.environ)
            with patch.object(run_native, "guard", side_effect=guard):
                self.assertEqual(self.run_quietly(), 0)
            self.assertEqual(dict(os.environ), before)

    def test_environment_is_restored_even_when_guard_raises(self):
        inherited = {"GGML_EXPERIMENT": "original", "AIRLLM_EXPERIMENT": "original",
                     "UNCHANGED_FIXTURE": "keep"}

        def failing_guard(args):
            self.assertNotIn("GGML_EXPERIMENT", os.environ)
            self.assertNotIn("AIRLLM_EXPERIMENT", os.environ)
            os.environ["GUARD_ADDED_FIXTURE"] = "must not leak"
            raise RuntimeError("fixture guard failure")

        with patch.dict(os.environ, inherited, clear=True):
            before = dict(os.environ)
            with patch.object(run_native, "guard", side_effect=failing_guard), \
                 patch("research.glm53_m5.compare_fidelity.compare") as compare:
                with self.assertRaisesRegex(RuntimeError, "fixture guard failure"):
                    self.run_quietly()
                compare.assert_not_called()
            self.assertEqual(dict(os.environ), before)

    def test_existing_experiment_is_refused_without_running_or_overwriting(self):
        self.args.output.mkdir()
        receipt = self.args.output / "receipt.txt"
        receipt.write_text("preserve this evidence")
        with patch.object(run_native, "guard") as guard, \
             patch("research.glm53_m5.compare_fidelity.compare") as compare:
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                self.run_quietly()
        guard.assert_not_called()
        compare.assert_not_called()
        self.assertEqual(receipt.read_text(), "preserve this evidence")
        self.assertEqual(list(self.args.output.iterdir()), [receipt])


if __name__ == "__main__":
    unittest.main()
