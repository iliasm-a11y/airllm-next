"""Proposed portable regression tests; fake responses do not prove GLM inference.

Intended destination: integrations/ronin/tests/test_partial_status.py after the
paired source patch is reviewed and applied. Do not run against the old source.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1] / "air_llm"), str(ROOT / "engine"),
                str(ROOT / "runtime"), str(ROOT), str(Path(__file__).resolve().parent)]
from adapter import Engine
from ronin_runtime import Runtime
from test_ronin_glm import FakeNative
import ronin_glm


class EosNative(FakeNative):
    def stream_chat(self, messages, **options):
        self.calls.append((messages, options))
        yield {"content": "OK", "tokens": [8], "stop": False}
        yield {"content": "", "tokens": [], "stop": True, "stop_type": "eos"}


class PartialStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.environment = mock.patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.runtimes = []

    def tearDown(self):
        for runtime in self.runtimes:
            try:
                runtime.dispatch("shutdown", {})
                runtime.store.db.close()
            except sqlite3.ProgrammingError:
                pass  # The CLI already closed this private fixture connection.
        self.environment.stop()
        self.temp.cleanup()

    def runtime(self, native=FakeNative):
        runtime = Runtime(self.home / str(len(self.runtimes)))
        runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=native)
        runtime.dispatch("settings", {"model_path": "/fixture/model"})
        self.runtimes.append(runtime)
        return runtime

    def test_background_run_preserves_partial_and_records_explicit_decision_state(self):
        runtime = self.runtime()
        mission = runtime.create({"prompt": "Question"})
        runtime.run(mission["id"])
        self.assertEqual(runtime.get(mission["id"])["status"], "partial")
        events = runtime.detail(mission["id"])["events"]
        self.assertNotIn("model_completed", [event["type"] for event in events])
        partial = next(event for event in events if event["type"] == "model_partial")
        self.assertEqual(json.loads(partial["body"])["decision_state"], "partial")
        self.assertEqual(runtime.engine._model.calls[0][1]["max_new_tokens"], 32)
        self.assertEqual(len(runtime.engine._model.calls), 1)

    def test_background_run_requires_eos_for_completed(self):
        runtime = self.runtime(EosNative)
        mission = runtime.create({"prompt": "Question"})
        runtime.run(mission["id"])
        self.assertEqual(runtime.get(mission["id"])["status"], "completed")
        event = next(event for event in runtime.detail(mission["id"])["events"]
                     if event["type"] == "model_completed")
        payload = json.loads(event["body"])
        self.assertEqual(payload["decision_state"], "completed")
        self.assertEqual(payload["stop_type"], "eos")

    def test_shutdown_keeps_partial_state_and_restart_does_not_resume(self):
        runtime = self.runtime()
        mission = runtime.create({"prompt": "Question"})
        runtime.run(mission["id"])
        runtime.dispatch("shutdown", {})
        self.assertEqual(runtime.get(mission["id"])["status"], "partial")
        home = runtime.store.home
        runtime.store.db.close()
        self.runtimes.remove(runtime)
        restored = Runtime(home)
        self.runtimes.append(restored)
        self.assertEqual(restored.get(mission["id"])["status"], "partial")
        self.assertEqual(restored.workers, {})
        self.assertIsNone(restored.engine._model)

    def test_stop_between_generation_and_final_state_is_not_overwritten(self):
        for native in (FakeNative, EosNative):
            with self.subTest(native=native.__name__):
                runtime = self.runtime(native)
                mission = runtime.create({"prompt": "Question"})
                original = runtime.state

                def stop_before_publish(mid, state):
                    if state in ("partial", "completed"):
                        runtime.stop(mid)
                    return original(mid, state)

                with mock.patch.object(runtime, "state", side_effect=stop_before_publish):
                    runtime.run(mission["id"])
                self.assertEqual(runtime.get(mission["id"])["status"], "stopped")

    def test_cli_returns_and_persists_explicit_partial_or_completed_state(self):
        for native, expected in ((FakeNative, "partial"), (EosNative, "completed")):
            with self.subTest(expected=expected):
                runtime = self.runtime(native)
                output = io.StringIO()
                with mock.patch.object(ronin_glm, "Runtime", return_value=runtime), \
                     mock.patch.object(sys, "argv", ["ronin_glm.py", "generate", "--prompt", "Question",
                                                      "--max-tokens", "2"]), \
                     contextlib.redirect_stdout(output):
                    self.assertEqual(ronin_glm.main(), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["decision_state"], expected)
                self.assertEqual(result["mission_state"], expected)
                with sqlite3.connect(runtime.store.path) as saved:
                    state, = saved.execute("SELECT status FROM missions WHERE id=?", (result["mission_id"],)).fetchone()
                self.assertEqual(state, expected)
                self.assertEqual(len(runtime.engine._model.calls), 1)
                self.assertEqual(runtime.engine._model.calls[0][1]["max_new_tokens"], 2)

    def test_cli_preserves_stop_between_generation_and_state_publication(self):
        runtime = self.runtime(EosNative)
        original = runtime.state

        def stop_before_publish(mid, state):
            if state in ("partial", "completed"):
                runtime.stop(mid)
            return original(mid, state)

        output = io.StringIO()
        with mock.patch.object(ronin_glm, "Runtime", return_value=runtime), \
             mock.patch.object(runtime, "state", side_effect=stop_before_publish), \
             mock.patch.object(sys, "argv", ["ronin_glm.py", "generate", "--prompt", "Question"]), \
             contextlib.redirect_stdout(output):
            self.assertEqual(ronin_glm.main(), 2)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], "CANCELLED")
        with sqlite3.connect(runtime.store.path) as saved:
            self.assertEqual(saved.execute("SELECT status FROM missions").fetchone(), ("stopped",))


if __name__ == "__main__":
    unittest.main()
