"""Portable cancellation controls without a model, native process or GPU."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1] / "air_llm"), str(ROOT / "engine"),
                str(ROOT / "runtime"), str(Path(__file__).resolve().parent)]
from adapter import Engine, NativeGlmError
from ronin_runtime import Fault, Runtime
from test_ronin_glm import FakeNative


class ControlStateTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.runtime = Runtime(self.temporary.name)
        self.mission = self.runtime.create({"prompt": "Question de contrôle"})["id"]
        self.runtime.dispatch("settings", {"model_path": "/fixture/model"})
        self.workers = []

    def tearDown(self):
        # Release fixture loops even if a preceding assertion failed. This
        # sends no signal and starts no real process.
        if self.runtime.engine:
            self.runtime.engine.request_shutdown()
        for event in self.runtime.cancels.values():
            event.set()
        for worker in self.workers:
            worker.join(2)
        self.runtime.dispatch("shutdown", {})
        self.runtime.store.db.close()
        self.temporary.cleanup()
        self.environment.stop()

    def use(self, native):
        self.runtime.engine = Engine(config={"runtime_binary": "/fixture/runtime"}, model_factory=native)

    def generate_in_thread(self, errors):
        def generate():
            try:
                self.runtime.dispatch("generate", {"id": self.mission, "max_new_tokens": 2})
            except Fault as error:
                errors.append(error.code)
        worker = threading.Thread(target=generate, daemon=True)
        self.workers.append(worker)
        worker.start()
        return worker

    def test_start_cannot_replace_event_of_active_direct_generation_and_stop_cancels_it(self):
        entered = threading.Event()
        class WaitingNative(FakeNative):
            def stream_chat(self, messages, **options):
                self.calls.append((messages, options))
                entered.set()
                while not options["cancel"].is_set():
                    time.sleep(.001)
                raise NativeGlmError("CANCELLED", "fixture mission cancellation")
                yield
        self.use(WaitingNative)
        errors = []
        worker = self.generate_in_thread(errors)
        self.assertTrue(entered.wait(2))
        original_event = self.runtime.cancels[self.mission]
        initial_state = self.runtime.get(self.mission)["status"]
        self.assertEqual(self.runtime.active_generations[self.mission], 1)
        with self.assertRaises(Fault) as caught:
            self.runtime.start(self.mission)
        self.assertEqual(caught.exception.code, "BUSY")
        self.assertIs(self.runtime.cancels[self.mission], original_event)
        self.assertFalse(original_event.is_set())
        self.assertEqual(self.runtime.get(self.mission)["status"], initial_state)
        self.assertNotIn(self.mission, self.runtime.workers)
        result = self.runtime.stop(self.mission)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["stopped"])
        self.assertTrue(original_event.is_set())
        self.assertEqual(errors, ["CANCELLED"])
        self.assertEqual(len(self.runtime.engine._model.calls), 1)
        self.assertEqual(self.runtime.active_generations, {})
        self.assertEqual(self.runtime.engine.sessions(self.mission), [])
        decisions = self.runtime.agent_contexts.snapshot(1, self.mission)["decisions"]
        self.assertEqual([row["state"] for row in decisions], ["cancelled"])

    def test_global_cancel_on_direct_generation_of_partial_mission_is_cancelled_not_failed(self):
        entered = threading.Event()
        class PartialThenWaitingNative(FakeNative):
            def stream_chat(self, messages, **options):
                self.calls.append((messages, options))
                if len(self.calls) == 1:
                    yield {"content": "brouillon", "tokens": [19], "stop": False}
                    yield {"content": "", "tokens": [], "stop": True, "stop_type": "limit"}
                    return
                entered.set()
                while not options["cancel"].is_set():
                    time.sleep(.001)
                raise NativeGlmError("CANCELLED", "fixture global engine cancellation")
        self.use(PartialThenWaitingNative)
        first = self.runtime.dispatch("generate", {"id": self.mission, "max_new_tokens": 2})
        self.assertEqual(first["decision_state"], "partial")
        self.runtime.state(self.mission, "partial")
        mission_event = self.runtime.cancels[self.mission]
        errors = []
        worker = self.generate_in_thread(errors)
        self.assertTrue(entered.wait(2))
        self.runtime.dispatch("shutdown", {})
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(mission_event.is_set())  # This route is global cancellation only.
        self.assertTrue(self.runtime.engine._shutdown_cancel.is_set())
        self.assertEqual(self.runtime.get(self.mission)["status"], "partial")
        self.assertEqual(errors, ["CANCELLED"])
        context = self.runtime.agent_contexts.snapshot(1, self.mission)
        self.assertEqual([row["state"] for row in context["decisions"]], ["partial", "cancelled"])
        last = context["decisions"][-1]
        self.assertEqual(last["error_code"], "CANCELLED")
        previous = self.runtime.agent_contexts.decision(1, self.mission, first["decision_id"])
        self.assertEqual(previous["tokens"], [19])
        self.assertEqual(previous["output_text"], "brouillon")
        self.assertEqual(len(self.runtime.engine._model.calls), 2)
        self.assertEqual(self.runtime.active_generations, {})
        self.assertEqual(self.runtime.engine.sessions(self.mission), [])

    def test_resource_failure_stays_failed_and_preserves_partial_output_without_retry(self):
        class ResourceFailureNative(FakeNative):
            def stream_chat(self, messages, **options):
                self.calls.append((messages, options))
                yield {"content": "partie enregistrée", "tokens": [4], "stop": False}
                raise NativeGlmError("SWAP_LIMIT", "fixture unchanged guard stop")
        self.use(ResourceFailureNative)
        with self.assertRaises(Fault) as caught:
            self.runtime.dispatch("generate", {"id": self.mission, "max_new_tokens": 2})
        self.assertEqual(caught.exception.code, "SWAP_LIMIT")
        self.assertFalse(self.runtime.cancels[self.mission].is_set())
        self.assertFalse(self.runtime.engine._shutdown_cancel.is_set())
        decisions = self.runtime.agent_contexts.snapshot(1, self.mission)["decisions"]
        self.assertEqual([row["state"] for row in decisions], ["failed"])
        self.assertEqual(decisions[0]["error_code"], "SWAP_LIMIT")
        saved = self.runtime.agent_contexts.decision(1, self.mission, decisions[0]["id"])
        self.assertEqual(saved["tokens"], [4])
        self.assertEqual(saved["output_text"], "partie enregistrée")
        self.assertEqual(len(self.runtime.engine._model.calls), 1)
        self.assertEqual(self.runtime.active_generations, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
