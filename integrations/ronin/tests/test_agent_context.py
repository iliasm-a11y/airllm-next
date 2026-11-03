"""Portable control/state tests. Fake outputs do not qualify GLM inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "air_llm"))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import Engine, NativeGlmError
from agent_context import AgentContextError
from ronin_runtime import Fault, Runtime
from test_ronin_glm import FakeNative


class AgentContextTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Runtime(self.temp.name)
        self.runtime.engine = Engine(config={"runtime_binary": "/test/llama-server"}, model_factory=FakeNative)
        self.runtime.dispatch("settings", {"model_path": "/test/model"})
        self.first = self.runtime.create({"prompt": "mission alpha"})["id"]
        self.second = self.runtime.create({"prompt": "mission beta"})["id"]

    def tearDown(self):
        self.runtime.dispatch("shutdown", {})
        self.runtime.store.db.close()
        self.temp.cleanup()
        self.environment.stop()

    def generate(self, mid=None, agent_id=1, **extra):
        return self.runtime.dispatch("generate", {"id": mid or self.first, "agent_id": agent_id,
                                                   "max_new_tokens": 2, **extra})

    def decision(self, result, mid=None):
        return self.runtime.dispatch("agent_decision", {"id": mid or self.first,
                    "agent_id": result["agent_id"], "decision_id": result["decision_id"]})

    def test_600_persistent_identities_have_real_instruction_fields(self):
        agents = self.runtime.dispatch("agent_identities", {})
        self.assertEqual(len(agents), 600)
        self.assertEqual(len({agent["name"] for agent in agents}), 600)
        self.assertTrue(all(agent["instructions"] and agent["specialization"] for agent in agents))
        # Reading an identity never invokes the model.
        self.assertIsNone(self.runtime.engine._model)
        self.runtime.agent_contexts.memory(600, self.first, "mémoire durable")
        self.runtime.engine.close()
        self.runtime.store.db.close()
        self.runtime = Runtime(self.temp.name)
        self.assertEqual(self.runtime.dispatch("agent_identities", {}), agents)
        self.assertEqual(self.runtime.agent_contexts.snapshot(600, self.first)["working_memory"], "mémoire durable")

    def test_agent_and_mission_contexts_are_isolated_and_one_model_is_reused(self):
        self.runtime.agent_contexts.memory(1, self.first, "source alpha uniquement")
        first = self.generate(prompt="demande alpha")
        model = self.runtime.engine._model
        messages = model.calls[-1][0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("RONIN-001", messages[0]["content"])
        self.assertIn("source alpha uniquement", messages[0]["content"])
        self.generate(agent_id=600, prompt="demande gamma")
        self.assertIs(self.runtime.engine._model, model)
        self.assertNotIn("source alpha uniquement", json.dumps(model.calls[-1][0]))
        self.assertNotIn("demande alpha", json.dumps(model.calls[-1][0]))
        self.generate(mid=self.second, prompt="demande beta")
        self.assertNotIn("source alpha uniquement", json.dumps(model.calls[-1][0]))
        self.assertNotIn("demande alpha", json.dumps(model.calls[-1][0]))
        self.generate(prompt="suite alpha")
        replay = model.calls[-1][0]
        self.assertEqual(replay[1:3], [{"role": "user", "content": "demande alpha"},
                                      {"role": "assistant", "content": "é"}])
        self.assertIn(first["decision_id"], replay[0]["content"])
        self.assertEqual(model.options.get("gpu_layers", 0), 0)
        self.assertNotIn("cache_prompt", model.calls[-1][1])

    def test_decisions_preserve_native_ids_context_provenance_and_partial_state(self):
        result = self.generate()
        record = self.decision(result)
        self.assertEqual(record["state"], "partial")  # stop=True alone never proves EOS.
        self.assertEqual(record["tokens"], [19])
        self.assertEqual(record["output_text"], "é")
        self.assertEqual(record["context_sha256"], result["context_sha256"])
        self.assertTrue(record["provenance"]["native_output_observed"])
        self.assertEqual(record["provenance"]["verification"], "model_output_unverified")
        self.assertFalse(result["tools_executed_from_output"])
        self.assertEqual(self.runtime.engine.sessions(self.first), [])
        with self.assertRaises(Fault) as caught:
            self.runtime.dispatch("agent_decision", {"id": self.second, "agent_id": 1,
                                                      "decision_id": result["decision_id"]})
        self.assertEqual(caught.exception.code, "DECISION_NOT_FOUND")

    def test_eos_must_be_explicit(self):
        class EosNative(FakeNative):
            def stream_chat(self, messages, **options):
                self.calls.append((messages, options))
                yield {"content": "réponse", "tokens": [8], "stop": False}
                yield {"content": "", "tokens": [], "stop": True, "stop_type": "eos"}
        self.runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=EosNative)
        result = self.generate()
        self.assertEqual(result["decision_state"], "completed")
        self.assertEqual(self.decision(result)["terminal"]["stop_type"], "eos")

    def test_cancelled_partial_output_is_archived_and_excluded_from_replay(self):
        cancel = self.runtime.cancels.setdefault(self.first, threading.Event())
        class CancelNative(FakeNative):
            def stream_chat(self, messages, **options):
                self.calls.append((messages, options))
                cancel.set()
                yield {"content": "brouillon annulé", "tokens": [7], "stop": False}
        self.runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=CancelNative)
        with self.assertRaises(Fault) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "CANCELLED")
        context = self.runtime.agent_contexts.snapshot(1, self.first)
        decision_id = context["decisions"][0]["id"]
        record = self.runtime.agent_contexts.decision(1, self.first, decision_id)
        self.assertEqual(record["state"], "cancelled")
        self.assertEqual(record["tokens"], [7])
        self.assertEqual(record["output_text"], "brouillon annulé")
        cancel.clear()
        self.runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=FakeNative)
        self.generate(prompt="nouvelle tentative")
        messages = self.runtime.engine._model.calls[-1][0]
        self.assertEqual(len(messages), 2)
        self.assertIn(decision_id, messages[0]["content"])
        self.assertNotIn("brouillon annulé", json.dumps(messages))
        self.assertEqual(self.runtime.active_generations, {})

    def test_failed_output_and_crash_interruption_remain_distinct(self):
        class FailedNative(FakeNative):
            def stream_chat(self, messages, **options):
                yield {"content": "partiel", "tokens": [4], "stop": False}
                raise NativeGlmError("SWAP_LIMIT", "fixture guard stop")
        self.runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=FailedNative)
        with self.assertRaises(Fault) as caught:
            self.generate()
        self.assertEqual(caught.exception.code, "SWAP_LIMIT")
        record = self.runtime.store.query("SELECT state,error_code,tokens FROM glm_agent_decisions")[0]
        self.assertEqual(record, {"state": "failed", "error_code": "SWAP_LIMIT", "tokens": "[4]"})
        begun = self.runtime.agent_contexts.begin(2, self.first, "crash fixture", {"fixture": True})
        self.runtime.agent_contexts.append(begun["id"], "premier token", [9])
        self.runtime.engine.close()
        self.runtime.store.db.close()
        self.runtime = Runtime(self.temp.name)
        restored = self.runtime.agent_contexts.decision(2, self.first, begun["id"])
        self.assertEqual(restored["state"], "interrupted")
        self.assertEqual(restored["tokens"], [9])
        self.assertEqual(restored["error_code"], "PROCESS_INTERRUPTED")

    def test_pair_is_busy_while_other_agents_can_prepare(self):
        contexts = self.runtime.agent_contexts
        first = contexts.begin(1, self.first, "a", {"fixture": True})
        with self.assertRaises(AgentContextError) as caught:
            contexts.begin(1, self.first, "b", {"fixture": True})
        self.assertEqual(caught.exception.code, "AGENT_BUSY")
        with self.assertRaises(AgentContextError):
            contexts.memory(1, self.first, "ne pas modifier en vol")
        other = contexts.begin(2, self.first, "c", {"fixture": True})
        contexts.finish(first["id"], "failed", provenance={"fixture": True})
        contexts.finish(other["id"], "failed", provenance={"fixture": True})

    def test_limits_and_invalid_agent_refuse_before_model_load_without_truncation(self):
        for agent in (True, 0, 601, "1"):
            with self.assertRaises(Fault) as caught:
                self.generate(agent_id=agent)
            self.assertEqual(caught.exception.code, "INVALID_AGENT")
        with self.assertRaises(AgentContextError) as caught:
            self.runtime.agent_contexts.memory(1, self.first, "é" * 4097)
        self.assertEqual(caught.exception.code, "AGENT_MEMORY_LIMIT")
        with self.assertRaises(Fault) as caught:
            self.generate(prompt="x" * 65536)
        self.assertEqual(caught.exception.code, "AGENT_CONTEXT_LIMIT")
        self.assertEqual(self.runtime.engine.counters()["load_attempts"], 0)
        self.assertEqual(self.runtime.store.query("SELECT id FROM glm_agent_decisions"), [])

    def test_history_limit_is_explicit_and_retains_prior_record(self):
        self.runtime.agent_contexts.MAX_DECISIONS_PER_CONTEXT = 1
        original = self.generate()
        with self.assertRaises(Fault) as caught:
            self.generate(prompt="suite refusée")
        self.assertEqual(caught.exception.code, "AGENT_HISTORY_LIMIT")
        self.assertEqual(self.decision(original)["tokens"], [19])

    def test_output_limit_preserves_already_committed_tokens(self):
        contexts = self.runtime.agent_contexts
        contexts.MAX_OUTPUT_BYTES = 3
        record = contexts.begin(1, self.first, "a", {"fixture": True})
        contexts.append(record["id"], "ab", [1])
        with self.assertRaises(AgentContextError) as caught:
            contexts.append(record["id"], "cd", [2])
        self.assertEqual(caught.exception.code, "AGENT_OUTPUT_LIMIT")
        contexts.finish(record["id"], "failed", provenance={"fixture": True})
        saved = contexts.decision(1, self.first, record["id"])
        self.assertEqual(saved["tokens"], [1])
        self.assertEqual(saved["output_text"], "ab")

    def test_secret_split_between_events_is_redacted_before_replay_and_raw_buffer_released(self):
        contexts = self.runtime.agent_contexts
        decision = contexts.begin(1, self.first, "a", {"fixture": True})
        contexts.append(decision["id"], "pass", [1])
        contexts.append(decision["id"], "word=abc123", [2])
        contexts.append(decision["id"], "tail", [3])
        contexts.finish(decision["id"], "partial", provenance={"fixture": True},
                        terminal={"stop": True, "content": "password=abc123tail"})
        saved = contexts.decision(1, self.first, decision["id"])
        self.assertEqual(saved["tokens"], [1, 2, 3])
        self.assertNotIn("abc123", saved["output_text"])
        self.assertNotIn("tail", saved["output_text"])
        self.assertNotIn("abc123", saved["terminal"]["content"])
        self.assertNotIn(decision["id"], contexts._raw_outputs)
        followup = contexts.begin(1, self.first, "b", {"fixture": True})
        self.assertNotIn("abc123", json.dumps(followup["messages"]))
        contexts.finish(followup["id"], "failed", provenance={"fixture": True})

    def test_stop_waits_for_direct_generation_cleanup(self):
        entered = threading.Event()
        class WaitingNative(FakeNative):
            def stream_chat(self, messages, **options):
                entered.set()
                while not options["cancel"].is_set():
                    time.sleep(.001)
                raise NativeGlmError("CANCELLED", "fixture cancellation")
                yield  # Mark as generator without emitting an invented token.
        self.runtime.engine = Engine(config={"runtime_binary": "/unused"}, model_factory=WaitingNative)
        errors = []
        def generate():
            try:
                self.generate()
            except Fault as error:
                errors.append(error.code)
        worker = threading.Thread(target=generate)
        worker.start()
        self.assertTrue(entered.wait(2))
        result = self.runtime.stop(self.first)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result["stopped"])
        self.assertEqual(errors, ["CANCELLED"])
        self.assertEqual(self.runtime.active_generations, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
