"""RONIN control-flow tests; fake events are not inference evidence."""
from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "air_llm"))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "runtime"))

from adapter import Engine, EngineError, NativeGlmError
from ronin_runtime import Fault, Runtime


class FakeNative:
    instances = []

    def __init__(self, path, **options):
        self.path, self.options = path, options
        self.loaded = False
        self.calls = []
        self.close_count = 0
        self.instances.append(self)

    def status(self):
        return {"loaded": self.loaded, "ready": self.loaded,
                "checkpoint": {"errors": []},
                "scheduler": {"physical_limit": 1, "max_pending": 1024}}

    def load(self, cancel=None):
        self.loaded = True
        return self.status()

    def close(self):
        self.loaded = False
        self.close_count += 1

    def stream_chat(self, messages, **options):
        self.calls.append((messages, options))
        yield {"content": "é", "tokens": [19], "stop": False}
        yield {"content": "", "tokens": [], "stop": True,
               "timings": {"predicted_n": 1, "predicted_per_second": 0.1}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        FakeNative.instances.clear()
        self.engine = Engine(config={"runtime_binary": "/test/llama-server"}, model_factory=FakeNative)

    def test_dense_prefetch_environment_and_explicit_json_priority(self):
        with mock.patch.dict(os.environ, {"RONIN_GLM_DENSE_PREFETCH": "1"}):
            self.assertTrue(self.engine._options()["dense_prefetch"])
            explicit = Engine(config={"dense_prefetch": False}, model_factory=FakeNative)
            self.assertFalse(explicit._options()["dense_prefetch"])

    def test_600_logical_sessions_share_one_native_object(self):
        sessions = [self.engine.create_session("mission-" + str(i)) for i in range(600)]
        self.engine.load("/test/model")
        self.assertEqual(len({s["id"] for s in sessions}), 600)
        self.assertEqual(len(FakeNative.instances), 1)
        self.assertEqual(self.engine.status()["capabilities"]["max_concurrent_inferences"], 1)
        self.assertEqual(self.engine.status()["native"]["scheduler"]["max_pending"], 1024)

    def test_native_events_and_ids_are_preserved(self):
        self.engine.load("/test/model")
        events = list(self.engine.stream([{"role": "user", "content": "Question"}], max_new_tokens=2))
        self.assertEqual(events[0], {"content": "é", "tokens": [19], "stop": False})
        self.assertEqual(events[-1]["timings"]["predicted_n"], 1)
        self.assertEqual(self.engine.counters()["generated_tokens"], 1)
        self.assertEqual(self.engine.counters()["requests_in_flight"], 0)

    def test_validation_precedes_generation(self):
        for budget in (False, 0, 1.5, 163841):
            with self.assertRaises(EngineError) as result:
                self.engine.stream([{"role": "user", "content": "x"}], max_new_tokens=budget)
            self.assertEqual(result.exception.code, "INVALID_TOKEN_BUDGET")
        with self.assertRaises(EngineError) as result:
            self.engine.stream([{"role": "user", "content": "x"}], reasoning_effort="unlimited")
        self.assertEqual(result.exception.code, "INVALID_REASONING_EFFORT")
        self.assertEqual(self.engine.counters()["generation_attempts"], 0)

    def test_default_preserves_template_and_explicit_efforts_are_forwarded(self):
        self.engine.load('/test/model')
        for effort in (None, 'low', 'high', 'max'):
            list(self.engine.stream([{'role':'user','content':'x'}],reasoning_effort=effort))
            options=self.engine._model.calls[-1][1]
            if effort is None: self.assertNotIn('chat_template_kwargs',options)
            else: self.assertEqual(options['chat_template_kwargs'],{'reasoning_effort':effort})

    def test_unknown_generation_argument_is_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.stream([{'role':'user','content':'x'}],unlimited_thinking=True)

    def test_cancellation_before_native_call(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(EngineError) as result:
            self.engine.stream([{"role": "user", "content": "x"}], cancel=cancel)
        self.assertEqual(result.exception.code, "CANCELLED")
        self.assertEqual(FakeNative.instances, [])

    def test_session_cannot_be_closed_by_another_mission(self):
        session = self.engine.create_session("A")
        with self.assertRaises(EngineError):
            self.engine.close_session(session["id"], "B")
        self.assertEqual(self.engine.sessions("B"), [])
        self.assertEqual(len(self.engine.sessions("A")), 1)

    def test_model_change_refused_during_active_stream(self):
        self.engine.load("/test/first")
        stream = self.engine.stream([{"role": "user", "content": "x"}])
        next(stream)
        with self.assertRaises(EngineError) as result:
            self.engine.load("/test/second")
        self.assertEqual(result.exception.code, "ENGINE_BUSY")
        self.assertEqual(FakeNative.instances[0].close_count, 0)
        stream.close()
        self.engine.load("/test/second")
        self.assertEqual(FakeNative.instances[0].close_count, 1)

    def test_model_change_before_first_next_refuses_old_generator(self):
        self.engine.load("/test/first")
        stream = self.engine.stream([{"role": "user", "content": "x"}])
        self.engine.load("/test/second")
        with self.assertRaises(EngineError) as result:
            next(stream)
        self.assertEqual(result.exception.code, "ENGINE_MODEL_CHANGED")
        self.assertEqual(self.engine.counters()["generation_attempts"], 0)

    def test_native_admission_error_keeps_code_and_details(self):
        self.engine.load("/test/model")
        with mock.patch.object(self.engine._model, "load", side_effect=NativeGlmError("MEMORY_ADMISSION", "refused", {"available": 1})):
            with self.assertRaises(EngineError) as result:
                self.engine.load("/test/model")
        self.assertEqual(result.exception.code, "MEMORY_ADMISSION")
        self.assertEqual(result.exception.details, {"available": 1})


    def test_closing_engine_cancels_manual_load_without_mission(self):
        entered = threading.Event()
        outcome = []
        class LoadingNative(FakeNative):
            def load(self, cancel=None):
                entered.set()
                while not cancel.is_set():
                    time.sleep(.01)
                raise NativeGlmError("CANCELLED", "cancelled by shutdown")
        engine = Engine(config={"runtime_binary": "/unused"}, model_factory=LoadingNative)
        def load():
            try:
                engine.load("/model")
            except EngineError as error:
                outcome.append(error.code)
        loader = threading.Thread(target=load)
        loader.start()
        self.assertTrue(entered.wait(timeout=1))
        engine.close()
        loader.join(timeout=1)
        self.assertFalse(loader.is_alive())
        self.assertEqual(outcome, ["CANCELLED"])
        with self.assertRaises(EngineError) as caught:
            engine.load("/model")
        self.assertEqual(caught.exception.code, "ENGINE_CLOSED")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.runtime = Runtime(self.temp.name)
        self.runtime.engine = Engine(config={"runtime_binary": "/test/llama-server"}, model_factory=FakeNative)
        self.runtime.dispatch("settings", {"model_path": "/test/model"})
        self.mission = self.runtime.dispatch("create", {"prompt": "Question"})

    def tearDown(self):
        self.runtime.dispatch("shutdown", {})
        self.runtime.store.db.close()
        self.environment.stop()
        self.temp.cleanup()

    def test_config_paths_follow_config_directory_and_default_state_follows_data_home(self):
        root=Path(self.temp.name); resources=root/'Resources'; resources.mkdir()
        file=resources/'runtime-config.json'
        file.write_text(json.dumps({'runtime_binary':'native-runtime/bin/llama-server',
            'runtime_manifest_file':'native-runtime/manifest.json','verification_file':'model-verification.json',
            'model_path':'~/Models/external-model','expected_runtime_sha256':'0'*64}))
        home=root/'isolated-data'
        with mock.patch.dict(os.environ,{'RONIN_GLM_CONFIG':str(file)},clear=True):
            runtime=Runtime(home)
            try:
                config=runtime.engine.config
                for key,value in (('runtime_binary','native-runtime/bin/llama-server'),
                    ('runtime_manifest_file','native-runtime/manifest.json'),('verification_file','model-verification.json')):
                    self.assertEqual(config[key],str(resources/value))
                self.assertEqual(config['state_dir'],str(home/'engine-state'))
                self.assertEqual(config['expected_runtime_sha256'],'0'*64)
                self.assertEqual(runtime.setting('model_path'),'~/Models/external-model')
            finally: runtime.engine.close(); runtime.store.db.close()

    def test_explicit_relative_state_is_relative_to_config(self):
        root=Path(self.temp.name); file=root/'runtime-config.json'
        file.write_text(json.dumps({'state_dir':'relative-state'}))
        with mock.patch.dict(os.environ,{'RONIN_GLM_CONFIG':str(file)},clear=True):
            runtime=Runtime(root/'other-data')
            try: self.assertEqual(runtime.engine.config['state_dir'],str(root/'relative-state'))
            finally: runtime.engine.close(); runtime.store.db.close()

    def test_generate_passes_through_ronin_and_records_unverified_evidence(self):
        result = self.runtime.dispatch("generate", {"id": self.mission["id"], "max_new_tokens": 2})
        self.assertEqual(result["text"], "é")
        self.assertEqual(result["tokens"], [19])
        self.assertEqual(result["verification"], "model_output_unverified")
        self.assertFalse(result["tools_executed_from_output"])
        evidence = self.runtime.store.query("SELECT * FROM evidence WHERE id=?", (result["evidence_id"],))[0]
        self.assertEqual(evidence["status"], "model_output_unverified")
        self.assertEqual(self.runtime.engine.sessions(self.mission["id"]), [])
        self.assertEqual(len(self.runtime.dispatch("agents", {})), 600)

    def test_mission_run_reaches_native_generation(self):
        self.runtime.run(self.mission["id"])
        self.assertEqual(self.runtime.get(self.mission["id"])["status"], "partial")
        events = self.runtime.detail(self.mission["id"])["events"]
        self.assertIn("model_partial", [e["type"] for e in events])
        self.assertNotIn("model_completed", [e["type"] for e in events])
        self.assertNotIn("model_blocked", [e["type"] for e in events])

    def test_stopped_mission_does_not_generate(self):
        self.runtime.stop(self.mission["id"])
        with self.assertRaises(Fault) as result:
            self.runtime.dispatch("generate", {"id": self.mission["id"]})
        self.assertEqual(result.exception.code, "CANCELLED")
        self.assertEqual(self.runtime.engine.counters()["load_attempts"], 0)

    def test_runtime_explicit_effort_reaches_template_and_is_recorded(self):
        result=self.runtime.dispatch('generate',{'id':self.mission['id'],'max_new_tokens':2,'reasoning_effort':'low'})
        self.assertEqual(result['reasoning_effort'],'low')
        self.assertEqual(self.runtime.engine._model.calls[-1][1]['chat_template_kwargs'],{'reasoning_effort':'low'})

    def test_unknown_runtime_flag_and_invalid_effort_refuse_before_load(self):
        for extra,code in (({'unlimited_thinking':True},'INVALID_ARGUMENT'),({'reasoning_effort':'unlimited'},'INVALID_REASONING_EFFORT')):
            with self.subTest(extra=extra),self.assertRaises(Fault) as result:
                self.runtime.dispatch('generate',{'id':self.mission['id'],**extra})
            self.assertEqual(result.exception.code,code)
        self.assertEqual(self.runtime.engine.counters()['load_attempts'],0)

    def test_cli_rejects_invalid_or_ignored_effort_before_runtime_creation(self):
        import ronin_glm
        for argv in (['generate','--reasoning-effort','unlimited'],['status','--reasoning-effort','low']):
            with mock.patch.object(sys,'argv',['ronin_glm.py',*argv]),mock.patch.object(ronin_glm,'Runtime') as factory:
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): ronin_glm.main()
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
