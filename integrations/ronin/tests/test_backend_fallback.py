"""Preactivation policy only; fake backends provide no GLM inference evidence."""
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "air_llm"))
sys.path.insert(0, str(ROOT / "engine"))
from adapter import Engine, EngineError, NativeGlmError


class Backend:
    instances = []

    def __init__(self, path, **options):
        self.options = options
        self.loaded = False
        self.requests = 0
        self.instances.append(self)

    def load(self, cancel=None):
        if self.options.get("gpu_layers") and not self.options.get("allow_unqualified_gpu"):
            raise NativeGlmError("GPU_FIDELITY_UNQUALIFIED", "rejected")
        self.loaded = True

    def status(self):
        return {"loaded": self.loaded, "ready": self.loaded, "checkpoint": {"errors": []}}

    def close(self):
        self.loaded = False

    def stream_chat(self, messages, **options):
        self.requests += 1
        yield {"content": "partial", "tokens": [1], "stop": False}
        raise NativeGlmError("SWAP_LIMIT", "resource stop")


class BackendFallbackTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        Backend.instances.clear()

    def tearDown(self):
        self.environment.stop()

    def engine(self, **options):
        return Engine(config={"runtime_binary": "/unused", **options}, model_factory=Backend)

    def test_unqualified_candidate_is_never_constructed_and_cpu_preserves_guards(self):
        limits = {"max_rss_bytes": 8_000_000_000, "max_swap_growth_bytes": 32 * 1024**2}
        engine = self.engine(gpu_layers=6, cpu_moe=True, moe_cache_mib=384, limits=limits)
        with mock.patch("adapter.os.cpu_count", return_value=10):
            status = engine.load("/model")
        self.assertEqual(len(Backend.instances), 1)
        backend = Backend.instances[0]
        self.assertEqual(backend.options["gpu_layers"], 0)
        self.assertEqual(backend.options["threads"], 10)
        self.assertFalse(backend.options["cpu_moe"])
        self.assertEqual(backend.options["moe_cache_mib"], 0)
        self.assertEqual(backend.options["limits"].max_rss_bytes, limits["max_rss_bytes"])
        self.assertEqual(backend.options["limits"].max_swap_growth_bytes, limits["max_swap_growth_bytes"])
        self.assertTrue(status["engine"]["backend_selection"]["fallback_before_activation"])
        self.assertEqual(status["engine"]["backend_selection"]["rejection"], "GPU_FIDELITY_UNQUALIFIED")

    def test_explicit_research_opt_in_is_preserved(self):
        engine = self.engine(gpu_layers=6, allow_unqualified_gpu=True)
        status = engine.load("/model")
        self.assertEqual(engine._model.options["gpu_layers"], 6)
        self.assertFalse(status["engine"]["backend_selection"]["fallback_before_activation"])

    def test_rejection_can_remain_visible_to_research_caller(self):
        engine = self.engine(gpu_layers=6, fallback_to_cpu_on_rejection=False)
        with self.assertRaises(EngineError) as result:
            engine.load("/model")
        self.assertEqual(result.exception.code, "GPU_FIDELITY_UNQUALIFIED")
        self.assertEqual(len(Backend.instances), 1)

    def test_environment_gpu_request_also_falls_back(self):
        with mock.patch.dict(os.environ, {"RONIN_GLM_GPU_LAYERS": "6"}):
            engine = self.engine()
            engine.load("/model")
        self.assertEqual(engine._model.options["gpu_layers"], 0)

    def test_no_replay_after_partial_output_or_resource_error(self):
        engine = self.engine()
        engine.load("/model")
        stream = engine.stream([{"role": "user", "content": "x"}])
        self.assertEqual(next(stream)["tokens"], [1])
        with self.assertRaises(EngineError) as result:
            next(stream)
        self.assertEqual(result.exception.code, "SWAP_LIMIT")
        self.assertEqual(len(Backend.instances), 1)
        self.assertEqual(engine._model.requests, 1)

    def test_cancelled_load_does_not_start_fallback(self):
        engine = self.engine(gpu_layers=6)
        cancel = threading.Event()
        cancel.set()
        with mock.patch.object(Backend, "load", side_effect=NativeGlmError("CANCELLED", "stopped")):
            with self.assertRaises(EngineError) as result:
                engine.load("/model", cancel=cancel)
        self.assertEqual(result.exception.code, "CANCELLED")
        self.assertEqual(len(Backend.instances), 1)

    def test_invalid_boolean_is_not_converted_to_an_opt_in_or_fallback(self):
        for value in ("true", 1, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.engine(allow_unqualified_gpu=value).load("/model")
        self.assertEqual(Backend.instances, [])


if __name__ == "__main__":
    unittest.main()
