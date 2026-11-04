"""Configuration contracts only; no native process or checkpoint is opened."""
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "air_llm"), str(ROOT / "integrations/ronin/engine")]
from airllm.nextgen.glm_native import NativeGlmModel, NativeGlmError, NativeLimits, MiB, GiB
from adapter import Engine


class DensePrefillOptionsTests(unittest.TestCase):
    def model(self, **options):
        with mock.patch("airllm.nextgen.glm_native.os.cpu_count", return_value=10):
            return NativeGlmModel("/unused-model", runtime_binary="/unused-runtime", **options)

    def prefix_model(self, **options):
        base = {"threads": 10, "shared_prefix_tokens": list(range(208)),
                "shared_prefix_scratch_mib": 16,
                "limits": NativeLimits(max_rss_bytes=17 * GiB // 2, max_swap_growth_bytes=32 * MiB)}
        return self.model(**{**base, **options})

    def test_defaults_do_not_enable_experiments_or_change_prefill_thread_policy(self):
        model = self.model()
        command = model._build_command(1234)
        self.assertEqual(command[command.index("--ubatch-size") + 1], "8")
        self.assertNotIn("--threads-batch", command)
        self.assertNotIn("--cache-ram", command)
        with mock.patch.dict(os.environ, {"AIRLLM_DENSE_LOOKAHEAD": "1", "AIRLLM_SHARED_PREFIX_MANIFEST": "/foreign"}):
            environment = model._runtime_environment()
        self.assertNotIn("AIRLLM_DENSE_LOOKAHEAD", environment)
        self.assertNotIn("AIRLLM_SHARED_PREFIX_MANIFEST", environment)

    def test_explicit_options_preserve_batch_and_select_only_requested_environment(self):
        model = self.model(batch_size=32, ubatch_size=8, threads_batch=10,
                           dense_lookahead=True, dense_lookahead_mib=16, dense_lookahead_max_cols=8)
        command = model._build_command(1234)
        self.assertEqual(command[command.index("--batch-size") + 1], "32")
        self.assertEqual(command[command.index("--ubatch-size") + 1], "8")
        self.assertEqual(command[command.index("--threads-batch") + 1], "10")
        environment = model._runtime_environment()
        self.assertEqual(environment["AIRLLM_DENSE_LOOKAHEAD_MAX_COLS"], "8")
        self.assertEqual(environment["AIRLLM_DENSE_LOOKAHEAD_MIB"], "16")
        self.assertNotIn("AIRLLM_DENSE_PREFETCH", environment)

    def test_invalid_options_fail_before_any_process_can_start(self):
        invalid = [{"ubatch_size": 9}, {"ubatch_size": True}, {"threads_batch": 0},
                   {"dense_lookahead": "1"}, {"dense_lookahead_mib": 1024},
                   {"dense_lookahead_max_cols": 8}, {"dense_prefetch_max_cols": 8},
                   {"dense_lock_names": []}, {"dense_lock_names": ["output.weight"]},
                   {"dense_lock_names": ["blk.3.ssm_beta.weight"] * 2}, {"dense_lock_mib": 512}]
        with mock.patch("airllm.nextgen.glm_native.subprocess.Popen", side_effect=AssertionError("process forbidden")):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.model(**options)

    def test_multicolumn_cannot_silently_run_against_an_old_manifest(self):
        model = self.model(dense_lookahead=True, dense_lookahead_max_cols=8)
        with self.assertRaises(NativeGlmError):
            model._check_requested_features()
        model._runtime_artifacts_verified = True
        model._runtime_manifest_info = {"patch_names": ["llama-dense-lookahead.patch"], "airllm_runtime_features": []}
        with self.assertRaises(NativeGlmError) as result:
            model._check_requested_features()
        self.assertEqual(result.exception.code, "RUNTIME_FEATURE_UNAVAILABLE")
        model._runtime_manifest_info["airllm_runtime_features"] = ["dense_lookahead_multicolumn_v1"]
        model._check_requested_features()

    def test_shared_prefix_requires_complete_bounded_cpu_profile(self):
        for options in ({"threads_batch": 4}, {"batch_size": 16}, {"shared_prefix_tokens": [True] * 208},
                        {"shared_prefix_tokens": list(range(209))}, {"shared_prefix_scratch_mib": None},
                        {"limits": NativeLimits()}, {"shared_prefix_render_options": {"unknown": True}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prefix_model(**options)
        model = self.prefix_model()
        command = model._build_command(1234)
        self.assertEqual(command[command.index("--ctx-checkpoints") + 1], "0")
        self.assertEqual(command[command.index("--cache-ram") + 1], "0")
        model._checkpoint = {"integrity_verified": False}
        self.assertFalse(model.status()["configuration"]["shared_prefix"]["numerically_qualified"])

    def test_template_options_cannot_reuse_a_prefix_from_another_rendering(self):
        model = self.prefix_model()
        model.scheduler.slot = lambda cancel: nullcontext()
        with mock.patch.object(model, "load"), mock.patch.object(model, "_admit"), \
             mock.patch.object(model, "_check_identity"), \
             mock.patch.object(model, "_http_json", return_value={"prompt": "rendered"}), \
             mock.patch.object(model, "_payload", return_value={}) as payload, \
             mock.patch.object(model, "_events", return_value=iter([])):
            list(model.stream_chat([{"role": "user", "content": "x"}], chat_template_kwargs={"reasoning_effort": "low"}, cache_prompt=True))
            self.assertFalse(payload.call_args.args[2])
        with mock.patch.object(model, "load"), mock.patch.object(model, "_admit"), \
             mock.patch.object(model, "_check_identity"), \
             mock.patch.object(model, "_http_json", return_value={"prompt": "rendered"}), \
             mock.patch.object(model, "_payload", return_value={}) as payload, \
             mock.patch.object(model, "_events", return_value=iter([])):
            list(model.stream_chat([{"role": "user", "content": "x"}]))
            self.assertTrue(payload.call_args.args[2])

    def test_adapter_configuration_has_priority_over_environment(self):
        with mock.patch.dict(os.environ, {"RONIN_GLM_DENSE_LOOKAHEAD": "1", "RONIN_GLM_UBATCH": "8",
                                          "RONIN_GLM_THREADS_BATCH": "10"}, clear=True):
            options = Engine(config={"dense_lookahead": False})._options()
        self.assertFalse(options["dense_lookahead"])
        self.assertEqual(options["ubatch_size"], 8)
        self.assertEqual(options["threads_batch"], 10)

    def test_abandoned_outer_stream_closes_the_native_generator(self):
        model = self.model()
        model.scheduler.slot = lambda cancel: nullcontext()
        closed = []

        def native_events(*args):
            try:
                yield {"content": "first", "tokens": [1]}
                yield {"content": "second", "tokens": [2]}
            finally:
                closed.append(True)

        with mock.patch.object(model, "load"), mock.patch.object(model, "_admit"), \
             mock.patch.object(model, "_check_identity"), \
             mock.patch.object(model, "_http_json", return_value={"prompt": "rendered"}), \
             mock.patch.object(model, "_payload", return_value={}), \
             mock.patch.object(model, "_events", side_effect=native_events):
            stream = model.stream_chat([{"role": "user", "content": "x"}])
            next(stream)
            stream.close()
        self.assertEqual(closed, [True])

    def test_dense_retention_requires_built_patch_and_explicit_names(self):
        model = self.model(dense_lock_names=["blk.3.ssm_beta.weight"], dense_lock_mib=128)
        self.assertEqual(model._runtime_environment()["AIRLLM_DENSE_LOCK_NAMES"], "blk.3.ssm_beta.weight")
        model._runtime_artifacts_verified = True
        model._runtime_manifest_info = {"patch_names": [], "airllm_runtime_features": []}
        with self.assertRaises(NativeGlmError):
            model._check_requested_features()


if __name__ == "__main__":
    unittest.main()
