"""Portable process-observation/configuration tests; no model/process is launched.

Run this module directly or through unittest. No ML framework is imported.
"""
from pathlib import Path
import os
import signal
import subprocess
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/ronin/engine"))
from airllm.nextgen import glm_native as native
from adapter import Engine


class OwnedProcessFixture:
    pid = 71003

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


class OwnedProcessExitTests(unittest.TestCase):
    def model(self, **kwargs):
        model = native.NativeGlmModel("/missing-model", runtime_binary="/missing-runtime",
                                      **({"threads": 1} | kwargs))
        model._checkpoint = {"errors": []}
        return model

    def test_exit_is_polled_before_close_drops_handle_and_survives_repeated_close(self):
        model = self.model()
        process = OwnedProcessFixture(returncode=0)
        model._process = process
        with mock.patch.object(native.os, "killpg") as kill:
            model.close()
            record = model.last_owned_process_exit
            model.close()
        kill.assert_not_called()
        self.assertIsNone(model._process)
        self.assertEqual(model.last_owned_process_exit, record)
        self.assertEqual(record["pid"], process.pid)
        self.assertEqual(record["returncode"], 0)
        self.assertTrue(record["exit_observed"])
        self.assertEqual(record["observation"], "popen_poll_of_owned_process")
        self.assertEqual(record["source"], "close_after_termination_attempt")
        self.assertGreater(record["observed_monotonic"], 0)
        self.assertGreater(record["observed_unix_seconds"], 0)

    def test_normal_close_keeps_existing_sigterm_and_wait_policy(self):
        model = self.model()
        process = OwnedProcessFixture()
        model._process = process
        with mock.patch.object(native.os, "killpg") as kill:
            model.close()
        kill.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(process.wait_timeouts, [5])
        self.assertEqual(model.status()["last_owned_process_exit"]["returncode"], -signal.SIGTERM)
        self.assertTrue(model.last_owned_process_exit["exit_observed"])

    def test_resource_watchdog_retains_exit_and_original_guard_code(self):
        model = self.model()
        process = OwnedProcessFixture()
        model._process = process
        sample = {"available_bytes": 4 * native.GiB, "rss_bytes": 5 * native.GiB,
                  "swap_used_bytes": 200 * native.MiB, "pressure_level": 2}
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(native, "resource_snapshot", return_value=sample), \
             mock.patch.object(model, "_guard_resource_sample", return_value="SWAP_LIMIT"), \
             mock.patch.object(native.os, "killpg") as kill:
            model._watch({"swap_used_bytes": 0})
            initial = model.last_owned_process_exit
            model.close()
        kill.assert_called_once_with(process.pid, signal.SIGTERM)
        self.assertEqual(initial["source"], "resource_guard")
        self.assertEqual(initial["guard_failure_code"], "SWAP_LIMIT")
        self.assertTrue(initial["exit_observed"])
        self.assertEqual(model.last_owned_process_exit["guard_failure_code"], "SWAP_LIMIT")
        self.assertEqual(model._failure["code"], "SWAP_LIMIT")

    def test_watchdog_monitor_failure_does_not_lose_owned_exit_observation(self):
        model = self.model()
        process = OwnedProcessFixture()
        model._process = process
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(native, "resource_snapshot", side_effect=OSError("fixture monitor failure")), \
             mock.patch.object(native.os, "killpg"):
            model._watch({})
        self.assertEqual(model._failure["code"], "WATCHDOG_ERROR")
        self.assertEqual(model.last_owned_process_exit["source"], "watchdog_error")
        self.assertTrue(model.last_owned_process_exit["exit_observed"])

    def test_termination_error_is_not_hidden_or_reported_as_confirmed_exit(self):
        model = self.model()
        process = OwnedProcessFixture()
        model._process = process
        error = subprocess.TimeoutExpired("fixture", 5)
        with mock.patch.object(model, "_terminate_process", side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired):
                model.close()
        self.assertIs(model._process, process)  # Original failed-close behavior is preserved.
        self.assertFalse(model.last_owned_process_exit["exit_observed"])
        self.assertIsNone(model.last_owned_process_exit["returncode"])

    def test_poll_error_and_missing_handle_cannot_fabricate_an_exit(self):
        model = self.model()
        model.close()
        self.assertIsNone(model.last_owned_process_exit)
        process = OwnedProcessFixture()
        model._process = process
        with mock.patch.object(model, "_terminate_process"), \
             mock.patch.object(process, "poll", side_effect=OSError("fixture poll failure")):
            model.close()
        record = model.last_owned_process_exit
        self.assertFalse(record["exit_observed"])
        self.assertIsNone(record["returncode"])
        self.assertEqual(record["poll_error"], "OSError")

    def test_late_old_generation_cannot_overwrite_newer_exit(self):
        model = self.model()
        model._owned_process_generation = 2
        current = OwnedProcessFixture(returncode=0)
        model._observe_owned_process_exit(current, "fixture_current", 2)
        expected = model.last_owned_process_exit
        old = OwnedProcessFixture(returncode=-signal.SIGKILL)
        with mock.patch.object(old, "poll") as poll:
            model._observe_owned_process_exit(old, "fixture_stale_watcher", 1)
        poll.assert_not_called()
        self.assertEqual(model.last_owned_process_exit, expected)

    def test_status_returns_detached_exit_record(self):
        model = self.model()
        model._observe_owned_process_exit(OwnedProcessFixture(returncode=0), "fixture")
        status = model.status()
        status["last_owned_process_exit"]["pid"] = -1
        self.assertEqual(model.last_owned_process_exit["pid"], OwnedProcessFixture.pid)

    def test_stale_watchdog_cannot_sample_guard_mutate_or_terminate_new_process(self):
        model = self.model()
        model._owned_process_generation = 2
        current = OwnedProcessFixture()
        model._process = current
        sentinel = {"code": "NEW_GENERATION_SENTINEL"}
        model._failure = sentinel
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(current, "poll") as poll, \
             mock.patch.object(native, "resource_snapshot") as sample, \
             mock.patch.object(model, "_guard_resource_sample") as guard, \
             mock.patch.object(model, "_terminate_process") as terminate:
            model._watch({}, owned_generation=1)
        poll.assert_not_called()
        sample.assert_not_called()
        guard.assert_not_called()
        terminate.assert_not_called()
        self.assertIs(model._failure, sentinel)
        self.assertEqual(list(model._samples), [])
        self.assertIsNone(model.last_owned_process_exit)

    def test_replaced_popen_identity_is_rechecked_after_poll_before_sampling(self):
        model = self.model()
        old, current = OwnedProcessFixture(), OwnedProcessFixture()
        model._process = old
        model._owned_process_generation = 1
        def replace_during_poll():
            with model._exit_observation_lock:
                model._process = current  # Same PID and generation: object identity still matters.
            return None
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(old, "poll", side_effect=replace_during_poll), \
             mock.patch.object(native, "resource_snapshot") as sample, \
             mock.patch.object(model, "_guard_resource_sample") as guard, \
             mock.patch.object(model, "_terminate_process") as terminate:
            model._watch({}, owned_generation=1)
        sample.assert_not_called()
        guard.assert_not_called()
        terminate.assert_not_called()
        self.assertIs(model._process, current)
        self.assertIsNone(model._failure)

    def test_reload_during_snapshot_discards_entire_sample_without_guard_mutation(self):
        model = self.model()
        old, current = OwnedProcessFixture(), OwnedProcessFixture()
        model._process = old
        model._owned_process_generation = 1
        context = {"sentinel": "new state must stay intact"}
        model._pressure_context = context
        def snapshot_then_reload(pid):
            self.assertEqual(pid, old.pid)
            with model._exit_observation_lock:
                model._process = current
                model._owned_process_generation = 2
            return {"available_bytes": 1, "rss_bytes": 100 * native.GiB,
                    "swap_used_bytes": 100 * native.GiB, "pressure_level": 4}
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(native, "resource_snapshot", side_effect=snapshot_then_reload), \
             mock.patch.object(model, "_guard_resource_sample") as guard, \
             mock.patch.object(model, "_terminate_process") as terminate:
            model._watch({}, owned_generation=1)
        guard.assert_not_called()
        terminate.assert_not_called()
        self.assertEqual(list(model._samples), [])
        self.assertEqual(model._peak_rss_bytes, 0)
        self.assertIs(model._pressure_context, context)
        self.assertIsNone(model._failure)
        self.assertIsNone(model.last_owned_process_exit)

    def test_stale_snapshot_error_cannot_set_watchdog_error_on_replacement(self):
        model = self.model()
        model._process = OwnedProcessFixture()
        model._owned_process_generation = 1
        current = OwnedProcessFixture()
        def error_after_reload(pid):
            with model._exit_observation_lock:
                model._process = current
                model._owned_process_generation = 2
            raise OSError("old process vanished while sampling")
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(native, "resource_snapshot", side_effect=error_after_reload), \
             mock.patch.object(model, "_guard_resource_sample") as guard, \
             mock.patch.object(model, "_terminate_process") as terminate:
            model._watch({}, owned_generation=1)
        guard.assert_not_called()
        terminate.assert_not_called()
        self.assertIsNone(model._failure)
        self.assertIs(model._process, current)

    def test_owner_publication_cannot_interleave_between_guard_and_termination(self):
        model = self.model()
        old, current = OwnedProcessFixture(), OwnedProcessFixture()
        model._process = old
        model._owned_process_generation = 1
        started, finished = threading.Event(), threading.Event()
        publishers = []
        def publish_replacement():
            started.set()
            with model._exit_observation_lock:
                model._process = current
                model._owned_process_generation = 2
                model._last_owned_process_exit = None
                model._failure = None
            finished.set()
        def guard(sample, baseline, now):
            publisher = threading.Thread(target=publish_replacement)
            publishers.append(publisher)
            publisher.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(finished.is_set())
            return "SWAP_LIMIT"
        def terminate(process):
            self.assertIs(process, old)
            self.assertIs(model._process, old)
            self.assertEqual(model._owned_process_generation, 1)
            self.assertFalse(finished.is_set())
            old.returncode = -signal.SIGTERM
        with mock.patch.object(model._watch_stop, "wait", return_value=False), \
             mock.patch.object(native, "resource_snapshot", return_value={"rss_bytes": 1}), \
             mock.patch.object(model, "_guard_resource_sample", side_effect=guard), \
             mock.patch.object(model, "_terminate_process", side_effect=terminate) as termination:
            model._watch({}, owned_generation=1)
        for publisher in publishers:
            publisher.join(2)
            self.assertFalse(publisher.is_alive())
        self.assertTrue(finished.is_set())
        termination.assert_called_once_with(old)
        self.assertIs(model._process, current)
        self.assertEqual(model._owned_process_generation, 2)
        self.assertIsNone(model._failure)
        self.assertIsNone(model.last_owned_process_exit)


class MmapNoCacheConfigurationTests(unittest.TestCase):
    def model(self, **kwargs):
        model = native.NativeGlmModel("/missing-model", runtime_binary="/missing-runtime",
                                      **({"threads": 1} | kwargs))
        model._checkpoint = {"errors": []}
        return model

    def test_nocache_is_explicit_boolean_and_off_by_default(self):
        model = self.model()
        self.assertFalse(model.mmap_nocache)
        self.assertFalse(model.status()["configuration"]["mmap_nocache"])
        for invalid in (None, 0, 1, "1", "false"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model(mmap_nocache=invalid)

    def test_inherited_flags_are_cleaned_and_only_explicit_nocache_is_forwarded(self):
        with mock.patch.dict(os.environ, {"AIRLLM_MMAP_NOCACHE": "1", "AIRLLM_UNKNOWN": "1",
                                         "LLAMA_ARG_SPEC_TYPE": "ngram-simple", "DYLD_INSERT_LIBRARIES": "/fixture"}, clear=True):
            reference = self.model()
            candidate = self.model(mmap_nocache=True)
            reference_env = reference._runtime_environment()
            candidate_env = candidate._runtime_environment()
        self.assertNotIn("AIRLLM_MMAP_NOCACHE", reference_env)
        self.assertEqual(candidate_env, reference_env | {"AIRLLM_MMAP_NOCACHE": "1"})
        self.assertNotIn("AIRLLM_UNKNOWN", candidate_env)
        self.assertNotIn("LLAMA_ARG_SPEC_TYPE", candidate_env)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", candidate_env)
        self.assertTrue(candidate.status()["configuration"]["mmap_nocache"])

    def test_nocache_changes_neither_native_command_nor_resource_limits(self):
        reference = self.model(mmap_demand=True, expert_prefetch=True, dense_prefetch=True)
        candidate = self.model(mmap_demand=True, expert_prefetch=True, dense_prefetch=True, mmap_nocache=True)
        self.assertEqual(candidate._build_command(1234), reference._build_command(1234))
        self.assertEqual(candidate.limits, reference.limits)
        self.assertEqual(candidate.threads, reference.threads)
        self.assertEqual(candidate.batch_size, reference.batch_size)
        self.assertEqual(candidate.gpu_layers, reference.gpu_layers)

    def test_adapter_json_priority_and_existing_boolean_environment_contract(self):
        with mock.patch.dict(os.environ, {"RONIN_GLM_MMAP_NOCACHE": "1"}, clear=True):
            implicit = Engine(config={"runtime_binary": "/fixture"})
            explicit = Engine(config={"runtime_binary": "/fixture", "mmap_nocache": False})
            self.assertIs(implicit._options()["mmap_nocache"], True)
            self.assertIs(explicit._options()["mmap_nocache"], False)
            self.assertIsNone(implicit._model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
