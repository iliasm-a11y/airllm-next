"""Portable tests: python tests_next/test_glm_native.py (no ML framework imports)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import fcntl
import hashlib
import json
import os
import signal
import subprocess
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
from airllm.nextgen import glm_native as native


def gguf(fields):
    value = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for key, data in fields.items():
        name = key.encode()
        value += struct.pack("<Q", len(name)) + name
        if isinstance(data, str):
            raw = data.encode()
            value += struct.pack("<IQ", 8, len(raw)) + raw
        else:
            value += struct.pack("<II", 4, data)
    return value


class MetadataTests(unittest.TestCase):
    def test_metadata_does_not_require_tensor_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "header"
            p.write_bytes(gguf({"general.architecture": "glm5-next", "glm5-next.expert_used_count": 8}))
            self.assertEqual(native._gguf_metadata(p)["glm5-next.expert_used_count"], 8)

    def test_malformed_length_is_bounded_before_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "header"
            p.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 2**40))
            with self.assertRaisesRegex(ValueError, "limit"):
                native._gguf_metadata(p)

    def test_truncated_metadata_is_rejected_with_frozen_size_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "header"
            p.write_bytes(gguf({"general.architecture": "glm5-next"})[:-1])
            with self.assertRaises(ValueError):
                native._gguf_metadata(p)

    def test_missing_shards_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = native.inspect_native_checkpoint(tmp)
            self.assertFalse(result["integrity_verified"])
            self.assertFalse(result["weights_present"])
            self.assertEqual(len(result["errors"]), 2)

    def test_receipt_does_not_allow_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "shard.gguf"
            p.write_bytes(gguf({"general.architecture": "glm5-next"}))
            import hashlib
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            receipt = Path(tmp) / "receipt.json"
            receipt.write_text(json.dumps({"revision": native.MODEL_REVISION, "files": [
                {"name": p.name, "identity": native._identity(p), "actual_sha256": digest}]}))
            with patch.object(native, "FILES", ((p.name, p.stat().st_size, digest),)):
                self.assertTrue(native.inspect_native_checkpoint(tmp, receipt)["integrity_verified"])
                p.write_bytes(p.read_bytes())
                self.assertFalse(native.inspect_native_checkpoint(tmp, receipt)["integrity_verified"])

    def test_nonobject_receipts_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.json"
            for invalid in ([], None, 2, {"files": None}):
                receipt.write_text(json.dumps(invalid))
                result = native.inspect_native_checkpoint(tmp, receipt)
                self.assertFalse(result["integrity_verified"])

    def test_wrong_revision_cannot_reuse_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "shard.gguf"
            p.write_bytes(gguf({"general.architecture": "glm5-next"}))
            receipt = Path(tmp) / "receipt.json"
            receipt.write_text(json.dumps({"revision": "other", "files": [
                {"name": p.name, "identity": native._identity(p), "actual_sha256": "expected"}]}))
            with patch.object(native, "FILES", ((p.name, p.stat().st_size, "expected"),)):
                self.assertFalse(native.inspect_native_checkpoint(tmp, receipt)["integrity_verified"])


class SchedulerTests(unittest.TestCase):
    def test_600_logical_jobs_never_overlap_physically(self):
        scheduler = native._SerialQueue(1024)
        active, peak = 0, 0
        lock = threading.Lock()
        def task(i):
            nonlocal active, peak
            with scheduler.slot():
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.0001)
                with lock:
                    active -= 1
            return i
        with ThreadPoolExecutor(max_workers=32) as pool:
            self.assertEqual(list(pool.map(task, range(600))), list(range(600)))
        self.assertEqual(peak, 1)
        self.assertEqual(scheduler.status()["completed"], 600)

    def test_cancelled_waiter_leaves_queue(self):
        scheduler = native._SerialQueue(2)
        cancel = threading.Event()
        outcome = []
        def waiter():
            try:
                with scheduler.slot(cancel):
                    outcome.append("executed")
            except native.NativeGlmError as exc:
                outcome.append(exc.code)
        with scheduler.slot():
            worker = threading.Thread(target=waiter)
            worker.start()
            while scheduler.status()["pending"] != 1:
                time.sleep(.001)
            cancel.set()
            worker.join(timeout=2)
            self.assertEqual(outcome, ["CANCELLED"])
            self.assertEqual(scheduler.status()["pending"], 0)

    def test_queue_overflow_is_explicit(self):
        scheduler = native._SerialQueue(1)
        cancel = threading.Event()
        def wait():
            try:
                with scheduler.slot(cancel):
                    pass
            except native.NativeGlmError:
                pass
        with scheduler.slot():
            worker = threading.Thread(target=wait)
            worker.start()
            while scheduler.status()["pending"] != 1:
                time.sleep(.001)
            with self.assertRaisesRegex(native.NativeGlmError, "full"):
                with scheduler.slot():
                    pass
            cancel.set()
            worker.join(timeout=2)


class ConfigurationTests(unittest.TestCase):
    def model(self, **kwargs):
        return native.NativeGlmModel("/missing", runtime_binary="/missing-server", **kwargs)

    def test_target_lock_key_is_independent_of_path_and_state_directory(self):
        first = native.NativeGlmModel("/models/original", runtime_binary="/runtime/a",
                                      state_dir="/state/first")
        alias = native.NativeGlmModel("/models/hardlink-alias", runtime_binary="/runtime/b",
                                      state_dir="/state/second")
        self.assertEqual(first._model_lock_key(), alias._model_lock_key())
        self.assertEqual(len(first._model_lock_key()), 24)

    def test_cpu_command_cannot_offload_or_repack(self):
        command = self.model()._build_command(12345)
        self.assertIn("--no-repack", command)
        self.assertIn("--no-op-offload", command)
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertEqual(command[command.index("--device") + 1], "none")

    def test_native_environment_ignores_inherited_model_flags(self):
        env = {"LLAMA_ARG_MODEL": "/other/model", "AIRLLM_MMAP_DEMAND": "1", "AIRLLM_OTHER": "x"}
        with patch.dict(native.os.environ, env):
            self.assertNotIn("AIRLLM_MMAP_DEMAND", self.model()._runtime_environment())
            selected = self.model(mmap_demand=True)._runtime_environment()
            self.assertEqual(selected["AIRLLM_MMAP_DEMAND"], "1")
            self.assertNotIn("LLAMA_ARG_MODEL", selected)
            self.assertNotIn("AIRLLM_OTHER", selected)

    def test_expert_prefetch_is_opt_in_and_scrubs_inherited_flag(self):
        with patch.dict(native.os.environ, {"AIRLLM_EXPERT_PREFETCH": "1"}):
            self.assertNotIn("AIRLLM_EXPERT_PREFETCH", self.model()._runtime_environment())
            self.assertEqual(self.model(expert_prefetch=True)._runtime_environment()["AIRLLM_EXPERT_PREFETCH"], "1")
        with self.assertRaises(ValueError):
            self.model(expert_prefetch="1")

    def test_dense_prefetch_is_opt_in_and_scrubs_inherited_flag(self):
        with patch.dict(native.os.environ, {"AIRLLM_DENSE_PREFETCH": "1"}):
            self.assertNotIn("AIRLLM_DENSE_PREFETCH", self.model()._runtime_environment())
            self.assertEqual(self.model(dense_prefetch=True)._runtime_environment()["AIRLLM_DENSE_PREFETCH"], "1")
        with self.assertRaises(ValueError):
            self.model(dense_prefetch="1")

    def test_hybrid_experts_stay_cpu_until_explicit_cache(self):
        command = self.model(gpu_layers=46, cpu_moe=True, moe_cache_mib=256)._build_command(12345)
        self.assertIn("--cpu-moe", command)
        self.assertNotIn("--device", command)
        self.assertEqual(command[command.index("--moe-cache-mib") + 1], "256")

    def test_gpu_without_explicit_experimental_optin_is_refused_before_any_spawn(self):
        model = self.model(gpu_layers=6, cpu_moe=True)
        with patch.object(native, "inspect_native_checkpoint") as inspect, patch.object(native.subprocess, "Popen") as spawn:
            with self.assertRaises(native.NativeGlmError) as caught:
                model.load()
        self.assertEqual(caught.exception.code, "GPU_FIDELITY_UNQUALIFIED")
        inspect.assert_not_called()
        spawn.assert_not_called()
        self.assertFalse(caught.exception.details["gpu_qualified"])
        with self.assertRaises(ValueError):
            self.model(allow_unqualified_gpu="true")

    def test_explicit_gpu_optin_still_checks_device_and_memory_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-server"
            binary.write_bytes(b"wrapper")
            binary.chmod(0o700)
            model = native.NativeGlmModel("/missing", runtime_binary=binary, gpu_layers=6,
                                          cpu_moe=True, allow_unqualified_gpu=True)
            checkpoint = {"errors": [], "architecture_matches": True, "integrity_verified": True, "files": []}
            versions = [subprocess.CompletedProcess([], 0, stdout="commit 3f8af15", stderr=""),
                        subprocess.CompletedProcess([], 0, stdout="Available devices:\n MTL0: Apple M5", stderr="")]
            with patch.object(native, "inspect_native_checkpoint", return_value=checkpoint), \
                 patch.object(native.subprocess, "run", side_effect=versions) as command, \
                 patch.object(model, "_admit", side_effect=native.NativeGlmError("ADMISSION_PROBE", "bounded memory")) as admit, \
                 patch.object(native.subprocess, "Popen") as spawn:
                with self.assertRaises(native.NativeGlmError) as caught:
                    model.load()
            self.assertEqual(caught.exception.code, "ADMISSION_PROBE")
            self.assertEqual(command.call_count, 2)
            admit.assert_called_once()
            spawn.assert_not_called()
            versions[1] = subprocess.CompletedProcess([], 0, stdout=" BLAS: Accelerate", stderr="")
            with patch.object(native, "inspect_native_checkpoint", return_value=checkpoint), \
                 patch.object(native.subprocess, "run", side_effect=versions), \
                 patch.object(model, "_admit") as admit:
                with self.assertRaises(native.NativeGlmError) as caught:
                    model.load()
            self.assertEqual(caught.exception.code, "GPU_UNAVAILABLE")
            admit.assert_not_called()

    def test_cpu_numerical_scope_does_not_claim_global_or_gpu_qualification(self):
        qualification = self.model().status()["numerical_qualification"]
        self.assertEqual(qualification["selected_path"], "cpu")
        self.assertFalse(qualification["global_numerical_equivalence_validated"])
        self.assertFalse(qualification["gpu_qualified"])
        self.assertEqual(qualification["cpu_evidence_scope"]["full_logits_compared"], 619520)

    def test_gpu_admission_counts_pinned_last_block_weights(self):
        model = self.model(gpu_layers=6, cpu_moe=True)
        self.assertEqual(model._gpu_weight_bytes(), 1605374848)
        with patch.object(native, "resource_snapshot", return_value={"available_bytes": 2 * native.GiB, "pressure_level": 1}):
            with self.assertRaises(native.NativeGlmError) as caught:
                model._admit()
        self.assertEqual(caught.exception.details["gpu_weight_and_cache_bytes"], 1605374848)
        model._process = type("Running", (), {"poll": lambda self: None})()
        with patch.object(native, "resource_snapshot", return_value={"available_bytes": 2 * native.GiB, "pressure_level": 1}):
            model._admit()  # Resident GPU weights are not counted twice after loading.

    def test_gpu_layer_placement_includes_output_before_transformer_blocks(self):
        expected = {0: 0, 1: 674054144, 2: 858540160, 6: 1605374848,
                    8: 1983237504, 45: 9147183104, 46: 9460642816}
        for layers, size in expected.items():
            with self.subTest(layers=layers):
                self.assertEqual(self.model(gpu_layers=layers, cpu_moe=True)._gpu_weight_bytes(), size)

    def test_gpu_expert_banks_are_not_admitted_without_streaming(self):
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model(gpu_layers=46)._admit()
        self.assertEqual(caught.exception.code, "GPU_ADMISSION")

    def test_loopback_server_requires_private_authentication(self):
        model = self.model()
        model._api_key = "test-secret"
        model._key_file = Path("/private/key")
        self.assertEqual(model._headers()["Authorization"], "Bearer test-secret")
        command = model._build_command(12345)
        self.assertIn("--api-key-file", command)
        self.assertNotIn("test-secret", command)
        self.assertIn("--no-cors-credentials", command)
        with patch.dict(native.os.environ, {"LLAMA_API_KEY": "inherited"}):
            self.assertNotIn("LLAMA_API_KEY", model._runtime_environment())

    def test_close_interrupts_inflight_checkpoint_verification(self):
        model = self.model()
        entered = threading.Event()
        result = []
        def inspect(*args, cancel=None, **kwargs):
            if not kwargs.get("verify_hashes"):
                return {}
            entered.set()
            while not cancel.is_set():
                time.sleep(.01)
            raise native.NativeGlmError("CANCELLED", "cancelled")
        def load():
            try:
                model.load()
            except native.NativeGlmError as error:
                result.append(error.code)
        with patch.object(native, "inspect_native_checkpoint", side_effect=inspect):
            loader = threading.Thread(target=load)
            loader.start()
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            model.close()
            loader.join(timeout=1)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(result, ["CANCELLED"])

    def test_pinned_mtl_device_listing_is_recognized(self):
        listing = "Available devices:\n  MTL0: Apple M5 (Memory: 16384 MiB)\n  BLAS: Accelerate\n"
        self.assertTrue(native._metal_device_present(listing))
        self.assertTrue(native._metal_device_present("Metal0: Apple M5"))
        self.assertFalse(native._metal_device_present("Available devices:\n BLAS: Accelerate"))
        self.assertFalse(native._metal_device_present("Metal support compiled in, no usable GPU"))

    def test_invalid_config_rejected(self):
        for options in ({"context_tokens": 2**20}, {"threads": 0}, {"gpu_layers": 47},
                        {"moe_cache_mib": 1}, {"batch_size": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options)

    def test_mmap_admits_under_6_gib_but_reserves_workspace(self):
        model = self.model()
        with patch.object(native, "resource_snapshot", return_value={"available_bytes": 3 * native.GiB, "pressure_level": 1}):
            model._admit()
        with patch.object(native, "resource_snapshot", return_value={"available_bytes": 600 * native.MiB, "pressure_level": 1}):
            with self.assertRaises(native.NativeGlmError) as caught:
                model._admit()
        self.assertEqual(caught.exception.code, "MEMORY_ADMISSION")

    def test_warning_pressure_is_not_admitted(self):
        with patch.object(native, "resource_snapshot", return_value={"available_bytes": 10 * native.GiB, "pressure_level": 2}):
            with self.assertRaises(native.NativeGlmError):
                self.model()._admit()

    def test_context_budget_is_checked_before_generation(self):
        model = self.model(context_tokens=64)
        model._http_json = lambda *a, **k: {"tokens": list(range(60))}
        with self.assertRaises(native.NativeGlmError) as caught:
            model._payload("prompt", 8, False, 0)
        self.assertEqual(caught.exception.code, "CONTEXT_LIMIT")

    def test_loading_cancelled_before_io(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model().load(cancel=cancel)
        self.assertEqual(caught.exception.code, "CANCELLED")

    def test_dispatch_uses_no_transformers(self):
        from airllm import AutoModel
        model = AutoModel.from_pretrained("/missing", backend="next-glm-native", runtime_binary="/missing-server")
        self.assertIsInstance(model, native.NativeGlmModel)



class RuntimeManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable = self.bin / "llama-server"
        self.executable.write_bytes(b"native-wrapper")
        self.library = self.bin / "libggml.0.dylib"
        self.library.write_bytes(b"native-math")
        self.link = self.bin / "libggml.dylib"
        self.link.symlink_to(self.library.name)
        self.manifest = self.root / "manifest.json"
        def record(path):
            return {"path": "bin/" + path.name, "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.document = {"status": "built", "source_revision": native.RUNTIME_REVISION,
                         "artifacts": [record(self.executable), record(self.library),
                                       {"path": "bin/" + self.link.name, "symlink": self.library.name}]}
        self.save()
        self.model = native.NativeGlmModel("/missing", runtime_binary=self.executable,
                                          runtime_manifest_file=self.manifest)

    def tearDown(self):
        self.model.close()
        self.temporary.cleanup()

    def save(self, document=None):
        self.manifest.write_text(json.dumps(self.document if document is None else document))

    def test_manifest_verifies_executable_library_and_internal_symlink(self):
        self.model._verify_runtime_manifest()
        self.assertTrue(self.model._runtime_artifacts_verified)
        self.assertEqual(self.model._runtime_manifest_info["artifact_count"], 3)
        self.model._check_runtime_identity()

    def test_library_hash_mismatch_is_rejected_before_spawn(self):
        self.document["artifacts"][1]["sha256"] = "0" * 64
        self.save()
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model._verify_runtime_manifest()
        self.assertEqual(caught.exception.code, "RUNTIME_MANIFEST")
        self.assertIn("SHA256 mismatch", str(caught.exception))
        self.assertFalse(self.model._runtime_artifacts_verified)

    def test_library_change_discards_previously_verified_runtime(self):
        self.model._verify_runtime_manifest()
        self.library.write_bytes(b"changed-math")
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model._check_running()
        self.assertEqual(caught.exception.code, "RUNTIME_ARTIFACT_CHANGED")
        self.assertFalse(self.model._runtime_artifacts_verified)

    def test_manifest_traversal_and_external_symlinks_are_rejected(self):
        self.document["artifacts"][0]["path"] = "bin/../llama-server"
        self.save()
        with self.assertRaises(native.NativeGlmError):
            self.model._verify_runtime_manifest()
        self.document["artifacts"][0]["path"] = "bin/llama-server"
        self.document["artifacts"][2]["symlink"] = "../external.dylib"
        self.save()
        with self.assertRaises(native.NativeGlmError):
            self.model._verify_runtime_manifest()

    def test_nonobject_and_wrong_source_manifest_are_rejected(self):
        for document in ([], 3, "built", {"status": "built", "source_revision": "other"}):
            with self.subTest(document=document):
                self.save(document)
                with self.assertRaises(native.NativeGlmError) as caught:
                    self.model._verify_runtime_manifest()
                self.assertEqual(caught.exception.code, "RUNTIME_MANIFEST")

    def test_unlisted_packaged_library_cannot_escape_verification(self):
        (self.bin / "libunverified.dylib").write_bytes(b"extra")
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model._verify_runtime_manifest()
        self.assertIn("missing from runtime manifest", str(caught.exception))

    def test_legacy_runtime_does_not_claim_dependency_qualification(self):
        legacy = native.NativeGlmModel("/missing", runtime_binary=self.executable)
        legacy._verify_runtime_manifest()
        self.assertFalse(legacy.status()["runtime_artifacts_verified"])
        self.assertIn("unqualified", legacy.status()["runtime_artifact_scope"])

    def test_runtime_manifest_hashing_observes_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(native.NativeGlmError) as caught:
            self.model._verify_runtime_manifest(cancel)
        self.assertEqual(caught.exception.code, "CANCELLED")

    def test_dyld_inherited_overrides_are_removed(self):
        with patch.dict(native.os.environ, {"DYLD_LIBRARY_PATH": "/other/libs", "DYLD_INSERT_LIBRARIES": "/other/lib.dylib"}):
            env = self.model._runtime_environment()
        self.assertNotIn("DYLD_LIBRARY_PATH", env)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", env)


class ProcessOwnershipTests(unittest.TestCase):
    def test_inherited_lock_survives_python_parent_sigkill(self):
        program = """import fcntl, subprocess, sys, time
stream = open(sys.argv[1], 'a')
fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
child = subprocess.Popen(['/bin/sleep', '30'], pass_fds=(stream.fileno(),))
print(child.pid, flush=True)
time.sleep(30)
"""
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "model.lock"
            parent = subprocess.Popen([sys.executable, "-c", program, str(lock_path)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child_pid = None
            try:
                line = parent.stdout.readline()
                self.assertTrue(line.strip(), parent.stderr.read() if parent.poll() is not None else "missing child pid")
                child_pid = int(line)
                parent.kill()
                parent.wait(timeout=2)
                with lock_path.open("a") as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.kill(child_pid, signal.SIGTERM)
                    child_pid = None
                    deadline = time.monotonic() + 2
                    while True:
                        try:
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                self.fail("model lock remained held after its native owner exited")
                            time.sleep(.01)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=2)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                parent.stdout.close()
                parent.stderr.close()


class PressureGuardTests(unittest.TestCase):
    def setUp(self):
        self.model = native.NativeGlmModel("/missing", runtime_binary="/missing-server")
        self.baseline = {"swap_used_bytes": 10 * native.MiB}

    def sample(self, *, pressure=2, available=None, swap_growth=0, rss=None):
        return {"pressure_level": pressure,
                "available_bytes": available if available is not None else int(3.6 * native.GiB),
                "swap_used_bytes": self.baseline["swap_used_bytes"] + swap_growth,
                "rss_bytes": rss if rss is not None else 6 * native.GiB}

    def test_warning_with_measured_headroom_can_continue_230_seconds(self):
        sample = self.sample(swap_growth=3 * native.MiB)
        for now in (0, 19, 20, 140, 230):
            self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, now))
        self.assertEqual(self.model._pressure_context["action"], "continue_under_monitoring")
        self.assertEqual(self.model._pressure_context["warning_seconds"], 230)
        self.assertIn("other workloads unqualified", self.model._pressure_context["evidence_scope"])

    def test_warning_and_low_headroom_stops_after_warning_grace(self):
        sample = self.sample(available=self.model.limits.reserve_bytes + 128 * native.MiB)
        self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, 0))
        self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, 19))
        self.assertEqual(self.model._guard_resource_sample(sample, self.baseline, 20), "MEMORY_PRESSURE_HEADROOM")

    def test_warning_and_swap_growth_33_mib_stops_after_grace(self):
        sample = self.sample(swap_growth=33 * native.MiB)
        self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, 0))
        self.assertEqual(self.model._guard_resource_sample(sample, self.baseline, 20), "MEMORY_PRESSURE_SWAP")

    def test_critical_pressure_rss_and_swap_caps_remain_immediate(self):
        cases = ((self.sample(pressure=4), "MEMORY_PRESSURE_CRITICAL"),
                 (self.sample(rss=9 * native.GiB + 1), "RSS_LIMIT"),
                 (self.sample(pressure=1, swap_growth=64 * native.MiB + 1), "SWAP_LIMIT"))
        for sample, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.model._guard_resource_sample(sample, self.baseline, 0), expected)

    def test_available_below_reserve_stops_after_short_grace_even_without_warning(self):
        sample = self.sample(pressure=1, available=self.model.limits.reserve_bytes - 1)
        self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, 0))
        self.assertIsNone(self.model._guard_resource_sample(sample, self.baseline, 2.9))
        self.assertEqual(self.model._guard_resource_sample(sample, self.baseline, 3), "MEMORY_HEADROOM_LIMIT")

    def loaded(self):
        self.model._process = type("Running", (), {"poll": lambda self: None})()
        self.model._resource_baseline = self.baseline

    def test_warm_warning_admission_requires_workspace_and_bounded_swap(self):
        self.loaded()
        with patch.object(native, "resource_snapshot", return_value=self.sample(swap_growth=3 * native.MiB)):
            self.model._admit()
        for sample in (self.sample(available=1700 * native.MiB), self.sample(swap_growth=33 * native.MiB)):
            with self.subTest(sample=sample), patch.object(native, "resource_snapshot", return_value=sample):
                with self.assertRaises(native.NativeGlmError) as caught:
                    self.model._admit()
                self.assertEqual(caught.exception.code, "MEMORY_ADMISSION")

    def test_warm_admission_does_not_ignore_existing_guard_failure(self):
        self.loaded()
        self.model._failure = {"code": "SWAP_LIMIT", "message": "stopped"}
        with patch.object(native, "resource_snapshot", return_value=self.sample()):
            with self.assertRaises(native.NativeGlmError) as caught:
                self.model._admit()
        self.assertEqual(caught.exception.code, "SWAP_LIMIT")


class StreamTests(unittest.TestCase):
    @contextmanager
    def server(self, terminal=True):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.payload = payload
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": ping\n\n")
                self.wfile.write(b'data: {"content":"Paris","tokens":[123],"stop":false}\n\n')
                if terminal:
                    self.wfile.write(b'data: {"content":"","tokens":[],"stop":true,"timings":{"predicted_n":1}}\n\n')
                self.wfile.flush()
        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                # Disconnect is the expected server-side result of cancellation.
                pass
        server = QuietServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def model(self, server):
        model = native.NativeGlmModel("/missing", runtime_binary="/missing-server")
        model._port = server.server_port
        model._check_running = lambda: None
        model._check_identity = lambda: None
        return model

    def test_sse_preserves_native_tokens_and_terminal_timings(self):
        with self.server() as server:
            model = self.model(server)
            events = list(model._events({"prompt": "x", "stream": True}, None))
            self.assertEqual(events[0]["tokens"], [123])
            self.assertTrue(events[-1]["stop"])
            self.assertEqual(model.last_request["streamed_token_count"], 1)
            self.assertEqual(model.last_request["native"]["timings"]["predicted_n"], 1)

    def test_failed_request_clears_previous_success_metrics(self):
        with self.server(terminal=False) as server:
            model = self.model(server)
            model.last_request = {"seconds": 1, "native": {"stop": True}}
            with self.assertRaises(native.NativeGlmError):
                list(model._events({"prompt": "x"}, None))
            self.assertIsNone(model.last_request)

    def test_broken_stream_is_never_reported_complete(self):
        with self.server(terminal=False) as server:
            model = self.model(server)
            with self.assertRaises(native.NativeGlmError) as caught:
                list(model._events({"prompt": "x"}, None))
            self.assertEqual(caught.exception.code, "INCOMPLETE_STREAM")
            self.assertIsNone(model.last_request)

    def test_cancellation_unblocks_reader_before_http_headers(self):
        import socket
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        received, release, cancel = threading.Event(), threading.Event(), threading.Event()
        outcome = []
        def silent_server():
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)
                received.set()
                release.wait(timeout=4)
        worker = threading.Thread(target=silent_server, daemon=True)
        worker.start()
        model = native.NativeGlmModel("/missing", runtime_binary="/missing-server")
        model._port = listener.getsockname()[1]
        model._check_running = lambda: None
        def consume():
            try:
                list(model._events({"prompt": "x"}, cancel))
            except native.NativeGlmError as error:
                outcome.append(error.code)
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(received.wait(timeout=2))
        started = time.monotonic()
        cancel.set()
        consumer.join(timeout=1)
        elapsed = time.monotonic() - started
        release.set()
        listener.close()
        worker.join(timeout=2)
        self.assertFalse(consumer.is_alive())
        self.assertLess(elapsed, 1)
        self.assertEqual(outcome, ["CANCELLED"])

    def test_cancellation_discards_native_state(self):
        with self.server() as server:
            model = self.model(server)
            cancel = threading.Event()
            cancel.set()
            with patch.object(model, "close") as close:
                with self.assertRaises(native.NativeGlmError) as caught:
                    list(model._events({"prompt": "x"}, cancel))
                self.assertEqual(caught.exception.code, "CANCELLED")
                close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
