"""Pinned GLM-5.3-Flash GGUF through a persistent, guarded native runtime.

No GGUF tensor is dequantized by Python. File-backed mmap pages are reclaimable;
startup admission budgets anonymous workspace, then the watchdog checks the real
process and system. The 1 GiB base exceeds the observed 887584792-byte peak
physical footprint at CPU4/context512/batch8; context/batch growth below is a
conservative estimate, not measured qualification of other configurations.
Warning pressure alone was observed for 230 s with stable mapped weights,
>3 GB available and 2.95 MB swap growth. Warm admission therefore also checks
headroom and swap; this observation does not qualify other workloads.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time

MODEL_ID = "ggml-org/GLM-5.3-Flash-GGUF"
MODEL_REVISION = "119034bc3d9d76eb41f91508e6f9fb19fde42672"
RUNTIME_REVISION = "3f8af15a76ac8675812e856da45454bafc90f12c"
FILES = (
    ("GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf", 9430112,
     "22e3599ff1cac02e854269b610030800e319151e1ed01838d7087f20b187091d"),
    ("GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf", 110017922752,
     "61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4"),
)
MiB = 1024**2
GiB = 1024**3


class NativeGlmError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def _identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _sha256_file(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if cancel is not None and cancel.is_set():
                raise NativeGlmError("CANCELLED", "Runtime verification cancelled")
            chunk = stream.read(4 * MiB)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _link_identity(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _gguf_metadata(path):
    """Bounded metadata-only parser; skip tokenizer arrays without allocating them."""
    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with path.open("rb") as stream:
        metadata_limit = min(os.fstat(stream.fileno()).st_size, 64 * MiB)
        def read(n):
            if n < 0 or stream.tell() + n > metadata_limit:
                raise ValueError("GGUF metadata limit exceeded")
            value = stream.read(n)
            if len(value) != n:
                raise ValueError("Truncated GGUF metadata")
            return value

        def string(keep=True):
            n = struct.unpack("<Q", read(8))[0]
            value = read(n)
            return value.decode("utf-8") if keep else None

        def value(kind, keep=True, depth=0):
            if kind == 8:
                return string(keep)
            if kind == 9:
                if depth:
                    raise ValueError("Nested GGUF array")
                sub, n = struct.unpack("<IQ", read(12))
                if n > 2_000_000:
                    raise ValueError("GGUF array limit exceeded")
                for _ in range(n):
                    value(sub, False, depth + 1)
                return {"element_type": sub, "count": n}
            fmt = scalars.get(kind)
            if fmt is None:
                raise ValueError("Unknown GGUF metadata type")
            result = struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]
            return result if keep else None

        if read(4) != b"GGUF":
            raise ValueError("Not GGUF")
        version, tensors, fields = struct.unpack("<IQQ", read(20))
        if version != 3 or fields > 4096 or tensors > 100_000:
            raise ValueError("Unsupported GGUF header")
        metadata = {}
        for _ in range(fields):
            key = string()
            kind = struct.unpack("<I", read(4))[0]
            result = value(kind, not key.startswith("tokenizer."))
            if not key.startswith("tokenizer."):
                metadata[key] = result
        return metadata


def inspect_native_checkpoint(model_path, verification_file=None, *, verify_hashes=False, cancel=None):
    """Inspect only the target; re-use a SHA receipt solely for unchanged files."""
    root = Path(model_path).expanduser().resolve()
    result = {"path": str(root), "model_id": MODEL_ID, "revision": MODEL_REVISION,
              "quantization": "Q2_K_S", "exists": root.is_dir(),
              "architecture_matches": False, "weights_present": False,
              "integrity_verified": False, "errors": [], "files": []}
    receipt = {}
    if verification_file:
        try:
            evidence = Path(verification_file)
            if evidence.stat().st_size > MiB:
                raise ValueError("Verification receipt exceeds limit")
            receipt = json.loads(evidence.read_text())
        except (OSError, ValueError, TypeError):
            receipt = {}
    if not isinstance(receipt, dict) or not isinstance(receipt.get("files", []), list):
        receipt = {}
    old = {item.get("name"): item for item in receipt.get("files", []) if isinstance(item, dict)}
    for name, size, expected in FILES:
        path = root / name
        item = {"name": name, "expected_bytes": size, "sha256": expected,
                "integrity_verified": False}
        try:
            identity = _identity(path)
            item.update(identity=identity, bytes=identity[2])
            if identity[2] != size:
                raise ValueError("Unexpected shard length: " + name)
            previous = old.get(name, {})
            if (receipt.get("revision") == MODEL_REVISION
                    and previous.get("identity") == identity
                    and previous.get("actual_sha256") == expected):
                item["integrity_verified"] = True
                item["verification"] = "unchanged SHA256 receipt"
            elif verify_hashes:
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while chunk := source.read(4 * MiB):
                        if cancel is not None and cancel.is_set():
                            raise NativeGlmError("CANCELLED", "Checkpoint verification cancelled")
                        digest.update(chunk)
                actual = digest.hexdigest()
                if _identity(path) != identity:
                    raise ValueError("Shard changed during SHA256 verification")
                item.update(actual_sha256=actual, integrity_verified=actual == expected,
                            verification="full SHA256")
                if actual != expected:
                    raise ValueError("SHA256 mismatch: " + name)
        except (OSError, ValueError) as exc:
            result["errors"].append(str(exc))
        result["files"].append(item)
    result["weights_present"] = all(f.get("bytes") == size for f, (_, size, _) in zip(result["files"], FILES))
    result["integrity_verified"] = all(f["integrity_verified"] for f in result["files"])
    if result["weights_present"]:
        try:
            metadata = _gguf_metadata(root / FILES[0][0])
            result["metadata"] = metadata
            expected = {"general.architecture": "glm5-next", "glm5-next.block_count": 45,
                        "glm5-next.expert_count": 288, "glm5-next.expert_used_count": 8,
                        "glm5-next.hyper_connection.count": 4, "split.count": 2}
            result["architecture_matches"] = all(metadata.get(k) == v for k, v in expected.items())
            if not result["architecture_matches"]:
                result["errors"].append("GGUF architecture does not match pinned GLM-5.3-Flash")
        except (OSError, ValueError, UnicodeError, struct.error) as exc:
            result["errors"].append(str(exc))
    return result


@dataclass(frozen=True)
class NativeLimits:
    reserve_bytes: int = 768 * MiB
    base_workspace_bytes: int = 1 * GiB
    max_rss_bytes: int = 9 * GiB
    max_swap_growth_bytes: int = 64 * MiB
    warning_swap_growth_bytes: int = 32 * MiB
    warning_headroom_bytes: int = 512 * MiB
    headroom_grace_seconds: float = 3
    load_timeout_seconds: float = 900
    request_timeout_seconds: float = 1800
    warning_grace_seconds: float = 20
    poll_seconds: float = 1
    max_pending: int = 1024

    def __post_init__(self):
        for name in ("reserve_bytes", "base_workspace_bytes", "max_rss_bytes",
                     "max_swap_growth_bytes", "warning_swap_growth_bytes", "warning_headroom_bytes", "max_pending"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Invalid limit: " + name)
        for name in ("load_timeout_seconds", "request_timeout_seconds", "poll_seconds", "warning_grace_seconds", "headroom_grace_seconds"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError("Invalid limit: " + name)


def _command_output(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=5, check=False).stdout.strip()


def resource_snapshot(pid=None):
    """OS-accounted counters, not physical NAND traffic or private memory."""
    result = {"monotonic": time.monotonic()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        result.update(available_bytes=vm.available, total_bytes=vm.total,
                      swap_used_bytes=psutil.swap_memory().used)
        if pid is not None:
            process = psutil.Process(pid)
            result.update(rss_bytes=process.memory_info().rss,
                          cpu_seconds=sum(process.cpu_times()[:2]))
    except ImportError:
        pass
    if os.uname().sysname == "Darwin":
        result["pressure_level"] = int(_command_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) or "0")
        if "available_bytes" not in result:
            vm = _command_output(["/usr/bin/vm_stat"])
            page = int(re.search(r"page size of (\d+)", vm).group(1))
            def pages(label):
                found = re.search(re.escape(label) + r":\s+(\d+)", vm)
                return int(found.group(1)) if found else 0
            result["available_bytes"] = page * sum(pages(k) for k in ("Pages free", "Pages inactive", "Pages speculative"))
            swap = _command_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
            found = re.search(r"used = ([\d.]+)([MG])", swap)
            result["swap_used_bytes"] = int(float(found[1]) * (MiB if found[2] == "M" else GiB)) if found else 0
        if pid is not None and "rss_bytes" not in result:
            result["rss_bytes"] = int(_command_output(["/bin/ps", "-p", str(pid), "-o", "rss="]) or "0") * 1024
    if "available_bytes" not in result:
        result["available_bytes"] = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return result


class _SerialQueue:
    def __init__(self, max_pending):
        self.condition = threading.Condition()
        self.waiters = deque()
        self.active = False
        self.max_pending = max_pending
        self.completed = 0

    @contextmanager
    def slot(self, cancel=None):
        ticket = object()
        with self.condition:
            if len(self.waiters) >= self.max_pending:
                raise NativeGlmError("QUEUE_FULL", "Logical inference queue is full")
            self.waiters.append(ticket)
            try:
                while self.active or self.waiters[0] is not ticket:
                    if cancel is not None and cancel.is_set():
                        raise NativeGlmError("CANCELLED", "Cancelled while waiting")
                    self.condition.wait(.1)
                if cancel is not None and cancel.is_set():
                    raise NativeGlmError("CANCELLED", "Cancelled before execution")
                self.waiters.popleft()
                self.active = True
            except BaseException:
                self.waiters.remove(ticket)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.active = False
                self.completed += 1
                self.condition.notify_all()

    def status(self):
        with self.condition:
            return {"physical_limit": 1, "active": int(self.active), "pending": len(self.waiters),
                    "max_pending": self.max_pending, "completed": self.completed}


def _metal_device_present(listing):
    # The pinned runtime advertises Apple's backend as MTL0, not the word Metal.
    return bool(re.search(r"(?m)^\s*(?:MTL|Metal)[0-9]+\s*:", listing))


class _AnyCancellation:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event is not None and event.is_set() for event in self.events)


class NativeGlmModel:
    """Share one object across logical agents. There is one native server/slot."""
    def __init__(self, model_path, *, runtime_binary, verification_file=None,
                 context_tokens=512, threads=4, batch_size=8, gpu_layers=0,
                 cpu_moe=False, moe_cache_mib=0, mmap_demand=False, expert_prefetch=False, dense_prefetch=False, mmap_nocache=False, state_dir=None, limits=None,
                 expected_runtime_sha256=None, runtime_manifest_file=None, allow_unqualified_gpu=False,
                 ubatch_size=None, threads_batch=None, dense_lookahead=False,
                 dense_lookahead_mib=32, dense_lookahead_max_cols=1,
                 dense_prefetch_max_cols=1, shared_prefix_tokens=None,
                 shared_prefix_render_options=None, shared_prefix_scratch_mib=None,
                 dense_lock_names=None, dense_lock_mib=128):
        if type(context_tokens) is not int or not 64 <= context_tokens <= 8192:
            raise ValueError("Validated integration context must be between 64 and 8192")
        if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
            raise ValueError("Invalid thread count")
        if type(batch_size) is not int or not 1 <= batch_size <= 512:
            raise ValueError("Invalid batch size")
        if ubatch_size is not None and (type(ubatch_size) is not int or not 1 <= ubatch_size <= batch_size):
            raise ValueError("Microbatch must be between 1 and batch_size")
        if threads_batch is not None and (type(threads_batch) is not int or not 1 <= threads_batch <= (os.cpu_count() or 1)):
            raise ValueError("Invalid prefill thread count")
        if type(dense_lookahead) is not bool:
            raise ValueError("dense_lookahead must be an explicit boolean")
        if type(dense_lookahead_mib) is not int or dense_lookahead_mib not in (16, 32, 64, 128):
            raise ValueError("Invalid dense lookahead window")
        for name, value in (("dense_lookahead_max_cols", dense_lookahead_max_cols),
                            ("dense_prefetch_max_cols", dense_prefetch_max_cols)):
            if type(value) is not int or value not in (1, 2, 4, 8):
                raise ValueError("Invalid dense column limit: " + name)
        if dense_lookahead_max_cols > 1 and not dense_lookahead:
            raise ValueError("Multicolumn lookahead requires dense_lookahead")
        if dense_prefetch_max_cols > 1 and not dense_prefetch:
            raise ValueError("Multicolumn prefetch requires dense_prefetch")
        if type(dense_lock_mib) is not int or dense_lock_mib not in (128, 256):
            raise ValueError("Dense retention is limited to 128 or 256 MiB")
        self.dense_lock_mib = dense_lock_mib
        self.dense_lock_names = None
        if dense_lock_names is not None:
            family = (r"(?:hc_attn_fn|hc_ffn_fn|ffn_gate_inp|ssm_beta|ssm_f_a|ssm_f_b|ssm_g_a|ssm_g_b|"
                      r"attn_k_b|attn_v_b|attn_kv_a_mqa|indexer\.attn_q_b|indexer\.attn_k|indexer\.proj|indexer_compressor_gate)\.weight")
            pattern = re.compile(r"blk\.(?:[0-9]|[1-3][0-9]|4[0-4])\." + family)
            if (not isinstance(dense_lock_names, list) or not 1 <= len(dense_lock_names) <= 128 or
                    any(not isinstance(n, str) or pattern.fullmatch(n) is None for n in dense_lock_names) or
                    len(set(dense_lock_names)) != len(dense_lock_names)):
                raise ValueError("An explicit unique list of original GLM dense weight names is required")
            self.dense_lock_names = tuple(dense_lock_names)
        if type(gpu_layers) is not int or not 0 <= gpu_layers <= 46:
            raise ValueError("Invalid GPU layer count")
        if self.dense_lock_names is not None and gpu_layers:
            raise ValueError("Dense retention research requires GPU0")
        if type(moe_cache_mib) is not int or not 0 <= moe_cache_mib <= 4096:
            raise ValueError("Invalid expert cache budget")
        if moe_cache_mib and not cpu_moe:
            raise ValueError("Expert cache requires CPU expert storage")
        if type(allow_unqualified_gpu) is not bool:
            raise ValueError("allow_unqualified_gpu must be an explicit boolean")
        self.allow_unqualified_gpu = allow_unqualified_gpu
        self.model_path = Path(model_path).expanduser().resolve()
        self.runtime_binary = Path(runtime_binary).expanduser().resolve()
        self.verification_file = verification_file
        self.context_tokens, self.threads, self.batch_size = context_tokens, threads, batch_size
        self.ubatch_size = batch_size if ubatch_size is None else ubatch_size
        self.threads_batch = threads_batch
        self.dense_lookahead = dense_lookahead
        self.dense_lookahead_mib = dense_lookahead_mib
        self.dense_lookahead_max_cols = dense_lookahead_max_cols
        self.dense_prefetch_max_cols = dense_prefetch_max_cols
        self.shared_prefix_tokens = None
        self.shared_prefix_render_options = {}
        self.shared_prefix_scratch_mib = shared_prefix_scratch_mib
        self._prefix_file = None
        self._prefix_key = None
        if shared_prefix_tokens is not None:
            if (not isinstance(shared_prefix_tokens, list) or not shared_prefix_tokens or
                    len(shared_prefix_tokens) >= context_tokens or len(shared_prefix_tokens) % self.ubatch_size or
                    any(type(t) is not int or not 0 <= t < 154880 for t in shared_prefix_tokens)):
                raise ValueError("An aligned native GLM prefix with a private suffix is required")
            if gpu_layers or threads != 10 or batch_size != 8 or self.ubatch_size != 8 or cpu_moe or moe_cache_mib:
                raise ValueError("Shared prefix research is limited to CPU10/GPU0 and batch/ubatch8")
            if threads_batch is not None and threads_batch != 10:
                raise ValueError("Shared prefix research requires prefill threads10")
            if type(shared_prefix_scratch_mib) is not int or not 1 <= shared_prefix_scratch_mib <= 64:
                raise ValueError("Shared prefix requires an explicit serialization scratch policy budget")
            rendering = {} if shared_prefix_render_options is None else shared_prefix_render_options
            if (not isinstance(rendering, dict) or set(rendering) - {"reasoning_effort"} or
                    ("reasoning_effort" in rendering and rendering["reasoning_effort"] not in ("low", "high", "max"))):
                raise ValueError("Unsupported shared prefix template options")
            self.shared_prefix_tokens = tuple(shared_prefix_tokens)
            self.shared_prefix_render_options = dict(rendering)
            self.threads_batch = threads if threads_batch is None else threads_batch
        elif shared_prefix_render_options is not None or shared_prefix_scratch_mib is not None:
            raise ValueError("Shared prefix options require native prefix tokens")
        if type(mmap_demand) is not bool:
            raise ValueError("mmap_demand must be explicit boolean")
        if type(mmap_nocache) is not bool:
            raise ValueError("mmap_nocache must be explicit boolean")
        self.gpu_layers, self.cpu_moe, self.moe_cache_mib = gpu_layers, cpu_moe, moe_cache_mib
        if type(expert_prefetch) is not bool:
            raise ValueError("expert_prefetch must be explicit boolean")
        self.mmap_demand = mmap_demand
        self.mmap_nocache = mmap_nocache
        self.expert_prefetch = expert_prefetch
        if type(dense_prefetch) is not bool:
            raise ValueError("dense_prefetch must be explicit boolean")
        self.dense_prefetch = dense_prefetch
        self.limits = limits or NativeLimits()
        if self.shared_prefix_tokens is not None and (
                self.limits.max_rss_bytes > 17 * GiB // 2 or self.limits.max_swap_growth_bytes > 32 * MiB):
            raise ValueError("Shared prefix research requires RSS<=8.5GiB and swap growth<=32MiB")
        self.state_dir = Path(state_dir or (Path(tempfile.gettempdir()) / ("airllm-glm-" + str(os.getuid()))))
        self.expected_runtime_sha256 = expected_runtime_sha256
        self.runtime_manifest_file = Path(runtime_manifest_file).expanduser().resolve() if runtime_manifest_file else None
        self._runtime_artifacts_verified = False
        self._runtime_artifact_identities = {}
        self._runtime_manifest_info = None
        self.scheduler = _SerialQueue(self.limits.max_pending)
        self._load_lock = threading.RLock()
        self._process = self._lock_file = self._log_file = None
        self._exit_observation_lock = threading.Lock()
        self._owned_process_generation = 0
        self._last_owned_process_exit = None
        self._watch_stop = threading.Event()
        self._load_stop = threading.Event()
        self._api_key = None
        self._key_file = None
        self._peak_rss_bytes = 0
        self._resource_baseline = None
        self._warning_started = None
        self._headroom_low_started = None
        self._pressure_context = None
        self._failure = None
        self._checkpoint = None
        self._port = None
        self._runtime_sha256 = None
        self._samples = deque(maxlen=3600)
        self.last_request = None
        self._counters = {"loads": 0, "requests": 0, "generated_tokens": 0}

    def status(self):
        loaded = self._process is not None and self._process.poll() is None
        return {"ready": loaded and self._failure is None and (not self.gpu_layers or self.allow_unqualified_gpu),
                "loaded": loaded, "numerical_qualification": self._numerical_qualification(),
                "backend": "next-glm-native", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "expected_runtime_revision": RUNTIME_REVISION, "runtime_sha256": self._runtime_sha256,
                "runtime_version": getattr(self, "_runtime_version", None),
                "runtime_artifacts_verified": self._runtime_artifacts_verified,
                "runtime_artifact_scope": ("executable_and_all_packaged_dylibs_verified_against_local_build_manifest"
                    if self._runtime_artifacts_verified else "dependencies_unqualified_without_validated_manifest"),
                "runtime_manifest": self._runtime_manifest_info,
                "checkpoint": self._checkpoint or inspect_native_checkpoint(self.model_path, self.verification_file),
                "scheduler": self.scheduler.status(), "counters": dict(self._counters),
                "configuration": {"context_tokens": self.context_tokens, "threads": self.threads,
                                  "batch_size": self.batch_size, "gpu_layers": self.gpu_layers,
                                  "ubatch_size": self.ubatch_size, "threads_batch": self.threads_batch,
                                  "dense_lookahead": self.dense_lookahead,
                                  "dense_lookahead_mib": self.dense_lookahead_mib,
                                  "dense_lookahead_max_cols": self.dense_lookahead_max_cols,
                                  "dense_prefetch_max_cols": self.dense_prefetch_max_cols,
                                  "dense_lock": {"requested": self.dense_lock_names is not None,
                                      "names": list(self.dense_lock_names or ()), "budget_mib": self.dense_lock_mib},
                                  "shared_prefix": {"requested": self.shared_prefix_tokens is not None,
                                      "tokens": len(self.shared_prefix_tokens or ()), "key_sha256": self._prefix_key,
                                      "numerically_qualified": False},
                                  "cpu_moe": self.cpu_moe, "moe_cache_mib": self.moe_cache_mib,
                                  "mmap_demand": self.mmap_demand,
                                  "mmap_nocache": self.mmap_nocache,
                                  "expert_prefetch": self.expert_prefetch,
                                  "dense_prefetch": self.dense_prefetch,
                                  "allow_unqualified_gpu": self.allow_unqualified_gpu},
                "failure": self._failure, "last_request": self.last_request,
                "last_owned_process_exit": self.last_owned_process_exit,
                "pressure_context": self._pressure_context,
                "last_resource_sample": self._samples[-1] if self._samples else None,
                "peak_sampled_rss_bytes": self._peak_rss_bytes}

    def _numerical_qualification(self):
        return {"selected_path": "experimental_gpu" if self.gpu_layers else "cpu",
                "gpu_qualified": False, "allow_unqualified_gpu": self.allow_unqualified_gpu,
                "global_numerical_equivalence_validated": False,
                "cpu_evidence_scope": {"generated_tokens": 4, "full_logits_compared": 619520,
                                       "ordered_route_elements_compared": 3024,
                                       "context_tokens": 512, "batch_size": 8,
                                       "scope": "recorded targeted CPU comparison only; no guarantee for all configurations or contexts"},
                "gpu_evidence_scope": {"gpu_layers": 6, "cpu_moe": True, "moe_cache_mib": 256,
                                       "forced_identical_prefixes": True, "generated_tokens": 4,
                                       "max_abs_logit_difference": 0.5348455905914307,
                                       "ordered_route_element_mismatches": 364,
                                       "comparison_passed": False,
                                       "scope": "observed hybrid6 deviation from CPU reference; other GPU configurations are unqualified"}}

    def _model_lock_key(self):
        # One UID-wide engine for this pinned target, including distinct hard-link aliases.
        target = MODEL_ID + "@" + MODEL_REVISION + "@Q2_K_S"
        return hashlib.sha256(target.encode()).hexdigest()[:24]

    def _gpu_weight_bytes(self):
        # Pinned native placement counts output as one GPU layer (MTP layer count is zero).
        # ngl=1 is output/norm only; ngl=N adds the last N-1 transformer blocks.
        blocks = [313459712 if i < 3 else 193376640 if (i - 3) % 4 == 0 else 184486016
                  for i in range(45)]
        n = min(max(self.gpu_layers - 1, 0), 45)
        return (sum(blocks[-n:]) if n else 0) + (674054144 if self.gpu_layers else 0)

    def _admit(self):
        if self.gpu_layers and not self.cpu_moe:
            raise NativeGlmError("GPU_ADMISSION", "This 110 GB target requires CPU-mapped experts for Metal offload")
        sample = resource_snapshot()
        loading = self._process is None or self._process.poll() is not None
        if not loading:
            self._check_running()
        workspace = self.limits.base_workspace_bytes + self.context_tokens * 64 * 1024 + self.batch_size * 4 * MiB
        resident_gpu = self._gpu_weight_bytes() + self.moe_cache_mib * MiB if loading else 0
        retained_dense = self.dense_lock_mib * MiB if loading and self.dense_lock_names is not None else 0
        required = self.limits.reserve_bytes + workspace + resident_gpu + retained_dense
        if loading and self.shared_prefix_tokens is not None:
            required = max(required, 11 * GiB // 2)
        pressure = sample.get("pressure_level", 0)
        baseline = self._resource_baseline or sample
        swap_growth = max(0, sample.get("swap_used_bytes", 0) - baseline.get("swap_used_bytes", 0))
        pressure_blocked = pressure >= 4 or (loading and pressure >= 2)
        swap_blocked = (not loading and (swap_growth > self.limits.max_swap_growth_bytes
                        or (pressure >= 2 and swap_growth > self.limits.warning_swap_growth_bytes)))
        if pressure_blocked or swap_blocked or sample["available_bytes"] < required:
            raise NativeGlmError("MEMORY_ADMISSION", "Insufficient safe mmap workspace", {
                "sample": sample, "required_bytes": required, "workspace_estimate_bytes": workspace,
                "gpu_weight_and_cache_bytes": resident_gpu, "loading": loading,
                "retained_dense_reservation_bytes": retained_dense,
                "swap_growth_bytes": swap_growth, "pressure_blocked": pressure_blocked,
                "swap_blocked": swap_blocked,
                "reason": "cold startup rejects warning pressure; warm requests require workspace headroom and bounded swap"})
        return sample

    def _build_command(self, port):
        args = [str(self.runtime_binary), "--offline", "--fit", "off", "--model",
                str(self.model_path / FILES[0][0]), "--load-mode", "mmap", "--no-repack",
                "--no-warmup", "--ctx-size", str(self.context_tokens), "--batch-size", str(self.batch_size),
                "--ubatch-size", str(self.ubatch_size), "--threads", str(self.threads),
                "--gpu-layers", str(self.gpu_layers), "--parallel", "1", "--host", "127.0.0.1",
                "--port", str(port), "--no-webui", "--no-context-shift", "--sse-ping-interval", "1",
                "--log-colors", "off", "--perf", "--cors-origins", "localhost", "--no-cors-credentials"]
        if self._key_file is not None:
            args.extend(["--api-key-file", str(self._key_file)])
        if self.threads_batch is not None:
            args.extend(["--threads-batch", str(self.threads_batch)])
        if self.shared_prefix_tokens is not None:
            args.extend(["--cache-ram", "0", "--ctx-checkpoints", "0"])
        if self.gpu_layers == 0:
            args.extend(["--device", "none", "--no-op-offload"])
        if self.cpu_moe:
            args.append("--cpu-moe")
        if self.moe_cache_mib:
            args.extend(["--moe-cache-mib", str(self.moe_cache_mib)])
        return args

    def _runtime_environment(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("LLAMA_ARG_", "AIRLLM_", "DYLD_")) and k != "LLAMA_API_KEY"}
        if self.mmap_demand:
            env["AIRLLM_MMAP_DEMAND"] = "1"
        if self.mmap_nocache:
            env["AIRLLM_MMAP_NOCACHE"] = "1"
        if self.expert_prefetch:
            env["AIRLLM_EXPERT_PREFETCH"] = "1"
        if self.dense_prefetch:
            env["AIRLLM_DENSE_PREFETCH"] = "1"
            env["AIRLLM_DENSE_PREFETCH_MAX_COLS"] = str(self.dense_prefetch_max_cols)
        if self.dense_lookahead:
            env["AIRLLM_DENSE_LOOKAHEAD"] = "1"
            env["AIRLLM_DENSE_LOOKAHEAD_MIB"] = str(self.dense_lookahead_mib)
            env["AIRLLM_DENSE_LOOKAHEAD_MAX_COLS"] = str(self.dense_lookahead_max_cols)
        if self._prefix_file is not None:
            env["AIRLLM_SHARED_PREFIX_MANIFEST"] = str(self._prefix_file)
        if self.dense_lock_names is not None:
            env["AIRLLM_DENSE_LOCK"] = "1"
            env["AIRLLM_DENSE_LOCK_MIB"] = str(self.dense_lock_mib)
            env["AIRLLM_DENSE_LOCK_NAMES"] = ",".join(self.dense_lock_names)
        return env

    def _check_requested_features(self):
        if not (self.dense_lookahead or self.dense_prefetch_max_cols > 1 or self.shared_prefix_tokens is not None or self.dense_lock_names is not None):
            return
        if not self._runtime_artifacts_verified or not self._runtime_manifest_info:
            raise NativeGlmError("RUNTIME_FEATURE_UNVERIFIED", "Dense experiments require a verified runtime manifest")
        patches = set(self._runtime_manifest_info.get("patch_names", []))
        required = set()
        if self.dense_lookahead:
            required.add("llama-dense-lookahead.patch")
        if self.dense_prefetch_max_cols > 1:
            required.add("llama-dense-prefetch-columns.patch")
        if self.dense_lock_names is not None:
            required.add("llama-dense-lock.patch")
        if not required <= patches:
            raise NativeGlmError("RUNTIME_FEATURE_UNAVAILABLE", "Selected runtime lacks the requested dense patches")
        features = self._runtime_manifest_info.get("airllm_runtime_features", [])
        if self.dense_lookahead_max_cols > 1 and "dense_lookahead_multicolumn_v1" not in features:
            raise NativeGlmError("RUNTIME_FEATURE_UNAVAILABLE", "Multicolumn lookahead requires a newly built runtime")
        if self.shared_prefix_tokens is not None and "shared_prefix_snapshot_v1" not in features:
            raise NativeGlmError("RUNTIME_FEATURE_UNAVAILABLE", "Shared prefix requires a newly built research runtime")

    def _prepare_prefix_manifest(self):
        if self.shared_prefix_tokens is None:
            return
        dependencies = {"model_revision": MODEL_REVISION,
            "shards": [{"sha256": digest, "bytes": size} for _, size, digest in FILES],
            "tokenizer_and_template_identity": "original pinned GGUF shards",
            "runtime_revision": RUNTIME_REVISION, "runtime_sha256": self._runtime_sha256,
            "runtime_artifacts": self._runtime_manifest_info["artifact_sha256"],
            "context": self.context_tokens, "batch": self.batch_size, "ubatch": self.ubatch_size,
            "threads": self.threads, "threads_batch": self.threads_batch,
            "gpu_layers": 0, "prefix_tokens": self.shared_prefix_tokens,
            "prefix_positions": list(range(len(self.shared_prefix_tokens))),
            "render_options": self.shared_prefix_render_options}
        canonical = json.dumps(dependencies, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._prefix_key = hashlib.sha256(canonical).hexdigest()
        value = {"schema": 1, "key_sha256": self._prefix_key,
                 "prefix_tokens": self.shared_prefix_tokens,
                 "scratch_bytes": self.shared_prefix_scratch_mib * MiB}
        self._prefix_file = self.state_dir / (".native-prefix-" + secrets.token_hex(8) + ".json")
        fd = os.open(self._prefix_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)

    def _verify_runtime_manifest(self, cancel=None):
        self._runtime_artifacts_verified = False
        self._runtime_artifact_identities = {}
        self._runtime_manifest_info = None
        if self.runtime_manifest_file is None:
            return
        manifest_path = self.runtime_manifest_file
        try:
            before_manifest = _identity(manifest_path)
            if before_manifest[2] > 2 * MiB:
                raise ValueError("Runtime manifest exceeds size limit")
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("Runtime manifest must be an object")
            features = manifest.get("airllm_runtime_features", [])
            if not isinstance(features, list) or any(not isinstance(f, str) for f in features):
                raise ValueError("Runtime feature list must contain strings")
            if manifest.get("status") != "built" or manifest.get("source_revision") != RUNTIME_REVISION:
                raise ValueError("Runtime manifest is not a built package of the pinned source revision")
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 512:
                raise ValueError("Invalid runtime artifact list")
            root = manifest_path.parent
            bin_dir = root / "bin"
            if bin_dir.is_symlink() or not bin_dir.is_dir():
                raise ValueError("Runtime bin must be an ordinary package directory")
            names, rows, total = set(), [], 0
            simple_name = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
            for row in artifacts:
                if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                    raise ValueError("Invalid runtime artifact record")
                parts = row["path"].split("/")
                if len(parts) != 2 or parts[0] != "bin" or not simple_name.fullmatch(parts[1]):
                    raise ValueError("Runtime artifact paths must be simple bin/ filenames")
                name = parts[1]
                if name in names:
                    raise ValueError("Duplicate runtime artifact path")
                names.add(name)
                path = bin_dir / name
                if "symlink" in row:
                    target = row["symlink"]
                    if not isinstance(target, str) or not simple_name.fullmatch(target):
                        raise ValueError("Runtime symlinks must target a simple filename in bin")
                    if not path.is_symlink() or os.readlink(path) != target:
                        raise ValueError("Runtime symlink does not match manifest")
                    if "sha256" in row:
                        raise ValueError("Runtime symlink record must not masquerade as a file")
                else:
                    size, expected = row.get("bytes"), row.get("sha256")
                    if (type(size) is not int or not 0 <= size <= 512 * MiB
                            or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                        raise ValueError("Invalid runtime file size or SHA256")
                    total += size
                    if total > 2 * GiB:
                        raise ValueError("Runtime package exceeds verification limit")
                    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                        raise ValueError("Runtime artifact size/type does not match manifest")
                rows.append((row, path))
            if self.runtime_binary.parent != bin_dir or self.runtime_binary.name not in names:
                raise ValueError("Selected runtime executable is not covered by the manifest")
            actual_libraries = {p.name for p in bin_dir.iterdir() if p.name.endswith(".dylib")}
            if not actual_libraries.issubset(names):
                raise ValueError("Packaged native libraries are missing from runtime manifest")
            identities = {}
            for row, path in rows:
                if cancel is not None and cancel.is_set():
                    raise NativeGlmError("CANCELLED", "Runtime verification cancelled")
                before = _link_identity(path)
                if "symlink" in row:
                    if row["symlink"] not in names or path.resolve(strict=True).parent != bin_dir:
                        raise ValueError("Runtime symlink target is outside the declared bin artifacts")
                elif _sha256_file(path, cancel) != row["sha256"]:
                    raise ValueError("Runtime artifact SHA256 mismatch: " + path.name)
                if _link_identity(path) != before:
                    raise ValueError("Runtime artifact changed during verification")
                identities[path] = before
            if _identity(manifest_path) != before_manifest:
                raise ValueError("Runtime manifest changed during verification")
            identities[manifest_path] = _link_identity(manifest_path)
            self._runtime_artifact_identities = identities
            self._runtime_manifest_info = {"path": str(manifest_path), "source_revision": RUNTIME_REVISION,
                "artifact_count": len(rows), "verified_bytes": total,
                "artifact_sha256": {row["path"]: row["sha256"] for row, _ in rows if "sha256" in row},
                "patch_names": [p["name"] for p in manifest.get("patches", [])
                                if isinstance(p, dict) and isinstance(p.get("name"), str)],
                "airllm_runtime_features": manifest.get("airllm_runtime_features", []),
                "experiment_defaults": manifest.get("experiment_defaults", {}),
                "scope": "local package integrity; source/build claims are those of the supplied manifest"}
            self._runtime_artifacts_verified = True
        except NativeGlmError:
            raise
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            raise NativeGlmError("RUNTIME_MANIFEST", str(error)) from error

    def _check_runtime_identity(self):
        if not self._runtime_artifacts_verified:
            return
        try:
            unchanged = all(_link_identity(path) == identity
                            for path, identity in self._runtime_artifact_identities.items())
        except OSError:
            unchanged = False
        if not unchanged:
            self._runtime_artifacts_verified = False
            self.close()
            raise NativeGlmError("RUNTIME_ARTIFACT_CHANGED", "Native executable, library or manifest changed; engine discarded")

    def load(self, cancel=None):
        if self.gpu_layers and not self.allow_unqualified_gpu:
            raise NativeGlmError("GPU_FIDELITY_UNQUALIFIED",
                "GPU numerical fidelity is unqualified; explicit experimental opt-in is required",
                self._numerical_qualification())
        with self._load_lock:
            self._load_stop.clear()
            cancel = _AnyCancellation(cancel, self._load_stop)
            if cancel.is_set():
                raise NativeGlmError("CANCELLED", "Loading cancelled before startup")
            if self.status()["ready"]:
                return self.status()
            self._close_locked()
            self._failure = None
            self._checkpoint = inspect_native_checkpoint(self.model_path, self.verification_file, verify_hashes=True, cancel=cancel)
            if (self._checkpoint["errors"] or not self._checkpoint["architecture_matches"]
                    or not self._checkpoint["integrity_verified"]):
                raise NativeGlmError("MODEL_INTEGRITY", "Pinned GLM checkpoint verification failed", self._checkpoint)
            if not self.runtime_binary.is_file() or not os.access(self.runtime_binary, os.X_OK):
                raise NativeGlmError("RUNTIME_MISSING", "Native llama-server binary is missing")
            self._verify_runtime_manifest(cancel)
            self._check_requested_features()
            self._runtime_sha256 = _sha256_file(self.runtime_binary, cancel)
            if self.expected_runtime_sha256 and self._runtime_sha256 != self.expected_runtime_sha256:
                raise NativeGlmError("RUNTIME_INTEGRITY", "Native runtime hash mismatch")
            version = subprocess.run([str(self.runtime_binary), "--version"], capture_output=True,
                                     text=True, timeout=15, check=True)
            self._runtime_version = version.stdout + version.stderr
            if RUNTIME_REVISION[:7] not in self._runtime_version:
                raise NativeGlmError("RUNTIME_REVISION", "Native runtime is not the qualified source revision")
            if self.gpu_layers:
                devices = subprocess.run([str(self.runtime_binary), "--list-devices"], capture_output=True,
                                         text=True, timeout=60, check=True)
                if not _metal_device_present(devices.stdout + devices.stderr):
                    raise NativeGlmError("GPU_UNAVAILABLE", "Requested Metal path is unavailable in native runtime")
            baseline = self._admit()
            self._resource_baseline = baseline
            self._warning_started = self._headroom_low_started = None
            self._pressure_context = None
            self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            # The model lock is global for the UID, even if callers choose different state directories.
            lock_root = Path(tempfile.gettempdir()) / ("airllm-native-locks-" + str(os.getuid()))
            lock_root.mkdir(mode=0o700, exist_ok=True)
            key = self._model_lock_key()
            self._lock_file = (lock_root / (key + ".lock")).open("a")
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock_file.close()
                self._lock_file = None
                raise NativeGlmError("ENGINE_BUSY", "Another native engine already owns this model") from exc
            started = time.monotonic()
            try:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    self._port = sock.getsockname()[1]
                self._api_key = secrets.token_urlsafe(32)
                self._key_file = self.state_dir / (".native-api-key-" + secrets.token_hex(8))
                fd = os.open(self._key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as key_stream:
                    key_stream.write(self._api_key + "\n")
                self._prepare_prefix_manifest()
                self._log_file = (self.state_dir / "native-server.log").open("ab")
                env = self._runtime_environment()
                self._check_runtime_identity()
                process = subprocess.Popen(self._build_command(self._port), stdin=subprocess.DEVNULL,
                    stdout=self._log_file, stderr=self._log_file, env=env, start_new_session=True,
                    # A surviving native child retains model ownership even if Python is SIGKILLed.
                    pass_fds=(self._lock_file.fileno(),))
                with self._exit_observation_lock:
                    self._process = process
                    self._owned_process_generation += 1
                    self._last_owned_process_exit = None
                self._watch_stop.clear()
                self._watcher = threading.Thread(target=self._watch, args=(baseline, self._owned_process_generation), daemon=True)
                self._watcher.start()
                while time.monotonic() - started < self.limits.load_timeout_seconds:
                    if cancel is not None and cancel.is_set():
                        raise NativeGlmError("CANCELLED", "Loading cancelled; native state discarded")
                    self._check_running()
                    try:
                        if self._http_json("GET", "/health", timeout=1).get("status") == "ok":
                            self._counters["loads"] += 1
                            self.load_seconds = time.monotonic() - started
                            return self.status()
                    except (OSError, http.client.HTTPException, NativeGlmError):
                        pass
                    time.sleep(.2)
                raise NativeGlmError("LOAD_TIMEOUT", "Native model startup exceeded time limit")
            except BaseException:
                self.close()
                raise

    def _check_identity(self):
        try:
            unchanged = all(_identity(self.model_path / item["name"]) == item["identity"]
                            for item in self._checkpoint["files"])
        except OSError:
            unchanged = False
        if not unchanged:
            self.close()
            raise NativeGlmError("MODEL_CHANGED", "Mapped checkpoint files changed; engine discarded")

    def _check_running(self):
        self._check_runtime_identity()
        if self._failure:
            raise NativeGlmError(self._failure["code"], self._failure["message"], self._failure)
        if self._process is None or self._process.poll() is not None:
            raise NativeGlmError("RUNTIME_EXITED", "Native engine exited; inspect native-server.log")

    def _guard_resource_sample(self, sample, baseline, now):
        """Pressure warning is contextual; critical pressure and hard caps are immediate."""
        pressure = sample.get("pressure_level", 0)
        available = sample["available_bytes"]
        swap_growth = max(0, sample.get("swap_used_bytes", 0) - baseline.get("swap_used_bytes", 0))
        if pressure >= 2:
            if self._warning_started is None:
                self._warning_started = now
        else:
            self._warning_started = None
        if available < self.limits.reserve_bytes:
            if self._headroom_low_started is None:
                self._headroom_low_started = now
        else:
            self._headroom_low_started = None
        warning_seconds = now - self._warning_started if self._warning_started is not None else 0
        low_seconds = now - self._headroom_low_started if self._headroom_low_started is not None else 0
        warning_headroom_low = available < self.limits.reserve_bytes + self.limits.warning_headroom_bytes
        warning_swap_high = swap_growth > self.limits.warning_swap_growth_bytes
        code = None
        if sample.get("rss_bytes", 0) > self.limits.max_rss_bytes:
            code = "RSS_LIMIT"
        if swap_growth > self.limits.max_swap_growth_bytes:
            code = "SWAP_LIMIT"
        if pressure >= 4:
            code = "MEMORY_PRESSURE_CRITICAL"
        if self._headroom_low_started is not None and low_seconds >= self.limits.headroom_grace_seconds:
            code = code or "MEMORY_HEADROOM_LIMIT"
        if pressure >= 2 and warning_seconds >= self.limits.warning_grace_seconds:
            if warning_headroom_low:
                code = code or "MEMORY_PRESSURE_HEADROOM"
            elif warning_swap_high:
                code = code or "MEMORY_PRESSURE_SWAP"
        self._pressure_context = {
            "pressure_level": pressure, "warning_seconds": warning_seconds,
            "available_bytes": available, "reserve_bytes": self.limits.reserve_bytes,
            "warning_headroom_floor_bytes": self.limits.reserve_bytes + self.limits.warning_headroom_bytes,
            "swap_growth_bytes": swap_growth, "warning_swap_growth_bytes": self.limits.warning_swap_growth_bytes,
            "warning_headroom_low": warning_headroom_low, "warning_swap_high": warning_swap_high,
            "headroom_low_seconds": low_seconds,
            "action": "stop" if code else "continue_under_monitoring", "stop_code": code,
            "evidence_scope": "M5 16 GiB, same pinned GLM, CPU4/context512/batch8: stable mmap warning pressure observed; other workloads unqualified"}
        return code

    def _watch_owns_process_locked(self, process, owned_generation):
        """Caller holds _exit_observation_lock; PID equality is insufficient."""
        return (process is not None and process is self._process
                and owned_generation == self._owned_process_generation)

    def _watch(self, baseline, owned_generation=None):
        if owned_generation is None:
            with self._exit_observation_lock:
                owned_generation = self._owned_process_generation
        while not self._watch_stop.wait(self.limits.poll_seconds):
            with self._exit_observation_lock:
                process = self._process
                if not self._watch_owns_process_locked(process, owned_generation):
                    return
            if process.poll() is not None:
                return
            termination_source = None
            try:
                with self._exit_observation_lock:
                    if not self._watch_owns_process_locked(process, owned_generation):
                        return
                sample = resource_snapshot(process.pid)
                # Sampling can yield while close/reload changes the owner. Keep
                # every guard-state mutation and its resulting termination in
                # the same owner lock as process/generation publication.
                with self._exit_observation_lock:
                    if not self._watch_owns_process_locked(process, owned_generation):
                        return
                    self._samples.append(sample)
                    self._peak_rss_bytes = max(self._peak_rss_bytes, sample.get("rss_bytes", 0))
                    code = self._guard_resource_sample(sample, baseline, time.monotonic())
                    if not self._watch_owns_process_locked(process, owned_generation):
                        return
                    if code:
                        self._failure = {"code": code, "message": "Native inference stopped by resource guard",
                                         "sample": sample, "pressure_context": self._pressure_context}
                        termination_source = "resource_guard"
                        if not self._watch_owns_process_locked(process, owned_generation):
                            return
                        self._terminate_process(process)
                if code:
                    return
            except Exception as exc:
                if self._watch_stop.is_set():
                    return
                with self._exit_observation_lock:
                    if not self._watch_owns_process_locked(process, owned_generation):
                        return
                    self._failure = {"code": "WATCHDOG_ERROR", "message": str(exc)}
                    termination_source = "watchdog_error"
                    if not self._watch_owns_process_locked(process, owned_generation):
                        return
                    self._terminate_process(process)
                return
            finally:
                # This observer takes the same lock internally; invoke it only
                # after leaving the guard/termination critical section.
                if termination_source is not None:
                    self._observe_owned_process_exit(process, termination_source, owned_generation)

    @property
    def last_owned_process_exit(self):
        """A detached observation, never proof based only on a cleared handle."""
        with self._exit_observation_lock:
            return dict(self._last_owned_process_exit) if self._last_owned_process_exit is not None else None

    def _observe_owned_process_exit(self, process, source, owned_generation=None):
        """Retain poll evidence before dropping an owned Popen handle.

        This adds no signal, wait or retry. Observation failures remain explicit
        and cannot turn an unknown/live process into a confirmed exit.
        """
        if process is None:
            return
        generation = self._owned_process_generation if owned_generation is None else owned_generation
        if generation != self._owned_process_generation:
            return  # A late old watcher must not overwrite a newer process record.
        try:
            pid = process.pid
        except Exception:
            pid = None
        observation = {"pid": pid, "generation": generation, "returncode": None,
                       "exit_observed": False, "observation": "popen_poll_of_owned_process",
                       "source": source, "observed_monotonic": time.monotonic(),
                       "observed_unix_seconds": time.time(),
                       "guard_failure_code": self._failure.get("code") if isinstance(self._failure, dict) else None}
        try:
            returncode = process.poll()
            if type(returncode) is int:
                observation.update(returncode=returncode, exit_observed=True)
            elif returncode is not None:
                observation["poll_error"] = "INVALID_RETURN_CODE"
        except Exception as error:
            observation["poll_error"] = type(error).__name__
        with self._exit_observation_lock:
            if generation == self._owned_process_generation:
                self._last_owned_process_exit = observation

    @staticmethod
    def _terminate_process(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def close(self):
        self._load_stop.set()
        return self._close_locked()

    def _close_locked(self):
        with self._load_lock:
            self._watch_stop.set()
            with self._exit_observation_lock:
                process = self._process
                owned_generation = self._owned_process_generation
            try:
                self._terminate_process(process)
            finally:
                self._observe_owned_process_exit(process, "close_after_termination_attempt", owned_generation)
            with self._exit_observation_lock:
                if self._process is process and self._owned_process_generation == owned_generation:
                    self._process = None
            watcher = getattr(self, "_watcher", None)
            if watcher and watcher is not threading.current_thread():
                watcher.join(timeout=6)
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            self._port = None
            self._api_key = None
            if self._key_file is not None:
                self._key_file.unlink(missing_ok=True)
                self._key_file = None
            if self._prefix_file is not None:
                self._prefix_file.unlink(missing_ok=True)
                self._prefix_file = None
            self._prefix_key = None
        return {"loaded": False}

    unload = close

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        return headers

    def _http_json(self, method, path, payload=None, timeout=30):
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        try:
            encoded = json.dumps(payload).encode() if payload is not None else None
            connection.request(method, path, body=encoded, headers=self._headers())
            response = connection.getresponse()
            raw = response.read(16 * MiB + 1)
            if len(raw) > 16 * MiB:
                raise NativeGlmError("RESPONSE_LIMIT", "Native metadata response exceeds limit")
            value = json.loads(raw)
            if response.status != 200:
                raise NativeGlmError("HTTP_ERROR", str(value), {"status": response.status})
            return value
        finally:
            connection.close()

    def _events(self, payload, cancel):
        self.last_request = None
        events = queue.Queue(maxsize=64)
        connection = http.client.HTTPConnection("127.0.0.1", self._port,
                                                timeout=self.limits.request_timeout_seconds)
        done = threading.Event()

        def publish(item):
            while not done.is_set():
                try:
                    events.put(item, timeout=.1)
                    return
                except queue.Full:
                    pass

        def read_stream():
            try:
                connection.request("POST", "/completion", body=json.dumps(payload).encode(),
                                   headers=self._headers())
                response = connection.getresponse()
                if response.status != 200:
                    raise NativeGlmError("HTTP_ERROR", response.read(64 * 1024).decode(errors="replace"))
                data = []
                while not done.is_set():
                    line = response.readline(16 * MiB + 1)
                    if len(line) > 16 * MiB:
                        raise NativeGlmError("RESPONSE_LIMIT", "Native SSE event exceeds limit")
                    if not line:
                        break
                    line = line.decode("utf-8").rstrip("\r\n")
                    if not line and data:
                        raw = "\n".join(data)
                        data = []
                        if raw != "[DONE]":
                            publish(("data", json.loads(raw)))
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip())
                publish(("end", None))
            except Exception as exc:
                publish(("error", exc))
            finally:
                connection.close()

        worker = threading.Thread(target=read_stream, daemon=True)
        worker.start()
        started, ttft, token_count, final = time.monotonic(), None, 0, None
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise NativeGlmError("CANCELLED", "Generation cancelled; native state discarded")
                self._check_running()
                if time.monotonic() - started > self.limits.request_timeout_seconds:
                    raise NativeGlmError("REQUEST_TIMEOUT", "Native generation exceeded time limit")
                try:
                    kind, event = events.get(timeout=.1)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise event
                if kind == "end":
                    if final is None:
                        raise NativeGlmError("INCOMPLETE_STREAM", "Native stream ended without terminal event")
                    break
                if "error" in event:
                    raise NativeGlmError("NATIVE_ERROR", str(event["error"]))
                if ttft is None and (event.get("tokens") or event.get("content")):
                    ttft = time.monotonic() - started
                token_count += len(event.get("tokens", []))
                if event.get("stop"):
                    final = event
                yield event
            self._check_identity()
            self.last_request = {"seconds": time.monotonic() - started, "ttft_seconds": ttft,
                                 "streamed_token_count": token_count, "native": final}
            self._counters["requests"] += 1
            self._counters["generated_tokens"] += token_count
        finally:
            done.set()
            try:
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            # Stop native work immediately on cancellation or an abandoned generator.
            if final is None:
                self.close()
            worker.join(timeout=2)
            if worker.is_alive():
                self.close()

    def _payload(self, prompt, max_new_tokens, cache_prompt, n_probs):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < self.context_tokens:
            raise ValueError("Token budget must fit configured context")
        if type(n_probs) is not int or not 0 <= n_probs <= 128:
            raise ValueError("Invalid probability count")
        tokens = self._http_json("POST", "/tokenize", {"content": prompt, "add_special": True})["tokens"]
        if len(tokens) + max_new_tokens > self.context_tokens:
            raise NativeGlmError("CONTEXT_LIMIT", "Prompt and generation exceed configured context",
                                 {"prompt_tokens": len(tokens), "context_tokens": self.context_tokens})
        return {"prompt": prompt, "n_predict": max_new_tokens, "stream": True,
                "temperature": 0, "seed": 42, "cache_prompt": bool(cache_prompt), "id_slot": 0,
                "return_tokens": True, "n_probs": n_probs, "post_sampling_probs": False,
                "timings_per_token": True, "sse_ping_interval": 1}

    def stream_completion(self, prompt, max_new_tokens=128, *, cancel=None, cache_prompt=False, n_probs=0):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("A nonempty text prompt is required")
        with self.scheduler.slot(cancel):
            self.load(cancel=cancel)
            self._admit()
            self._check_identity()
            payload = self._payload(prompt, max_new_tokens, False if self.shared_prefix_tokens is not None else cache_prompt, n_probs)
            yield from self._events(payload, cancel)

    def stream_chat(self, messages, max_new_tokens=128, *, cancel=None, cache_prompt=False,
                    n_probs=0, chat_template_kwargs=None):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Nonempty chat messages are required")
        for message in messages:
            if (not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)):
                raise ValueError("Only text system/user/assistant messages are supported")
        requested = time.monotonic()
        with self.scheduler.slot(cancel):
            acquired = time.monotonic()
            self.load(cancel=cancel)
            loaded = time.monotonic()
            self._admit()
            self._check_identity()
            admitted = time.monotonic()
            body = {"messages": messages}
            if chat_template_kwargs is not None:
                if not isinstance(chat_template_kwargs, dict):
                    raise ValueError("Invalid chat template options")
                body["chat_template_kwargs"] = chat_template_kwargs
            prompt = self._http_json("POST", "/apply-template", body)["prompt"]
            rendered = time.monotonic()
            if self.shared_prefix_tokens is not None:
                cache_prompt = (chat_template_kwargs or {}) == self.shared_prefix_render_options
            payload = self._payload(prompt, max_new_tokens, cache_prompt, n_probs)
            tokenized = time.monotonic()
            first_output = None
            native_stream = self._events(payload, cancel)
            try:
                for event in native_stream:
                    if first_output is None and (event.get("tokens") or event.get("content")):
                        first_output = time.monotonic()
                    yield event
            finally:
                close = getattr(native_stream, "close", None)
                if close is not None:
                    close()
            if self.last_request is not None:
                phases = {"queue_seconds": acquired - requested, "load_seconds": loaded - acquired,
                          "admission_seconds": admitted - loaded, "template_seconds": rendered - admitted,
                          "tokenize_seconds": tokenized - rendered}
                phases["request_seconds_including_queue"] = time.monotonic() - requested
                self.last_request["host_phases"] = phases
                self.last_request["ttft_including_queue_seconds"] = (
                    first_output - requested if first_output is not None else None)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.close()
