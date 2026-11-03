"""Opt-in, isolated RONIN adapter candidate for the pinned V4 slot-file API.

No entry point starts a model. make_model_class() supplies a subclass only after
the coordinator explicitly chooses this research adapter. B8 is rechecked before
native load. The HTTP path itself remains unqualified, including after mocks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import threading
import time
import uuid

MiB = 1024 ** 2
PREFIX = 208
RAW_BYTES = 156739588
RAW_SHA256 = "4c4d6f24b2cf3c4f30a7894b8609035960a3ec5687521c942aa6a578e2e5a3cd"
RAW_HEADER = struct.pack("<II", 0xaf143cd8, 0)
FILE_MAGIC, FILE_VERSION, TOKEN_VERSION = 0x67677371, 4, 1
MAX_FILE = 256 * MiB
SOURCE_HASHES = {
    "research.glm53_batched_prefill.native_compare": "0019976ae27a7296e2bdafea2d67ff4a77dd77c77a6586b838d7da0ea629d6de",
    "research.glm53_batched_prefill.snapshot_key": "64260e4000da92653f8446e51abcaddff867e209d5cc1e720cdc983c7553af46",
    "research.glm53_m5.compare_fidelity": "75166d9590e26afbe51529e8ec9a141db5d57491ae7fd0c4b7810c1b695309a8",
}
VARIANT_COMPARE_SHA256 = "07939855afeb9bdb4fb087b27698e99ec6225ff1f0ae0df7a37e6cc3568d4df7"
VARIANT_PARAMETERS = {"prefill_chunk_policy": "cpu8-tail-compatible-v1",
                      "prefill_chunk_policy_version": 1, "tail_reference_microbatch": 8}
EXPECTED_PARAMETERS = {
    "n_ctx": 2048, "n_batch": 8, "n_ubatch": 8, "n_seq_max": 1,
    "n_rs_seq": 0, "n_outputs_max": 0, "n_outputs_max_per_seq": 1,
    "n_threads": 10, "n_threads_batch": 10, "type_k": 1, "type_v": 1,
    "flash_attn": -1, "offload_kqv": False, "op_offload": False,
    "n_gpu_layers": 0, "rope_scaling_type": -1, "rope_freq_base": 0.0,
    "rope_freq_scale": 0.0, "yarn_ext_factor": -1.0, "yarn_attn_factor": -1.0,
    "yarn_beta_fast": -1.0, "yarn_beta_slow": -1.0, "yarn_orig_ctx": 0,
    "swa_full": True, "kv_unified": False, "ctx_type": 0,
    "pooling_type": -1, "attention_type": -1, "embeddings": False,
    "moe_cache_size": 0, "n_samplers": 0,
}
EXTRA_FLAGS = [
    "--threads-batch", "10", "--swa-full", "--no-kv-offload", "--no-kv-unified",
    "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "auto",
    "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-idle-slots",
    "--cache-reuse", "0", "--no-cache-prompt", "--spec-type", "none", "--slots", "--jinja",
]
ENVIRONMENT_NAMES = {
    "AIRLLM_DENSE_PREFETCH_MAX_COLS", "AIRLLM_MMAP_DEMAND", "AIRLLM_EXPERT_PREFETCH",
    "AIRLLM_DENSE_PREFETCH", "AIRLLM_MMAP_NOCACHE",
}
PROOF_ONLY_ENVIRONMENT = {"RONIN_PREFILL_DIAGNOSTIC": "1"}


def server_knobs(proof_environment):
    """Keep diagnostic capture identity in the proof, never in the server env."""
    return {name: value for name, value in proof_environment.items() if name in ENVIRONMENT_NAMES}


class AdapterError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def require(condition, message, code="PREFIX_PROTOCOL"):
    if not condition:
        raise AdapterError(code, message)


def identity(path):
    s = Path(path).lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1,
            "ordinary, owned, unaliased evidence file required", "PREFIX_FILE")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(MiB), b""):
            check()
            h.update(chunk)
    check()
    return h.hexdigest()


def read_json(path, cap=64 * MiB):
    path = Path(path)
    before = identity(path)
    require(0 < before[2] <= cap, "bounded evidence JSON required")
    value = json.loads(path.read_text())
    require(identity(path) == before, "evidence changed during metadata read")
    return value


class CampaignBudget:
    """Create before Runtime/load: startup and common-prefix work consume7200s."""
    def __init__(self, seconds=7200, *, clock=time.monotonic):
        require(type(seconds) in (int, float) and 0 < seconds <= 7200, "campaign ceiling is7200s")
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + seconds

    def remaining(self, deadline=None):
        end = self.deadline if deadline is None else min(self.deadline, deadline)
        left = end - self.clock()
        require(left > 0, "logical or campaign deadline reached", "PREFIX_DEADLINE")
        return left


class Cancellation:
    def __init__(self, budget, deadline, event=None):
        self.budget, self.deadline, self.event = budget, deadline, event

    def is_set(self):
        return ((self.event is not None and self.event.is_set()) or
                self.budget.clock() >= min(self.deadline, self.budget.deadline))


@dataclass(frozen=True)
class SnapshotSpec:
    token_ids: tuple[int, ...]
    raw_bytes: int = RAW_BYTES
    raw_sha256: str = RAW_SHA256

    @property
    def packed_bytes(self):
        return 16 + len(self.token_ids) * 4

    @property
    def offset(self):
        return 12 + self.packed_bytes

    @property
    def file_bytes(self):
        return self.offset + self.raw_bytes - len(RAW_HEADER)


def validate_slot_file(path, spec, *, check=lambda: None):
    """Stream only: never allocate the entire156MB state in Python.

    SnapshotSpec can describe tiny unit-fixture payloads. Production admission
    below independently fixes RAW_BYTES/RAW_SHA256 and208 actual native IDs.
    """
    path = Path(path)
    before = identity(path)
    require(0 < before[2] <= MAX_FILE and before[2] == spec.file_bytes,
            "slot file size/cap differs before payload read", "PREFIX_FILE")
    require(len(spec.token_ids) == PREFIX, "exactly208 native IDs required")
    digest, canonical = hashlib.sha256(), hashlib.sha256(RAW_HEADER)
    with path.open("rb") as f:
        head = f.read(12)
        require(len(head) == 12 and struct.unpack("<III", head) ==
                (FILE_MAGIC, FILE_VERSION, spec.packed_bytes // 4), "slot header differs", "PREFIX_FILE")
        packed = f.read(spec.packed_bytes)
        require(len(packed) == spec.packed_bytes, "short packed token metadata", "PREFIX_FILE")
        marker, version, count = struct.unpack("<iII", packed[:12])
        require((marker, version, count) == (-1, TOKEN_VERSION, PREFIX), "server-token format differs", "PREFIX_FILE")
        ids = struct.unpack("<" + "i" * PREFIX, packed[12:12 + PREFIX * 4])
        require(ids == spec.token_ids and packed[-4:] == b"\0\0\0\0", "native IDs/media count differ", "PREFIX_FILE")
        digest.update(head)
        digest.update(packed)
        remaining = spec.raw_bytes - len(RAW_HEADER)
        while remaining:
            check()
            chunk = f.read(min(MiB, remaining))
            require(bool(chunk), "short native memory payload", "PREFIX_FILE")
            canonical.update(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        require(f.read(1) == b"", "checkpoint/draft/trailing bytes are forbidden", "PREFIX_FILE")
    check()
    require(identity(path) == before, "slot file changed during validation", "PREFIX_FILE")
    require(canonical.hexdigest() == spec.raw_sha256, "complete native state differs from B8 checkpoint", "PREFIX_FILE")
    return {"path": str(path), "file_bytes": before[2], "file_sha256": digest.hexdigest(),
            "canonical_raw_bytes": spec.raw_bytes, "canonical_raw_sha256": canonical.hexdigest(),
            "memory_offset": spec.offset, "token_count": PREFIX, "identity": before}


def strict_b8_comparison(repository, reference, restored, reference_guard, restored_guard):
    """Recheck closed raw B8 evidence; no model load, no runtime process."""
    repository = Path(repository).resolve()
    sys.path[:0] = [str(repository), str(repository / "air_llm")]
    for module, expected in SOURCE_HASHES.items():
        path = repository / (module.replace(".", "/") + ".py")
        require(sha(path) == expected, "delegated comparison source identity changed", "B8_UNQUALIFIED")
    for module in SOURCE_HASHES:
        path = repository / (module.replace(".", "/") + ".py")
        loaded = importlib.import_module(module)
        require(Path(loaded.__file__).resolve() == path, "comparison import comes from another checkout", "B8_UNQUALIFIED")
    canonical_key = importlib.import_module("research.glm53_batched_prefill.snapshot_key").canonical_key
    for root, guard_path in ((reference, reference_guard), (restored, restored_guard)):
        receipt = read_json(guard_path)
        report = read_json(Path(root) / "report.json", 2 * MiB)
        key, _ = canonical_key(receipt["preflight"]["snapshot_manifest"])
        require(key == report.get("snapshot_key_sha256") == receipt["preflight"].get("snapshot_key_sha256"),
                "each native profile must retain its own recomputed key", "B8_UNQUALIFIED")
    module = importlib.import_module("research.glm53_batched_prefill.native_compare")
    result = module.compare(reference, restored, reference_guard, restored_guard)
    for module_name, expected in SOURCE_HASHES.items():
        require(sha(repository / (module_name.replace(".", "/") + ".py")) == expected,
                "delegated source changed during native comparison", "B8_UNQUALIFIED")
    return result


def strict_variant_bridge(repository, candidate_root, candidate_guard, bridge):
    """Only the reviewed two-build comparator may establish CPU8→variant16."""
    required = {"variant_comparator", "control_root", "control_guard", "reference_build", "candidate_build", "binding"}
    require(isinstance(bridge, dict) and set(bridge) == required, "complete actual two-build bridge required", "B8_UNQUALIFIED")
    path = Path(bridge["variant_comparator"]).resolve()
    require(sha(path) == VARIANT_COMPARE_SHA256, "variant comparator identity changed", "B8_UNQUALIFIED")
    name = "ronin_prefix_reviewed_variant_comparison"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.REPO.resolve() == Path(repository).resolve(), "variant comparison checkout differs", "B8_UNQUALIFIED")
    result = module.compare(bridge["control_root"], candidate_root, bridge["control_guard"], candidate_guard,
                            bridge["reference_build"], bridge["candidate_build"], bridge["binding"])
    require(sha(path) == VARIANT_COMPARE_SHA256, "variant comparator changed during comparison", "B8_UNQUALIFIED")
    return result


class NativePrefixGate:
    """Closed numerical evidence, independent of directory names.

    Profile8 needs full8→restore8. Profile16 additionally needs the actual
    reviewed two-build CPU8→full16 bridge; a declared16 flag cannot admit it.
    The HTTP backend keeps its own fixed8 profile and a separate identity.
    """
    def __init__(self, repository, reference, restored, reference_guard, restored_guard, *, cpu8_bridge=None):
        self.repository, self.reference, self.restored = map(Path, (repository, reference, restored))
        self.reference_guard, self.restored_guard = map(Path, (reference_guard, restored_guard))
        self.cpu8_bridge = cpu8_bridge
        self.admitted = None

    def verify(self, check=lambda: None):
        check()
        a, b = [read_json(root / "report.json", 2 * MiB) for root in (self.reference, self.restored)]
        ub = a.get("microbatch")
        require(type(ub) is int and ub in (8, 16) and b.get("microbatch") == ub,
                "only exact8 or explicitly bridged16 profiles are supported", "B8_UNQUALIFIED")
        native_parameters = dict(EXPECTED_PARAMETERS)
        if ub == 16:
            native_parameters.update(n_batch=16, n_ubatch=16, **VARIANT_PARAMETERS)
            require(self.cpu8_bridge is not None, "variant16 requires its real CPU8 two-build bridge", "B8_UNQUALIFIED")
        for report, mode in ((a, "full"), (b, "restore")):
            require(report.get("complete") is True and report.get("mode") == mode and
                    report.get("microbatch") == ub and report.get("state_parameters") == native_parameters,
                    "completed exact native full/restore evidence required", "B8_UNQUALIFIED")
        metadata_paths = [self.reference / "report.json", self.restored / "report.json",
                          self.reference_guard, self.restored_guard]
        frozen_metadata = {path: identity(path) for path in metadata_paths}
        guards = [read_json(p) for p in (self.reference_guard, self.restored_guard)]
        manifests = [g["preflight"]["snapshot_manifest"] for g in guards]
        require(manifests[0] == manifests[1] and manifests[0]["state_parameters"] == native_parameters,
                "B8 dependency manifests differ", "B8_UNQUALIFIED")
        rendering = manifests[0]["render_options"]
        require(rendering.get("reasoning_effort_override") is None and rendering.get("native_add_special") is True and
                rendering.get("native_parse_special") is True, "native default Max/tokenizer options required", "B8_UNQUALIFIED")
        prefix = manifests[0]["prefix_tokens"]
        require(len(prefix) == PREFIX and all(type(t) is int and t >= 0 for t in prefix),
                "native B8 prefix IDs required", "B8_UNQUALIFIED")
        result = strict_b8_comparison(self.repository, self.reference, self.restored,
                                      self.reference_guard, self.restored_guard)
        check()
        boundary = result.get("direct_restore_boundary", {})
        require(result.get("passed") is True and result.get("errors") == [] and
                result.get("criteria") == {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0} and
                boundary.get("passed") is True and boundary.get("boundary") == PREFIX and
                boundary.get("snapshot_bytes") == RAW_BYTES and boundary.get("snapshot_sha256") == RAW_SHA256,
                "strict completed B8 and exact208 checkpoint are not qualified", "B8_UNQUALIFIED")
        cross_profile = None
        if ub == 16:
            cross_profile = strict_variant_bridge(self.repository, self.reference, self.reference_guard, self.cpu8_bridge)
            require(cross_profile.get("passed") is True and cross_profile.get("errors") == [] and
                    cross_profile.get("criteria") == {"bitexact_f32": True, "atol": 0.0, "rtol": 0.0} and
                    cross_profile.get("reference_microbatch") == 8 and cross_profile.get("candidate_microbatch") == 16 and
                    cross_profile.get("checkpoint208_compared") is True and
                    cross_profile.get("build_binding", {}).get("validated") is True,
                    "actual two-build CPU8→variant16 fidelity bridge did not pass", "B8_UNQUALIFIED")
            check()
        environment = manifests[0]["render_options"]["native_environment"]
        require(set(environment) <= ENVIRONMENT_NAMES | set(PROOF_ONLY_ENVIRONMENT) and
                all(value == "1" for value in environment.values()) and
                all(environment.get(name) == value for name, value in PROOF_ONLY_ENVIRONMENT.items()),
                "B8 environment is outside the isolated CPU profile", "B8_UNQUALIFIED")
        require(guards[0]["preflight"]["source_sha256"] == guards[1]["preflight"]["source_sha256"],
                "B8 source dependencies differ", "B8_UNQUALIFIED")
        require(all(identity(path) == value for path, value in frozen_metadata.items()),
                "closed native evidence changed during qualification", "B8_UNQUALIFIED")
        self.admitted = {"manifest": manifests[0], "source_sha256": guards[0]["preflight"]["source_sha256"],
                         "comparison": result, "native_microbatch": ub, "cross_profile_comparison": cross_profile,
                         "proof_only_environment": dict(PROOF_ONLY_ENVIRONMENT), "server_environment": server_knobs(environment),
                         "spec": SnapshotSpec(tuple(prefix))}
        return self.admitted


# Compatibility name for earlier source fixtures; neither class inspects a
# directory name, and no admitted receipt or build is bundled with this module.
B8Gate = NativePrefixGate


class SlotPrefixMixin:
    """Mixin placed before the unchanged NativeGlmModel, never selected by default."""
    def __init__(self, *args, enable_shared_prefix=False, native_gate=None, b8_gate=None, campaign_budget=None,
                 admission_guard=None, evidence_dir=None, **kwargs):
        require(enable_shared_prefix is True, "research shared prefix is disabled by default", "PREFIX_DISABLED")
        require(not (native_gate is not None and b8_gate is not None), "select one explicit native qualification gate")
        selected_gate = native_gate if native_gate is not None else b8_gate
        require(isinstance(selected_gate, NativePrefixGate) and isinstance(campaign_budget, CampaignBudget) and
                callable(admission_guard) and evidence_dir is not None,
                "B8 gate, campaign budget and existing external admission guard required", "PREFIX_DISABLED")
        expected = {"context_tokens": 2048, "threads": 10, "batch_size": 8,
                    "gpu_layers": 0, "cpu_moe": False, "moe_cache_mib": 0, "allow_unqualified_gpu": False}
        for name, value in expected.items():
            require(type(kwargs.get(name)) is type(value) and kwargs[name] == value, "exact CPU8 option required: " + name)
        limits = kwargs.get("limits")
        require(limits is not None and limits.max_rss_bytes <= int(8.5 * 1024 ** 3) and
                limits.max_swap_growth_bytes <= 32 * MiB and limits.warning_swap_growth_bytes <= 32 * MiB and
                limits.request_timeout_seconds <= 2700, "existing native limits must retain research caps")
        defaults = vars(type(limits)())
        require(set(vars(limits)) == set(defaults), "complete original native limits required")
        for name, default in defaults.items():
            value = getattr(limits, name)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0 and
                    (type(default) is not int or type(value) is int), "positive typed native guard limit required: " + name)
            ceiling = 2700 if name == "request_timeout_seconds" else default
            require(value >= default if name in ("reserve_bytes", "base_workspace_bytes", "warning_headroom_bytes") else value <= ceiling,
                    "native policy was relaxed: " + name)
        self._prefix_gate, self._campaign, self._admission_guard = selected_gate, campaign_budget, admission_guard
        self._prefix_load_lock = threading.RLock()
        self._prefix_io_lock = threading.RLock()
        self._prefix_snapshot = None
        self._prefix_evidence = None
        self._prefix_poisoned = False
        self._prefix_logical = 0
        self._prefix_physical = 0
        self._prefix_discarded = 0
        self._prefix_journal_bytes = 0
        self._prefix_dir = Path(evidence_dir).absolute()
        require(not self._prefix_dir.exists() and not self._prefix_dir.is_symlink(), "fresh isolated evidence directory required")
        self._prefix_dir.mkdir(mode=0o700, parents=True)
        self._prefix_journal = self._prefix_dir / "physical.jsonl"
        self._prefix_journal.touch(mode=0o600, exist_ok=False)
        self._prefix_config_dir = self._prefix_dir / "isolated-config"
        self._prefix_config_dir.mkdir(mode=0o700)
        self._prefix_owned = set()
        self._prefix_http_scope = None
        self._prefix_external_deadline = None
        super().__init__(*args, **kwargs)

    @contextmanager
    def logical_scope(self):
        """Start before Runtime.dispatch, including host history and engine.load."""
        require(self._prefix_external_deadline is None, "one logical RONIN request at a time")
        self._prefix_external_deadline = min(self._campaign.deadline, self._campaign.clock() + 2700)
        try:
            with self._bounded(self._prefix_external_deadline) as cancel:
                yield cancel
        finally:
            self._prefix_external_deadline = None

    def _journal(self, record):
        with self._prefix_io_lock:
            record = dict(record, t_campaign=self._campaign.clock() - self._campaign.started)
            encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
            require(self._prefix_journal_bytes + len(encoded) <= 16 * MiB, "bounded physical journal exhausted")
            with self._prefix_journal.open("ab") as f:
                f.write(encoded)
            self._prefix_journal_bytes += len(encoded)

    def _check(self, deadline, cancel=None):
        self._campaign.remaining(deadline)
        require(cancel is None or not cancel.is_set(), "request cancelled", "CANCELLED")
        self._assert_no_presets()
        process = getattr(self, "_process", None)
        phase = "preload" if process is None else "loaded"
        if process is not None:
            self._check_running()
            self._prefix_admission_sample = self._admit()
            self._check_identity()
        receipt = self._admission_guard(self, phase=phase)
        require(isinstance(receipt, dict) and receipt.get("phase") == phase and all(receipt.get(k) is True for k in
                ("passed", "power_context_passed", "native_limits_unchanged")),
                "existing power/native guard refused its actual phase", "PREFIX_ADMISSION")
        if process is None:
            require(receipt.get("no_conflicting_uid_engine") is True and receipt.get("uid_exclusive") is not True and
                    receipt.get("identity_passed") is not True,
                    "preload must observe no competing engine without claiming unacquired ownership", "PREFIX_ADMISSION")
        else:
            native_lock = getattr(self, "_lock_file", None)
            require(receipt.get("uid_exclusive") is True and receipt.get("identity_passed") is True and
                    native_lock is not None and not native_lock.closed,
                    "loaded phase requires retained native lock and owned child identity", "PREFIX_ADMISSION")

    @contextmanager
    def _bounded(self, deadline, cancel=None):
        combined = Cancellation(self._campaign, deadline, cancel)
        stop = threading.Event()
        def monitor():
            while not stop.wait(.1):
                if combined.is_set():
                    self._prefix_poisoned = True
                    # Existing close sets _load_stop before taking the load lock.
                    # It owns/terminates the retained child; no new PID lookup.
                    self.close()
                    return
        watcher = threading.Thread(target=monitor, name="research-prefix-deadline", daemon=True)
        watcher.start()
        try:
            self._check(deadline, combined)
            yield combined
            self._check(deadline, combined)
        finally:
            stop.set()
            watcher.join(timeout=12)
            require(not watcher.is_alive(), "deadline cleanup did not finish", "PREFIX_CLEANUP")

    def _build_command(self, port):
        self._assert_no_presets()
        command = super()._build_command(port)
        require("--no-repack" in command and "--no-op-offload" in command and
                command[command.index("--parallel") + 1] == "1", "native CPU command contract changed")
        return command + EXTRA_FLAGS + ["--slot-save-path", str(self._prefix_dir)]

    def _runtime_environment(self):
        require(self._prefix_evidence is not None, "B8 must be checked before native environment construction")
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ("LLAMA_ARG_", "AIRLLM_", "GGML_", "DYLD_", "RONIN_PREFILL_", "RONIN_PREFIX_")) and k != "LLAMA_API_KEY"}
        env.update(server_knobs(self._prefix_evidence["manifest"]["render_options"]["native_environment"]))
        env["XDG_CONFIG_HOME"] = str(self._prefix_config_dir)
        return env

    def _assert_no_presets(self):
        # V4 automatically reads these before CLI. Never read or edit a user's
        # preset; reject the system file and isolate the process's user path.
        for path in (Path("/etc/llama.cpp/config.ini"), self._prefix_config_dir / "llama.cpp/config.ini"):
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            raise AdapterError("PREFIX_PRESET", "automatic V4 preset exists; exact helper parameters cannot be assumed")

    def _verify_profile_binding(self):
        evidence = self._prefix_evidence
        require(sha(self.runtime_manifest_file) == evidence["source_sha256"]["native_manifest"],
                "server package differs from B8 native manifest")
        require(self._runtime_artifacts_verified is True and self._checkpoint.get("integrity_verified") is True,
                "unchanged native integrity verification did not complete")
        actual = [{"sha256": r["sha256"], "size_bytes": r["bytes"]} for r in self._checkpoint["files"]]
        require(actual == evidence["manifest"]["model_shards"], "model shards differ from B8")
        require(sha(self.model_path / "tokenizer.json") == evidence["manifest"]["tokenizer_sha256"],
                "companion tokenizer provenance differs from B8")

    def _verify_environment_options(self):
        admitted = self._prefix_evidence["manifest"]["render_options"]["native_environment"]
        for attribute, name in (("mmap_demand", "AIRLLM_MMAP_DEMAND"), ("mmap_nocache", "AIRLLM_MMAP_NOCACHE"),
                                ("expert_prefetch", "AIRLLM_EXPERT_PREFETCH"), ("dense_prefetch", "AIRLLM_DENSE_PREFETCH")):
            require(getattr(self, attribute, False) is (admitted.get(name) == "1"),
                    "configured native environment differs from B8: " + attribute)

    def _current_profile(self):
        self._assert_no_presets()
        actual = {name: getattr(self, name) for name in
                  ("context_tokens", "threads", "batch_size", "gpu_layers", "cpu_moe", "moe_cache_mib", "allow_unqualified_gpu")}
        require(actual == {"context_tokens": 2048, "threads": 10, "batch_size": 8, "gpu_layers": 0,
                           "cpu_moe": False, "moe_cache_mib": 0, "allow_unqualified_gpu": False}, "live CPU options changed")
        fingerprint = getattr(self, "_runtime_sha256", None)
        require(isinstance(fingerprint, str) and len(fingerprint) == 64 and
                all(c in "0123456789abcdef" for c in fingerprint), "loaded server fingerprint missing")
        profile = {"schema": "ronin-slot-prefix-http-v1", "actual_backend": "native_cpu_server_slot_file",
            "qualified_native_dependency_manifest": self._prefix_evidence["manifest"], "server_sha256": fingerprint,
            "adapter_source_sha256": sha(Path(__file__)), "native_policy_source_sha256": sha(self._native_policy_source),
            "research_source_sha256": {name: sha(Path(__file__).parent / name)
                                       for name in ("guards.py", "campaign.py")
                                       if (Path(__file__).parent / name).is_file()},
            "external_guard_source_sha256": sha(Path(inspect.getfile(type(self._admission_guard))))
                if not inspect.isfunction(self._admission_guard) else sha(Path(inspect.getfile(self._admission_guard))),
            "external_guard_dependency_sha256": self._admission_guard.provenance()
                if hasattr(self._admission_guard, "provenance") else {},
            "state_parameters": EXPECTED_PARAMETERS, "constructor_options": actual, "limits": vars(self.limits),
            "extra_flags": EXTRA_FLAGS, "slot": 0, "use_extra_bufts": False,
            "proof_only_environment": dict(PROOF_ONLY_ENVIRONMENT),
            "server_environment": server_knobs(self._prefix_evidence["manifest"]["render_options"]["native_environment"]),
            "preparation_policy": "n_predict0; exactly one counted discarded sample; no private history insertion",
            "template_policy": "native default Max; no override", "snapshot_flags": 0,
            "request_seconds": 2700, "campaign_seconds_max": 7200,
            "automatic_preset_policy": "system config absent; process-only private empty XDG_CONFIG_HOME; monitored without reading presets",
            "http_path_numerically_qualified": False}
        encoded = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return profile, hashlib.sha256(encoded).hexdigest()

    def _http(self, method, path, payload, deadline, cancel, phase, logical=None, maximum_timeout=None):
        self._check(deadline, cancel)
        self._prefix_physical += 1
        number, start = self._prefix_physical, self._campaign.clock()
        self._journal({"physical": number, "logical": logical, "phase": phase, "method": method,
                       "path": path, "payload": payload, "status": "started"})
        try:
            timeout = self._campaign.remaining(deadline)
            if maximum_timeout is not None:
                timeout = min(timeout, maximum_timeout)
            value = super()._http_json(method, path, payload, timeout=timeout)
            self._check(deadline, cancel)
            self._journal({"physical": number, "logical": logical, "phase": phase,
                           "status": "completed", "seconds": self._campaign.clock() - start, "response": value})
            return value
        except BaseException as error:
            self._journal({"physical": number, "logical": logical, "phase": phase,
                           "status": "failed", "error": type(error).__name__})
            raise

    def _http_json(self, method, path, payload=None, timeout=30):
        # Include original startup health calls in the physical journal without
        # turning a1s readiness probe into a2700s socket wait.
        require(self._prefix_http_scope is not None, "unscoped native HTTP call refused")
        deadline, cancel = self._prefix_http_scope
        return self._http(method, path, payload, deadline, cancel, "native_startup_http", maximum_timeout=timeout)

    def _erase(self, deadline, cancel, logical=None):
        value = self._http("POST", "/slots/0?action=erase", {}, deadline, cancel, "erase", logical)
        require(value.get("id_slot") == 0 and type(value.get("n_erased")) is int and value["n_erased"] >= 0, "erase receipt missing")
        self._slot(0, deadline, cancel, logical)

    def _slot(self, expected, deadline, cancel, logical=None):
        value = self._http("GET", "/slots", None, deadline, cancel, "observe_slot", logical)
        require(isinstance(value, list) and len(value) == 1 and value[0].get("id") == 0 and
                value[0].get("is_processing") is False and value[0].get("n_prompt_tokens", 0 if expected == 0 else None) == expected,
                "exact idle slot token count not observed")

    def _save(self, deadline, cancel, phase, logical=None):
        name = "quarantine-" + uuid.uuid4().hex + ".bin"
        path = self._prefix_dir / name
        spec = self._prefix_evidence["spec"]
        require(0 < spec.file_bytes <= MAX_FILE, "expected complete file exceeds256MiB before save")
        self._prefix_owned.add(path)
        response = self._http("POST", "/slots/0?action=save", {"filename": name}, deadline, cancel, phase, logical)
        require(response.get("id_slot") == 0 and response.get("filename") == name and
                response.get("n_saved") == PREFIX and response.get("n_written") == spec.file_bytes,
                "save receipt differs from exact bounded208 format")
        result = validate_slot_file(path, spec, check=lambda: self._check(deadline, cancel))
        self._journal({"phase": phase + "_validated", "logical": logical, **result})
        return path, result

    def _prime(self, deadline, cancel):
        spec = self._prefix_evidence["spec"]
        properties = self._http("GET", "/props", None, deadline, cancel, "verify_native_template")
        require(properties.get("total_slots") == 1 and properties.get("endpoint_slots") is True and
                properties.get("default_generation_settings", {}).get("n_ctx") == 2048 and
                isinstance(properties.get("chat_template"), str) and hashlib.sha256(properties["chat_template"].encode()).hexdigest() ==
                self._prefix_evidence["manifest"]["template_sha256"], "native template/slot/context properties differ")
        self._erase(deadline, cancel)
        payload = {"prompt": list(spec.token_ids), "n_predict": 0, "stream": False,
                   "temperature": 0, "seed": 42, "cache_prompt": False,
                   "id_slot": 0, "return_tokens": True, "timings_per_token": True}
        result = self._http("POST", "/completion", payload, deadline, cancel, "prepare_common_prefix")
        tokens, timings = result.get("tokens"), result.get("timings", {})
        require(isinstance(tokens, list) and len(tokens) == 1 and type(tokens[0]) is int and 0 <= tokens[0] < 154880 and
                timings.get("predicted_n") == 1 and timings.get("prompt_n") == PREFIX and
                timings.get("cache_n") == 0 and result.get("truncated") is False and
                result.get("stop_type") in ("limit", "eos"), "n_predict0 must yield exactly one counted preparation sample")
        self._prefix_discarded += 1
        self._journal({"phase": "preparation_sample_discarded", "tokens": tokens,
                       "native_stop_type": result["stop_type"], "logical_history_inserted": False})
        self._slot(PREFIX, deadline, cancel)
        path, validated = self._save(deadline, cancel, "save_common_quarantine")
        admitted = self._prefix_dir / ("immutable-" + uuid.uuid4().hex + ".bin")
        path.rename(admitted)
        self._prefix_owned.remove(path)
        self._prefix_owned.add(admitted)
        admitted.chmod(0o400)
        validated.update(path=str(admitted), identity=identity(admitted))
        profile, key = self._current_profile()
        validated["http_profile_sha256"] = key
        self._prefix_snapshot = validated
        self._journal({"phase": "http_profile_bound", "http_profile_sha256": key, "profile": profile})
        self._erase(deadline, cancel)

    def load(self, cancel=None):
        try:
            with self._prefix_load_lock:
                require(not self._prefix_poisoned, "adapter stopped; no automatic model reload", "PREFIX_POISONED")
                deadline = min(self._campaign.deadline, self._campaign.clock() + self.limits.load_timeout_seconds,
                               self._prefix_external_deadline or self._campaign.deadline)
                with self._bounded(deadline, cancel) as combined:
                    if self._prefix_evidence is None:
                        self._prefix_evidence = self._prefix_gate.verify(lambda: self._check(deadline, combined))
                    self._verify_environment_options()
                    self._prefix_http_scope = deadline, combined
                    try:
                        super().load(cancel=combined)
                    finally:
                        self._prefix_http_scope = None
                    self._verify_profile_binding()
                    if self._prefix_snapshot is None:
                        self._prime(deadline, combined)
        except BaseException:
            self._prefix_poisoned = True
            self.close()
            raise
        return self.status()

    def _restore_prefix(self, deadline, cancel, logical):
        self._erase(deadline, cancel, logical)
        known = self._prefix_snapshot
        require(known is not None, "no admitted immutable prefix")
        require(self._current_profile()[1] == known["http_profile_sha256"], "HTTP profile changed since quarantine admission")
        path = Path(known["path"])
        require(identity(path) == known["identity"], "immutable prefix identity changed")
        checked = validate_slot_file(path, self._prefix_evidence["spec"], check=lambda: self._check(deadline, cancel))
        require(checked["file_sha256"] == known["file_sha256"], "immutable prefix changed")
        result = self._http("POST", "/slots/0?action=restore", {"filename": path.name}, deadline, cancel, "restore_prefix", logical)
        require(result.get("id_slot") == 0 and result.get("filename") == path.name and
                result.get("n_restored") == PREFIX and result.get("n_read") == known["file_bytes"], "restore receipt differs")
        self._slot(PREFIX, deadline, cancel, logical)
        # Dirty-state path proof: serialize the actual reloaded state, validate
        # its complete canonical payload, then discard only this owned copy.
        probe, _ = self._save(deadline, cancel, "verify_actual_restored_state", logical)
        probe.unlink()
        self._prefix_owned.remove(probe)

    def stream_chat(self, messages, max_new_tokens=512, *, cancel=None, cache_prompt=False,
                    n_probs=0, chat_template_kwargs=None):
        require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 512, "logical output budget is at most512")
        require(cache_prompt is False and n_probs == 0 and chat_template_kwargs is None,
                "only native default Max; no caller cache/sampling/template override")
        require(isinstance(messages, list) and bool(messages) and all(isinstance(m, dict) and
                set(m) == {"role", "content"} and m["role"] in ("system", "user", "assistant") and
                isinstance(m["content"], str) for m in messages), "complete text messages required")
        deadline = min(self._campaign.deadline, self._campaign.clock() + 2700,
                       self._prefix_external_deadline or self._campaign.deadline)
        self._prefix_logical += 1
        logical = self._prefix_logical
        terminal = None
        generation_record = None
        try:
            with self._bounded(deadline, cancel) as combined, self.scheduler.slot(combined):
                self.load(cancel=combined)
                body = {"messages": messages}
                rendered = self._http("POST", "/apply-template", body, deadline, combined, "apply_template", logical)
                require(isinstance(rendered.get("prompt"), str) and rendered["prompt"], "native rendered prompt missing")
                ids = self._http("POST", "/tokenize", {"content": rendered["prompt"], "add_special": True,
                    "parse_special": True}, deadline, combined, "native_tokenize", logical).get("tokens")
                require(isinstance(ids, list) and len(ids) > PREFIX and all(type(t) is int and t >= 0 for t in ids), "native full IDs missing")
                require(tuple(ids[:PREFIX]) == self._prefix_evidence["spec"].token_ids, "request native prefix differs")
                require(len(ids) + max_new_tokens <= 2048, "full prompt plus output exceeds context; no truncation")
                self._restore_prefix(deadline, combined, logical)
                payload = {"prompt": ids, "n_predict": max_new_tokens, "stream": True, "temperature": 0,
                           "seed": 42, "cache_prompt": True, "id_slot": 0, "return_tokens": True,
                           "n_probs": 0, "post_sampling_probs": False, "timings_per_token": True, "sse_ping_interval": 1}
                self._prefix_physical += 1
                physical, start = self._prefix_physical, self._campaign.clock()
                generation_record = physical
                self._journal({"physical": physical, "logical": logical, "phase": "generate", "payload": payload, "status": "started"})
                total_tokens = 0
                for event in super()._events(payload, combined):
                    self._check(deadline, combined)
                    require(isinstance(event, dict), "native event object required")
                    tokens = event.get("tokens", [])
                    require(isinstance(tokens, list) and all(type(t) is int and 0 <= t < 154880 for t in tokens), "native token IDs missing")
                    total_tokens += len(tokens)
                    require(total_tokens <= max_new_tokens, "native logical token budget exceeded")
                    self._journal({"physical": physical, "logical": logical, "phase": "generate_event", "response": event})
                    if event.get("stop"):
                        require(terminal is None, "multiple native terminal events")
                        terminal = event
                        timings = event.get("timings", {})
                        require(timings.get("cache_n") == PREFIX and timings.get("prompt_n") == len(ids) - PREFIX and
                                timings.get("predicted_n") == total_tokens and total_tokens > 0 and
                                event.get("tokens_predicted") == total_tokens and event.get("tokens_evaluated") == len(ids) and
                                event.get("truncated") is False and event.get("stop_type") in ("eos", "limit"),
                                "restored208 suffix-only processing not observed")
                    yield event
                require(terminal is not None, "native stream ended without a terminal")
                self._journal({"physical": physical, "logical": logical, "phase": "generate", "status": "completed",
                               "seconds": self._campaign.clock() - start, "generated_tokens": total_tokens,
                               "stop_type": terminal["stop_type"], "functional_qualification": False})
                self._erase(deadline, combined, logical)
        except BaseException as error:
            self._prefix_poisoned = True
            try:
                self._journal({"logical": logical, "physical": generation_record, "phase": "logical_request",
                               "status": "failed", "error": type(error).__name__, "code": getattr(error, "code", None)})
            finally:
                self.close()
            raise

    def stream_completion(self, *args, **kwargs):
        raise AdapterError("PREFIX_UNSUPPORTED", "use complete RONIN messages, not a raw completion")

    def status(self):
        result = super().status()
        result["research_shared_prefix"] = {"enabled": True, "native_prefix_admitted": self._prefix_evidence is not None,
            "qualified_native_microbatch": (self._prefix_evidence or {}).get("native_microbatch"),
            "immutable_http_prefix_accepted": self._prefix_snapshot is not None,
            "http_profile_sha256": (self._prefix_snapshot or {}).get("http_profile_sha256"),
            "logical_requests_attempted": self._prefix_logical, "physical_requests_started": self._prefix_physical,
            "preparation_samples_discarded": self._prefix_discarded, "poisoned": self._prefix_poisoned,
            "http_path_numerically_qualified": False, "functional_C_qualified": False,
            "scope": "isolated research candidate; native helper admission is not HTTP/C qualification"}
        return result

    def close(self):
        try:
            return super().close()
        finally:
            self._prefix_snapshot = None
            self._prefix_evidence = None
            for path in tuple(self._prefix_owned):
                path.unlink(missing_ok=True)
                self._prefix_owned.discard(path)

    unload = close


def make_model_class(native_model_class):
    """For an isolated research factory; never patches Engine or production code."""
    return type("ResearchSlotPrefixModel", (SlotPrefixMixin, native_model_class),
                {"__module__": __name__, "_native_policy_source": Path(inspect.getfile(native_model_class)).resolve()})
