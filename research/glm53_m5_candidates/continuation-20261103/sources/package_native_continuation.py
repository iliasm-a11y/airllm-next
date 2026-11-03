"""Package only the two closed noBLAS model captures, in an idle window.

This is evidence packaging, never inference or qualification. Preserves native
failed/cancelled guards and partial JSONL exactly. No model/credential/app data.
Prepared while C16 is active; do not execute until all hardware tasks are closed.
"""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUTPUT = ROOT / "outputs"
RUNS = ("native-noblas-full32", "native-noblas-v4-full32")
ARCHIVE_CAP = 4 * 1024**3
DISK_RESERVE = 8 * 1024**3
# Exact files emitted by the frozen native_run wrapper, and the flat capture
# files emitted by native_prefill.cpp v3/v4. Unknown files are never exported.
RUN_FILES = frozenset({
    "guard.json", "guard-samples.jsonl", "run-prompts.json",
    "snapshot-key-manifest.json", "native-stderr.log", "native-stdout.log",
})
# Consolidation journal only; its presence grants no native qualification.
OPTIONAL_RUN_FILES = frozenset({".closed-capture-dedup-events.jsonl"})
CAPTURE_METADATA = frozenset({
    "report.json", "operations.jsonl", "tensors.jsonl", "batches.jsonl",
    "state_hashes.jsonl", "state_checkpoints.jsonl",
    "continuation_tensors.jsonl", "continuation_states.jsonl",
})
CAPTURE_PAYLOAD_PATTERNS = (
    r"tensor-(?:0|[1-9][0-9]*)\.bin",
    r"logits-[012]\.f32",
    r"final-state-[012]\.bin",
    r"continuation-logits-[012]-[0-3]\.f32",
    r"continuation-state-[012]\.bin",
)
EXCLUDED_RUN_LOCK = ".closed-capture-dedup.lock"


def source_members(source):
    """Validate only the known, flat producer layout, without following links."""
    if not stat.S_ISDIR(source.lstat().st_mode):
        raise ValueError("Expected the original closed run directory")
    top = {path.name: path for path in source.iterdir()}
    allowed = RUN_FILES | OPTIONAL_RUN_FILES | {"capture", EXCLUDED_RUN_LOCK}
    if set(top) - allowed or not (RUN_FILES | {"capture"}) <= set(top):
        raise ValueError("Unexpected or missing closed-run member")
    selected_top = RUN_FILES | (OPTIONAL_RUN_FILES & set(top))
    for name in selected_top:
        if not stat.S_ISREG(top[name].lstat().st_mode):
            raise ValueError("Run metadata must be a regular file, never a symlink")
    if EXCLUDED_RUN_LOCK in top and not stat.S_ISREG(top[EXCLUDED_RUN_LOCK].lstat().st_mode):
        raise ValueError("Excluded dedup lock is not a regular file")
    capture = top["capture"]
    if not stat.S_ISDIR(capture.lstat().st_mode):
        raise ValueError("Capture directory must not be a symlink")
    payloads = sorted(capture.iterdir())
    names = {path.name for path in payloads}
    if not CAPTURE_METADATA <= names:
        raise ValueError("Missing expected native capture metadata")
    for path in payloads:
        if (path.name not in CAPTURE_METADATA and
                not any(re.fullmatch(pattern, path.name) for pattern in CAPTURE_PAYLOAD_PATTERNS)):
            raise ValueError("Unexpected native capture member")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Native capture members must be flat regular files")
    return [source, capture, *payloads, *(top[name] for name in sorted(selected_top))]


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def main():
    start = time.perf_counter()
    archive = OUTPUT / "Preuves-natives-noBLAS-M5.tar.gz"
    temporary = WORK / (archive.name + ".partial")
    manifest_file = OUTPUT / "Preuves-natives-noBLAS-M5.manifest.json"
    if archive.exists() or temporary.exists() or manifest_file.exists():
        raise FileExistsError("New evidence output required")
    frozen = {}
    for name in RUNS:
        path = WORK / name / "guard.json"
        data = path.read_bytes()
        guard = json.loads(data)
        if (guard["status"] not in ("failed", "cancelled") or guard["returncode"] != -15
                or guard["cleanup"]["owned_server_stopped"] is not True):
            raise ValueError("Expected the actual closed partial native failure")
        frozen[str(path)] = data
    # The exact same benchmark and UID locks prevent accidental packaging during
    # the functional pilot, a new build or a model. No runtime imports are used.
    target = "ggml-org/GLM-5.3-Flash-GGUF@119034bc3d9d76eb41f91508e6f9fb19fde42672@Q2_K_S"
    lock_name = hashlib.sha256(target.encode()).hexdigest()[:24] + ".lock"
    uid_path = Path(tempfile.gettempdir()) / ("airllm-native-locks-" + str(os.getuid())) / lock_name
    if not uid_path.is_file():
        raise FileNotFoundError("Existing canonical UID lock required")
    manifest = {}
    directories = set()
    with Path("/tmp/airllm-glm53-benchmark.lock").open("a") as bench, uid_path.open("a") as uid:
        fcntl.flock(bench, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(uid, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(OUTPUT).free < ARCHIVE_CAP + DISK_RESERVE:
            raise RuntimeError("Evidence archive disk admission refused")
        # Validate both complete name lists before opening an output archive.
        # This does not inspect tensor contents or treat partial captures as full.
        admitted = [(WORK / name, source_members(WORK / name)) for name in RUNS]
        with temporary.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", dereference=False) as tar:
                for source, members in admitted:
                    for path in members:
                        if path.is_symlink():
                            raise ValueError("Symlink refused")
                        if path.name.startswith(".native-api-key") or path.suffix in (".gguf", ".dylib") or "engine-state" in path.parts:
                            raise ValueError("Forbidden credential/model/runtime path")
                        member = str(path.relative_to(WORK))
                        if ".." in Path(member).parts or Path(member).is_absolute():
                            raise ValueError("Unsafe member")
                        before = path.stat()
                        info = tar.gettarinfo(str(path), arcname=member)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        if info.isreg():
                            with path.open("rb") as stream:
                                reader = HashReader(stream)
                                tar.addfile(info, reader)
                            manifest[member] = {"bytes": info.size, "sha256": reader.digest.hexdigest(), "kind": "file"}
                        elif info.islnk():
                            target_info = manifest[info.linkname]
                            manifest[member] = {**target_info, "kind": "hardlink", "target": info.linkname}
                            tar.addfile(info)
                        elif info.isdir():
                            directories.add(member)
                            tar.addfile(info)
                        else:
                            raise ValueError("Unsupported member")
                        after = path.stat()
                        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                        if identity(before) != identity(after):
                            raise ValueError("Closed source changed while packaging")
                        if raw.tell() > ARCHIVE_CAP or shutil.disk_usage(OUTPUT).free < DISK_RESERVE:
                            raise RuntimeError("Evidence archive disk cap reached; partial file retained as failure")
        if temporary.stat().st_size > ARCHIVE_CAP or shutil.disk_usage(OUTPUT).free < DISK_RESERVE:
            raise RuntimeError("Final compressed archive exceeds disk cap; partial retained under work")
        # A separate read of every exported payload verifies tar/hardlinks/SHA;
        # it is deliberately performed only after model/build closure.
        verified = 0
        seen = set()
        seen_directories = set()
        with tarfile.open(temporary, "r:gz") as tar:
            for info in tar:
                if info.name in seen or info.name in seen_directories:
                    raise ValueError("Duplicate archive member")
                if info.isdir():
                    if info.name not in directories:
                        raise ValueError("Unexpected archived directory")
                    seen_directories.add(info.name)
                    continue
                if info.name not in manifest:
                    raise ValueError("Unexpected archived file")
                row = manifest[info.name]
                if info.islnk():
                    if (row["kind"] != "hardlink" or row.get("target") != info.linkname
                            or info.linkname not in seen):
                        raise ValueError("Hardlink mismatch")
                    target_row = manifest[info.linkname]
                    if row["bytes"] != target_row["bytes"] or row["sha256"] != target_row["sha256"]:
                        raise ValueError("Hardlink target payload mismatch")
                elif info.isreg():
                    if row["kind"] != "file":
                        raise ValueError("Archive member kind mismatch")
                    digest = hashlib.sha256()
                    with tar.extractfile(info) as stream:
                        for part in iter(lambda: stream.read(4 << 20), b""):
                            digest.update(part)
                    if info.size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                        raise ValueError("Payload verification failed")
                else:
                    raise ValueError("Unexpected archived type")
                seen.add(info.name)
                verified += 1
        if seen != set(manifest) or seen_directories != directories:
            raise ValueError("Archive member set is incomplete")
        for path, data in frozen.items():
            if Path(path).read_bytes() != data:
                raise ValueError("Closed guard changed")
        temporary.rename(archive)
        result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "exact closed failed/cancelled v3/v4 native captures; no complete-profile/EOS/cache/GPU/performance qualification",
            "archive": archive.name, "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha_file(archive), "verified_members": verified,
            "hardlinks": sum(r["kind"] == "hardlink" for r in manifest.values()),
            "member_set_exact_and_unique": True,
            "selection_policy": {"run_files": sorted(RUN_FILES),
                "optional_run_files": sorted(OPTIONAL_RUN_FILES),
                "capture_metadata": sorted(CAPTURE_METADATA),
                "capture_payload_patterns": CAPTURE_PAYLOAD_PATTERNS,
                "excluded_run_lock": EXCLUDED_RUN_LOCK,
                "unexpected_member_policy": "refuse; no recursive catch-all export",
                "content_scope": "named outputs of pinned synthetic diagnostic producers; no generic secret-content scan"},
            "elapsed_seconds": time.perf_counter()-start, "members": manifest,
            "exclusions": ["model weights", "API credentials", "personal application data", "native binaries/libraries"],
            "partial_jsonl_policy": "Preserved unchanged; trailing incomplete record remains incomplete"}
        manifest_file.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "members"}))


if __name__ == "__main__":
    main()
