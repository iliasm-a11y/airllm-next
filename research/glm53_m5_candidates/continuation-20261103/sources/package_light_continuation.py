"""Reviewed allowlist delivery only. Default is plan; no model or publication API.

Prepared as SOURCE ONLY. --execute requires a separately approved, SHA-bound
extra_allowlist_final.json, closed receipts and both exclusive hardware locks.
Historical source/receipt bytes are never edited or represented as new proof.
"""
from __future__ import annotations

import argparse
import codecs
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
WORK, OUTPUTS = ROOT / "work", ROOT / "outputs"
PUBLIC_REL = PurePosixPath("research/glm53_m5_candidates/continuation-20261103")
PUBLIC = WORK / "publication" / str(PUBLIC_REL)
SELECTION = WORK / "continuation-selection-compact.json"
EXTRAS = WORK / "extra_allowlist_final.json"
ZIP_NAME = "Sources-et-preuves-continuation-M5.zip"
MANIFEST_NAME = "Sources-et-preuves-continuation-M5.manifest.json"
GiB, MiB = 1024**3, 1024**2
DISK_RESERVE, WRITE_BUDGET = 32 * GiB, 32 * MiB
TREE_CAP, INPUT_CAP, FILE_CAP, FILE_COUNT_CAP = 7_000_000, 96 * MiB, 32 * MiB, 384
CHUNK = 256 * 1024
CODE_EXT = {".py", ".cpp", ".h", ".inc", ".mm", ".metal", ".patch", ".md", ".cmake"}
TEXT_EXT = CODE_EXT | {".json", ".jsonl", ".log", ".txt"}
SOURCE_JSON_NAMES = {
    "source-manifest.json", "prepare-manifest.json", "prepared-build-recipe.json",
    "build-plan.json", "build-recipe.json", "binding-template.json", "build-binding.json",
    "actual-build-binding.json", "actual-binding-check.json", "probe-compare-prepare.json",
    "selection.json", "frozen-http8-source-manifest.json", "plan.json", "prepare.json",
    "run-config.planned.json",
}
SOURCE_JSON_PATHS = {"cpu10-reference.json", "native-nocopy-probe-config.json"}
OUTPUT_NAMES = {
    "Rapport-final-RONIN-M5.md", "Rapport-intermediaire-RONIN-M5.md",
    "Metriques-RONIN-M5.json", "Preuves-reprise-M5.json", "Integration-RONIN-M5.md",
    "LIVRAISON-M5.md", "Empreintes-build-M5.json", "Empreintes-livraison-M5.json",
    "Statut-execution-M5.json",
}
FORBIDDEN_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "private-app-memory",
    "engine-state", "checkouts", "model_weights", "models",
}
ROLES = {
    "source_or_documentation", "parent_receipt", "parent_source_review_receipt",
    "closed_parent_metadata_or_source_review", "closed_parent_metadata_or_build_log",
    "closed_parent_metadata_or_log", "closed_synthetic_prompt_fixture",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_rel(value):
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
            "Invalid explicit relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
            all(p not in (".", "..", "") for p in path.parts), "Unsafe relative path")
    lower = [part.lower() for part in path.parts]
    require(not FORBIDDEN_PARTS.intersection(lower), "Private/checkout path refused")
    require(not any("api-key" in p or "api_key" in p or "apikey" in p or
                    p in {".env", "credentials.json", "secrets.json"} for p in lower),
            "Credential filename refused; argv path strings are not credential files")
    require(path.suffix.lower() in TEXT_EXT, "Weights, raw arrays, binaries and archives refused")
    return path


def safe_path(root_name, rel):
    require(root_name in {"work", "outputs"}, "Unknown source root")
    rel = canonical_rel(rel)
    require(not (root_name == "work" and rel.parts[0] in
                 {"publication", "airllm-next", "metal-integration-v4-source"}),
            "Whole checkout / previous publication is not a source root")
    root = WORK if root_name == "work" else OUTPUTS
    current = root
    require(stat.S_ISDIR(root.lstat().st_mode), "Source root is not a plain directory")
    for part in rel.parts:
        current = current / part
        mode = current.lstat().st_mode
        require(not stat.S_ISLNK(mode), "Source symlink refused")
    require(stat.S_ISREG(current.lstat().st_mode), "Only regular source files admitted")
    return current


def identity(path):
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode), "Source type changed")
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def open_regular(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        actual = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        require(stat.S_ISREG(st.st_mode) and actual == tuple(expected), "Source FD identity differs before read")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK), b""):
            digest.update(data)
    return digest.hexdigest()


def load_json(path, cap=MiB):
    before = identity(path)
    require(before[2] <= cap, "JSON admission cap exceeded")
    with open_regular(path, before) as stream:
        data = stream.read()
    require(identity(path) == before, "JSON changed while reading")
    return json.loads(data), hashlib.sha256(data).hexdigest(), before


def describe_entries(selection, extra):
    entries, seen = [], set()
    require(selection.get("schema_version") == 1 and selection.get("status") in {
        "selection_only_no_copy_no_publication",
        "FINAL_explicit_frozen_selection_no_copy_no_archive_no_publication",
    }, "Wrong selection schema/status")
    require(Path(selection["source_work_root"]) == WORK, "Selection belongs to another workspace")
    for row in selection["allowed_entries"]:
        require(row["role"] in ROLES, "Unreviewed base role")
        rel = canonical_rel(row["source_relative_to_work"])
        delivery = "source" if (rel.suffix in CODE_EXT or rel.name in SOURCE_JSON_NAMES or
                                 str(rel) in SOURCE_JSON_PATHS or row["role"] == "source_or_documentation") else "evidence"
        entries.append({"root": "work", "relative_path": str(rel), "bytes": row["bytes_at_selection"],
                        "expected_sha256": row.get("sha256"), "delivery": delivery,
                        "role": row["role"], "group": row["group"], "origin": "base_selection"})
    require(len(entries) == selection["counts"]["allowlist_files"], "Base selection count mismatch")
    require(len(extra.get("entries", [])) <= 64, "Too many final additions")
    for row in extra.get("entries", []):
        require(row.get("closed_artifact") is True, "Final entry lacks explicit closed/frozen attestation")
        rel = canonical_rel(row["relative_path"])
        require(row["root"] in {"work", "outputs"}, "Invalid final root")
        require(row["delivery"] in {"source", "evidence", "output_only"}, "Invalid delivery kind")
        if row["root"] == "outputs":
            require(len(rel.parts) == 1 and rel.name in OUTPUT_NAMES and row["delivery"] == "output_only",
                    "Only named final reports/metadata may enter outputs-only archive section")
        else:
            require(row["delivery"] != "output_only", "Output-only metadata must come from outputs")
        require(re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None,
                "Final additions require actual SHA256 pins")
        entries.append({"root": row["root"], "relative_path": str(rel), "bytes": row["bytes"],
                        "expected_sha256": row["sha256"], "delivery": row["delivery"],
                        "role": row["role"], "group": row.get("group", "final_root_selection"),
                        "origin": "approved_final_allowlist"})
    for row in entries:
        key = row["root"], row["relative_path"]
        require(key not in seen, "Duplicate/override refused; version a changed final artifact explicitly")
        seen.add(key)
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= FILE_CAP, "Per-file bound exceeded")
    require(len(entries) <= FILE_COUNT_CAP and sum(row["bytes"] for row in entries) <= INPUT_CAP,
            "Light selection count/total input cap exceeded")
    return entries


def inspect_and_pin(row, accept_deferred):
    path = safe_path(row["root"], row["relative_path"])
    before = identity(path)
    require(before[2] == row["bytes"], "Selected size changed; no automatic refresh")
    require(row["expected_sha256"] is not None or accept_deferred,
            "Deferred inventory SHA requires explicit root acceptance at quiescent freeze")
    digest, decoder, tail = hashlib.sha256(), codecs.getincrementaldecoder("utf-8")(), b""
    with open_regular(path, before) as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            decoder.decode(chunk)
            require(b"\x00" not in chunk, "Binary data disguised as text refused")
            text = tail + chunk
            require(re.search(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", text) is None and
                    re.search(rb"(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{24,}", text) is None,
                    "Possible credential content; refuse without redaction")
            tail = text[-256:]
            digest.update(chunk)
    decoder.decode(b"", final=True)
    require(identity(path) == before, "Source changed during freeze")
    actual = digest.hexdigest()
    require(row["expected_sha256"] in (None, actual), "Source SHA differs from reviewed pin")
    # Only actual guard-like documents receive this additional running check.
    if path.suffix == ".json" and path.name in {"guard.json", "build-guard.json", "report.json"}:
        doc, _, _ = load_json(path, FILE_CAP)
        require(doc.get("status") not in {"running", "active", "launching", "initializing"},
                "Active guard/report refused")
    return {**row, "original_path": str(path), "original_sha256": actual,
            "identity": before, "hash_scope": "quiescent export freeze; not retroactive measured-build proof"}


def verify_closures(extra):
    require(extra.get("closure_checks"), "At least one actual final closure receipt is required")
    records = []
    for check in extra["closure_checks"]:
        path = safe_path(check["root"], check["relative_path"])
        require(path.suffix == ".json", "Closure must be JSON metadata")
        doc, digest, ident = load_json(path, FILE_CAP)
        require(check["bytes"] == ident[2] and check["sha256"] == digest, "Closure receipt pin differs")
        expected = check["expected_fields"]
        require(expected and doc.get("status") in
                {"completed", "failed", "cancelled", "blocked", "functional_rejected_REQUEST_TIMEOUT"},
                "Receipt is not an admitted closed status")
        for dotted, value in expected.items():
            current = doc
            for key in dotted.split("."):
                require(isinstance(current, dict) and key in current, "Missing closure field")
                current = current[key]
            require(type(current) is type(value) and current == value, "Closure condition differs")
        records.append({"original_path": str(path), "original_sha256": digest, "bytes": ident[2],
                        "identity": ident, "expected_fields": expected})
    return records


def free_space():
    return min(shutil.disk_usage(WORK).free, shutil.disk_usage(OUTPUTS).free)


class Budget:
    def __init__(self):
        self.used = 0

    def charge(self, increase):
        require(self.used + increase <= WRITE_BUDGET and free_space() >= DISK_RESERVE + increase,
                "Light write budget / 32GiB disk reserve refused")
        self.used += increase


class BudgetFile:
    def __init__(self, path, budget):
        self.file, self.budget, self.high = path.open("x+b"), budget, 0

    def write(self, data):
        end = self.file.tell() + len(data)
        self.budget.charge(max(0, end - self.high))
        count = self.file.write(data)
        require(count == len(data), "Short archive write")
        self.high = max(self.high, end)
        return count

    def __getattr__(self, name):
        return getattr(self.file, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()


class DigestSink:
    def __init__(self, target):
        self.target, self.digest, self.bytes = target, hashlib.sha256(), 0

    def write(self, data):
        count = self.target.write(data)
        require(count == len(data), "Short gzip member write")
        self.digest.update(data)
        self.bytes += count
        return count

    def flush(self):
        return self.target.flush()


def copy_checked(row, target):
    path, digest = Path(row["original_path"]), hashlib.sha256()
    require(identity(path) == tuple(row["identity"]), "Source changed after freeze")
    with open_regular(path, row["identity"]) as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            target.write(chunk)
            digest.update(chunk)
    require(identity(path) == tuple(row["identity"]) and digest.hexdigest() == row["original_sha256"],
            "Source changed while exporting")


def write_bytes(path, data, budget):
    path.parent.mkdir(parents=True, exist_ok=True)
    with BudgetFile(path, budget) as target:
        target.write(data)
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def verify_evidence(path, rows):
    expected = {row["gzip_member"]: row for row in rows}
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist()]
        require(len(names) == len(set(names)) and set(names) == set(expected), "Evidence ZIP member set differs")
        for name, row in expected.items():
            digest, count, packed_sha, packed_count = hashlib.sha256(), 0, hashlib.sha256(), 0
            with archive.open(name) as compressed:
                for chunk in iter(lambda: compressed.read(CHUNK), b""):
                    packed_sha.update(chunk)
                    packed_count += len(chunk)
            require(packed_count == row["gzip_bytes"] and packed_sha.hexdigest() == row["gzip_sha256"],
                    "Gzip bytes differ")
            with archive.open(name) as compressed:
                with gzip.GzipFile(fileobj=compressed) as stream:
                    for chunk in iter(lambda: stream.read(CHUNK), b""):
                        count += len(chunk)
                        require(count <= row["bytes"], "Decompression exceeds declared original size")
                        digest.update(chunk)
            require(count == row["bytes"] and digest.hexdigest() == row["original_sha256"], "Evidence raw SHA differs")


def publish_exclusive(stage, public_files, output_zip, output_manifest):
    """Exclusive creates/links only. Roll back only inodes created by this call."""
    created_files, created_dirs = [], []
    try:
        require(not PUBLIC.exists() and not PUBLIC.is_symlink(), "New namespace already exists")
        PUBLIC.parent.mkdir(parents=True, exist_ok=True)
        PUBLIC.mkdir()
        st = PUBLIC.lstat()
        created_dirs.append((PUBLIC, (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode))))
        for rel in sorted(public_files):
            destination = PUBLIC / rel
            missing = []
            parent = destination.parent
            while parent != PUBLIC and not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for directory in reversed(missing):
                directory.mkdir()
                st = directory.lstat()
                created_dirs.append((directory, (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode))))
            os.link(stage / rel, destination, follow_symlinks=False)
            created_files.append((destination, identity(destination)))
        for source, destination in ((output_zip, OUTPUTS / ZIP_NAME), (output_manifest, OUTPUTS / MANIFEST_NAME)):
            os.link(source, destination, follow_symlinks=False)
            created_files.append((destination, identity(destination)))
    except BaseException:
        for path, ident in reversed(created_files):
            if path.exists() and identity(path) == ident:
                path.unlink()
        for directory, original in reversed(created_dirs):
            try:
                st = directory.lstat()
                if (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode)) == original:
                    directory.rmdir()  # refuses unknown files and replacement directories
            except OSError:
                pass
        raise


def execute(selection, selection_sha, selection_id, extra, extra_sha, extra_id):
    require(extra.get("schema_version") == 1 and extra.get("approved_for_delivery") is True and
            extra.get("all_base_entries_frozen") is True, "Root final review/approval is absent")
    require(extra.get("selection_sha256") == selection_sha, "Final allowlist does not pin this exact selection")
    entries = describe_entries(selection, extra)
    require(not any(path.exists() or path.is_symlink() for path in
                    (PUBLIC, OUTPUTS / ZIP_NAME, OUTPUTS / MANIFEST_NAME)), "Fresh namespace/output names required")
    require(OUTPUTS.is_dir() and (WORK / "publication").is_dir(), "Existing publication/outputs roots required")
    for root in (WORK / "publication", OUTPUTS):
        require(stat.S_ISDIR(root.lstat().st_mode), "Publication/output roots cannot be symlinks")
    for part_count in range(1, len(PUBLIC_REL.parts)):
        parent = WORK / "publication" / str(PurePosixPath(*PUBLIC_REL.parts[:part_count]))
        if parent.exists() or parent.is_symlink():
            require(stat.S_ISDIR(parent.lstat().st_mode), "Namespace parent cannot be a symlink or file")
    target = "ggml-org/GLM-5.3-Flash-GGUF@119034bc3d9d76eb41f91508e6f9fb19fde42672@Q2_K_S"
    key = hashlib.sha256(target.encode()).hexdigest()[:24] + ".lock"
    uid_path = Path(extra["uid_lock_path"])
    canonical_uid = Path(tempfile.gettempdir()) / ("airllm-native-locks-" + str(os.getuid())) / key
    require(uid_path.is_absolute() and uid_path.resolve(strict=True) == canonical_uid.resolve(strict=True),
            "Full canonical NativeModel UID lock path required; homonymous locks refused")
    with ExitStack() as stack:
        for path in (Path("/tmp/airllm-glm53-benchmark.lock"), uid_path):
            before = path.stat()
            require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(), "Existing owned lock required")
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            actual = os.fstat(fd)
            require((actual.st_dev, actual.st_ino) == (before.st_dev, before.st_ino), "Lock inode changed")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(free_space() >= DISK_RESERVE + WRITE_BUDGET, "32 GiB reserve plus 32 MiB light budget required")
        closures = verify_closures(extra)
        pinned = [inspect_and_pin(row, extra.get("accept_deferred_base_hashes") is True) for row in entries]
        budget = Budget()
        with tempfile.TemporaryDirectory(prefix=".light-continuation-", dir=WORK) as temp:
            temp, public_files = Path(temp), {}
            stage = temp / "namespace"
            stage.mkdir()
            evidence_rows = []
            evidence_path = stage / "evidence-continuation.zip"
            with BudgetFile(evidence_path, budget) as raw:
                with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
                    for row in pinned:
                        if row["delivery"] != "evidence":
                            continue
                        member = row["root"] + "/" + row["relative_path"] + ".gz"
                        with archive.open(member, "w") as zip_stream:
                            sink = DigestSink(zip_stream)
                            with gzip.GzipFile(fileobj=sink, filename="", mode="wb", mtime=0, compresslevel=6) as zipped:
                                copy_checked(row, zipped)
                        evidence_rows.append({**row, "gzip_member": member, "gzip_bytes": sink.bytes,
                                              "gzip_sha256": sink.digest.hexdigest()})
            verify_evidence(evidence_path, evidence_rows)
            public_files["evidence-continuation.zip"] = sha(evidence_path)
            source_rows = []
            for row in pinned:
                if row["delivery"] != "source":
                    continue
                rel = "sources/" + row["relative_path"]
                path = stage / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with BudgetFile(path, budget) as target_stream:
                    copy_checked(row, target_stream)
                public_files[rel] = row["original_sha256"]
                source_rows.append({**row, "public_path": str(PUBLIC_REL / rel),
                                    "exported_sha256": row["original_sha256"]})
            readme = ("# Continuation M5 — exact source and closed metadata snapshot\n\n"
                "Source and historical receipts retain exact bytes, paths and hashes; no redaction or automatic portability claim.\n"
                "evidence-continuation.zip contains lossless gzip members indexed with original and gzip SHA256.\n"
                "No public tensors, logits, states, model, binaries, credentials or private app memories are included.\n"
                "CPU8 raw archive is a separate LOCAL outputs deliverable (~1.585GB), not a GitHub download.\n"
                "Other large raw captures remain local; receipts/hashes are not payload availability.\n"
                "Q8-tail arrays were compared in memory and were not exported.\n"
                "Historical source-only/failed/partial statuses are preserved; final reports are separate explicit outputs metadata.\n"
                "This package grants no full-model/cache/EOS/600-agent/performance qualification.\n").encode()
            public_files["README.md"] = write_bytes(stage / "README.md", readme, budget)
            index = {"schema_version": 1, "namespace": str(PUBLIC_REL), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                     "selection_sha256": selection_sha, "final_allowlist_sha256": extra_sha,
                     "base_publication_commit": selection["base_publication_commit"],
                     "already_public_reconstruction_references": selection["already_public_reconstruction_references"],
                     "closure_checks": closures, "sources": source_rows, "evidence": evidence_rows,
                     "evidence_archive_sha256": public_files["evidence-continuation.zip"],
                     "public_raw_payloads_included": False, "heavy_local_evidence_not_public": selection["heavy_local_evidence_not_public"],
                     "qualification": "none inherited from packaging; see exact scoped receipts and final reports"}
            public_files["index.json"] = write_bytes(stage / "index.json", json_bytes(index), budget)
            tree_bytes = sum((stage / rel).stat().st_size for rel in public_files)
            require(tree_bytes < TREE_CAP, "GitHub namespace must remain below 7,000,000 bytes")
            outer = temp / ZIP_NAME
            outer_members = {str(PUBLIC_REL / rel): (stage / rel, digest) for rel, digest in public_files.items()}
            for row in pinned:
                if row["delivery"] == "output_only":
                    outer_members["outputs/" + row["relative_path"]] = (Path(row["original_path"]), row["original_sha256"])
            with BudgetFile(outer, budget) as raw:
                with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    for member, (path, digest) in sorted(outer_members.items()):
                        before, copied = identity(path), hashlib.sha256()
                        with open_regular(path, before) as source_stream, archive.open(member, "w") as target_stream:
                            for data in iter(lambda: source_stream.read(CHUNK), b""):
                                target_stream.write(data)
                                copied.update(data)
                        require(identity(path) == before and copied.hexdigest() == digest,
                                "Outer ZIP source changed during copy")
            with zipfile.ZipFile(outer) as archive:
                names = [info.filename for info in archive.infolist()]
                require(len(names) == len(set(names)) and set(names) == set(outer_members), "Outer ZIP member set differs")
                for member, (_, digest) in outer_members.items():
                    h = hashlib.sha256()
                    with archive.open(member) as stream:
                        for data in iter(lambda: stream.read(CHUNK), b""):
                            h.update(data)
                    require(h.hexdigest() == digest, "Outer ZIP bytes differ")
            for row in pinned + closures:
                require(identity(Path(row["original_path"])) == tuple(row["identity"]), "Frozen source/closure changed before commit")
            require(identity(SELECTION) == selection_id and identity(EXTRAS) == extra_id, "Allowlist changed before commit")
            result = {"schema_version": 1, "archive": ZIP_NAME, "archive_bytes": outer.stat().st_size,
                      "archive_sha256": sha(outer), "namespace": str(PUBLIC_REL), "namespace_bytes": tree_bytes,
                      "namespace_files_sha256": public_files, "selection_sha256": selection_sha,
                      "final_allowlist_sha256": extra_sha, "output_only_metadata":
                      [row for row in pinned if row["delivery"] == "output_only"],
                      "write_budget_cap_bytes": WRITE_BUDGET, "disk_reserve_bytes": DISK_RESERVE,
                      "public_raw_payloads_included": False, "remote_publication_performed": False}
            manifest = temp / MANIFEST_NAME
            write_bytes(manifest, json_bytes(result), budget)
            require(free_space() >= DISK_RESERVE, "Final disk reserve refused")
            publish_exclusive(stage, public_files, outer, manifest)
            return {k: value for k, value in result.items() if k not in {"namespace_files_sha256", "output_only_metadata"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Root-approved actual copy/archive under both locks")
    args = parser.parse_args()
    selection, selection_sha, selection_id = load_json(SELECTION)
    extra, extra_sha, extra_id = load_json(EXTRAS)
    if not args.execute:
        print(json.dumps({"status": "plan_only_no_source_hash_copy_compression_publication", "base_entries": len(selection["allowed_entries"]),
                          "extra_entries": len(extra.get("entries", [])), "namespace": str(PUBLIC_REL),
                          "approval_present": extra.get("approved_for_delivery") is True}))
        return
    print(json.dumps(execute(selection, selection_sha, selection_id, extra, extra_sha, extra_id)))


if __name__ == "__main__":
    main()
