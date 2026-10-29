"""Pinned GLM-5.3 Flash mixed-GGUF downloads with bounded RAM/disk reserves.

Uses no Hub cache and never copies an entire shard. A verified .partial file is
renamed in its destination directory. Merely calling preflight never downloads
weights or creates directories. This module does not claim inference readiness.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen

MODEL_ID = "ggml-org/GLM-5.3-Flash-GGUF"
REVISION = "119034bc3d9d76eb41f91508e6f9fb19fde42672"
FILES = (
    {"name": "GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf", "size": 9430112,
     "sha256": "22e3599ff1cac02e854269b610030800e319151e1ed01838d7087f20b187091d"},
    {"name": "GLM-5.3-Flash-Q2_K_S-00002-of-00002.gguf", "size": 110017922752,
     "sha256": "61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4"},
)
# ``name`` is the destination basename; ``path`` retains the Hub subdirectory.
# LFS sha256 values were read from the immutable Hub revision's blobs metadata.
AJ_MODEL_ID = "aj9o9/GLM-5.3-Flash-GGUF"
AJ_REVISION = "07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2"
AJ_FILES = (
    {"name": "GLM-5.3-Flash-AJ-IQ3_XXS-00001-of-00003.gguf",
     "path": "AJ-IQ3_XXS/GLM-5.3-Flash-AJ-IQ3_XXS-00001-of-00003.gguf",
     "size": 39842483616,
     "sha256": "0f98b1934ecae99b403a667e94a79ee3630a0d66c48cd719e5210da728603c80"},
    {"name": "GLM-5.3-Flash-AJ-IQ3_XXS-00002-of-00003.gguf",
     "path": "AJ-IQ3_XXS/GLM-5.3-Flash-AJ-IQ3_XXS-00002-of-00003.gguf",
     "size": 39974541184,
     "sha256": "abecd5f47280372702d6fa4a093ef3ffe149fae5a1d3f28e1f1feb8cc94fdf89"},
    {"name": "GLM-5.3-Flash-AJ-IQ3_XXS-00003-of-00003.gguf",
     "path": "AJ-IQ3_XXS/GLM-5.3-Flash-AJ-IQ3_XXS-00003-of-00003.gguf",
     "size": 32594142624,
     "sha256": "5651239c136bb26344e28001176a3f6b2dfe78664d0ab90e8762d013c9c349f8"},
)
PROFILE_CHOICES = ("Q2_K_S", "AJ-IQ3_XXS")

TOTAL_BYTES = sum(entry["size"] for entry in FILES)
GIB = 1024 ** 3
MIN_RESERVE_BYTES = 32 * GIB
METADATA_HEADROOM_BYTES = 64 * 1024 ** 2
IO_CHUNK_BYTES = 4 * 1024 ** 2


class DownloadSafetyError(ValueError):
    """A resource, identity or range check prevented a download."""


def _digest(path: Path, length: int | None = None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining is None or remaining:
            data = stream.read(IO_CHUNK_BYTES if remaining is None else min(IO_CHUNK_BYTES, remaining))
            if not data:
                if remaining:
                    raise DownloadSafetyError("Truncated local partial file")
                break
            digest.update(data)
            if remaining is not None:
                remaining -= len(data)
    return digest


def _regular(path: Path):
    if path.is_symlink() or path.exists() and not path.is_file():
        raise DownloadSafetyError(f"Refusing a non-regular or symbolic-link shard: {path}")


def verify_file(path: Path, entry: dict) -> None:
    path = Path(path)
    _regular(path)
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise DownloadSafetyError(f"File size mismatch: {path.name}")
    if _digest(path).hexdigest() != entry["sha256"]:
        raise DownloadSafetyError(f"SHA256 mismatch: {path.name}; existing file preserved")


def _profile(profile: str = "Q2_K_S") -> dict:
    if profile == "Q2_K_S":
        # Preserve the legacy constants/API, including test and caller overrides.
        return {"variant": profile, "model_id": MODEL_ID, "revision": REVISION,
                "files": FILES, "total_bytes": sum(e["size"] for e in FILES)}
    if profile == "AJ-IQ3_XXS":
        return {"variant": profile, "model_id": AJ_MODEL_ID, "revision": AJ_REVISION,
                "files": AJ_FILES, "total_bytes": sum(e["size"] for e in AJ_FILES)}
    raise DownloadSafetyError(f"Unknown checkpoint profile: {profile!r}")


def _identity(entry: dict, profile: str = "Q2_K_S") -> dict:
    spec = _profile(profile)
    return {"model_id": spec["model_id"], "revision": spec["revision"], **entry}


def _paths(root: Path, entry: dict):
    final = root / entry["name"]
    partial = root / (entry["name"] + ".partial")
    marker = root / (entry["name"] + ".partial.json")
    return final, partial, marker


def _partial_state(root: Path, entry: dict, profile: str = "Q2_K_S"):
    _, partial, marker = _paths(root, entry)
    _regular(partial)
    _regular(marker)
    if partial.exists() and partial.stat().st_nlink != 1:
        raise DownloadSafetyError(f"Refusing a hard-linked partial shard: {partial.name}")
    if not marker.exists():
        if partial.exists():
            raise DownloadSafetyError(f"Unidentified partial file preserved: {partial.name}")
        return 0, hashlib.sha256()
    try:
        with marker.open("rb") as marker_stream:
            marker_raw = marker_stream.read(4097)
        if len(marker_raw) > 4096:
            raise ValueError("partial marker exceeds bounded size")
        state = json.loads(marker_raw)
        for key, value in _identity(entry, profile).items():
            if state[key] != value:
                raise ValueError("identity differs")
        length = state["downloaded_bytes"]
        if type(length) is not int or not 0 <= length <= entry["size"]:
            raise ValueError("invalid partial length")
        if not partial.exists() or partial.stat().st_size < length:
            raise ValueError("partial is missing or truncated")
        digest = _digest(partial, length)
        if digest.hexdigest() != state["prefix_sha256"]:
            raise ValueError("partial prefix digest differs")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DownloadSafetyError(f"Partial identity verification failed: {partial.name}") from exc
    return length, digest


def _disk_path(root: Path) -> Path:
    probe = root
    while not probe.exists():
        if probe == probe.parent:
            raise DownloadSafetyError("Cannot locate the destination filesystem")
        probe = probe.parent
    if not probe.is_dir():
        raise DownloadSafetyError("Destination ancestor is not a directory")
    return probe


def _reserve(value: int) -> int:
    if type(value) is not int or value < MIN_RESERVE_BYTES:
        raise DownloadSafetyError("The disk reserve must be at least 32 GiB")
    return value


def _system_volume_free():
    return shutil.disk_usage(Path("/")).free if platform.system() == "Darwin" else None


def _shared_system_storage(probe: Path) -> bool | None:
    """Identify shared boot/destination storage without summing APFS capacities.

    Only disk_usage free blocks are used; no purgeable estimate is added. Separate
    APFS volumes may have different st_dev values while sharing one container.
    Unknown discovery is reported and treated conservatively by preflight.
    """
    if probe.stat().st_dev == Path("/").stat().st_dev:
        return True
    try:
        volumes = []
        for path in (Path("/"), probe):
            result = subprocess.run(["/usr/sbin/diskutil", "info", "-plist", str(path)],
                                    capture_output=True, check=True, timeout=5)
            if len(result.stdout) > 1024 * 1024:
                return None
            info = plistlib.loads(result.stdout)
            if not isinstance(info, dict):
                return None
            volumes.append(info)
        # Compare the same identifier kind on both volumes. A disk reference
        # such as "disk3" is not comparable to another response's UUID.
        for key in ("APFSContainerUUID", "APFSContainerReference"):
            identifiers = [info.get(key) for info in volumes]
            if all(isinstance(value, str) and value for value in identifiers):
                return identifiers[0] == identifiers[1]
        return None
    except (OSError, ValueError, subprocess.SubprocessError, plistlib.InvalidFileException):
        return None


def preflight(destination, *, reserve_bytes: int = MIN_RESERVE_BYTES,
              profile: str = "Q2_K_S") -> dict:
    """Inspect the actual destination filesystem and verify any reusable bytes.

    The 32 GiB reserve is an operational policy, not a macOS safety guarantee.
    The result reports both decimal bytes and binary GiB without conflating them.
    """
    _reserve(reserve_bytes)
    spec = _profile(profile)
    root = Path(destination).expanduser().resolve()
    probe = _disk_path(root)
    free = shutil.disk_usage(probe).free
    needed = 0
    records = []
    for entry in spec["files"]:
        final, partial, marker = _paths(root, entry)
        _regular(final)
        if final.exists():
            verify_file(final, entry)
            records.append({**entry, "status": "verified", "remaining_bytes": 0})
        else:
            length, _ = _partial_state(root, entry, profile)
            remaining = entry["size"] - length
            needed += remaining
            records.append({**entry, "status": "partial_verified" if length else "absent",
                            "verified_partial_bytes": length, "remaining_bytes": remaining})
    after = free - needed - METADATA_HEADROOM_BYTES
    system_free = _system_volume_free()
    shared = _shared_system_storage(probe) if system_free is not None else None
    # Do not add capacities or reserves for two APFS volumes sharing a pool.
    # If discovery is unavailable, subtract incoming bytes conservatively.
    system_after = (None if system_free is None else
                    system_free - (needed + METADATA_HEADROOM_BYTES if shared is not False else 0))
    system_allowed = system_after is None or system_after >= MIN_RESERVE_BYTES
    return {"model_id": spec["model_id"], "revision": spec["revision"], "variant": profile,
            "destination": str(root), "filesystem_probe": str(probe),
            "payload_bytes": spec["total_bytes"], "payload_gb": spec["total_bytes"] / 10**9,
            "payload_gib": spec["total_bytes"] / GIB,
            "free_bytes": free, "free_gb": free / 10**9, "free_gib": free / GIB,
            "remaining_download_bytes": needed,
            "temporary_duplicate_weight_bytes": 0,
            "metadata_headroom_bytes": METADATA_HEADROOM_BYTES,
            "reserve_bytes": reserve_bytes, "estimated_free_after_bytes": after,
            "download_allowed": after >= reserve_bytes and system_allowed, "files": records,
            "host_system": platform.system(),
            "macos_system_volume_free_bytes": system_free,
            "macos_system_volume_estimated_free_after_bytes": system_after,
            "destination_shares_system_storage": shared,
            "macos_system_volume_minimum_bytes": MIN_RESERVE_BYTES if system_free is not None else None,
            "macos_system_volume_allowed": system_allowed if system_free is not None else None,
            "vision": False, "mtp": False, "engine_validated": False}


def verify_remote_manifest(*, opener=None, profile: str = "Q2_K_S") -> None:
    """Compare immutable Hub LFS metadata against the checked-in manifest."""
    opener = opener or urlopen
    spec = _profile(profile)
    url = f"https://huggingface.co/api/models/{spec['model_id']}/revision/{spec['revision']}?blobs=true"
    with opener(Request(url), timeout=30) as response:
        raw = response.read(2 * 1024 ** 2 + 1)
    if len(raw) > 2 * 1024 ** 2:
        raise DownloadSafetyError("Hub metadata exceeds the bounded response size")
    data = json.loads(raw)
    if data.get("sha") != spec["revision"] or data.get("gated") or data.get("disabled"):
        raise DownloadSafetyError("Pinned public Hub revision differs or is unavailable")
    entries = {item["rfilename"]: item for item in data["siblings"]}
    for entry in spec["files"]:
        actual = entries.get(entry.get("path", entry["name"]), {})
        lfs = actual.get("lfs") or {}
        if (actual.get("size") != entry["size"] or lfs.get("size") != entry["size"]
                or lfs.get("sha256") != entry["sha256"]):
            raise DownloadSafetyError(f"Hub identity differs: {entry['name']}")


def _write_marker(marker: Path, entry: dict, length: int, digest,
                  profile: str = "Q2_K_S") -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=marker.parent,
                                         prefix=marker.name + ".tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({**_identity(entry, profile), "downloaded_bytes": length,
                       "prefix_sha256": digest.hexdigest()}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_checkpoint(destination, *, reserve_bytes: int = MIN_RESERVE_BYTES,
                        opener=None, progress=None, profile: str = "Q2_K_S") -> dict:
    """Download exact pinned files only after resource and identity checks pass.

    Every response must honor the requested range and total size. Interrupted
    bytes are reused only with their persisted prefix hash. Checksum failures
    preserve the partial for diagnosis and never produce a final GGUF filename.
    """
    opener = opener or urlopen
    spec = _profile(profile)
    report = preflight(destination, reserve_bytes=reserve_bytes, profile=profile)
    if not report["download_allowed"]:
        raise DownloadSafetyError("Insufficient destination or macOS system-volume space after preserving the 32 GiB minimum reserve")
    verify_remote_manifest(opener=opener, profile=profile)
    root = Path(report["destination"])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".glm53-download.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DownloadSafetyError("A download lock already exists; inspect it before retrying") from exc
    try:
        with os.fdopen(lock_fd, "w") as lock_stream:
            lock_stream.write(str(os.getpid()) + "\n")
        # Repeat under our lock: another invocation may have completed earlier.
        report = preflight(root, reserve_bytes=reserve_bytes, profile=profile)
        if not report["download_allowed"]:
            raise DownloadSafetyError("Free space changed before the download started")
        for entry in spec["files"]:
            final, partial, marker = _paths(root, entry)
            if final.exists():
                verify_file(final, entry)
                continue
            length, digest = _partial_state(root, entry, profile)
            if not partial.exists():
                with partial.open("xb"):
                    pass
                _write_marker(marker, entry, 0, digest, profile)
            with partial.open("r+b", buffering=0) as output:
                # Discard only our own uncommitted tail after a crash.
                output.truncate(length)
                output.seek(length)
                while length < entry["size"]:
                    system_free = _system_volume_free()
                    if system_free is not None and system_free < MIN_RESERVE_BYTES:
                        raise DownloadSafetyError("Download stopped to preserve the macOS system volume reserve")
                    if shutil.disk_usage(root).free < reserve_bytes + METADATA_HEADROOM_BYTES + IO_CHUNK_BYTES:
                        raise DownloadSafetyError("Download stopped to preserve the disk reserve")
                    end = min(entry["size"], length + IO_CHUNK_BYTES) - 1
                    if (system_free is not None and report["destination_shares_system_storage"] is not False
                            and system_free < MIN_RESERVE_BYTES + METADATA_HEADROOM_BYTES + end - length + 1):
                        raise DownloadSafetyError("Download stopped to preserve shared macOS/APFS space")
                    url = (f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{entry.get('path', entry['name'])}"
                           f"?download=true&range_start={length}")
                    request = Request(url, headers={"Range": f"bytes={length}-{end}"})
                    with opener(request, timeout=60) as response:
                        expected = f"bytes {length}-{end}/{entry['size']}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise DownloadSafetyError("Server did not honor the exact requested byte range")
                        count = end - length + 1
                        content_length = response.headers.get("Content-Length")
                        if content_length is not None and content_length != str(count):
                            raise DownloadSafetyError("Server range Content-Length differs")
                        if response.headers.get("Content-Encoding", "identity") != "identity":
                            raise DownloadSafetyError("Compressed HTTP range is not a raw GGUF payload")
                        payload = response.read(count + 1)
                        if len(payload) != count:
                            raise DownloadSafetyError("Truncated or excessive HTTP range payload")
                    written = output.write(payload)
                    if written != len(payload):
                        raise OSError("Short write to destination")
                    digest.update(payload)
                    length += written
                    output.flush()
                    os.fsync(output.fileno())
                    _write_marker(marker, entry, length, digest, profile)
                    if progress:
                        progress({"file": entry["name"], "completed_bytes": length,
                                  "total_bytes": entry["size"]})
            verify_file(partial, entry)
            if final.exists():
                raise DownloadSafetyError("Final filename appeared during download; preserving both files")
            partial.rename(final)
            marker.unlink()
        return preflight(root, reserve_bytes=reserve_bytes, profile=profile)
    finally:
        lock.unlink(missing_ok=True)
