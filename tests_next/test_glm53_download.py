"""Real pinned identities, bounded transfers and interruption/storage failures."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from airllm.nextgen import glm53_download as download


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}


def fake_checkpoint(monkeypatch, payloads=(b"abcdefghij", b"klmnopqrstuv")):
    files = tuple({"name": f"shard-{index}.gguf", "size": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}
                  for index, payload in enumerate(payloads))
    monkeypatch.setattr(download, "FILES", files)
    monkeypatch.setattr(download, "TOTAL_BYTES", sum(len(p) for p in payloads))
    monkeypatch.setattr(download, "IO_CHUNK_BYTES", 4)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(free=1000 * download.GIB))
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if "/api/models/" in request.full_url:
            data = {"sha": download.REVISION, "gated": False, "siblings": [
                {"rfilename": entry["name"], "size": entry["size"],
                 "lfs": {"size": entry["size"], "sha256": entry["sha256"]}} for entry in files]}
            return Response(json.dumps(data).encode())
        entry = next(e for e in files if e["name"] in request.full_url)
        index = files.index(entry)
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        return Response(payloads[index][start:end+1], 206,
                        {"Content-Range": f"bytes {start}-{end}/{entry['size']}"})

    return files, payloads, requests, opener


def test_official_variant_is_exact_and_excludes_optional_sidecars():
    assert download.REVISION == "119034bc3d9d76eb41f91508e6f9fb19fde42672"
    assert download.TOTAL_BYTES == 110027352864
    assert len(download.FILES) == 2
    assert all("Q2_K_S" in e["name"] for e in download.FILES)
    assert download.FILES[0]["sha256"] == "22e3599ff1cac02e854269b610030800e319151e1ed01838d7087f20b187091d"
    assert download.FILES[1]["sha256"] == "61ed849acb67dc7832eda82e9fad0f0257edcd20ced1fc7640e5e329ed6951b4"


def test_preflight_refuses_old_125_decimal_gb_budget_and_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(free=125450000000))
    target = tmp_path / "external" / "model"
    result = download.preflight(target)
    assert not result["download_allowed"]
    assert result["estimated_free_after_bytes"] == 15355538272
    assert not target.exists()
    assert result["temporary_duplicate_weight_bytes"] == 0


@pytest.mark.parametrize("reserve", [0, 31*download.GIB, -1, 32.0*download.GIB])
def test_minimum_reserve_cannot_be_disabled(tmp_path, reserve):
    with pytest.raises(download.DownloadSafetyError, match="at least 32 GiB"):
        download.preflight(tmp_path, reserve_bytes=reserve)


def test_actual_destination_filesystem_is_checked(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: observed.append(path) or SimpleNamespace(free=1000*download.GIB))
    download.preflight(tmp_path / "absent" / "model")
    assert observed == [tmp_path]


def test_download_no_cache_no_duplicate_and_reuse_is_verified(tmp_path, monkeypatch):
    files, payloads, requests, opener = fake_checkpoint(monkeypatch)
    result = download.download_checkpoint(tmp_path, opener=opener)
    assert all(e["status"] == "verified" for e in result["files"])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(e["name"] for e in files)
    for entry, payload in zip(files, payloads):
        assert (tmp_path / entry["name"]).read_bytes() == payload
    weight_requests = [r for r in requests if "/api/models/" not in r.full_url]
    assert len(weight_requests) == 6
    assert all(int(r.get_header("Range").split("-")[1])-int(r.get_header("Range").split("=")[1].split("-")[0])+1 <= 4 for r in weight_requests)
    requests.clear()
    download.download_checkpoint(tmp_path, opener=opener)
    assert all("/api/models/" in r.full_url for r in requests)


def test_resume_reuses_only_verified_prefix_and_discards_own_crash_tail(tmp_path, monkeypatch):
    files, payloads, requests, opener = fake_checkpoint(monkeypatch)
    entry = files[0]
    _, partial, marker = download._paths(tmp_path, entry)
    partial.write_bytes(payloads[0][:4] + b"uncommitted tail")
    download._write_marker(marker, entry, 4, hashlib.sha256(payloads[0][:4]))
    download.download_checkpoint(tmp_path, opener=opener)
    first = next(r for r in requests if "/api/models/" not in r.full_url)
    assert first.get_header("Range") == "bytes=4-7"
    assert (tmp_path / entry["name"]).read_bytes() == payloads[0]


def test_foreign_or_corrupt_partial_is_preserved(tmp_path, monkeypatch):
    files, payloads, _, opener = fake_checkpoint(monkeypatch)
    _, partial, marker = download._paths(tmp_path, files[0])
    partial.write_bytes(b"foreign")
    with pytest.raises(download.DownloadSafetyError, match="Unidentified"):
        download.download_checkpoint(tmp_path, opener=opener)
    assert partial.read_bytes() == b"foreign"
    download._write_marker(marker, files[0], 4, hashlib.sha256(b"abcd"))
    with pytest.raises(download.DownloadSafetyError, match="identity verification"):
        download.download_checkpoint(tmp_path, opener=opener)
    assert partial.read_bytes() == b"foreign"


def test_full_size_corruption_and_symlink_are_not_reused(tmp_path, monkeypatch):
    files, _, _, opener = fake_checkpoint(monkeypatch)
    target = tmp_path / files[0]["name"]
    target.write_bytes(b"x" * files[0]["size"])
    with pytest.raises(download.DownloadSafetyError, match="SHA256 mismatch"):
        download.download_checkpoint(tmp_path, opener=opener)
    assert target.read_bytes() == b"x" * files[0]["size"]
    target.unlink()
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(download.DownloadSafetyError, match="symbolic-link"):
        download.preflight(tmp_path)


@pytest.mark.parametrize("status,range_value,payload", [(200,"bytes 0-3/10",b"abcd"), (206,"bytes 1-4/10",b"abcd"), (206,"bytes 0-3/10",b"abc"), (206,"bytes 0-3/10",b"abcde")])
def test_ignored_ranges_truncation_and_excess_payload_fail_closed(tmp_path, monkeypatch, status, range_value, payload):
    files, _, _, opener = fake_checkpoint(monkeypatch)
    def broken(request, timeout):
        if "/api/models/" in request.full_url:
            return opener(request, timeout)
        return Response(payload, status, {"Content-Range": range_value})
    with pytest.raises(download.DownloadSafetyError):
        download.download_checkpoint(tmp_path, opener=broken)
    assert not (tmp_path / files[0]["name"]).exists()
    assert not (tmp_path / ".glm53-download.lock").exists()


def test_disk_pressure_stops_before_any_weight_request(tmp_path, monkeypatch):
    _, _, requests, opener = fake_checkpoint(monkeypatch)
    calls = iter([1000*download.GIB, 1000*download.GIB, 32*download.GIB])
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(calls)))
    with pytest.raises(download.DownloadSafetyError, match="preserve the disk reserve"):
        download.download_checkpoint(tmp_path, opener=opener)
    assert all("/api/models/" in r.full_url for r in requests)


def test_changed_remote_manifest_does_not_create_destination(tmp_path, monkeypatch):
    fake_checkpoint(monkeypatch)
    def broken(request, timeout):
        return Response(json.dumps({"sha": "another-revision", "siblings": []}).encode())
    target = tmp_path / "new"
    with pytest.raises(download.DownloadSafetyError, match="Pinned public"):
        download.download_checkpoint(target, opener=broken)
    assert not target.exists()


def test_external_ssd_does_not_bypass_macos_boot_volume_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=20*download.GIB if path == Path("/") else 1000*download.GIB))
    report = download.preflight(tmp_path)
    assert not report["download_allowed"]
    assert report["macos_system_volume_free_bytes"] == 20*download.GIB
    assert report["macos_system_volume_allowed"] is False


def test_hardlinked_partial_cannot_modify_another_file(tmp_path, monkeypatch):
    import os
    files, payloads, _, _ = fake_checkpoint(monkeypatch)
    original = tmp_path / "personal-file"
    original.write_bytes(b"preserve")
    _, partial, _ = download._paths(tmp_path, files[0])
    os.link(original, partial)
    with pytest.raises(download.DownloadSafetyError, match="hard-linked"):
        download.preflight(tmp_path)
    assert original.read_bytes() == b"preserve"


def test_incoming_checksum_mismatch_never_becomes_final_shard(tmp_path, monkeypatch):
    files, _, _, opener = fake_checkpoint(monkeypatch)
    def corrupt(request, timeout):
        result = opener(request, timeout)
        if "/api/models/" in request.full_url:
            return result
        return Response(b"x" * len(result.getvalue()), result.status, result.headers)
    with pytest.raises(download.DownloadSafetyError, match="SHA256 mismatch"):
        download.download_checkpoint(tmp_path, opener=corrupt)
    assert not (tmp_path / files[0]["name"]).exists()
    assert (tmp_path / (files[0]["name"] + ".partial")).exists()


def test_cooperative_download_lock_preserves_existing_lock(tmp_path, monkeypatch):
    _, _, requests, opener = fake_checkpoint(monkeypatch)
    lock = tmp_path / ".glm53-download.lock"
    lock.write_text("existing process\n")
    with pytest.raises(download.DownloadSafetyError, match="lock already exists"):
        download.download_checkpoint(tmp_path, opener=opener)
    assert lock.read_text() == "existing process\n"
    assert all("/api/models/" in r.full_url for r in requests)


def test_partial_metadata_read_has_a_bound(tmp_path, monkeypatch):
    files, _, _, _ = fake_checkpoint(monkeypatch)
    _, partial, marker = download._paths(tmp_path, files[0])
    partial.write_bytes(b"")
    marker.write_bytes(b" " * 4097)
    with pytest.raises(download.DownloadSafetyError, match="identity verification"):
        download.preflight(tmp_path)
