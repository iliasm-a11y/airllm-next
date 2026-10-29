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


def fake_checkpoint(monkeypatch, payloads=(b"abcdefghij", b"klmnopqrstuv"),
                    profile="Q2_K_S"):
    files = tuple({"name": f"shard-{index}.gguf", "size": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}
                  for index, payload in enumerate(payloads))
    if profile == "AJ-IQ3_XXS":
        files = tuple({**entry, "path": "AJ-IQ3_XXS/" + entry["name"]} for entry in files)
    monkeypatch.setattr(download, "AJ_FILES" if profile == "AJ-IQ3_XXS" else "FILES", files)
    monkeypatch.setattr(download, "TOTAL_BYTES", sum(len(p) for p in payloads))
    monkeypatch.setattr(download, "IO_CHUNK_BYTES", 4)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(free=1000 * download.GIB))
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if "/api/models/" in request.full_url:
            data = {"sha": download._profile(profile)["revision"], "gated": False, "siblings": [
                {"rfilename": entry.get("path", entry["name"]), "size": entry["size"],
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


# AJ is an explicit additive profile; the legacy constants and default remain Q2.
def test_aj_profile_has_three_immutable_shards_and_hub_paths():
    spec = download._profile("AJ-IQ3_XXS")
    assert spec["model_id"] == "aj9o9/GLM-5.3-Flash-GGUF"
    assert spec["revision"] == "07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2"
    assert spec["total_bytes"] == 112411167424
    assert [e["size"] for e in spec["files"]] == [39842483616, 39974541184, 32594142624]
    for index, entry in enumerate(spec["files"], 1):
        assert entry["name"].endswith(f"{index:05d}-of-00003.gguf")
        assert entry["path"] == "AJ-IQ3_XXS/" + entry["name"]
        assert len(entry["sha256"]) == 64
    assert download._profile()["files"] is download.FILES
    manifest = json.loads((Path(__file__).parents[1] / "research/ronin_aj_iq3xxs/checkpoint-manifest.json").read_text())
    assert manifest["model_id"] == spec["model_id"]
    assert manifest["revision"] == spec["revision"]
    assert manifest["files"] == list(spec["files"])
    assert manifest["total_bytes"] == spec["total_bytes"]


def test_unknown_profile_is_rejected_before_filesystem_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: pytest.fail("unexpected disk probe"))
    with pytest.raises(download.DownloadSafetyError, match="Unknown checkpoint profile"):
        download.preflight(tmp_path / "new", profile="AJ-IQ2_XXS")
    assert not (tmp_path / "new").exists()


def test_aj_dry_run_uses_exact_bytes_and_creates_nothing(tmp_path, monkeypatch):
    free = 140_000_000_000
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    target = tmp_path / "absent"
    report = download.preflight(target, profile="AJ-IQ3_XXS")
    assert report["remaining_download_bytes"] == 112411167424
    assert report["estimated_free_after_bytes"] == free - 112411167424 - 64 * 1024**2
    assert report["payload_gb"] == 112411167424 / 10**9
    assert report["payload_gib"] == 112411167424 / 1024**3
    assert not report["download_allowed"]
    assert not target.exists()


def test_aj_download_uses_all_three_remote_paths_without_cache_or_merge(tmp_path, monkeypatch):
    files, payloads, requests, opener = fake_checkpoint(
        monkeypatch, (b"a"*10, b"b"*12, b"c"*9), profile="AJ-IQ3_XXS")
    report = download.download_checkpoint(tmp_path, profile="AJ-IQ3_XXS", opener=opener)
    assert report["variant"] == "AJ-IQ3_XXS"
    assert [e["status"] for e in report["files"]] == ["verified"]*3
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(e["name"] for e in files)
    for entry, payload in zip(files, payloads):
        assert (tmp_path / entry["name"]).read_bytes() == payload
        assert any(entry["path"] in request.full_url for request in requests)
    assert all(download.AJ_REVISION in request.full_url for request in requests)
    assert download.REVISION == "119034bc3d9d76eb41f91508e6f9fb19fde42672"


def test_aj_resume_preserves_profile_identity_and_reuses_verified_prefix(tmp_path, monkeypatch):
    files, payloads, requests, opener = fake_checkpoint(
        monkeypatch, (b"a"*10, b"b"*12, b"c"*9), profile="AJ-IQ3_XXS")
    _, partial, marker = download._paths(tmp_path, files[0])
    partial.write_bytes(payloads[0][:4])
    download._write_marker(marker, files[0], 4, hashlib.sha256(payloads[0][:4]), "AJ-IQ3_XXS")
    download.download_checkpoint(tmp_path, profile="AJ-IQ3_XXS", opener=opener)
    first = next(r for r in requests if "/api/models/" not in r.full_url)
    assert first.get_header("Range") == "bytes=4-7"


def test_aj_rejects_partial_recorded_for_other_repository(tmp_path, monkeypatch):
    files, payloads, _, opener = fake_checkpoint(
        monkeypatch, (b"a"*10, b"b"*12, b"c"*9), profile="AJ-IQ3_XXS")
    _, partial, marker = download._paths(tmp_path, files[0])
    partial.write_bytes(payloads[0][:4])
    download._write_marker(marker, files[0], 4, hashlib.sha256(payloads[0][:4]))
    with pytest.raises(download.DownloadSafetyError, match="identity verification"):
        download.download_checkpoint(tmp_path, profile="AJ-IQ3_XXS", opener=opener)
    assert partial.read_bytes() == payloads[0][:4]


@pytest.mark.parametrize("headers", [{"Content-Length": "5"}, {"Content-Encoding": "gzip"}])
def test_http_range_length_and_encoding_are_validated_before_body(tmp_path, monkeypatch, headers):
    _, _, _, opener = fake_checkpoint(monkeypatch)
    class UnreadableResponse(Response):
        def read(self, *args):
            pytest.fail("HTTP payload read before range header validation")
    def broken(request, timeout):
        if "/api/models/" in request.full_url:
            return opener(request, timeout)
        return UnreadableResponse(b"never read", 206,
                                  {"Content-Range": "bytes 0-3/10", **headers})
    with pytest.raises(download.DownloadSafetyError):
        download.download_checkpoint(tmp_path, opener=broken)


def test_ignored_http_range_is_not_read(tmp_path, monkeypatch):
    _, _, _, opener = fake_checkpoint(monkeypatch)
    class UnreadableResponse(Response):
        def read(self, *args):
            pytest.fail("server's full checkpoint body must never be read")
    def broken(request, timeout):
        if "/api/models/" in request.full_url:
            return opener(request, timeout)
        return UnreadableResponse(b"never read", 200)
    with pytest.raises(download.DownloadSafetyError, match="exact requested byte range"):
        download.download_checkpoint(tmp_path, opener=broken)


def test_aj_remote_etag_is_not_accepted_instead_of_lfs_sha256(tmp_path, monkeypatch):
    files, _, _, _ = fake_checkpoint(monkeypatch, (b"a", b"b", b"c"), profile="AJ-IQ3_XXS")
    def wrong_identity(request, timeout):
        return Response(json.dumps({"sha": download.AJ_REVISION, "siblings": [
            {"rfilename": e["path"], "size": e["size"], "etag": e["sha256"],
             "lfs": {"size": e["size"], "oid": e["sha256"]}} for e in files]}).encode())
    target = tmp_path / "new"
    with pytest.raises(download.DownloadSafetyError, match="Hub identity differs"):
        download.download_checkpoint(target, opener=wrong_identity, profile="AJ-IQ3_XXS")
    assert not target.exists()


@pytest.mark.parametrize("shared", [True, None])
def test_shared_or_unknown_apfs_pool_accounts_for_incoming_bytes(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download, "_shared_system_storage", lambda _: shared)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=130_000_000_000 if path == Path("/") else 300_000_000_000))
    report = download.preflight(tmp_path, profile="AJ-IQ3_XXS")
    assert report["estimated_free_after_bytes"] > download.MIN_RESERVE_BYTES
    assert report["macos_system_volume_free_bytes"] > download.MIN_RESERVE_BYTES
    assert report["macos_system_volume_estimated_free_after_bytes"] < download.MIN_RESERVE_BYTES
    assert not report["download_allowed"]
    assert report["destination_shares_system_storage"] is shared


def test_separate_storage_does_not_subtract_payload_from_boot_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(download.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(download, "_shared_system_storage", lambda _: False)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=40*download.GIB if path == Path("/") else 300*download.GIB))
    report = download.preflight(tmp_path, profile="AJ-IQ3_XXS")
    assert report["download_allowed"]
    assert report["macos_system_volume_estimated_free_after_bytes"] == 40*download.GIB


@pytest.mark.parametrize("volumes,expected", [
    ([{"APFSContainerReference": "disk3", "APFSContainerUUID": "same-pool"},
      {"APFSContainerUUID": "same-pool"}], True),
    ([{"APFSContainerReference": "disk3"},
      {"APFSContainerUUID": "uncomparable-uuid"}], None),
    ([{"APFSContainerUUID": "boot-pool"},
      {"APFSContainerUUID": "external-pool"}], False),
])
def test_apfs_discovery_compares_a_common_identifier_kind(monkeypatch, volumes, expected):
    import plistlib
    boot_device = Path("/").stat().st_dev
    class DifferentFilesystem:
        def stat(self):
            return SimpleNamespace(st_dev=boot_device + 1)
        def __str__(self):
            return "/mounted-volume"
    replies = iter(volumes)
    monkeypatch.setattr(download.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=plistlib.dumps(next(replies))))
    assert download._shared_system_storage(DifferentFilesystem()) is expected
