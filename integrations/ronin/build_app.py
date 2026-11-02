#!/usr/bin/env python3
"""Build a separate RONIN GLM native app without replacing an installed app."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def absolute_python_path(path):
    """Keep a venv launcher symlink; resolving it selects the base interpreter."""
    result = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not result.is_file() or not os.access(result, os.X_OK):
        raise ValueError("Python executable is missing or not executable: " + str(result))
    return result


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_monitor_manifest(resources):
    """Refresh destination hashes after signing a bundled Mach-O extension."""
    path = resources / "monitor-manifest.json"
    manifest = json.loads(path.read_text())
    for item in manifest.get("files", []):
        file = resources / item["path"]
        item.update(bundled_sha256=file_hash(file), bundled_bytes=file.stat().st_size)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def bundle_monitor(package_path, resources):
    """Copy only psutil and its license, without importing its native module."""
    manifest = {"schema_version": 1, "bundled": False, "files": [],
                "selection": "interpreter psutil if available; otherwise existing stdlib/subprocess fallback",
                "guard_thresholds_changed": False}
    if package_path is not None:
        source = package_path.expanduser().resolve()
        if source.name != "psutil" or not (source / "__init__.py").is_file() or not list(source.glob("_psutil_osx*.so")):
            raise ValueError("--monitor-package must identify the installed macOS psutil package directory")
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Monitor package symlinks are not copied")
        found = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)', (source / "__init__.py").read_text(), re.MULTILINE)
        version = found[1] if found else None
        if not version:
            raise ValueError("Cannot determine psutil source version without importing it")
        distribution = source.parent / ("psutil-" + version + ".dist-info")
        destination = resources / "air_llm/psutil"
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest.update(bundled=True, name="psutil", version=version, source_package=str(source),
                        selection="bundled psutil, matching the benchmark monitor implementation",
                        binary_compatibility="Use a wheel compatible with the selected external Python and macOS architecture; copied files are not executed by the builder",
                        license_bundled=False, distribution_metadata={})
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                original = source / file.relative_to(destination)
                manifest["files"].append({"path": str(file.relative_to(resources)), "source_sha256": file_hash(original)})
        for name in ("METADATA", "WHEEL"):
            file = distribution / name
            if file.is_file():
                manifest["distribution_metadata"][name] = {"source_path": str(file), "sha256": file_hash(file)}
                if name == "WHEEL":
                    manifest["wheel_tags"] = re.findall(r'^Tag:\s*(.+)$', file.read_text(), re.MULTILINE)
        license_file = distribution / "LICENSE"
        if license_file.is_file():
            target = resources / "third-party/psutil/LICENSE"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(license_file, target)
            manifest["license_bundled"] = True
            manifest["files"].append({"path": str(target.relative_to(resources)), "source_sha256": file_hash(license_file)})
    (resources / "monitor-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return refresh_monitor_manifest(resources)


def config_path(value, directory):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (directory / path).absolute()


def verify_runtime_artifacts(root, manifest, binary, expected_sha256=None):
    """Verify pristine inputs before copying or signing can change their hashes."""
    if manifest.get("status") != "built" or manifest.get("source_revision") != "3f8af15a76ac8675812e856da45454bafc90f12c":
        raise ValueError("Runtime must be a completed build of the pinned source revision")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 512:
        raise ValueError("Invalid runtime artifact list")
    bin_dir = root / "bin"
    if bin_dir.is_symlink() or not bin_dir.is_dir():
        raise ValueError("Runtime bin must be an ordinary directory")
    names, hashes, links, total = set(), {}, [], 0
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Invalid runtime artifact record")
        parts = row["path"].split("/")
        if len(parts) != 2 or parts[0] != "bin" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", parts[1]):
            raise ValueError("Runtime artifacts must be simple bin/ filenames")
        name = parts[1]
        if name in names: raise ValueError("Duplicate runtime artifact")
        names.add(name); path = bin_dir / name
        if "symlink" in row:
            target = row["symlink"]
            if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", target):
                raise ValueError("Runtime symlink target must stay in bin")
            if not path.is_symlink() or os.readlink(path) != target or "sha256" in row:
                raise ValueError("Runtime symlink differs from original manifest")
            links.append((path, target))
        else:
            size, digest = row.get("bytes"), row.get("sha256")
            if type(size) is not int or not 0 <= size <= 512 * 1024**2 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid original runtime size or hash")
            total += size
            if total > 2 * 1024**3 or path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                raise ValueError("Runtime artifact size/type mismatch: " + name)
            actual = file_hash(path)
            if actual != digest: raise ValueError("Runtime artifact hash mismatch: " + name)
            hashes[name] = actual
    if {path.name for path in bin_dir.iterdir()} != names:
        raise ValueError("Runtime bin contains missing or undeclared artifacts")
    for path, target in links:
        if target not in names or path.resolve(strict=True).parent != bin_dir.resolve():
            raise ValueError("Runtime symlink escapes declared bin artifacts")
    if binary.parent.resolve() != bin_dir.resolve() or binary.name not in names or binary.is_symlink():
        raise ValueError("Runtime executable is not a declared regular artifact")
    if expected_sha256 is not None and hashes[binary.name] != expected_sha256:
        raise ValueError("Configured expected_runtime_sha256 differs from pristine executable")
    return {"artifact_count": len(entries), "verified_bytes": total, "binary_sha256": hashes[binary.name]}


def refresh_runtime_manifest(resources):
    """Rehash the embedded runtime after signing; retain original build evidence."""
    root = resources / "native-runtime"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    packaging = manifest.get("bundle_packaging", {})
    if packaging.get("source_artifacts_verified") is not True or packaging.get("copied_artifacts_verified") is not True:
        raise ValueError("Refusing to refresh runtime hashes without verified pristine source and copy")
    manifest.setdefault("pre_bundle_artifacts", manifest.get("artifacts", []))
    artifacts = []
    binary_root = root / "bin"
    for path in sorted(binary_root.iterdir()):
        item = {"path": str(path.relative_to(root))}
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute() or not path.resolve().is_relative_to(binary_root.resolve()):
                raise ValueError("Embedded runtime symlink escapes bin: " + str(path))
            item["symlink"] = target
        elif path.is_file():
            item.update(bytes=path.stat().st_size, sha256=file_hash(path))
        else:
            raise ValueError("Unexpected embedded runtime artifact: " + str(path))
        artifacts.append(item)
    manifest["artifacts"] = artifacts
    manifest.setdefault("bundle_packaging", {})["artifact_hashes_refreshed_after_packaging"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config_file = resources / "runtime-config.json"
    config = json.loads(config_file.read_text())
    binary = config_path(config["runtime_binary"], resources)
    if not binary.resolve().is_relative_to(binary_root.resolve()):
        raise ValueError("Embedded runtime configuration points outside bin")
    config["expected_runtime_sha256"] = file_hash(binary)
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    return manifest


def bundle_runtime(config, resources, config_directory):
    """Copy native code, licenses and a small verification receipt; never weights."""
    config = dict(config)
    binary = config_path(config["runtime_binary"], config_directory)
    root = binary.parent.parent
    manifest_path = config_path(config.get("runtime_manifest_file", root / "manifest.json"), config_directory)
    if manifest_path.resolve() != (root / "manifest.json").resolve() or not binary.is_file():
        raise ValueError("Embedded runtime binary and manifest must share the runtime root")
    manifest_bytes = manifest_path.read_bytes()
    if len(manifest_bytes) > 2 * 1024**2: raise ValueError("Runtime build manifest exceeds metadata size limit")
    manifest = json.loads(manifest_bytes)
    verified = verify_runtime_artifacts(root, manifest, binary, config.get("expected_runtime_sha256"))
    receipt = config_path(config["verification_file"], config_directory)
    if receipt.stat().st_size > 4 * 1024**2:
        raise ValueError("Model verification receipt exceeds metadata-only size limit")
    receipt_data = json.loads(receipt.read_text())
    if not isinstance(receipt_data, dict) or not isinstance(receipt_data.get("files"), list):
        raise ValueError("A model verification metadata receipt is required")
    destination = resources / "native-runtime"
    destination.mkdir()
    shutil.copytree(root / "bin", destination / "bin", symlinks=True)
    shutil.copytree(root / "licenses", destination / "licenses", symlinks=True)
    (destination / "manifest.json").write_bytes(manifest_bytes)
    if (root / "build.log").is_file():
        shutil.copy2(root / "build.log", destination / "build.log")
    verify_runtime_artifacts(destination, manifest, destination / "bin" / binary.name, config.get("expected_runtime_sha256"))
    shutil.copy2(receipt, resources / "model-verification.json")
    copied_manifest = json.loads(manifest_bytes)
    copied_manifest["bundle_packaging"] = {"source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_runtime_directory": str(root), "weights_copied": False,
        "source_artifacts_verified": True, "copied_artifacts_verified": True,
        "pristine_binary_sha256": verified["binary_sha256"]}
    (destination / "manifest.json").write_text(json.dumps(copied_manifest, indent=2) + "\n")
    config.update(runtime_binary="native-runtime/bin/" + binary.name,
                  runtime_manifest_file="native-runtime/manifest.json", verification_file="model-verification.json")
    config.pop("state_dir", None)
    (resources / "runtime-config.json").write_text(json.dumps(config, indent=2) + "\n")
    refresh_runtime_manifest(resources)
    return json.loads((resources / "runtime-config.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--monitor-package", type=Path, help="Installed macOS psutil package directory to bundle, with its available license")
    parser.add_argument("--embed-runtime", action="store_true", help="Embed native runtime, licenses and model verification metadata; weights remain external")
    parser.add_argument("--data-home", type=Path, default=Path.home() / "Library/Application Support/RONIN GLM")
    args = parser.parse_args()
    python_path = absolute_python_path(args.python)
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict) or not all(config.get(k) for k in ("model_path", "runtime_binary")):
        parser.error("config requires model_path and runtime_binary")
    app = args.destination.resolve() / "RONIN GLM.app"
    if app.exists():
        parser.error("destination app already exists; choose an empty output directory")
    resources = app / "Contents/Resources"
    macos = app / "Contents/MacOS"
    backend = resources / "backend"
    backend.mkdir(parents=True)
    macos.mkdir()
    integration = resources / "integrations/ronin"
    shutil.copytree(ROOT, integration, ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"))
    shutil.copytree(REPO / "air_llm", resources / "air_llm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    monitor = bundle_monitor(args.monitor_package, resources)
    if args.embed_runtime:
        config = bundle_runtime(config, resources, args.config.expanduser().absolute().parent)
    else:
        for key in ("runtime_binary", "runtime_manifest_file", "verification_file", "state_dir"):
            if config.get(key): config[key] = str(config_path(config[key], args.config.expanduser().absolute().parent))
    (resources / "runtime-config.json").write_text(json.dumps(config, indent=2) + "\n")
    launcher = backend / "ronin-backend"
    launcher.write_text(
        "#!/bin/sh\nset -eu\n"
        'resources=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\n"
        f"exec {shlex.quote(str(python_path))} "
        '"$resources/integrations/ronin/ronin_glm.py" stdio '
        '--config "$resources/runtime-config.json" '
        f"--home {shlex.quote(str(args.data_home.expanduser().resolve()))}\n"
    )
    launcher.chmod(0o755)
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")
    subprocess.run(["swiftc", "-sdk", str(sdk), "-swift-version", "5", "-parse-as-library", "-O",
                    str(ROOT / "native/Ronin.swift"), "-o", str(macos / "RONIN"),
                    "-framework", "SwiftUI", "-framework", "AppKit",
                    "-framework", "UniformTypeIdentifiers", "-framework", "Security"], check=True)
    plist = {"CFBundleName": "RONIN GLM", "CFBundleDisplayName": "RONIN GLM",
             "CFBundleIdentifier": "local.ronin.glm.prototype", "CFBundleExecutable": "RONIN",
             "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "0.2.0",
             "CFBundleVersion": "2", "LSMinimumSystemVersion": "26.0", "NSHighResolutionCapable": True}
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps(plist))
    if args.embed_runtime:
        embedded = resources / "native-runtime"
        verify_runtime_artifacts(embedded, json.loads((embedded / "manifest.json").read_text()),
            config_path(config["runtime_binary"], resources), config.get("expected_runtime_sha256"))
    subprocess.run(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    if args.embed_runtime:
        refresh_runtime_manifest(resources)
        config = json.loads((resources / "runtime-config.json").read_text())
    monitor = refresh_monitor_manifest(resources)
    # The manifest changes the outer resource seal; nested code keeps its signature.
    subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(app)], check=True)
    subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    print(json.dumps({"app": str(app), "python": str(python_path), "monitor": monitor,
                      "native_runtime": config["runtime_binary"], "signature": "ad-hoc", "notarized": False}))


if __name__ == "__main__":
    main()
