#!/usr/bin/env python3
"""Build the pinned private GLM runtime and diagnostic without modifying source.

Builds the default-off V2 Metal, residency and profiling research patches. Experiments
remain disabled unless their explicit AIRLLM_* process flags are set; the dense
column limit remains one unless an experiment explicitly requests 2, 4 or 8. This command does not load a model or benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

REVISION = "3f8af15a76ac8675812e856da45454bafc90f12c"
UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PATCHES = tuple(RESEARCH / "glm53_m5/patches" / name for name in (
    "llama-mmap-demand.patch", "llama-expert-prefetch.patch", "llama-dense-prefetch.patch", "llama-dense-prefetch-columns.patch"))
PATCHES += tuple(RESEARCH / "glm53_gpu/patches" / name for name in (
    "llama-layer-local-hc.patch", "llama-moe-cache-split.patch"))
PATCHES += tuple(RESEARCH / "glm53_gpu_v2/patches" / name for name in (
    "llama-q8-cpuarith.patch", "llama-metal-command-timing.patch", "llama-hc-kda-trace.patch",
    "llama-q8-selective.patch", "llama-metal-mapped-cpu-read.patch", "llama-q8-shared-activation.patch", "llama-expert-drop.patch"))
PATCHES += tuple(ROOT / "patches" / name for name in ("llama-mmap-nocache.patch", "llama-dense-lookahead.patch"))
PROBE = ROOT / "native_probe.cpp"
AUXILIARIES = tuple(ROOT / name for name in ("state_capture.inc", "source_capture.inc", "self_vm.inc"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def build(source, destination, jobs=3, dense_lock_experiment=False):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This qualified build recipe requires Apple Silicon macOS")
    if type(jobs) is not int or not 1 <= jobs <= (os.cpu_count() or 1):
        raise ValueError("Invalid build parallelism")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty runtime destination")
    if source is not None:
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("--source must name an existing Git checkout")
        # Do not create output anywhere inside the source whose preservation was requested.
        if destination == source or destination.is_relative_to(source):
            raise ValueError("The destination must be outside the source checkout")
    patches = PATCHES + ((ROOT / 'patches/llama-dense-lock.patch',) if dense_lock_experiment else ())
    for path in (*patches, PROBE, *AUXILIARIES):
        if not path.is_file():
            raise ValueError("Missing build input: " + str(path))
    frozen_patches = [(path.name, path.read_bytes()) for path in patches]
    frozen_probe = PROBE.read_bytes()
    frozen_aux = [(path.name,path.read_bytes()) for path in AUXILIARIES]
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "build.log"
    blocked_prefixes = ("LLAMA_ARG_", "AIRLLM_", "DYLD_", "CMAKE_", "GGML_")
    blocked_names = {"LLAMA_API_KEY", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}
    removed_environment_names = sorted(k for k in os.environ
        if k in blocked_names or k.startswith(blocked_prefixes))
    env = {k: v for k, v in os.environ.items() if k not in removed_environment_names}
    commands = []
    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "source_url": UPSTREAM, "source_revision": REVISION, "status": "building",
                "source_checkout_modified": False, "jobs": jobs,
                "dense_lock_experiment_built": bool(dense_lock_experiment),
                "removed_environment_names": removed_environment_names,
                "platform": {"system": platform.system(), "release": platform.release(),
                             "machine": platform.machine()},
                "patches": [{"name": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in frozen_patches],
                "probe_source_sha256": hashlib.sha256(frozen_probe).hexdigest(),
                "probe_auxiliaries": [{"name":n,"sha256":hashlib.sha256(b).hexdigest()} for n,b in frozen_aux], "commands": commands,
                "experiment_defaults": {"mmap_demand": False, "expert_prefetch": False, "dense_prefetch": False,
                                        "mmap_nocache": False, "dense_lookahead": False, "dense_lookahead_mib": 32, "dense_lock": False, "dense_prefetch_max_cols": 1, "q8_cpu_arith": False, "q8_only": False, "mapped_cpu_weights": False, "q8_shared": False, "expert_drop": False}}
    manifest_path = destination / "manifest.json"

    def save():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    with log_path.open("w") as log:
        def run(args, *, cwd=None, capture=False):
            args = [str(arg) for arg in args]
            commands.append({"argv": args, "cwd": str(cwd) if cwd else None})
            log.write(json.dumps(commands[-1]) + "\n")
            log.flush()
            if capture:
                completed = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                log.write(completed.stdout)
                log.flush()
            else:
                completed = subprocess.run(args, cwd=cwd, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, text=True)
            if completed.returncode:
                raise RuntimeError("Build command failed; inspect " + str(log_path))
            return completed.stdout if capture else None

        save()
        try:
            # The caller may run from inside --source. Keep all temporary writes
            # in the admitted destination, which was verified outside that source.
            with tempfile.TemporaryDirectory(prefix=".glm-native-build-", dir=destination) as temporary:
                workspace = Path(temporary)
                checkout, cmake_build = workspace / "source", workspace / "build"
                inputs = workspace / "inputs"
                inputs.mkdir()
                for name, data in frozen_patches:
                    (inputs / name).write_bytes(data)
                for name,data in frozen_aux:
                    (inputs/name).write_bytes(data)
                probe_source = inputs / PROBE.name
                probe_source.write_bytes(frozen_probe)
                clone = ["git", "clone", "--no-checkout"]
                if source:
                    clone.append("--no-hardlinks")
                run([*clone, str(source) if source else UPSTREAM, checkout])
                run(["git", "checkout", "--detach", REVISION], cwd=checkout)
                actual = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).strip()
                if actual != REVISION:
                    raise RuntimeError("The isolated checkout is not the pinned revision")
                for name, _ in frozen_patches:
                    patch = inputs / name
                    run(["git", "apply", "--check", patch], cwd=checkout)
                    run(["git", "apply", patch], cwd=checkout)
                manifest["applied_source_diff_sha256"] = hashlib.sha256(
                    run(["git", "diff", "--binary", "HEAD"], cwd=checkout, capture=True).encode()).hexdigest()
                compiler_c = run(["xcrun", "--find", "clang"], capture=True).strip()
                compiler_cxx = run(["xcrun", "--find", "clang++"], capture=True).strip()
                manifest["compiler_version"] = run([compiler_cxx, "--version"], capture=True).strip()
                sdk = run(["xcrun", "--show-sdk-path"], capture=True).strip()
                if not Path(sdk).is_dir():
                    raise RuntimeError("xcrun did not return an existing macOS SDK")
                manifest["macos_sdk_path"] = sdk
                manifest["cmake_version"] = run(["cmake", "--version"], capture=True).splitlines()[0]
                flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + compiler_c,
                         "-DCMAKE_CXX_COMPILER=" + compiler_cxx, "-DCMAKE_OSX_SYSROOT=" + sdk, "-DGGML_NATIVE=ON", "-DGGML_METAL=ON",
                         "-DGGML_METAL_EMBED_LIBRARY=ON", "-DBUILD_SHARED_LIBS=ON",
                         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                         "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_OPENSSL=OFF",
                         "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=@loader_path"]
                run(["cmake", "-S", checkout, "-B", cmake_build, "-G", "Ninja", *flags])
                run(["cmake", "--build", cmake_build, "--target", "llama-server", "llama-completion", "-j", jobs])
                packaged = destination / "bin"
                packaged.mkdir()
                for name in ("llama-server", "llama-completion"):
                    shutil.copy2(cmake_build / "bin" / name, packaged / name)
                libraries = sorted((cmake_build / "bin").glob("*.dylib"))
                if not libraries:
                    raise RuntimeError("No native libraries produced")
                for library in libraries:
                    shutil.copy2(library, packaged / library.name, follow_symlinks=False)
                license_sources = ("LICENSE", "licenses/LICENSE-jsonhpp", "vendor/cpp-httplib/LICENSE")
                for relative in license_sources:
                    target = destination / "licenses" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkout / relative, target)
                manifest["licenses"] = ["licenses/" + relative for relative in license_sources]
                probe = packaged / "glm-native-fidelity"
                run([compiler_cxx, "-isysroot", sdk, "-std=c++17", "-O3", "-I", checkout / "include",
                     "-I", checkout / "ggml/include", probe_source, "-L", packaged, "-lllama", "-lggml",
                     "-lggml-base", "-lggml-cpu", "-Wl,-rpath,@loader_path", "-o", probe])
                # Ad hoc signatures cover the exact packaged Mach-O bytes and require no credentials.
                for path in sorted(packaged.iterdir()):
                    if path.is_file() and not path.is_symlink():
                        run(["/usr/bin/codesign", "--force", "--sign", "-", path])
                manifest["runtime_version"] = run([packaged / "llama-server", "--version"], capture=True).strip()
                if REVISION[:7] not in manifest["runtime_version"]:
                    raise RuntimeError("Packaged runtime version is inconsistent")
                manifest["dynamic_dependencies"] = run(["/usr/bin/otool", "-L", packaged / "llama-server"], capture=True)
                artifacts = []
                for path in sorted(packaged.iterdir()):
                    item = {"path": str(path.relative_to(destination))}
                    if path.is_symlink():
                        item["symlink"] = os.readlink(path)
                        if Path(item["symlink"]).is_absolute() or not path.resolve().is_relative_to(packaged):
                            raise RuntimeError("Build-time library symlink escaped package")
                    else:
                        item.update(bytes=path.stat().st_size, sha256=sha256(path))
                    artifacts.append(item)
                manifest.update(status="built", artifacts=artifacts,
                                completed_at=datetime.now(timezone.utc).isoformat(),
                                benchmark_performed=False, model_loaded=False)
                save()
        except BaseException as error:
            manifest.update(status="failed", error=str(error), completed_at=datetime.now(timezone.utc).isoformat())
            save()
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing Git checkout; clone it without modifying it")
    parser.add_argument("--destination", required=True, type=Path, help="Empty directory for bin/, manifest.json and build.log")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument('--dense-lock-experiment', action='store_true', help='Include16th default-off bounded dense mlock research patch')
    args = parser.parse_args()
    try:
        result = build(args.source, args.destination, args.jobs, args.dense_lock_experiment)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"status": result["status"], "destination": str(args.destination.resolve()),
                      "source_revision": result["source_revision"], "model_loaded": False}))


if __name__ == "__main__":
    main()
