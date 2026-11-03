"""Build metadata-only helper and compare every complete prompt to native Jinja/IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research/glm53_gpu_v3"))
from process_io import ProcessIOObserver
from prefix_audit import common_prefix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_prompts(companion, native):
    expected, observed = companion["agents"], native["agents"]
    matches = []
    for index, (left, right) in enumerate(zip(expected, observed)):
        matches.append({"index": index, "agent_id": left["agent_id"],
                        "same_agent_id": left["agent_id"] == right["agent_id"],
                        "same_rendered_bytes": left["rendered_prompt"].encode() == right["rendered_prompt"].encode(),
                        "same_full_token_ids": left["token_ids"] == right["token_ids"],
                        "native_prompt_tokens": len(right["token_ids"])})
    lcp = common_prefix(a["token_ids"] for a in observed)
    return {"passed": len(expected) == len(observed) and bool(matches) and all(
                row["same_agent_id"] and row["same_rendered_bytes"] and row["same_full_token_ids"] for row in matches),
            "expected_prompt_count": len(expected), "native_prompt_count": len(observed),
            "matches": matches, "native_common_token_count": len(lcp), "native_common_token_ids": lcp,
            "microbatch8_aligned_candidate": len(lcp) // 8 * 8,
            "qualification": "Template/tokenization only. No prefill state, reset/restore, logits or route equivalence claimed."}


def build(source, runtime, output):
    command = ["clang++", "-std=c++17", "-O2"]
    for directory in ("include", "common", "ggml/include", "vendor"):
        command += ["-I", str(source / directory)]
    command += [str(Path(__file__).with_name("native_prefix.cpp")), "-L", str(runtime),
                "-Wl,-rpath," + str(runtime), "-lllama", "-lllama-common", "-o", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return {"command": command, "returncode": completed.returncode,
            "helper_sha256": sha(output),
            "sources": {name: sha(Path(__file__).with_name(name)) for name in
                        ("native_prefix.cpp", "native_audit.py", "prefix_audit.py")},
            "headers": {name: sha(source / name) for name in ("include/llama.h", "common/chat.h", "common/common.h")},
            "libraries": {p.name: sha(p) for p in runtime.glob("*.dylib") if not p.is_symlink()}}


def run(args):
    if args.gguf.stat().st_size > 64 * 1024**2:
        raise ValueError("This metadata-only audit requires the pinned small first GGUF (<=64MiB)")
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / "native-prefix"
    provenance = build(args.source, args.runtime, binary)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYLD_", "AIRLLM_", "LLAMA_ARG_"))}
    output = args.output / "native-prompts.json"
    log = args.output / "native-stderr.log"
    observations = []
    started = time.perf_counter()
    with output.open("wb") as stdout, log.open("wb") as stderr:
        child = subprocess.Popen([str(binary), str(args.gguf), str(args.input)], stdout=stdout, stderr=stderr, env=env)
        observer = ProcessIOObserver(child.pid)
        try:
            while child.poll() is None:
                observations.append(observer.snapshot())
                if time.perf_counter() - started > 120:
                    raise TimeoutError("native metadata-only audit exceeded 120s")
                time.sleep(.05)
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    elapsed = time.perf_counter() - started
    report = {"scope": "metadata-only tokenization/template audit, not model prefill timing",
              "build": provenance, "input_sha256": sha(args.input), "first_gguf_sha256": sha(args.gguf),
              "native_output_sha256": sha(output), "stderr_sha256": sha(log),
              "returncode": child.returncode, "elapsed_seconds": elapsed,
              "observations": observations, "owned_process_reaped": child.poll() is not None,
              "physical_nand_read_bytes": None,
              "scope_limits": "OS process counters are sampled; only the small first metadata GGUF is hashed. No second tensor-shard checksum, inference or performance gain measured."}
    valid = [o for o in observations if o.get("available")]
    report["peak_sampled_rss_bytes"] = max((o["rusage"]["resident_bytes"] for o in valid), default=None)
    report["maximum_observed_os_storage_read_bytes"] = max((o["rusage"]["os_storage_read_bytes"] for o in valid), default=None)
    if child.returncode == 0:
        companion = json.loads(args.input.read_text())
        native = json.loads(output.read_text())
        report["comparison"] = compare_prompts(companion, native)
        report["native_template_sha256"] = hashlib.sha256(native["template"].encode()).hexdigest()
        report["native_template_matches_companion_file"] = report["native_template_sha256"] == companion["files"]["chat_template.jinja"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"returncode": child.returncode, "report": str(args.output / "report.json"),
                      "comparison_passed": report.get("comparison", {}).get("passed"),
                      "native_common_tokens": report.get("comparison", {}).get("native_common_token_count")}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "runtime", "gguf", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = run(parser.parse_args())
    return 0 if report.get("comparison", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
