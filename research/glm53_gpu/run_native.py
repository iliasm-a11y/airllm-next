"""Run one owned GLM placement diagnostic with fixed fidelity and memory gates.

This captures tensors and serializes parts of execution. Its wall time is not
decode throughput. Production RONIN settings and its single slot are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from research.glm53_m5.run_guarded_probe import run as guard


def run(args):
    if args.policy in ("cpu", "head-only") and args.gpu_layers:
        raise ValueError("cpu/head-only require zero GPU weight layers")
    if args.policy in ("weights-only", "legacy") and not args.gpu_layers:
        raise ValueError("weights-only/legacy require positive GPU layers")
    if args.policy in ("cpu", "head-only") and args.cache_mib:
        raise ValueError("Expert cache requires offloaded weight layers")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an experiment")
    args.output.mkdir(parents=True)
    forced = "-"
    if args.reference:
        rows = [json.loads(x) for x in (args.reference / "tokens.jsonl").read_text().splitlines() if x.strip()]
        if len(rows) < args.tokens:
            raise ValueError("Reference has too few forced tokens")
        forced = str((args.output / "forced.jsonl").resolve())
        # A forced capture can have a different argmax from its actual input.
        trajectory = [{"token": r.get("next_input_token", r["token"])} for r in rows[:args.tokens]]
        if any(type(r["token"]) is not int or r["token"] < 0 for r in trajectory):
            raise ValueError("Invalid reference input token")
        Path(forced).write_text("".join(json.dumps(r) + "\n" for r in trajectory))
    knobs = {"AIRLLM_MMAP_DEMAND": "1", "AIRLLM_EXPERT_PREFETCH": "1",
             "AIRLLM_DENSE_PREFETCH": "1", "AIRLLM_DENSE_PREFETCH_MAX_COLS": "1",
             "AIRLLM_GPU_CACHE_MIB": str(args.cache_mib)}
    if args.layer_local_hc:
        knobs["AIRLLM_GLM_LAYER_LOCAL_HC"] = "1"
    if args.disable_hc_post:
        knobs["AIRLLM_GLM_HC_POST"] = "0"
    if args.cache_lookup_split:
        knobs["AIRLLM_GLM_CACHE_SPLIT"] = "1"
    # Do not inherit unrelated experiment switches or loader injections.
    previous = dict(os.environ)
    clean = {k: v for k, v in previous.items()
             if not k.startswith(("AIRLLM_", "LLAMA_ARG_", "GGML_", "DYLD_"))}
    clean.update(knobs)
    command = [str(args.probe.resolve()), str(args.model.resolve()),
               str((args.output / "capture").resolve()), str(args.threads),
               str(args.gpu_layers), str(args.tokens), args.prompt, forced, args.policy]
    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                "scope": "forced-input tensor diagnostic; not throughput", "command": command,
                "environment": knobs, "model_files": [], "production_changed": False}
    for path in sorted(args.model.parent.glob("*.gguf")):
        st = path.stat()
        manifest["model_files"].append({"file": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    manifest["weight_integrity_scope"] = "file sizes and modification times; no new full 110GB hash"
    manifest["probe_source_sha256"] = hashlib.sha256(Path(__file__).with_name("native_probe.cpp").read_bytes()).hexdigest()
    try:
        os.environ.clear()
        os.environ.update(clean)
        result = guard(SimpleNamespace(
            command=command, output=args.output / "guard.json", max_rss_gib=8.5,
            max_swap_mib=32, timeout_seconds=400, poll_seconds=.5))
    finally:
        os.environ.clear()
        os.environ.update(previous)
    manifest["guard_status"] = result["status"]
    manifest["model_file_stats_unchanged"] = all(
        (args.model.parent / row["file"]).stat().st_size == row["size"] and
        (args.model.parent / row["file"]).stat().st_mtime_ns == row["mtime_ns"]
        for row in manifest["model_files"])
    passed = False
    if result["status"] == "completed" and args.reference:
        from research.glm53_m5.compare_fidelity import compare
        comparison = compare(args.reference, args.output / "capture", "cpu" if args.policy == "cpu" else "metal")
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        passed = comparison["passed"] and manifest["model_file_stats_unchanged"]
        manifest.update(fidelity_passed=passed, full_logits=comparison["full_logits_summary"],
                        coverage=comparison["coverage"])
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (args.output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"guard_status": result["status"], "fidelity_passed": passed,
                      "guard_violation": result["guard_violation"]}))
    if result["status"] != "completed":
        return 2
    return 3 if args.reference and not passed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--policy", choices=("cpu", "head-only", "weights-only", "legacy"), default="cpu")
    p.add_argument("--gpu-layers", type=int, choices=range(47), default=0)
    p.add_argument("--cache-mib", type=int, choices=(0, 128, 256, 384, 512), default=0)
    p.add_argument("--threads", type=int, choices=range(1, 11), default=10)
    p.add_argument("--tokens", type=int, choices=range(1, 65), default=4)
    p.add_argument("--prompt", default="La capitale de la France est")
    p.add_argument("--layer-local-hc", action="store_true")
    p.add_argument("--disable-hc-post", action="store_true")
    p.add_argument("--cache-lookup-split", action="store_true")
    args = p.parse_args(argv)
    try:
        with Path("/tmp/airllm-glm53-benchmark.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run(args)
    except (ValueError, OSError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
