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
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from research.glm53_m5.run_guarded_probe import run as guard
from .system_memory import SystemMemoryObserver
from .dense_lock_plan import load_plan, lock_environment, check_plan, retained_log


def run(args):
    if args.requests == 2 and not args.state_hashes:
        raise ValueError("Repeated-request validation requires observed state hashes")
    if args.policy in ("cpu", "head-only", "quant-tail") and args.gpu_layers:
        raise ValueError("cpu/head-only require zero GPU weight layers")
    if args.policy in ("weights-only", "legacy") and not args.gpu_layers:
        raise ValueError("weights-only/legacy require positive GPU layers")
    if args.policy in ("cpu", "head-only", "quant-tail") and args.cache_mib:
        raise ValueError("Expert cache requires offloaded weight layers")
    if args.q8_shared and (args.policy != "head-only" or not args.q8_cpu_arith):
        raise ValueError("Shared head requires head-only and CPU-compatible arithmetic")
    dense_lock = load_plan(args.dense_lock_plan, args.model) if getattr(args, 'dense_lock_plan', None) else None
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
    if args.source_plan: knobs['AIRLLM_SOURCE_PLAN'] = '1'
    if args.state_hashes: knobs['AIRLLM_STATE_HASHES'] = '1'
    knobs['AIRLLM_DIAGNOSTIC_REQUESTS'] = str(args.requests)
    if args.mmap_nocache: knobs['AIRLLM_MMAP_NOCACHE'] = '1'
    if args.dense_lookahead:
        knobs['AIRLLM_DENSE_LOOKAHEAD'] = '1'
        knobs['AIRLLM_DENSE_LOOKAHEAD_MIB'] = str(args.lookahead_mib)
    if dense_lock:
        knobs.update(lock_environment(dense_lock))
    if args.profile_ops: knobs['AIRLLM_PROFILE_OPS'] = '1'
    if args.profile_pages: knobs['AIRLLM_PROFILE_PAGES'] = '1'
    if args.layer_local_hc:
        knobs["AIRLLM_GLM_LAYER_LOCAL_HC"] = "1"
    if args.disable_hc_post:
        knobs["AIRLLM_GLM_HC_POST"] = "0"
    if args.cache_lookup_split:
        knobs["AIRLLM_GLM_CACHE_SPLIT"] = "1"
    if args.metal_timing:
        knobs["AIRLLM_METAL_COMMAND_TIMING"] = "1"
    if args.expert_drop:
        knobs["AIRLLM_EXPERT_DROP_AFTER_USE"] = "1"
    if args.q8_shared:
        knobs["AIRLLM_METAL_Q8_SHARED"] = "1"
    if args.mapped_cpu_weights:
        knobs["AIRLLM_METAL_CPU_MAPPED_WEIGHTS"] = "1"
    if args.q8_cpu_arith:
        knobs["AIRLLM_METAL_Q8_CPU_ARITH"] = "1"
    if args.policy == "quant-tail":
        from research.glm53_gpu_v2.placement import plan
        placement = plan(args.tail_layers)
        (args.output / "placement.json").write_text(json.dumps(placement, indent=2)+"\n")
        knobs.update(AIRLLM_METAL_Q8_CPU_ARITH="2", AIRLLM_GLM_Q8_ONLY="1",
                     AIRLLM_Q8_TAIL_PATTERN=placement["tensor_pattern"])
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
                "environment": knobs, "model_files": [], "production_changed": False,
                "guard_limits": {"max_rss_gib": 8.5, "max_swap_mib": 32,
                                 "timeout_seconds": 900 if args.tokens > 16 else 400}}
    if dense_lock:
        manifest['dense_lock_plan'] = dense_lock
    for path in sorted(args.model.parent.glob("*.gguf")):
        st = path.stat()
        manifest["model_files"].append({"file": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    manifest["weight_integrity_scope"] = "file sizes and modification times; no new full 110GB hash"
    manifest["probe_binary_sha256"] = hashlib.sha256(args.probe.read_bytes()).hexdigest()
    libraries = args.runtime_libraries
    if libraries is None:
        listing = subprocess.check_output(["/usr/bin/otool", "-l", str(args.probe.resolve())], text=True)
        for candidate in re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", listing):
            directory = Path(candidate.replace("@loader_path", str(args.probe.resolve().parent)).replace("@executable_path", str(args.probe.resolve().parent)))
            if (directory / "libllama.dylib").exists():
                libraries = directory
                break
    manifest["native_libraries_sha256"] = ({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in libraries.glob("*.dylib") if not p.is_symlink()} if libraries else {})
    observer = SystemMemoryObserver()
    manifest['system_before'] = observer.snapshot()
    if dense_lock:
        manifest['dense_lock_prelaunch'] = check_plan(dense_lock, args.model)
    try:
        os.environ.clear()
        os.environ.update(clean)
        result = guard(SimpleNamespace(
            command=command, output=args.output / "guard.json", max_rss_gib=8.5,
            max_swap_mib=32, timeout_seconds=manifest["guard_limits"]["timeout_seconds"], poll_seconds=.5))
    finally:
        os.environ.clear()
        os.environ.update(previous)
    manifest["system_after"] = observer.snapshot()
    manifest["guard_status"] = result["status"]
    dense_lock_passed = True
    if dense_lock:
        manifest['dense_lock'] = retained_log(result.get('stderr_file', args.output/'guard.stderr.log'), dense_lock)
        try:
            manifest['dense_lock_postrun'] = check_plan(dense_lock, args.model)
        except (ValueError, OSError) as error:
            manifest['dense_lock_postrun'] = {'passed': False, 'error': str(error)}
        dense_lock_passed = manifest['dense_lock']['passed'] and manifest['dense_lock_postrun']['passed']
    manifest["model_file_stats_unchanged"] = all(
        (args.model.parent / row["file"]).stat().st_size == row["size"] and
        (args.model.parent / row["file"]).stat().st_mtime_ns == row["mtime_ns"]
        for row in manifest["model_files"])
    passed = False
    reset = None
    if result["status"] == "completed" and args.requests == 2:
        from .state_fidelity import compare_reset
        reset = compare_reset(args.output / 'capture', args.tokens)
        manifest['request_reset'] = reset
    if result["status"] == "completed" and args.reference:
        from research.glm53_m5.compare_fidelity import compare
        comparison = compare(args.reference, args.output / "capture", "cpu" if args.policy == "cpu" else "metal")
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        passed = comparison["passed"] and manifest["model_file_stats_unchanged"]
        manifest.update(fidelity_passed=passed, numerical_fidelity_passed=passed, full_logits=comparison["full_logits_summary"],
                        coverage=comparison["coverage"])
        if args.state_hashes:
            from .state_fidelity import compare_states
            states=compare_states(args.reference/'state_hashes.jsonl',args.output/'capture/state_hashes.jsonl')
            manifest['observed_state_writes']=states
            manifest['state_reference_available']=(args.reference/'state_hashes.jsonl').is_file()
            manifest['fidelity_scope']='captured logits/routes/router/KDA; observed-write parity only when a state reference is present'
            if (args.reference/'state_hashes.jsonl').exists():
                passed=passed and states['passed'];manifest['fidelity_passed']=passed
        if reset is not None:
            passed=passed and reset['passed'];manifest['fidelity_passed']=passed
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if dense_lock:
        manifest['dense_lock_accepted'] = bool(result['status'] == 'completed' and dense_lock_passed and manifest['model_file_stats_unchanged'])
        if args.reference:
            passed = bool(passed and manifest['dense_lock_accepted'])
            manifest['fidelity_passed'] = passed
    (args.output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"guard_status": result["status"], "fidelity_passed": passed,
                      "guard_violation": result["guard_violation"]}))
    if result["status"] != "completed":
        return 2
    if reset is not None and not reset['passed']:
        return 3
    if dense_lock and not manifest['dense_lock_accepted']:
        return 3
    return 3 if args.reference and not passed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan', action='store_true')
    p.add_argument('--state-hashes', action='store_true')
    p.add_argument('--requests', type=int, choices=(1,2), default=1)
    p.add_argument('--mmap-nocache', action='store_true')
    p.add_argument('--dense-lookahead', action='store_true')
    p.add_argument('--dense-lock-plan', type=Path, help='Default-off selector manifest for bounded original dense mmap user locks')
    p.add_argument('--lookahead-mib', type=int, choices=(16,32,64,128), default=32)
    p.add_argument('--profile-ops', action='store_true')
    p.add_argument('--profile-pages', action='store_true')
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--policy", choices=("cpu", "head-only", "quant-tail", "weights-only", "legacy"), default="cpu")
    p.add_argument("--gpu-layers", type=int, choices=range(47), default=0)
    p.add_argument("--cache-mib", type=int, choices=(0, 128, 256, 384, 512), default=0)
    p.add_argument("--threads", type=int, choices=range(1, 11), default=10)
    p.add_argument("--tokens", type=int, choices=range(1, 65), default=4)
    p.add_argument("--prompt", default="La capitale de la France est")
    p.add_argument("--layer-local-hc", action="store_true")
    p.add_argument("--disable-hc-post", action="store_true")
    p.add_argument("--cache-lookup-split", action="store_true")
    p.add_argument("--q8-cpu-arith", action="store_true")
    p.add_argument("--q8-shared", action="store_true")
    p.add_argument("--expert-drop", action="store_true")
    p.add_argument("--metal-timing", action="store_true")
    p.add_argument("--runtime-libraries", type=Path)
    p.add_argument("--mapped-cpu-weights", action="store_true")
    p.add_argument("--tail-layers", type=int, default=5)
    args = p.parse_args(argv)
    try:
        with Path("/tmp/airllm-glm53-benchmark.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run(args)
    except (ValueError, OSError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
