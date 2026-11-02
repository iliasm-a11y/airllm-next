"""Serial fresh-server ABBA benchmark campaign, gated by matching fidelity.

The saved plan precedes every launch. Production configuration is untouched.
Timing samples describe these runs only; short fidelity evidence is not long
context qualification, and no confidence interval is inferred from two samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CASES = ("head-inline", "head-shared", "quant-tail", "drop-cpu")


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_fidelity(path, candidate, model, tail_layers=None, *, binary):
    report = json.loads(Path(path).read_text())
    if (report.get("fidelity_passed") is not True or report.get("guard_status") != "completed" or
            report.get("model_file_stats_unchanged") is not True):
        raise ValueError("candidate requires a completed, passed fidelity experiment with unchanged model stats")
    if not {"routes", "router", "kda"}.issubset(report.get("coverage", {})):
        raise ValueError("candidate fidelity must include routes, router and KDA coverage")
    fidelity_libraries = report.get("native_libraries_sha256", {})
    current_libraries = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(binary).resolve().parent.glob("*.dylib") if not p.is_symlink()}
    if not fidelity_libraries or fidelity_libraries != current_libraries:
        raise ValueError("native library hashes are missing or differ from the passed fidelity runtime")
    for label, metrics in [("logits", report.get("full_logits", {})), *report.get("coverage", {}).items()]:
        if (metrics.get("passed") is not True or metrics.get("finite") is not True or
                metrics.get("bit_mismatch_count") != 0):
            raise ValueError("campaign requires bit-identical captured fidelity: " + label)
    env = report.get("environment", {})
    policy = report.get("command", [None])[-1]
    flags = []
    if candidate in ("head-inline", "head-shared"):
        if policy != "head-only" or env.get("AIRLLM_METAL_Q8_CPU_ARITH") != "1":
            raise ValueError("fidelity report does not match CPU-compatible output-head candidate")
        shared = env.get("AIRLLM_METAL_Q8_SHARED") == "1"
        if shared != (candidate == "head-shared"):
            raise ValueError("fidelity report does not match inline/shared arithmetic mode")
        flags = ["--head-only", "--q8-cpu-arith"] + (["--q8-shared"] if shared else [])
    elif candidate == "quant-tail":
        from .placement import plan
        placement = plan(tail_layers)
        if (policy != "quant-tail" or env.get("AIRLLM_METAL_Q8_CPU_ARITH") != "2" or
                env.get("AIRLLM_GLM_Q8_ONLY") != "1" or
                env.get("AIRLLM_Q8_TAIL_PATTERN") != placement["tensor_pattern"]):
            raise ValueError("fidelity report does not match the requested quantized tail placement")
        flags = ["--quant-tail", str(tail_layers)]
    elif candidate == "drop-cpu":
        if policy != "cpu" or env.get("AIRLLM_EXPERT_DROP_AFTER_USE") != "1":
            raise ValueError("fidelity report does not match CPU expert drop-after-use")
        flags = ["--expert-drop"]
    else:
        raise ValueError("unknown candidate")
    if env.get("AIRLLM_METAL_CPU_MAPPED_WEIGHTS") == "1":
        flags.append("--mapped-cpu-weights")
    for row in report.get("model_files", []):
        stat = (Path(model).parent / row["file"]).stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("model file stats differ from the fidelity experiment")
    if not report.get("model_files"):
        raise ValueError("fidelity experiment has no model identity records")
    scope = {"fidelity_report": str(Path(path).resolve()),
             "fidelity_report_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
             "captured_steps": report["full_logits"].get("compared"),
             "full_logits": report["full_logits"], "coverage": report.get("coverage", {}),
             "scope": "passed captured trajectory only; not qualification of every benchmark token or long context",
             "fidelity_native_libraries_sha256": fidelity_libraries,
             "current_native_libraries_sha256": current_libraries}
    return flags, scope


def make_plan(args):
    if args.output.exists():
        raise ValueError("refusing to reuse a campaign output directory")
    if args.tokens not in (16, 64) or type(args.cycles) is not int or not 1 <= args.cycles <= 3:
        raise ValueError("campaign tokens must be16/64 and cycles1..3")
    if args.candidate == "quant-tail" and (type(args.tail_layers) is not int or not 1 <= args.tail_layers <= 45):
        raise ValueError("quant-tail requires --tail-layers1..45")
    if args.candidate != "quant-tail" and args.tail_layers is not None:
        raise ValueError("tail-layers is only valid for quant-tail")
    flags, fidelity = validate_fidelity(args.candidate_fidelity, args.candidate, args.model, args.tail_layers,
                                        binary=args.binary)
    common = [sys.executable, "-m", "research.glm53_gpu_v2.benchmark_native",
              "--binary", str(args.binary.resolve()), "--model", str(args.model.resolve()),
              "--threads", "10", "--context", "512", "--batch", "8", "--gpu-layers", "0",
              "--tokens", str(args.tokens), "--repeats", "1", "--prompt", args.prompt,
              "--max-rss-gib", "8.5", "--max-swap-mib", "32", "--timeout", "900",
              "--mmap-demand", "--expert-prefetch", "--dense-prefetch", "--metal-timing"]
    jobs = []
    for cycle in range(args.cycles):
        for letter in "ABBA":
            index = len(jobs)
            report = args.output.resolve() / "runs" / f"{index:02d}-{letter}.json"
            jobs.append({"index": index, "cycle": cycle, "arm": letter,
                         "candidate": "cpu10" if letter == "A" else args.candidate,
                         "report": str(report), "runner_log": str(report.with_suffix(".runner.log")),
                         "command": common + ["--output", str(report)] + (flags if letter == "B" else [])})
    return {"schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "planned", "sequence": "ABBA" * args.cycles, "jobs": jobs,
            "tokens_requested": args.tokens, "fidelity": fidelity, "production_changed": False,
            "limits": ["fresh server per stage; macOS file cache and background load uncontrolled",
                       "fidelity scope is the attached captured trajectory, not every long generation",
                       "small samples; no confidence intervals, p-values or universal speed claims",
                       "drop-cpu does not establish GPU utilization or GPU speedup"]}


def owned_process(pid):
    import psutil
    return psutil.Process(pid)


def cancel_owned(child, identity, *, get_process=owned_process, get_group=os.getpgid, kill_group=os.killpg):
    """Interrupt only the owned benchmark; let its finally stop its server."""
    if child.poll() is not None:
        return {"status": "already_exited"}
    if identity is None:
        return {"status": "ownership_unknown_no_signal"}
    try:
        current = get_process(child.pid)
        if (child.pid != identity["pid"] or current.create_time() != identity["create_time"] or
                get_group(child.pid) != child.pid):
            return {"status": "ownership_mismatch_no_signal"}
        kill_group(child.pid, signal.SIGINT)
        child.wait(timeout=20)
        return {"status": "interrupted_and_exited"}
    except subprocess.TimeoutExpired:
        return {"status": "cleanup_pending_no_unverified_forced_kill", "owned_benchmark": identity}
    except ProcessLookupError:
        return {"status": "already_exited"}
    except Exception as exc:
        if type(exc).__name__ == "NoSuchProcess":
            return {"status": "already_exited"}
        return {"status": "ownership_check_failed_no_signal", "error": str(exc)}


def summarize(plan, stages):
    groups = {"A": [], "B": []}
    for job, stage in zip(plan["jobs"], stages):
        report = stage.get("report_data")
        if (stage.get("status") == "complete" and report and
                report.get("status") == "complete" and not report.get("guard_violation")):
            request = report["requests"][0]
            groups[job["arm"]].append({"index": job["index"], "tokens": request.get("tokens", []),
                "ttft_seconds": request.get("ttft_seconds"),
                "decode_tokens_per_second": request.get("observed_decode_tokens_per_second")})
    metrics = {}
    for arm, runs in groups.items():
        metrics[arm] = {"runs": runs}
        for name in ("ttft_seconds", "decode_tokens_per_second"):
            values = [run[name] for run in runs if isinstance(run[name], (int, float)) and math.isfinite(run[name])]
            metrics[arm][name] = {"values": values, "median": statistics.median(values) if values else None,
                                  "range": [min(values), max(values)] if values else None}
    pairs = []
    by_index = {run["index"]: run for rows in groups.values() for run in rows}
    for index in range(0, len(plan["jobs"]), 2):
        if index in by_index and index + 1 in by_index:
            a, b = by_index[index], by_index[index + 1]
            pairs.append({"stage_indices": [index, index + 1], "generated_ids_identical": bool(a["tokens"]) and a["tokens"] == b["tokens"],
                          "token_counts": [len(a["tokens"]), len(b["tokens"])]})
    differences = {}
    for name in ("ttft_seconds", "decode_tokens_per_second"):
        a, b = metrics["A"][name]["median"], metrics["B"][name]["median"]
        differences[name] = {"B_minus_A": b - a if a is not None and b is not None else None,
                             "B_relative_to_A_percent": (b/a - 1)*100 if a and b is not None else None}
    complete = len(stages) == len(plan["jobs"]) and all(s.get("status") == "complete" for s in stages)
    swap_growth = [s.get("report_data", {}).get("peak_swap_growth_bytes")
                   for s in stages if s.get("report_data")]
    all_reports_guard_clean = all((s.get("report_data") or {}).get("status") == "complete" and
                                  (s.get("report_data") or {}).get("guard_violation") is None for s in stages)
    memory_qualified = (complete and all_reports_guard_clean and len(swap_growth) == len(plan["jobs"]) and
                        all(type(v) is int and v == 0 for v in swap_growth))
    return {"arms": metrics, "paired_generated_ids": pairs, "median_differences": differences,
            "all_stages_complete": complete,
            "comparison_valid_on_stated_scope": bool(complete and len(pairs) == len(plan["jobs"])//2 and
                                        all(p["generated_ids_identical"] for p in pairs)),
            "optimization_qualified": False, "memory_stability_qualified": memory_qualified,
            "observed_peak_swap_growth_bytes": swap_growth,
            "acceptance_scope": "matching passed fidelity gate plus generated-ID agreement; not new full-logit validation of timed runs"}


def run_campaign(args, plan):
    # Fail before creating an experiment or launching anything if the selected
    # Python environment lacks the process-identity dependency.
    owned_process(os.getpid())
    args.output.mkdir(parents=True)
    (args.output / "runs").mkdir()
    save(args.output / "plan.json", plan)
    stages = []
    campaign = {"plan": plan, "stages": stages, "status": "running"}
    save(args.output / "campaign.json", campaign)
    child = identity = None
    try:
        for job in plan["jobs"]:
            print(json.dumps({"stage": job["index"], "arm": job["arm"], "status": "starting"}), flush=True)
            stage = {"index": job["index"], "arm": job["arm"], "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
            stages.append(stage)
            save(args.output / "campaign.json", campaign)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("AIRLLM_", "GGML_", "LLAMA_ARG_", "DYLD_"))}
            with Path(job["runner_log"]).open("x") as output:
                child = subprocess.Popen(job["command"], cwd=REPO, env=env, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                identity = {"pid": child.pid, "create_time": owned_process(child.pid).create_time()}
                stage["owned_benchmark"] = identity
                save(args.output / "campaign.json", campaign)
                code = child.wait()
            stage["exit_code"] = code
            raw_path = Path(job["report"])
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else None
            stage["report_data"] = raw
            gpu_expected = job["arm"] == "B" and args.candidate != "drop-cpu"
            gpu_observed = bool(raw and raw.get("gpu_command_times", {}).get("valid_commands", 0) > 0)
            stage["gpu_commands_required"] = gpu_expected
            stage["gpu_commands_observed"] = gpu_observed
            stage["status"] = "complete" if (code == 0 and raw and raw.get("status") == "complete" and
                not raw.get("guard_violation") and (not gpu_expected or gpu_observed)) else "failed"
            if gpu_expected and not gpu_observed:
                stage["failure_reason"] = "candidate GPU command instrumentation recorded no valid completed commands"
            stage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            child = identity = None
            save(args.output / "campaign.json", campaign)
            print(json.dumps({"stage": job["index"], "arm": job["arm"], "status": stage["status"], "report": job["report"]}), flush=True)
            if stage["status"] != "complete":
                raise RuntimeError("stage failed or its guard rejected it; remaining stages not launched")
        campaign["status"] = "complete"
    except KeyboardInterrupt:
        campaign["status"] = "cancelled"
        if child is not None:
            campaign["cancellation"] = cancel_owned(child, identity)
            if campaign["cancellation"]["status"] not in ("already_exited", "interrupted_and_exited"):
                campaign["status"] = "cancelled_needs_attention"
        if stages and stages[-1]["status"] == "running":
            stages[-1]["status"] = "cancelled"
    except Exception as exc:
        campaign.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if child is not None:
            campaign["cancellation"] = cancel_owned(child, identity)
    finally:
        campaign["summary"] = summarize(plan, stages)
        save(args.output / "campaign.json", campaign)
    return 0 if campaign["status"] == "complete" and campaign["summary"]["comparison_valid_on_stated_scope"] else 1


def interrupt_campaign(*unused):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-fidelity", type=Path, required=True)
    parser.add_argument("--candidate", choices=CASES, required=True)
    parser.add_argument("--tail-layers", type=int)
    parser.add_argument("--tokens", type=int, choices=(16, 64), default=16)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--prompt", default="La capitale de la France est")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        with Path("/tmp/airllm-glm53-campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous_handler = signal.signal(signal.SIGTERM, interrupt_campaign)
            try:
                return run_campaign(args, plan)
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
