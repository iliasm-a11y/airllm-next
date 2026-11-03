"""Bounded RONIN batch experiments; default sequence1→600→1, native reasoning.

Reuses the PR22 guarded lifecycle with no shared cache or production changes.
A prefill timing run emits only one ID per full prompt and cannot qualify EOS.
Different microbatches and all Metal paths remain numerically unqualified until
separate full logits/routes/state captures pass the unchanged fidelity gates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "air_llm")]
from research.glm53_shared_prefill import qualify_agents as qa
from airllm.nextgen.glm_native import resource_snapshot, MiB, GiB
from research.glm53_shared_prefill.qualify_agents import (
    TASKS, memory_for, QualificationError, NoCacheResearchModel, metadata_preflight,
    verify_cache_flags, final_answer, answer_checks, observe_owned_argv, Runtime,
    BENCHMARK_LOCK, ProbeInterrupted, _cleanup, _graceful_stop, _write,
    PowerGate, ScopedAwake, power_evidence_matches, ProcessIOObserver, validate_completion)

BATCHES = (8, 16, 32, 64, 128)


def profile_config(config, microbatch=8, logical_batch=None, gpu_layers=0):
    if type(microbatch) is not int or microbatch not in BATCHES:
        raise ValueError("Physical microbatch must be8/16/32/64/128")
    logical_batch = microbatch if logical_batch is None else logical_batch
    if type(logical_batch) is not int or not microbatch <= logical_batch <= 512:
        raise ValueError("Logical batch must cover microbatch and remain<=512")
    if type(gpu_layers) is not int or not 0 <= gpu_layers <= 12:
        raise ValueError("Experimental tail Metal offload limited to0..12 layers")
    return {"microbatch": microbatch, "logical_batch": logical_batch, "gpu_layers": gpu_layers,
            "full_model_fidelity_qualified": False, "cache_shared": False,
            "backend_options": {"batch_size": logical_batch, "gpu_layers": gpu_layers,
                "cpu_moe": bool(gpu_layers), "moe_cache_mib": 0,
                "allow_unqualified_gpu": bool(gpu_layers)},
            "scope": "explicit isolated research profile; batch size alone does not prove multi-token Metal work"}


def adaptive_admission(config, requested, *, gpu_layers=0, sample=None):
    """Conservative preflight estimate, then NativeGlmModel rechecks at load/request.

    No OOM probing: select a smaller *new* microbatch before execution, never
    replay an already-started decision. This estimate is not a measured peak.
    """
    sample = resource_snapshot() if sample is None else sample
    if requested not in BATCHES or type(requested) is not int:
        raise ValueError("Unknown requested microbatch")
    limits = qa.NativeLimits(**config["limits"])
    accepted, rows = None, []
    for size in reversed([v for v in BATCHES if v <= requested]):
        profile = profile_config(config, size, size, gpu_layers)
        probe = object.__new__(qa.NativeGlmModel)
        probe.gpu_layers = gpu_layers
        resident = probe._gpu_weight_bytes()
        required = (limits.reserve_bytes + limits.base_workspace_bytes +
                    config["context_tokens"] * 64 * 1024 + size * 4 * MiB + resident)
        enough = type(sample.get("available_bytes")) is int and sample["available_bytes"] >= required
        normal = sample.get("pressure_level") == 1
        passed = enough and normal
        rows.append({"microbatch": size, "required_bytes_estimate": required,
                     "gpu_weight_bytes_estimate": resident, "passed": passed})
        if passed and accepted is None:
            accepted = size
    return {"requested_microbatch": requested, "selected_microbatch": accepted,
            "admitted": accepted is not None, "sample": sample, "candidates": rows,
            "scope": "same workspace/reserve estimate as native guard, strict normal pressure; not peak GPU activation measurement"}


class BatchedResearchModel(NoCacheResearchModel):
    def __init__(self, *args, microbatch=8, **kwargs):
        if microbatch not in BATCHES or type(microbatch) is not int:
            raise ValueError("Unsupported physical microbatch")
        self.research_microbatch = microbatch
        super().__init__(*args, **kwargs)
        if self.batch_size < microbatch:
            raise ValueError("Logical batch smaller than physical microbatch")

    def _build_command(self, port):
        command = super()._build_command(port)
        index = command.index("--ubatch-size") + 1
        command[index] = str(self.research_microbatch)
        self.research_constructed_command["argv"] = command
        self.research_constructed_command["microbatch"] = self.research_microbatch
        return command


def measurement(row):
    final = row.get("final") or {}
    timings = final.get("timings") or {}
    prompt_n, prompt_ms = timings.get("prompt_n"), timings.get("prompt_ms")
    valid = type(prompt_n) is int and prompt_n > 0 and type(prompt_ms) in (int, float) and prompt_ms > 0
    first = next((event.get("t") for event in row.get("events", [])
                  if event.get("response", {}).get("tokens")), None)
    return {"timings_available": valid, "prompt_tokens": len(row.get("prompt_token_ids", [])) or None,
            "server_prompt_tokens_processed": prompt_n, "prefill_seconds": prompt_ms / 1000 if valid else None,
            "prompt_tokens_per_second": 1000 * prompt_n / prompt_ms if valid else None,
            "ttft_seconds": first, "decoding_seconds": timings.get("predicted_ms", 0) / 1000 if type(timings.get("predicted_ms")) in (int, float) else None,
            "output_tokens": len(row.get("tokens", [])), "stop_type": final.get("stop_type"),
            "cpu_compute_seconds": None, "gpu_compute_seconds": None, "weight_wait_seconds": None,
            "synchronization_seconds": None,
            "scope": "native final prompt/decode timings and SSE observed TTFT; component attribution unavailable"}


def plan(config_path, *, requested=128, gpu_layers=0, gate_factory=None):
    config, digest = qa.configuration(config_path, 256)
    power = (gate_factory or PowerGate)("ac").sample()
    admission = adaptive_admission(config, requested, gpu_layers=gpu_layers)
    return {"schema_version": 1, "status": "planned" if power.get("passed") and admission["admitted"] else "blocked",
            "executed": False, "model_loads": 0, "requests": [], "config_sha256": digest,
            "power": power, "admission": admission, "guard_limits": config["limits"],
            "planned_sequence": [1, 600, 1], "reasoning_effort": None,
            "cache_shared": False, "measurements": None}


def run_campaign(config_path, output, tokens=256, *, max_seconds=7200, request_seconds=1800,
                 agent_order=(1, 600, 1), runtime_factory=None,
                 mode="agents", microbatch=8, logical_batch=None, gpu_layers=0, reasoning_effort=None,
                 gate_factory=None, awake_factory=None, io_factory=None, monitor_interval=.5):
    if (tuple(agent_order) not in ((1, 600), (600, 1), (1, 600, 1))
            or any(type(agent) is not int for agent in agent_order)):
        raise ValueError("Choose1,600 or600,1 or isolation sequence1,600,1")
    tasks_by_id = {task["agent_id"]: task for task in TASKS}
    planned_tasks = tuple(tasks_by_id[agent] for agent in agent_order)
    config, config_hash = qa.configuration(config_path, max(128, tokens), request_seconds=request_seconds)
    if mode not in ("agents", "prefill") or (mode == "agents" and not 128 <= tokens <= 512) or (mode == "prefill" and tokens != 1):
        raise ValueError("Agents require128..512 outputs; prefill timing requires exactly1")
    if reasoning_effort not in (None, "low", "high", "max"):
        raise ValueError("Unknown reasoning effort")
    profile = profile_config(config, microbatch, logical_batch, gpu_layers)
    config.update(profile["backend_options"])
    if request_seconds == 2700 and max_seconds > 7200:
        raise ValueError("45-minute requests retain a hard 7200s campaign ceiling")
    if not 0 < monitor_interval <= .5 or not 1 <= max_seconds <= 10800:
        raise ValueError("Sampling<=0.5s and campaign deadline1..10800s required")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    runner_source = Path(__file__).resolve()
    runner_sha256 = hashlib.sha256(runner_source.read_bytes()).hexdigest()
    started = time.perf_counter()
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "runner_source_path": str(runner_source), "runner_source_sha256_at_start": runner_sha256,
              "status": "running", "mode": mode, "profile": profile,
              "reasoning_effort": reasoning_effort, "reasoning_policy": "native GGUF default when null; no override or shortened identity",
              "config_sha256": config_hash, "guard_limits": config["limits"],
              "planned_agents": list(planned_tasks), "requested_tokens_each": tokens, "max_campaign_seconds": max_seconds,
              "explicit_request_budget_seconds": request_seconds,
              "context_tokens": config["context_tokens"], "prompt_policy": "full original RONIN identity/instructions/memory and task; no truncation or summarization",
              "functional_two_agents_qualified": False, "exact_state_qualified": False,
              "full_qualification": False, "requests": [],
              "numerical_scope": "No new logits/routes/KDA/KV/mHC equivalence capture in this server runner; reset is uncached recomputation, no restore enabled"}
    runtime = awake = watcher = process = None
    previous = os.environ.get("RONIN_GLM_CONFIG")
    stop = threading.Event()
    monitor_errors, handlers = [], {}
    gate = (gate_factory or PowerGate)("ac")
    cleaning = False

    def cancel_all():
        if runtime is not None:
            for event in list(runtime.cancels.values()):
                event.set()
            runtime.engine.request_shutdown()

    def interrupt(signum, frame):
        if not cleaning:
            cancel_all()
            raise ProbeInterrupted(signum)

    def monitor():
        while not stop.wait(monitor_interval):
            try:
                awake.check()
                if runtime is not None and not report.get("observed_native_argv", {}).get("available"):
                    current_model = getattr(runtime.engine, "_model", None)
                    current_process = getattr(current_model, "_process", None)
                    if current_process is not None:
                        report["observed_native_argv"] = observe_owned_argv(current_process, config["runtime_binary"])
                if time.perf_counter() - started > max_seconds:
                    raise QualificationError("CAMPAIGN_DEADLINE", "Bounded campaign deadline reached; no automatic replay")
                if not gate.sample()["passed"]:
                    raise QualificationError("POWER_CONTEXT_CHANGED", "AC/lid/clock continuity rejected")
            except Exception as error:
                monitor_errors.append({"code": getattr(error, "code", type(error).__name__), "error": str(error)})
                cancel_all()
                return

    with BENCHMARK_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, mode=0o700)
        try:
            report["metadata_preflight"] = metadata_preflight(config)
            current_runner_sha256 = hashlib.sha256(runner_source.read_bytes()).hexdigest()
            if current_runner_sha256 != runner_sha256:
                raise QualificationError("RUNNER_SOURCE_CHANGED", "Runner source changed before native admission")
            report["metadata_preflight"].setdefault("sha256", {})[str(runner_source)] = runner_sha256
            report["power_preflight"] = gate.sample()
            if not report["power_preflight"]["passed"]:
                raise QualificationError("POWER_CONTEXT_UNAVAILABLE", "Open lid and stable AC power required; no model constructed or loaded")
            time.sleep(.5)
            report["power_admission_confirmation"] = gate.sample()
            if not report["power_admission_confirmation"]["passed"]:
                raise QualificationError("POWER_CONTEXT_UNAVAILABLE", "AC/lid changed during admission; no model constructed or loaded")
            admission = adaptive_admission(config, microbatch, gpu_layers=gpu_layers)
            report["batch_admission"] = admission
            if not admission["admitted"] and gpu_layers:
                report["gpu_admission_fallback"] = "GPU workspace estimate rejected before model activation; select CPU0 and recheck"
                gpu_layers = 0
                admission = adaptive_admission(config, microbatch, gpu_layers=0)
                report["cpu_fallback_admission"] = admission
            if not admission["admitted"]:
                raise QualificationError("BATCH_ADMISSION", "No safely admitted microbatch, no model constructed")
            microbatch = admission["selected_microbatch"]
            profile = profile_config(config, microbatch, logical_batch, gpu_layers)
            config.update(profile["backend_options"])
            report["profile"] = profile
            report["cache_control_preflight"] = verify_cache_flags(config["runtime_binary"])
            receipt = Path(config["verification_file"]).read_bytes()
            copied = output / "model-verification.json"
            copied.write_bytes(receipt)
            copied.chmod(0o600)
            config.update(verification_file=str(copied), state_dir=str(output / "engine-state"))
            _write(output / "run-config.json", config)
            os.environ["RONIN_GLM_CONFIG"] = str(output / "run-config.json")
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupt)
            runtime = (runtime_factory or Runtime)(output / "state")
            # Only this isolated research Engine receives the stricter command.
            def research_model_factory(*args, **kwargs):
                selected = BatchedResearchModel(*args, microbatch=microbatch, **kwargs)
                selected._research_power_gate = gate
                return selected
            runtime.engine._model_factory = research_model_factory
            report["registered_identities"] = len(runtime.dispatch("agent_identities", {}))
            if report["registered_identities"] != 600:
                raise QualificationError("IDENTITY_COUNT", "Exactly600 logical identities required")
            owner = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
            awake = (awake_factory or ScopedAwake)(owner)
            awake.__enter__()
            watcher = threading.Thread(target=monitor, daemon=True, name="two-agent-power")
            watcher.start()
            if not gate.sample()["passed"]:
                raise QualificationError("POWER_CONTEXT_CHANGED", "AC/lid/clock rejected before model load")
            runtime.engine.load(config["model_path"])
            model = runtime.engine._model
            process = model._process
            if process is None or process.poll() is not None:
                raise QualificationError("SERVER_NOT_RUNNING", "No live owned native process after load")
            report.update(owned_pid=process.pid, engine_after_load=runtime.engine.status(),
                          constructed_native_command=getattr(model, "research_constructed_command", None),
                          load_seconds=time.perf_counter()-started)
            observer = (io_factory or ProcessIOObserver)(process.pid)
            missions = {}
            for request_index, task in enumerate(planned_tasks):
                if monitor_errors:
                    raise QualificationError("SUPERVISION_REJECTED", "Campaign supervision already rejected")
                if task["agent_id"] not in missions:
                    missions[task["agent_id"]] = runtime.create({"prompt": task["prompt"]})
                    runtime.dispatch("agent_memory", {"id": missions[task["agent_id"]]["id"], "agent_id": task["agent_id"], "text": memory_for(task)})
                mission = missions[task["agent_id"]]
                row = {"request_index": request_index, "agent_id": task["agent_id"], "mission_id": mission["id"], "marker": task["marker"],
                       "events": [], "tokens": [], "text": "", "final": None, "owned_pid": process.pid}
                report["requests"].append(row)
                _write(output / "report.json", report)
                request_output = output / ("request-%02d-agent-%d" % (request_index + 1, task["agent_id"]))
                request_output.mkdir(mode=0o700)
                restore = qa.capture_native(model, row, request_output, observer)
                try:
                    parameters = {"id": mission["id"], "agent_id": task["agent_id"], "max_new_tokens": tokens}
                    if reasoning_effort is not None:
                        parameters["reasoning_effort"] = reasoning_effort
                    row["result"] = runtime.dispatch("generate", parameters)
                    row["decision"] = runtime.dispatch("agent_decision", {"id": mission["id"], "agent_id": task["agent_id"],
                        "decision_id": row["result"]["decision_id"]})
                    row["engine_after"] = runtime.engine.status()
                    row["completion_contract"] = validate_completion(row, tokens)
                    row["answer_checks"] = answer_checks(task, row["text"])
                    row["final_answer"] = final_answer(row["text"])
                    row["measurement"] = measurement(row)
                except Exception as error:
                    row.update(error_code=getattr(error, "code", type(error).__name__), error=str(error),
                        error_details=getattr(error, "details", None),
                        context=runtime.dispatch("agent_context", {"id": mission["id"], "agent_id": task["agent_id"]}))
                    raise
                finally:
                    restore()
                    _write(output / "report.json", report)
            report["engine_after_stream"] = runtime.engine.status()
            report["status"] = "completed"
        except ProbeInterrupted as error:
            report.update(status="cancelled", error_code=error.code, error=str(error), interrupted_signal=error.signum)
        except Exception as error:
            report.update(status="blocked" if runtime is None else "failed", error_code=getattr(error, "code", type(error).__name__),
                          error=str(error), error_details=getattr(error, "details", None))
        finally:
            cleaning = True
            stop.set()
            if watcher is not None:
                watcher.join(timeout=6)
                if watcher.is_alive():
                    monitor_errors.append({"code": "POWER_MONITOR_NOT_STOPPED"})
            try:
                if runtime is not None:
                    current_model = getattr(runtime.engine, "_model", None)
                    report["constructed_native_command"] = getattr(current_model, "research_constructed_command", None)
                    try:
                        _graceful_stop(process, report)
                    finally:
                        _cleanup(runtime, report)
                    report["engine_after_cleanup"] = runtime.engine.status()
                    model = getattr(runtime.engine, "_model", None)
                    if model is not None:
                        _write(output / "native-resource-samples.json", list(getattr(model, "_samples", [])))
            except Exception as error:
                report.update(status="failed", cleanup_error=str(error))
            try:
                if awake is not None:
                    awake.close()
                gate.sample(final=True)
                report["power_gate"] = gate.summary()
                if awake is not None:
                    report["power_gate"]["idle_assertion"] = awake.record
            except Exception as error:
                report.update(status="failed", final_power_error=str(error))
            report["power_continuity_validated"] = power_evidence_matches({**report, "owned_pid": os.getpid()}, "ac")
            try:
                final_runner_sha256 = hashlib.sha256(runner_source.read_bytes()).hexdigest()
                report["runner_source_identity"] = {"passed": final_runner_sha256 == runner_sha256,
                    "path": str(runner_source), "sha256_at_start": runner_sha256,
                    "sha256_postflight": final_runner_sha256,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat()}
            except OSError as error:
                report["runner_source_identity"] = {"passed": False, "path": str(runner_source),
                    "sha256_at_start": runner_sha256, "sha256_postflight": None, "error": str(error)}
            if report["runner_source_identity"]["passed"] is not True:
                report["status"] = "failed"
                report.setdefault("error_code", "RUNNER_SOURCE_CHANGED")
                report.setdefault("error", "Runner source identity could not be preserved through postflight")
            if monitor_errors:
                report.update(status="failed", supervision_errors=monitor_errors)
            report["functional_checks"] = campaign_checks(report)
            report["functional_two_agents_qualified"] = mode == "agents" and report["status"] == "completed" and all(report["functional_checks"].values())
            # A completed timing transcript still needs every structural and
            # resource check. EOS/useful answers belong to agent qualification.
            report["prefill_admission_checks"] = {key: value for key, value in report["functional_checks"].items()
                if key not in ("natural_eos_each", "useful_final_each", "native_reasoning_preserved")}
            report["prefill_measurements_complete"] = (report["status"] == "completed"
                and all(report["prefill_admission_checks"].values())
                and len(report["requests"]) == len(planned_tasks)
                and all(r.get("measurement", {}).get("timings_available") is True
                        and r.get("completion_contract", {}).get("passed") is True for r in report["requests"]))
            report["adopted_optimization"] = False
            if report["status"] == "completed" and (
                    mode == "agents" and not report["functional_two_agents_qualified"] or
                    mode == "prefill" and not report["prefill_measurements_complete"]):
                report["status"] = "rejected"
            report["elapsed_seconds"] = time.perf_counter() - started
            if previous is None:
                os.environ.pop("RONIN_GLM_CONFIG", None)
            else:
                os.environ["RONIN_GLM_CONFIG"] = previous
            try:
                _write(output / "report.json", report)
            finally:
                for signum, previous_handler in handlers.items():
                    signal.signal(signum, previous_handler)
    return report



def campaign_checks(report):
    rows = report.get("requests", [])
    planned = [task["agent_id"] for task in report.get("planned_agents", [])]
    native = report.get("engine_after_stream", {}).get("native", {})
    final_native = report.get("engine_after_cleanup", {}).get("native", {})
    tasks = {task["agent_id"]: task for task in TASKS}
    def isolated(row, key):
        text = row.get(key, "") if key == "rendered_prompt" else json.dumps(row.get("decision", {}).get("messages", []), ensure_ascii=False)
        own = tasks.get(row.get("agent_id"), {})
        return own.get("marker", "__missing__") in text and all(task["marker"] not in text for task in TASKS if task["agent_id"] != row.get("agent_id"))
    enough = len(rows) == len(planned) and len(rows) >= 2
    return {
        "planned_sequence_completed": enough and [row.get("agent_id") for row in rows] == planned,
        "600_logical_identities": report.get("registered_identities") == 600,
        "one_owned_pid": enough and len({row.get("owned_pid") for row in rows}) == 1,
        "one_model_load": native.get("counters", {}).get("loads") == 1,
        "native_request_count": native.get("counters", {}).get("requests") == len(planned),
        "one_physical_slot": native.get("scheduler", {}).get("physical_limit") == 1,
        "natural_eos_each": enough and all((row.get("final") or {}).get("stop_type") == "eos" for row in rows),
        "complete_transcripts": enough and all(row.get("completion_contract", {}).get("passed") is True for row in rows),
        "useful_final_each": enough and all(len(row.get("answer_checks", {})) > 1 and all(value for key,value in row["answer_checks"].items() if key != "scope") for row in rows),
        "host_contexts_isolated": enough and all(isolated(row, "decision") for row in rows),
        "native_prompts_isolated": enough and all(isolated(row, "rendered_prompt") for row in rows),
        "actual_prompt_ids_each": enough and all(isinstance(row.get("prompt_token_ids"), list) and bool(row["prompt_token_ids"]) and all(type(token) is int and token >= 0 for token in row["prompt_token_ids"]) for row in rows),
        "process_io_each": enough and all(row.get("process_delta", {}).get("available") is True for row in rows),
        "full_prompts_no_truncation": enough and all((row.get("final") or {}).get("truncated") is False for row in rows),
        "same_mission_on_agent1_return": planned != [1,600,1] or enough and rows[0].get("mission_id") == rows[2].get("mission_id"),
        "cache_disabled_each": enough and all(row.get("body", {}).get("cache_prompt") is False and row.get("body", {}).get("id_slot") == 0 for row in rows),
        "normal_native_exit": report.get("owned_server_stopped") is True and report.get("owned_server_returncode") == 0,
        "stable_ac": report.get("power_continuity_validated") is True,
        "weights_integrity": native.get("checkpoint", {}).get("integrity_verified") is True,
        "runtime_integrity": native.get("runtime_artifacts_verified") is True,
        "resource_guards_clear": bool(final_native) and native.get("failure") is None and final_native.get("failure") is None,
        "native_reasoning_preserved": "reasoning_effort" in report and report["reasoning_effort"] in (None, "max"),
        "runner_source_unchanged": report.get("runner_source_identity", {}).get("passed") is True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "prefill", "agents"), default="plan")
    parser.add_argument("--microbatch", type=int, choices=BATCHES, default=8)
    parser.add_argument("--logical-batch", type=int)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", choices=("native", "low", "high", "max"), default="native")
    parser.add_argument("--agent-order", choices=("1,600", "600,1", "1,600,1"), default="1,600,1")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--request-seconds", type=int, choices=(1800,2700), default=1800)
    args = parser.parse_args()
    if args.mode == "plan":
        report = plan(args.config, requested=args.microbatch, gpu_layers=args.gpu_layers)
        args.output.mkdir(parents=True, exist_ok=False)
        _write(args.output / "report.json", report)
        print(json.dumps(report))
        return 0 if report["status"] == "planned" else 2
    report = run_campaign(args.config, args.output, 1 if args.mode == "prefill" else args.tokens,
        mode=args.mode, microbatch=args.microbatch, logical_batch=args.logical_batch,
        gpu_layers=args.gpu_layers, reasoning_effort=None if args.reasoning_effort == "native" else args.reasoning_effort,
        agent_order=tuple(map(int,args.agent_order.split(","))), max_seconds=args.max_seconds,
        request_seconds=args.request_seconds)
    print(json.dumps({"status": report["status"], "functional_two_agents_qualified": report["functional_two_agents_qualified"],
        "prefill_measurements_complete": report["prefill_measurements_complete"], "report": str(args.output / "report.json")}))
    return 0 if report["functional_two_agents_qualified"] or args.mode == "prefill" and report["prefill_measurements_complete"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
