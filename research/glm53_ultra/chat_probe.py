"""Guarded CPU chat/EOS probe with one user message and the native GGUF template.

An EOS result qualifies this bounded request only. No new full-logit, expert-route
or recurrent-state fidelity is established by its text or generated token IDs.
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

from .agent_probe import BENCHMARK_LOCK, ProbeInterrupted, _cleanup
from research.glm53_gpu_v3.completion_contract import validate_completion
from research.glm53_gpu_v3.power_state import PowerGate, ScopedAwake, power_evidence_matches

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "air_llm"), str(REPO / "integrations/ronin/engine")]
from adapter import Engine


class ChatProbeError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _configuration(path, tokens):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("A structured CPU runtime configuration is required")
    for key, expected in (("threads", 10), ("gpu_layers", 0), ("context_tokens", 512), ("batch_size", 8)):
        value = config.get(key, expected if key != "threads" else None)
        if type(value) is not int or value != expected:
            raise ValueError("This probe requires CPU10/GPU0/context512/batch8: " + key)
        config[key] = expected
    if type(tokens) is not int or not 1 <= tokens <= 128:
        raise ValueError("Chat generation budget must be an integer in1..128")
    if config.get("cpu_moe", False) is not False or config.get("moe_cache_mib", 0) != 0:
        raise ValueError("The CPU reference must not enable an expert GPU cache")
    config.update(cpu_moe=False, moe_cache_mib=0)
    config["allow_unqualified_gpu"] = False
    limits = dict(config.get("limits", {}))
    for key, cap in (("max_rss_bytes", int(8.5 * 1024**3)),
                     ("max_swap_growth_bytes", 32 * 1024**2),
                     ("warning_swap_growth_bytes", 32 * 1024**2)):
        value = limits.get(key, cap)
        if type(value) is not int or value <= 0:
            raise ValueError("Positive integer guard limit required: " + key)
        limits[key] = min(value, cap)
    config["limits"] = limits
    for key in ("model_path", "runtime_binary", "runtime_manifest_file", "verification_file"):
        value = config.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError("Nonempty path required: " + key)
            selected = Path(value).expanduser()
            config[key] = str((path.parent / selected).resolve() if not selected.is_absolute() else selected.resolve())
    if not all(config.get(key) for key in ("model_path", "runtime_binary", "verification_file")):
        raise ValueError("Model, runtime and a copyable verification receipt are required")
    return config, hashlib.sha256(raw).hexdigest()


def _graceful_stop(process, report):
    """Signal only the retained owned Popen; Native close remains the fallback."""
    if process is None:
        return
    observation = {"pid": process.pid, "requested_signal": None, "exit_code": process.poll()}
    report["graceful_shutdown"] = observation
    if observation["exit_code"] is None:
        try:
            process.send_signal(signal.SIGINT)
            observation["requested_signal"] = "SIGINT"
            observation["exit_code"] = process.wait(timeout=5)
        except Exception as error:
            observation["error"] = type(error).__name__ + ": " + str(error)


def run_probe(config_path, output, tokens=128, *, prompt="Réponds uniquement par OK.",
              engine_factory=None, gate_factory=None, awake_factory=None, monitor_interval=.5):
    config, config_hash = _configuration(config_path, tokens)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty user request is required")
    if not 0 < monitor_interval <= .5:
        raise ValueError("Power sampling interval must be positive and <=0.5s")
    started = time.perf_counter()
    request = {"body": {"n_predict": tokens, "messages": [{"role": "user", "content": prompt}],
                         "chat_template_kwargs": {"reasoning_effort": "low"}, "cache_prompt": False},
               "events": [], "tokens": [], "text": "", "final": None}
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "one short-user chat through the native GGUF template; no full-logit/route/state fidelity",
              "status": "running", "eos_qualified": False, "native_fidelity_qualified": False,
              "config_sha256": config_hash, "guard_limits": config["limits"],
              "context_tokens": 512, "reasoning_effort": "low", "request": request,
              "reasoning_effort_scope": "explicit conditioning for this request; global default unchanged"}
    engine = process = awake = watcher = None
    stop, cancel = threading.Event(), threading.Event()
    monitor_errors = []
    gate = (gate_factory or PowerGate)("ac")
    handlers, interrupt_state = {}, {"cleaning": False}

    def interrupt(signum, frame):
        if not interrupt_state["cleaning"]:
            cancel.set()
            raise ProbeInterrupted(signum)

    def monitor():
        while not stop.wait(monitor_interval):
            try:
                awake.check()
                if not gate.sample()["passed"]:
                    raise ChatProbeError("POWER_CONTEXT_CHANGED", "Power/lid/clock continuity rejected")
            except Exception as error:
                monitor_errors.append({"code": getattr(error, "code", type(error).__name__), "error": str(error)})
                cancel.set()
                return

    with BENCHMARK_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, mode=0o700)
        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupt)
            report["power_preflight"] = gate.sample()
            if not report["power_preflight"]["passed"]:
                raise ChatProbeError("POWER_CONTEXT_UNAVAILABLE", "An open lid and stable AC power are required for this measurement")
            receipt_path = Path(config["verification_file"])
            if receipt_path.stat().st_size > 4 * 1024**2:
                raise ValueError("Verification receipt exceeds metadata budget")
            receipt = receipt_path.read_bytes()
            copied = output / "model-verification.json"
            copied.write_bytes(receipt)
            copied.chmod(0o600)
            report["input_model_receipt"] = {"path": str(receipt_path), "sha256": hashlib.sha256(receipt).hexdigest(),
                                              "copied_to": str(copied), "source_modified_by_probe": False}
            model_path = config.pop("model_path")
            config.update(verification_file=str(copied), state_dir=str(output / "engine-state"))
            _write(output / "run-config.json", {"model_path": model_path, **config})
            engine = (engine_factory or Engine)(config=config)
            # Tie the idle assertion to this CLI PID so it also covers model load.
            # The helper ends automatically if this probe dies; it cannot bypass a closed lid.
            owner = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
            awake = (awake_factory or ScopedAwake)(owner)
            awake.__enter__()
            watcher = threading.Thread(target=monitor, daemon=True, name="chat-probe-power")
            watcher.start()
            engine.load(model_path, cancel=cancel)
            process = getattr(getattr(engine, "_model", None), "_process", None)
            if process is None or process.poll() is not None:
                raise ChatProbeError("SERVER_NOT_RUNNING", "No live owned native process after model load")
            report.update(owned_pid=process.pid, load_seconds=time.perf_counter()-started,
                          engine_after_load=engine.status())
            _write(output / "report.json", report)
            request_start = time.perf_counter()
            with (output / "events.jsonl").open("x", encoding="utf-8") as transcript:
                os.fchmod(transcript.fileno(), 0o600)
                for event in engine.stream(request["body"]["messages"], max_new_tokens=tokens,
                                           cancel=cancel, reasoning_effort="low"):
                    row = {"t": time.perf_counter()-request_start, "response": event}
                    request["events"].append(row)
                    transcript.write(json.dumps(row, ensure_ascii=False) + "\n")
                    transcript.flush()
                    ids, content = event.get("tokens", []), event.get("content", "")
                    if not isinstance(ids, list) or not isinstance(content, str):
                        raise ChatProbeError("INVALID_NATIVE_EVENT", "Native text/token event is malformed")
                    request["tokens"].extend(ids)
                    request["text"] += content
                    if event.get("stop"):
                        request["final"] = event
                    if cancel.is_set():
                        raise ChatProbeError("POWER_CONTEXT_CHANGED" if monitor_errors else "CANCELLED", "Probe cancelled under supervision")
            request["wall_seconds"] = time.perf_counter()-request_start
            report["engine_after_stream"] = engine.status()
            report["completion_contract"] = validate_completion(request, tokens)
            if not report["completion_contract"]["passed"]:
                raise ChatProbeError("INCOMPLETE_STREAM", "; ".join(report["completion_contract"]["reasons"]))
            report["status"] = "completed" if request["final"]["stop_type"] == "eos" else "incomplete"
        except ProbeInterrupted as error:
            report.update(status="cancelled", error_code=error.code, error=str(error), interrupted_signal=error.signum)
        except Exception as error:
            report.update(status="failed", error_code=getattr(error, "code", type(error).__name__), error=str(error))
        finally:
            interrupt_state["cleaning"] = True
            stop.set()
            if watcher is not None:
                watcher.join(timeout=6)
                if watcher.is_alive():
                    monitor_errors.append({"code": "POWER_MONITOR_NOT_STOPPED", "error": "Power observer did not finish"})
            try:
                if engine is not None:
                    process = process or getattr(getattr(engine, "_model", None), "_process", None)
                    try:
                        _graceful_stop(process, report)
                    finally:
                        shim = SimpleNamespace(engine=engine, dispatch=lambda *unused: engine.close())
                        _cleanup(shim, report)
                        report["engine_after_cleanup"] = engine.status()
            except Exception as error:
                report.update(status="failed", cleanup_error=type(error).__name__ + ": " + str(error))
            finally:
                try:
                    if awake is not None:
                        awake.close()
                except Exception as error:
                    report.update(status="failed", awake_cleanup_error=type(error).__name__ + ": " + str(error))
                try:
                    gate.sample(final=True)
                except Exception as error:
                    report.update(status="failed", final_power_error=type(error).__name__ + ": " + str(error))
                report["power_gate"] = gate.summary()
                if awake is not None:
                    report["power_gate"]["idle_assertion"] = awake.record
                report["idle_assertion_scope"] = "owned probe CLI PID across load and generation; native server PID is separate"
                report["power_continuity_validated"] = power_evidence_matches(
                    {**report, "owned_pid": os.getpid()}, "ac")
                if monitor_errors:
                    report.update(status="failed", power_monitor_errors=monitor_errors)
                stream_native = report.get("engine_after_stream", {}).get("native", {})
                final_native = report.get("engine_after_cleanup", {}).get("native", {})
                # A failed request has no after-stream snapshot. Integrity and
                # model-load facts can still be established independently of
                # request completion; keep the snapshot source explicit.
                native, native_source = {}, None
                for key in ("engine_after_stream", "engine_after_cleanup", "engine_after_load"):
                    candidate = report.get(key, {}).get("native", {})
                    if isinstance(candidate, dict) and candidate:
                        native, native_source = candidate, key + ".native"
                        break
                report["native_evidence_source"] = native_source
                checks = {"natural_eos": bool(request["final"] and request["final"].get("stop_type") == "eos"),
                          "complete_transcript": report.get("completion_contract", {}).get("passed") is True,
                          "normal_native_exit": report.get("owned_server_returncode") == 0 and report.get("owned_server_stopped") is True,
                          "power_continuity": report["power_continuity_validated"],
                          "model_integrity": native.get("checkpoint", {}).get("integrity_verified") is True,
                          "runtime_artifacts": native.get("runtime_artifacts_verified") is True,
                          "resource_guards_clear": bool(final_native) and stream_native.get("failure") is None and final_native.get("failure") is None,
                          "one_model_load": native.get("counters", {}).get("loads") == 1,
                          "one_native_request": native.get("counters", {}).get("requests") == 1,
                          "one_physical_slot": native.get("scheduler", {}).get("physical_limit") == 1}
                report["eos_checks"] = checks
                report["eos_qualified"] = report["status"] == "completed" and all(checks.values())
                if report["status"] == "completed" and not report["eos_qualified"]:
                    report.update(status="failed", error_code="EOS_QUALIFICATION_REJECTED")
                report["elapsed_seconds"] = time.perf_counter()-started
                copied_receipt = output / "model-verification.json"
                if copied_receipt.is_file():
                    report["output_model_receipt"] = {"path": str(copied_receipt),
                        "sha256": hashlib.sha256(copied_receipt.read_bytes()).hexdigest()}
                try:
                    _write(output / "report.json", report)
                finally:
                    for signum, previous in handlers.items():
                        signal.signal(signum, previous)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Réponds uniquement par OK.")
    args = parser.parse_args()
    report = run_probe(args.config, args.output, args.tokens, prompt=args.prompt)
    print(json.dumps({"status": report["status"], "eos_qualified": report["eos_qualified"],
                      "report": str(args.output / "report.json")}))
    return 0 if report["eos_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
