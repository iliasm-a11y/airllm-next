"""Two real logical agents through one guarded RONIN CPU server.

This integration probe is separate from native throughput/fidelity campaigns.
It neither qualifies all 600 identities nor captures complete numerical state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "air_llm"), str(REPO / "integrations/ronin/runtime"),
                str(REPO / "integrations/ronin/engine")]
from ronin_runtime import Runtime

BENCHMARK_LOCK = Path("/tmp/airllm-glm53-benchmark.lock")


class ProbeInterrupted(RuntimeError):
    code = "CANCELLED"

    def __init__(self, signum):
        super().__init__("Integration probe interrupted by " + signal.Signals(signum).name)
        self.signum = signum


class PowerContextError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def observe_power_context():
    """Read macOS lid and power state only; battery power is informational."""
    observation = {"platform": sys.platform, "guard_applicable": sys.platform == "darwin",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "monotonic_start": time.monotonic(), "clamshell_closed": None,
                   "power_source": None, "battery_percent": None, "sources": []}
    if not observation["guard_applicable"]:
        observation["scope"] = "macOS lid guard not applicable on this platform"
        observation["monotonic_end"] = time.monotonic()
        return observation
    queries = (("ioreg_clamshell", ["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"]),
               ("pmset_battery", ["/usr/bin/pmset", "-g", "batt"]))
    for label, command in queries:
        source = {"name": label, "command": command}
        observation["sources"].append(source)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
            if len(result.stdout.encode("utf-8")) > 64 * 1024:
                raise ValueError("Power observation exceeds its metadata budget")
            if label == "ioreg_clamshell":
                values = re.findall(r'"AppleClamshellState"\s*=\s*(Yes|No|true|false)\b',
                                    result.stdout, flags=re.IGNORECASE)
                source["observed_values"] = values
                source["matching_lines"] = [line.strip() for line in result.stdout.splitlines()
                                             if '"AppleClamshellState"' in line]
                if len(values) == 1:
                    observation["clamshell_closed"] = values[0].lower() in ("yes", "true")
            else:
                source["output"] = result.stdout.strip()
                power = re.search(r"Now drawing from '([^']+)'", result.stdout)
                percent = re.search(r"\b(\d{1,3})%;", result.stdout)
                observation["power_source"] = power[1] if power else None
                if percent and 0 <= int(percent[1]) <= 100:
                    observation["battery_percent"] = int(percent[1])
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            source.update(error_code=type(error).__name__, error=str(error))
    observation["scope"] = "read-only macOS lid admission; power source and battery are observations, not an AC requirement"
    observation["monotonic_end"] = time.monotonic()
    return observation


def _cleanup(runtime, report):
    """Attempt every cleanup step, retaining the actual owned process handle."""
    errors = []

    def attempt(step, action):
        try:
            return action()
        except Exception as error:
            errors.append({"step": step, "error_code": getattr(error, "code", type(error).__name__),
                           "error": str(error)})
            return None

    if runtime is not None:
        engine = getattr(runtime, "engine", None)
        model = getattr(engine, "_model", None)
        process = getattr(model, "_process", None)
        report["owned_pid_before_cleanup"] = getattr(process, "pid", None)
        if engine is not None:
            report["engine_before_cleanup"] = attempt("engine_status", engine.status)
        attempt("runtime_shutdown", lambda: runtime.dispatch("shutdown", {}))
        # Native close is idempotent and still runs if the high-level shutdown fails.
        if engine is not None:
            attempt("engine_close", engine.close)
        if process is not None:
            returncode = attempt("owned_process_poll", process.poll)
            report["owned_server_returncode"] = returncode
            report["owned_server_stopped"] = returncode is not None
            report["owned_server_observation"] = "poll_of_process_captured_before_close"
            if returncode is None:
                errors.append({"step": "owned_server_exit", "error_code": "SERVER_STILL_RUNNING",
                               "error": "The observed owned native process has no exit code after cleanup"})
        else:
            # A resource guard or cancelled SSE may have already closed the
            # model. Use its retained poll evidence, never handle=None alone.
            observation = attempt("native_exit_observation", lambda: getattr(model, "last_owned_process_exit", None))
            report["last_owned_process_exit"] = observation
            valid_pid = isinstance(observation, dict) and type(observation.get("pid")) is int and observation["pid"] > 0
            valid_exit = (valid_pid and observation.get("observation") == "popen_poll_of_owned_process"
                          and observation.get("exit_observed") is True
                          and type(observation.get("returncode")) is int)
            report["owned_server_stopped"] = True if valid_exit else False if valid_pid else None
            report["owned_server_observation"] = ("retained_native_owned_process_poll" if valid_exit
                else "retained_native_process_exit_unconfirmed" if valid_pid
                else "no_owned_native_process_observed_before_cleanup")
            if valid_pid:
                report["owned_pid_before_cleanup"] = observation["pid"]
                report["owned_server_returncode"] = observation.get("returncode")
            if valid_pid and not valid_exit:
                errors.append({"step": "owned_server_exit", "error_code": "SERVER_EXIT_UNCONFIRMED",
                               "error": "The retained owned process observation does not confirm an exit"})
            elif not valid_exit and report["status"] == "completed":
                errors.append({"step": "owned_server_exit", "error_code": "SERVER_NOT_OBSERVED",
                               "error": "No owned native process was available to attest cleanup"})
        database = getattr(getattr(runtime, "store", None), "db", None)
        if database is not None:
            attempt("database_close", database.close)
    if errors:
        report["cleanup_errors"] = errors
        report["status_before_cleanup"] = report["status"]
        report["status"] = "failed"


def run_probe(config_path, output, tokens=4, *, runtime_factory=None):
    """Run in a new isolated output directory; fake factories are test-only."""
    args = argparse.Namespace(config=Path(config_path), output=Path(output), tokens=tokens)
    if type(tokens) is not int or not 1 <= tokens <= 32:
        raise ValueError("integration output budget must be 1..32")
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("A structured runtime configuration is required")
    gpu = config.get("gpu_layers", 0)
    if type(gpu) is not int or gpu != 0 or type(config.get("threads")) is not int or config["threads"] != 10:
        raise ValueError("This probe requires the CPU10/GPU0 reference")
    limits = dict(config.get("limits", {}))
    for key, cap in {"max_rss_bytes": int(8.5 * 1024**3), "max_swap_growth_bytes": 32 * 1024**2,
                     "warning_swap_growth_bytes": 32 * 1024**2}.items():
        requested = limits.get(key, cap)
        if type(requested) is not int or requested <= 0:
            raise ValueError("A positive integer guard limit is required: " + key)
        limits[key] = min(requested, cap)
    config["limits"] = limits
    # Config paths are resolved before writing the isolated run configuration.
    for key in ("runtime_binary", "runtime_manifest_file", "verification_file", "model_path"):
        value = config.get(key)
        if value:
            path = Path(value).expanduser()
            config[key] = str(path if path.is_absolute() else (args.config.parent / path).resolve())
    started = time.monotonic()
    report = {"scope": "two real agent/mission contexts, shared CPU server; no full-state or 600-agent qualification",
              "config_sha256": hashlib.sha256(raw).hexdigest(), "status": "running", "requests": [],
              "guard_limits": limits, "native_fidelity_qualified": False}
    runtime = None
    with BENCHMARK_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        config["state_dir"] = str(args.output.resolve() / "engine-state")
        selected_config = args.output.resolve() / "run-config.json"
        selected_config.write_text(json.dumps(config, indent=2) + "\n")
        previous = os.environ.get("RONIN_GLM_CONFIG")
        os.environ["RONIN_GLM_CONFIG"] = str(selected_config)
        previous_handlers = {}
        interrupt_state = {"cleaning": False}

        def interrupt(signum, frame):
            # Repeated signals must not interrupt native termination/DB cleanup.
            if not interrupt_state["cleaning"]:
                raise ProbeInterrupted(signum)

        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.signal(signum, interrupt)
            report["interrupt_handlers_installed"] = bool(previous_handlers)
            report["power_context_before_runtime"] = power = observe_power_context()
            if power.get("guard_applicable") and power.get("clamshell_closed") is not False:
                closed = power.get("clamshell_closed") is True
                raise PowerContextError("POWER_CONTEXT_CLOSED" if closed else "POWER_CONTEXT_UNKNOWN",
                                        "Open the Mac lid before inference" if closed else
                                        "The Mac lid state could not be established before inference")
            runtime = (runtime_factory or Runtime)(args.output.resolve() / "state")
            report["registered_identities"] = len(runtime.dispatch("agent_identities", {}))
            for agent, marker in ((1, "ALPHA"), (600, "OMEGA")):
                mission = runtime.create({"prompt": "Réponds uniquement par OK."})
                runtime.dispatch("agent_memory", {"id": mission["id"], "agent_id": agent,
                                                   "text": "Repère privé de cette mission : " + marker})
                try:
                    result = runtime.dispatch("generate", {"id": mission["id"], "agent_id": agent,
                                                         "max_new_tokens": args.tokens, "reasoning_effort": "low"})
                    decision = runtime.dispatch("agent_decision", {"id": mission["id"], "agent_id": agent,
                                                                  "decision_id": result["decision_id"]})
                    state = runtime.engine.status()
                    report["requests"].append({"agent_id": agent, "mission_id": mission["id"],
                                               "result": result, "decision": decision, "engine_after": state,
                                               "owned_pid": runtime.engine._model._process.pid})
                except Exception as error:
                    report["requests"].append({"agent_id": agent, "mission_id": mission["id"],
                                               "error_code": getattr(error, "code", type(error).__name__),
                                               "error": str(error),
                                               "context": runtime.dispatch("agent_context", {"id": mission["id"], "agent_id": agent})})
                    raise
            pids = {row["owned_pid"] for row in report["requests"]}
            state = runtime.engine.status()
            decisions = [row["decision"] for row in report["requests"]]
            # These checks examine host context isolation and actual native IDs;
            # identical-looking text is not numerical equivalence evidence.
            report["checks"] = {
                "one_native_pid": len(pids) == 1,
                "one_model_load": state["native"]["counters"]["loads"] == 1,
                "two_native_requests": state["native"]["counters"]["requests"] == 2,
                "actual_token_ids": all(row["result"]["tokens"] for row in report["requests"]),
                "different_context_hashes": decisions[0]["context_sha256"] != decisions[1]["context_sha256"],
                "host_contexts_isolated": ("ALPHA" in json.dumps(decisions[0]["messages"])
                                          and "OMEGA" in json.dumps(decisions[1]["messages"])
                                          and "OMEGA" not in json.dumps(decisions[0]["messages"])
                                          and "ALPHA" not in json.dumps(decisions[1]["messages"])),
                "one_physical_slot": state["native"]["scheduler"]["physical_limit"] == 1,
            }
            report["status"] = "completed" if all(report["checks"].values()) else "rejected"
        except ProbeInterrupted as error:
            report.update(status="cancelled", error_code=error.code, error=str(error),
                          interrupted_signal=error.signum)
        except Exception as error:
            report.update(status="failed", error_code=getattr(error, "code", type(error).__name__), error=str(error))
        finally:
            interrupt_state["cleaning"] = True
            try:
                _cleanup(runtime, report)
            finally:
                try:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                finally:
                    if previous is None:
                        os.environ.pop("RONIN_GLM_CONFIG", None)
                    else:
                        os.environ["RONIN_GLM_CONFIG"] = previous
                    report["elapsed_seconds"] = time.monotonic() - started
                    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 32:
        parser.error("integration output budget must be 1..32")
    report = run_probe(args.config, args.output, args.tokens)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
