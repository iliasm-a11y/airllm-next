"""Bounded, fail-closed CPU campaign for two useful, isolated RONIN decisions.

No model starts on battery. Cache controls are research-only. Raw native SSE,
rendered prompts and their actual native token IDs are observed in existing
calls, without extra inference or tokenization. Functional completion is kept
separate from numerical reset/restore equivalence, which this runner cannot
establish without an instrumented native-state campaign.
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
from types import SimpleNamespace
import psutil

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "air_llm"), str(REPO / "integrations/ronin/runtime"),
                str(REPO / "integrations/ronin/engine")]
from airllm.nextgen.glm_native import NativeGlmModel, NativeLimits, inspect_native_checkpoint
from ronin_runtime import Runtime
from research.glm53_ultra.agent_probe import BENCHMARK_LOCK, ProbeInterrupted, _cleanup
from research.glm53_ultra.chat_probe import _graceful_stop, _write
from research.glm53_gpu_v3.completion_contract import validate_completion
from research.glm53_gpu_v3.power_state import PowerGate, ScopedAwake, power_evidence_matches
from research.glm53_gpu_v3.process_io import ProcessIOObserver, compare_process

TASKS = (
    {"agent_id": 1, "marker": "RONIN-ALPHA-713", "prompt":
     "Un atelier prépare 17 lots de 24 pièces et dispose de 420 pièces. Calcule la quantité utilisée et la marge restante, puis vérifie si le stock suffit. Donne le calcul et une conclusion utile, en rappelant le repère privé de ta mémoire."},
    {"agent_id": 600, "marker": "RONIN-OMEGA-962", "prompt":
     "Pour une commande de 12 unités, le fournisseur A facture 35 € de frais fixes puis 8 € par unité ; B facture 10 € puis 11 € par unité. Compare les coûts, choisis le moins cher et calcule l'économie. Donne le calcul et ta justification, en rappelant le repère privé de ta mémoire."},
)
CACHE_FLAGS = ["--cache-ram", "0", "--ctx-checkpoints", "0", "--no-cache-idle-slots",
               "--cache-reuse", "0", "--no-cache-prompt", "--spec-type", "none"]


def memory_for(task):
    return ("Repère privé de cette mission : " + task["marker"]
            + ". Ce repère appartient uniquement à cette identité et à cette mission.")


class QualificationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class NoCacheResearchModel(NativeGlmModel):
    """Explicitly disabled server caches; never selected by production Engine."""
    def _build_command(self, port):
        gate = getattr(self, "_research_power_gate", None)
        if gate is not None and not gate.sample()["passed"]:
            raise QualificationError("POWER_CONTEXT_CHANGED", "AC/lid/clock rejected immediately before native command construction")
        command = super()._build_command(port) + CACHE_FLAGS
        self.research_constructed_command = {
            "argv": command, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "constructed by research _build_command; construction alone does not attest Popen spawn or OS-observed argv"}
        return command


def observe_owned_argv(process, expected_executable, *, process_factory=psutil.Process):
    """Read only an existing owned child's argv, with identity/lifetime brackets."""
    if process is None or process.poll() is not None:
        return {"available": False, "reason": "no live owned Popen child"}
    try:
        target = process_factory(process.pid)
        started = target.create_time()
        executable = target.exe()
        argv = target.cmdline()
        if not target.is_running() or target.create_time() != started or process.poll() is not None:
            raise ValueError("owned PID lifetime changed during argv observation")
        if os.path.realpath(executable) != os.path.realpath(expected_executable):
            raise ValueError("owned executable differs from pinned runtime")
        if not argv or sum(len(str(part)) for part in argv) > 65536:
            raise ValueError("argv unavailable or exceeds metadata budget")
        return {"available": True, "pid": process.pid, "create_time": started,
                "executable": executable, "argv": argv, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "OS argv of one retained live owned Popen child; identity and lifetime bracketed; no process enumeration"}
    except (OSError, ValueError, psutil.Error) as error:
        return {"available": False, "pid": process.pid, "reason": type(error).__name__ + ": " + str(error)}


def configuration(path, tokens, *, request_seconds=1800):
    if type(request_seconds) is not int or request_seconds not in (1800, 2700):
        raise ValueError("Explicit request budget must be 1800s or 2700s")
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("A structured CPU configuration is required")
    for key, expected in (("threads", 10), ("gpu_layers", 0), ("batch_size", 8)):
        if type(config.get(key)) is not int or config[key] != expected:
            raise ValueError("Required CPU10/GPU0/batch8 reference: " + key)
    context = config.get("context_tokens")
    if type(context) is not int or not 1024 <= context <= 2048:
        raise ValueError("Use an explicit context1024..2048; full agent prompts are never truncated")
    if type(tokens) is not int or not 128 <= tokens <= 512 or tokens >= context:
        raise ValueError("An output budget128..512 fitting the full context is required")
    if config.get("cpu_moe", False) is not False or config.get("moe_cache_mib", 0) != 0:
        raise ValueError("Expert GPU caches must remain off")
    config.update(cpu_moe=False, moe_cache_mib=0, allow_unqualified_gpu=False)
    limits = dict(config.get("limits", {}))
    for key, cap in (("max_rss_bytes", int(8.5 * 1024**3)),
                     ("max_swap_growth_bytes", 32 * 1024**2),
                     ("warning_swap_growth_bytes", 32 * 1024**2)):
        value = limits.get(key, cap)
        if type(value) is not int or value <= 0:
            raise ValueError("A positive integer guard is required: " + key)
        limits[key] = min(cap, value)
    config["limits"] = limits
    defaults = {**vars(NativeLimits()), "request_timeout_seconds": request_seconds}
    if set(limits) - defaults.keys():
        raise ValueError("Unknown resource guard settings")
    # Preserve every existing pressure/identity-watchdog protection. More
    # reserve/headroom is stricter; shorter polling/grace/timeouts are stricter.
    for key, default in defaults.items():
        value = limits.get(key, default)
        if type(value) not in (int, float) or not 0 < value < float("inf"):
            raise ValueError("Positive finite resource guard required: " + key)
        if type(default) is int and type(value) is not int:
            raise ValueError("Integer resource guard required: " + key)
        limits[key] = (max(value, default) if key in (
            "reserve_bytes", "base_workspace_bytes", "warning_headroom_bytes") else min(value, default))
    NativeLimits(**limits)
    for key in ("model_path", "runtime_binary", "runtime_manifest_file", "verification_file"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("A pinned artifact path is required: " + key)
        selected = Path(value).expanduser()
        config[key] = str((path.parent / selected).resolve() if not selected.is_absolute() else selected.resolve())
    return config, hashlib.sha256(raw).hexdigest()


def metadata_preflight(config):
    """Read metadata and existing receipts, never hash the110GB shard here."""
    files = [Path(config[k]) for k in ("runtime_binary", "runtime_manifest_file", "verification_file")]
    files += [REPO / name for name in (
        "air_llm/airllm/nextgen/glm_native.py", "integrations/ronin/engine/adapter.py",
        "integrations/ronin/runtime/agent_context.py", "integrations/ronin/runtime/ronin_runtime.py",
        "research/glm53_shared_prefill/qualify_agents.py")]
    hashes = {}
    for path in files:
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Metadata artifact exceeds64MiB: " + str(path))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = inspect_native_checkpoint(config["model_path"], config["verification_file"], verify_hashes=False)
    return {"sha256": hashes, "checkpoint": checkpoint,
            "scope": "metadata and unchanged-file receipt reuse; no new full-shard SHA scan; native load revalidates before inference"}


def verify_cache_flags(binary):
    result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=15, check=True)
    help_text = result.stdout + result.stderr
    needed = ["--cache-ram", "--ctx-checkpoints", "--no-cache-idle-slots", "--cache-reuse", "--no-cache-prompt", "--spec-type"]
    if any(flag not in help_text for flag in needed):
        raise QualificationError("CACHE_CONTROLS_UNAVAILABLE", "Pinned runtime does not expose every required no-cache control")
    return {"flags": CACHE_FLAGS, "help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
            "scope": "CLI options verified; no-cache command recorded again before load; distinct from old PR20 default-cache profile"}


def final_answer(text):
    """Split emitted native thinking delimiters only; never synthesize reasoning."""
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    else:
        # The native GLM template already opens its reasoning span. Without
        # an emitted closing delimiter, text can still be reasoning at EOS.
        return ""
    return text.strip()


def explicit_conclusion(agent_id, final):
    """Require an affirmative bounded clause; raw answers still need review."""
    clauses = re.split(r"[.!?;,\n]+", " ".join(final.casefold().split()))
    negated = re.compile(r"\b(?:ne|pas|jamais|aucun|aucune)\b|\bn['’]")
    affirmative = [clause for clause in clauses if not negated.search(clause)]
    if agent_id == 1:
        stock = r"\bstock(?:\s+(?:actuel|disponible))?\s+(?:(?:est|reste|sera)\s+(?:donc\s+)?)?(?:suffit|suffisant)\b"
        contradicted = any("stock" in clause and (negated.search(clause) or "insuffisant" in clause)
                          and re.search(r"suffi", clause) for clause in clauses)
        return not contradicted and any(re.search(stock, clause) for clause in affirmative)

    def chooses(supplier, clause):
        label = r"\b" + supplier + r"\b"
        cheaper = label + r"\s*(?:est|reste|revient|:)\s*(?:donc\s+)?(?:le\s+)?(?:moins\s+cher|moins\s+coûteux|plus\s+économique)\b"
        choice = (r"\b(?:chois(?:is|issons|it)|reti(?:ens|ent)|retenons|privilégi(?:e|ons)|recommand(?:e|ons))"
                  r"\s+(?:donc\s+)?(?:(?:le\s+)?fournisseur\s+)?" + label)
        return bool(re.search(cheaper, clause) or re.search(choice, clause))

    return (any(chooses("a", clause) for clause in affirmative)
            and not any(chooses("b", clause) for clause in affirmative))


def answer_checks(task, text):
    final = final_answer(text)
    other = next(t["marker"] for t in TASKS if t["agent_id"] != task["agent_id"])
    numbers = set(re.findall(r"(?<!\d)\d+(?!\d)", final))
    expected = {"408", "12"} if task["agent_id"] == 1 else {"131", "142", "11"}
    rationale = bool(re.search(r"[=×*]|(?:calcul|car |donc|reste|suffi|économ|moins cher|marge)", final, re.I))
    conclusion = explicit_conclusion(task["agent_id"], final)
    return {"nonempty_final_answer": bool(final), "own_marker_in_final": task["marker"] in final,
            "peer_marker_absent": other not in text, "expected_result_numbers": expected <= numbers,
            "visible_calculation_or_justification": rationale, "correct_conclusion": conclusion,
            "scope": "conservative bounded heuristics; raw native final answer requires human review, not universal reasoning quality"}


def capture_native(model, row, output, observer):
    """Observe exact existing metadata calls and SSE, with no duplicate request."""
    original_http, original_events = model._http_json, model._events
    def http(method, path, payload=None, timeout=30):
        value = original_http(method, path, payload, timeout)
        if path == "/apply-template":
            row["template_request"] = payload
            row["rendered_prompt"] = value.get("prompt")
        elif path == "/tokenize":
            row["tokenize_request"] = payload
            row["prompt_token_ids"] = value.get("tokens")
        return value
    def events(payload, cancel):
        row["body"] = payload
        row["io_before"] = observer.snapshot()
        started = time.perf_counter()
        try:
            with (output / ("agent-" + str(row["agent_id"]) + "-events.jsonl")).open("x", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                for event in original_events(payload, cancel):
                    entry = {"t": time.perf_counter() - started, "response": event}
                    row["events"].append(entry)
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    stream.flush()
                    row["tokens"].extend(event.get("tokens", []))
                    row["text"] += event.get("content", "")
                    if event.get("stop"):
                        row["final"] = event
                    yield event
        except Exception as error:
            row["native_error_details"] = getattr(error, "details", None)
            raise
        finally:
            row["wall_seconds"] = time.perf_counter() - started
            row["io_after"] = observer.snapshot()
            row["process_delta"] = compare_process(row["io_before"], row["io_after"])
    model._http_json, model._events = http, events
    return lambda: (setattr(model, "_http_json", original_http), setattr(model, "_events", original_events))


def run_campaign(config_path, output, tokens=256, *, max_seconds=7200, request_seconds=1800,
                 agent_order=(1, 600), runtime_factory=None,
                 gate_factory=None, awake_factory=None, io_factory=None, monitor_interval=.5):
    if (tuple(agent_order) not in ((1, 600), (600, 1))
            or any(type(agent) is not int for agent in agent_order)):
        raise ValueError("Both agents1/600 exactly once required")
    tasks_by_id = {task["agent_id"]: task for task in TASKS}
    planned_tasks = tuple(tasks_by_id[agent] for agent in agent_order)
    config, config_hash = configuration(config_path, tokens, request_seconds=request_seconds)
    if request_seconds == 2700 and max_seconds > 7200:
        raise ValueError("45-minute requests retain a hard 7200s campaign ceiling")
    if not 0 < monitor_interval <= .5 or not 1 <= max_seconds <= 10800:
        raise ValueError("Sampling<=0.5s and campaign deadline1..10800s required")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "config_sha256": config_hash, "guard_limits": config["limits"],
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
            report["power_preflight"] = gate.sample()
            if not report["power_preflight"]["passed"]:
                raise QualificationError("POWER_CONTEXT_UNAVAILABLE", "Open lid and stable AC power required; no model constructed or loaded")
            time.sleep(.5)
            report["power_admission_confirmation"] = gate.sample()
            if not report["power_admission_confirmation"]["passed"]:
                raise QualificationError("POWER_CONTEXT_UNAVAILABLE", "AC/lid changed during admission; no model constructed or loaded")
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
                selected = NoCacheResearchModel(*args, **kwargs)
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
            for task in planned_tasks:
                if monitor_errors:
                    raise QualificationError("SUPERVISION_REJECTED", "Campaign supervision already rejected")
                mission = runtime.create({"prompt": task["prompt"]})
                runtime.dispatch("agent_memory", {"id": mission["id"], "agent_id": task["agent_id"], "text": memory_for(task)})
                row = {"agent_id": task["agent_id"], "mission_id": mission["id"], "marker": task["marker"],
                       "events": [], "tokens": [], "text": "", "final": None, "owned_pid": process.pid}
                report["requests"].append(row)
                _write(output / "report.json", report)
                restore = capture_native(model, row, output, observer)
                try:
                    row["result"] = runtime.dispatch("generate", {"id": mission["id"], "agent_id": task["agent_id"],
                        "max_new_tokens": tokens, "reasoning_effort": "low"})
                    row["decision"] = runtime.dispatch("agent_decision", {"id": mission["id"], "agent_id": task["agent_id"],
                        "decision_id": row["result"]["decision_id"]})
                    row["engine_after"] = runtime.engine.status()
                    row["completion_contract"] = validate_completion(row, tokens)
                    row["answer_checks"] = answer_checks(task, row["text"])
                    row["final_answer"] = final_answer(row["text"])
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
            if monitor_errors:
                report.update(status="failed", supervision_errors=monitor_errors)
            report["functional_checks"] = campaign_checks(report)
            report["functional_two_agents_qualified"] = report["status"] == "completed" and all(report["functional_checks"].values())
            if report["status"] == "completed" and not report["functional_two_agents_qualified"]:
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
    rows = sorted(report.get("requests", []), key=lambda row: str(row.get("agent_id", "")))
    native = report.get("engine_after_stream", {}).get("native", {})
    final_native = report.get("engine_after_cleanup", {}).get("native", {})
    decisions = [row.get("decision", {}) for row in rows]
    isolated = len(rows) == 2 and all(
        task["marker"] in json.dumps(decision.get("messages", []), ensure_ascii=False)
        and all(peer["marker"] not in json.dumps(decision.get("messages", []), ensure_ascii=False)
                for peer in TASKS if peer != task) for task, decision in zip(TASKS, decisions))
    return {
        "two_distinct_agents": [r.get("agent_id") for r in rows] == [1, 600],
        "600_logical_identities": report.get("registered_identities") == 600,
        "one_owned_pid": len(rows) == 2 and len({r.get("owned_pid") for r in rows}) == 1,
        "one_model_load": native.get("counters", {}).get("loads") == 1,
        "two_native_requests": native.get("counters", {}).get("requests") == 2,
        "one_physical_slot": native.get("scheduler", {}).get("physical_limit") == 1,
        "natural_eos_each": len(rows) == 2 and all((r.get("final") or {}).get("stop_type") == "eos" for r in rows),
        "complete_transcripts": len(rows) == 2 and all(r.get("completion_contract", {}).get("passed") is True for r in rows),
        "useful_final_each": len(rows) == 2 and all(all(v for k, v in r.get("answer_checks", {}).items() if k != "scope")
            and len(r.get("answer_checks", {})) > 1 for r in rows),
        "host_contexts_isolated": isolated,
        "native_prompts_isolated": len(rows) == 2 and all(task["marker"] in row.get("rendered_prompt", "")
            and all(peer["marker"] not in row.get("rendered_prompt", "") for peer in TASKS if peer != task)
            for task, row in zip(TASKS, rows)),
        "actual_prompt_ids_each": len(rows) == 2 and all(isinstance(r.get("prompt_token_ids"), list) and bool(r["prompt_token_ids"])
            and all(type(t) is int and t >= 0 for t in r["prompt_token_ids"]) for r in rows),
        "full_prompts_no_truncation": len(rows) == 2 and all((r.get("final") or {}).get("truncated") is False for r in rows),
        "process_io_each": len(rows) == 2 and all(r.get("process_delta", {}).get("available") is True for r in rows),
        "cache_disabled_each": len(rows) == 2 and all(r.get("body", {}).get("cache_prompt") is False and r.get("body", {}).get("id_slot") == 0 for r in rows),
        "normal_native_exit": report.get("owned_server_stopped") is True and report.get("owned_server_returncode") == 0,
        "stable_ac": report.get("power_continuity_validated") is True,
        "weights_integrity": native.get("checkpoint", {}).get("integrity_verified") is True,
        "runtime_integrity": native.get("runtime_artifacts_verified") is True,
        "resource_guards_clear": bool(final_native) and native.get("failure") is None and final_native.get("failure") is None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--agent-order", choices=("1,600", "600,1"), default="1,600")
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--request-seconds", type=int, choices=(1800, 2700), default=1800,
                        help="Explicit bounded request time; memory/PID/cancel guards unchanged")
    args = parser.parse_args()
    report = run_campaign(args.config, args.output, args.tokens, max_seconds=args.max_seconds,
                          request_seconds=args.request_seconds,
                          agent_order=tuple(map(int, args.agent_order.split(","))))
    print(json.dumps({"status": report["status"], "functional_two_agents_qualified": report["functional_two_agents_qualified"],
                      "full_qualification": report["full_qualification"], "report": str(args.output / "report.json")}))
    return 0 if report["functional_two_agents_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
