"""Isolated, opt-in C research runner; import and default plan start no engine.

No positive native B evidence exists when this source is delivered. The runner
never turns a mock, first exact request, incomplete arm or failed guard into B.
HTTP numerical qualification remains separate even after useful natural EOS.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

from prefix_adapter import AdapterError, CampaignBudget, NativePrefixGate, make_model_class, require, sha
from guards import PhaseAwareGuard

HERE = Path(__file__).resolve().parent
SOURCES = ("prefix_adapter.py", "guards.py", "campaign.py")
ORDER = (1, 600, 1)
UID_OBSERVER_SHA256 = "7857d7d0996767cfdaacf71c5e908100f6f09ffe1280ea10cdd22d8a3ab7c219"


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def source_hashes():
    return {str(HERE / name): sha(HERE / name) for name in SOURCES}


def verify_checkout_binding(repository, native_gate):
    require(isinstance(native_gate, NativePrefixGate), "real closed-evidence native gate required")
    repository = Path(repository).resolve()
    require(native_gate.repository.resolve() == repository, "runner and native gate must use one checkout", "PREFIX_SOURCE")
    if native_gate.cpu8_bridge is not None:
        path = Path(native_gate.cpu8_bridge["variant_comparator"]).resolve()
        # The reviewed/pinned comparator derives REPO from this exact sibling
        # layout. Check before importing any repository modules; never rewrite
        # its constant or normalize another checkout into its provenance.
        require(repository == (path.parent.parent / "airllm-next").resolve(),
                "reviewed CPU16 bridge is bound to its sibling airllm-next checkout", "PREFIX_SOURCE")


def load_uid_observer(repository):
    """The measured sibling exists outside the frozen repository; never copy it."""
    repository = Path(repository).resolve()
    path = repository.parent / "uid_supervisor.py"
    require(path.is_file() and not path.is_symlink() and sha(path) == UID_OBSERVER_SHA256,
            "original sibling UID observer source identity missing or changed", "PREFIX_SOURCE")
    name = "ronin_prefix_existing_uid_observer"
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(inspect.getfile(module)).resolve() == path, "UID observer cached from another path", "PREFIX_SOURCE")
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    require(module.REPO.resolve() == repository and callable(module.observe_lock_fd) and sha(path) == UID_OBSERVER_SHA256,
            "UID observer changed or belongs to another checkout", "PREFIX_SOURCE")
    return module


def dependencies(repository):
    """Use the explicitly selected repository, refusing pre-imported shadows."""
    repository = Path(repository).resolve()
    sys.path[:0] = [str(repository), str(repository / "air_llm"),
                   str(repository / "integrations/ronin/runtime"),
                   str(repository / "integrations/ronin/engine")]
    modules = {}
    for name in ("airllm.nextgen.glm_native", "research.glm53_shared_prefill.qualify_agents",
                 "research.glm53_batched_prefill.campaign",
                 "research.glm53_gpu_v3.power_state", "research.glm53_gpu_v3.completion_contract",
                 "research.glm53_gpu_v3.process_io", "research.glm53_ultra.agent_probe",
                 "research.glm53_ultra.chat_probe", "ronin_runtime", "adapter", "agent_context"):
        module = importlib.import_module(name)
        require(Path(inspect.getfile(module)).resolve().is_relative_to(repository),
                "imported dependency is outside selected repository: " + name, "PREFIX_SOURCE")
        modules[name] = module
    uid = load_uid_observer(repository)
    modules["existing_owned_uid_observer"] = uid
    qa = modules["research.glm53_shared_prefill.qualify_agents"]
    return SimpleNamespace(qa=qa, native=modules["airllm.nextgen.glm_native"],
        batched=modules["research.glm53_batched_prefill.campaign"],
        uid=uid,
        hashes={str(Path(inspect.getfile(module)).resolve()): sha(inspect.getfile(module))
                for module in modules.values()})


def plan():
    return {"schema_version": 1, "status": "source_candidate", "executed": False,
            "model_loads": 0, "native_B_qualified": False, "HTTP_numerically_qualified": False,
            "functional_C_qualified": False, "planned_agent_order": list(ORDER),
            "logical_identities": 600, "physical_slots": 1, "HTTP_microbatch": 8,
            "requested_tokens_each": 512, "request_seconds": 2700, "campaign_seconds": 7200,
            "reasoning_effort": None, "reasoning_policy": "native default Max; no override",
            "HTTP16_implemented": False, "source_sha256": source_hashes(),
            "scope": "source/mocks only; future execution requires actual complete native evidence and fresh HTTP quarantine"}


def read_physical(path, logical):
    """Observe already journalled HTTP/SSE; never issue a second native call."""
    path = Path(path)
    require(path.stat().st_size <= 16 * 1024 ** 2, "bounded physical journal required")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [row for row in rows if row.get("logical") == logical]
    started = [row for row in selected if row.get("phase") == "generate" and row.get("status") == "started"]
    require(len(started) <= 1, "multiple generation calls for one logical request")
    start = started[0]["t_campaign"] if started else None
    events = [{"t": row["t_campaign"] - start, "response": row["response"]}
              for row in selected if row.get("phase") == "generate_event"] if start is not None else []
    final = [event["response"] for event in events if event["response"].get("stop")]
    require(len(final) <= 1, "multiple terminal native events")
    def completed(phase, field):
        matches = [row["response"].get(field) for row in selected
                   if row.get("phase") == phase and row.get("status") == "completed"]
        require(len(matches) <= 1, "duplicate native metadata call: " + phase)
        return matches[0] if matches else None
    return {"body": started[0]["payload"] if started else None, "events": events,
            "generation_started_t_campaign": start,
            "tokens": [token for event in events for token in event["response"].get("tokens", [])],
            "text": "".join(event["response"].get("content", "") for event in events),
            "final": final[0] if final else None,
            "rendered_prompt": completed("apply_template", "prompt"),
            "prompt_token_ids": completed("native_tokenize", "tokens"),
            "physical_requests": [row for row in selected if row.get("status") == "started"],
            "physical_observation_scope": "actual adapter journal, no monkeypatch or duplicate inference"}


def history_checks(rows, tasks):
    if len(rows) != 3 or [row.get("agent_id") for row in rows] != list(ORDER):
        return {"ordered_three_requests": False, "same_mission": False,
                "private_markers": False, "agent1_full_history_returned": False}
    by_id = {task["agent_id"]: task for task in tasks}
    markers = True
    for row in rows:
        host = json.dumps(row.get("decision", {}).get("messages", []), ensure_ascii=False)
        native = row.get("rendered_prompt") or ""
        own = by_id[row["agent_id"]]["marker"]
        peer = by_id[600 if row["agent_id"] == 1 else 1]["marker"]
        markers = markers and own in host and own in native and peer not in host and peer not in native and peer not in row.get("text", "")
    first, third = rows[0], rows[2]
    previous = first.get("decision", {})
    replayed = third.get("decision", {}).get("messages", [])
    expected = previous.get("output_text")
    # The store applies the existing RONIN text policy. Compare its exact stored
    # output, rather than substituting raw native text when sanitization differs.
    first_messages = previous.get("messages", [])
    first_user = next((m["content"] for m in reversed(first_messages) if m.get("role") == "user"), None)
    expected_pair = [{"role": "user", "content": first_user}, {"role": "assistant", "content": expected}]
    pair_matches = isinstance(expected, str) and bool(expected) and isinstance(first_user, str) and replayed[1:3] == expected_pair
    native_history = isinstance(expected, str) and bool(expected) and expected in (third.get("rendered_prompt") or "") and (
        isinstance(first_user, str) and first_user in (third.get("rendered_prompt") or ""))
    return {"ordered_three_requests": True, "same_mission": first.get("mission_id") == third.get("mission_id"),
            "private_markers": markers, "agent1_full_history_returned": pair_matches and native_history}


def functional_checks(report, qa):
    rows = report["requests"]
    checks = history_checks(rows, qa.TASKS)
    native = report.get("engine_after_stream", {}).get("native", {})
    final = report.get("engine_after_cleanup", {}).get("native", {})
    checks.update({
        "600_identities": report.get("registered_identities") == 600,
        "one_owned_pid": len(rows) == 3 and len({row.get("owned_pid") for row in rows}) == 1,
        "one_model_load": native.get("counters", {}).get("loads") == 1,
        "three_logical_generations": native.get("counters", {}).get("requests") == 3,
        "one_physical_slot": native.get("scheduler", {}).get("physical_limit") == 1,
        "native_EOS_each": len(rows) == 3 and all((row.get("final") or {}).get("stop_type") == "eos" for row in rows),
        "complete_transcripts": len(rows) == 3 and all(row.get("completion_contract", {}).get("passed") is True for row in rows),
        "useful_final_each": len(rows) == 3 and all(row.get("answer_checks") and all(
            value for key, value in row["answer_checks"].items() if key != "scope") for row in rows),
        "restored_208_each": len(rows) == 3 and all((row.get("final") or {}).get("timings", {}).get("cache_n") == 208 for row in rows),
        "full_prompt_ids_each": len(rows) == 3 and all(isinstance(row.get("prompt_token_ids"), list) and
            (row.get("final") or {}).get("tokens_evaluated") == len(row["prompt_token_ids"]) and
            (row.get("final") or {}).get("truncated") is False for row in rows),
        "process_io_each": len(rows) == 3 and all(row.get("process_delta", {}).get("available") is True for row in rows),
        "normal_native_exit": report.get("owned_server_stopped") is True and report.get("owned_server_returncode") == 0,
        "stable_AC": report.get("power_continuity_validated") is True,
        "source_unchanged": report.get("sources_unchanged") is True,
        "resource_guards_clear": bool(native) and bool(final) and native.get("failure") is None and final.get("failure") is None,
        "integrity": native.get("runtime_artifacts_verified") is True and native.get("checkpoint", {}).get("integrity_verified") is True,
        "native_Max_512": report.get("reasoning_effort") is None and report.get("requested_tokens_each") == 512,
        "actual_native_B": report.get("native_B_qualified") is True,
        "external_guard_clear": report.get("external_guard", {}).get("passed") is True,
    })
    return checks


def two_agent_observation(report):
    """Retain useful outputs already observed if the return-to-agent1 fails."""
    rows = report.get("requests", [])[:2]
    checks = {
        "distinct_agents_1_600": len(rows) == 2 and [row.get("agent_id") for row in rows] == [1, 600],
        "native_EOS_each": len(rows) == 2 and all((row.get("final") or {}).get("stop_type") == "eos" for row in rows),
        "complete_transcripts": len(rows) == 2 and all(row.get("completion_contract", {}).get("passed") is True for row in rows),
        "useful_final_each": len(rows) == 2 and all(row.get("answer_checks") and all(
            value for key, value in row["answer_checks"].items() if key != "scope") for row in rows),
    }
    return {"observed": all(checks.values()), "checks": checks, "request_indices": [0, 1],
        "scope": "only the two first actually recorded useful natural-EOS outputs; no claim of full campaign, agent1 history return, numerical fidelity or guard qualification"}


def run_campaign(repository, config_path, output, native_gate, *, execute_research=False,
                 deps=None, runtime_factory=None, power_factory=None, awake_factory=None,
                 io_factory=None, guard_factory=None, monitor_seconds=.5):
    require(execute_research is True, "research execution is opt-in; default plan starts no engine", "PREFIX_DISABLED")
    require(0 < monitor_seconds <= .5, "campaign sampling must be <=0.5s")
    budget = CampaignBudget(7200)  # Before dependencies, metadata, Runtime or load.
    verify_checkout_binding(repository, native_gate)
    deps = dependencies(repository) if deps is None else deps
    qa = deps.qa
    config, config_hash = qa.configuration(config_path, 512, request_seconds=2700)
    require(config["context_tokens"] == 2048, "exact native helper context2048 required")
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink(), "fresh isolated output/state required")
    hashes = {**source_hashes(), **deps.hashes}
    report = {**plan(), "status": "starting", "executed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes, "config_sha256": config_hash, "guard_limits": config["limits"],
        "requests": [], "physical_requests_started": 0, "functional_two_agents_qualified": False,
        "exact_state_qualified": False, "full_qualification": False, "adopted_optimization": False}
    power = (power_factory or qa.PowerGate)("ac")
    runtime = model = process = awake = watcher = guard = None
    stop = threading.Event()
    errors, handlers = [], {}
    previous = os.environ.get("RONIN_GLM_CONFIG")
    cleaning = False

    def check_sources():
        require(all(sha(path) == digest for path, digest in hashes.items()),
                "research/native policy sources changed", "PREFIX_SOURCE")

    def cancel_all():
        if runtime is not None:
            for event in list(runtime.cancels.values()):
                event.set()
            runtime.engine.request_shutdown()

    def interrupted(signum, frame):
        if not cleaning:
            cancel_all()
            raise qa.ProbeInterrupted(signum)

    def monitor():
        while not stop.wait(monitor_seconds):
            try:
                budget.remaining()
                awake.check()
                if model is not None:
                    model._check(budget.deadline)
                else:
                    require(power.sample()["passed"] is True and power.violation is None, "power context rejected")
            except Exception as error:
                errors.append({"code": getattr(error, "code", type(error).__name__), "error": str(error)})
                cancel_all()
                if model is not None:
                    model.close()
                return

    with qa.BENCHMARK_LOCK.open("a") as benchmark:
        fcntl.flock(benchmark, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(mode=0o700, parents=True)
        try:
            require(isinstance(native_gate, NativePrefixGate), "real closed-evidence native gate required")
            # Native gate runs before even constructing Runtime/model. Its
            # check callback charges proof I/O and guards power/clock globally.
            last_proof_power = [float("-inf")]
            def proof_check():
                budget.remaining()
                now = budget.clock()
                include_power = now - last_proof_power[0] >= .5
                require(power.sample(include_power=include_power)["passed"] and power.violation is None, "power/clock rejected")
                if include_power:
                    last_proof_power[0] = now
            require(power.sample()["passed"] is True, "stable AC and open lid required before native admission")
            report["native_admission"] = {key: value for key, value in native_gate.verify(proof_check).items()
                                          if key not in ("spec", "manifest")}
            require(power.sample()["passed"] is True and power.violation is None, "power changed during native admission")
            report["native_B_qualified"] = True
            report["metadata_preflight"] = qa.metadata_preflight(config)
            hashes.update(report["metadata_preflight"]["sha256"])
            receipt = Path(config["verification_file"])
            require(receipt.stat().st_size <= 64 * 1024 ** 2, "bounded verification receipt required")
            copied = output / "model-verification.json"
            copied.write_bytes(receipt.read_bytes()); copied.chmod(0o600)
            config.update(verification_file=str(copied), state_dir=str(output / "engine-state"))
            write(output / "run-config.json", config)
            os.environ["RONIN_GLM_CONFIG"] = str(output / "run-config.json")
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupted)
            guard = (guard_factory or PhaseAwareGuard)(power, qa.NativeLimits, fd_observer=deps.uid.observe_lock_fd)
            ResearchModel = make_model_class(qa.NativeGlmModel)
            runtime = (runtime_factory or qa.Runtime)(output / "state")
            factory_calls = []
            def factory(*args, **kwargs):
                nonlocal model
                check_sources()
                require(not factory_calls, "one physical native model only; no automatic reload")
                factory_calls.append(True)
                report["model_construction_attempted"] = True
                model = ResearchModel(*args, enable_shared_prefix=True, native_gate=native_gate,
                    campaign_budget=budget, admission_guard=guard, evidence_dir=output / "prefix-evidence", **kwargs)
                return model
            runtime.engine._model_factory = factory
            report["registered_identities"] = len(runtime.dispatch("agent_identities", {}))
            require(report["registered_identities"] == 600, "exactly600 logical identities required")
            awake = (awake_factory or qa.ScopedAwake)(SimpleNamespace(pid=os.getpid(), poll=lambda: None))
            awake.__enter__()
            watcher = threading.Thread(target=monitor, name="research-prefix-power-uid", daemon=True)
            watcher.start()
            check_sources()
            runtime.engine.load(config["model_path"])
            process = model._process
            require(process is not None and process.poll() is None, "retained live native server required")
            report.update(executed=True, status="running", owned_pid=process.pid,
                          engine_after_load=runtime.engine.status(), common_startup_seconds=budget.clock()-budget.started)
            report["observed_native_argv"] = qa.observe_owned_argv(process, config["runtime_binary"])
            require(report["observed_native_argv"].get("available") is True, "owned native argv not observed")
            observer = (io_factory or qa.ProcessIOObserver)(process.pid)
            tasks = {task["agent_id"]: task for task in qa.TASKS}
            missions = {}
            for index, agent_id in enumerate(ORDER):
                check_sources()
                require(not errors, "campaign supervision already rejected")
                task = tasks[agent_id]
                if agent_id not in missions:
                    missions[agent_id] = runtime.create({"prompt": task["prompt"]})
                    runtime.dispatch("agent_memory", {"id": missions[agent_id]["id"], "agent_id": agent_id, "text": qa.memory_for(task)})
                mission = missions[agent_id]
                row = {"request_index": index, "agent_id": agent_id, "mission_id": mission["id"],
                       "marker": task["marker"], "owned_pid": process.pid}
                report["requests"].append(row)
                logical = model._prefix_logical + 1
                before = observer.snapshot()
                started = budget.clock()
                row["logical_started_t_campaign"] = started - budget.started
                try:
                    with model.logical_scope():
                        row["result"] = runtime.dispatch("generate", {"id": mission["id"], "agent_id": agent_id, "max_new_tokens": 512})
                    row["decision"] = runtime.dispatch("agent_decision", {"id": mission["id"], "agent_id": agent_id,
                        "decision_id": row["result"]["decision_id"]})
                finally:
                    row.update(read_physical(model._prefix_journal, logical))
                    after = observer.snapshot()
                    row.update(logical_wall_seconds=budget.clock()-started, io_before=before, io_after=after,
                               process_delta=qa.compare_process(before, after), engine_after=runtime.engine.status())
                    row["completion_contract"] = qa.validate_completion(row, 512)
                    row["answer_checks"] = qa.answer_checks(task, row["text"])
                    row["final_answer"] = qa.final_answer(row["text"])
                    row["measurement"] = deps.batched.measurement(row)
                    first_token = next((event["t"] for event in row["events"] if event["response"].get("tokens")), None)
                    generation_start = row.get("generation_started_t_campaign")
                    row["measurement"].update(logical_wall_seconds=row["logical_wall_seconds"],
                        logical_ttft_seconds=(generation_start - row["logical_started_t_campaign"] + first_token)
                            if first_token is not None and generation_start is not None else None,
                        generation_stream_ttft_seconds=first_token,
                        ttft_scope="ttft_seconds is native generation stream only; logical_ttft_seconds includes host/restore/proof stages; common startup separately in campaign")
                    write(output / "report.json", report)
            report.update(engine_after_stream=runtime.engine.status(), status="completed")
        except BaseException as error:
            report.update(status="blocked" if not report["executed"] else "failed",
                          error_code=getattr(error, "code", type(error).__name__), error=str(error),
                          error_details=getattr(error, "details", None))
        finally:
            cleaning = True
            stop.set()
            if watcher is not None:
                watcher.join(timeout=8)
                if watcher.is_alive():
                    errors.append({"code": "SUPERVISION_THREAD_NOT_STOPPED"})
            try:
                if runtime is not None:
                    if model is not None:
                        if model._prefix_physical > 0 or getattr(model, "_process", None) is not None:
                            report["executed"] = True
                            if report["status"] == "blocked":
                                report["status"] = "failed"
                        report["adapter_before_cleanup"] = model.status().get("research_shared_prefix")
                        report["physical_requests_started"] = model._prefix_physical
                        write(output / "native-resource-samples.json", list(getattr(model, "_samples", [])))
                    try:
                        qa._graceful_stop(process, report)
                    finally:
                        qa._cleanup(runtime, report)
                    report["engine_after_cleanup"] = runtime.engine.status()
            except Exception as error:
                report.update(status="failed", cleanup_error=str(error))
            try:
                if awake is not None:
                    awake.close()
                power.sample(final=True)
                report["power_gate"] = power.summary()
                if awake is not None:
                    report["power_gate"]["idle_assertion"] = awake.record
                report["power_continuity_validated"] = qa.power_evidence_matches({**report, "owned_pid": os.getpid()}, "ac")
                report["external_guard"] = guard.summary() if guard is not None else {"passed": False}
                report["sources_unchanged"] = all(sha(path) == digest for path, digest in hashes.items())
            except Exception as error:
                report.update(status="failed", final_guard_error=str(error), sources_unchanged=False)
            if errors or not report.get("sources_unchanged"):
                report.update(status="failed", supervision_errors=errors)
            report["functional_checks"] = functional_checks(report, qa)
            report["two_distinct_agents_useful_EOS_observation"] = two_agent_observation(report)
            report["functional_two_agents_qualified"] = (report["status"] == "completed" and all(report["functional_checks"].values()))
            report["functional_C_qualified"] = report["functional_two_agents_qualified"]
            # Server/slots EOS can establish functional evidence only. No new
            # full HTTP logits/routes/KDA/KV capture is claimed by this runner.
            report["HTTP_numerically_qualified"] = False
            report["exact_state_qualified"] = report["full_qualification"] = False
            if report["status"] == "completed" and not report["functional_C_qualified"]:
                report["status"] = "rejected"
            report["elapsed_seconds"] = budget.clock() - budget.started
            if previous is None:
                os.environ.pop("RONIN_GLM_CONFIG", None)
            else:
                os.environ["RONIN_GLM_CONFIG"] = previous
            write(output / "report.json", report)
            for signum, previous_handler in handlers.items():
                signal.signal(signum, previous_handler)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-research", action="store_true")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-gate-json", type=Path)
    args = parser.parse_args()
    if not args.execute_research:
        print(json.dumps(plan(), ensure_ascii=False))
        return 0
    require(all((args.repository, args.config, args.output, args.native_gate_json)), "explicit repository/config/output/evidence required")
    require(args.native_gate_json.stat().st_size <= 64 * 1024, "bounded gate path configuration required")
    paths = json.loads(args.native_gate_json.read_text())
    require(set(paths) <= {"reference", "restored", "reference_guard", "restored_guard", "cpu8_bridge"}, "unknown native gate option")
    gate = NativePrefixGate(args.repository, **paths)
    report = run_campaign(args.repository, args.config, args.output, gate, execute_research=True)
    print(json.dumps({"status": report["status"], "functional_C_qualified": report["functional_C_qualified"],
                      "HTTP_numerically_qualified": False, "report": str(args.output / "report.json")}))
    return 0 if report["functional_C_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
