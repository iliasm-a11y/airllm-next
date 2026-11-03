"""Isolated CPU-to-Metal Q8 runner; original watchdog/UID ownership retained.

This is a native-state diagnostic, not generation or EOS qualification. The16MiB
native serialization scratch envelope is an explicit conservative policy budget,
not a measurement of allocation peaks. A snapshot remains unqualified until raw
state/logit/router/KDA captures are compared separately.
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
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "airllm-next"
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "air_llm")]
import psutil
from airllm.nextgen.glm_native import NativeGlmModel, NativeLimits, resource_snapshot, _identity, FILES
from research.glm53_shared_prefill import qualify_agents as qa
from research.glm53_gpu_v3.power_state import PowerGate, ScopedAwake, power_evidence_matches
from research.glm53_gpu_v3.process_io import ProcessIOObserver
from research.glm53_batched_prefill.snapshot_key import canonical_key
import gpu_contract as gc
NATIVE_GUARD_SOURCE = Path(sys.modules[NativeGlmModel.__module__].__file__).resolve()

MiB, GiB = 1024**2, 1024**3
SCRATCH_POLICY = 16 * MiB
BATCHES = (8,16,32,64,128)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(MiB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def contract(function,*args,**kwargs):
    try:
        return function(*args,**kwargs)
    except gc.GpuContractError as error:
        raise qa.QualificationError(error.code,str(error)) from error


def observe_uid_fd(child,binary,identity,record,process_factory=psutil.Process):
    identify(child,binary,identity,process_factory=process_factory)
    completed=subprocess.run(["/usr/sbin/lsof","-a","-p",str(child.pid),"-d",str(record["pass_fd"]),"-Ffn"],
        capture_output=True,text=True,timeout=3,check=False)
    matched=False
    for line in completed.stdout.splitlines():
        if not line.startswith("n"):continue
        try:
            stat=Path(line[1:]).stat()
            matched=matched or (stat.st_dev,stat.st_ino)==(record["device"],record["inode"])
        except OSError:pass
    identify(child,binary,identity,process_factory=process_factory)
    return {"observed_matching_lock_fd":matched,"lsof_returncode":completed.returncode,
        "scope":"one owned helper PID/FD, bracketed lifetime/executable checks"}


def clean_environment(config, diagnostic=False):
    env = {k:v for k,v in os.environ.items() if not k.startswith(("LLAMA_ARG_","AIRLLM_","GGML_","DYLD_","RONIN_PREFILL_")) and k != "LLAMA_API_KEY"}
    knobs = {"AIRLLM_DENSE_PREFETCH_MAX_COLS":"1"}
    for option, name in (("mmap_demand","AIRLLM_MMAP_DEMAND"), ("expert_prefetch","AIRLLM_EXPERT_PREFETCH"),
                         ("dense_prefetch","AIRLLM_DENSE_PREFETCH"), ("mmap_nocache","AIRLLM_MMAP_NOCACHE")):
        if config.get(option) is True:
            knobs[name] = "1"
    if diagnostic:
        knobs["RONIN_PREFILL_DIAGNOSTIC"] = "1"
    env.update(knobs)
    return env, knobs


def admit(config, microbatch, mode, diagnostic, sample):
    limits = NativeLimits(**config["limits"])
    # Charge one256MiB common snapshot and one transient256MiB final-state
    # diagnostic separately, in addition to the native workspace estimate.
    snapshot_bytes = (256*MiB + SCRATCH_POLICY) if mode == "restore" else 0
    diagnostic_bytes = (256*MiB + SCRATCH_POLICY) if diagnostic else 0
    required = limits.reserve_bytes + limits.base_workspace_bytes + 2048*64*1024 + microbatch*4*MiB + snapshot_bytes + diagnostic_bytes + gc.ADDITIONAL_RESERVATION
    if type(sample.get("available_bytes")) is not int or type(sample.get("pressure_level")) is not int:
        raise qa.QualificationError("MEMORY_ADMISSION","Finite integer memory admission metadata required")
    if sample.get("pressure_level") != 1 or sample.get("available_bytes", 0) < required:
        raise qa.QualificationError("MEMORY_ADMISSION", "Normal memory pressure and reserved native/snapshot workspace required before launch")
    return {"required_bytes_estimate":required,"snapshot_envelope_bytes":snapshot_bytes,"diagnostic_envelope_bytes":diagnostic_bytes,
            "native_scratch_policy_bytes":SCRATCH_POLICY,"gpu_additional_reservation_bytes":gc.ADDITIONAL_RESERVATION,
            "gpu_payload_limit_bytes":gc.DATA_LIMIT,"gpu_framework_driver_policy_bytes":64*MiB,"sample":sample,"passed":True,
            "scope":"conservative policy reservation, not measured runtime allocation peaks"}


def guard_model(limits):
    model = object.__new__(NativeGlmModel)
    model.limits = limits
    model._warning_started = None
    model._headroom_low_started = None
    model._pressure_context = None
    return model


def acquire_uid_model_lock():
    """Exactly the production UID/model key; native child inherits this FD."""
    root = Path(tempfile.gettempdir()) / ("airllm-native-locks-" + str(os.getuid()))
    root.mkdir(mode=0o700, exist_ok=True)
    key = NativeGlmModel._model_lock_key(object.__new__(NativeGlmModel))
    path = root / (key + ".lock")
    holder = path.open("a")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        holder.close()
        raise qa.QualificationError("ENGINE_BUSY", "Another native engine already owns this UID/model lock") from error
    stat=os.fstat(holder.fileno())
    return holder, {"path":str(path), "uid":os.getuid(), "model_key":key,"device":stat.st_dev,"inode":stat.st_ino,
        "acquired_at_utc":datetime.now(timezone.utc).isoformat(), "acquired_before_spawn":True,
        "pass_fd":holder.fileno(), "pass_fds_requested":False,
        "scope":"same UID-wide pinned model lock as NativeGlmModel.load; inherited Popen FD survives owner death"}


def identify(child, binary, previous=None, *, process_factory=psutil.Process):
    if child.poll() is not None:
        raise qa.QualificationError("OWNED_PROCESS_EXITED", "Owned native helper is no longer live")
    target = process_factory(child.pid)
    started, exe = target.create_time(), Path(target.exe()).resolve()
    if not target.is_running() or exe != Path(binary).resolve() or previous is not None and started != previous["create_time"]:
        raise qa.QualificationError("OWNED_PROCESS_IDENTITY", "Owned native helper executable or PID lifetime changed")
    return {"pid":child.pid,"create_time":started,"executable":str(exe)}


def stop_owned(child, binary, identity, *, process_factory=psutil.Process, grace_seconds=5):
    """Signal only a retained unreaped Popen child after lifetime/exe verification."""
    record = {"signals":[],"returncode":child.poll()}
    if child.poll() is None:
        identify(child,binary,identity,process_factory=process_factory)
        child.send_signal(signal.SIGTERM)
        record["signals"].append("SIGTERM")
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                identify(child,binary,identity,process_factory=process_factory)
                child.send_signal(signal.SIGKILL)
                record["signals"].append("SIGKILL")
            child.wait(timeout=grace_seconds)
    record["returncode"] = child.poll()
    record["owned_server_stopped"] = record["returncode"] is not None
    return record


def prepare(config, binary, helper_source, prompts_path, libraries, microbatch, env, knobs, *, gpu_document, gpu_profile_path, provider_build_path, driver_build_path, uid_fd):
    evidence = qa.metadata_preflight(config)
    checkpoint = evidence["checkpoint"]
    if checkpoint.get("integrity_verified") is not True or checkpoint.get("errors"):
        raise qa.QualificationError("MODEL_INTEGRITY", "Unchanged original model SHA receipts are required; no full-shard rescan")
    binary, helper_source, libraries = Path(binary).resolve(), Path(helper_source).resolve(), Path(libraries).resolve()
    if libraries != Path(config["runtime_binary"]).resolve().parent:
        raise ValueError("Helper runtime libraries must match the configured verified native package")
    if not binary.is_file() or not os.access(binary,os.X_OK) or binary.stat().st_size > 64*MiB:
        raise ValueError("A bounded executable helper is required")
    native = NativeGlmModel(config["model_path"], runtime_binary=config["runtime_binary"],
        runtime_manifest_file=config["runtime_manifest_file"], verification_file=config["verification_file"],
        limits=NativeLimits(**config["limits"]))
    native._verify_runtime_manifest()
    if not native._runtime_artifacts_verified:
        raise ValueError("Runtime manifest was not verified")
    if config.get("expected_runtime_sha256") and sha(config["runtime_binary"]) != config["expected_runtime_sha256"]:
        raise ValueError("Configured runtime executable fingerprint differs")
    linked = subprocess.run(["/usr/bin/otool","-L",str(binary)],capture_output=True,text=True,check=True,timeout=10).stdout
    rpaths = subprocess.run(["/usr/bin/otool","-l",str(binary)],capture_output=True,text=True,check=True,timeout=10).stdout
    paths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset",rpaths)
    resolved = [str(Path(p.replace("@loader_path",str(binary.parent)).replace("@executable_path",str(binary.parent))).resolve()) for p in paths]
    if str(libraries) not in resolved:
        raise ValueError("Native helper has no explicit rpath to the pinned runtime package")
    if any(".dylib" in line and "@rpath/" not in line and not line.strip().startswith(("/usr/lib/","/System/Library/",str(libraries)+"/")) for line in linked.splitlines()[1:]):
        raise ValueError("Native helper has an unexpected non-system library dependency")
    if Path(prompts_path).stat().st_size > 16*MiB:
        raise ValueError("Native prompt JSON exceeds16MiB")
    prompts = json.loads(Path(prompts_path).read_text())
    agents = {row["agent_id"]:row for row in prompts["agents"]}
    if not {1,600} <= agents.keys() or not isinstance(prompts.get("template"),str):
        raise ValueError("Full native-rendered prompts and raw native template required")
    prefix = agents[1]["token_ids"][:208]
    if len(prefix) != 208 or prefix != agents[600]["token_ids"][:208]:
        raise ValueError("The208-token common boundary must match native IDs")
    state = json.loads(subprocess.run([str(binary),"--state-parameters",str(microbatch)],env=env,pass_fds=(uid_fd,),capture_output=True,text=True,check=True,timeout=15).stdout)
    # Reserved metadata envelope fields are validated explicitly, never silently
    # omitted from an effective parameter dictionary that the report retains.
    native_linkage=state.pop("native_linkage",None)
    native_resource_context=state.pop("native_resource_context",None)
    if state.get("n_ctx") != 2048 or state.get("n_batch") != microbatch or state.get("n_ubatch") != microbatch:
        raise ValueError("Native helper context/microbatch descriptor differs from the requested profile")
    contract(gc.validate_state,state,gpu_document,microbatch)
    loaded_images=contract(gc.validate_loaded_images,native_linkage,gpu_document,libraries)
    contract(gc.validate_resource_context,native_resource_context,int(env["AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES"]))
    hashes = {"helper":sha(binary), **{p.name:sha(p) for p in libraries.glob("*.dylib") if not p.is_symlink()}}
    auxiliary=helper_source.parent.parent/"glm53_gpu_v3/state_capture.inc"
    sources = {"helper_source":sha(helper_source), "snapshot_source":sha(helper_source.with_name("snapshot.h")),"state_capture_source":sha(auxiliary),
               "native_manifest":sha(config["runtime_manifest_file"]), "wrapper_source":sha(__file__)}
    tokenizer = Path(config["model_path"]) / "tokenizer.json"
    manifest = {"schema":1,"model_shards":[{"sha256":row["sha256"],"size_bytes":row["bytes"]} for row in checkpoint["files"]],
        "tokenizer_sha256":sha(tokenizer),"template_sha256":hashlib.sha256(prompts["template"].encode()).hexdigest(),
        "runtime_source_sha256":hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "runtime_binaries":hashes,"prefix_tokens":prefix,"prefix_positions":list(range(208)),
        "state_parameters":state,"render_options":{"native_environment":knobs,"reasoning_effort_override":None,
            "native_add_special":prompts.get("add_special"),"native_parse_special":prompts.get("parse_special"),
            "tokenizer_scope":"tokenizer.json hash plus original GGUF shards identify native tokenizer container"}}
    manifest["render_options"].update(gpu_document["render_options_extension"])
    # native_environment already includes both GPU knobs and the canonical profile digest.
    sources.update(gpu_contract_source=sha(gc.__file__),gpu_profile_source=sha(HERE/"profile.py"),
        gpu_profile_document=sha(gpu_profile_path),provider_build_receipt=sha(provider_build_path),
        gpu_driver_build_receipt=sha(driver_build_path),native_guard_source=sha(NATIVE_GUARD_SOURCE))
    manifest["runtime_source_sha256"]=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    key,canonical = canonical_key(manifest)
    prompts.update(snapshot_key_sha256=key,native_snapshot_scratch_bound_bytes=SCRATCH_POLICY)
    tracked = [binary,helper_source,helper_source.with_name("snapshot.h"),auxiliary,Path(__file__).resolve(),Path(prompts_path).resolve(),tokenizer]
    tracked += [Path(config["model_path"])/row["name"] for row in checkpoint["files"]]
    tracked += list(native._runtime_artifact_identities)
    tracked += [Path(gc.__file__),HERE/"profile.py",Path(gpu_profile_path),Path(provider_build_path),Path(driver_build_path),NATIVE_GUARD_SOURCE]
    identities = {str(p):_identity(p) for p in tracked}
    return {"metadata":evidence,"runtime_manifest":native._runtime_manifest_info,"source_sha256":sources,
        "actual_loaded_native_images":loaded_images,
        "actual_loaded_native_image_paths":native_linkage["loaded_native_images"],
        "native_resource_context":native_resource_context,
        "runtime_binaries":hashes,"snapshot_manifest":manifest,"snapshot_key_sha256":key,
        "snapshot_canonical_bytes":len(canonical),"tracked_identities":identities,
        "helper_linkage":{"otool_L":linked,"runtime_rpaths":resolved},
        "source_build_scope":"exact supplied source/binary fingerprints and verified linked package; compiler provenance supplied externally"},prompts


def identities_pass(before):
    try:
        return all(_identity(Path(path)) == expected for path,expected in before.items())
    except OSError:
        return False


def run(config_path, binary, helper_source, prompts, libraries, output, *, gpu_profile, provider_build, driver_build, microbatch=8, mode="full", diagnostic=False,
        max_seconds=1800, poll_seconds=.5, gate_factory=None, awake_factory=None, popen_factory=None, process_factory=psutil.Process):
    if type(microbatch) is not int or microbatch not in BATCHES or mode not in ("full","split","restore"):
        raise ValueError("Unknown native bounded profile")
    if not 0 < poll_seconds <= .5 or not 1 <= max_seconds <= 10800:
        raise ValueError("Sampling<=0.5s and native deadline<=10800s required")
    config,config_hash = qa.configuration(config_path,512,request_seconds=2700)
    if config["context_tokens"] != 2048:
        raise ValueError("Native helper requires the same explicit context2048")
    limits = NativeLimits(**config["limits"])
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    binary,helper_source = Path(binary).resolve(),Path(helper_source).resolve()
    env,knobs = clean_environment(config,diagnostic)
    report = {"schema_version":1,"started_at_utc":datetime.now(timezone.utc).isoformat(),"status":"running",
        "config_sha256":config_hash,"mode":mode,"microbatch":microbatch,"diagnostic":diagnostic,
        "requested_backend":gc.BACKEND,"actual_gpu":False,"runner_source_sha256_at_start":sha(__file__),
        "limits":{"rss_bytes":limits.max_rss_bytes,"swap_growth_bytes":limits.max_swap_growth_bytes,"native":vars(limits)},
        "sampling_interval_seconds":poll_seconds,"max_seconds":max_seconds,"guard_violation":None,"samples":[],
        "peak_rss_bytes":0,"peak_swap_growth_bytes":0,"returncode":None,"power_context_passed":False,"integrity_passed":False,
        "diagnostic_continuation_requested":diagnostic,"generation_performed":False,"functional_agents_qualified":False,"snapshot_qualified":False,"environment":knobs}
    child=awake=identity=None
    uid_lock=None
    gate=(gate_factory or PowerGate)("ac")
    handlers={}
    started=time.perf_counter()
    def interrupted(signum,frame):
        raise qa.ProbeInterrupted(signum)
    with qa.BENCHMARK_LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,mode=0o700)
        try:
            for signum in (signal.SIGINT,signal.SIGTERM):
                handlers[signum]=signal.signal(signum,interrupted)
            report["power_admission"]=gate.sample()
            if not report["power_admission"]["passed"]:
                raise qa.QualificationError("POWER_CONTEXT_UNAVAILABLE","Stable AC and open lid required before helper startup")
            # UID ownership starts before any metadata helper can load the new backend.
            uid_lock, report["uid_model_lock"] = acquire_uid_model_lock()
            document=contract(gc.validate_profile,contract(gc.read_json,gpu_profile))
            contract(gc.validate_limits,config["limits"],document)
            if psutil.__version__ != "7.2.2":
                raise qa.QualificationError("GPU_RESOURCE_SAMPLER","Pinned psutil7.2.2 available-memory formula required")
            require_config = type(config.get("gpu_layers")) is int and config["gpu_layers"]==0 and config.get("context_tokens")==2048
            if not require_config:
                raise qa.QualificationError("GPU_WEIGHT_PLACEMENT","GPU bridge requires CPU-only original weight placement")
            report["gpu_artifact_preflight"]=contract(gc.validate_artifacts,document,
                contract(gc.read_json,provider_build),contract(gc.read_json,driver_build),
                libraries=libraries,binary=binary,helper_source=helper_source,
                runner_source=__file__,guard_source=NATIVE_GUARD_SOURCE,provider_receipt_sha256=sha(provider_build))
            env,knobs=clean_environment(config,diagnostic)
            env.update(document["environment"]);knobs.update(document["environment"])
            report["environment"]=knobs
            report["gpu_execution_profile"]=document
            report["configured_backend"]=gc.BACKEND
            baseline=resource_snapshot()
            if type(baseline.get("swap_used_bytes")) is not int or not 0 <= baseline["swap_used_bytes"] <= 2**64-1:
                raise qa.QualificationError("GPU_RESOURCE_BASELINE","Finite initial native swap counter required")
            report["admission"]=admit(config,microbatch,mode,diagnostic,baseline)
            report["baseline"]=baseline
            # A dynamic guard input, not an arithmetic mode: record it explicitly
            # outside the dependency key, exactly as the original baseline sample.
            env["AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES"]=str(baseline["swap_used_bytes"])
            report["gpu_runtime_guard_inputs"]={"native_guard_swap_baseline_bytes":baseline["swap_used_bytes"],
                "environment_field":"AIRLLM_CPU_METAL_Q8_SWAP_BASELINE_BYTES",
                "scope":"same initial global watchdog sample; dynamic gauge, not a mathematical snapshot dependency"}
            guard=guard_model(limits)
            metadata,prepared = prepare(config,binary,helper_source,prompts,libraries,microbatch,env,knobs,
                gpu_document=document,gpu_profile_path=gpu_profile,provider_build_path=provider_build,
                driver_build_path=driver_build,uid_fd=uid_lock.fileno())
            report["preflight"]=metadata
            after_prepare=resource_snapshot()
            report["pre_activation_admission"]=admit(config,microbatch,mode,diagnostic,after_prepare)
            preflight_code=NativeGlmModel._guard_resource_sample(guard,after_prepare,baseline,time.monotonic())
            if preflight_code:
                raise qa.QualificationError(preflight_code,"Resource guard rejected metadata preparation")
            write(output/"snapshot-key-manifest.json",metadata["snapshot_manifest"])
            write(output/"run-prompts.json",prepared)
            command=[str(binary),str(Path(config["model_path"])/FILES[0][0]),str(output/"run-prompts.json"),str(output/"capture"),str(microbatch),mode]
            report["command"]=command
            if not gate.sample()["passed"] or not identities_pass(metadata["tracked_identities"]):
                raise qa.QualificationError("PREFLIGHT_CHANGED","Power or artifact identity changed before activation")
            write(output/"guard.json",report)
            with (output/"native-stdout.log").open("wb") as stdout,(output/"native-stderr.log").open("wb") as stderr,(output/"guard-samples.jsonl").open("x") as samples:
                child=(popen_factory or subprocess.Popen)(command,env=env,stdout=stdout,stderr=stderr,
                    pass_fds=(uid_lock.fileno(),))
                report["uid_model_lock"]["pass_fds_requested"] = True
                report["uid_model_lock"]["owned_child_pid"] = child.pid
                report["owned_pid"]=child.pid
                identity=identify(child,binary,process_factory=process_factory)
                report["uid_model_lock"]["fd_observation"]=observe_uid_fd(child,binary,identity,report["uid_model_lock"],process_factory)
                if not report["uid_model_lock"]["fd_observation"]["observed_matching_lock_fd"]:
                    raise qa.QualificationError("UID_LOCK_INHERITANCE","Owned helper UID lock FD was not observed")
                report["owned_identity"]=identity
                awake=(awake_factory or ScopedAwake)(child)
                awake.__enter__()
                io=ProcessIOObserver(child.pid)
                while child.poll() is None:
                    identify(child,binary,identity,process_factory=process_factory)
                    awake.check()
                    power=gate.sample()
                    sample=resource_snapshot(child.pid)
                    sample.update(t=time.perf_counter()-started,swap_growth_bytes=max(0,sample.get("swap_used_bytes",0)-baseline.get("swap_used_bytes",0)),power_passed=power["passed"])
                    sample["process_io"]=io.snapshot()
                    report["samples"].append(sample)
                    report["peak_rss_bytes"]=max(report["peak_rss_bytes"],sample.get("rss_bytes",0))
                    report["peak_swap_growth_bytes"]=max(report["peak_swap_growth_bytes"],sample["swap_growth_bytes"])
                    samples.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+"\n");samples.flush()
                    code=NativeGlmModel._guard_resource_sample(guard,sample,baseline,time.monotonic())
                    if not power["passed"]: code="POWER_CONTEXT_CHANGED"
                    if not identities_pass(metadata["tracked_identities"]): code="ARTIFACT_IDENTITY_CHANGED"
                    if time.perf_counter()-started>max_seconds: code="DEADLINE"
                    if code:
                        report["guard_violation"]={"code":code,"sample":sample,"pressure_context":guard._pressure_context}
                        raise qa.QualificationError(code,"Native diagnostic stopped under unchanged resource/power/identity guards")
                    if len(report["samples"])%10==0:
                        write(output/"guard.json",report)
                    time.sleep(poll_seconds)
                report["returncode"]=child.wait(timeout=5)
                report["status"]="completed" if report["returncode"]==0 else "failed"
                if report["returncode"]!=0:
                    report["error_code"]="NATIVE_NONZERO_EXIT"
        except qa.ProbeInterrupted as error:
            report.update(status="cancelled",error_code=error.code,error=str(error))
            report["guard_violation"]={"code":error.code,"signal":error.signum}
        except Exception as error:
            report.update(status="blocked" if child is None else "failed",error_code=getattr(error,"code",type(error).__name__),error=str(error))
        finally:
            # Ignore repeat user signals while stopping only the retained owned child.
            for signum in handlers:
                signal.signal(signum,signal.SIG_IGN)
            try:
                if child is not None:
                    report["cleanup"]=stop_owned(child,binary,identity,process_factory=process_factory)
                    report["returncode"]=child.poll()
            except Exception as error:
                report.update(status="failed",cleanup_error=str(error))
            try:
                if awake is not None: awake.close()
                gate.sample(final=True)
                report["power_gate"]=gate.summary()
                if awake is not None: report["power_gate"]["idle_assertion"]=awake.record
                report["power_context_passed"]=power_evidence_matches(report,"ac")
            except Exception as error:
                report.update(status="failed",power_cleanup_error=str(error))
            if uid_lock is not None:
                report["uid_model_lock"]["child_poll_before_parent_fd_close"] = child.poll() if child is not None else None
                uid_lock.close()
                report["uid_model_lock"]["parent_fd_closed_at_utc"] = datetime.now(timezone.utc).isoformat()
                report["uid_model_lock"]["parent_fd_closed"] = True
            metadata=report.get("preflight")
            report["integrity_passed"]=bool(metadata) and identities_pass(metadata["tracked_identities"])
            capture_report=output/"capture/report.json"
            if capture_report.is_file():
                try:
                    captured=gc.read_json(capture_report,16*MiB)
                    report["capture_contract"]=gc.validate_capture_report(captured,metadata,report["gpu_execution_profile"],
                        libraries,report["baseline"]["swap_used_bytes"],microbatch)
                    report["generation_performed"]=any(bool(row.get("continuation")) for row in captured.get("requests",[]))
                    report["observed_continuation_token_count"]=sum(len(row.get("continuation",[])) for row in captured.get("requests",[]))
                except (OSError,ValueError,KeyError,TypeError) as error:
                    report["capture_report_error"]={"code":getattr(error,"code",type(error).__name__),"error":str(error)}
                    if report["status"]=="completed":
                        report.update(status="failed",error_code=getattr(error,"code","GPU_CAPTURE_REPORT"),error=str(error))
            elif report["status"]=="completed":
                report.update(status="failed",error_code="GPU_CAPTURE_REPORT",error="Native capture report absent")
            # Preserve any earlier error class; trace validation cannot turn a
            # timeout, resource stop, nonzero exit or identity failure into success.
            stderr_path=output/"native-stderr.log"
            if child is not None and stderr_path.is_file():
                try:
                    if stderr_path.stat().st_size>32*MiB:
                        raise gc.GpuContractError("GPU_TRACE_SIZE","GPU log exceeds32MiB")
                    text=stderr_path.read_text(errors="replace")
                    report["gpu_submissions_observed"]=sum("event=submitted actual_gpu=1" in line for line in text.splitlines())
                    report["actual_gpu"]=report["gpu_submissions_observed"]>0
                    report["actual_backend"]=gc.BACKEND if report["actual_gpu"] else "no_gpu_command_observed"
                    report["gpu_trace"]=gc.parse_trace(text,report["gpu_execution_profile"],baseline_swap=report["baseline"]["swap_used_bytes"])
                except (OSError,ValueError,KeyError,TypeError) as error:
                    report["gpu_trace_error"]={"code":getattr(error,"code",type(error).__name__),"error":str(error)}
                    if report["status"]=="completed":
                        report.update(status="failed",error_code=getattr(error,"code","GPU_TRACE_REJECTED"),error=str(error))
            report["runner_source_identity_unchanged"]=sha(__file__)==report["runner_source_sha256_at_start"]
            if not report["runner_source_identity_unchanged"] and report["status"]=="completed":
                report.update(status="failed",error_code="ARTIFACT_IDENTITY_CHANGED")
            report["elapsed_seconds"]=time.perf_counter()-started
            if report["status"]=="completed" and (not report["power_context_passed"] or not report["integrity_passed"] or not report["samples"]):
                report.update(status="failed",error_code="FINAL_EVIDENCE_REJECTED")
            write(output/"guard.json",report)
            for signum,handler in handlers.items():
                signal.signal(signum,handler)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","binary","helper-source","prompts","runtime-libraries","output","gpu-profile","provider-build","driver-build"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--microbatch",type=int,choices=BATCHES,default=8)
    p.add_argument("--mode",choices=("full","split","restore"),default="full")
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--max-seconds",type=int,default=1800)
    args=p.parse_args()
    report=run(args.config,args.binary,args.helper_source,args.prompts,args.runtime_libraries,args.output,
        gpu_profile=args.gpu_profile,provider_build=args.provider_build,driver_build=args.driver_build,
        microbatch=args.microbatch,mode=args.mode,diagnostic=args.diagnostic,max_seconds=args.max_seconds)
    print(json.dumps({"status":report["status"],"returncode":report["returncode"],"guard_violation":report["guard_violation"],"guard":str(args.output/"guard.json")}))
    return 0 if report["status"]=="completed" else 2

if __name__=="__main__":
    raise SystemExit(main())
