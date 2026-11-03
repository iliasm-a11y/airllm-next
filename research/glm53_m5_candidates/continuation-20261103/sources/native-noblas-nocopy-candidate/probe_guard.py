"""SOURCE ONLY: resource validation of a closed partial GPU128 diagnostic.

The existing complete comparator remains unchanged and continues to reject
complete=false. This distinct validator accepts only probe_only/complete=false,
then preserves the pinned native comparator resource checks without modification.
It does not validate numerical fidelity, a complete prompt or generation.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
GiB,MiB=1024**3,1024**2

def _record(value):
    if type(value) is not dict:raise ValueError('explicit parsed probe guard required')
    return value

def validate_probe_guard(receipt):
    """Require a conservative sufficient domain of the existing native policy.

    Warning pressure alone is allowed only with full warning headroom and
    bounded warning swap. Reserve shortfalls are refused without relying on
    grace timestamps. The existing watchdog is replayed as a consistency
    check; no native threshold or numeric-fidelity criterion is changed.
    """
    from air_llm.airllm.nextgen.glm_native import NativeGlmModel, NativeLimits

    def number(value, name, *, minimum=0, maximum=2**53):
        if (type(value) not in (int, float) or not minimum <= value <= maximum or
                not math.isfinite(value)):
            raise ValueError("missing or nonfinite numeric guard metadata: " + name)
        return value

    def byte_count(value, name, *, minimum=0, maximum=2**64 - 1):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("missing or invalid integer guard metadata: " + name)
        return value

    row = _record(receipt)
    if not isinstance(row, dict):
        raise ValueError("an explicit guard receipt is required")
    if (row.get("status") != "probe_only" or row.get("complete") is not False or
            row.get("diagnostic_window_complete") is not True or row.get("full_prompt_computed") is not False or
            row.get("whole_model_qualified") is not False or row.get("full_campaign_qualified") is not False or
            type(row.get("returncode")) is not int or
            row["returncode"] != 0 or "guard_violation" not in row or row["guard_violation"] is not None):
        raise ValueError("native run did not exit successfully under clear guards")
    if row.get("power_context_passed") is not True or row.get("integrity_passed") is not True:
        raise ValueError("AC/lid/continuity and integrity receipts are required")
    limits = row.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("complete native resource limits are required")
    native = limits.get("native")
    defaults = vars(NativeLimits())
    if not isinstance(native, dict) or set(native) != set(defaults):
        raise ValueError("complete native watchdog limits are required")
    for name, default in defaults.items():
        value = native[name]
        if type(default) is int:
            byte_count(value, name, minimum=1)
        else:
            number(value, name, minimum=0)
            if value <= 0:
                raise ValueError("positive native watchdog limit required: " + name)
        if name in ("reserve_bytes", "base_workspace_bytes", "warning_headroom_bytes"):
            if value < default:
                raise ValueError("native resource reserve was relaxed: " + name)
        elif name not in ("max_rss_bytes", "max_swap_growth_bytes", "request_timeout_seconds") and value > default:
            raise ValueError("native watchdog limit was relaxed: " + name)
    if not 0 < native["request_timeout_seconds"] <= 2700:
        raise ValueError("native request timeout exceeds the authorized ceiling")
    rss = byte_count(limits.get("rss_bytes"), "limits.rss_bytes", minimum=1, maximum=int(8.5 * GiB))
    swap = byte_count(limits.get("swap_growth_bytes"), "limits.swap_growth_bytes", minimum=1, maximum=32 * MiB)
    if native["max_rss_bytes"] != rss or native["max_swap_growth_bytes"] != swap:
        raise ValueError("native and reported hard resource limits differ")
    selected_limits = NativeLimits(**native)
    interval = number(row.get("sampling_interval_seconds"), "sampling_interval_seconds", maximum=.5)
    if interval <= 0:
        raise ValueError("positive sampling interval required")

    policy_path = Path(NativeGlmModel._guard_resource_sample.__code__.co_filename).resolve()
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    preflight = row.get("preflight")
    metadata = preflight.get("metadata") if isinstance(preflight, dict) else None
    recorded_hashes = metadata.get("sha256") if isinstance(metadata, dict) else None
    if not isinstance(recorded_hashes, dict):
        raise ValueError("original native watchdog source identity is required")
    policies = [value for name, value in recorded_hashes.items()
                if isinstance(name, str) and name.replace("\\", "/").endswith("/airllm/nextgen/glm_native.py")]
    if len(policies) != 1 or policies[0] != policy_sha256:
        raise ValueError("native watchdog source identity missing or changed")

    baseline = row.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("native baseline metadata is required")
    baseline_time = number(baseline.get("monotonic"), "baseline.monotonic")
    byte_count(baseline.get("swap_used_bytes"), "baseline.swap_used_bytes")
    byte_count(baseline.get("available_bytes"), "baseline.available_bytes")
    if type(baseline.get("pressure_level")) is not int or baseline["pressure_level"] != 1:
        raise ValueError("native startup requires normal pressure")
    admission = row.get("admission")
    if not isinstance(admission, dict) or admission.get("passed") is not True:
        raise ValueError("successful native startup admission receipt required")
    admitted_sample = admission.get("sample")
    if not isinstance(admitted_sample, dict):
        raise ValueError("native startup admission sample is required")
    number(admitted_sample.get("monotonic"), "admission.sample.monotonic")
    byte_count(admitted_sample.get("swap_used_bytes"), "admission.sample.swap_used_bytes")
    byte_count(admitted_sample.get("available_bytes"), "admission.sample.available_bytes")
    if (type(admitted_sample.get("pressure_level")) is not int or admitted_sample["pressure_level"] != 1 or
            any(admitted_sample.get(key) != baseline[key] for key in
                ("monotonic", "swap_used_bytes", "available_bytes", "pressure_level"))):
        raise ValueError("native startup admission and baseline differ")
    required = byte_count(admission.get("required_bytes_estimate"), "admission.required_bytes_estimate", minimum=1)
    if required < selected_limits.reserve_bytes + selected_limits.base_workspace_bytes or baseline["available_bytes"] < required:
        raise ValueError("native startup workspace and reserve admission failed")

    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("no guard observations")
    guard = object.__new__(NativeGlmModel)
    guard.limits = selected_limits
    guard._warning_started = guard._headroom_low_started = None
    guard._pressure_context = None
    previous_time = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("structured native resource sample required")
        now = number(sample.get("monotonic"), "sample.monotonic")
        if now < baseline_time or previous_time is not None and now <= previous_time:
            raise ValueError("native resource sample clocks are reversed or repeated")
        previous_time = now
        byte_count(sample.get("rss_bytes"), "sample.rss_bytes")
        byte_count(sample.get("available_bytes"), "sample.available_bytes")
        used = byte_count(sample.get("swap_used_bytes"), "sample.swap_used_bytes")
        growth = byte_count(sample.get("swap_growth_bytes"), "sample.swap_growth_bytes")
        if growth != max(0, used - baseline["swap_used_bytes"]):
            raise ValueError("reported swap growth differs from baseline counters")
        pressure = sample.get("pressure_level")
        if type(pressure) is not int or pressure not in (1, 2):
            raise ValueError("only observed normal/contextual-warning pressure is admitted")
        if sample["available_bytes"] < selected_limits.reserve_bytes:
            raise ValueError("reserve shortfall refused without inferring native grace timing")
        if pressure == 2 and (
                sample["available_bytes"] < selected_limits.reserve_bytes + selected_limits.warning_headroom_bytes or
                growth > selected_limits.warning_swap_growth_bytes):
            raise ValueError("warning-pressure sufficient headroom/swap bounds failed")
        code = NativeGlmModel._guard_resource_sample(guard, sample, baseline, now)
        if code is not None:
            raise ValueError(f"existing native resource policy rejected sample{index}: {code}")
    return row

