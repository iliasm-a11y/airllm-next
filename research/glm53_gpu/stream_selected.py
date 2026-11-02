"""Replay one captured ordered MoE selection through exact cache + Metal Q2_K.

This is a projection microbenchmark using a seeded synthetic activation. It
does not run the router, SwiGLU, MoE reduction, KDA, or GLM decoding, and cannot
establish full-model fidelity or a CPU10 decoding speedup.

Requires --experimental to execute. Keep the owning process memory/swap monitor
active as well: local packed/device payload bounds do not bound total RSS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np

from research.glm53_m5.compare_fidelity import json_lines, tensor_view
from .expert_cache import ByteSpan, CacheUnavailable, ExpertByteCache, ExpertRef
from .quantized import Q2KMetalMatvec
from .quantized_probe import NativeOracle, compare, read_selected_rows


MIB = 1024**2


def read_route_selection(capture, *, layer=3, step=1, column=0):
    """Read exact ordered IDs using the native comparator's GGML stride view."""
    root = Path(capture)
    manifest = root / "tensors.jsonl"
    if manifest.stat().st_size > 16 * MIB:
        raise ValueError("capture manifest exceeds replay parsing budget")
    if any(type(n) is not int or n < 0 for n in (layer, step, column)):
        raise ValueError("layer, step and column must be nonnegative integers")
    name = f"ffn_moe_topk-{layer}"
    rows = [r for r in json_lines(manifest) if r.get("step") == step and r.get("name") == name]
    if len(rows) != 1:
        raise ValueError("replay requires exactly one selected layer/step route tensor")
    row = rows[0]
    shape = row.get("shape", [])
    if (row.get("type") != 26 or len(shape) != 4 or shape[0] != 8 or
            shape[2:] != [1, 1] or not 0 <= column < shape[1] or
            type(row.get("bytes")) is not int or not 0 < row["bytes"] <= MIB):
        raise ValueError("expected bounded I32 top-8 routes with a valid token column")
    view = tensor_view(root, row)
    ids = [int(v) for v in view[:, column, 0, 0]]
    if any(v < 0 for v in ids):
        raise ValueError("negative selected expert ID")
    return ids, {
        "capture_directory_name": root.name, "route_tensor": name,
        "step": step, "token_column": column, "shape": shape,
        "ordered_expert_ids": ids,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "route_payload_sha256": hashlib.sha256((root / row["file"]).read_bytes()).hexdigest(),
    }


def selected_projection_refs(gguf, ordered_ids, *, layer=3, projection="gate"):
    """Get one metadata row, then use complete original packed expert spans."""
    if projection not in ("gate", "up", "down"):
        raise ValueError("projection must be gate, up or down")
    if type(layer) is not int or layer < 0 or len(ordered_ids) != 8:
        raise ValueError("replay requires one layer and exactly eight ordered expert IDs")
    if any(type(e) is not int or e < 0 for e in ordered_ids):
        raise ValueError("expert IDs must be nonnegative integers")
    tensor = f"blk.{layer}.ffn_{projection}_exps.weight"
    first_expert = ordered_ids[0]
    header_row, width, metadata = read_selected_rows(gguf, tensor, first_expert, 0, 1)
    width_from_dims, rows, experts = metadata["dims"]
    if width_from_dims != width or width > 65536 or not 1 <= rows <= 8192:
        raise ValueError("projection exceeds the bounded Metal matrix dimensions")
    if any(e >= experts for e in ordered_ids):
        raise ValueError("selected expert is outside the GGUF bank")
    row_bytes = width // 256 * 84
    expert_bytes = rows * row_bytes
    # read_selected_rows returned row zero of first_expert, not bank row zero.
    bank_offset = metadata["absolute_offset"] - first_expert * expert_bytes
    refs = [ExpertRef((tensor, e), (ByteSpan(bank_offset + e * expert_bytes, expert_bytes),))
            for e in ordered_ids]
    file_size = Path(gguf).stat().st_size
    if bank_offset < 0 or any(r.spans[0].offset + r.nbytes > file_size for r in refs):
        raise ValueError("computed complete expert span is outside the shard")
    return refs, width, rows, {
        "file": Path(gguf).name, "file_bytes": file_size, "tensor": tensor,
        "dims": metadata["dims"], "bank_absolute_offset": bank_offset,
        "row_bytes": row_bytes, "expert_bytes": expert_bytes,
        "header_probe_read_bytes": int(header_row.nbytes),
        "header_probe_sha256": metadata["sha256_selected_bytes"],
    }


class MemoryAdmission:
    """Fail-closed macOS pressure + available-memory check, without psutil dependency.

    This is a local admission check, not the independent process/swap watchdog.
    Available-memory estimates and pressure can change after a check.
    """
    def __init__(self, *, min_available_bytes=512 * MIB, device_allowance_bytes=16 * MIB):
        self.minimum = min_available_bytes
        self.device_allowance = device_allowance_bytes
        self.last_snapshot = None

    @staticmethod
    def _text(args):
        return subprocess.check_output(args, text=True, timeout=3).strip()

    def __call__(self, incoming_bytes):
        try:
            pressure = int(self._text(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))
            try:
                import psutil
            except ImportError:
                psutil = None
            if psutil is not None:
                available = int(psutil.virtual_memory().available)
                source = "psutil.virtual_memory.available"
            else:
                text = self._text(["/usr/bin/vm_stat"])
                page = re.search(r"page size of (\d+) bytes", text)
                counts = []
                for label in ("Pages free", "Pages inactive", "Pages speculative"):
                    found = re.search(r"^" + re.escape(label) + r":\s*(\d+)\.", text, re.M)
                    if found is None:
                        raise ValueError("vm_stat omitted required page counts")
                    counts.append(int(found.group(1)))
                if page is None:
                    raise ValueError("vm_stat omitted page size")
                available = sum(counts) * int(page.group(1))
                source = "vm_stat free+inactive+speculative estimate"
            approved = pressure == 1 and available >= self.minimum + incoming_bytes + self.device_allowance
            self.last_snapshot = {"pressure_level": pressure, "available_bytes": available,
                                  "available_source": source, "incoming_packed_bytes": incoming_bytes,
                                  "device_allowance_bytes": self.device_allowance,
                                  "minimum_available_bytes": self.minimum, "approved": approved}
            return approved
        except Exception as exc:
            self.last_snapshot = {"approved": False, "error": f"{type(exc).__name__}: {exc}"}
            return False


def _sha_array(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def run_replay(gguf, refs, width, rows, libraries, *, admission,
               cache_bytes=64 * MIB, device_bytes=16 * MIB, passes=2,
               seed=5305, matrix_factory=Q2KMetalMatvec, oracle_factory=NativeOracle):
    """Run bounded ordered projection replay. Call only inside an external guard."""
    if (type(cache_bytes) is not int or not 1 <= cache_bytes <= 64 * MIB or
            type(device_bytes) is not int or not 1 <= device_bytes <= 16 * MIB or
            passes != 2):
        raise ValueError("replay requires two passes, cache <=64 MiB and device budget <=16 MiB")
    if (type(width) is not int or not 256 <= width <= 65536 or width % 256 or
            type(rows) is not int or not 1 <= rows <= 8192):
        raise ValueError("replay width/rows exceed bounded Q2_K matrix dimensions")
    if len(refs) != 8 or sum({r.key: r.nbytes for r in refs}.values()) > cache_bytes:
        raise ValueError("the exact top-8 selection must fit the shared packed-byte budget")
    row_bytes = width // 256 * 84
    if any(len(r.spans) != 1 or r.nbytes != rows * row_bytes for r in refs):
        raise ValueError("replay requires one complete projection span per selected expert")
    x = np.random.default_rng(seed).normal(size=width).astype("f4")
    oracle = oracle_factory(libraries)
    all_passes = []
    passed = True
    with ExpertByteCache(gguf, cache_bytes, max_inflight_bytes=cache_bytes,
                         workers=1, enabled=True, admission=admission) as cache:
        for pass_index in range(passes):
            began = time.perf_counter()
            cache.prefetch_selected(refs)
            submitted = time.perf_counter()
            with cache.acquire_selected(refs) as leases:
                acquired = time.perf_counter()
                records = []
                for lease, ref in zip(leases, refs):
                    if not admission(device_bytes):
                        raise CacheUnavailable("memory admission rejected the next GPU projection")
                    packed = np.frombuffer(lease.data, dtype="u1").reshape(rows, row_bytes)
                    cpu_began = time.perf_counter()
                    native = oracle.matvec(packed, x)
                    cpu_ended = time.perf_counter()
                    prepared = time.perf_counter()
                    with matrix_factory(packed, width, budget_bytes=device_bytes) as matrix:
                        gpu_began = time.perf_counter()
                        actual = np.asarray(matrix(x, mode="cpu_q8k", synchronize=True))
                        gpu_ended = time.perf_counter()
                        known_payload = matrix.payload_bytes
                    finite = bool(np.isfinite(native).all() and np.isfinite(actual).all())
                    agreement = compare(actual, native) if finite else {
                        "bit_identical": False, "max_abs": None, "rmse": None,
                        "elements": int(native.size)}
                    passed = passed and finite and agreement["bit_identical"]
                    records.append({
                        "expert_id": ref.key[1], "absolute_offset": ref.spans[0].offset,
                        "length": ref.nbytes, "sha256_packed_bytes": hashlib.sha256(lease.data).hexdigest(),
                        "native_output_sha256": _sha_array(native), "gpu_output_sha256": _sha_array(actual),
                        "finite": finite, "comparison_vs_native_arm_dot": agreement,
                        "known_matrix_payload_bytes": known_payload,
                        "native_ctypes_single_caller_seconds": cpu_ended - cpu_began,
                        "gpu_prepare_seconds": gpu_began - prepared,
                        "gpu_evaluate_synchronized_seconds": gpu_ended - gpu_began,
                    })
                    # Calls synchronize; release host/device owners before the
                    # next expert. Never queue multiple full projection weights.
                    del packed, native, actual, matrix
            all_passes.append({
                "pass": pass_index, "prefetch_submission_seconds": submitted - began,
                "acquire_wait_seconds": acquired - submitted,
                "total_seconds": time.perf_counter() - began,
                "ordered_experts": records, "cache": cache.snapshot(),
            })
        before_close = cache.snapshot()
    return {
        "passed_bitexact_projection_replay": bool(passed),
        "activation_seed": seed, "activation_sha256": _sha_array(x),
        "activation_origin": "seeded synthetic standard normal float32; not captured model activations",
        "cache_budget_bytes": cache_bytes, "matrix_payload_budget_bytes": device_bytes,
        "passes": all_passes, "cache_before_close": before_close,
        "cache_after_close": cache.snapshot(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", action="store_true", help="explicitly enable this research execution")
    parser.add_argument("--gguf", required=True, help="GGUF shard containing expert banks")
    parser.add_argument("--capture", required=True, help="native capture directory with tensors.jsonl")
    parser.add_argument("--libraries", required=True, help="pinned native GGML dylib directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--column", type=int, default=0, help="explicit prefill token column; decode has one")
    parser.add_argument("--projection", choices=("gate", "up", "down"), default="gate")
    args = parser.parse_args(argv)
    if not args.experimental:
        parser.error("execution is disabled by default; pass --experimental inside the independent memory watchdog")
    admission = MemoryAdmission()
    if not admission(0):
        raise CacheUnavailable("initial memory/pressure admission failed: " + json.dumps(admission.last_snapshot))
    ids, route = read_route_selection(args.capture, layer=args.layer, step=args.step, column=args.column)
    refs, width, rows, projection = selected_projection_refs(args.gguf, ids, layer=args.layer,
                                                            projection=args.projection)
    import mlx.core as mx
    mx.set_cache_limit(0)
    mx.set_memory_limit(128 * MIB)  # guideline, not the independent RSS/swap watchdog
    result = run_replay(args.gguf, refs, width, rows, args.libraries, admission=admission)
    result.update({
        "schema": 1, "scope": "ordered captured top-8 IDs, one Q2_K projection, synthetic activation; not GLM decode",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "route_source": route, "projection": projection, "mlx_version": mx.__version__,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "last_local_memory_admission": admission.last_snapshot,
        "native_libraries_sha256": {
            name: hashlib.sha256((Path(args.libraries) / name).read_bytes()).hexdigest()
            for name in ("libggml-base.dylib", "libggml-cpu.dylib")},
        "limitations": [
            "replays selected expert IDs; does not execute or validate the router",
            "one projection only: no SwiGLU, gate/up combination, down path or MoE reduction unless separately selected",
            "activation is synthetic, not the actual captured expert input",
            "native reference is one dot per row via ctypes with one caller, not CPU10 decoding",
            "macOS file cache, background workload and compiler cache are uncontrolled",
            "logical selected-byte reads are not physical SSD traffic or NAND measurements",
            "local payload budgets and MLX guideline do not replace the independent process/swap watchdog",
            "no end-to-end token-rate gain or complete GLM fidelity is established",
        ],
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed_bitexact_projection_replay": result["passed_bitexact_projection_replay"],
                      "route_ids": ids, "cache": result["cache_before_close"]}, indent=2))
    return 0 if result["passed_bitexact_projection_replay"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
