#!/usr/bin/env python3
"""Reproduce small synthetic transport timings or Metal exporter lifetimes.

No checkpoint is downloaded or opened. CPU timings use normal OS cache policy;
they do not measure cold storage, physical SSD throughput or model generation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import tempfile
from time import perf_counter
import tracemalloc
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "air_llm"))
from airllm.nextgen.gguf import GGUFReader, PackedBufferBudget
from airllm.nextgen.gguf_packed_io import read_packed_rows_into


def _fixture(path, width, rows, raw):
    """One Q4_K tensor, no metadata, GGUFv3 with default 32-byte alignment."""
    name = b"w"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    header += struct.pack("<Q", len(name)) + name
    header += struct.pack("<IQQIQ", 2, width, rows, 12, 0)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + raw.tobytes())
    return path


def cpu_transport(repeats=51):
    raw = np.random.default_rng(20261024).integers(0, 256, (1024, 1152), dtype=np.uint8)
    records = {name: [] for name in ("baseline_two_host_copies", "readonly_copyfree", "direct_writable")}
    with tempfile.TemporaryDirectory(prefix="airllm-direct-cpu-") as temp:
        path = _fixture(Path(temp) / "synthetic.gguf", 2048, 1024, raw)
        with GGUFReader(path, memory_budget_bytes=32 << 20) as cp:
            budget = PackedBufferBudget(8 << 20)
            def baseline():
                value = cp.read_packed_rows("w", 0, 1024)
                return value.copy()
            functions = {
                "baseline_two_host_copies": baseline,
                "readonly_copyfree": lambda: cp.read_packed_rows("w", 0, 1024, copy=False, owner_budget=budget),
                "direct_writable": lambda: read_packed_rows_into(cp, "w", 0, 1024, owner_budget=budget),
            }
            for fn in functions.values():
                for _ in range(4):
                    value = fn()
                    del value
            order, rng = list(functions), random.Random(51)
            for _ in range(repeats):
                rng.shuffle(order)
                for name in order:
                    start = perf_counter()
                    value = functions[name]()
                    records[name].append(perf_counter() - start)
                    del value
            peaks, hashes, writable = {}, {}, {}
            for name, fn in functions.items():
                gc.collect()
                tracemalloc.start()
                value = fn()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                hashes[name] = hashlib.sha256(value).hexdigest()
                writable[name] = bool(value.flags.writeable)
                peaks[name] = {"current_bytes": current, "peak_bytes": peak}
                del value
                gc.collect()
            if len(set(hashes.values())) != 1 or cp._reserved or budget.live_bytes:
                raise AssertionError("Transport parity or final ownership accounting failed")
            return {
                "scope": "synthetic warm OS-cache CPU transport only; no model, Metal, cold/external SSD or physical storage claim",
                "platform": platform.platform(), "payload_bytes_per_read": raw.nbytes, "repeats": repeats,
                "order": "randomized each repetition after 4 warmups/method",
                "median_seconds": {name: statistics.median(v) for name, v in records.items()},
                "min_seconds": {name: min(v) for name, v in records.items()},
                "max_seconds": {name: max(v) for name, v in records.items()},
                "tracemalloc_single_read": peaks, "writable": writable, "sha256": hashes,
                "raw_seconds": records, "reader_stats": dict(cp.stats),
            }


def metal_lifetime(imports=100):
    import mlx.core as mx
    from airllm.nextgen.gguf_metal import packed_matmul
    if not mx.metal.is_available():
        return {"scope": "synthetic Metal lifetime experiment", "metal_available": False, "records": {}}
    raw = np.zeros((8, 144), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(.01).tobytes(), np.uint8)
    mx.set_cache_limit(1 << 20)
    records = {}
    with tempfile.TemporaryDirectory(prefix="airllm-lifetime-") as temp:
        path = _fixture(Path(temp) / "tiny.gguf", 256, 8, raw)
        for mode, zero_copy in (("readonly", False), ("direct", False), ("direct", True)):
            label = mode + ("_dlpack" if zero_copy else "_array")
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            budget = PackedBufferBudget(1 << 20)
            stages, refs, outputs, stats = [], [], [], {}
            with GGUFReader(path, memory_budget_bytes=1 << 20) as cp:
                def stage(name):
                    stages.append({
                        "stage": name, "live_bytes": budget.live_bytes, "peak_bytes": budget.peak_bytes,
                        "numpy_live": sum(ref() is not None for ref in refs),
                        "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
                    })
                with mx.stream(mx.new_stream(mx.gpu)):
                    x = mx.ones((1, 256), dtype=mx.float32)
                    mx.eval(x)
                    stage("start")
                    for index in range(imports):
                        array = (cp.read_packed_rows("w", 0, 8, copy=False, owner_budget=budget)
                                 if mode == "readonly" else read_packed_rows_into(cp, "w", 0, 8, owner_budget=budget))
                        refs.append(weakref.ref(array))
                        value = packed_matmul(array, x, 12, zero_copy=zero_copy, stats=stats)
                        del array
                        outputs.append(value)
                        del value
                        if index in (0, 9, 49, imports - 1):
                            stage("evaluated_" + str(index + 1))
                    stage("after_eval_all_outputs_retained")
                    outputs.clear()
                    stage("after_del_outputs")
                    mx.synchronize()
                    stage("after_synchronize")
                    gc.collect()
                    stage("after_gc_collect")
                    mx.clear_cache()
                    stage("after_clear_cache")
                    mx.synchronize()
                    stage("after_second_synchronize")
                    gc.collect()
                    stage("after_second_gc_collect")
                    del x
            records[label] = {"stages": stages, "stats": stats}
    return {
        "scope": f"{imports} synthetic packed Metal matmul calls; each packed allocation 1152 B, no model weights",
        "metal_available": True, "platform": platform.platform(), "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "metal-lifetime"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--imports", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000 or not 1 <= args.imports <= 100:
        parser.error("repeats must be 1..1000 and imports 1..100")
    report = cpu_transport(args.repeats) if args.mode == "cpu" else metal_lifetime(args.imports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "output": str(args.output.resolve()),
                      "scope": report["scope"]}, indent=2))


if __name__ == "__main__":
    main()
