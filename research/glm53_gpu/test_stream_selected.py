"""Portable integration fixtures; no Metal/model performance is measured."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import struct
import unittest

import numpy as np

from research.glm53_gpu.expert_cache import ByteSpan, CacheUnavailable, ExpertRef
from research.glm53_gpu.stream_selected import (
    MemoryAdmission, main, read_route_selection, run_replay, selected_projection_refs)


def write_bank(path, *, rows=2, width=256, experts=12):
    name = b"blk.3.ffn_gate_exps.weight"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    header += struct.pack("<Q", len(name)) + name + struct.pack("<IQQQIQ", 3, width, rows, experts, 10, 0)
    header += b"\0" * (-len(header) % 32)
    base = len(header)
    bank = np.arange(experts * rows * (width // 256 * 84), dtype="u1")
    path.write_bytes(header + bank.tobytes())
    return base, bank


def write_capture(root, *, columns=1, name="ffn_moe_topk-3", step=1):
    # GGML dimension zero contiguous; deliberate token-column padding.
    raw = bytearray(columns * 48)
    ids = np.ndarray((8, columns, 1, 1), dtype="<i4", buffer=raw, strides=(4, 48, 48 * columns, 48 * columns))
    for column in range(columns):
        ids[:, column, 0, 0] = np.array([9, 2, 7, 1, 6, 0, 8, 3]) + column
    (root / "routes.bin").write_bytes(raw)
    row = {"step": step, "name": name, "type": 26, "shape": list(ids.shape),
           "strides": list(ids.strides), "bytes": len(raw), "file": "routes.bin"}
    (root / "tensors.jsonl").write_text(json.dumps(row) + "\n")
    return row


class FakeOracle:
    def __init__(self, ignored):
        pass

    def matvec(self, packed, x):
        return packed.sum(axis=1, dtype="u4").astype("f4")


class FakeMatrix:
    active = 0
    peak_active = 0

    def __init__(self, packed, width, *, budget_bytes):
        self.packed = packed
        self.payload_bytes = 2 * packed.nbytes

    def __enter__(self):
        FakeMatrix.active += 1
        FakeMatrix.peak_active = max(FakeMatrix.peak_active, FakeMatrix.active)
        return self

    def __exit__(self, *unused):
        FakeMatrix.active -= 1

    def __call__(self, x, *, mode, synchronize):
        assert mode == "cpu_q8k" and synchronize is True
        return self.packed.sum(axis=1, dtype="u4").astype("f4")


class StreamSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gguf = self.root / "fixture.gguf"
        self.base, self.bank = write_bank(self.gguf)
        write_capture(self.root)
        FakeMatrix.active = FakeMatrix.peak_active = 0

    def tearDown(self):
        self.temp.cleanup()

    def test_route_stride_view_preserves_original_order(self):
        ids, metadata = read_route_selection(self.root)
        self.assertEqual(ids, [9, 2, 7, 1, 6, 0, 8, 3])
        self.assertEqual(metadata["shape"], [8, 1, 1, 1])
        self.assertEqual(metadata["route_payload_sha256"],
                         hashlib.sha256((self.root / "routes.bin").read_bytes()).hexdigest())

    def test_prefill_column_is_explicit_and_bounds_checked(self):
        write_capture(self.root, columns=3)
        ids, metadata = read_route_selection(self.root, column=2)
        self.assertEqual(ids, [11, 4, 9, 3, 8, 2, 10, 5])
        self.assertEqual(metadata["token_column"], 2)
        with self.assertRaises(ValueError):
            read_route_selection(self.root, column=3)

    def test_missing_duplicate_or_non_top8_route_is_rejected(self):
        with self.assertRaises(ValueError):
            read_route_selection(self.root, step=2)
        row = write_capture(self.root)
        (self.root / "tensors.jsonl").write_text((json.dumps(row) + "\n") * 2)
        with self.assertRaises(ValueError):
            read_route_selection(self.root)
        row["shape"][0] = 7
        (self.root / "tensors.jsonl").write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            read_route_selection(self.root)

    def test_metadata_probe_corrects_first_expert_to_bank_base(self):
        ids, _ = read_route_selection(self.root)
        refs, width, rows, metadata = selected_projection_refs(self.gguf, ids)
        self.assertEqual((width, rows), (256, 2))
        self.assertEqual(metadata["bank_absolute_offset"], self.base)
        self.assertEqual(refs[0].spans[0], ByteSpan(self.base + 9 * 168, 168))
        self.assertEqual(refs[1].spans[0], ByteSpan(self.base + 2 * 168, 168))
        self.assertEqual(metadata["header_probe_read_bytes"], 84)
        with self.assertRaises(ValueError):
            selected_projection_refs(self.gguf, [12] * 8)

    def test_two_pass_cache_kernel_integration_without_gpu(self):
        ids, _ = read_route_selection(self.root)
        refs, width, rows, _ = selected_projection_refs(self.gguf, ids)
        result = run_replay(self.gguf, refs, width, rows, "unused",
                            admission=lambda incoming: True,
                            matrix_factory=FakeMatrix, oracle_factory=FakeOracle)
        self.assertTrue(result["passed_bitexact_projection_replay"])
        self.assertEqual(FakeMatrix.peak_active, 1)
        self.assertEqual(FakeMatrix.active, 0)
        self.assertEqual(len(result["passes"]), 2)
        for record in result["passes"]:
            self.assertEqual([r["expert_id"] for r in record["ordered_experts"]], ids)
            self.assertEqual(record["cache"]["logical_bytes_read"], 8 * 168)
            self.assertEqual(record["cache"]["fills_submitted"], 8)
            for ref, expert in zip(refs, record["ordered_experts"]):
                original = self.gguf.read_bytes()[ref.spans[0].offset:ref.spans[0].offset + 168]
                self.assertEqual(expert["sha256_packed_bytes"], hashlib.sha256(original).hexdigest())
                self.assertEqual(expert["native_output_sha256"], expert["gpu_output_sha256"])
        self.assertEqual(result["cache_after_close"]["reserved_bytes"], 0)
        self.assertEqual(result["cache_before_close"]["reused_requests"], 24)

    def test_numerical_mismatch_is_recorded_as_failure(self):
        class WrongMatrix(FakeMatrix):
            def __call__(self, *args, **kwargs):
                return super().__call__(*args, **kwargs) + np.float32(1)
        ids, _ = read_route_selection(self.root)
        refs, width, rows, _ = selected_projection_refs(self.gguf, ids)
        result = run_replay(self.gguf, refs, width, rows, "unused", admission=lambda n: True,
                            matrix_factory=WrongMatrix, oracle_factory=FakeOracle)
        self.assertFalse(result["passed_bitexact_projection_replay"])

    def test_device_admission_rejects_before_constructing_matrix(self):
        ids, _ = read_route_selection(self.root)
        refs, width, rows, _ = selected_projection_refs(self.gguf, ids)
        with self.assertRaises(CacheUnavailable):
            run_replay(self.gguf, refs, width, rows, "unused",
                       admission=lambda n: n < 16 * 1024**2,
                       matrix_factory=FakeMatrix, oracle_factory=FakeOracle)
        self.assertEqual(FakeMatrix.peak_active, 0)

    def test_no_execution_without_experimental_switch(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--gguf", "missing", "--capture", "missing", "--libraries", "missing", "--output", "missing"])
        self.assertEqual(caught.exception.code, 2)

    def test_memory_admission_errors_fail_closed(self):
        guard = MemoryAdmission()
        with patch.object(guard, "_text", side_effect=OSError("no pressure information")):
            self.assertFalse(guard(0))
        self.assertFalse(guard.last_snapshot["approved"])

    def test_budgets_are_bounded_before_loading(self):
        with self.assertRaises(ValueError):
            run_replay(self.gguf, [], 256, 2, "unused", admission=lambda n: True,
                       cache_bytes=65 * 1024**2, matrix_factory=FakeMatrix, oracle_factory=FakeOracle)


if __name__ == "__main__":
    unittest.main()
