"""Focused regression tests for experimental packed GGUF transport."""
import gc
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from airllm.nextgen.gguf import GGUFMemoryError, GGUFReader, PackedBufferBudget, TensorInfo, GGMLType


class TestPackedBufferBudget(unittest.TestCase):
    def test_capacity_and_release(self):
        b = PackedBufferBudget(16)
        b._acquire(12)
        self.assertEqual(b.live_bytes, 12)
        with self.assertRaises(GGUFMemoryError):
            b._acquire(5)
        b._release(12)
        self.assertEqual(b.live_bytes, 0)
        self.assertEqual(b.peak_bytes, 12)

    def test_read_only_zero_copy_and_lifetime(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "weights.bin"
            path.write_bytes(bytes(range(144)))
            reader = GGUFReader.__new__(GGUFReader)
            reader._fd = os.open(path, os.O_RDONLY)
            reader.file_size = 144
            reader._lock = __import__('threading').RLock()
            reader._metadata_bytes = 0
            reader._reserved = 0
            reader.memory_budget_bytes = 1024
            reader.max_read_bytes = 144
            reader.tensors = {"w": TensorInfo("w", (256, 1), GGMLType.Q4_K, 0, 144, 144)}
            reader.stats = {"read_calls": 0, "bytes_read": 0, "logical_bytes": 0,
                            "tensor_bytes": 0, "index_bytes": 0,
                            "peak_operation_bytes": 0, "peak_reserved_bytes": 0}
            budget = PackedBufferBudget(144)
            try:
                arr = reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
                self.assertFalse(arr.flags.writeable)
                self.assertEqual(arr.tobytes(), bytes(range(144)))
                self.assertEqual(budget.live_bytes, 144)
                with self.assertRaises(GGUFMemoryError):
                    reader.read_packed_rows("w", 0, 1, copy=False, owner_budget=budget)
                del arr
                gc.collect()
                self.assertEqual(budget.live_bytes, 0)
                arr = reader.read_packed_rows("w", 0, 1)
                self.assertTrue(arr.flags.writeable)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
