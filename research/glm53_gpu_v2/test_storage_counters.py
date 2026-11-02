"""Portable counter/timestamp fixtures; no device queries or GPU execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.glm53_gpu_v2.storage_counters import compare_snapshots, parse_gpu_times


def snapshot(*devices):
    return {"available": True, "devices": list(devices),
            "scope": "system-wide IOBlockStorageDriver counters; controller reads, not NAND"}


def device(registry_id, read_bytes=0, write_bytes=0):
    return {"registry_id": registry_id, "statistics": {
        "Bytes (Read)": read_bytes, "Bytes (Write)": write_bytes,
        "Operations (Read)": read_bytes // 10, "Operations (Write)": write_bytes // 10}}


class StorageCounterTests(unittest.TestCase):
    def gpu(self, content):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "native.log"
            path.write_text(content)
            return parse_gpu_times(path)

    def test_device_global_deltas_preserve_actual_scope(self):
        result = compare_snapshots(snapshot(device(1, 100, 50)), snapshot(device(1, 160, 70)))
        self.assertTrue(result["available"])
        self.assertEqual(result["devices"][0]["delta"], {
            "Bytes (Read)": 60, "Bytes (Write)": 20, "Operations (Read)": 6, "Operations (Write)": 2})
        self.assertIn("system-wide", result["scope"])
        self.assertIn("other processes", result["limits"])

    def test_counter_reset_or_missing_snapshot_cannot_be_reported_as_zero_reads(self):
        result = compare_snapshots(snapshot(device(1, 100)), snapshot(device(1, 90)))
        self.assertFalse(result["available"])
        self.assertIn("reset", result["error"])
        self.assertFalse(compare_snapshots({"available": False}, snapshot(device(1)))["available"])

    def test_disappearing_device_cannot_silently_become_an_available_partial_total(self):
        before = snapshot(device(1, 100), device(2, 200))
        after = snapshot(device(1, 120))
        result = compare_snapshots(before, after)
        self.assertFalse(result["available"])
        self.assertFalse(compare_snapshots(before, snapshot())["available"])
        self.assertFalse(compare_snapshots(snapshot(device(1)), snapshot(device(1), device(2)))["available"])

    def test_gpu_completed_interval_union_counts_overlap_once_and_excludes_invalid_status(self):
        result = self.gpu(
            "AIRLLM_GPU_TIME origin=compute start=1.0 end=3.0 seconds=2.0 status=4\n"
            "AIRLLM_GPU_TIME origin=copy start=2.0 end=5.0 seconds=3.0 status=4\n"
            "AIRLLM_GPU_TIME origin=nested start=2.5 end=2.75 seconds=0.25 status=4\n"
            "AIRLLM_GPU_TIME origin=disjoint start=6.0 end=7.0 seconds=1.0 status=4\n"
            "AIRLLM_GPU_TIME origin=failed start=6.0 end=9.0 seconds=3.0 status=5\n"
            "AIRLLM_GPU_TIME origin=backwards start=8.0 end=7.0 seconds=-1.0 status=4\n")
        self.assertEqual(result["recorded_commands"], 6)
        self.assertEqual(result["valid_commands"], 4)
        self.assertAlmostEqual(result["sum_command_seconds"], 6.25)
        self.assertAlmostEqual(result["union_command_seconds"], 5)
        self.assertFalse(result["complete_capture_guaranteed"])

    def test_invalid_gpu_duration_is_not_a_valid_negative_compute_measurement(self):
        result = self.gpu(
            "AIRLLM_GPU_TIME origin=negative start=1.0 end=2.0 seconds=-3.0 status=4\n"
            "AIRLLM_GPU_TIME origin=inconsistent start=1.0 end=2.0 seconds=0.5 status=4\n")
        self.assertEqual(result["recorded_commands"], 2)
        self.assertEqual(result["valid_commands"], 0)
        self.assertEqual(result["sum_command_seconds"], 0)
        self.assertEqual(result["union_command_seconds"], 0)

    def test_log_without_instrumentation_has_no_recorded_gpu_observations(self):
        result = self.gpu("ordinary server log; no timestamp instrumentation was enabled\n")
        self.assertEqual(result["recorded_commands"], 0)
        self.assertEqual(result["valid_commands"], 0)
        self.assertEqual(result["rows"], [])
        self.assertFalse(result["complete_capture_guaranteed"])


if __name__ == "__main__":
    unittest.main()
