"""Trace/reuse-pipeline fixtures, without full model or GPU execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from research.glm53_gpu_v2.cache_replay import read_routes, replay_io, replay_policy, replay_reuse


class CacheReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "weights.gguf"
        self.source.write_bytes(bytes(range(256)) * 4)
        self.metadata = {layer: [{"offset": 0, "expert_bytes": 4}] for layer in range(3, 45)}

    def tearDown(self):
        self.temp.cleanup()

    def capture(self):
        manifest = []
        for step in range(4):
            for layer in range(3, 45):
                values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="<i4")
                filename = f"{step}-{layer}.bin"
                (self.root / filename).write_bytes(values.tobytes())
                manifest.append({"step": step, "name": f"ffn_moe_topk-{layer}", "type": 26,
                                 "shape": [8, 1, 1, 1], "strides": [4, 32, 32, 32],
                                 "bytes": 32, "file": filename})
        path = self.root / "tensors.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        return manifest

    def test_complete_four_step_42_layer_capture_and_hash(self):
        self.capture()
        rows, source = read_routes(self.root)
        self.assertEqual(len(rows), 4 * 42)
        self.assertEqual(source["selected_expert_elements"], 4 * 42 * 8)
        self.assertEqual(rows[0], {"step": 0, "layer": 3, "column": 0, "ids": list(range(1, 9))})
        self.assertEqual(len(source["route_ids_sha256"]), 64)
        manifest = self.capture()[:-1]
        (self.root / "tensors.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
        with self.assertRaises(ValueError):
            read_routes(self.root)

    def test_policy_replay_bounds_logical_payload_and_rejects_unmeasured_weighted(self):
        self.capture()
        records, _ = read_routes(self.root)
        result = replay_policy(records, self.metadata, capacity_bytes=64, policy="lru")
        self.assertLessEqual(result["peak_payload_bytes"], 64)
        self.assertEqual(result["hits"] + result["misses"], 4 * 42 * 8)
        self.assertEqual(result["logical_miss_bytes"], result["misses"] * 4)
        self.assertFalse(result["physical_io_measured"])
        with self.assertRaises(ValueError):
            replay_policy(records, self.metadata, capacity_bytes=64, policy="weighted")

    def test_policy_phase_split_preserves_totals_without_calling_prefill_hits_decode(self):
        records = [
            {"layer": 3, "step": 0, "column": 0, "ids": list(range(1, 9))},
            {"layer": 3, "step": 1, "column": 0, "ids": list(range(1, 9))},
            {"layer": 3, "step": 2, "column": 0, "ids": list(range(9, 17))}]
        result = replay_policy(records, self.metadata, capacity_bytes=64, policy="lru")
        self.assertEqual(result["by_phase"]["prefill"], {
            "hits": 0, "misses": 8, "logical_miss_bytes": 32, "selected_groups": 1})
        self.assertEqual(result["by_phase"]["decode"], {
            "hits": 8, "misses": 8, "logical_miss_bytes": 32, "selected_groups": 2})
        for name in ("hits", "misses", "logical_miss_bytes"):
            self.assertEqual(result[name], sum(phase[name] for phase in result["by_phase"].values()))

    def test_actual_small_io_policies_preserve_hashes_and_shared_read_budget(self):
        records = [{"layer": 3, "step": 1, "column": 0, "ids": list(range(1, 9))},
                   {"layer": 3, "step": 2, "column": 0, "ids": list(range(9, 17))}]
        result = replay_io(self.source, records, self.metadata, admission=lambda n: True,
                           capacity_bytes=24, max_read_bytes=256, max_demands=12)
        self.assertTrue(result["same_original_bytes_across_policies"])
        self.assertEqual(result["shared_logical_read_bytes"], 4 * 12 * 4)
        for policy in result["results"]:
            self.assertEqual(policy["cache_after_close"]["reserved_bytes"], 0)
            self.assertLessEqual(policy["cache_before_close"]["peak_reserved_bytes"], 24)
        with self.assertRaises(ValueError):
            replay_io(self.source, records, self.metadata, admission=lambda n: True,
                      capacity_bytes=24, max_read_bytes=40, max_demands=12)

    def test_consecutive_actual_routes_reuse_buffers_and_defer_shrink_until_release(self):
        records = [{"layer": 40, "step": 1, "column": 0, "ids": list(range(1, 9))},
                   {"layer": 40, "step": 2, "column": 0, "ids": [1, 2, 5, 3, 4, 6, 9, 10]}]
        result = replay_reuse(self.source, records, self.metadata, admission=lambda n: True,
                              capacity_bytes=48, max_read_bytes=64)
        self.assertEqual(result["same_resident_buffers_reused"], 6)
        self.assertEqual(result["logical_bytes_read"], 40)
        self.assertEqual(result["tokens"][1]["cache_while_pinned"]["hits"], 6)
        shrink = result["tokens"][1]["shrink_test"]
        self.assertFalse(shrink["immediately_applied"])
        self.assertEqual(shrink["while_selected_leases_pinned"]["reserved_bytes"], 32)
        self.assertEqual(result["cache_after_close"]["reserved_bytes"], 0)
        for token in result["tokens"]:
            self.assertEqual([row["expert_id"] for row in token["experts"]], token["original_ordered_ids"])


if __name__ == "__main__":
    unittest.main()
