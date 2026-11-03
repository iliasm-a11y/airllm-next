"""Portable policy counterexamples, no model, GPU or expert payload allocation."""
from pathlib import Path
import unittest
from unittest.mock import patch

from research.glm53_gpu_v3.moe_reuse import (adjacent_decode_reuse, expert_sizes,
    simulate_global_lru, simulate_layer_last_k)


def records():
    return [{"step": step, "layer": layer, "ids": list(range(4*(step-1), 4*(step-1)+8)), "phase": "decode"}
            for step in (1, 2, 3) for layer in (3, 4)]


class MoEReuseTests(unittest.TestCase):
    def test_cold_and_past_route_warm_start_have_distinct_denominators(self):
        sizes = {3: 10, 4: 10}
        cold = simulate_layer_last_k(records(), sizes)
        warm = simulate_layer_last_k(records(), sizes, initial={3: list(range(8)), 4: list(range(8))})
        self.assertEqual((cold["hits"], cold["misses"], cold["logical_miss_bytes"]), (16, 32, 320))
        self.assertEqual((warm["hits"], warm["misses"], warm["logical_miss_bytes"]), (32, 16, 160))
        self.assertEqual(cold["persistent_raw_payload_capacity_bytes"], 160)
        self.assertEqual(cold["steps"], warm["steps"])
        self.assertFalse(cold["physical_storage_reads_measured"])

    def test_original_first_k_ranks_do_not_choose_next_token_experts(self):
        rows = [{"step": 1, "layer": 3, "ids": list(range(2, 10))},
                {"step": 2, "layer": 3, "ids": list(range(8))}]
        result = simulate_layer_last_k(rows, {3: 10}, keep=2, initial={3: list(range(8))})
        # Initial cache holds0,1, so the first demand has zero hits even though
        # six other previous-route IDs occur in it. No future-aware replacement.
        self.assertEqual([r["hits"] for r in result["per_step"]], [0, 2])
        self.assertEqual(result["persistent_raw_payload_capacity_bytes"], 20)

    def test_global_lru_scan_thrashing_differs_from_per_layer_retention(self):
        rows = [{"step": step, "layer": layer, "ids": list(range(8)), "phase": "decode"}
                for step in (1, 2) for layer in (3, 4, 5)]
        lru = simulate_global_lru(rows, {3: 10, 4: 10, 5: 10}, capacity_bytes=160)
        retained = simulate_layer_last_k(rows, {3: 10, 4: 10, 5: 10})
        self.assertEqual(lru["decode"]["hits"], 0)
        self.assertEqual(lru["decode"]["misses"], 48)
        self.assertEqual(retained["hits"], 24)
        self.assertLessEqual(lru["peak_packed_payload_bytes"], 160)
        self.assertEqual(retained["persistent_raw_payload_capacity_bytes"], 240)

    def test_lru_group_admission_preserves_hits_in_current_selected_set(self):
        rows = [{"step": 0, "layer": 3, "ids": list(range(8)), "phase": "prefill"},
                {"step": 1, "layer": 3, "ids": list(range(4, 12)), "phase": "decode"}]
        result = simulate_global_lru(rows, {3: 10}, capacity_bytes=80)
        self.assertEqual(result["decode"]["hits"], 4)
        self.assertEqual(result["decode"]["logical_miss_bytes"], 40)
        self.assertEqual(result["peak_packed_payload_bytes"], 80)

    def test_adjacent_union_is_previous_payload_plus_actual_new_ids(self):
        routes = {1: {3: list(range(8)), 4: list(range(8))},
                  2: {3: list(range(4, 12)), 4: list(range(4, 12))}}
        result = adjacent_decode_reuse(routes, {3: 10, 4: 10})
        self.assertEqual(result["hits"], 8)
        self.assertEqual(result["misses"], 8)
        self.assertEqual(result["mean_logical_miss_bytes_per_step"], 80)
        self.assertEqual(result["two_token_expert_union_bytes"], {"minimum": 240, "mean": 240, "maximum": 240})
        self.assertEqual(result["per_layer"][0]["mean_shared_experts_out_of_eight"], 4)

    def test_invalid_sizes_rank_duplicates_or_inadequate_group_budget_rejected(self):
        for sizes in ({3: 0}, {3: -1}, {3: True}):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                simulate_layer_last_k(records()[:1], sizes)
        with self.assertRaises(ValueError):
            simulate_global_lru(records()[:1], {3: 10}, capacity_bytes=79)
        with self.assertRaises(ValueError):
            simulate_layer_last_k(records(), {3: 10, 4: 10}, keep=9)
        with self.assertRaises(ValueError):
            simulate_layer_last_k([{"step": 1, "layer": 3, "ids": [0]*8}], {3: 10})

    def test_inventory_only_size_check_never_opens_declared_GGUF_payload(self):
        tensors = [{"file": "NEVER-OPEN.gguf", "name": "output.weight", "dims": [4, 2],
                    "type": "F32", "offset": 0, "bytes_with_alignment": 32}]
        for index, projection in enumerate(("gate", "up", "down")):
            tensors.append({"file": "NEVER-OPEN.gguf", "name": f"blk.3.ffn_{projection}_exps.weight",
                "dims": [256, 1, 288], "type": "Q2_K", "offset": 32+index*24192,
                "bytes_with_alignment": 24192})
        inventory = {"files": [{"file": "NEVER-OPEN.gguf", "data_offset": 64,
                                "file_bytes": 64+32+3*24192}], "tensors": tensors}
        with patch.object(Path, "open", side_effect=AssertionError("payload read forbidden")):
            sizes, dense = expert_sizes(inventory)
        self.assertEqual(sizes, {3: 252})
        self.assertEqual(dense, 32)


if __name__ == "__main__":
    unittest.main()
