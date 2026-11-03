"""Offline range-accounting counterexamples; no model/GPU/large file reads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.glm53_gpu_v3.source_union import (merge_intervals, packed_layout, prove_rows,
    read_closed_capture, union_bytes, validate_inventory, validate_source)


def fixture():
    tensors = []
    offset = 0
    for name, shape, codec in [("blk.0.attn_q.weight", [4, 2], "F32"), *[
        (f"blk.3.ffn_{projection}_exps.weight", [256, 1, 288], "Q2_K")
        for projection in ("gate", "up", "down")]]:
        ne, nb, payload = packed_layout(shape, codec)
        tensors.append({"file": "payload.gguf", "name": name, "dims": shape, "type": codec,
                        "offset": offset, "bytes_with_alignment": (payload+31)//32*32})
        offset += (payload+31)//32*32
    inventory = {"revision": "fixture", "files": [{"file": "payload.gguf", "data_offset": 64,
                  "file_bytes": 64+offset}], "tensors": tensors}
    joined = validate_inventory(inventory)
    plan = []
    for step in (0, 1):
        for name, base in joined.items():
            plan.append({"step": step, "op": "MUL_MAT_ID" if "_exps." in name else "MUL_MAT",
                         "source_weight": {"name": name, "base_name": name,
                             "type": 10 if "_exps." in name else 0,
                             "bytes": base["payload_bytes"], "alias_depth": 0, "alias_resolved": True,
                             "dimensions": base["dimensions"], "strides": base["strides"], "offset_from_base": 0}})
    routes = {(0, 3): [list(range(8)), list(range(4, 12))], (1, 3): [list(range(8))]}
    return inventory, plan, routes


class SourceUnionTests(unittest.TestCase):
    def test_file_qualified_union_deduplicates_overlaps_and_adjacency(self):
        spans = [("a", 0, 10), ("a", 5, 15), ("a", 15, 20), ("b", 0, 10)]
        self.assertEqual(merge_intervals(spans), {"a": [[0, 20]], "b": [[0, 10]]})
        self.assertEqual(union_bytes(spans), 30)

    def test_exact_sources_and_current_route_union_close_identity_gap(self):
        inventory, plan, routes = fixture()
        proof = prove_rows(inventory, plan, routes, captured_steps=2)
        self.assertTrue(proof["all_steps_closed"])
        prefill, decode = proof["per_step"]
        self.assertEqual(prefill["distinct_selected_expert_operand_bytes"], 12*84*3)
        self.assertEqual(decode["distinct_selected_expert_operand_bytes"], 8*84*3)
        self.assertEqual(decode["distinct_dense_operand_bytes"], 32)
        self.assertEqual(proof["decode_distinct_known_bytes_range"], [2048, 2048])
        self.assertEqual(decode["dense_expert_overlap_bytes"], 0)
        self.assertFalse(proof["physical_ssd_bytes_measured"])
        self.assertFalse(proof["model_payload_read_by_tool"])

    def test_reset_request_prefills_do_not_inflate_decode_unique_volume(self):
        inventory, plan, routes = fixture()
        second = deepcopy(plan)
        for row in second:
            row["step"] += 2
        routes.update({(2, 3): routes[(0, 3)], (3, 3): routes[(1, 3)]})
        proof = prove_rows(inventory, plan+second, routes, captured_steps=4, prefill_steps=(0, 2))
        self.assertTrue(proof["all_steps_closed"])
        self.assertEqual([row["phase"] for row in proof["per_step"]], ["prefill", "decode", "prefill", "decode"])
        self.assertEqual(proof["decode_distinct_known_bytes_range"], [2048, 2048])

    def test_multi_column_unknown_request_boundary_does_not_qualify_as_decode(self):
        inventory, plan, routes = fixture()
        routes[(1, 3)] = routes[(0, 3)]
        proof = prove_rows(inventory, plan, routes, captured_steps=2)
        self.assertFalse(proof["decode_steps_closed"])
        self.assertTrue(proof["per_step"][1]["unknown_sources"])

    def test_repeated_source_operations_do_not_inflate_unique_bytes(self):
        inventory, plan, routes = fixture()
        plan.append(deepcopy(plan[4]))
        proof = prove_rows(inventory, plan, routes, captured_steps=2)
        decode = proof["per_step"][1]
        self.assertTrue(proof["all_steps_closed"])
        self.assertEqual(decode["logical_dense_operand_bytes"], 64)
        self.assertEqual(decode["distinct_dense_operand_bytes"], 32)
        self.assertEqual(decode["repeated_known_operand_bytes"], 32)

    def test_contiguous_slices_must_jointly_cover_expected_base(self):
        inventory, plan, routes = fixture()
        full = plan.pop(4)
        for offset in (0, 16):
            row = deepcopy(full)
            row["source_weight"].update(name=f"slice-{offset}", alias_depth=1, bytes=16,
                dimensions=[4, 1, 1, 1], strides=[4, 16, 16, 16], offset_from_base=offset)
            plan.append(row)
        self.assertTrue(prove_rows(inventory, plan, routes, captured_steps=2)["all_steps_closed"])
        plan.pop()
        proof = prove_rows(inventory, plan, routes, captured_steps=2)
        self.assertFalse(proof["all_steps_closed"])
        self.assertEqual(proof["per_step"][1]["incomplete_dense_payload_names"], [full["source_weight"]["base_name"]])

    def test_unresolved_gapped_or_out_of_bounds_views_remain_unknown(self):
        changes = [{"alias_resolved": False}, {"offset_from_base": 16},
                   {"type": 8}, {"bytes": 31},
                   {"alias_depth": 1, "name": "gapped", "dimensions": [2, 2, 1, 1],
                    "strides": [4, 16, 32, 32], "bytes": 24}]
        for change in changes:
            with self.subTest(change=change):
                inventory, plan, routes = fixture()
                plan[4]["source_weight"].update(change)
                proof = prove_rows(inventory, plan, routes, captured_steps=2)
                self.assertFalse(proof["all_steps_closed"])
                self.assertTrue(proof["per_step"][1]["unknown_sources"])

    def test_nonquantized_contiguous_permutation_has_exact_full_byte_coverage(self):
        inventory, plan, _ = fixture()
        base = validate_inventory(inventory)[plan[0]["source_weight"]["base_name"]]
        permuted = deepcopy(plan[0]["source_weight"])
        permuted.update(name="permuted", alias_depth=1, dimensions=[2, 4, 1, 1], strides=[16, 4, 32, 32])
        self.assertEqual(validate_source(permuted, base), (0, 32))

    def test_unknown_operand_is_only_excluded_with_explicit_compute_buffer(self):
        for usage, closed in ((None, False), (1, False), (0, False), (2, True)):
            with self.subTest(usage=usage):
                inventory, plan, routes = fixture()
                extra = {"step": 1, "op": "MUL_MAT", "source_weight": {"name": "activation",
                         "base_name": "activation", **({"buffer_usage": usage} if usage is not None else {})}}
                proof = prove_rows(inventory, [*plan, extra], routes, captured_steps=2)
                self.assertEqual(proof["all_steps_closed"], closed)

    def test_missing_or_invalid_actual_routes_cannot_use_predicted_experts(self):
        for bad in (None, [[0]*8], [[0, 1, 2, 3, 4, 5, 6, 288]]):
            with self.subTest(route=bad):
                inventory, plan, routes = fixture()
                if bad is None:
                    del routes[(1, 3)]
                else:
                    routes[(1, 3)] = bad
                proof = prove_rows(inventory, plan, routes, captured_steps=2)
                self.assertFalse(proof["all_steps_closed"])
                self.assertEqual(proof["per_step"][1]["distinct_selected_expert_operand_bytes"], 0)

    def test_header_padding_excluded_and_ambiguous_tensor_overlap_rejected(self):
        inventory, _, _ = fixture()
        inventory["tensors"][0].update(dims=[1, 1], bytes_with_alignment=32)
        joined = validate_inventory(inventory)
        self.assertEqual(joined["blk.0.attn_q.weight"]["payload_bytes"], 4)
        inventory["tensors"][1]["offset"] = 0
        with self.assertRaises(ValueError):
            validate_inventory(inventory)

    def test_incomplete_capture_is_rejected_before_any_route_payload_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"complete.json").write_text(json.dumps({"completed": False, "captured_steps": 1}))
            (root/"tokens.jsonl").write_text('{"step":0}\n')
            with patch("research.glm53_gpu_v3.source_union.tensor_view") as view:
                with self.assertRaises(ValueError):
                    read_closed_capture(root)
                view.assert_not_called()


if __name__ == "__main__":
    unittest.main()
