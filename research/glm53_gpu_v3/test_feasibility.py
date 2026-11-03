"""Portable accounting and counterexample tests; no model or GPU execution."""
import math
import unittest

from research.glm53_gpu_v3.feasibility import (build_report, closed_source_bounds, conditional_scenarios, critical_path_reduction,
                                              io_coverage_bound, notes_markdown, required_coverage, summarize_operations)


def op(step, name, seconds, *, operation="MUL_MAT", logical=0, status="ok", cpu=1., os_bytes=0):
    return {"step": step, "name": name, "op": operation, "seconds": seconds,
            "cpu_seconds": cpu, "os_read_bytes": os_bytes, "pageins": 0,
            "weight_pages_before": {"status": status, "logical_selected_weight_bytes": logical,
                                    "inspection_seconds": .125}}


class FeasibilityTests(unittest.TestCase):
    @staticmethod
    def closed_proof():
        return {"all_steps_closed": True, "decode_steps_closed": True, "expected_dense_matrices": 1,
                "expected_expert_banks": 1, "decode_distinct_known_bytes_range": [100, 100],
                "per_step": [{"step": step, "all_expected_fixed_matrices_and_expert_banks_resolved": True,
                    "unknown_sources": [], "missing_dense_names": [], "missing_expert_banks": [],
                    "incomplete_dense_payload_names": [], "unexpected_weight_sources": [],
                    "distinct_dense_names": 1, "distinct_expert_bank_names": 1,
                    "distinct_dense_operand_bytes": 80, "distinct_selected_expert_operand_bytes": 20,
                    "dense_expert_overlap_bytes": 0, "distinct_known_matmul_operand_bytes": 100}
                    for step in (0, 1)]}

    def test_interface_bound_accounts_for_retained_coverage_and_nonbinding_case(self):
        result = io_coverage_bound(12_000_000_000, 4_000_000_000, 1_000_000_000)
        self.assertEqual(result["minimum_interface_bytes_per_token"], 8_000_000_000)
        self.assertEqual(result["minimum_storage_service_seconds_per_token"], 8)
        self.assertEqual(result["maximum_io_only_tokens_per_second"], .125)
        covered = io_coverage_bound(10, 20, 1)
        self.assertEqual(covered["minimum_interface_bytes_per_token"], 0)
        self.assertIsNone(covered["maximum_io_only_tokens_per_second"])
        self.assertTrue(covered["bound_nonbinding_when_transfer_zero"])

    def test_target_coverage_uses_necessary_whole_byte_ceiling(self):
        self.assertEqual(required_coverage(100, 9, 2)["minimum_retained_original_byte_coverage"], 96)
        self.assertEqual(required_coverage(100, 200, 1)["minimum_retained_original_byte_coverage"], 0)
        self.assertAlmostEqual(critical_path_reduction(1.2), 1/6)
        self.assertEqual(critical_path_reduction(1), 0)

    def test_invalid_physical_inputs_rejected(self):
        for args in [(-1, 0, 1), (1, -1, 1), (True, 0, 1), (1.5, 0, 1), (1, 0, 0),
                     (1, 0, math.inf), (1, 0, math.nan)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                io_coverage_bound(*args)
        for ratio in (0, .9, math.nan):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                critical_path_reduction(ratio)

    def test_serialized_partition_excludes_prefill_and_does_not_subtract_cpu(self):
        rows = [op(0, "huge-prefill", 100, logical=1000)]
        for step in (1, 2):
            rows += [op(step, "dense", 4, logical=80, cpu=30),
                     op(step, "expert", 1, logical=20, operation="MUL_MAT_ID", cpu=8),
                     op(step, "norm", 1, operation="RMS_NORM", status="not_applicable", cpu=3)]
        result = summarize_operations(rows)
        self.assertEqual(result["decode_steps"], 2)
        self.assertEqual(result["total"]["mean_per_decode_step"]["op_wall_seconds"], 6)
        self.assertEqual(result["total"]["mean_per_decode_step"]["cpu_seconds"], 41)
        self.assertAlmostEqual(sum(g["op_wall_fraction_of_serialized_decode"] for g in result["categories"].values()), 1)
        self.assertEqual(result["categories"]["dense_weight_matmul"]["mean_per_decode_step"]["logical_weight_bytes_known_sum"], 80)
        self.assertIsNone(result["pure_io_wait_seconds_per_token"])

    def test_duplicate_logical_accesses_never_become_unique_or_physical(self):
        # Both rows can access the same bytes; the source profile does not say.
        report = build_report(summarize_operations([op(1, "same-weight", 1, logical=10),
                                                  op(1, "same-weight", 1, logical=10)]))
        self.assertEqual(report["profile"]["total"]["logical_weight_bytes_known_sum"], 20)
        self.assertIsNone(report["profile"]["unique_required_weight_bytes_per_token"])
        self.assertIsNone(report["profile"]["physical_ssd_bytes_per_token"])
        self.assertFalse(report["observed_physical_feasibility"]["unique_required_bytes_verified_by_this_helper"])
        self.assertEqual(report["logical_volume_sensitivity_only"]["assumed_unique_bytes_per_token"], 20)
        self.assertEqual(report["observed_physical_feasibility"]["unconditional_minimum_ssd_bytes_from_these_inputs"], 0)

    def test_missing_counters_and_failed_residency_remain_unknown(self):
        rows = [op(1, "failed", 1, logical=10, status="error", cpu=None), op(1, "good", 1, logical=20)]
        report = summarize_operations(rows)
        self.assertIsNone(report["total"]["cpu_seconds"])
        self.assertEqual(report["total"]["cpu_seconds_known_sum"], 1)
        self.assertEqual(report["unknown_weight_observations"], 1)
        self.assertEqual(report["total"]["logical_weight_bytes_known_sum"], 20)
        self.assertIsNone(report["file_cache_hit_bytes_per_token"])

    def test_bad_or_empty_metadata_cannot_create_a_zero_cost_proof(self):
        for rows in ([], [op(0, "prefill", 1)], [op(True, "invalid-step", 1)], [op(1, "negative-time", -1)],
                     [op(1, "negative-byte-counter", 1, os_bytes=-1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_operations(rows)

    def test_generated_notes_and_scenarios_keep_assumptions_explicit(self):
        result = build_report(summarize_operations([op(1, "dense", 1, logical=100)]), bandwidth_bytes_per_second=2e9)
        notes = notes_markdown(result)
        self.assertIn("interface à 2 Go/s", notes)
        self.assertIn("hypothèse doit être vérifiée", notes)
        self.assertIn("**GPU serial32**", notes)
        scenarios = conditional_scenarios(100, budgets=(0, 50, 100))
        self.assertEqual([x["minimum_interface_bytes_per_token"] for x in scenarios["cache_scenarios"]], [100, 50, 0])
        self.assertIn("ALL cache tiers", scenarios["retention_scope"])

    def test_independent_closed_source_union_supersedes_logical_volume_only_for_its_own_scope(self):
        proof = self.closed_proof()
        report = build_report(summarize_operations([op(1, "logical", 1, logical=20)]), source_union=proof)
        self.assertEqual(report["logical_volume_sensitivity_only"]["assumed_unique_bytes_per_token"], 20)
        closed = report["closed_source_operand_proof"]
        self.assertEqual(closed["conditional_storage_model"]["assumed_unique_bytes_per_token"], 100)
        self.assertTrue(report["observed_physical_feasibility"]["distinct_declared_operands_verified_by_supplied_closed_proof"])
        self.assertFalse(report["observed_physical_feasibility"]["physical_ssd_or_nand_bytes_measured_by_this_helper"])
        self.assertIn("capture complète distincte", notes_markdown(report))

    def test_closed_label_cannot_hide_unknown_or_inconsistent_source_accounting(self):
        changes = [{"unknown_sources": ["unresolved"]}, {"distinct_known_matmul_operand_bytes": 99},
                   {"distinct_dense_names": 2}, {"dense_expert_overlap_bytes": 30}]
        for change in changes:
            with self.subTest(change=change):
                proof = self.closed_proof()
                proof["per_step"][1].update(change)
                with self.assertRaises(ValueError):
                    closed_source_bounds(proof)

    def test_incomplete_source_proof_does_not_qualify_even_with_positive_known_ranges(self):
        proof = self.closed_proof()
        proof["decode_steps_closed"] = False
        with self.assertRaises(ValueError):
            closed_source_bounds(proof)


if __name__ == "__main__":
    unittest.main()
