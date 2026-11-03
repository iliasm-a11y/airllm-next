"""Portable metadata checks; no model, runtime build, or hardware access."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from research.glm53_ultra.structural import (
    analyze_bundle, capture_trajectories, recurrent_state_bytes, simple_draft, trajectory_opportunities,
)


class StructuralTests(unittest.TestCase):
    def test_native_ignores_position_zero_and_current_pattern(self):
        self.assertEqual(simple_draft([7, 8, 9, 10, 11], 7, size_n=1, size_m=1), [])

    def test_native_reversed_search_and_m_minimum(self):
        history = [99, 1, 2, 7, 8, 1, 2, 3, 4, 5, 1]
        self.assertEqual(simple_draft(history, 2, size_n=2, size_m=2), [3, 4])
        self.assertEqual(simple_draft(history, 2, size_n=2, size_m=1), [])

    def test_opportunity_counts_do_not_claim_runtime_acceptance(self):
        result = trajectory_opportunities([99, 1, 2, 3, 4, 5], [1, 2, 3], size_n=1, size_m=1)
        self.assertEqual(result["offer_count"], 2)
        self.assertEqual(result["first_token_agreement_count"], 2)
        self.assertIsNone(result["throughput_estimate"])
        self.assertFalse(result["speculative_fidelity_validated"])

    def test_malformed_tokens_are_rejected(self):
        for invalid in ([True], [-1], [1.0], "123"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                trajectory_opportunities(invalid, [])

    def test_request_boundary_does_not_merge_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "prompt_tokens.json").write_text("[9, 8]")
            (p / "complete.json").write_text('{"completed":true,"captured_steps":2}')
            (p / "tokens.jsonl").write_text(
                json.dumps({"step":0,"request":0,"token":3,"next_input_token":3}) + "\n" +
                json.dumps({"step":1,"request":1,"token":4,"next_input_token":4}) + "\n")
            rows = capture_trajectories(p)
            self.assertEqual([row["generated"] for row in rows], [[3], [4]])
            self.assertEqual([row["prompt"] for row in rows], [[9, 8], [9, 8]])
            (p / "complete.json").write_text('{"completed":false,"captured_steps":2}')
            with self.assertRaises(ValueError):
                capture_trajectories(p)

    def test_glm_dimensions_reproduce_logical_rs_allocation(self):
        metadata = {"glm5-next.block_count":45, "glm5-next.attention.head_count":64,
                    "glm5-next.kda.head_dim":128, "glm5-next.ssm.conv_kernel":4,
                    "glm5-next.attention.head_count_kv":[int(i % 4 == 3) for i in range(45)]}
        result = recurrent_state_bytes(metadata)
        self.assertEqual(result["recurrent_layers"], 34)
        self.assertEqual(result["logical_r_plus_s_bytes"], 152633344)
        with self.assertRaises(ValueError):
            recurrent_state_bytes(metadata, cells=0)

    def test_published_bundle_replays_without_original_absolute_paths(self):
        bundle = Path(__file__).parent / "evidence" / "structural-inputs"
        result = analyze_bundle(bundle)
        self.assertEqual(len(result["captures"]), 4)
        self.assertEqual(len(result["timed_benchmark_continuations"]), 12)
        self.assertTrue(all(not Path(row["capture_directory"]).is_absolute()
                            for row in result["captures"]))
        for capture in result["captures"]:
            configs = capture["requests"][0]["configs"]
            self.assertEqual([(row["offer_count"], row["first_token_agreement_count"])
                              for row in configs], [(0, 0), (0, 0), (1, 0), (2, 1), (0, 0)])

    def test_bundle_refuses_modified_input(self):
        bundle = Path(__file__).parent / "evidence" / "structural-inputs"
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "inputs"
            shutil.copytree(bundle, copied)
            (copied / "native/v4-cpu-control16/prompt_tokens.json").write_text("[1]")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                analyze_bundle(copied)


if __name__ == "__main__":
    unittest.main()
