"""Synthetic validation of the comparator, not evidence of model fidelity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_fidelity import aggregate, compare, compare_arrays, CRITERIA, tensor_view


def make_capture(root, *, tokens=(7, 8), forced_inputs=None, router_delta=0.0, route_delta=0):
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"count": len(tokens), "forced": forced_inputs is not None}))
    (root / "prompt_tokens.json").write_text("[1,2]")
    (root / "complete.json").write_text(json.dumps({"completed": True, "captured_steps": len(tokens)}))
    token_rows, rows = [], []
    for step, token in enumerate(tokens):
        token_rows.append({"step": step, "token": token, "vocab": 10,
                           "next_input_token": forced_inputs[step] if forced_inputs else token})
        logits = np.arange(10, dtype="<f4")
        logits[token] = 20
        logits.tofile(root / ("logits-" + str(step) + ".f32"))
        for index, name in enumerate(("ffn_moe_topk-0", "ffn_moe_logits-0", "kda_scan_out-0", "result_output")):
            route = index == 0
            values = np.array([1 + (route_delta if route else router_delta), 2], dtype="<i4" if route else "<f4")
            filename = str(step) + "-" + str(index) + ".bin"
            values.tofile(root / filename)
            rows.append({"step": step, "name": name, "file": filename, "type": 26 if route else 0,
                         "bytes": 8, "shape": [2, 1, 1, 1], "strides": [4, 8, 8, 8]})
    (root / "tokens.jsonl").write_text("".join(json.dumps(r) + "\n" for r in token_rows))
    (root / "tensors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class FidelityComparatorTests(unittest.TestCase):
    def test_byte_strides_compare_logical_values_with_different_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            views = []
            for name, strides, size in (("a", [4, 16, 32, 32], 24), ("b", [4, 8, 16, 16], 16)):
                raw = bytearray(size)
                view = np.ndarray((2, 2, 1, 1), dtype="<f4", buffer=raw, strides=strides)
                view[:, :, 0, 0] = [[1, 3], [2, 4]]
                (root / name).write_bytes(raw)
                views.append(tensor_view(root, {"file": name, "type": 0, "shape": [2, 2, 1, 1], "strides": strides, "bytes": size}))
            self.assertTrue(compare_arrays(*views, CRITERIA["cpu"])["passed"])

    def test_signed_zero_fails_bitexact_but_passes_metal_numeric_tolerance(self):
        a, b = np.array([0.0], dtype="<f4"), np.array([-0.0], dtype="<f4")
        result = compare_arrays(a, b, CRITERIA["cpu"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["bit_mismatch_count"], 1)
        self.assertEqual(result["numeric_mismatch_count"], 0)
        self.assertTrue(compare_arrays(a, b, CRITERIA["metal"])["passed"])

    def test_nonfinite_never_passes(self):
        a = np.array([np.nan, np.inf], dtype="<f4")
        result = compare_arrays(a, a, CRITERIA["cpu"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["finite"])
        self.assertEqual(result["nonfinite_reference"], 2)
        self.assertIsNone(aggregate([{**result, "comparable": True}])["max_abs_error_finite_pairs"])

    def test_criteria_are_fixed_and_integer_routes_exact_in_metal(self):
        self.assertEqual(CRITERIA["metal"], {"bitexact_f32": False, "atol": 1e-3, "rtol": 1e-4})
        a, b = np.array([10000], dtype="<i4"), np.array([10001], dtype="<i4")
        self.assertFalse(compare_arrays(a, b, CRITERIA["metal"], integer=True)["passed"])

    def test_free_token_divergence_prevents_later_numeric_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            make_capture(a)
            make_capture(b, tokens=(6, 8))
            result = compare(a, b, "metal")
            self.assertFalse(result["passed"])
            self.assertEqual(result["first_incomparable_step"], 1)
            self.assertTrue(result["full_logits"][0]["comparable"])
            self.assertFalse(result["full_logits"][1]["comparable"])

    def test_forced_equal_input_keeps_numeric_comparison_after_argmax_divergence(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            make_capture(a)
            make_capture(b, tokens=(6, 8), forced_inputs=(7, 8))
            result = compare(a, b, "metal")
            self.assertFalse(result["passed"])
            self.assertTrue(result["complete_same_input_sequence"])
            self.assertTrue(all(r["comparable"] for r in result["full_logits"]))

    def test_identical_complete_capture_passes_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            make_capture(a)
            make_capture(b)
            result = compare(a, b, "cpu")
            self.assertTrue(result["passed"])
            self.assertEqual(result["coverage"]["routes"]["elements"], 4)

    def test_interrupted_capture_cannot_pass_without_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            make_capture(a)
            make_capture(b)
            (b / "complete.json").unlink()
            with self.assertRaises(ValueError):
                compare(a, b, "cpu")

    def test_metal_allclose_does_not_relax_cpu_and_rejects_large_error(self):
        a, b = np.array([1.0], dtype="<f4"), np.array([1.0005], dtype="<f4")
        self.assertTrue(compare_arrays(a, b, CRITERIA["metal"])["passed"])
        self.assertFalse(compare_arrays(a, b, CRITERIA["cpu"])["passed"])
        b[0] = 1.1
        self.assertFalse(compare_arrays(a, b, CRITERIA["metal"])["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
