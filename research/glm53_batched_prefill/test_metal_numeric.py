"""Portable comparison contract checks; no library, GPU or model is loaded."""
import unittest
import numpy as np
from research.glm53_batched_prefill.metal_probe import (
    numerical_comparison, q8_buffer_bytes, representative_token_counts,
)


class NumericalComparisonTests(unittest.TestCase):
    def test_exact(self):
        values = np.array([0, 1, -3], np.float32)
        result = numerical_comparison(values, values.copy())
        self.assertTrue(result['bit_identical'])
        self.assertTrue(result['metal_tolerance']['passed'])

    def test_one_bit_difference_is_reported_separately(self):
        expected = np.array([1], np.float32)
        actual = np.nextafter(expected, np.float32(2))
        result = numerical_comparison(actual, expected)
        self.assertFalse(result['bit_identical'])
        self.assertEqual(result['bit_mismatches'], 1)
        self.assertTrue(result['metal_tolerance']['passed'])

    def test_signed_zero_preserves_bit_contract(self):
        result = numerical_comparison(np.array([-0.0], np.float32), np.zeros(1, np.float32))
        self.assertFalse(result['bit_identical'])
        self.assertTrue(result['metal_tolerance']['passed'])

    def test_outside_existing_tolerance(self):
        result = numerical_comparison(np.array([1.01], np.float32), np.array([1], np.float32))
        self.assertFalse(result['bit_identical'])
        self.assertFalse(result['metal_tolerance']['passed'])
        self.assertEqual(result['metal_tolerance']['atol'], 1e-3)
        self.assertEqual(result['metal_tolerance']['rtol'], 1e-4)

    def test_nonfinite_fails_both_even_with_same_bits(self):
        for value in (np.nan, np.inf, -np.inf):
            values = np.array([value], np.float32)
            result = numerical_comparison(values, values.copy())
            self.assertFalse(result['finite'])
            self.assertFalse(result['bit_identical'])
            self.assertFalse(result['metal_tolerance']['passed'])

    def test_rejects_incompatible_inputs(self):
        for actual, expected in [
            (np.ones(1, np.float32), np.ones(2, np.float32)),
            (np.ones(1, np.float64), np.ones(1, np.float32)),
            (np.empty(0, np.float32), np.empty(0, np.float32)),
        ]:
            with self.assertRaises(ValueError):
                numerical_comparison(actual, expected)


class ProjectionPlanningTests(unittest.TestCase):
    def test_default_stops_at_32(self):
        self.assertEqual(representative_token_counts(), (8, 16, 32))

    def test_larger_batches_require_explicit_maximum(self):
        self.assertEqual(representative_token_counts(64), (8, 16, 32, 64))
        self.assertEqual(representative_token_counts(128), (8, 16, 32, 64, 128))
        for value in (8, 16, 256, True, '128'):
            with self.assertRaises(ValueError):
                representative_token_counts(value)

    def test_representative_128_exact_buffer_inventory(self):
        sizes = q8_buffer_bytes(1536, 4096, 128)
        self.assertEqual(sizes, {
            'packed':6684672, 'x':2097152, 'shape':12,
            'qs':524288, 'scales':32768, 'output':786432,
        })
        self.assertEqual(sum(sizes.values()), 10125324)
        self.assertEqual(2 * (sum(sizes.values()) - sizes['shape']), 20250624)
        self.assertLess(sum(sizes.values()), 64 << 20)
        self.assertLess(2 * (sum(sizes.values()) - sizes['shape']), 64 << 20)

    def test_invalid_buffer_dimensions(self):
        for dimensions in ((0,4096,128), (1536,4095,128), (1536,4096,0)):
            with self.assertRaises(ValueError):
                q8_buffer_bytes(*dimensions)


if __name__ == '__main__':
    unittest.main()
