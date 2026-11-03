"""Small synthetic arrays only; no native runtime, build, GPU or model."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'airllm-next'
spec = importlib.util.spec_from_file_location('strict_fidelity_fixture_source', REPO / 'research/glm53_m5/compare_fidelity.py')
fidelity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fidelity)
sys.path.insert(0, str(HERE))
import numerics
MODULES = {'fidelity': fidelity, 'batch': types.SimpleNamespace(TOKEN_AXES={'router': 1, 'routes': 1, 'kda': 2})}


class NumericCriteria(unittest.TestCase):
    def test_original_criteria_and_diagnostic_remain_distinct(self):
        a = np.array([1, 2, 3], dtype='<f4')
        b = a + np.float32(0.0001)
        result = numerics.both(a, b, MODULES)
        self.assertTrue(result['metal']['passed'])
        self.assertFalse(result['cpu']['passed'])
        self.assertEqual(fidelity.CRITERIA['metal'], {'bitexact_f32': False, 'atol': 1e-3, 'rtol': 1e-4})

    def test_routes_are_ordered_exact_even_under_metal(self):
        a, b = np.array([1, 2], dtype='<i4'), np.array([2, 1], dtype='<i4')
        result = numerics.both(a, b, MODULES, integer=True)
        self.assertFalse(result['metal']['passed'])
        self.assertEqual(result['metal']['mismatch_count'], 2)

    def test_equal_nonfinite_values_never_pass(self):
        for value in (np.nan, np.inf, -np.inf):
            result = numerics.both(np.array([value], dtype='<f4'), np.array([value], dtype='<f4'), MODULES)
            self.assertFalse(result['metal']['passed'])
            self.assertFalse(result['cpu']['passed'])

    def test_real_strides_and_different_chunks_keep_exact_token_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            def row(file, values, offset):
                data = np.zeros((2, len(values) * 2), dtype='<f4')
                data[0, ::2] = values
                data[1, ::2] = np.array(values) + 20
                path = root / file
                path.write_bytes(data.tobytes())
                return {'file': file, 'type': 0, 'bytes': path.stat().st_size,
                        'shape': [2, len(values), 1, 1], 'strides': [len(values) * 8, 8, 4, 4],
                        'offset': offset, 'input_tokens': len(values)}
            a = [row('a0', [1, 2], 0), row('a1', [3, 4], 2)]
            b = [row('b0', [1, 2, 3, 4], 0)]
            result = numerics.token_series(root, root, a, b, 'ffn_moe_logits-3', MODULES)
            self.assertTrue(result['metal_passed'])
            self.assertTrue(result['cpu_passed'])
            self.assertEqual((result['tokens'], result['elements']), (4, 8))
            b[0] = row('b1', [1, 2, 3.1, 4], 0)
            result = numerics.token_series(root, root, a, b, 'ffn_moe_logits-3', MODULES)
            self.assertFalse(result['metal_passed'])
            self.assertEqual(result['first_metal_token_divergence'], 2)

    def test_gap_and_truncated_series_are_rejected(self):
        # Position checks happen before opening a payload.
        with self.assertRaisesRegex(ValueError, 'positions'):
            numerics.token_series(Path('.'), Path('.'), [{'offset': 0}], [{'offset': 1}], 'ffn_moe_logits-3', MODULES)
        with self.assertRaisesRegex(ValueError, 'ended'):
            numerics.token_series(Path('.'), Path('.'), [], [{'offset': 0}], 'ffn_moe_logits-3', MODULES)


if __name__ == '__main__':
    unittest.main()
