"""Portable source scheduling properties; no matmul, model or native process."""
from pathlib import Path
import importlib.util
import json
import unittest
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import geometry as g
import fallback_eligibility as f


class ColumnEligibility(unittest.TestCase):
    def test_n30_demonstrates_v3_failure_and_v4_changes_only_remainder(self):
        legacy = f.layout(8192, 30)
        self.assertEqual((legacy['dr0'], legacy['dr1']), (16, 15))
        self.assertTrue(all(rows == ((0, 8192, 1),) for rows in legacy['nrc_by_column']))
        current = g.plan(32, 512, 542)
        self.assertEqual([r['tokens'] for r in current], [16, 14])
        self.assertEqual(f.schedule_map(8192, current), (((0, 8192, 2),),) * 30)

    def test_explicit_remainders_and_large_windows(self):
        cases = {9: [8, 1], 11: [8, 3], 13: [8, 5], 14: [14], 15: [8, 7],
                 18: [16, 2], 30: [16, 14], 38: [32, 6], 44: [32, 12],
                 48: [48], 72: [64, 8], 88: [80, 8], 120: [112, 8]}
        for n, expected in cases.items():
            with self.subTest(n=n):
                # No intermediate absolute fence in this bounded interval.
                self.assertEqual([r['tokens'] for r in g.plan(128, 512, 512 + n)], expected)

    def test_observed_large_m_domain_preserves_exact_nrc_map_for_all_window_lengths(self):
        # Actual ordinary Q8 M values from the closed v3 inventory, with
        # 2D/non-NUMA/10-worker assumptions explicit; not model-fidelity proof.
        for m in (1536, 2048, 4096, 8192, 12288, 16384, 154880):
            for n in range(1, 129):
                ref = g.plan(8, 512, 512 + n)
                for ub in g.BATCHES:
                    with self.subTest(m=m, n=n, ub=ub):
                        self.assertEqual(f.schedule_map(m, g.plan(ub, 512, 512 + n)), f.schedule_map(m, ref))

    def test_small_m_thread_rechunk_counterexample_prevents_universal_claim(self):
        self.assertTrue(f.layout(128, 8)['thread_rechunk'])
        self.assertFalse(f.layout(128, 128)['thread_rechunk'])
        self.assertNotEqual(f.schedule_map(128, g.plan(128, 512, 640)),
                            f.schedule_map(128, g.plan(8, 512, 640)))

    def test_all_bounded_lengths_preserve_column_eligibility_in_admitted_large_m_domain(self):
        # Once row chunks are even and the thread-rechunk fallback is absent,
        # nrc depends only on call N and its actual dr1 column chunk lengths.
        choices = {n: f.layout(1536, n)['nrc_by_column'] for n in range(1, 129)}
        for end in range(1, 1801):
            reference = tuple(x for r in g.plan(8, 0, end) for x in choices[r['tokens']])
            for ub in g.BATCHES:
                actual = tuple(x for r in g.plan(ub, 0, end) for x in choices[r['tokens']])
                self.assertEqual(actual, reference, (ub, end))

    def test_source_guard_resource_api_and_version_are_not_renamed(self):
        source = (ROOT / 'research/glm53_batched_prefill/native_prefill.cpp').read_text()
        self.assertIn('rusage_info_v2 u{}', source)
        self.assertIn('RUSAGE_INFO_V2', source)
        self.assertIn('raw > 16 && raw % 16 != 0', source)
        self.assertIn('const int aligned = prefill_aligned_take(raw)', source)
        self.assertIn('{"chunk16_split",aligned_take!=unadjusted_take}', source)
        self.assertIn('{"odd_tail_split",take!=aligned_take}', source)
        self.assertIn('{"fallback_column_chunk",16}', source)
        self.assertNotIn('glm53_gpu_v4', source)
        self.assertNotIn('RUSAGE_INFO_V4', source)


if __name__ == '__main__':
    unittest.main()
