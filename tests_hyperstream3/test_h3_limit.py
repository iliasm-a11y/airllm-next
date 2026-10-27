"""Stdlib arithmetic checks with invented tiny units labelled synthetic tests."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from estimate_hyperstream3_limit import estimate_limit


class LimitTests(unittest.TestCase):
    def inputs(self, **changes):
        values = dict(logical_bytes_per_token=100., effective_transport_bytes_per_second=50.,
                      measurement_label='synthetic arithmetic test, not a hardware measurement')
        values.update(changes)
        return values

    def test_transport_only_estimate_has_no_compute_or_physical_ssd_claim(self):
        result = estimate_limit(**self.inputs())
        self.assertEqual(result['status'], 'conditional_scenario_not_measured')
        self.assertIs(result['physical_ssd_bandwidth_certified'], False)
        self.assertIs(result['caller_inputs_verified'], False)
        estimates = result['estimates']
        self.assertEqual(estimates['logical_bytes_per_verified_final_token'], 100.)
        self.assertEqual(estimates['transport_seconds_per_block'], 2.)
        self.assertEqual(estimates['conditional_transport_ceiling_tokens_per_second'], .5)
        self.assertIsNone(estimates['ideal_overlap_tokens_per_second'])
        self.assertIsNone(estimates['serial_tokens_per_second'])

    def test_draft_replay_cache_and_verified_emission_are_explicit(self):
        result = estimate_limit(**self.inputs(additional_cache_coverage=.25,
            accepted_tokens_per_block=4., target_pass_equivalents_per_block=2.,
            draft_logical_bytes_per_block=50., draft_seconds_per_block=1.,
            compute_seconds_per_block=3.))
        estimates = result['estimates']
        self.assertEqual(estimates['target_logical_bytes_per_block'], 150.)
        self.assertEqual(estimates['total_logical_bytes_per_block'], 200.)
        self.assertEqual(estimates['logical_bytes_per_verified_final_token'], 50.)
        self.assertEqual(estimates['transport_seconds_per_block'], 4.)
        self.assertEqual(estimates['ideal_overlap_seconds_per_block'], 4.)
        self.assertEqual(estimates['ideal_overlap_tokens_per_second'], 1.)
        self.assertEqual(estimates['serial_seconds_per_block'], 8.)
        self.assertEqual(estimates['serial_tokens_per_second'], .5)

    def test_verification_and_replay_can_remove_the_scenario_gain(self):
        ordinary = estimate_limit(**self.inputs())['estimates']
        speculative = estimate_limit(**self.inputs(accepted_tokens_per_block=2.,
            target_pass_equivalents_per_block=3.))['estimates']
        self.assertGreater(speculative['logical_bytes_per_verified_final_token'],
                           ordinary['logical_bytes_per_verified_final_token'])
        self.assertLess(speculative['conditional_transport_ceiling_tokens_per_second'],
                        ordinary['conditional_transport_ceiling_tokens_per_second'])

    def test_zero_remaining_traffic_gives_no_finite_transport_bound(self):
        estimates = estimate_limit(**self.inputs(additional_cache_coverage=1.,
            compute_seconds_per_block=2.))['estimates']
        self.assertEqual(estimates['transport_seconds_per_block'], 0.)
        self.assertIsNone(estimates['conditional_transport_ceiling_tokens_per_second'])
        self.assertEqual(estimates['ideal_overlap_tokens_per_second'], .5)
        self.assertNotIn('Infinity', json.dumps(estimates, allow_nan=False))

    def test_invalid_or_infinite_inputs_and_missing_provenance_are_rejected(self):
        invalid = [dict(logical_bytes_per_token=-1), dict(logical_bytes_per_token=True),
                   dict(effective_transport_bytes_per_second=0),
                   dict(effective_transport_bytes_per_second=float('inf')),
                   dict(accepted_tokens_per_block=0), dict(target_pass_equivalents_per_block=-1),
                   dict(additional_cache_coverage=1.01), dict(additional_cache_coverage=float('nan')),
                   dict(draft_logical_bytes_per_block=-1), dict(draft_seconds_per_block=-1),
                   dict(compute_seconds_per_block=-1), dict(measurement_label='  '),
                   dict(logical_bytes_per_token=10**1000),
                   dict(logical_bytes_per_token=1e308, target_pass_equivalents_per_block=1e308)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                estimate_limit(**self.inputs(**changes))

    def test_cli_requires_a_declared_rate_and_has_no_bandwidth_default(self):
        base = [sys.executable, '-S', str(ROOT / 'benchmarks/estimate_hyperstream3_limit.py'),
                '--logical-bytes-per-token', '100', '--measurement-label', 'synthetic test']
        missing = subprocess.run(base, capture_output=True, text=True, timeout=10)
        self.assertEqual(missing.returncode, 2)
        self.assertIn('--effective-transport-bytes-per-second', missing.stderr)
        complete = subprocess.run([*base, '--effective-transport-bytes-per-second', '50'],
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertEqual(json.loads(complete.stdout)['status'], 'conditional_scenario_not_measured')


if __name__ == '__main__':
    unittest.main()
