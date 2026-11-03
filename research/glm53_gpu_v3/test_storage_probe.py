"""Tiny temporary-file and fake-counter checks; no model or hardware reads."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import hashlib
import json
import unittest

from research.glm53_gpu_v3.storage_probe import (
    GIB, MIB, controller_ratio, run_probe, validate_ranges)


def sample_sequence(final_swap=0, final_pressure=1):
    calls = [0]
    def sample():
        index = calls[0]
        calls[0] += 1
        process = {'available': True, 'identity': {'pid': 1234, 'start_abstime': 100},
            'monotonic_start': 2 * index, 'monotonic_end': 2 * index + .1,
            'rusage': {key: 10 for key in ('user_ticks', 'system_ticks', 'pageins', 'os_storage_read_bytes',
                'os_storage_write_bytes', 'resident_bytes', 'physical_footprint_bytes', 'wired_bytes')},
            'task_info': {'available': False}, 'cpu_seconds': 1.}
        return {'available': True, 'process': process,
                'system': {'available': True, 'pressure_level': 1 if index == 0 else final_pressure,
                           'swap_used_bytes': 0 if index == 0 else final_swap}}
    return sample


def block_snapshot(read_bytes):
    return {'available': True, 'scope': 'global fixture controller',
            'devices': [{'registry_id': 1, 'statistics': {'Bytes (Read)': read_bytes}}]}


class StorageProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic.gguf'
        self.original = bytes(range(256))
        self.source.write_bytes(self.original)
        self.counter = Mock(side_effect=[block_snapshot(100), block_snapshot(200)])

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_selected_ranges_repetitions_and_streamed_hash(self):
        plan = [{'offset': 16, 'length': 24, 'repetitions': 2, 'label': 'dense-fixture'},
                {'offset': 100, 'length': 12}]
        output = self.root / 'normal.json'
        result = run_probe(self.source, plan, 'normal', output, scratch_bytes=8,
                           sample_fn=sample_sequence(), block_fn=self.counter,
                           fd_adviser=Mock())
        self.assertEqual(result['status'], 'complete')
        expected = self.original[16:40] * 2 + self.original[100:112]
        self.assertEqual(result['completed_logical_bytes'], len(expected))
        self.assertEqual(result['sha256_completed_ordered_stream'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertTrue(result['selected_file_stats_unchanged'])
        self.assertEqual(result['scratch_payload_bytes'], 8)
        reads = [json.loads(line) for line in output.with_suffix('.reads.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['bytes'] for row in reads), len(expected))
        self.assertTrue(all(row['bytes'] <= 8 for row in reads))
        self.assertIsNone(result['physical_nand_read_bytes'])

    def test_nocache_is_only_an_explicit_descriptor_request(self):
        adviser = Mock()
        result = run_probe(self.source, [{'offset': 0, 'length': 12}], 'nocache', self.root / 'nocache.json',
                           sample_fn=sample_sequence(), block_fn=self.counter, fd_adviser=adviser)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(adviser.call_args.args[1], 'nocache')
        self.assertTrue(result['fd_nocache_requested'])
        self.assertFalse(result['physically_cold_proven'])
        self.assertIsNone(result['file_cache_hit_bytes'])

    def test_hard_budgets_invalid_ranges_and_existing_evidence_are_refused(self):
        for plan in ([{'offset': 0, 'length': GIB + 1}],
                     [{'offset': 0, 'length': GIB, 'repetitions': 2}],
                     [{'offset': -1, 'length': 1}], [{'offset': True, 'length': 1}]):
            with self.assertRaises(ValueError):
                validate_ranges(plan, 4 * GIB)
        output = self.root / 'existing.json'
        output.write_text('preserve')
        with self.assertRaises(ValueError):
            run_probe(self.source, [{'offset': 0, 'length': 1}], 'normal', output)
        self.assertEqual(output.read_text(), 'preserve')
        with self.assertRaises(ValueError):
            run_probe(self.source, [{'offset': 0, 'length': 1}], 'normal', self.root / 'bad.json', scratch_bytes=8 * MIB + 1)

    def test_launch_pressure_and_final_swap_growth_stop_and_persist(self):
        initial = sample_sequence()()
        initial['system']['pressure_level'] = 2
        adviser = Mock()
        launch = run_probe(self.source, [{'offset': 0, 'length': 12}], 'normal', self.root / 'launch.json',
                           sample_fn=lambda: deepcopy(initial), block_fn=self.counter, fd_adviser=adviser)
        self.assertEqual(launch['status'], 'stopped_by_guard')
        self.assertEqual(launch['completed_logical_bytes'], 0)
        adviser.assert_not_called()
        final = run_probe(self.source, [{'offset': 0, 'length': 12}], 'normal', self.root / 'swap.json',
                          sample_fn=sample_sequence(final_swap=33 * MIB),
                          block_fn=Mock(side_effect=[block_snapshot(0), block_snapshot(0)]), fd_adviser=Mock())
        self.assertEqual(final['status'], 'stopped_by_guard')
        self.assertEqual(final['peak_global_swap_growth_bytes'], 33 * MIB)
        self.assertTrue((self.root / 'swap.json').exists())

    def test_advice_failure_and_source_mutation_are_saved_as_failures(self):
        result = run_probe(self.source, [{'offset': 0, 'length': 12}], 'nocache', self.root / 'advice.json',
                           sample_fn=sample_sequence(), block_fn=self.counter,
                           fd_adviser=Mock(side_effect=OSError('fixture advice rejected')))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['completed_logical_bytes'], 0)
        self.assertIn('advice rejected', result['error'])
        def mutate(fd, mode):
            self.source.write_bytes(b'changed')
        result = run_probe(self.source, [{'offset': 0, 'length': 12}], 'normal', self.root / 'changed.json',
                           sample_fn=sample_sequence(), block_fn=Mock(side_effect=[block_snapshot(0), block_snapshot(0)]),
                           fd_adviser=mutate)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['selected_file_stats_unchanged'])

    def test_critical_pressure_midrange_stops_before_the_next_read(self):
        ticks = [0.]
        def clock():
            ticks[0] += .1
            return ticks[0]
        result = run_probe(self.source, [{'offset': 0, 'length': 16}], 'normal', self.root / 'midrange.json',
                           scratch_bytes=4, clock=clock, sample_fn=sample_sequence(final_pressure=4),
                           block_fn=self.counter, fd_adviser=Mock())
        self.assertEqual(result['status'], 'stopped_by_guard')
        self.assertEqual(result['completed_logical_bytes'], 4)
        self.assertEqual(result['read_calls'], 1)
        self.assertIn('critical', result['guard_violation'])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_controller_ratio_can_exceed_one_and_missing_or_reset_is_unknown(self):
        observed = controller_ratio(block_snapshot(100), block_snapshot(400), 100)
        self.assertEqual(observed['controller_to_logical_ratio'], 3)
        self.assertIn('not a cache-hit fraction', observed['scope'])
        self.assertIsNone(controller_ratio(block_snapshot(100), block_snapshot(90), 100)['controller_to_logical_ratio'])
        self.assertIsNone(controller_ratio({'available': False}, block_snapshot(90), 100)['controller_to_logical_ratio'])
        missing = block_snapshot(100)
        missing['devices'][0]['statistics'] = {}
        self.assertIsNone(controller_ratio(missing, missing, 100)['controller_to_logical_ratio'])


if __name__ == '__main__':
    unittest.main()
