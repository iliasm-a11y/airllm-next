"""Tiny offline fixtures only; no native libraries, hardware counters or model reads."""
import json
from pathlib import Path
import tempfile
import unittest

from .summarize_observed import summarize_campaign, summarize_request


class ObservedSummaryTests(unittest.TestCase):
    def fixture(self):
        job = {'index': 1, 'cycle': 0, 'arm': 'B', 'candidate': 'head-stream',
               'gpu_commands_required': True}
        owned = {'available': True, 'pid': 77, 'identity': {'pid': 77, 'start_abstime': 100},
                 'mach_timebase': {'numer': 7, 'denom': 2}}
        report = {'owned_pid': 77, 'status': 'complete', 'samples': [{'owned_process': owned}],
                  'peak_swap_growth_bytes': 0, 'peak_rss_bytes': 123, 'guard_violation': None}
        request = {'usage_before': {'user_ticks': 10, 'system_ticks': 20,
                                    'storage_read_bytes': 100, 'pageins': 1},
                   'usage_after': {'user_ticks': 30, 'system_ticks': 50,
                                   'storage_read_bytes': 250, 'pageins': 4},
                   'block_device_before': {'available': True, 'devices': [
                       {'registry_id': 4, 'statistics': {'Bytes (Read)': 400}}]},
                   'block_device_after': {'available': True, 'devices': [
                       {'registry_id': 4, 'statistics': {'Bytes (Read)': 600}}]},
                   'final': {'stop': True, 'stop_type': 'limit', 'tokens_evaluated': 6},
                   'tokens': [1, 2], 'body': {'n_predict': 2}, 'ttft_seconds': 5,
                   'wall_seconds': 8, 'observed_decode_tokens_per_second': 1/3,
                   'gpu_command_times': {'recorded_commands': 2, 'valid_commands': 2,
                       'sum_command_seconds': 1.2, 'union_command_seconds': 1,
                       'complete_capture_guaranteed': False}}
        return job, report, request

    def test_measured_timebase_and_request_scope(self):
        job, report, request = self.fixture()
        row = summarize_request(job, {'status': 'complete'}, report, request, 0)
        self.assertAlmostEqual(row['request_owned_pid_cpu_seconds'], 175e-9)
        self.assertEqual(row['request_owned_pid_os_read_bytes']['value'], 150)
        self.assertEqual(row['request_global_block_read_bytes']['value'], 200)
        self.assertEqual(row['prefill_tokens_evaluated'], 6)
        self.assertIn('including prefill', row['counter_scope'])
        self.assertIsNone(row['decode_only_counter_delta'])
        self.assertFalse(row['request_gpu_command_times']['complete_capture_guaranteed'])

    def test_missing_timebase_and_reset_are_unknown(self):
        job, report, request = self.fixture()
        report['samples'] = []
        request['usage_after']['user_ticks'] = 1
        request['usage_after']['storage_read_bytes'] = 99
        row = summarize_request(job, {'status': 'complete'}, report, request, 0)
        self.assertIsNone(row['request_owned_pid_cpu_seconds'])
        self.assertIsNone(row['recorded_mach_timebase']['value'])
        self.assertIn('reset', row['request_cpu_ticks']['user_ticks']['reason'])
        self.assertIsNone(row['request_owned_pid_os_read_bytes']['value'])
        self.assertIsNone(row['decode_only_counter_delta'])

    def test_changed_device_or_pid_lifetime_rejects_delta(self):
        job, report, request = self.fixture()
        request['block_device_after']['devices'][0]['registry_id'] = 5
        other = dict(report['samples'][0]['owned_process'],
                     identity={'pid': 77, 'start_abstime': 101})
        report['samples'].append({'owned_process': other})
        row = summarize_request(job, {'status': 'failed'}, report, request, 0)
        self.assertIsNone(row['request_global_block_read_bytes']['value'])
        self.assertIn('device set changed', row['request_global_block_read_bytes']['reason'])
        self.assertIsNone(row['recorded_mach_timebase']['value'])
        self.assertIsNone(row['request_owned_pid_cpu_seconds'])

    def test_closed_failed_campaign_preserves_unlaunched_and_source_hashes(self):
        job, report, request = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, manifest = root/'run.json', root/'campaign.json'
            report['requests'] = [request]
            raw_path.write_text(json.dumps(report))
            job['report'] = str(raw_path)
            unlaunched = dict(job, index=2, report=str(root/'absent.json'))
            data = {'status': 'failed', 'plan': {'jobs': [job, unlaunched]},
                    'stages': [{'status': 'failed', 'report_data': report}]}
            manifest.write_text(json.dumps(data))
            before = manifest.read_bytes()
            result = summarize_campaign(manifest)
            self.assertEqual(manifest.read_bytes(), before)
            self.assertTrue(result['raw_report_sources'][0]['matches_embedded_report'])
            self.assertEqual(result['rows'][1]['stage_status'], 'not_launched')
            self.assertIsNone(result['rows'][1]['request_owned_pid_os_read_bytes']['value'])
            self.assertFalse(result['optimization_qualified'])
            self.assertEqual(len(result['campaign_sha256']), 64)
            data['status'] = 'running'
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'closed'):
                summarize_campaign(manifest)


if __name__ == '__main__':
    unittest.main()
