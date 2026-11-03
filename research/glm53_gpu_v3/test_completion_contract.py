"""Portable completion/shutdown metadata regressions; no subprocess or model."""
from copy import deepcopy
import unittest

from .completion_contract import record_server_exit, validate_completion


class CompletionContractTests(unittest.TestCase):
    def request(self, count, requested, stop_type='limit'):
        return {'body': {'n_predict': requested}, 'tokens': list(range(count)),
            'final': {'stop': True, 'stop_type': stop_type, 'tokens_predicted': count,
                      'timings': {'predicted_n': count}, 'truncated': False},
            'events': [{'response': {'tokens': [token]}} for token in range(count)]}

    def test_limit_requires_exact_budget_and_consistent_final_count(self):
        request = self.request(8, 8)
        self.assertTrue(validate_completion(request, 8)['passed'])
        self.assertFalse(validate_completion(self.request(7, 8), 8)['passed'])
        request['final']['tokens_predicted'] = 9
        self.assertFalse(validate_completion(request, 8)['passed'])

    def test_natural_eos27_in_request64_requires_complete_id_transcript(self):
        request = self.request(27, 64, 'eos')
        self.assertTrue(validate_completion(request, 64)['passed'])
        missing = deepcopy(request)
        missing['tokens'].pop()
        missing['events'].pop()
        result = validate_completion(missing, 64)
        self.assertFalse(result['passed'])
        self.assertFalse(result['eos_id_omission_accepted'])

    def test_truncated_word_or_mismatching_event_transcript_rejected(self):
        request = self.request(8, 8)
        request['final']['truncated'] = True
        self.assertFalse(validate_completion(request, 8)['passed'])
        request['final']['truncated'] = False
        request['final']['stop_type'] = 'word'
        self.assertFalse(validate_completion(request, 8)['passed'])
        request['final']['stop_type'] = 'limit'
        request['events'][2]['response']['tokens'] = [999]
        self.assertFalse(validate_completion(request, 8)['passed'])

    def test_only_normal_owned_exit_preserves_completed_status(self):
        for code in (1, -9, -15, None):
            report = {'status': 'complete'}
            self.assertFalse(record_server_exit(report, code))
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['server_shutdown']['normal_exit'])
        report = {'status': 'complete'}
        self.assertTrue(record_server_exit(report, 0))
        self.assertEqual(report['status'], 'complete')
        rejected = {'status': 'failed', 'error': 'earlier guard violation'}
        record_server_exit(rejected, 0)
        self.assertEqual(rejected['status'], 'failed')
        self.assertEqual(rejected['error'], 'earlier guard violation')
