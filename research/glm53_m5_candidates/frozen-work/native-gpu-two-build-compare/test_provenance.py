"""Negative metadata fixtures only; no successful build or model is simulated."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as p


class ProvenanceRejections(unittest.TestCase):
    def test_null_or_malformed_candidate_hash_refused(self):
        for value in (None, '', '0' * 63, 'A' * 64):
            with self.assertRaises(ValueError):
                p.hash_value(value)

    def test_actual_binary_change_refused(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'helper'
            path.write_bytes(b'synthetic old helper')
            record = p.descriptor(path)
            path.write_bytes(b'synthetic new helper')
            with self.assertRaisesRegex(ValueError, 'actual file bytes'):
                p.checked_file(record)

    def test_actual_header_drift_refused(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'include/llama.h'
            path.parent.mkdir()
            path.write_bytes(b'synthetic header before')
            build = {'inputs_sha256': {str(path): p.sha(path)}}
            path.write_bytes(b'synthetic header changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                p.build_input(build, 'include/llama.h')

    def test_duplicate_or_missing_source_input_refused(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            p.build_input({'inputs_sha256': {}}, 'glm53_batched_prefill/native_prefill.cpp')
        with self.assertRaisesRegex(ValueError, 'unique'):
            p.build_input({'inputs_sha256': {'/fixture/a/include/llama.h': '0' * 64,
                                           '/fixture/b/include/llama.h': '0' * 64}}, 'include/llama.h')

    def test_wrong_own_key_refused_before_dependency_normalization(self):
        key = types.SimpleNamespace(canonical_key=lambda manifest: ('1' * 64, b'canonical fixture'))
        report = {'snapshot_key_sha256': '2' * 64}
        receipt = {'preflight': {'snapshot_manifest': {}, 'snapshot_key_sha256': '1' * 64}}
        with self.assertRaisesRegex(ValueError, 'own native/report key'):
            p.dependencies(report, receipt, {}, {}, {'key': key})

    def test_source_compound_or_keyed_state_cannot_be_substituted(self):
        key = types.SimpleNamespace(canonical_key=lambda manifest: ('1' * 64, b'canonical fixture'))
        manifest = {'state_parameters': {'n_ctx': 2048}}
        report = {'snapshot_key_sha256': '1' * 64, 'state_parameters': {'n_ctx': 4096}}
        receipt = {'preflight': {'snapshot_manifest': manifest, 'snapshot_key_sha256': '1' * 64}}
        with self.assertRaisesRegex(ValueError, 'state_parameters'):
            p.dependencies(report, receipt, {}, {}, {'key': key})

    def test_odd_tail_policy_requires_actual_cpu8_remainder(self):
        report = {'mode': 'full', 'microbatch': 16, 'requests': [{'prompt_tokens': 523}]}
        batches = [{'request': 0, 'offset': offset, 'tokens': 16, 'unadjusted_take': 16,
                    'odd_tail_split': False, 'prefill_chunk_policy': p.POLICY} for offset in range(0, 512, 16)]
        # An even-plus-one policy would change CPU8's final3 scalar columns.
        wrong = batches + [{'request': 0, 'offset': 512, 'tokens': 10}, {'request': 0, 'offset': 522, 'tokens': 1}]
        with self.assertRaisesRegex(ValueError, 'actual chunks'):
            p.check_chunks(report, wrong, gpu=True)
        correct = batches + [{'request': 0, 'offset': 512, 'tokens': 8, 'unadjusted_take': 11,
                              'odd_tail_split': True, 'prefill_chunk_policy': p.POLICY},
                             {'request': 0, 'offset': 520, 'tokens': 3, 'unadjusted_take': 3,
                              'odd_tail_split': False, 'prefill_chunk_policy': p.POLICY}]
        p.check_chunks(report, correct, gpu=True)
        correct[-1]['prefill_chunk_policy'] = 'unreviewed-tail-policy'
        with self.assertRaisesRegex(ValueError, 'receipt missing or false'):
            p.check_chunks(report, correct, gpu=True)


if __name__ == '__main__':
    unittest.main()
