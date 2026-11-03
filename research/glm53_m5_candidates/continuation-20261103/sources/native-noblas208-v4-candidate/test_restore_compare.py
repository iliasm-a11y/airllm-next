"""Prepared metadata/refusal fixtures, synthetic fixture checks only.

No real build, model, runtime, guard, full comparison or tensor data is mocked
as qualified. These small checks exercise the additional metadata refusals.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('unmeasured_v4_restore_review', HERE / 'restore_compare.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def fixture(mode):
    report = {'mode': mode, 'microbatch': 32,
              'requests': [{'prompt_tokens': 521, 'sequence_bytes': 4, 'sequence_sha256': '1' * 64}] * 3}
    batches = []
    for request in (0, 1, 2):
        import sys
        sys.path.insert(0, str(HERE))
        import geometry
        batches.extend({**row, 'request': request} for row in geometry.request_plan(32, 521, mode, request))
    expected = {(row['request'], row['offset']+row['tokens']) for row in batches
                if (row['offset']+row['tokens']) % 128 == 0 or row['offset']+row['tokens'] in (208, 521)}
    records = [{'request': request, 'token_end': end, 'next_position': end, 'status': 'hashed',
                'state_seq_flags': 0, 'bytes': 4, 'sha256': '1' * 64} for request, end in sorted(expected)]
    return report, batches, records


class RestoreMetadata(unittest.TestCase):
    def test_restore_has_one_prefix_and_all_actual_later_checkpoints(self):
        report, batches, rows = fixture('restore')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'state_checkpoints.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = v.checkpoint_records(root, report, batches)
            self.assertIn((0, 128), result); self.assertIn((0, 208), result)
            self.assertNotIn((1, 128), result); self.assertNotIn((2, 208), result)
            self.assertTrue({(r, n) for r in (0, 1, 2) for n in (256, 384, 512, 521)} <= set(result))

    def test_missing_changed_or_extra_restore_checkpoint_is_refused(self):
        report, batches, rows = fixture('restore')
        cases = [rows[:-1], [row for row in rows if (row['request'], row['token_end']) != (1, 384)],
                 rows + [{'request': 2, 'token_end': 208, 'next_position': 208, 'status': 'hashed',
                          'state_seq_flags': 0, 'bytes': 4, 'sha256': '1' * 64}]]
        changed_final = [dict(row) for row in rows]; changed_final[-1]['sha256'] = '2' * 64
        cases.append(changed_final)
        for selected in cases:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'state_checkpoints.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
                with self.assertRaises(ValueError): v.checkpoint_records(root, report, batches)

    def test_restore_intermediate_difference_cannot_be_hidden_by_exact_final_state(self):
        full = {(r, end): {'bytes': 4, 'sha256': '1'*64} for r in (0, 1, 2) for end in (128, 208, 256, 384, 512, 521)}
        restored = {key: dict(value) for key, value in full.items() if key[0] == 0 or key[1] > 208}
        self.assertTrue(v.compare_checkpoints(full, restored)['passed'])
        restored[1, 384]['sha256'] = '2'*64
        self.assertFalse(v.compare_checkpoints(full, restored)['passed'])

    def test_own_callback_metadata_must_bind_to_actual_batch_index_tokens_and_phase(self):
        batch = {'request': 0, 'offset': 208, 'tokens': 32, 'phase': 'suffix'}
        row = {'request': 0, 'offset': 208, 'input_tokens': 32, 'batch_index': 0, 'phase': 'suffix'}
        self.assertTrue(v.own_batch_metadata({}, [batch], {(0, 208, 'fixture', 0): row}, [row])['passed'])
        for field, value in (('input_tokens', 16), ('batch_index', 1), ('phase', 'full')):
            changed = {**row, field: value}
            with self.assertRaises(ValueError): v.own_batch_metadata({}, [batch], {}, [changed])

    def test_full_fidelity_failure_stops_before_raw_restore_or_numpy_import(self):
        partial = SimpleNamespace(compare=lambda *args: {'passed': False, 'scope': 'early128 is not complete fidelity'})
        with mock.patch.object(v, 'binding_module', return_value=partial):
            result = v.compare(*(['unread synthetic path'] * 9))
        self.assertFalse(result['passed'])
        self.assertFalse(result['snapshot_exact_evidence_passed'])
        self.assertIn('complete raw', result['errors'][0])


if __name__ == '__main__': unittest.main()
