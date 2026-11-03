"""Portable exhaustive scheduling properties; no model/compiler/native runtime."""
from pathlib import Path
import json
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import geometry as g


def parity(rows, end):
    values = bytearray(end)
    for row in rows:
        values[row['offset']:row['offset'] + row['tokens']] = bytes([2 if row['tokens'] % 2 == 0 else 1]) * row['tokens']
    return values


class Geometry(unittest.TestCase):
    def test_all_bounded_lengths_preserve_cpu8_parity_and_all_fences(self):
        for end in range(1, 1801):
            reference = parity(g.plan(8, 0, end), end)
            for ub in g.BATCHES:
                rows = g.plan(ub, 0, end)
                self.assertEqual(parity(rows, end), reference, (ub, end))
                self.assertEqual(sum(row['tokens'] for row in rows), end)
                ends = {row['offset'] + row['tokens'] for row in rows}
                self.assertTrue({f for f in g.FENCES if f <= end} <= ends)
                self.assertTrue(all(0 < row['tokens'] <= ub for row in rows))

    def test_all_private_suffixes_have_identical_actual_boundaries(self):
        for end in range(209, 1801):
            for ub in g.BATCHES:
                full = g.plan(ub, 0, end)
                suffix = [(r['offset'], r['tokens']) for r in full if r['offset'] >= 208]
                for mode in ('split', 'restore'):
                    for request in (0, 1, 2):
                        actual = g.request_plan(ub, end, mode, request)
                        self.assertEqual(suffix, [(r['offset'], r['tokens']) for r in actual if r['offset'] >= 208], (ub, end, mode, request))

    def test_current_geometric_cost_and_unadmitted_gpu_shapes_are_explicit(self):
        self.assertEqual([len(g.plan(ub, 0, 521)) for ub in g.BATCHES], [66, 34, 19, 11, 7])
        self.assertEqual([len(g.plan(ub, 0, 542)) for ub in g.BATCHES], [68, 34, 19, 11, 7])
        self.assertTrue({48, 80} <= {r['tokens'] for r in g.plan(128, 0, 521)})

    def test_native_source_uses_absolute_fences_and_exact_policy(self):
        source = (ROOT / 'research/glm53_batched_prefill/native_prefill.cpp').read_text()
        values = re.search(r'prefill_fences = \{([^}]+)\}', source).group(1)
        self.assertEqual(tuple(map(int, values.split(','))), g.FENCES)
        self.assertIn('prefill_take(ub,offset,end)', source)
        self.assertIn('prefill_fence_end(offset,end)-offset', source)
        self.assertIn(g.POLICY, source)
        self.assertIn('{"prefill_chunk_policy_version",4}', source)
        self.assertIn('{"prefill_fences",prefill_fences}', source)
        self.assertIn('offset%128==0 || offset==208', source)

    def test_malformed_receipts_cannot_replace_observed_fences(self):
        report = {'state_parameters': g.PARAMETERS.copy(), **g.PARAMETERS,
            'prefill_chunk_policy_qualified': False, 'microbatch': 32, 'mode': 'full',
            'requests': [{'prompt_tokens': 521}] * 3}
        batches = [{**row, 'request': request} for request in range(3) for row in g.request_plan(32, 521, 'full', request)]
        self.assertTrue(g.check(report, batches)['passed'])
        batches[6]['tokens'] = 32  # skip the actual208 boundary while declaring v4
        with self.assertRaisesRegex(ValueError, 'boundary/fence/tail'):
            g.check(report, batches)

    def test_missing208_or_changed_final_checkpoint_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {'mode': 'full', 'requests': [{'prompt_tokens': 521, 'sequence_bytes': 4, 'sequence_sha256': '1' * 64}] * 3}
            rows = [{'request': request, 'token_end': n, 'next_position': n, 'status': 'hashed',
                     'state_seq_flags': 0, 'bytes': 4, 'sha256': '1' * 64}
                    for request in range(3) for n in (128, 208, 256, 384, 512, 521)]
            path = root / 'state_checkpoints.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows if r['token_end'] != 208))
            with self.assertRaisesRegex(ValueError, 'including208'):
                g.checkpoints(root, report)
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(len(g.checkpoints(root, report)), 18)
            rows[-1]['sha256'] = '2' * 64
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'final checkpoint'):
                g.checkpoints(root, report)

    def test_invalid_bounds_and_unknown_profiles_are_refused(self):
        for args in ((32, 0, 0), (32, -1, 208), (256, 0, 521), (True, 0, 521), (32, 0, 1801)):
            with self.assertRaises(ValueError):
                g.plan(*args)

    def test_direct_restore_uses_real208_states_and_refuses_wrong_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def report(mode):
                return {'state_parameters': g.PARAMETERS.copy(), **g.PARAMETERS,
                    'prefill_chunk_policy_qualified': False, 'microbatch': 32, 'mode': mode,
                    'requests': [{'prompt_tokens': 521, 'sequence_bytes': 4, 'sequence_sha256': '1' * 64}] * 3}
            full, restored = report('full'), report('restore')
            rows = [{'request': request, 'token_end': n, 'next_position': n, 'status': 'hashed',
                     'state_seq_flags': 0, 'bytes': 4, 'sha256': '1' * 64}
                    for request in range(3) for n in (128, 208, 256, 384, 512, 521)]
            (root / 'state_checkpoints.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            batches = lambda mode: [{**r, 'request': request} for request in range(3)
                                    for r in g.request_plan(32, 521, mode, request)]
            restored['snapshot'] = {'bytes': 4, 'sha256': '2' * 64}
            with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                g.direct_restore(root, full, restored, batches('full'), batches('restore'))
            restored['snapshot']['sha256'] = '1' * 64
            # Metadata geometry fixture only. This does not simulate a model,
            # guard receipt, successful build or full comparison verdict.
            self.assertTrue(g.direct_restore(root, full, restored, batches('full'), batches('restore'))['passed'])
            full['state_parameters'] = {**g.PARAMETERS, 'prefill_fences': [128.0, 208, 256, 384, 512]}
            with self.assertRaisesRegex(ValueError, 'integer native-token'):
                g.check(full, batches('full'))


if __name__ == '__main__':
    unittest.main()
