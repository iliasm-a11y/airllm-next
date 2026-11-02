"""Pilot schema/argument checks with synthetic data; no server or model starts."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from benchmark_two_slots import parse_args, probability_rows, validate_slot


def probability():
    return {'id': 10, 'logprob': -0.25,
            'top_logprobs': [{'id': i, 'logprob': -float(i + 1)} for i in range(8)]}


class TwoSlotSchemaTests(unittest.TestCase):
    def test_partial_slot_may_be_absent_default_or_matching(self):
        for event in ({'stop': False}, {'stop': False, 'id_slot': -1}, {'stop': False, 'id_slot': 1}):
            with self.subTest(event=event): validate_slot(event, 1)

    def test_identified_partial_slot_must_match(self):
        for value in (0, -2, True, '1', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_slot({'stop': False, 'id_slot': value}, 1)

    def test_terminal_slot_is_required_and_exact(self):
        validate_slot({'stop': True, 'id_slot': 1}, 1)
        for event in ({'stop': True}, {'stop': True, 'id_slot': -1}, {'stop': True, 'id_slot': 0}):
            with self.subTest(event=event), self.assertRaises(ValueError): validate_slot(event, 1)

    def test_preserves_order_ids_values_and_both_native_field_names(self):
        row = probability()
        expected = [(10, -0.25, [(i, -float(i + 1)) for i in range(8)])]
        self.assertEqual(probability_rows([{'completion_probabilities': [row]}]), expected)
        self.assertEqual(probability_rows([{'probs': [row]}]), expected)

    def test_missing_top8_nonfinite_and_duplicate_ids_rejected(self):
        for mutation in ('short', 'nan', 'duplicate', 'invalid_id'):
            row = probability()
            if mutation == 'short': row['top_logprobs'].pop()
            if mutation == 'nan': row['logprob'] = float('nan')
            if mutation == 'duplicate': row['top_logprobs'][1]['id'] = 0
            if mutation == 'invalid_id': row['id'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probability_rows([{'completion_probabilities': [row]}])

    def test_dense_column_limit_is_explicit_bounded_and_defaults_one(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / 'GLM-5.3-Flash-Q2_K_S-00001-of-00002.gguf'
            model.write_bytes(b'fixture-only')
            base = ['--binary', '/unused/server', '--model', str(model), '--output', str(Path(temp) / 'pilot.json')]
            self.assertEqual(parse_args(base).dense_prefetch_max_cols, 1)
            for value in (1, 2, 4, 8):
                self.assertEqual(parse_args(base + ['--dense-prefetch-max-cols', str(value)]).dense_prefetch_max_cols, value)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + ['--dense-prefetch-max-cols', '3'])


if __name__ == '__main__': unittest.main(verbosity=2)
