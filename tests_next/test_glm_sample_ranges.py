"""A range-ignoring server must never trigger a multi-GB weight download."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_path = Path(__file__).resolve().parents[1] / 'research/v5/validate_glm53_sample.py'
_spec = importlib.util.spec_from_file_location('glm_sample_protocol', _path)
sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample)


class Response:
    def __init__(self, *, status=206, body=b'xyz', start=7, end=9, total=100,
                 declared=3):
        self.status, self.body, self.read_calls = status, body, 0
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}
        if declared is not None:
            self.headers['Content-Length'] = str(declared)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, count):
        self.read_calls += 1
        return self.body[:count]


class SampleRangeGuards(unittest.TestCase):
    def test_http200_rejected_before_any_body_read(self):
        response = Response(status=200)
        with patch.object(sample, 'urlopen', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'before body read'):
                sample.Ranges().read('https://official.invalid/model', 7, 3)
        self.assertEqual(response.read_calls, 0)

    def test_wrong_range_or_length_rejected_before_body_read(self):
        for response in (Response(start=0, end=2), Response(declared=100)):
            with patch.object(sample, 'urlopen', return_value=response):
                with self.assertRaises(RuntimeError):
                    sample.Ranges().read('https://official.invalid/model', 7, 3)
            self.assertEqual(response.read_calls, 0)

    def test_exact206_records_only_requested_body(self):
        response = Response()
        with patch.object(sample, 'urlopen', return_value=response):
            fetch = sample.Ranges()
            self.assertEqual(fetch.read('https://official.invalid/model', 7, 3), b'xyz')
        self.assertEqual(fetch.used, 3)
        self.assertEqual(fetch.requests[0]['content_range'], 'bytes 7-9/100')

    def test_truncated_and_overlong_bodies_are_rejected(self):
        for body in (b'xy', b'xyzz'):
            with patch.object(sample, 'urlopen', return_value=Response(body=body, declared=None)):
                with self.assertRaises(EOFError):
                    sample.Ranges().read('https://official.invalid/model', 7, 3)

    def test_allowance_reserves_overlong_probe_before_network(self):
        fetch = sample.Ranges()
        fetch.used = sample.MAX_BODY_BYTES - 3
        with patch.object(sample, 'urlopen') as opener:
            with self.assertRaises(MemoryError):
                fetch.read('https://official.invalid/model', 7, 3)
        opener.assert_not_called()

    def test_inconsistent_resource_size_is_rejected_before_read(self):
        fetch = sample.Ranges()
        with patch.object(sample, 'urlopen', side_effect=[Response(), Response(total=200)]) as opener:
            fetch.read('https://official.invalid/model', 7, 3)
            with self.assertRaisesRegex(RuntimeError, 'size changed'):
                fetch.read('https://official.invalid/model', 7, 3)


if __name__ == '__main__':
    unittest.main()
