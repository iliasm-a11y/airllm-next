"""Bounded range audit: HTTP refusal happens before unbounded body reads."""
from contextlib import AbstractContextManager
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import unittest

_path = Path(__file__).resolve().parents[1] / 'scripts' / 'inspect_glm53_aj.py'
_spec = importlib.util.spec_from_file_location('aj_inspection', _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


class Response(AbstractContextManager):
    def __init__(self, body=b'', *, status=206, headers=None):
        self.body, self.status = body, status
        self.headers = headers or {}
        self.reads = []

    def read(self, count):
        self.reads.append(count)
        return self.body[:count]

    def __exit__(self, *_):
        return False


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def file_with_tensor(kind=22, width=256):
    metadata = [('general.architecture', 8, string('glm5next')),
                ('split.no', 2, struct.pack('<H', 0)),
                ('split.count', 2, struct.pack('<H', 3)),
                ('split.tensors.count', 10, struct.pack('<Q', 3))]
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for key, value_type, value in metadata:
        data += string(key) + struct.pack('<I', value_type) + value
    data += string('blk.3.ffn_gate_exps.weight') + struct.pack('<IQQQIQ', 3, width, 2, 3, kind, 0)
    data += b'\0' * ((-len(data)) % 32)
    return data + b'\0' * 492


class InspectionTests(unittest.TestCase):
    def test_200_refused_without_reading_body(self):
        response = Response(status=200)
        http = audit.BoundedHTTP(opener=lambda *_args, **_kw: response)
        with self.assertRaisesRegex(audit.AuditError, 'non-partial'):
            http.read_range('https://example.test/file', 0, 4, 40_000_000_000)
        self.assertEqual(response.reads, [])
        self.assertEqual(http.bytes_read, 0)

    def test_wrong_content_range_refused_before_read(self):
        response = Response(headers={'Content-Range': 'bytes 0-3/5'})
        http = audit.BoundedHTTP(opener=lambda *_args, **_kw: response)
        with self.assertRaisesRegex(audit.AuditError, 'Content-Range'):
            http.read_range('https://example.test/file', 0, 4, 6)
        self.assertEqual(response.reads, [])

    def test_wrong_content_length_refused_before_read(self):
        response = Response(headers={'Content-Range': 'bytes 0-3/6', 'Content-Length': '6'})
        http = audit.BoundedHTTP(opener=lambda *_args, **_kw: response)
        with self.assertRaisesRegex(audit.AuditError, 'Content-Length'):
            http.read_range('https://example.test/file', 0, 4, 6)
        self.assertEqual(response.reads, [])

    def test_budget_refused_before_open(self):
        def forbidden(*_args, **_kw):
            self.fail('Opener must not be called after budget exhaustion')
        http = audit.BoundedHTTP(max_bytes=3, opener=forbidden)
        with self.assertRaisesRegex(audit.AuditError, 'budget'):
            http.read_range('https://example.test/file', 0, 4, 100)

    def test_oversized_response_is_bounded_and_rejected(self):
        response = Response(b'12345', headers={'Content-Range': 'bytes 0-3/6'})
        http = audit.BoundedHTTP(opener=lambda *_args, **_kw: response)
        with self.assertRaisesRegex(audit.AuditError, 'oversized'):
            http.read_range('https://example.test/file', 0, 4, 6)
        self.assertEqual(response.reads, [5])

    def test_budget_reserves_overrun_detection_byte(self):
        def forbidden(*_args, **_kw):
            self.fail('The overrun guard byte must be budgeted before open')
        http = audit.BoundedHTTP(max_bytes=4, opener=forbidden)
        with self.assertRaisesRegex(audit.AuditError, 'budget'):
            http.read_range('https://example.test/file', 0, 4, 100)

    def test_parser_uses_real_type_layout_and_dimension_order(self):
        raw = file_with_tensor()
        def opener(req, **_kw):
            bounds = req.headers['Range'].split('=')[1]
            start, end = map(int, bounds.split('-'))
            body = raw[start:end + 1]
            return Response(body, headers={'Content-Range': f'bytes {start}-{end}/{len(raw)}',
                                           'Content-Length': str(len(body))})
        http = audit.BoundedHTTP(max_bytes=4096, max_request_bytes=128, opener=opener)
        result = audit.parse_header(audit.Prefix(http, 'https://example.test/file',
                                                len(raw), cap=1024, chunk_bytes=128))
        self.assertEqual(result['metadata']['general.architecture'], 'glm5next')
        self.assertEqual(result['tensor_count'], 1)
        self.assertEqual(result['tensors'][0]['dimensions'], [256, 2, 3])
        self.assertEqual(result['tensors'][0]['row_bytes'], 82)
        self.assertEqual(result['tensors'][0]['nbytes'], 492)
        self.assertEqual(result['header_sha256'], hashlib.sha256(raw[:result['header_bytes']]).hexdigest())
        self.assertLess(http.bytes_read, len(raw))

    def test_invalid_width_is_refused(self):
        raw = file_with_tensor(width=255)
        class MemoryHTTP:
            def read_range(self, _url, start, count, _size):
                return raw[start:start + count]
        with self.assertRaisesRegex(audit.AuditError, 'row width/block'):
            audit.parse_header(audit.Prefix(MemoryHTTP(), '', len(raw), chunk_bytes=128))

    def test_header_cap_prevents_network_request(self):
        class NeverHTTP:
            def read_range(self, *_):
                raise AssertionError('No request is allowed outside header cap')
        prefix = audit.Prefix(NeverHTTP(), '', 1000, cap=64, chunk_bytes=64)
        with self.assertRaisesRegex(audit.AuditError, 'prefix allowance'):
            prefix.read(65)

    def test_captured_real_split_headers_are_consistent(self):
        root = Path(__file__).resolve().parents[1] / 'research' / 'ronin_aj_iq3xxs'
        summary = json.loads((root / 'checkpoint-audit-summary.json').read_text())
        shards = [json.loads((root / f'shard-{i:02d}-header.json').read_text())
                  for i in (1, 2, 3)]
        self.assertEqual([s['tensor_count'] for s in shards], [523, 469, 391])
        self.assertEqual(sum(s['tensor_count'] for s in shards), 1383)
        self.assertEqual(summary['tensor_type_counts'],
                         {'Q6_K': 339, 'F32': 704, 'Q8_0': 214,
                          'IQ3_S': 8, 'IQ2_S': 84, 'IQ3_XXS': 34})
        self.assertLess(summary['http_bytes_read'], summary['http_budget_bytes'])
        for i, shard in enumerate(shards):
            self.assertEqual(shard['metadata']['split.no'], i)
            self.assertEqual(shard['metadata']['split.count'], 3)
            self.assertEqual(shard['metadata']['split.tensors.count'], 1383)

    def test_captured_aj_tensors_preserve_q2_names_and_shapes(self):
        root = Path(__file__).resolve().parents[1] / 'research'
        aj = json.loads((root / 'ronin_aj_iq3xxs' / 'checkpoint-tensor-directory.json').read_text())
        q2 = json.loads((root / 'ronin_glm53' / 'checkpoint-tensor-directory.json').read_text())
        self.assertEqual({t['name']: t['dimensions'] for t in aj['tensors']},
                         {t['name']: t['dimensions'] for t in q2['tensors']})
        iq3s = [t for t in aj['tensors'] if t['type_id'] == 21]
        self.assertEqual({t['name'] for t in iq3s},
                         {f'blk.{i}.ffn_down_exps.weight' for i in (3, 4, 5, 6, 41, 42, 43, 44)})


if __name__ == '__main__':
    unittest.main()
