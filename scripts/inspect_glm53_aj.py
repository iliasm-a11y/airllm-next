#!/usr/bin/env python3
"""Audit an immutable split GGUF directory through bounded HTTP ranges.

Only bounded prefixes covering the headers (and explicitly requested tiny codec
samples) are fetched. Prefix overread may include initial tensor bytes, but no
full weight shard or unbounded tensor allocation is used. This parses metadata
without importing the inference runtime so an unsupported architecture cannot
be silently normalized during the audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.parse
import urllib.request


class AuditError(ValueError):
    pass


# GGML layouts verified against llama.cpp at this immutable revision.
GGML_REFERENCE = '7df8b876b91208a86a6d792e1140b3deec14535b'
TYPE_LAYOUT = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 8: ('Q8_0', 32, 34),
    10: ('Q2_K', 256, 84), 12: ('Q4_K', 256, 144),
    14: ('Q6_K', 256, 210), 18: ('IQ3_XXS', 256, 98),
    21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82),
    30: ('BF16', 1, 2),
}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
          6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


class BoundedHTTP:
    def __init__(self, max_bytes=24 << 20, max_request_bytes=1 << 20,
                 timeout=60, opener=None):
        if type(max_bytes) is not int or max_bytes < 1:
            raise AuditError('max_bytes must be positive')
        if type(max_request_bytes) is not int or max_request_bytes < 1:
            raise AuditError('max_request_bytes must be positive')
        self.max_bytes = max_bytes
        self.max_request_bytes = max_request_bytes
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.bytes_read = 0
        self.requests = []

    def read_range(self, url, start, count, file_size):
        if any(type(x) is not int for x in (start, count, file_size)):
            raise AuditError('Range bounds must be integers')
        if start < 0 or count < 1 or start + count > file_size:
            raise AuditError('Range is outside the pinned file')
        # Reserve the single overrun-detection byte as well. Even an oversized
        # response must never consume bytes beyond the aggregate allowance.
        if count > self.max_request_bytes or count + 1 > self.max_bytes - self.bytes_read:
            raise AuditError('HTTP range budget exhausted before request')
        end = start + count - 1
        req = urllib.request.Request(url, headers={
            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        with self.opener(req, timeout=self.timeout) as response:
            # Check headers before any body read. A 200 response could be 40 GB.
            if response.status != 206:
                raise AuditError(f'Refusing non-partial HTTP response: {response.status}')
            expected = f'bytes {start}-{end}/{file_size}'
            if response.headers.get('Content-Range') != expected:
                raise AuditError('Content-Range differs from pinned bounds/size')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise AuditError('Encoded range body is unsupported')
            length = response.headers.get('Content-Length')
            if length is not None and length != str(count):
                raise AuditError('Content-Length differs from requested range')
            # Bounded to count+1 even for a dishonest server, never read-to-EOF.
            payload = response.read(count + 1)
            self.bytes_read += len(payload)
            if len(payload) != count:
                raise AuditError('HTTP range body is truncated or oversized')
        self.requests.append({'start': start, 'count': count, 'file_size': file_size,
                              'sha256': hashlib.sha256(payload).hexdigest()})
        return payload


class Prefix:
    def __init__(self, http, url, file_size, cap=16 << 20, chunk_bytes=1 << 20):
        self.http, self.url, self.file_size = http, url, file_size
        self.cap, self.chunk_bytes = cap, chunk_bytes
        self.data = bytearray()
        self.pos = 0
        self.values_seen = 0

    def ensure(self, end):
        if end < 0 or end > min(self.cap, self.file_size):
            raise AuditError('Header exceeds its bounded prefix allowance')
        while len(self.data) < end:
            count = min(self.chunk_bytes, self.cap - len(self.data),
                        self.file_size - len(self.data))
            self.data.extend(self.http.read_range(self.url, len(self.data), count,
                                                 self.file_size))

    def read(self, size):
        if type(size) is not int or size < 0:
            raise AuditError('Invalid parser read')
        self.ensure(self.pos + size)
        start, self.pos = self.pos, self.pos + size
        return bytes(self.data[start:self.pos])

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self, retain=True):
        size = self.unpack('Q')
        if size > 16 << 20:
            raise AuditError('GGUF string exceeds parser limit')
        if retain:
            try:
                return self.read(size).decode('utf-8')
            except UnicodeDecodeError as exc:
                raise AuditError('Invalid UTF-8 in GGUF') from exc
        self.ensure(self.pos + size)
        self.pos += size
        return None

    def value(self, kind, retain=True, depth=0):
        self.values_seen += 1
        if self.values_seen > 2_000_000 or depth > 2:
            raise AuditError('Metadata nesting/item allowance exceeded')
        if kind in SCALAR:
            value = self.unpack(SCALAR[kind])
            return value if retain else None
        if kind == 8:
            return self.string(retain)
        if kind != 9:
            raise AuditError(f'Unsupported GGUF metadata type {kind}')
        element_type, count = self.unpack('I'), self.unpack('Q')
        if count > 1_000_000 or element_type == 9:
            raise AuditError('Invalid or excessive GGUF array')
        values = [] if retain and count <= 128 else None
        start = self.pos
        for _ in range(count):
            v = self.value(element_type, retain=values is not None, depth=depth + 1)
            if values is not None:
                values.append(v)
        if values is not None:
            return values
        if not retain:
            return None
        return {'array_type': element_type, 'count': count,
                'serialized_values_sha256': hashlib.sha256(
                    memoryview(self.data)[start:self.pos]).hexdigest()}


def parse_header(prefix):
    if prefix.read(4) != b'GGUF':
        raise AuditError('Not a little-endian GGUF file')
    version = prefix.unpack('I')
    if version not in (2, 3):
        raise AuditError(f'Unsupported GGUF version {version}')
    tensor_count, metadata_count = prefix.unpack('Q'), prefix.unpack('Q')
    if tensor_count > 100_000 or metadata_count > 4096:
        raise AuditError('GGUF directory exceeds parser allowance')
    metadata = {}
    for _ in range(metadata_count):
        key, kind = prefix.string(), prefix.unpack('I')
        if key in metadata:
            raise AuditError('Duplicate metadata key')
        metadata[key] = prefix.value(kind)
    alignment = metadata.get('general.alignment', 32)
    if type(alignment) is not int or alignment < 1 or alignment > 1 << 20:
        raise AuditError('Invalid GGUF alignment')
    tensors, seen = [], set()
    for _ in range(tensor_count):
        name, ndims = prefix.string(), prefix.unpack('I')
        if name in seen or not 1 <= ndims <= 4:
            raise AuditError('Duplicate tensor or invalid dimension count')
        seen.add(name)
        dimensions = [prefix.unpack('Q') for _ in range(ndims)]
        type_id, offset = prefix.unpack('I'), prefix.unpack('Q')
        if any(d < 1 or d > 1 << 40 for d in dimensions) or type_id not in TYPE_LAYOUT:
            raise AuditError(f'Invalid tensor dimensions or unsupported type {type_id}')
        label, block, block_bytes = TYPE_LAYOUT[type_id]
        if dimensions[0] % block or offset % alignment:
            raise AuditError('Tensor row width/block or offset/alignment mismatch')
        row_bytes = dimensions[0] // block * block_bytes
        nbytes = row_bytes * math.prod(dimensions[1:])
        tensors.append({'name': name, 'dimensions': dimensions, 'type_id': type_id,
                        'type': label, 'offset': offset, 'nbytes': nbytes,
                        'row_bytes': row_bytes})
    header_bytes = prefix.pos
    data_start = (header_bytes + alignment - 1) // alignment * alignment
    previous_end = data_start
    for tensor in sorted(tensors, key=lambda t: t['offset']):
        absolute = data_start + tensor['offset']
        if absolute < previous_end or tensor['nbytes'] > prefix.file_size - absolute:
            raise AuditError('Overlapping tensor or tensor extends beyond pinned file')
        tensor['absolute_offset'] = absolute
        previous_end = absolute + tensor['nbytes']
    return {'gguf_version': version, 'metadata_count': metadata_count,
            'tensor_count': tensor_count, 'alignment': alignment,
            'header_bytes': header_bytes, 'data_start': data_start,
            'header_sha256': hashlib.sha256(memoryview(prefix.data)[:header_bytes]).hexdigest(),
            'prefix_downloaded_bytes': len(prefix.data),
            'metadata': metadata, 'tensors': tensors}


def inspect_manifest(manifest, output_dir, *, max_bytes=24 << 20,
                     max_header_bytes=16 << 20, chunk_bytes=1 << 20,
                     http=None, sample_blocks=0):
    model_id, revision = manifest['model_id'], manifest['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise AuditError('Manifest revision must be an immutable commit SHA')
    files = manifest['files']
    if len(files) != 3:
        raise AuditError('AJ-IQ3_XXS audit requires exactly three shards')
    if not 0 <= sample_blocks <= 16:
        raise AuditError('Codec samples must be bounded to at most 16 blocks/type')
    http = http or BoundedHTTP(max_bytes=max_bytes, max_request_bytes=chunk_bytes)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards, all_tensors, names, common = [], [], set(), {}
    samples, sampled_types = [], set()
    for i, entry in enumerate(files):
        remote_name = entry.get('path', entry.get('remote_path', entry['name']))
        url = ('https://huggingface.co/' + model_id + '/resolve/' + revision + '/' +
               urllib.parse.quote(remote_name, safe='/'))
        request_begin = len(http.requests)
        parsed = parse_header(Prefix(http, url, entry['size'], max_header_bytes,
                                     chunk_bytes))
        meta = parsed['metadata']
        if meta.get('split.no') != i or meta.get('split.count') != 3:
            raise AuditError('Shard split index/count disagrees with manifest order')
        for key, value in meta.items():
            if key == 'split.no':
                continue
            if key in common and common[key] != value:
                raise AuditError(f'Conflicting metadata across shards: {key}')
            common[key] = value
        for tensor in parsed['tensors']:
            if tensor['name'] in names:
                raise AuditError('Duplicate tensor across shards')
            names.add(tensor['name'])
            tensor['shard_index'] = i
            tensor['shard_name'] = entry['name']
            all_tensors.append(tensor)
            if sample_blocks and tensor['type_id'] in (18, 21, 22) and tensor['type_id'] not in sampled_types:
                block_bytes = TYPE_LAYOUT[tensor['type_id']][2]
                count = min(tensor['nbytes'] // block_bytes, sample_blocks) * block_bytes
                raw = http.read_range(url, tensor['absolute_offset'], count, entry['size'])
                sample_name = f"sample-{tensor['type']}.bin"
                (out / sample_name).write_bytes(raw)
                samples.append({'file': sample_name, 'type_id': tensor['type_id'],
                                'tensor': tensor['name'], 'remote_path': remote_name,
                                'absolute_offset': tensor['absolute_offset'], 'bytes': count,
                                'sha256': hashlib.sha256(raw).hexdigest()})
                sampled_types.add(tensor['type_id'])
        parsed['file'] = entry
        parsed['requests'] = http.requests[request_begin:]
        parsed['tensor_type_counts'] = dict(Counter(t['type'] for t in parsed['tensors']))
        shards.append(parsed)
        (out / f'shard-{i + 1:02d}-header.json').write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False) + '\n')
    if common.get('split.tensors.count') != len(all_tensors):
        raise AuditError('Global tensor count does not match split metadata')
    families = defaultdict(Counter)
    for tensor in all_tensors:
        family = re.sub(r'^blk\.\d+\.', 'blk.*.', tensor['name'])
        families[family][tensor['type']] += 1
    experts = [t for t in all_tensors if '_exps' in t['name']]
    architecture = {k: v for k, v in common.items() if not k.startswith('tokenizer.')}
    summary = {
        'model_id': model_id, 'revision': revision, 'ggml_reference': GGML_REFERENCE,
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': [s['file'] for s in shards], 'payload_bytes': sum(e['size'] for e in files),
        'http_bytes_read': http.bytes_read, 'http_budget_bytes': http.max_bytes,
        'tensor_count': len(all_tensors),
        'tensor_type_counts': dict(Counter(t['type'] for t in all_tensors)),
        'tensor_type_encoded_bytes': {kind: sum(t['nbytes'] for t in all_tensors if t['type'] == kind)
                                      for kind in sorted({t['type'] for t in all_tensors})},
        'tensor_families': {k: dict(v) for k, v in sorted(families.items())},
        'routed_experts': [{'name': t['name'], 'type': t['type'], 'dimensions': t['dimensions']}
                           for t in experts],
        'samples': samples, 'full_weights_downloaded': False,
        'real_checkpoint_generation_validated': False, 'm5_execution_validated': False,
    }
    for name, value in [('checkpoint-header-audit.json', {'summary': summary, 'shards': shards}),
                        ('checkpoint-architecture-metadata.json', architecture),
                        ('checkpoint-tokenizer-metadata.json', {k: v for k, v in common.items()
                                                                if k.startswith('tokenizer.')}),
                        ('checkpoint-tensor-directory.json', {'revision': revision,
                                                               'tensors': all_tensors}),
                        ('checkpoint-audit-summary.json', summary)]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-bytes', type=int, default=24 << 20)
    parser.add_argument('--max-header-bytes', type=int, default=16 << 20)
    parser.add_argument('--chunk-bytes', type=int, default=1 << 20)
    parser.add_argument('--sample-blocks', type=int, default=0)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    if manifest_path.stat().st_size > 2 << 20:
        parser.error('Manifest exceeds 2 MiB limit')
    summary = inspect_manifest(json.loads(manifest_path.read_text()), args.output_dir,
                               max_bytes=args.max_bytes, max_header_bytes=args.max_header_bytes,
                               chunk_bytes=args.chunk_bytes, sample_blocks=args.sample_blocks)
    print(json.dumps({k: summary[k] for k in ('revision', 'payload_bytes', 'http_bytes_read',
                                             'tensor_count', 'tensor_type_counts', 'samples')},
                     indent=2))


if __name__ == '__main__':
    main()
