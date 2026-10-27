#!/usr/bin/env python3
"""Paired GGUF index parsing; no tensor reads, inference, or cache purges.

Run with PYTHONPATH=air_llm. Both arms use bulk numeric metadata, isolating
the optional string-array buffer. Repeats alternate AB/BA on the same host;
results are process-warm and OS-cache uncontrolled, never cold-SSD evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import struct
import sys
from time import perf_counter


def _identity(path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _fingerprint(reader):
    digest = hashlib.sha256()
    def update(value):
        if isinstance(value, str):
            raw = value.encode('utf-8')
            digest.update(b's' + struct.pack('<Q', len(raw)))
            digest.update(raw)
        elif isinstance(value, (tuple, list)):
            digest.update(b'l' + struct.pack('<Q', len(value)))
            for item in value:
                update(item)
        elif isinstance(value, dict):
            digest.update(b'd' + struct.pack('<Q', len(value)))
            for key in sorted(value):
                update(key)
                update(value[key])
        elif type(value) is float:
            digest.update(b'f' + struct.pack('<d', value))
        else:
            update(f'{type(value).__name__}:{value}')
    update(reader.metadata)
    update([(key, value.dimensions, value.quant_type.name, value.offset,
             value.nbytes, value.row_bytes) for key, value in sorted(reader.tensors.items())])
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--string-buffer-bytes', type=int, default=65536)
    parser.add_argument('--tokenizer', action='store_true', help='Also construct the actual embedded tokenizer')
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error('--repeats must be at least two for balanced order')
    if not 1 <= args.string_buffer_bytes <= 65536:
        parser.error('--string-buffer-bytes must be between 1 and 65536')
    from airllm.nextgen.gguf import GGUFReader
    from airllm.nextgen.experiment_guard import system_snapshot, validate_output_paths
    source_paths = [Path(sys.modules[GGUFReader.__module__].__file__).resolve(), Path(__file__).resolve()]
    def source_hashes():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    sources = source_hashes()
    validate_output_paths(args.output, protected=[args.model])
    identity = _identity(args.model)
    result = {'kind': 'hyperstream_metadata_paired', 'python': sys.version,
              'platform': platform.platform(), 'model_identity': identity,
              'cache_regime': 'process-warm; OS cache uncontrolled; no purge',
              'bulk_metadata': True, 'string_buffer_bytes': args.string_buffer_bytes,
              'source_sha256_before': sources,
              'before': system_snapshot(), 'samples': []}
    expected = {}
    for retain in (False, True):
        for repeat in range(args.repeats):
            order = ('legacy', 'buffered') if repeat % 2 == 0 else ('buffered', 'legacy')
            for variant in order:
                gc.collect()
                start = perf_counter()
                with GGUFReader(args.model, memory_budget_bytes=256 << 20,
                                max_metadata_bytes=192 << 20, include_tokenizer=retain,
                                bulk_metadata=True,
                                string_buffer_bytes=0 if variant == 'legacy' else args.string_buffer_bytes) as reader:
                    parse_seconds = perf_counter() - start
                    tokenizer_seconds = None
                    if retain and args.tokenizer:
                        from airllm.nextgen.gguf_tokenizer import GGUFTokenizer
                        tokenizer_start = perf_counter()
                        tokenizer = GGUFTokenizer(reader.metadata)
                        tokenizer_seconds = perf_counter() - tokenizer_start
                        del tokenizer
                    fingerprint = _fingerprint(reader)
                    expected.setdefault(retain, fingerprint)
                    if fingerprint != expected[retain]:
                        raise AssertionError('GGUF metadata/index differs between paired arms')
                    if reader.stats['tensor_bytes']:
                        raise AssertionError('Index benchmark read tensor payload')
                    sample = {'variant': variant, 'retain_tokenizer_metadata': retain,
                              'repeat': repeat, 'order': list(order),
                              'parse_seconds': parse_seconds, 'tokenizer_seconds': tokenizer_seconds,
                              'metadata_bytes': reader.metadata_bytes, 'fingerprint': fingerprint,
                              'stats': dict(reader.stats)}
                del reader
                if _identity(args.model) != identity:
                    raise AssertionError('Input model changed during metadata benchmark')
                result['samples'].append(sample)
                print(json.dumps(sample), flush=True)
    result['summary'] = {}
    for retain in (False, True):
        grouped = {variant: [s for s in result['samples']
                             if s['variant'] == variant and s['retain_tokenizer_metadata'] == retain]
                   for variant in ('legacy', 'buffered')}
        result['summary'][str(retain).lower()] = {
            variant: {'median_parse_seconds': statistics.median(s['parse_seconds'] for s in samples),
                      'median_read_calls': statistics.median(s['stats']['read_calls'] for s in samples),
                      'median_index_bytes': statistics.median(s['stats']['index_bytes'] for s in samples)}
            for variant, samples in grouped.items()}
    result['after'] = system_snapshot()
    result['source_sha256_after'] = source_hashes()
    if result['source_sha256_after'] != sources:
        raise AssertionError('Metadata benchmark sources changed during measurement')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
