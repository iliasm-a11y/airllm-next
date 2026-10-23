#!/usr/bin/env python3
"""Read bounded GGUF metadata/tensor offsets; never mmap or load weight data."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct

TYPES = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
         4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
         10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
QUANTS = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
          8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
          13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 30: 'BF16'}


def inspect(path):
    with path.open('rb') as source:
        def take(n):
            if not 0 <= n <= 1024**2:
                raise ValueError('Metadata item exceeds the 1MiB read limit')
            value = source.read(n)
            if len(value) != n:
                raise ValueError('Truncated GGUF metadata')
            return value

        def number(fmt):
            return struct.unpack('<'+fmt, take(struct.calcsize(fmt)))[0]

        def string():
            return take(number('Q')).decode('utf-8')

        def value(kind, keep=True):
            if kind in TYPES:
                return number(TYPES[kind][0])
            if kind == 8:
                return string() if keep else take(number('Q')) and None
            if kind != 9:
                raise ValueError(f'Unsupported metadata kind {kind}')
            element, count = number('I'), number('Q')
            if count > 2000000:
                raise ValueError('Metadata array is too long')
            if element in TYPES and not keep:
                source.seek(count*TYPES[element][1], 1)
                return None
            if element == 9:
                raise ValueError('Nested GGUF arrays are unsupported')
            if not keep:
                for _ in range(count):
                    value(element, False)
                return None
            return [value(element) for _ in range(count)]

        if take(4) != b'GGUF' or number('I') != 3:
            raise ValueError('Expected little-endian GGUF version 3')
        tensors, metadata_count = number('Q'), number('Q')
        if tensors > 100000 or metadata_count > 10000:
            raise ValueError('Excessive GGUF index dimensions')
        metadata = {}
        for _ in range(metadata_count):
            key, kind = string(), number('I')
            keep = not key.startswith('tokenizer.')
            found = value(kind, keep)
            if keep:
                metadata[key] = found
        entries = []
        for _ in range(tensors):
            name, ndim = string(), number('I')
            if not 1 <= ndim <= 4:
                raise ValueError('Unsupported GGUF tensor rank')
            ggml_shape = [number('Q') for _ in range(ndim)]
            dtype, offset = number('I'), number('Q')
            entries.append(dict(name=name, shape=list(reversed(ggml_shape)),
                ggml_shape=ggml_shape, dtype_id=dtype, dtype=QUANTS.get(dtype, f'GGML_{dtype}'),
                relative_data_offset=offset, elements=math.prod(ggml_shape)))
        alignment = metadata.get('general.alignment', 32)
        if not isinstance(alignment, int) or not 1 <= alignment <= 65536:
            raise ValueError('Invalid GGUF alignment')
        data_offset = (source.tell()+alignment-1)//alignment*alignment
        for entry in entries:
            entry['file_offset'] = data_offset+entry['relative_data_offset']
            if entry['file_offset'] >= path.stat().st_size:
                raise ValueError('Tensor offset exceeds file size')
    return dict(file=str(path), file_bytes=path.stat().st_size, tensor_data_file_offset=data_offset,
        tensor_count=tensors, metadata_count=metadata_count, metadata=metadata,
        dtype_tensor_counts=dict(Counter(x['dtype'] for x in entries)), tensors=entries,
        weight_payload_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.model_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
