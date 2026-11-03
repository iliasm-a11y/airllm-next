"""Raw numerical rejection check for one closed request in an incomplete arm."""
import argparse
import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def digest(path):
    before = identity(path)
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    if identity(path) != before:
        raise ValueError('Raw payload changed during check')
    return value.hexdigest()


def compare(reference, candidate, index):
    a = json.loads((reference / 'report.json').read_text())['requests'][index]
    b = json.loads((candidate / 'report.json').read_text())['requests'][index]
    if not a.get('continuation_complete') or not b.get('continuation_complete'):
        raise ValueError('Both first requests need closed complete continuations')
    result = {
        'schema_version': 1, 'at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': f'request{index} numerical rejection only; no complete arm/resource/cache/EOS/speed qualification',
        'request': index,
        'full_qualification': False,
        'same_prompt': all(a[key] == b[key] for key in
                           ('prompt_ids_sha256', 'prompt_tokens', 'next_position', 'n_vocab')),
        'same_sampled_ids': [row['token'] for row in a['continuation']] ==
                            [row['token'] for row in b['continuation']],
        'checks': {},
    }
    names = [(f'logits-{index}.f32', a['logits_sha256'], b['logits_sha256'], a['n_vocab'] * 4)]
    for step in range(4):
        names.append((f'continuation-logits-{index}-{step}.f32',
                      a['continuation'][step]['logits_sha256'],
                      b['continuation'][step]['logits_sha256'], a['n_vocab'] * 4))
    names.extend([
        (f'final-state-{index}.bin', a['sequence_sha256'], b['sequence_sha256'], a['sequence_bytes']),
        (f'continuation-state-{index}.bin', a['continuation_state_sha256'],
         b['continuation_state_sha256'], a['continuation_state_bytes']),
    ])
    for name, expected_a, expected_b, size in names:
        pa, pb = reference / name, candidate / name
        if not 0 < size <= 256 * 1024**2 or pa.stat().st_size != size or pb.stat().st_size != size:
            raise ValueError('Raw payload size differs: ' + name)
        da, db = digest(pa), digest(pb)
        if da != expected_a or db != expected_b:
            raise ValueError('Raw payload differs from recorded SHA: ' + name)
        row = {'same_raw_sha256': da == db, 'reference_sha256': da,
               'candidate_sha256': db, 'bytes': size}
        if name.endswith('.f32'):
            left, right = pa.read_bytes(), pb.read_bytes()
            av, bv = array.array('f'), array.array('f')
            av.frombytes(left)
            bv.frombytes(right)
            if sys.byteorder != 'little':
                av.byteswap()
                bv.byteswap()
            finite = all(map(math.isfinite, av)) and all(map(math.isfinite, bv))
            row.update(values=len(av), finite=finite,
                       raw_bit_mismatches=sum(left[i:i+4] != right[i:i+4] for i in range(0, size, 4)),
                       max_abs=max(abs(x-y) for x, y in zip(av, bv)) if finite else None)
        result['checks'][name] = row
    result['request_raw_exact'] = bool(result['same_prompt'] and result['same_sampled_ids'] and
        all(row['same_raw_sha256'] and row.get('finite', True) for row in result['checks'].values()))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--request', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.request)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}))
    raise SystemExit(0 if result['request_raw_exact'] else 2)
