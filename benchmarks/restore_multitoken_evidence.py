#!/usr/bin/env python3
"""Restore and hash-check the losslessly compressed public JSON evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore(root):
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for entry in manifest['json_reports']:
        relative = Path(entry['path'])
        packed_relative = Path(entry['archive'])
        if any(path.is_absolute() or '..' in path.parts for path in (relative, packed_relative)):
            raise ValueError('Unsafe evidence path')
        packed = (root/packed_relative).read_bytes()
        if sha(packed) != entry['archive_sha256']:
            raise ValueError(f'Archive digest mismatch: {packed_relative}')
        raw = gzip.decompress(packed)
        if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
            raise ValueError(f'Restored digest mismatch: {relative}')
        target = root/relative
        if target.exists() and target.read_bytes() != raw:
            raise FileExistsError(f'Refusing to replace different evidence: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return len(manifest['json_reports'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'evidence-multitoken')
    args = parser.parse_args()
    print(f'Restored and verified {restore(args.root.resolve())} JSON reports')
