#!/usr/bin/env python3
"""Verify and restore public JSON/log/XML research evidence, never weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'evidence-research51/raw-evidence.tar.gz')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'evidence-research51/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('A new restoration directory is required')
    manifest = json.loads(args.manifest.read_text())
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    expected = manifest['files']
    if not isinstance(expected, dict) or len(expected) > 10000:
        raise ValueError('Invalid manifest size')
    maximum = 64 << 20
    if any(type(v['bytes']) is not int or not 0 <= v['bytes'] <= maximum for v in expected.values()):
        raise ValueError('Invalid member sizes')
    if sum(v['bytes'] for v in expected.values()) > maximum:
        raise ValueError('Public evidence exceeds restoration limit')
    validated = {}
    with tarfile.open(args.archive, 'r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or str(path) != member.name or member.name in validated
                    or member.name not in expected or path.suffix not in ('.json', '.log', '.xml', '.txt')):
                raise ValueError('Unexpected archive member')
            spec = expected[member.name]
            if member.size != spec['bytes']:
                raise ValueError('Member size mismatch')
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != spec['sha256']:
                raise ValueError('Member hash mismatch')
            validated[member.name] = payload
    if set(validated) != set(expected):
        raise ValueError('Archive member set differs from manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in validated.items():
        output = args.output.joinpath(*PurePosixPath(name).parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(payload)
    print(json.dumps(dict(files_verified=len(validated), bytes=sum(map(len, validated.values())),
                          output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
