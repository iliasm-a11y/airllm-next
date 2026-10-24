#!/usr/bin/env python3
"""Restore hash-verified public evidence into a NEW directory, without weights."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile


def main():
    root=Path(__file__).resolve().parents[1]/'evidence-throughput'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=root/'raw-json-logs.tar.gz')
    p.add_argument('--manifest',type=Path,default=root/'raw-manifest.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());raw=a.archive.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['archive_sha256']:raise ValueError('Archive SHA-256 mismatch')
    if a.output.exists():raise FileExistsError('Restore target must be a new directory')
    expected=manifest['files'];seen=set();total=0
    a.output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(a.archive,'r:gz') as archive:
        for member in archive:
            rel=PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected or unsafe archive member')
            if member.size>16<<20:raise ValueError('Evidence member exceeds 16 MiB bound')
            total+=member.size
            if total>64<<20:raise ValueError('Total evidence exceeds 64 MiB bound')
            stream=archive.extractfile(member)
            if stream is None:raise ValueError('Missing archive payload')
            data=stream.read()
            if hashlib.sha256(data).hexdigest()!=expected[member.name]:raise ValueError('Evidence SHA mismatch: '+member.name)
            path=a.output.joinpath(*rel.parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);seen.add(member.name)
    if seen!=set(expected):raise ValueError('Missing evidence members')
    print(json.dumps({'restored_files':len(seen),'uncompressed_bytes':total,'all_sha256_verified':True}))


if __name__=='__main__':main()
