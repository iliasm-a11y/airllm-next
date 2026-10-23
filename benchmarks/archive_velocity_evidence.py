#!/usr/bin/env python3
"""Losslessly archive or restore local Velocity JSON with SHA256 checks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'evidence-velocity'


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore',action='store_true')
    args=p.parse_args()
    manifest=ROOT/'MANIFEST.json'
    if args.restore:
        index=json.loads(manifest.read_text())
        for row in index['files']:
            archive=(ROOT/row['archive']).resolve()
            destination=(ROOT/row['name']).resolve()
            if not archive.is_relative_to(ROOT.resolve()) or not destination.is_relative_to(ROOT.resolve()):
                raise ValueError('Evidence path escapes its root')
            packed=archive.read_bytes()
            if digest(packed)!=row['archive_sha256']:
                raise ValueError('Compressed archive identity mismatch: '+row['name'])
            raw=gzip.decompress(packed)
            if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:
                raise ValueError('Archive identity mismatch: '+row['name'])
            if destination.exists() and digest(destination.read_bytes())!=row['sha256']:
                raise ValueError('Existing evidence differs; no overwrite: '+str(destination))
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(raw)
    else:
        target=ROOT/'published';target.mkdir(exist_ok=True)
        rows=[]
        for path in sorted(ROOT.rglob('*.json')):
            if path==manifest or 'published' in path.relative_to(ROOT).parts:continue
            name=str(path.relative_to(ROOT));raw=path.read_bytes()
            archive=target/(name.replace('/','__')+'.gz')
            archive.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
            rows.append(dict(name=name,bytes=len(raw),sha256=digest(raw),
                archive=str(archive.relative_to(ROOT)),archive_sha256=digest(archive.read_bytes())))
        manifest.write_text(json.dumps(dict(schema_version=1,files=rows,
            scope='Raw numerical/measurement JSON; no checkpoint weights or corpus text files'),indent=2)+'\n')
    print(json.dumps(dict(restore=args.restore,files=len(json.loads(manifest.read_text())['files']))))


if __name__=='__main__':main()
