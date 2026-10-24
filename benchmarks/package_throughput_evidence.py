#!/usr/bin/env python3
"""Package JSON/log evidence without rewriting originals or including tensors."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    root=Path(__file__).resolve().parents[1]/'evidence-throughput'
    archive=root/'raw-json-logs.tar.gz';manifest=root/'raw-manifest.json'
    if archive.exists() or manifest.exists():raise FileExistsError('Evidence package already exists; originals are never overwritten')
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
    content={p.relative_to(root).as_posix():p.read_bytes() for p in paths}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in content.items()}
    with archive.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as tar:
                for name,data in content.items():
                    info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(data))
    binaries={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.npy')}
    result={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':hashes,
        'local_binary_capture_hashes_not_packaged':binaries,'scope':'Experiment JSON/log/text only; no model, tokenizer asset, weight tensor or user document'}
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'files':len(hashes),'archive_bytes':archive.stat().st_size,'local_binary_captures':len(binaries)}))


if __name__=='__main__':main()
