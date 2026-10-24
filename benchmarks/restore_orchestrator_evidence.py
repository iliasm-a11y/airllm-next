"""Restore only manifest-listed regular evidence files after checking hashes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    archive, destination = Path(args.archive), Path(args.destination)
    manifest = json.loads(Path(args.manifest).read_text())
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['archive_sha256']:
            raise ValueError('Archive checksum mismatch')
    files = manifest['files']
    if len(files) > 10000 or sum(v['bytes'] for v in files.values()) > 64<<20:
        raise ValueError('Evidence exceeds bounded restore size')
    for name, record in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name or not name:
            raise ValueError('Invalid evidence path')
        if type(record['bytes']) is not int or record['bytes'] < 0:
            raise ValueError('Invalid evidence file size')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Restore destination must be fresh')
    destination.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as stream:
        seen = set()
        for member in stream:
            if not member.isfile() or member.name not in files or member.name in seen:
                raise ValueError('Unexpected archive member')
            record = files[member.name]
            if member.size != record['bytes']:
                raise ValueError('Member size mismatch')
            payload = stream.extractfile(member).read(record['bytes']+1)
            if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError('Evidence checksum mismatch')
            target = destination/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out:
                out.write(payload)
            seen.add(member.name)
        if seen != set(files):
            raise ValueError('Missing archive member')
    print(json.dumps({'verified_files':len(seen), 'restored':str(destination)}))


if __name__ == '__main__':
    main()
