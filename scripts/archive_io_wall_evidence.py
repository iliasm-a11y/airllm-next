#!/usr/bin/env python3
"""Archive local I/O-wall evidence without breaking original benchmark paths.

NPZ payloads are content-addressed in the sibling outputs/raw-evidence folder.
Completed originals are atomically replaced by hard links where supported; on
failure they remain intact. Treat every content-addressed payload and its links
as immutable evidence. XML originals are preserved alongside deterministic gzip.
Run after benchmark and test writers have finished. No NumPy data is loaded.
"""
import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4


MiB = 1024 ** 2


def identity(path):
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def stable_digest(path):
    before = identity(path)
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(MiB), b''):
            digest.update(block)
    if identity(path) != before:
        raise RuntimeError(f'Evidence changed during hashing; retry after writers finish: {path}')
    return digest.hexdigest(), before


def atomic_json(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(data, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_canonical(source, canonical, digest, before):
    if canonical.exists():
        actual, existing = stable_digest(canonical)
        if actual != digest or existing[2] != before[2]:
            raise ValueError(f'Content-addressed archive collision or corruption: {canonical}')
        return
    try:
        os.link(source, canonical)
    except FileExistsError:
        return install_canonical(source, canonical, digest, before)
    except OSError:
        descriptor, temporary = tempfile.mkstemp(prefix='.' + canonical.name + '-', dir=canonical.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(source, temporary)
            actual, copied = stable_digest(Path(temporary))
            if actual != digest or copied[2] != before[2] or identity(source) != before:
                raise RuntimeError(f'Evidence changed during archival copy: {source}')
            os.replace(temporary, canonical)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if identity(source) != before or stable_digest(canonical)[0] != digest:
        raise RuntimeError(f'Evidence changed during canonical archival: {source}')


def relink_original(source, canonical, before):
    if source.is_symlink():
        return 'preserved_symlink', None
    if os.path.samestat(source.stat(), canonical.stat()):
        return 'already_shared_inode', None
    temporary = source.with_name('.' + source.name + '-relink-' + uuid4().hex)
    try:
        os.link(canonical, temporary)
        if identity(source) != before:
            raise RuntimeError(f'Evidence changed before deduplication: {source}')
        os.replace(temporary, source)
        return 'relinked_atomically', None
    except OSError as error:
        # An unsupported or cross-device link never removes the original.
        return 'preserved_original', f'{type(error).__name__}: {error}'
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_gzip(source):
    before = identity(source)
    destination = source.with_suffix(source.suffix + '.gz')
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '-', dir=source.parent)
    try:
        with os.fdopen(descriptor, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as compressed:
                with source.open('rb') as original:
                    shutil.copyfileobj(original, compressed, length=MiB)
            raw.flush()
            os.fsync(raw.fileno())
        if identity(source) != before:
            raise RuntimeError(f'XML changed during compression; retry after test writers finish: {source}')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def archive(repo_root, *, compress_xml=True):
    repo_root = Path(repo_root).resolve()
    evidence = repo_root / 'evidence-io-wall'
    if not evidence.is_dir():
        raise FileNotFoundError(evidence)
    output_root = repo_root.parent
    canonical_root = output_root / 'raw-evidence'
    canonical_root.mkdir(exist_ok=True)
    manifest_path = evidence / 'logits-manifest.json'
    payloads = []
    for source in sorted(evidence.rglob('*.npz')):
        if not source.is_file():
            continue
        digest, before = stable_digest(source)
        canonical = canonical_root / (digest + '.npz')
        install_canonical(source, canonical, digest, before)
        status, error = relink_original(source, canonical, before)
        # Verify original paths remain readable and byte-identical after relink.
        if stable_digest(source)[0] != digest:
            raise RuntimeError(f'Original evidence verification failed: {source}')
        payloads.append({'original_path': source.relative_to(repo_root).as_posix(),
                         'sha256': digest, 'size_bytes': before[2],
                         'canonical_output_relative_path': canonical.relative_to(output_root).as_posix(),
                         'deduplication_status': status, 'deduplication_error': error})
    xml_archives = []
    if compress_xml:
        for source in sorted(evidence.rglob('*.xml')):
            if source.is_symlink() or not source.is_file():
                continue
            destination = deterministic_gzip(source)
            digest, before = stable_digest(destination)
            xml_archives.append({'original_path': source.relative_to(repo_root).as_posix(),
                'gzip_path': destination.relative_to(repo_root).as_posix(),
                'gzip_sha256': digest, 'gzip_size_bytes': before[2],
                'original_preserved': True, 'gzip_mtime': 0})
    inventory = []
    for source in sorted(evidence.rglob('*')):
        if source == manifest_path or not source.is_file():
            continue
        digest, before = stable_digest(source)
        inventory.append({'path': source.relative_to(repo_root).as_posix(),
                          'sha256': digest, 'size_bytes': before[2]})
    unique = {entry['sha256']: entry['size_bytes'] for entry in payloads}
    unique_size = sum(unique.values())
    original_size = sum(entry['size_bytes'] for entry in payloads)
    summary = {'npz_original_count': len(payloads), 'npz_unique_count': len(unique),
               'npz_original_logical_bytes': original_size,
               'npz_unique_logical_bytes': unique_size,
               'npz_duplicate_logical_bytes': original_size - unique_size,
               'npz_preserved_without_link_count': sum(entry['deduplication_status'] in
                   ('preserved_original', 'preserved_symlink') for entry in payloads),
               'xml_gzip_count': len(xml_archives),
               'xml_gzip_total_bytes': sum(entry['gzip_size_bytes'] for entry in xml_archives),
               'evidence_inventory_count': len(inventory)}
    manifest = {'schema_version': 1, 'summary': summary,
                'path_roots': {'original_paths': 'repository root',
                               'canonical_output_relative_paths': 'repository parent output directory'},
                'scope': 'all local NPZ evidence including failed attempts; all original paths preserved; no raw values in manifest',
                'deduplication_scope': 'logical identical bytes; filesystem physical-space savings are not measured',
                'immutable_payload_note': 'canonical payloads and original hard links share content; preserve them as immutable evidence',
                'inventory_excludes': ['evidence-io-wall/logits-manifest.json'],
                'payloads': payloads, 'xml_archives': xml_archives, 'evidence_files': inventory}
    atomic_json(manifest_path, manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--skip-xml', action='store_true', help='Preserve XML unchanged while test writers may still be active')
    args = parser.parse_args()
    print(json.dumps(archive(args.repo_root, compress_xml=not args.skip_xml), indent=2))


if __name__ == '__main__':
    main()
