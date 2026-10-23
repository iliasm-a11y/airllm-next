#!/usr/bin/env python3
"""Explicit, bounded download of one immutable public trained MoE checkpoint.

Only this IBM revision is selected. Ranges are bounded, independently checked,
restartable, and the complete payload must match its official LFS SHA256 before
it becomes model.safetensors. No authentication token or paid inference is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

MODEL_ID = 'ibm-granite/granite-3.0-1b-a400m-instruct'
REVISION = 'ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445'
WEIGHT_BYTES = 2669283096
WEIGHT_SHA256 = 'f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode(), usedforsecurity=False)
    with path.open('rb') as stream:
        while data := stream.read(1024**2):
            digest.update(data)
    return digest.hexdigest()


def verify_file(path, entry):
    checksum = sha256(path)
    if path.stat().st_size != entry.size:
        raise ValueError(f'File size mismatch: {entry.rfilename}')
    blob = None if entry.lfs else git_blob_sha1(path)
    if entry.lfs and checksum != entry.lfs['sha256'] or not entry.lfs and blob != entry.blob_id:
        raise ValueError(f'File content identity mismatch: {entry.rfilename}')
    return {'name':entry.rfilename, 'bytes':path.stat().st_size, 'sha256':checksum,
            'hub_lfs_sha256':entry.lfs['sha256'] if entry.lfs else None,
            'git_blob_sha1':blob, 'hub_git_blob_sha1':entry.blob_id}


def reusable_range(fd, marker, start, end):
    try:
        saved = json.loads(marker.read_text())
        if not isinstance(saved,dict) or saved.get('start') != start or saved.get('end') != end:
            return False
        existing = os.pread(fd, end-start+1, start)
        return len(existing) == end-start+1 and hashlib.sha256(existing).hexdigest() == saved.get('sha256')
    except (OSError,ValueError,TypeError):
        return False


def write_marker(marker, values):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=marker.parent, prefix=marker.name+'.tmp-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(values,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,marker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--workers', type=int, choices=range(1,9), default=8)
    parser.add_argument('--chunk-mib', type=int, choices=[8,16,32,64], default=32)
    parser.add_argument('--min-free-gib', type=float, default=20.)
    args = parser.parse_args()
    if not 0 <= args.min_free_gib < float('inf'):
        parser.error('min-free-gib must be finite and nonnegative')
    from huggingface_hub import HfApi, snapshot_download
    root = Path(args.destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root/'model.safetensors'
    if target.exists() and sha256(target) != WEIGHT_SHA256:
        raise ValueError('Existing model.safetensors has another identity; use a new destination')
    info = HfApi(token=False).model_info(MODEL_ID, revision=REVISION, files_metadata=True, token=False)
    files = [f for f in info.siblings if f.rfilename.endswith(('.json','.txt','.safetensors')) or f.rfilename == 'README.md']
    payload = sum(f.size for f in files)
    weight = next(f for f in files if f.rfilename == 'model.safetensors')
    if info.sha != REVISION or info.gated or weight.size != WEIGHT_BYTES or weight.lfs['sha256'] != WEIGHT_SHA256:
        raise ValueError('Public pinned repository metadata differs from expected identity')
    if payload > 4*1024**3 or shutil.disk_usage(root).free-payload < args.min_free_gib*1024**3:
        raise OSError('Download would exceed 4GiB payload or remaining disk reserve')
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=root,
        allow_patterns=[f.rfilename for f in files if f.rfilename != 'model.safetensors'], max_workers=2, token=False)
    if not target.exists() or sha256(target) != WEIGHT_SHA256:
        incomplete = root/'model.safetensors.direct-incomplete'
        markers = root/'.direct-range-markers'
        markers.mkdir(exist_ok=True)
        chunk = args.chunk_mib*1024**2
        fd = os.open(incomplete, os.O_CREAT|os.O_RDWR, 0o600)
        os.ftruncate(fd, WEIGHT_BYTES)
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.safetensors?download=true'
        def fetch(start):
            end = min(WEIGHT_BYTES, start+chunk)-1
            marker = markers/f'{start}-{end}'
            if reusable_range(fd,marker,start,end):
                return end-start+1
            for attempt in range(4):
                try:
                    request = Request(url+'&segment='+str(start), headers={'Range':f'bytes={start}-{end}'})
                    with urlopen(request, timeout=90) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{WEIGHT_BYTES}':
                            raise ValueError('Unexpected response range')
                        offset, checksum = start, hashlib.sha256()
                        while offset <= end:
                            data = response.read(min(1024**2, end-offset+1))
                            if not data:
                                raise OSError('Truncated range')
                            checksum.update(data)
                            if os.pwrite(fd, data, offset) != len(data):
                                raise OSError('Incomplete file write')
                            offset += len(data)
                        if response.read(1):
                            raise ValueError('Excess range payload')
                    write_marker(marker, {'sha256':checksum.hexdigest(), 'start':start, 'end':end})
                    return end-start+1
                except (OSError,ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(2+3*attempt)
        try:
            completed, begin = 0, time.monotonic()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch, start) for start in range(0, WEIGHT_BYTES, chunk)]
                for future in as_completed(futures):
                    completed += future.result()
                    print(json.dumps({'completed_bytes':completed,'total_bytes':WEIGHT_BYTES,
                        'elapsed_seconds':time.monotonic()-begin}), flush=True)
            os.fsync(fd)
        finally:
            os.close(fd)
        if sha256(incomplete) != WEIGHT_SHA256:
            raise ValueError('Full checkpoint LFS SHA256 mismatch; incomplete file preserved')
        incomplete.replace(target)
    provenance = {'model_id':MODEL_ID, 'revision':REVISION,
        'source':f'https://huggingface.co/{MODEL_ID}/tree/{REVISION}', 'gated':info.gated,
        'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selected_payload_bytes':payload, 'files':[]}
    for entry in files:
        file = root/entry.rfilename
        provenance['files'].append(verify_file(file,entry))
    (root/'download-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'complete':True,'model_directory':str(root),'checkpoint_sha256':WEIGHT_SHA256}),flush=True)


if __name__ == '__main__':
    main()
