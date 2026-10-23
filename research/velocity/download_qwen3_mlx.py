#!/usr/bin/env python3
"""Download one pinned real MLX MoE checkpoint with bounded disk/RAM use.

No credentials, conversion, duplicate cache or automatic model load is used.
Every completed file is SHA256 checked against Hugging Face LFS metadata.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

MODEL = 'Qwen/Qwen3-30B-A3B-MLX-6bit'
REVISION = 'b92024737fc38f0bce249aaefb7611dd22c38b58'
GiB = 1024**3


def range_download(entry, destination, reserve, workers):
    """Concurrent bounded HTTP ranges into one file, with a restart journal."""
    target = destination/entry['rfilename']
    partial = target.with_name(target.name+'.part')
    journal = target.with_name(target.name+'.ranges.json')
    expected = entry['size']
    if journal.exists():
        state = json.loads(journal.read_text())
        if state['expected_bytes'] != expected or state['revision'] != REVISION:
            raise ValueError('Mismatched range journal')
    else:
        # An existing sequential partial is a verified contiguous received prefix.
        prefix = partial.stat().st_size if partial.exists() else 0
        state = dict(expected_bytes=expected, revision=REVISION, prefix_bytes=prefix, completed=[])
    prefix = state['prefix_bytes']
    chunk_size = 64*1024**2
    ranges = [(start, min(expected, start+chunk_size)) for start in range(prefix, expected, chunk_size)]
    completed = {tuple(x) for x in state['completed']}
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, expected)

    def save():
        temp = journal.with_name(journal.name+'.tmp')
        temp.write_text(json.dumps(state)+'\n')
        temp.replace(journal)
    save()

    def download(bounds):
        start, stop = bounds
        last_error = None
        for attempt in range(3):
            request = urllib.request.Request(
                f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{entry["rfilename"]}?download=true&range={start}&attempt={attempt}&velocity={time.time_ns()}',
                headers={'Range': f'bytes={start}-{stop-1}'})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop-1}/{expected}':
                        raise ValueError('Server did not honor the exact byte range')
                    cursor = start
                    for block in iter(lambda: response.read(1024**2), b''):
                        if cursor+len(block) > stop:
                            raise ValueError('Excess range response bytes')
                        if shutil.disk_usage(destination).free < reserve+workers*1024**2:
                            raise RuntimeError('Disk reserve reached during range download')
                        written = 0
                        while written < len(block):
                            n = os.pwrite(fd, block[written:], cursor+written)
                            if n < 1:
                                raise OSError('Short positional write')
                            written += n
                        cursor += len(block)
                    if cursor != stop:
                        raise ValueError('Incomplete range response')
                return bounds
            except (OSError, ValueError) as exc:
                last_error = exc
        raise last_error

    try:
        started = time.perf_counter()
        initial = prefix+sum(stop-start for start, stop in completed)
        count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download, bounds) for bounds in ranges if bounds not in completed]
            for future in as_completed(futures):
                bounds = future.result()
                completed.add(bounds)
                state['completed'] = sorted(completed)
                save()
                count += 1
                if count % 4 == 0 or len(completed) == len(ranges):
                    received = prefix+sum(stop-start for start, stop in completed)
                    rate = (received-initial)/max(1e-9, time.perf_counter()-started)/1024**2
                    print(f'{entry["rfilename"]}: {received}/{expected} bytes; {rate:.2f} MiB/s; HTTP workers={workers}', flush=True)
    finally:
        os.close(fd)
    return partial, journal


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--reserve-gib', type=float, default=16)
    parser.add_argument('--http-workers', type=int, default=1, choices=range(1, 17))
    args = parser.parse_args()
    if args.reserve_gib < 16:
        parser.error('At least 16 GiB of disk reserve is required')
    args.destination.mkdir(parents=True, exist_ok=True)
    api = fetch_json(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true')
    if api['sha'] != REVISION:
        raise ValueError('Pinned revision mismatch')
    wanted = [entry for entry in api['siblings'] if entry['rfilename'].endswith(
        ('.safetensors', '.json', '.jinja', '.model', '.txt')) or entry['rfilename'] in ('LICENSE', 'README.md')]
    total = sum(entry['size'] for entry in wanted)
    remaining = sum(max(0, entry['size'] - (args.destination/entry['rfilename']).stat().st_size)
                    if (args.destination/entry['rfilename']).is_file() else entry['size'] for entry in wanted)
    free = shutil.disk_usage(args.destination).free
    reserve = int(args.reserve_gib*GiB)
    if free < remaining + reserve:
        raise RuntimeError(f'Disk reserve failed: {free=}, {remaining=}, {reserve=}')
    report = dict(model_id=MODEL, revision=REVISION, weight_file_bytes=sum(
        e['size'] for e in wanted if e['rfilename'].endswith('.safetensors')),
        requested_bytes=total, disk_free_before_bytes=free, reserve_bytes=reserve,
        initial_remaining_bytes=remaining, files=[], started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        complete=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    save()
    print(json.dumps({key: report[key] for key in ('model_id', 'weight_file_bytes', 'initial_remaining_bytes', 'disk_free_before_bytes', 'reserve_bytes')}), flush=True)
    # Metadata first, so architecture/shape inspection need not await all weights.
    wanted.sort(key=lambda e: (e['rfilename'].endswith('.safetensors'), e['rfilename']))
    for entry in wanted:
        name, expected = entry['rfilename'], entry['size']
        target = args.destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name+'.part')
        sha = hashlib.sha256()
        start = time.perf_counter()
        if target.exists():
            if target.stat().st_size != expected:
                raise ValueError(f'Existing completed file has wrong size: {target}')
            with target.open('rb') as handle:
                for chunk in iter(lambda: handle.read(4*1024**2), b''):
                    sha.update(chunk)
        elif args.http_workers > 1 and name.endswith('.safetensors'):
            partial, journal = range_download(entry, args.destination, reserve, args.http_workers)
            with partial.open('rb') as handle:
                for chunk in iter(lambda: handle.read(4*1024**2), b''):
                    sha.update(chunk)
            expected_sha = entry.get('lfs', {}).get('sha256')
            if not expected_sha or sha.hexdigest() != expected_sha:
                raise ValueError(f'Hash mismatch or absent LFS digest: {name}')
            partial.rename(target)
            journal.unlink()
        else:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise ValueError(f'Oversized partial file: {partial}')
            if offset:
                with partial.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(4*1024**2), b''):
                        sha.update(chunk)
            request = urllib.request.Request(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true&velocity={time.time_ns()}')
            if offset:
                request.add_header('Range', f'bytes={offset}-')
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise ValueError(f'Server ignored resume range for {name}')
                with partial.open('ab' if offset else 'wb') as handle:
                    downloaded = offset
                    next_log = ((offset//GiB)+1)*GiB
                    for chunk in iter(lambda: response.read(4*1024**2), b''):
                        if shutil.disk_usage(args.destination).free < reserve+len(chunk):
                            raise RuntimeError('Disk reserve reached during download')
                        if downloaded+len(chunk)>expected:
                            raise ValueError(f'Received excess bytes for {name}')
                        handle.write(chunk)
                        sha.update(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_log:
                            print(f'{name}: {downloaded}/{expected} bytes', flush=True)
                            next_log += GiB
            if partial.stat().st_size != expected:
                raise ValueError(f'Incomplete download: {name}')
            expected_sha = entry.get('lfs', {}).get('sha256')
            if expected_sha and sha.hexdigest() != expected_sha:
                raise ValueError(f'Hash mismatch: {name}')
            partial.rename(target)
        digest = sha.hexdigest()
        expected_sha = entry.get('lfs', {}).get('sha256')
        if expected_sha and digest != expected_sha:
            raise ValueError(f'Hash mismatch: {name}')
        report['files'].append(dict(name=name, size=expected, sha256=digest,
            expected_lfs_sha256=expected_sha, verified_lfs_sha256=bool(expected_sha), seconds=time.perf_counter()-start))
        save()
        print(f'Complete: {name} ({expected} bytes)', flush=True)
    report['complete'] = True
    report['disk_free_after_bytes'] = shutil.disk_usage(args.destination).free
    save()


if __name__ == '__main__':
    main()
