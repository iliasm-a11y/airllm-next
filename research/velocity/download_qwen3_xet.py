#!/usr/bin/env python3
"""Pinned Xet alternative, direct-to-file, no duplicated Hub model cache.

Existing HTTP range partials are preserved and never mistaken for complete
files. Run this only while the HTTP downloader is paused/stopped. This process
has disk/RSS/system-memory watchdogs and never opens model weights in MLX.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from time import perf_counter

from download_qwen3_mlx import MODEL, REVISION, GiB, fetch_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--only-shard', type=int, choices=range(1, 6))
    p.add_argument('--max-rss-mib', type=int, default=1024)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    # These variables affect only this process, not the user's global settings.
    os.environ['HF_XET_CACHE'] = str(args.destination.parent/'xet-cache')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.file_download import xet_get
    import psutil

    api = fetch_json(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true')
    shards = [entry for entry in api['siblings'] if entry['rfilename'].endswith('.safetensors')]
    # Try an untouched shard first so the HTTP reconstruction is preserved.
    shards.sort(key=lambda e: ('00001-of' in e['rfilename'], e['rfilename']))
    if args.only_shard is not None:
        shards = [entry for entry in shards if entry['rfilename'].startswith(f'model-{args.only_shard:05d}-')]
    remaining = sum(entry['size'] for entry in shards if not (args.destination/entry['rfilename']).is_file())
    if shutil.disk_usage(args.destination).free < remaining+16*GiB:
        raise RuntimeError('Xet download would exceed the 16GiB disk reserve')
    report = dict(model_id=MODEL, revision=REVISION, transport='hf_xet direct-to-file',
        status='started', files=[], rss_peak_bytes=0,
        min_disk_free_bytes=shutil.disk_usage(args.destination).free)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    save()
    stop = threading.Event()
    process = psutil.Process()

    def watchdog():
        while not stop.wait(.2):
            rss = process.memory_info().rss
            free = shutil.disk_usage(args.destination).free
            report['rss_peak_bytes'] = max(report['rss_peak_bytes'], rss)
            report['min_disk_free_bytes'] = min(report['min_disk_free_bytes'], free)
            if rss > args.max_rss_mib*1024**2 or free < 16*GiB or psutil.virtual_memory().available < 1.5*GiB:
                report.update(status='resource_guard_failed', rss_bytes=rss, disk_free_bytes=free)
                save()
                os._exit(75)
    monitor = threading.Thread(target=watchdog, daemon=True)
    monitor.start()
    try:
        for entry in shards:
            name = entry['rfilename']
            target = args.destination/name
            incomplete = target.with_name(target.name+'.xet.part')
            started = perf_counter()
            if not target.exists():
                metadata = get_hf_file_metadata(hf_hub_url(MODEL, filename=name, revision=REVISION), token=False)
                if metadata.commit_hash != REVISION or metadata.size != entry['size'] or metadata.xet_file_data is None:
                    raise ValueError('Pinned Xet metadata mismatch or unavailable Xet transport')
                print(f'Xet start: {name}, {entry["size"]} bytes', flush=True)
                xet_get(incomplete_path=incomplete, xet_file_data=metadata.xet_file_data,
                    headers={}, expected_size=entry['size'], displayed_filename=name)
                candidate = incomplete
            else:
                candidate = target
            if candidate.stat().st_size != entry['size']:
                raise ValueError('Incomplete Xet file')
            digest = hashlib.sha256()
            with candidate.open('rb') as handle:
                for block in iter(lambda: handle.read(4*1024**2), b''):
                    digest.update(block)
            expected_sha = entry.get('lfs', {}).get('sha256')
            if not expected_sha or digest.hexdigest() != expected_sha:
                raise ValueError('Xet completed file failed canonical LFS SHA256')
            if candidate != target:
                candidate.rename(target)
            seconds = perf_counter()-started
            record = dict(name=name, size=entry['size'], sha256=digest.hexdigest(),
                expected_lfs_sha256=expected_sha, verified_lfs_sha256=True,
                seconds=seconds, effective_MiB_per_second=entry['size']/seconds/1024**2)
            report['files'].append(record)
            save()
            print(json.dumps(record), flush=True)
            # Merge only verified completions into the canonical manifest.
            manifest = json.loads(args.manifest.read_text())
            manifest['files'] = [item for item in manifest['files'] if item['name'] != name]+[record]
            completed = {item['name'] for item in manifest['files'] if item.get('verified_lfs_sha256')}
            all_shards = {f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)}
            manifest['complete'] = all_shards.issubset(completed)
            manifest['disk_free_after_bytes'] = shutil.disk_usage(args.destination).free
            args.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop.set()
        monitor.join(timeout=1)
        save()


if __name__ == '__main__':
    main()
