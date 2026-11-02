"""Read at most 1GiB of the original model with fd-local F_NOCACHE requested.

This never writes the model or purges the system cache. Device-global counter
deltas determine whether block IO actually occurred; the flag alone proves none.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .storage_counters import block_snapshot, compare_snapshots


def run(path, offset, length):
    if type(length) is not int or not 0 < length <= 1024**3 or offset < 0:
        raise ValueError('Invalid bounded read')
    stat = path.stat()
    if offset+length > stat.st_size:
        raise ValueError('Read outside source')
    payload = bytearray(8 * 1024**2)
    before = block_snapshot()
    digest = hashlib.sha256()
    intervals = []
    with path.open('rb', buffering=0) as source:
        fcntl.fcntl(source.fileno(), 48, 1)  # F_NOCACHE, verified in this macOS SDK.
        source.seek(offset)
        remaining = length
        start = time.perf_counter()
        while remaining:
            t0 = time.perf_counter()
            count = source.readinto(memoryview(payload)[:min(remaining, len(payload))])
            elapsed = time.perf_counter()-t0
            if not count:
                raise OSError('Short model read')
            digest.update(memoryview(payload)[:count])
            intervals.append({'bytes': count, 'read_seconds': elapsed})
            remaining -= count
        wall = time.perf_counter()-start
    after = block_snapshot()
    final = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (final.st_size, final.st_mtime_ns, final.st_ino):
        raise RuntimeError('Model changed during read')
    seconds = sum(r['read_seconds'] for r in intervals)
    return {'schema_version': 1, 'file': path.name, 'offset': offset, 'logical_bytes': length,
            'fd_nocache_requested': True, 'model_unchanged_file_stats': True,
            'sha256_read_range': digest.hexdigest(), 'read_call_seconds': seconds,
            'wall_including_hash_seconds': wall, 'logical_bytes_per_read_second': length/seconds,
            'chunks': intervals, 'block_device_delta': compare_snapshots(before, after),
            'scope': 'Bounded internal-file sequential component; global block IO includes other processes, not NAND cells or GLM throughput'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--offset', type=int, required=True)
    p.add_argument('--bytes', type=int, default=1024**3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite evidence')
    result = run(a.gguf, a.offset, a.bytes)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'chunks'}, indent=2))
