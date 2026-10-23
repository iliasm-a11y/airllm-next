#!/usr/bin/env python3
"""Bounded SHA256 of an installed GGUF with macOS uncached reads and guards."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = dict(file=str(args.file), expected_sha256=args.expected_sha256,
        file_bytes=args.file.stat().st_size, complete=False, read_bytes=0,
        io_buffer_bytes=4*1024**2, minimum_available_bytes=psutil.virtual_memory().available,
        rss_peak_bytes=0, swap_growth_peak_bytes=0, uncached_fd=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start, swap = perf_counter(), psutil.swap_memory().used
    process = psutil.Process()
    digest, buffer = hashlib.sha256(), bytearray(report['io_buffer_bytes'])
    try:
        with args.file.open('rb', buffering=0) as handle:
            if sys.platform == 'darwin':
                fcntl.fcntl(handle.fileno(), 48, 1)  # Darwin F_NOCACHE; private fd.
                report['uncached_fd'] = True
            while True:
                available = psutil.virtual_memory().available
                growth = max(0, psutil.swap_memory().used-swap)
                report['minimum_available_bytes'] = min(report['minimum_available_bytes'], available)
                report['swap_growth_peak_bytes'] = max(report['swap_growth_peak_bytes'], growth)
                report['rss_peak_bytes'] = max(report['rss_peak_bytes'], process.memory_info().rss)
                if available < 1.5*1024**3 or growth > 256*1024**2:
                    raise MemoryError('System memory/swap guard reached during bounded hash')
                count = handle.readinto(buffer)
                if not count:
                    break
                digest.update(memoryview(buffer)[:count])
                report['read_bytes'] += count
        report['sha256'] = digest.hexdigest()
        if report['sha256'] != args.expected_sha256:
            raise ValueError('Complete installed GGUF SHA256 mismatch')
        report['complete'] = True
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['seconds'] = perf_counter()-start
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
