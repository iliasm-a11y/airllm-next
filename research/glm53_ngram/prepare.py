"""Read-only cache advice after an engine exits; never a model admission bypass.

No native model is initialized and no weight payload is traversed. Darwin may
make clean file-backed pages reclaimable. This is an advisory preparation step,
not a cold-cache guarantee or a measurement of physical RAM freed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import mmap
from pathlib import Path
import sys
import time

import psutil

from airllm.nextgen.experiment_guard import system_snapshot
from airllm.nextgen.glm_native import FILES, _identity, inspect_native_checkpoint
from research.glm53_batched_prefill.uid_supervisor import acquire_uid_lock
from research.glm53_ultra_decode.model_profile import REQUIRED_MEMORY


def prepare(model, receipt, *, clock=time.monotonic, sleep=time.sleep):
    if sys.platform != 'darwin' or not hasattr(mmap.mmap, 'madvise'):
        raise ValueError('this preparation is scoped to the measured Darwin environment')
    before = system_snapshot()
    if before['pressure_level'] != 1 or before['available_bytes'] < (3<<30)+(64<<20):
        raise MemoryError('metadata/cache preparation requires normal pressure and its small workspace reserve')
    inspection = inspect_native_checkpoint(model, receipt)
    if not inspection['integrity_verified'] or not inspection['architecture_matches']:
        raise ValueError('original checkpoint integrity must be verified before cache preparation')
    holder, lock = acquire_uid_lock()
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'before': before,
              'uid_lock': lock, 'advice': 'MADV_DONTNEED', 'files': [],
              'model_inference_started': False, 'performance_measured': False,
              'scope': 'read-only file-backed cache advice; no payload traversal; no cold-cache or causal RAM-release claim'}
    try:
        for entry in FILES:
            name = entry[0]
            path = model/name
            identity = _identity(path)
            rss = psutil.Process().memory_info().rss
            with path.open('rb') as stream:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                    mapping.madvise(mmap.MADV_DONTNEED)
            if _identity(path) != identity:
                raise RuntimeError('checkpoint identity changed during read-only cache preparation')
            result['files'].append({'name': name, 'read_only': True, 'identity_unchanged': True,
                                    'self_rss_before': rss, 'self_rss_after': psutil.Process().memory_info().rss})
        # A cache hint is not an admission result. Wait for a short stable
        # measured window before spawning a wrapper; never retry a model.
        start = clock()
        stable_since = None
        result['settle_samples'] = []
        while True:
            now = clock()
            sample = system_snapshot()
            result['settle_samples'].append({'elapsed_seconds': now-start, **sample})
            ready = sample['pressure_level'] == 1 and sample['available_bytes'] >= REQUIRED_MEMORY
            if ready:
                if stable_since is None:
                    stable_since = now
                if now-stable_since >= 2:
                    break
            else:
                stable_since = None
            if now-start >= 60:
                raise MemoryError('no stable unchanged5.5GiB admission window within60s; no model started')
            sleep(.5)
    finally:
        holder.close()
    result['after'] = system_snapshot()
    result['global_available_change_bytes'] = result['after']['available_bytes']-before['available_bytes']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'receipt', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.model.resolve()):
        raise ValueError('a fresh output outside the model is required')
    result = prepare(a.model, a.receipt)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'before': result['before'], 'after': result['after'], 'native_model_loaded': False}))
