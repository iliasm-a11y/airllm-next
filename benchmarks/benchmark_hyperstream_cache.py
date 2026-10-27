"""Synthetic cache-history memory probe; never imports MLX or a model.

Compare the cache source at an immutable Git revision with the current source.
The one-byte fake values isolate Python observation metadata. Tracemalloc is
enabled for every row, so elapsed time is diagnostic, not an inference claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import tracemalloc
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('air_llm/airllm/nextgen/cache.py')
DEFAULT_BASELINE = 'ca3a42935be876bbdbbc1e2adbd16c8547746175'


def _git(*arguments):
    return subprocess.check_output(['git', '-C', str(ROOT), *arguments], text=True).strip()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _source_manifest():
    paths = (ROOT / SOURCE, Path(__file__).resolve())
    return {path.relative_to(ROOT).as_posix(): _digest(path.read_bytes()) for path in paths}


def _positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Request counts must be positive integers')
    return number


def _output_path(value):
    path = Path(value).expanduser()
    if os.path.lexists(path):
        raise ValueError('Output already exists; choose a new output path')
    resolved = path.resolve()
    protected = (ROOT / SOURCE, Path(__file__).resolve())
    if resolved in protected or resolved.is_relative_to(ROOT / '.git'):
        raise ValueError('Output must not replace source inputs or Git metadata')
    return resolved


def measure(source, label, requests):
    module = ModuleType('measured_cache_' + label)
    exec(compile(source, label + '_cache.py', 'exec'), module.__dict__)
    gc.collect()
    tracemalloc.start()
    try:
        cache = module.WeightCache(2, 'lru')
        payload = SimpleNamespace(nbytes=1)
        started = perf_counter()
        for number in range(requests):
            key = 'expert-tile-' + str(number)
            if cache.get(key, 1) is not None:
                raise AssertionError('Unique demand unexpectedly hit the cache')
            cache.record_load(key, .001)
            if not cache.put(key, payload):
                raise AssertionError('One-byte value unexpectedly rejected')
        elapsed = perf_counter() - started
        live, peak = tracemalloc.get_traced_memory()
        return {'variant': label, 'requests': requests,
                'source_sha256': _digest(source.encode()),
                'resident_entries': len(cache.data), 'resident_payload_bytes': cache.bytes,
                'frequency_entries': len(cache.freq), 'cost_entries': len(cache.costs),
                'request_trace_entries': len(cache.trace), 'cost_trace_entries': len(cache.cost_trace),
                'load_trace_entries': len(cache.load_trace), 'stats': dict(cache.stats),
                'python_traced_live_bytes': live, 'python_traced_peak_bytes': peak,
                'seconds_with_tracemalloc': elapsed}
    finally:
        tracemalloc.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default=DEFAULT_BASELINE)
    parser.add_argument('--output', required=True, help='New JSON file; existing paths are refused')
    parser.add_argument('--requests', type=_positive_integer, nargs='+', default=[100_000, 200_000])
    args = parser.parse_args(argv)
    try:
        output = _output_path(args.output)
        # Resolve to a commit before constructing a blob selector. No user ref
        # can become a Git option or an alternate source path.
        baseline_commit = _git('rev-parse', '--verify', '--end-of-options', args.baseline_ref + '^{commit}')
        baseline = subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', baseline_commit + ':' + SOURCE.as_posix()], text=True)
        before = _source_manifest()
        candidate = (ROOT / SOURCE).read_text()
        if _digest(candidate.encode()) != before[SOURCE.as_posix()]:
            raise RuntimeError('Candidate source changed while being read')
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    rows = []
    for index, count in enumerate(args.requests):
        variants = [('baseline', baseline), ('candidate', candidate)]
        if index % 2:
            variants.reverse()
        for label, source in variants:
            rows.append(measure(source, label, count))
    after = _source_manifest()
    unchanged = before == after
    report = {
        'schema_version': 1, 'kind': 'synthetic_cache_metadata_memory_probe',
        'status': 'complete' if unchanged else 'interrupted_sources_changed',
        'model_inference_executed': False, 'python': sys.version, 'platform': platform.platform(),
        'baseline_commit': baseline_commit, 'baseline_source_sha256': _digest(baseline.encode()),
        'source_manifest_before': before, 'source_manifest_after': after,
        'sources_unchanged': unchanged,
        'protocol': 'One-byte fake payload, LRU capacity two, unique demand keys and successful load-cost observations; fresh cache and tracemalloc per row; alternate variant order.',
        'limits': ['Synthetic metadata churn, not model throughput or RSS.',
                   'One measurement per variant and request count.',
                   'Timing includes tracemalloc and is not used as a throughput claim.',
                   'No GPU, tensor payload, physical storage, or model generation measured.'],
        'measurements': rows,
    }
    encoded = json.dumps(report, indent=2) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwriting a path created during the
    # measurement, including an existing symlink or source input.
    with output.open('x') as stream:
        stream.write(encoded)
    print(json.dumps({'output': str(output), 'status': report['status'], 'rows': len(rows)}))
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
