#!/usr/bin/env python3
"""Reproduce predictor metadata cost and real-file route replay, without MLX.

The route replay reads a deterministic 8 MiB local fixture with os.pread. It
does not flush the OS cache or measure SSD throughput, and is not a trained
model benchmark. No sleeps, injected read latency, or simulated speedups enter
the timings. Pass a saved phase-1 specprefetch.py with --baseline-source.
"""
import argparse
import cProfile
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import random
import statistics
import sys
import tempfile
from time import perf_counter, process_time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))
from airllm.nextgen.cache import SingleFlight
from airllm.nextgen.specprefetch import SpecPrefetch


def load_control(path):
    name = 'airllm.nextgen._specprefetch_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.SpecPrefetch


def metadata_trial(predictor_type, edge_limit, calls, profile=False):
    io_scheduler = SingleFlight(0)
    predictor = predictor_type(io_scheduler, 0, max_transitions=edge_limit)
    try:
        # Exactly 128 layers x 32 transitions; varying edge caps makes the
        # global scan cost visible without submitting any IO.
        for layer in range(128):
            for expert in (*range(32), 0):
                predictor.observe(layer, [expert])
        start = perf_counter()
        cpu_start = process_time()
        profiler = cProfile.Profile() if profile else None
        if profiler:
            profiler.enable()
        for step in range(calls):
            predictor.predict(step % 128)
        if profiler:
            profiler.disable()
        wall_seconds = perf_counter() - start
        cpu_seconds = process_time() - cpu_start
        output = dict(edge_limit=edge_limit, calls=calls, wall_seconds=wall_seconds,
                      cpu_seconds=cpu_seconds, metrics=predictor.metrics())
        if profiler:
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('tottime').print_stats(15)
            output['cprofile_top15'] = text.getvalue()
        return output
    finally:
        predictor.close()
        io_scheduler.close()


def route_trial(predictor_type, path, routes, *, gate=True):
    expert_bytes = 256 * 1024
    io_scheduler = SingleFlight(4 * expert_bytes, 2)
    kwargs = {'min_accuracy': 0} if not gate else {}
    predictor = predictor_type(io_scheduler, 2 * expert_bytes, **kwargs)
    read_calls = []
    fd = os.open(path, os.O_RDONLY)
    digest = sha256()

    def read(expert):
        value = os.pread(fd, expert_bytes, expert * expert_bytes)
        if len(value) != expert_bytes:
            raise EOFError('Fixture truncated')
        read_calls.append(expert)
        return np.frombuffer(value, dtype=np.uint8)

    try:
        start = perf_counter()
        cpu_start = process_time()
        for actual in routes:
            for predicted in predictor.predict(0):
                predictor.submit(0, predicted, predicted, expert_bytes,
                                 lambda expert=predicted: read(expert))
            predictor.observe(0, [actual])
            value = predictor.take(actual)
            if value is None:
                value = io_scheduler.take(actual)
            if value is None:
                if not io_scheduler.submit(actual, expert_bytes, lambda: read(actual)):
                    predictor.discard()
                    io_scheduler.drain()
                    if not io_scheduler.submit(actual, expert_bytes, lambda: read(actual)):
                        raise RuntimeError('Could not reserve exact read')
                value = io_scheduler.take(actual)
            digest.update(value)
        predictor.discard()
        elapsed = perf_counter() - start
        return dict(wall_seconds=elapsed, cpu_seconds=process_time() - cpu_start,
                    exact_route_sha256=digest.hexdigest(), logical_read_calls=len(read_calls),
                    logical_read_bytes=len(read_calls) * expert_bytes,
                    metrics=predictor.metrics(), io_peak_bytes=io_scheduler.peak_bytes)
    finally:
        predictor.close()
        io_scheduler.close()
        os.close(fd)


def run(args):
    baseline = load_control(Path(args.baseline_source).resolve())
    variants = {'phase1': baseline, 'phase2': SpecPrefetch}
    output = {'scope': __doc__, 'python': sys.executable, 'numpy_version': np.__version__,
              'calls_per_metadata_trial': args.calls, 'rounds': args.rounds,
              'route_seed': 20261023, 'order_seed': args.seed,
              'baseline_source_sha256': sha256(Path(args.baseline_source).read_bytes()).hexdigest(),
              'current_source_sha256': sha256((ROOT / 'air_llm/airllm/nextgen/specprefetch.py').read_bytes()).hexdigest(),
              'shared_scheduler_source_sha256': sha256((ROOT / 'air_llm/airllm/nextgen/cache.py').read_bytes()).hexdigest(),
              'os_cache_flushed': False, 'physical_ssd_validated': False,
              'metadata': {}, 'pread_route_replay': {}}
    for name, predictor_type in variants.items():
        output['metadata'][name] = [metadata_trial(predictor_type, edges, args.calls)
                                    for edges in (128, 1024, 4096)]
        output['metadata'][name].append(metadata_trial(predictor_type, 4096, args.calls, True))
    with tempfile.TemporaryDirectory(prefix='airllm-specprefetch-control-') as temporary:
        fixture = Path(temporary) / 'experts.bin'
        fixture.write_bytes(np.random.default_rng(20261023).bytes(8 * 1024 ** 2))
        output['fixture_sha256'] = sha256(fixture.read_bytes()).hexdigest()
        rng = random.Random(20261023)
        traces = {'repeated_route': [1] * 128,
                  'random_routes': [rng.randrange(32) for _ in range(128)]}
        for trace, routes in traces.items():
            reports = {name: [] for name in (*variants, 'phase2_gate_disabled')}
            # Alternate order across rounds to reduce order bias. OS cache is
            # deliberately labelled uncontrolled, including the first round.
            for round_index in range(args.rounds):
                names = list(reports)
                random.Random(args.seed + round_index).shuffle(names)
                for name in names:
                    cls = baseline if name == 'phase1' else SpecPrefetch
                    reports[name].append(route_trial(cls, fixture, routes,
                                                      gate=name != 'phase2_gate_disabled'))
            hashes = {report['exact_route_sha256'] for rows in reports.values() for report in rows}
            if len(hashes) != 1:
                raise AssertionError('An exact route payload changed')
            output['pread_route_replay'][trace] = {
                name: {'median_seconds': statistics.median(row['wall_seconds'] for row in rows),
                       'runs': rows} for name, rows in reports.items()}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calls', type=int, default=8192)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    if args.calls < 1 or args.rounds < 1:
        parser.error('calls and rounds must be positive')
    result = run(args)
    print(json.dumps({key: {name: value['median_seconds'] for name, value in reports.items()}
                      for key, reports in result['pread_route_replay'].items()}, indent=2))


if __name__ == '__main__':
    main()
