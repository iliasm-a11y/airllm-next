#!/usr/bin/env python3
"""Separate native shadow-predictor bookkeeping from speculative IO startup.

This diagnostic wraps benchmark_io_wall without editing production sources.
``shadow`` rejects every speculative submission from process start. It still
predicts, observes exact routes and executes all ordinary demands. Count started
executor threads after shutdown, outside generation timing. No IO delays are
injected. Scope remains the supplied checkpoint and uncontrolled OS cache.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'air_llm'))
BENCHMARK = Path(__file__).with_name('benchmark_io_wall.py')
spec = importlib.util.spec_from_file_location('native_probe_io_wall', BENCHMARK)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def worker(args, remaining):
    from airllm.nextgen.cache import SingleFlight
    from airllm.nextgen.specprefetch import SpecPrefetch
    schedulers = []
    initialize = SingleFlight.__init__

    def record_scheduler(self, *values, **options):
        initialize(self, *values, **options)
        schedulers.append(self)

    SingleFlight.__init__ = record_scheduler
    if args.probe == 'shadow':
        def reject(self, *values, **options):
            self.stats['probe_shadow_no_io'] += 1
            return False
        SpecPrefetch.submit = reject
    elif args.probe == 'warmup':
        submit = SpecPrefetch.submit
        def warmup(self, layer, *values, **options):
            with self.lock:
                state = self.layers.get(layer)
                if state is None or len(state.accuracy) < 4:
                    self.stats['probe_warmup_rejections'] += 1
                    return False
            return submit(self, layer, *values, **options)
        SpecPrefetch.submit = warmup
    sys.argv = [str(BENCHMARK), '--worker',
                'reference' if args.probe == 'reference' else 'specprefetch',
                '--model-dir', args.model_dir, '--output', args.output, *remaining]
    benchmark.main()
    destination = Path(args.output)
    report = json.loads(destination.read_text())
    report['variant'] = args.probe
    report['native_probe'] = {
        'shadow_rejects_all_speculative_submissions': args.probe == 'shadow',
        'warmup_requires_four_shadow_predictions': args.probe == 'warmup',
        'started_executor_threads': [len(scheduler.pool._threads) for scheduler in schedulers],
        'thread_count_scope': 'CPython ThreadPoolExecutor started thread set, inspected after close; no timing instrumentation'}
    destination.write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--probe', choices=('reference', 'shadow', 'warmup', 'specprefetch'))
    args, remaining = parser.parse_known_args()
    if args.probe:
        worker(args, remaining)
        return
    output = Path(args.output).resolve()
    artifacts = output.parent / (output.stem + '-runs')
    artifacts.mkdir(parents=True, exist_ok=True)
    reports, logits = [], {}
    for iteration in range(args.rounds):
        order = ['reference', 'shadow', 'warmup', 'specprefetch']
        random.Random(args.seed + iteration).shuffle(order)
        for probe in order:
            destination = artifacts / f'{iteration:02d}-{probe}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--probe', probe,
                       '--model-dir', str(Path(args.model_dir).resolve()),
                       '--output', str(destination), *remaining]
            done = subprocess.run(command, capture_output=True, text=True, timeout=300,
                                  env=dict(os.environ, MLX_ENABLE_TF32='0', OMP_NUM_THREADS='1',
                                           OPENBLAS_NUM_THREADS='1'))
            destination.with_suffix('.log').write_text(done.stdout + done.stderr)
            if done.returncode:
                raise RuntimeError(f'{probe} failed: {destination.with_suffix(".log")}')
            report = json.loads(destination.read_text())
            report['artifact'] = destination.name
            reports.append(report)
            logits[destination.name] = destination.with_suffix('.npz')
    correctness = benchmark.compare_reports(reports, logits)
    summaries = {}
    for probe in ('reference', 'shadow', 'warmup', 'specprefetch'):
        runs = [report for report in reports if report['variant'] == probe]
        rows = [row for report in runs for row in report['rows'] if row['repeat'] > 0]
        summaries[probe] = {
            'warm_median_seconds': statistics.median(row['generation_seconds_including_drain_and_sampling'] for row in rows),
            'warm_median_logical_bytes': statistics.median(row['counter_delta']['logical_bytes'] for row in rows),
            'started_threads_per_worker': [report['native_probe']['started_executor_threads'] for report in runs],
            'spec_metrics': [report['final_metrics']['spec_prefetch'] for report in runs]}
    result = {'scope': __doc__, 'summaries': summaries, 'correctness': correctness,
              'worker_artifacts': [report['artifact'] for report in reports],
              'current_source_hashes_consistent_across_workers': all(
                  report['source_files_sha256_at_worker_start'] == reports[0]['source_files_sha256_at_worker_start']
                  for report in reports), 'physical_ssd_validated': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
