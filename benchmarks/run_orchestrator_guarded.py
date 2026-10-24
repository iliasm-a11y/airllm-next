"""Launch load tests in an independently supervised owned process group."""
import argparse
import os
from pathlib import Path
import sys

from airllm.nextgen.experiment_guard import ExperimentLimits, supervise, validate_output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'smollm', 'qwen-gguf'], default='synthetic')
    parser.add_argument('--model')
    parser.add_argument('--counts', default='1,10,100,500,600')
    parser.add_argument('--tokens', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    destinations = validate_output_paths(output, protected=[__file__]+([args.model] if args.model else []))
    if any(path.exists() for path in destinations):
        parser.error('Evidence output/log/guard already exists; use a fresh output path')
    command = [sys.executable, str(Path(__file__).with_name('benchmark_orchestrator.py')),
               '--mode', args.mode, '--counts', args.counts, '--tokens', str(args.tokens),
               '--output', str(output)]
    if args.model:
        command += ['--model', args.model]
    env = dict(os.environ, MLX_ENABLE_TF32='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    report = supervise(command, output, ExperimentLimits(timeout_seconds=args.timeout), env=env)
    print(report)
    if report.get('violation') is not None or report.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
