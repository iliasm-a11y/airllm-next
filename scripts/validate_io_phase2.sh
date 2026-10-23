#!/bin/bash
# Offline phase-2 correctness; model benchmarks remain explicit separate runs.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
mkdir -p evidence-io-phase2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -m pytest tests_next tests_ultra air_llm/tests research/exact_io/tests research/v5/test_streamed_reference.py \
  --ignore=air_llm/tests/test_streaming_gpu.py -q \
  --junitxml=evidence-io-phase2/tests-final.xml > evidence-io-phase2/tests-final.log 2>&1
"$task_python" -m compileall -q air_llm/airllm benchmarks
"$task_python" -m airllm.nextgen.cli run --help > evidence-io-phase2/cli-help.txt
"$task_python" benchmarks/benchmark_io_wall.py --environment-only \
  --output evidence-io-phase2/environment.json
git diff --check
