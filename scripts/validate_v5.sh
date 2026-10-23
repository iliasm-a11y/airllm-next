#!/bin/bash
# Offline source validation; trained/giant checkpoints are never downloaded here.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
mkdir -p evidence-v5/final
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -m pytest tests_next tests_ultra air_llm/tests research/exact_io/tests research/v5/test_streamed_reference.py \
  --ignore=air_llm/tests/test_streaming_gpu.py -q \
  --junitxml=evidence-v5/final/tests.xml > evidence-v5/final/tests.log 2>&1
"$task_python" -m compileall -q air_llm/airllm research/v5
"$task_python" -m airllm.nextgen.cli doctor > evidence-v5/final/doctor.json
git diff --check
