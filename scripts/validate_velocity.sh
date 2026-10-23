#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
mkdir -p evidence-velocity
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -m pytest tests_next tests_ultra air_llm/tests research/exact_io/tests research/v5/test_streamed_reference.py \
  benchmarks/test_quantized_granite_benchmark.py benchmarks/test_quantized_corpus_quality.py \
  benchmarks/test_velocity_quantization.py --ignore=air_llm/tests/test_streaming_gpu.py -q \
  --junitxml=evidence-velocity/tests-final.xml > evidence-velocity/tests-final.log 2>&1
"$task_python" -m compileall -q air_llm/airllm benchmarks research/velocity
"$task_python" -m airllm.nextgen.cli run --help > evidence-velocity/cli-help.txt
git diff --check
