#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
mkdir -p evidence-native-gguf
"$task_python" -m pytest tests_next tests_ultra air_llm/tests research/exact_io/tests \
  research/v5/test_streamed_reference.py benchmarks/test_quantized_granite_benchmark.py \
  benchmarks/test_quantized_corpus_quality.py benchmarks/test_velocity_quantization.py \
  --ignore=air_llm/tests/test_streaming_gpu.py -q \
  --junitxml=evidence-native-gguf/tests-reproduction.xml
"$task_python" -m compileall -q air_llm/airllm benchmarks research/native_gguf
git diff --check
