#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_ENABLE_TF32=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$task_python" - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import os, sys
from airllm.nextgen.experiment_guard import supervise, ExperimentLimits
folder=Path('evidence-throughput')/datetime.now(timezone.utc).strftime('validation-%Y%m%dT%H%M%SZ')
folder.mkdir(parents=True,exist_ok=False)
command=[sys.executable,'-m','pytest','tests_next','tests_ultra','air_llm/tests','research/exact_io/tests',
    'research/v5/test_streamed_reference.py','benchmarks/test_quantized_granite_benchmark.py',
    'benchmarks/test_quantized_corpus_quality.py','benchmarks/test_velocity_quantization.py','research/real_world/tests',
    '--ignore=air_llm/tests/test_streaming_gpu.py','-q',f'--junitxml={folder}/tests.xml']
report=supervise(command,folder/'tests.json',ExperimentLimits(admitted_payload_bytes=768<<20,timeout_seconds=900),env=dict(os.environ))
print((folder/'tests.log').read_text())
print('Guard:',report)
PY
"$task_python" -m compileall -q air_llm/airllm benchmarks/throughput_gguf.py \
  benchmarks/campaign_throughput.py benchmarks/analyze_throughput.py benchmarks/study_gguf_draft.py \
  benchmarks/package_throughput_evidence.py benchmarks/restore_throughput_evidence.py
git diff --check
