#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-python3}"
cd "$task_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -m pytest tests_next -q --junitxml=evidence-next/tests-next.xml > evidence-next/logs/tests-next.log 2>&1
"$task_python" -m pytest tests_ultra -q --junitxml=evidence-next/tests-ultra.xml > evidence-next/logs/tests-ultra.log 2>&1
"$task_python" -m pytest air_llm/tests --ignore=air_llm/tests/test_streaming_gpu.py -q --junitxml=evidence-next/tests-upstream.xml > evidence-next/logs/tests-upstream.log 2>&1
"$task_python" -m compileall -q air_llm/airllm
printf '%s\n' 'Tests et compilation Python terminés. Résultats : evidence-next/.'
