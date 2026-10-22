#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-$task_root/.venv-macos/bin/python}"
cd "$task_root"
if [[ ! -x "$task_python" ]]; then echo 'Set AIRLLM_PYTHON to an installed AirLLM Python environment.' >&2; exit 1; fi
mkdir -p validation-local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
"$task_python" -m airllm.ultra.cli doctor > validation-local/doctor.json
"$task_python" -m pytest tests_ultra -q --junitxml=validation-local/tests.xml
echo 'Rapports disponibles dans validation-local/. Pour le SSD :'
echo '  .venv-macos/bin/python -m airllm.ultra.cli io-bench --directory /Volumes/VOTRE_SSD'
