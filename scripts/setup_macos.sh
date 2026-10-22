#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Ce script nécessite macOS sur Apple Silicon.' >&2
  exit 1
fi
task_python="${AIRLLM_PYTHON:-python3}"
"$task_python" -c 'import sys; assert (3,12) <= sys.version_info[:2] <= (3,13), "Python 3.12 ou 3.13 requis"'
"$task_python" -m venv "$task_root/.venv-macos"
"$task_root/.venv-macos/bin/python" -m pip install -r "$task_root/requirements-macos.txt"
"$task_root/.venv-macos/bin/python" -m pip install --no-deps -e "$task_root/air_llm"
echo 'Installation terminée. Lancez scripts/validate_macos.sh avant un modèle géant.'
