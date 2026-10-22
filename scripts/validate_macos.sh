#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="$task_root/.venv-macos/bin/python"
cd "$task_root"
mkdir -p validation-local
"$task_python" -m airllm.ultra.cli doctor > validation-local/doctor.json
"$task_python" -m pytest tests_ultra -q --junitxml=validation-local/tests.xml
echo 'Rapports disponibles dans validation-local/. Pour le SSD :'
echo '  .venv-macos/bin/python -m airllm.ultra.cli io-bench --directory /Volumes/VOTRE_SSD'
