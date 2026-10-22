#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="$task_root/.venv-macos/bin/python"
if [[ ! -x "$task_python" ]]; then
  "$task_root/scripts/setup_macos.sh"
fi
exec "$task_python" -m airllm.ultra.cli run "$@"
