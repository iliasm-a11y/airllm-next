#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then
 echo 'Usage : bash scripts/launch_next_macos.sh /Volumes/SSD/GLM-5.3 --prompt "Bonjour"' >&2
 exit 2
fi
if [[ ! -x "$task_root/.venv-next/bin/airllm-next" ]]; then
 echo 'Exécutez bash scripts/setup_next_macos.sh auparavant.' >&2
 exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$task_root/.venv-next/bin/airllm-next" run "$@"
