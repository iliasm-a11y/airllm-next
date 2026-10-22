#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-$task_root/.venv-next/bin/python}"
if [[ ! -x "$task_python" ]]; then echo 'Installez avec setup_next_macos.sh' >&2;exit 1;fi
cd "$task_root"
mkdir -p evidence-next
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
"$task_python" -m airllm.nextgen.cli doctor > evidence-next/macos-environment.json
"$task_python" - <<'PY'
import mlx.core as mx,platform
if not mx.metal.is_available():raise SystemExit('Metal requis pour la validation Mac')
print('Validation sur',platform.platform(),'MLX',mx.__version__)
PY
AIRLLM_TEST_DEVICE=gpu "$task_python" -m pytest tests_next -q --junitxml=evidence-next/tests-macos.xml
"$task_python" benchmarks/benchmark_next.py --device gpu --output evidence-next/benchmark-macos.json --work-dir .benchmark-macos
printf '%s\n' 'Validation miniature terminée ; aucun GLM-5.3 complet ni SSD Fanxiang certifié par ces tests.'
