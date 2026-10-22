#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'macOS Apple Silicon natif requis ; Rosetta et Linux ne sont pas ce profil.' >&2
  exit 1
fi
task_python="${AIRLLM_PYTHON:-python3}"
"$task_python" - <<'PY'
import os,platform,shutil,sys,subprocess
if not (3,12)<=sys.version_info[:2]<=(3,13):raise SystemExit('Python 3.12 ou 3.13 requis')
version=tuple(int(x) for x in platform.mac_ver()[0].split('.')[:2])
if version<(14,0):raise SystemExit('macOS 14 minimum pour MLX')
if version<(26,2):print('macOS 26.2+ recommandé pour les Neural Accelerators du M5 via MLX.')
free=shutil.disk_usage('.').free
if free<4*1024**3:raise SystemExit('Au moins 4 Gio libres requis pour les dépendances et validations miniatures')
memory=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
print(f'RAM unifiée détectée : {memory/1024**3:.1f} Gio. Le moteur ajustera ses budgets à la mémoire disponible.')
print('Pré-vérification réussie. Aucun poids de grand modèle ne sera téléchargé.')
PY
if [[ ! -d "$task_root/.venv-next" ]]; then "$task_python" -m venv "$task_root/.venv-next"; fi
"$task_root/.venv-next/bin/python" -m pip install -r "$task_root/requirements-next-macos.txt"
"$task_root/.venv-next/bin/python" -m pip install --no-deps -e "$task_root/air_llm"
"$task_root/.venv-next/bin/airllm-next" doctor
echo 'Installation terminée. Validation : bash scripts/validate_next_macos.sh'
