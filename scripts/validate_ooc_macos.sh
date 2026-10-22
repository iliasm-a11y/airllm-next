#!/bin/bash
# Real Apple Silicon validation. No downloads, sudo, cache purge or SSD writes.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
task_python="${AIRLLM_PYTHON:-$task_root/.venv-next/bin/python}"
task_model=""
task_ssd_file=""
task_seconds=300
task_qwen_miniature=0
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --model || "$1" == --ssd-file || "$1" == --seconds ]]; then
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
      echo "Missing value for $1" >&2; exit 2
    fi
  fi
  case "$1" in
    --model) task_model="$2"; shift 2 ;;
    --ssd-file) task_ssd_file="$2"; shift 2 ;;
    --seconds) task_seconds="$2"; shift 2 ;;
    --qwen-miniature) task_qwen_miniature=1; shift ;;
    *) echo "Usage: $0 [--model /local/checkpoint] [--ssd-file /Volumes/SSD/existing-shard.safetensors] [--seconds 300] [--qwen-miniature]" >&2; exit 2 ;;
  esac
done
if [[ ! "$task_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo '--seconds must be a positive integer.' >&2; exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This validation requires a real Apple Silicon Mac.' >&2
  exit 1
fi
if [[ ! -x "$task_python" ]]; then
  echo 'Set AIRLLM_PYTHON or run scripts/setup_next_macos.sh first.' >&2
  exit 1
fi
cd "$task_root"
export PYTHONPATH="$task_root/air_llm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
mkdir -p "$task_root/evidence-ooc"
task_report="$(mktemp -d "$task_root/evidence-ooc/macos-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$task_python" - <<'PY' > "$task_report/hardware.json"
import json,platform,subprocess
import mlx.core as mx
if not mx.metal.is_available():
    raise SystemExit('A working Metal MLX device is required; CPU fallback is not a GPU validation.')
def read(command):
    try:return subprocess.check_output(command,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return None
print(json.dumps({'platform':platform.platform(),'machine':platform.machine(),
    'chip':read(['sysctl','-n','machdep.cpu.brand_string']),
    'memory_bytes':read(['sysctl','-n','hw.memsize']),
    'mlx_version':mx.__version__,'metal_available':mx.metal.is_available(),
    'macos_version':platform.mac_ver()[0],
    'm5_neural_acceleration_os_requirement_met':tuple(int(x) for x in platform.mac_ver()[0].split('.')[:2]) >= (26,2),
    'custom_tensorops_kernel_validated':False},indent=2))
PY
"$task_python" -m airllm.nextgen.cli doctor > "$task_report/environment.json"
pmset -g therm > "$task_report/thermal-before.txt" 2>&1 || true
vm_stat > "$task_report/vm-before.txt"
sysctl vm.swapusage > "$task_report/swap-before.txt"
# Reference regressions include CPU tests. The dedicated --device gpu benchmark
# below performs actual Metal numerical checks and timings separately.
"$task_python" -m pytest tests_next -q --junitxml="$task_report/tests.xml"
"$task_python" benchmarks/benchmark_ooc.py --cache-state warm --output "$task_report/io-warm.json"
"$task_python" benchmarks/benchmark_ooc.py --cache-state uncontrolled --output "$task_report/io-uncontrolled.json"
"$task_python" benchmarks/benchmark_next.py --device gpu --output "$task_report/miniature-metal.json" --work-dir .benchmark-ooc-macos
if [[ "$task_qwen_miniature" == 1 ]]; then
  "$task_python" benchmarks/validate_qwen_runtime.py --device gpu --output "$task_report/qwen-miniature-metal.json"
fi
if [[ -n "$task_ssd_file" ]]; then
  # 1 TiB cap is a READ cap, not a file allocation or download. The supplied
  # duration normally terminates first. Every allocation remains <= 4 MiB.
  "$task_python" benchmarks/benchmark_ooc.py --probe-file "$task_ssd_file" --probe-seconds "$task_seconds" --probe-max-mib 1048576 --output "$task_report/ssd-sequential.json"
  "$task_python" benchmarks/benchmark_ooc.py --probe-file "$task_ssd_file" --probe-seconds "$task_seconds" --probe-max-mib 1048576 --probe-random --output "$task_report/ssd-random.json"
fi
if [[ -n "$task_model" ]]; then
  "$task_python" -m airllm.nextgen.cli run "$task_model" --device gpu --prompt 'Explique pourquoi le ciel est bleu.' --max-new-tokens 32 --json-report "$task_report/real-generation.json" > "$task_report/real-generation.txt" 2> "$task_report/real-generation-metrics.txt"
  "$task_python" benchmarks/sustained_next.py "$task_model" --device gpu --seconds "$task_seconds" --tokens 32 --output "$task_report/sustained-inference.json"
fi
pmset -g therm > "$task_report/thermal-after.txt" 2>&1 || true
vm_stat > "$task_report/vm-after.txt"
sysctl vm.swapusage > "$task_report/swap-after.txt"
echo "Measurements saved to $task_report"
echo 'Miniature correctness and throughput are separate from real-checkpoint generation. Review scope in each JSON.'
