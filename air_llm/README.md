# AirLLM Next 5.0.0a1

Research alpha of the AirLLM derivative maintained at
[iliasm-a11y/airllm-next](https://github.com/iliasm-a11y/airllm-next).
The original AirLLM copyright and Apache 2.0 license are preserved. This is a
derivative research build, with separate model licenses for any user-supplied
weights. Model weights are not included.

The Next engine runs local checkpoints through selective Safetensors reads,
bounded weight and context caches, native MLX compute, and an optional verified
lossless transport. It implements GLM DSA/MoE and a separate bias-free Llama
path with default or linear RoPE. Qwen hybrid execution is explicitly experimental.
Full GLM-5.3 and Qwen3.8 generation on 16 GB has not been certified.

V5 adds optional compact source residency inside the weight cache budget,
decoded output tile admission, measured speculation cost control, read-only
resource preflight, and a bounded durable agent queue sharing one model and
one resident architecture context. Cache changes remain opt-in; fewer file
reads do not guarantee lower generation time. Context switches replay verified
tokens, which trades latency for bounded native-state memory.

Python 3.12 or newer is required. The macOS setup requires native ARM64
Python 3.12/3.13 and MLX 0.32.4; this build's execution checks use Python
3.12. From a repository checkout:

```sh
python3.12 -m venv .venv-next
.venv-next/bin/python -m pip install './air_llm[next]'
.venv-next/bin/airllm-next doctor
.venv-next/bin/airllm-next --help
```

For a downloaded wheel, install the wheel's `next` extra with pip. The extra
selects Metal MLX on macOS and CPU MLX on Linux. Other operating systems have
no Next MLX installation profile in this release.

```sh
airllm-next inspect /absolute/path/to/local/checkpoint
MLX_ENABLE_TF32=0 airllm-next run /absolute/path/to/local/llama \
  --backend llama --device gpu --prompt 'The capital of France is' \
  --max-new-tokens 16
```

Automatic budgets preserve a system memory reserve. The CLI exposes explicit
cache, IO, KV, workspace and reserve limits for smaller workloads. Check
`airllm-next run --help` before changing them. Checkpoint downloads are never
implicit in the Next engine.

See the repository README, usage guide, compatibility table and final validation
report for installation profiles, the loopback OpenAI API subset, reproducible
measurements, numerical precision settings and remaining limits. The bundled
`NOTICE_NEXT.txt` and `NOTICE_ULTRA.txt` retain the inherited provenance notices.
