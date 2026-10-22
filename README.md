# AirLLM Next — out-of-core research engine

Local inference with selective safetensors reads, bounded weight/context caches,
native MLX arithmetic and exact greedy draft verification. This branch extracts
the verified source archive into a normal Git tree and develops its internals.
It does not modify the official AirLLM repository.

**Validation boundary:** Linux CPU MLX miniature numerical tests and measured
storage/conversion experiments are available. Full GLM-5.3, full Qwen3.8,
M5/Metal performance and the external SSD remain unvalidated. No interactive
trillion-parameter performance claim follows from these tests.

## Changes in this branch

- Exact target-verified greedy speculative blocks, GLM KV rollback, accepted-token
  byte accounting, adaptive depth and fallback when the draft costs too much.
- Bounded grouped offset reads, optional consecutive embedding reads, selective
  FP8 scale tiles, and measured transfer/conversion cost cache retention.
- Tiled resumable quantization writes, content digests and explicit integrity
  audits, early allocation guards, and scheduler concurrency fixes.
- Experimental streamed Qwen hybrid backend, separate from GLM, with recurrent
  state transactions. Only independently tested floating miniature variants are
  eligible; newer swish-gate and NVIDIA ModelOpt/NVFP4 checkpoints fail explicitly.
- Loopback-only OpenAI-shaped non-streaming text/chat API, without remote inference.

Read [the measured report](docs/OOC_REPORT_FR.md),
[model status](docs/MODEL_STATUS.md),
[speculative algorithm](docs/SPECULATIVE_STREAMING.md),
[M5 installation/validation](docs/OOC_M5_GUIDE.md), and
[local API limits](docs/OOC_API.md).

## Install and run

On native arm64 macOS:

```bash
bash scripts/setup_next_macos.sh
source .venv-next/bin/activate
python -m airllm.nextgen.cli doctor
python -m airllm.nextgen.cli inspect /Volumes/Models/GLM-5.3
python -m airllm.nextgen.cli run /Volumes/Models/GLM-5.3 --prompt "Bonjour" --max-new-tokens 16
```

A local checkpoint and tokenizer must already exist. No command implicitly
downloads model shards. `inspect` reads metadata and does not run inference.
For reproducible CPU correctness validation, use `requirements-next-linux.txt`.

```bash
python -m pytest tests_next -q
python -m pytest tests_ultra -q
python benchmarks/benchmark_ooc.py --output io.json
```

## Explicit optimization controls

| Control | Behavior |
|---|---|
| No `--draft-model` | Ordinary autoregressive GLM generation |
| `--draft-model /local/draft --speculative-depth 4` | Greedy proposal/target verification; compatible local draft |
| `--no-adaptive-speculation` | Fixed speculative depth for controlled experiments |
| `--coalesce-reads` | Opt in to grouped reads; default disabled after no warm-path gain |
| `--no-selective-fp8-scales` | Reproduce full scale-matrix reads for comparison |
| `--cache-policy lru/slru/lfu/cost` | Explicit cache policy, overrides saved tuning |
| `--no-prefetch` | Disable predicted prefetch, without changing exact MoE selection |
| `--backend qwen-experimental` | Explicit floating Qwen research backend; no full Qwen3.8 claim |

Quantization changes the weights and is opt-in. FP8 checkpoint decoding does
not reproduce production dynamic activation-quantization kernels. INT4 is not
NVFP4. Read the model-status document before interpreting numerical parity.

## Reproducibility and provenance

Raw measurements, source hashes, failed/excluded timing windows and reference
versions live in `evidence-ooc/`. Historical input reports and old wheels remain
under their original names; build the current source with
`python -m pip wheel --no-deps ./air_llm -w dist-ooc`.
The original archive SHA-256 is
`7433255c37830b6d1603e276452154a81e67c7bdbd6eae826216230b1bb93e2d`.
Its Git blob hash matches the public repository's source ZIP.

Apache-2.0 source notices are preserved. Model weights have their own licenses.
The [upstream README](docs/README_UPSTREAM.md) and archived source/evidence are
retained for technical history; their historical claims are not new validations.
