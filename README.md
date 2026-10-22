# AirLLM Next 4.1.0rc1

Local inference with selective Safetensors reads, bounded caches, native MLX
arithmetic and optional reversible standalone storage. This is a **candidate for
the tested configurations**, not a certification of giant models on 16 GB.

A complete trained **SmolLM2-135M-Instruct** checkpoint has generated on the M5
CPU and Metal GPU. Its recorded greedy sequences match independent Transformers
results. GLM DSA and Qwen hybrid numerical/Metal tests use reduced random weights.
**Full GLM-5.3 and Qwen3.8 remain unvalidated.**

## Install and run

```bash
AIRLLM_PYTHON=python3.12 bash scripts/setup_next_macos.sh
export MLX_ENABLE_TF32=0
.venv-next/bin/airllm-next doctor
.venv-next/bin/airllm-next inspect /absolute/local/checkpoint
.venv-next/bin/airllm-next run /absolute/local/SmolLM2-135M-Instruct \
  --backend llama --device gpu --float32 \
  --prompt 'The capital of France is' --max-new-tokens 16 \
  --cache-mib 128 --io-mib 32 --kv-mib 32 --workspace-mib 128 \
  --reserve-mib 512 --max-tensor-mib 32 --row-chunk 2048
```

Checkpoints must exist locally; model downloads are never implicit. Automatic
configuration retains a conservative system reserve and may refuse inference
when applications already occupy the available memory. Explicit budgets above
were exercised on the accessible 16 GB Mac.

MLX can use reduced-precision matrix units with float32 arrays. Set
`MLX_ENABLE_TF32=0` before Python starts for the reference validation protocol.

## Integrated functionality

- Separate GLM DSA, default-RoPE bias-free Llama and experimental Qwen hybrid
  implementations; early rejection of unsupported architectures and formats.
- Offset, row and expert reads; bounded asynchronous demand/prefetch; optional
  coalescing; LRU/SLRU/LFU/cost caches; tiled vocabulary projection.
- Target-verified greedy drafts or prompt lookup for GLM/Llama, adaptive depth
  and append-only KV rollback. Numerical batch-invariance limits are explicit.
- Lossless overlays and complete standalone packs, per-block hashes and bounded
  local config/tokenizer assets. Quantized INT4/INT8 sidecars remain opt-in lossy
  derivatives; they are different from FP8 and NVFP4.
- Loopback text/chat API, diagnostic commands, controlled budgets and JSON metrics.

Compression saves storage in the measured example but slows warm reads. It is
optional. No miniature result is extrapolated to interactive trillion-parameter
inference.

## Documentation and evidence

- [Complete user guide](docs/USER_GUIDE.md)
- [Candidate architecture](docs/RELEASE_ARCHITECTURE.md)
- [Model compatibility](docs/RELEASE_COMPATIBILITY.md)
- [Technical and scientific limits](docs/TECHNICAL_LIMITS.md)
- [Final validation](FINAL_VALIDATION.md) and [release notes](RELEASE_NOTES.md)

Raw results: `evidence-final/`. Original archive, earlier reports, experiments and
wheels remain preserved. Their historical environments are distinct from this
candidate. The [exact-I/O scientific report](research/exact_io/RAPPORT_SCIENTIFIQUE_FR.md)
and its caveats remain available. Apache-2.0 software; model licenses are separate.
No merge into `main` and no stable release publication.
