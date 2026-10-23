# AirLLM Next 5.0.0a1 — research generation

The [multi-token MoE campaign](MULTITOKEN_REPORT_FR.md) compares exact expert
union, causal block verification, residency policies, real-model drafts and
lossy expert storage on trained Granite with bounded CPU/Metal runs. It keeps
negative results and separates known-position verification from generated-token
throughput. SpecPrefetch and the new residency/retry ablations remain opt-in;
these experiments do not validate 300–700B models or the external Fanxiang SSD.

The [second experimental phase](IO_PHASE2_RESEARCH_FR.md) adds explicit macOS
F_NOCACHE reads, OS-accounted process storage counters, improved small reads,
measured SpecPrefetch admission and an explicit experimental Granite MoE backend.
Physical measurements identify the internal Apple SSD; the Fanxiang is absent.
Positive, negative and preliminary measurements remain separately documented.
A complete trained Granite 1.3B MoE now passes the short strict CPU/Metal
protocol through the explicit Python backend, after fixing a worker CPU/GPU
device race. This does not validate giant models or general model quality.

The [October 23 I/O research validation](IO_WALL_RESEARCH_FR.md) adds opt-in
adaptive admission, bounded SpecPrefetch, output tile pipelining and offset-I/O
profiling. On the measured M5/SmolLM2 warm-file protocol with equal 256 MiB
software caches, adaptive admission plus output pipelining gives 1.39× and
46.4% fewer logical read bytes. External Fanxiang SSD and giant-model validation
remain pending; negative ablations and raw evidence hashes are included.

V5 extends the exact `4.1.0rc1` source commit
`9629c27b9d8bdf697ae866498ad468e11eb183d4`. It integrates an optional compact
source-weight cache inside the existing weight budget, reusable output tiles,
measured speculative cost control, resource preflight, and a durable local
agent queue sharing one model and one resident architecture session.
Its separate Llama path also accepts strictly validated linear RoPE scaling.

This is an **alpha research build**, not a stable giant-model release.
V5 measurements in `evidence-v5/` are Linux MLX CPU measurements. Historical M5
Metal measurements in `evidence-final/` belong to V4.1 and are not V5 results.
Full GLM-5.3 and Qwen3.8 generation on a 16 GB M5 remains unvalidated.

See [V5 validation](FINAL_VALIDATION_V5.md), [architecture](NEXT_GENERATION_ARCHITECTURE.md),
[measured comparisons](PERFORMANCE_BEFORE_AFTER.md), [model status](MODEL_COMPATIBILITY_V5.md),
[M5 protocol](M5_VALIDATION.md), and [RONIN guide](RONIN_INFERENCE_GUIDE.md).

```bash
airllm-next preflight /absolute/local/checkpoint --tokens 4096
airllm-next run /absolute/local/checkpoint --backend llama --device cpu \
  --cache-mib 300 --compact-cache-mib 300 --max-new-tokens 16
airllm-next agent-serve /absolute/local/checkpoint --backend llama \
  --state-dir /absolute/private/agent-state --quantum-tokens 8
```

Compact residency and output-tile admission are opt-in ablations. They can
increase conversion work or evict useful weights; no universal speedup is
assumed. Agent scheduling serializes inference and may replay prefixes when
switching requests; logical agents are not simultaneous model executions.

## Inherited 4.1 behavior and evidence

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
