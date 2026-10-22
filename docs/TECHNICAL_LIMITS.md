# Technical and scientific limits

Lossless transport reconstructs original bits, including signed zero/NaN
payloads. Same-runtime original/standalone cases assert bit-identical logits
and tokens. Independent Torch/MLX comparisons use explicit tolerances and
sequence checks. These assertions are different.

MLX 0.32.4 defaults to reduced-precision float32 matrix operations on supported
hardware. Validation sets `MLX_ENABLE_TF32=0` before initialization; the initial
M5 failure without it is preserved. Runtime metrics record the environment, not
a proof of a subsystem initialized earlier. See [official precision guidance](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

Greedy draft verification is exact under a target function whose argmax is
block-invariant. Floating reductions, batching, absorbed attention and ties can
violate that numerical assumption. No universal bitwise trajectory proof is
claimed. Speculation is opt-in; stochastic correction and MTP are absent.
Prior trained-expert counterexamples/theoretical caveats remain in exact-I/O
research. Leave speculation disabled for a strict ordinary-decoding trajectory.

Managed cache/I/O/KV and estimated scratch are bounded. Python/framework
metadata, native allocators, tokenizer state and OS cache add overhead. MLX
allocator limits are process-wide; multiple models/unrelated consumers require
a combined budget. Recorded RSS and system swap are observations on those runs,
not a bound for all checkpoints/contexts. Default reserves may reject a busy Mac.

Model/API calls serialize. There is no multi-user scheduling or persistent
conversation recovery protocol. Sidecar conversion resumes committed matrices;
incomplete standalone conversion retries in a fresh destination. Interrupted
forward retry restores tested committed prefixes. Power loss, network filesystems
and every physical external-drive interruption have not been reproduced.

The trained checkpoint fits the OS page cache. Its streaming software path does
not certify throughput when weights exceed RAM by orders of magnitude. macOS
physical reads, energy, GPU utilization and temperatures stay null when unknown.
No external physical disk was listed. Compression loses on measured warm reads;
cache/coalescing benefits depend on workload. GLM miniature timing shows no
robust general acceleration. Visible Metal/TensorOps is not a custom-kernel gain.

Remaining giant-model gates:

- Full GLM-5.3/Qwen3.8 generation, quality, route traces and useful draft acceptance.
- NVFP4/ModelOpt kernels, MTP, stochastic verification and broad quantized quality.
- Real SSD/cable/enclosure/random/cold throughput and long-context behavior.
- A storage representation proven to fit the 4.89 TB Qwen BF16 in 2 TB; SmolLM2
  compression cannot be extrapolated to prove this.
- Long sustained thermal/energy validation; the short recorded run does not
  certify a 30-minute plateau or interactive trillion-parameter use.

No unconditional impossibility theorem follows from these implementation limits.
The candidate is usable for its validated subset; the mission's ultimate giant
model objective remains open.
