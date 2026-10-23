# V5 validation protocol and scientific boundaries

The source baseline is the exact immutable Git tree at
`9629c27b9d8bdf697ae866498ad468e11eb183d4`. The baseline full suite was rerun
separately before integration. This run uses Linux x86_64 with MLX CPU,
not an Apple M5 or Metal device. The inherited M5 evidence remains attributed
to its original experiments and is preserved without overwriting.

`benchmark_trained.py` starts a fresh process for each source/feature variant.
All variants use the same pinned complete SmolLM2-135M-Instruct checkpoint,
FP32 computation, token counts, prompts, disjoint memory budgets and prefill
settings. The independent reference uses ordinary Transformers. Its all-ones
attention mask is explicit because EOS=PAD must not remove real chat markers
when EOS stopping is disabled. Logit tolerances are fixed before execution:
`atol=1e-4, rtol=5e-4`; all observed token sequences must also match. Raw final
logits are retained in compressed NumPy arrays and hashed. A hash of each
production module identifies the exact source snapshot used by each worker.

The first prompt's initial software-cold forward and subsequent generation
timings are separate. Later prompts reuse that worker's existing software caches;
their initial-forward timings are not relabeled as cold-cache observations.
The OS page cache is uncontrolled. A warm logical-byte reduction does not prove
an equal physical-read reduction, and neither metric is an external SSD rate.
Unfavorable comparisons are retained, including an ordinary in-memory Torch CPU
reference that is faster than the initial streaming CPU control. The small
checkpoint fits physical RAM; it cannot validate the mission's >RAM criterion.

`benchmark_large.py` reads an explicitly downloaded complete, pinned, unchanged
CodeLlama-7b-hf checkpoint. Its two safetensors shards total approximately
13.477 GB, exceeding both this environment's physical-host memory and 8 GiB
cgroup limit. The identical original checkpoint and conservative explicit
budgets can be executed with V4 and V5 in separate processes. A short generation
only establishes that observed workload; it does not establish long-context,
thermal stability, autonomous mission quality or GLM/Qwen compatibility.

The independent large reference calls the official Transformers
`LlamaDecoderLayer`, RMS normalization, rotary embedding and dynamic caches.
Each decoder block is instantiated on the meta device, populated with original
safetensors weights in FP32, evaluated, then released. Embeddings read only
requested rows; the head is independently tiled. This avoids loading the whole
model for reference verification. A separate miniature control compares this
streamed HF reference with ordinary fully resident HF prefill and cached decode.
The large-checkpoint tolerances are unchanged from the small protocol. A
failure is retained and cannot be converted to a success by widening them.

Physical Linux process read counters, process peak RSS, system swap and cgroup
memory limit/current/peak/events are recorded when available. Cgroup counters
include shared filesystem cache and other processes and must not be relabeled
as model/process allocations. No invasive cache purge, swap disabling, memory
pinning, synthetic storage delay or hidden model download is used.

The explicit downloader has a 15 GB ceiling, free-storage headroom, pinned
revision, streamed LFS SHA256 verification, two bounded workers, and resumable
partial files. It never runs from model construction or normal tests. Neither
GLM-5.3 nor Qwen3.8 giant shards are downloaded by these experiments.
