# Velocity: storage and materialization audit

## What the existing evidence establishes

The first Granite Metal `native_off` row in
`evidence-io-phase2/granite/granite-metal-after-stream-fix-native_off.json`
records 1.335497333 seconds of generation, 0.690126003 seconds of successful
weight-loader wall time, 0.381948421 seconds in the offset-read implementation,
and 0.660500891 seconds waiting for demand futures. It performs 3,119 loads and
100 output tiles. These are warm, uncontrolled OS page-cache measurements on a
model smaller than host RAM; they do not establish SSD bandwidth.

Three costly areas are therefore worth measuring in the current generation
campaign:

1. Storage service and repeated weight traversal: 0.382 seconds of read service
   in that Granite row. Exact prefetch reduces exposed waits in the comparable
   `native` row to 0.335 seconds, versus 0.661 seconds with prefetch disabled.
2. Loader work outside the read implementation: the Granite row has a
   0.308-second difference between loader and read time. This is an attribution
   gap, not an already measured copy cost. On the older real CodeLlama 7B CPU
   experiment, `evidence-v5/trained-large-candidate.json`, the corresponding
   counters are 81.824 seconds and 25.055 seconds. Shape handling, NumPy import,
   conversion, materialization, and scheduling need separate attribution.
3. Model computation, routing readbacks, Python orchestration, and synchronous
   expert/output lifetime boundaries: the old Granite row contains another
   approximately 0.645 seconds outside the sum of loader durations. This is a
   residual, not a GPU-time measurement. It cannot distinguish kernels from
   synchronization. Loader timers can also overlap GPU computation, so these
   figures must not be presented as a disjoint end-to-end breakdown.

The prior weight-residency analysis retains negative evidence: compact encoded
cache versus decoded cache was 0.836x–1.040x by order-specific median-time ratio;
output tile caching was 0.977x–1.104x. That experiment overlapped a large download
and did not establish a general cache-policy gain.

## Implemented load-stage attribution

`NextConfig(profile_io=True)` now enables `source_read_seconds/calls` and
`materialize_seconds/calls` in `NativeWeights.stats`. A decoded cache hit charges
neither stage. Compact-source reuse still charges materialization. Source
failures have an explicit failed-call counter. Packed sidecar transport is
included. Counter updates from readers are protected against lost updates.

The timings are sums of worker wall time, including CPU scheduling and blocking,
and may overlap each other across workers and overlap generation. They are not
CPU utilization or isolated GPU kernel durations. `dequant_seconds` is a subset
of `materialize_seconds`; adding both double-counts the FP8 work. Profiling is
disabled in comparative timing runs and enabled in separate diagnostic runs.

## Implemented exact host-buffer import

`NextConfig(zero_copy_weights=True)` tries `mlx.core.from_dlpack(copy=False)` for
whole, contiguous NumPy backing allocations. This avoids the import copy made by
`mx.array` when the installed allocator accepts that host buffer. The default
remains the existing copy path. Unsupported APIs, alignments, platforms, and
views fall back to an ordinary exact copy.

This uses a documented MLX mechanism, not a new algorithm. The
[MLX v0.32.4 conversion implementation](https://raw.githubusercontent.com/ml-explore/mlx/v0.32.4/python/src/convert.cpp)
describes wrapping a CPU host buffer for GPU access, retaining its ndarray
exporter until the resulting buffer is released, and rejecting unsupported
adoptions. The adoption path is separate from element-type conversion.

The whole-backing rule is necessary: a 1 KiB view of a 32 MiB coalesced buffer
must not retain 32 MiB while charging only 1 KiB to the weight cache. Offset and
lossless readers allocate private, non-recycled storage for each completed read.
FP8 lookup and BF16 conversion remain evaluated on explicit CPU streams before
reader completion. Neither the source checkpoint nor routing semantics changes.

`zero_copy_imports` counts successful imports, `zero_copy_import_bytes` counts
cumulative imported payload, and `zero_copy_fallbacks` counts copied imports.
These counters are not unique live residency or observed RSS savings. Foreign
buffers may be absent from MLX allocator totals; process RSS plus explicit cache
and I/O payload bounds remain necessary memory evidence.

## Validation completed in this session

On the installed MLX 0.32.4 and Apple M5, a 256-byte read-only NumPy buffer kept
the same data pointer after import. Its exporter survived garbage collection
while the MLX array existed. An explicit Metal matrix multiply matched the
expected exact values, and releasing the arrays released the exporter. This was
a functional smoke test, not a performance measurement.

The focused CPU suite passed **65 tests** in 2.62 seconds, covering the new import
and profile cases plus existing compact-cache, output-tiling, pipeline, and
Granite reader-view tests. Small GLM F32 and FP8 fixtures have bit-identical
logits, route counts, and eight generated tokens between import modes. Trained
Granite, larger trained model, and end-to-end performance results belong to the
separate Velocity campaign. No throughput improvement is claimed by this audit.

## Related primary research

- Apple's [SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching)
  evaluates interactions between expert replacement and constrained hardware.
  Its findings warn against assuming that LRU/LFU temporal locality determines
  every MoE workload. Its reported numbers are not measurements of this engine.
- [Who Should Own the Expert Cache?](https://arxiv.org/abs/2608.12103)
  compares kernel and application residency and discusses sensitivity to the
  method used to enforce memory pressure. Its lookahead experiments illustrate
  that predictor recall does not by itself establish end-to-end speedup. It uses
  different hardware and models, and does not certify a macOS cache policy.
- MLX's [unified-memory documentation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst)
  explains shared CPU/GPU access after an MLX array has been created. That alone
  does not mean that a NumPy-to-MLX import avoids its initial copy.
