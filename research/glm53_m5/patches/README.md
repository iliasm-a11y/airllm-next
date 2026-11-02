# Private mmap demand experiment

Base: llama.cpp `3f8af15a76ac8675812e856da45454bafc90f12c`.

`llama-mmap-demand.patch` adds a process-local Apple-only experiment. With
`AIRLLM_MMAP_DEMAND=1`, the initial full-file `POSIX_MADV_WILLNEED` request is
skipped. The model remains a read-only `MAP_SHARED` mapping. File bytes, tensor
layout, quantization, routing and numerical operations are unchanged. Every
other value, including an unset variable or `0`, preserves the original path.

This does not turn off all operating-system read-ahead. It only suppresses the
initial explicit request over the entire mapped file. Ordinary page faults and
macOS paging still operate. Existing lazy-range RANDOM advice remains intact.

Apply only to a separate private checkout. Do not change the historical test
checkout or submit this patch to the upstream llama.cpp project.

```sh
git -C "$GLM_PRIVATE_LLAMA_SOURCE" rev-parse HEAD
git -C "$GLM_PRIVATE_LLAMA_SOURCE" apply --check "$GLM_MMAP_PATCH"
git -C "$GLM_PRIVATE_LLAMA_SOURCE" apply "$GLM_MMAP_PATCH"
cmake -S "$GLM_PRIVATE_LLAMA_SOURCE" -B "$GLM_PRIVATE_LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF
cmake --build "$GLM_PRIVATE_LLAMA_BUILD" --target llama-completion llama-server -j 3
```

The caller supplies absolute paths for the source, patch and build variables.
Use precisely the same benchmark command and binary for both arms; the only
change between them is this environment variable:

```sh
AIRLLM_MMAP_DEMAND=0 "$GLM_PRIVATE_LLAMA_BUILD/bin/llama-completion" ...
AIRLLM_MMAP_DEMAND=1 "$GLM_PRIVATE_LLAMA_BUILD/bin/llama-completion" ...
```

Record the explicit demand-mode log line, process initialization duration,
first-token latency, subsequent token intervals, process storage counters,
peak RSS, memory pressure and swap deltas. Compare deterministic tokens and
logits. Use alternating arm order and multiple runs because file-cache state
affects the result. A process restart alone does not establish physically cold
SSD reads. Prefer a persistent server if it avoids repeated initial requests;
that improvement must be accounted separately from decode latency.

Validation performed while authoring: patch application check passes on the
pinned commit; ASCII and style review of the changed lines. No compile or
hardware performance claim is made by this file. Retain the option only after
the root benchmark confirms a benefit.

## Selected expert prefetch experiment

`llama-expert-prefetch.patch` applies separately to the same pinned commit and
can be combined with the mmap demand patch. The actual CPU matmul functions in
this revision are in `ggml/src/ggml-cpu/ggml-cpu.c`, rather than `ops.cpp`.

With `AIRLLM_EXPERT_PREFETCH=1`, thread zero issues page-aligned
`POSIX_MADV_WILLNEED` hints only for selected contiguous Q2_K expert ranges,
after the existing route-to-expert row counts have been constructed. Adjacent
selected expert IDs are merged into one hint. The existing barrier remains
unchanged. The other threads and all numerical operations retain their order.
No weights are read or modified by the helper itself. Failed hints are ignored.

This experiment is restricted to a single token (`ids.ne[1] == 1`) and buffers
marked as weights. Prefill batches, non-Q2_K matrices and non-Apple builds are
unchanged. For this target, each selected expert projection is 2752512 bytes;
at most eight are requested per projection during decode, rather than the full
792723456-byte bank. It does not predict future route choices, create a cache,
or reorder computation. It does not affect experts computed on Metal.

The hypothesis is that a small number of larger paging requests can reduce the
cost of repeated faults on 16 KiB pages. The hints may themselves block; the
thread-zero call occurs before the normal barrier. The resulting I/O may pollute
cache or run no faster than macOS automatic read-ahead. This can therefore be
slower. Compare the same patched binary with the flag `0` and `1`, and retain
only a measured improvement with identical deterministic token output and
explicit numerical criteria. Its contribution must be isolated from demand
mode and thread-count changes. The patch author performed application and
code inspection only, with no compile or inference run.

The production adapter intentionally scrubs inherited `AIRLLM_` variables.
The benchmark must enable this experiment explicitly in its controlled child
environment; setting a shell variable alone will not bypass that scrub.

## Dense Q8_0 prefetch experiment

`llama-dense-prefetch.patch` is independent of the other patches. It adds an
Apple-only `AIRLLM_DENSE_PREFETCH=1` option in the stock CPU `MUL_MAT` path.
Thread zero requests the contiguous source-weight range immediately before
the existing chunk-initialization barrier; no barrier is added. Eligibility
requires Q8_0, a buffer marked WEIGHTS, a contiguous tensor and one source
column (`src1.ne[1] == 1`). The code does not touch the numeric payload.

The source-column condition covers a single token for the 2-D KDA projections,
FFN/shared-expert matrices and output head. It also covers a one-token MLA
projection broadcast across attention heads; it does not require the source
weight to be 2-D. Activation matmuls are excluded by the weight-buffer marker,
and MoE Q2_K `MUL_MAT_ID` remains the separate expert experiment. Embedding
lookups are not matmuls and are unaffected. A one-token prefill microbatch is
eligible too: the backend cannot infer the caller's semantic phase from this
shape alone.

The pinned tiled dispatcher does not support Q8_0, so its early-return path
does not bypass these target matrices. The optional llamafile early path can
bypass this insertion on another build; it is compiled out for the M5 build's
ARM integer-matmul feature set. The patch does not change either dispatcher.

Each hint can cover a large matrix, up to 674037760 bytes for this checkpoint's
output head. `POSIX_MADV_WILLNEED` can block and consume storage bandwidth or
evict useful pages. Other threads can finish input conversion and wait at the
existing barrier while thread zero issues the hint. There is no claimed
asynchronous overlap or performance improvement. Test this option alone, then
combine only measured compatible improvements. CPU4 and CPU10, demand mode and
expert prefetch must be identified separately in the run configuration.

Author validation: independent application check and a combined three-patch
application check pass on the pinned source; no compilation or inference was
run. Retain only after a real timing comparison and fidelity validation.

## Bounded-column dense prefetch experiment

Apply `llama-dense-prefetch-columns.patch` **after** `llama-dense-prefetch.patch`.
This fourth patch leaves the original three patch files unchanged. It extends
the dense experiment's source-column guard with
`AIRLLM_DENSE_PREFETCH_MAX_COLS`: exact strings `2`, `4` and `8` permit at most
that many source columns. An unset value, `1`, or any other string retains the
original limit of one. No integer parser or unbounded value is accepted.

`AIRLLM_DENSE_PREFETCH=1` is still required; the column limit alone does not
enable prefetch. The Q8_0/contiguous/WEIGHTS eligibility, thread-zero placement,
page-alignment checks and existing barrier are unchanged. With a limit of one,
the original experiment's eligibility remains the same. Eligible hints still
cover the same weight matrix regardless of the number of columns.

The intended pilot is small physical batches or a few logical sequences
sharing an inference step. Source columns are a tensor shape, not an asserted
number of independent clients: prefill microbatches can be eligible too. A
limit of four can therefore affect both four-token prefill and a four-column
decode graph. It does not establish continuous-batching correctness or higher
useful system throughput. Matrix paging can remain blocking and can evict
other weights; any benefit must be measured with explicit batch, slot, context,
cache and column-limit metadata. The production one-slot configuration retains
the limit of one until a separate experiment is qualified.

All four patches passed sequential `git apply --check` and application in a
temporary copy of the two affected source files. The fourth patch depends on
the third's added lines, so passing four dependent patch files to one
`git apply --check` against an unmodified tree is not the validation procedure.
No historical checkout, raw measurement, compilation or inference was changed
or run by the patch author.
