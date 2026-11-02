# Experimental CPU expert advice after use

`patches/llama-expert-drop.patch` is an optional, default-off patch for the pinned
llama.cpp CPU `mul_mat_id` path. It is an IO/cache-policy experiment, not a
quantization or arithmetic change. It is not installed into RONIN by this file.

The motivating profile is the owning experiment's observed selected-page
residency and OS-accounted reads. A low pre-operation resident fraction and
repeated weight reads motivate testing whether short-lived expert pages compete
with dense weights for file cache. They do not prove that `MADV_DONTNEED` will
retain dense pages, reduce physical reads or improve decoding.

Enable only for a separate guarded experiment:

```sh
AIRLLM_EXPERT_DROP_AFTER_USE=1 <owned native experiment command>
```

With the variable unset or different from `1`, no new barrier or advice is
executed. The variable must remain fixed for the process lifetime: every worker
must take the same conditional barrier path. Prefill (`ids->ne[1] != 1`) is
excluded. A single physical inference slot remains required; this does not
coordinate independent readers of the same weights in another engine.

## Safety and synchronization

The inspected pinned `ggml_compute_forward_mul_mat_id` has no function-level
early return after its worker loops. The tiled dispatch exits an expert with
`continue`, then all workers still reach the function end. The patch adds a
common barrier after all selected experts in the current bank have finished.
Only worker zero issues advice; a second barrier prevents other workers from
advancing before advice completes. Existing vector dot products, activations,
route IDs, output writes and reductions are unchanged.

Advice is limited to contiguous Q2_K tensors with `WEIGHTS` usage and the exact
`CPU_Mapped` buffer name. That name alone is insufficient: generic mapped-type
allocations can be writable or anonymous. Each candidate range must also pass
Mach VM checks showing readable, non-writable protection and an external file
pager across its full range. Unknown or mismatched regions are skipped.

The existing completed `matrix_row_counts` deduplicates selected expert IDs.
An expert stride must be at most 64 MiB, the sum of selected strides at most
128 MiB, and the bank dimensions/strides must fit the tensor. Pointer arithmetic
is checked before use. The start is rounded **up** and the end **down** to page
boundaries, so advice never includes adjacent experts or other tensors. Partial
boundary pages remain untouched. Tiny spans with no complete page are skipped.

`madvise(MADV_DONTNEED)` changes advice only. It does not write model weights,
perform global cache purging, call garbage collection, or release any graph
buffer. Its success does not guarantee physical eviction or reduced NAND reads.
File-backed read-only bytes must remain identical when faulted back in; this is
checked by the CPU fixture and must also be checked by full-model forced-input
fidelity before accepting an inference result.

Each eligible bank reports `AIRLLM_EXPERT_DROP` with selected experts, successful
advice calls, advised bytes, skipped ranges, errors and first errno. Do not turn
`advised` bytes into an estimate of physical pages dropped. Advice failure is
reported; it does not substitute different bytes or fabricate success.

## Validation

The patch dry-run applies to the current pinned CPU source with the existing
expert-prefetch experiment. It is stored as an artifact; this subtask did not
modify the owning runtime source or install a runtime.

Portable/static checks:

```sh
python -m unittest research.glm53_gpu_v2.tests_expert_drop -v
```

Three portable checks pass: default-off/arithmetic-preserving diff shape,
readonly/file-backed guard requirements, and inward page alignment. These are
not evidence of runtime fidelity or memory reclamation. The native check is
skipped until explicitly pointed at a compiled fixture linked to the patched
CPU runtime:

```sh
c++ -std=c++17 -I"$SOURCE/ggml/include" \
  research/glm53_gpu_v2/tests_expert_drop_fixture.cpp \
  -L"$LIBRARIES" -lggml -lggml-cpu -lggml-base \
  -Wl,-rpath,"$LIBRARIES" -o /absolute/path/to/expert-drop-fixture

AIRLLM_EXPERT_DROP_FIXTURE=/absolute/path/to/expert-drop-fixture \
  python -m unittest research.glm53_gpu_v2.tests_expert_drop -v
```

The fixture uses four CPU threads and a temporary synthetic readonly file-backed
Q2_K bank. It runs complete `mul_mat_id` operations with flag off/on, compares
all result bits, checks unchanged original mapped bytes, tests duplicate expert
IDs (two selected experts then one unique expert), and checks that a two-token
prefill emits no advice. It requires actual successful advice with zero errors;
an unpatched library or an unsafe skipped mapping cannot pass that native test.
No model or GPU is loaded by the fixture. It compiles against the available
native headers/libraries; actual patched-runtime execution remains a separate
reported check.

Full-model fidelity and performance are still required. Keep the established
RSS/swap/pressure guards and compare identical tokens, routes and CPU arithmetic
against the same binary with this variable off. Compare OS-accounted reads and
device-global counters with their stated scope. Reject or leave the experiment
off if it adds latency or produces no reproducible benefit.
