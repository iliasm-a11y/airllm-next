# Packed GGUF IO optimization (experimental, 2026-10-24)

Base: research/real-world-2026-10-23 at 76fd683c37534807e21b5fda07a3d717b163215d.

## Changes

- Optional read-only zero-copy NumPy view of the immutable `pread` payload.
- Bounded host buffer ownership tracked until the last ndarray reference is released.
- Preserve the original copying path by default for controlled A/B comparisons.
- Optional bounded bulk parsing for numeric GGUF metadata arrays, with a 64 KiB chunk ceiling.
- Opt-in CLI flags: `--packed-copy-free` and `--bulk-metadata`.
- Targeted regression test for ownership, memory exhaustion, data equality and backward compatibility.

## Important limits

This branch is a GitHub integration of the optimization approach, **not** a claim that the full prior experimental artifact or its entire 90-test suite was uploaded. The original offline measurements were conducted on a different execution environment. The exact current GitHub revision must be revalidated; the GitHub integration has not been executed on Apple M5 or GLM 5.3. The current packed path is Qwen GGUF-specific, not a GLM throughput improvement.

Run the existing GGUF validation and the new regression test before enabling the flags. Compare identical checkpoints, prompts, token budgets, temperatures, SSD and thermal conditions. Measure end-to-end tok/s, TTFT, resident memory, swap and physical IO separately from logical read bytes. Keep both flags off by default until numerical parity and hardware safety checks pass.
