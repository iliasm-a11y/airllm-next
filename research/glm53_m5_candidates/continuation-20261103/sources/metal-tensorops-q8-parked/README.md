# TensorOps with parked CPU workers — distinct candidate

The original ggml_barrier loops while ith0 performs synchronous GPU work.
This experiment parks the other workers on one predicate-checked condition
variable after native quantization. A serial provider lock remains held by
ith0 through the final all-reader barrier, preventing overlapping epochs or
early shared-workspace reuse. Initial declines retain both original barriers.
Late deliberate fatal/error still terminates; it never wakes a CPU fallback
while a GPU command is pending. Limits, per-tile resource/power checks, shader,
all allocations and numerical criteria are unchanged.

The scale predicate uses FP16 exponent/mantissa bits; a real native audit
checked all65536 patterns equal to the prior conversion-based predicate.
This is not a full-kernel fidelity or timing transfer. A fresh native build,
finite raw-output checks, native trace, fallback and late-fatal cases must pass.
No source or receipt from the frozen V3 is modified. Build/runtime OFF.
