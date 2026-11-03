# Bounded original-weight NoCopy view — distinct candidate

Three fixed buffers, one bounded original-pointer view and one GPU command
per supported matrix. The view must cover a readable VM region, fit the
64MiB workspace/device admission, alias exactly, and be contained logically
in the original CPU weight buffer. The shader uses its precise offset/shape.
No copied weight backing or complete model copy; physical SSD reads remain
unobserved. View/command stay alive through terminal GPU status and staged
readback. Every error after submission remains fatal with no CPU fallback.
All scales are checked before this single submission. The old second-tile
subnormal case is now a genuine pre-submit decline, not a late-fatal test.
Pending-timeout/GPU-error injection remains untested for this new lifetime.

Original resource/power checks precede each command and follow readback;
the existing external guards, buffers/data ceilings and source tracking remain.
The fixture uses owned read-only anonymous mmap weights with64-byte offset
to test the alias path; its synthetic weights are not original GGUF tensors.
It adds actual Darwin lifetime RSS to the original sampled128MiB cap.
Kernel bytes and quantizer arithmetic are unchanged from the validated MPP
integer-dot component. A fresh native build and raw finite equality are required.
Build/runtime OFF; no GLM, SSD, EOS or performance qualification inherited.
