# Isolated no-BLAS candidate, not qualified

The original native runtime registers BLAS and CPU as compatible host backends.
BLAS admits eligible matrices at N>=32; its Q8 path dequantizes weights and uses
the original F32 activations. It is arithmetically distinct from CPU Q8 activation
quantization. The direct CPU component fixtures do not exercise that scheduler.

This helper uses the public ggml_backend_unload API before constructing a model,
requires BLAS present before removal and absent after, rejects any remaining
ACCEL device, and requires CPU present. A new metadata/key policy identifies it.
It changes no native library, model, prompt, guard, comparer or production path.
Absolute fences and odd-tail scheduling are copied from v2. Five metadata smokes
and a real registry-only selftest are required by its build script. That selftest
may initialize legacy Metal; it does not dispatch bridge matrices or load GLM.

Source preparation and component diagnostics do not establish model causality,
full parity, cache restoration, acceleration or useful EOS. Actual build bindings
and strict raw capture comparisons are required before any such claim.
