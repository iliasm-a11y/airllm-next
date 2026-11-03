# Q8 TensorOps — source candidate only

The scalar/shared-tile provider was slower than the ordinary CPU10 graph in
all twelve completed normal warm cases (3.7675..38.9248 times). This distinct
candidate replaces only the32-term integer dot with a Metal4 MPP matmul.
Q8_0 weights and CPU quantized activations remain unchanged. It keeps a
separate integer dot and explicit FP32 FMA for every32-element scale block.
It does not accumulate unscaled integers across blocks with different scales.

The installed SDK declares int8×int8→int32 support in
MPPTensorOpsMatMul2d.h. Apple documents inline tensor operations on Apple10
GPUs and cooperative participation requirements in the
[sample](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
and [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
The32×8 tile and four SIMD groups are candidate choices, not tuned results.
The local bounds, non-contiguous packed-weight strides, integer precision,
TensorOps address spaces and barrier semantics require actual validation.

Required host changes in a distinct package: Metal language4.0,128 threads
per group, gridceil(M/32)×ceil(N/8), embedded source hash, provider patch and
GPU profile fingerprints. Keep all existing resource/ownership/power checks,
buffers, terminal waits and final destination publication. No changes to
the already measured provider, baseline libraries or CPU32 experiment.

No build, dispatch, numerical equality, neural-accelerator counter or speed
is claimed for this source. An actual compiled profile and bounded component
comparison must pass before considering GLM. The prior component's success
cannot be transferred to this shader.
