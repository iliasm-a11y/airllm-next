// Bounded integer partial-dot experiment. No floating-point dot or model dispatch.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#pragma clang fp contract(off)

kernel void tensorops_int8_partials(
        device char * weights [[buffer(0)]],
        device char * masked_activation [[buffer(1)]],
        device int * result [[buffer(2)]]) {
    // Row-major A[16,32] x B[32,8] -> C[16,8]. Tensor extent 0 is contiguous.
    // Non-const tensor element types match MPP's int8_t specializations;
    // the two input buffers are read only in this kernel.
    auto a = tensor(weights, dextents<int32_t, 2>(32, 16));
    auto b = tensor(masked_activation, dextents<int32_t, 2>(8, 32));
    auto c = tensor(result, dextents<int32_t, 2>(8, 16));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            16, 8, 32, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<1>> op;
    op.run(a, b, c);
}
