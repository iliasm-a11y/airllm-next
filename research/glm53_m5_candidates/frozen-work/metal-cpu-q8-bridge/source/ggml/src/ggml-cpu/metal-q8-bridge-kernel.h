#pragma once
// Exact validated component source; only q8_batch_dot is dispatched here.
#define AIRLLM_CPU_METAL_Q8_KERNEL_SHA256 "9dfb78a4901a1a876a394671640fddef07313203dcf19e68cd1b014ceb75cb83"
static const char * const airllm_cpu_metal_q8_source = R"Q8BRIDGE(#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void q8_batch_quantize(
    device const float * x [[buffer(0)]],
    device char * qs [[buffer(1)]],
    device ushort * scales [[buffer(2)]],
    constant uint * shape [[buffer(3)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const float value = x[gid];
    const float maximum = simd_max(abs(value));
    const float d = maximum / 127.0f;
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    qs[gid] = char(rint(value * inverse));
    if (lane == 0) scales[gid / 32] = as_type<ushort>(half(d));
}

kernel void q8_batch_dot(
    device const uchar * packed [[buffer(0)]],
    device const char * qs [[buffer(1)]],
    device const ushort * scales [[buffer(2)]],
    device float * output [[buffer(3)]],
    constant uint * shape [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint width = shape[0], rows = shape[1], tokens = shape[2];
    const uint local_row = tid / 8, local_token = tid % 8;
    const uint row = group.x * 8 + local_row, token = group.y * 8 + local_token;
    const uint blocks = width / 32;
    threadgroup char weight_tile[8 * 32];
    threadgroup char activation_tile[8 * 32];
    threadgroup float weight_scales[8];
    threadgroup float activation_scales[8];
    float sum = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        for (uint i = tid; i < 8 * 32; i += 64) {
            const uint tile_row = i / 32, k = i % 32;
            const uint source_row = group.x * 8 + tile_row;
            const uint source_token = group.y * 8 + tile_row;
            weight_tile[i] = source_row < rows ? as_type<char>(packed[(source_row * blocks + block) * 34 + 2 + k]) : 0;
            activation_tile[i] = source_token < tokens ? qs[(source_token * blocks + block) * 32 + k] : 0;
        }
        if (tid < 8) {
            const uint source_row = group.x * 8 + tid;
            const uint source_token = group.y * 8 + tid;
            if (source_row < rows) {
                const uint offset = (source_row * blocks + block) * 34;
                weight_scales[tid] = float(as_type<half>(ushort(ushort(packed[offset]) | (ushort(packed[offset + 1]) << 8))));
            } else weight_scales[tid] = 0.0f;
            activation_scales[tid] = source_token < tokens ? float(as_type<half>(scales[source_token * blocks + block])) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int dot = 0;
        for (uint k = 0; k < 32; ++k) dot += int(weight_tile[local_row * 32 + k]) * int(activation_tile[local_token * 32 + k]);
        const float scale = weight_scales[local_row] * activation_scales[local_token];
        sum = fma(float(dot), scale, sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && token < tokens) output[token * rows + row] = sum;
}
)Q8BRIDGE";
