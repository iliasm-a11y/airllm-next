#include <metal_stdlib>
// Prepared candidate, not compiled or executed as of 2026-11-03 publication.
using namespace metal;
#pragma clang fp contract(off)

kernel void q2_batch_dot(
    device const uchar * packed [[buffer(0)]],
    device const uchar * activation [[buffer(1)]],
    device float * output [[buffer(2)]],
    constant uint * shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint width = shape[0], rows = shape[1];
    const uint row = group.x;
    const uint token = group.y * 8 + tid / 32;
    const uint lane = tid % 32;
    const uint blocks = width / 256;
    threadgroup uchar weight_block[84];
    float result = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        if (tid < 84) weight_block[tid] = packed[(row * blocks + block) * 84 + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device uchar * a = activation + (token * blocks + block) * 292;
        const uint scale_bits = uint(a[0]) | (uint(a[1]) << 8) | (uint(a[2]) << 16) | (uint(a[3]) << 24);
        const float dy = as_type<float>(scale_bits);
        int dot = 0, minimum = 0;
        for (uint j = 0; j < 8; ++j) {
            const uint k = lane + j * 32;
            const uint scale = weight_block[k / 16];
            const uint pos = 16 + (k / 128) * 32 + k % 32;
            const uint shift = ((k % 128) / 32) * 2;
            const int q = int((weight_block[pos] >> shift) & 3);
            const int value = int(as_type<char>(a[4 + k]));
            dot += int(scale & 15) * q * value;
            minimum += int(scale >> 4) * value;
        }
        dot = simd_sum(dot);
        minimum = simd_sum(minimum);
        if (lane == 0) {
            const float d = float(as_type<half>(ushort(ushort(weight_block[80]) | (ushort(weight_block[81]) << 8))));
            const float dm = float(as_type<half>(ushort(ushort(weight_block[82]) | (ushort(weight_block[83]) << 8))));
            const float md = (-dy) * dm;
            const float dd = dy * d;
            result = fma(md, float(minimum), result);
            result = fma(dd, float(dot), result);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[token * rows + row] = result;
}
