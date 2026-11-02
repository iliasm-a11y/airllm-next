// One full SIMD group quantizes once for its eight-row tile.
threadgroup char local_qs[4096];
threadgroup float local_scales[128];
const uint lane = thread_position_in_grid.x % 32;
for (uint block = 0; block < 128; ++block) {
    const float value = x[block * 32 + lane];
    const float maximum = simd_max(abs(value));
    const float d = maximum / 127.0f;
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    local_qs[block * 32 + lane] = char(rint(value * inverse));
    if (lane == 0) local_scales[block] = float(half(d));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint row = thread_position_in_grid.x / 4;
const uint part = thread_position_in_grid.x % 4;
if (row >= packed_shape[0]) return;
float even = 0.0f;
float odd = 0.0f;
for (uint block = 0; block < 128; block += 2) {
    const device uchar* p0 = packed + (row * 128 + block) * 34;
    const device uchar* p1 = p0 + 34;
    int dot0 = 0;
    int dot1 = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = part * 4 + k;
        dot0 += int(as_type<char>(p0[2+j])) * int(local_qs[block*32+j]);
        dot0 += int(as_type<char>(p0[18+j])) * int(local_qs[block*32+16+j]);
        dot1 += int(as_type<char>(p1[2+j])) * int(local_qs[(block+1)*32+j]);
        dot1 += int(as_type<char>(p1[18+j])) * int(local_qs[(block+1)*32+16+j]);
    }
    const float w0 = float(as_type<half>(ushort(ushort(p0[0]) | (ushort(p0[1]) << 8))));
    const float w1 = float(as_type<half>(ushort(ushort(p1[0]) | (ushort(p1[1]) << 8))));
    const float scale0 = w0 * local_scales[block];
    const float scale1 = w1 * local_scales[block+1];
    even = fma(float(dot0), scale0, even);
    odd = fma(float(dot1), scale1, odd);
}
const float even_pair = even + simd_shuffle_xor(even, 1);
const float odd_pair = odd + simd_shuffle_xor(odd, 1);
const float even_sum = even_pair + simd_shuffle_xor(even_pair, 2);
const float odd_sum = odd_pair + simd_shuffle_xor(odd_pair, 2);
if (part == 0) out[row] = even_sum + odd_sum;
