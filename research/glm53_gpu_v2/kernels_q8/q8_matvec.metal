// Four lanes reproduce one ARM NEON nrc=1 Q8_0 dot; eight rows per SIMD group.
const uint row = thread_position_in_grid.x / 4;
const uint part = thread_position_in_grid.x % 4;
const uint blocks = WIDTH / 32;
if (row >= packed_shape[0]) return;
float even = 0.0f;
float odd = 0.0f;
uint block = 0;
for (; block + 1 < blocks; block += 2) {
    const device uchar* p0 = packed + (row * blocks + block) * 34;
    const device uchar* p1 = p0 + 34;
    int dot0 = 0;
    int dot1 = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = part * 4 + k;
        dot0 += int(as_type<char>(p0[2+j])) * int(qs[block*32+j]);
        dot0 += int(as_type<char>(p0[18+j])) * int(qs[block*32+16+j]);
        dot1 += int(as_type<char>(p1[2+j])) * int(qs[(block+1)*32+j]);
        dot1 += int(as_type<char>(p1[18+j])) * int(qs[(block+1)*32+16+j]);
    }
    const float w0 = float(as_type<half>(ushort(ushort(p0[0]) | (ushort(p0[1]) << 8))));
    const float w1 = float(as_type<half>(ushort(ushort(p1[0]) | (ushort(p1[1]) << 8))));
    const float a0 = float(as_type<half>(scale_bits[block]));
    const float a1 = float(as_type<half>(scale_bits[block+1]));
    const float scale0 = w0 * a0;
    const float scale1 = w1 * a1;
    even = fma(float(dot0), scale0, even);
    odd = fma(float(dot1), scale1, odd);
}
const float even_pair = even + simd_shuffle_xor(even, 1);
const float odd_pair = odd + simd_shuffle_xor(odd, 1);
const float even_sum = even_pair + simd_shuffle_xor(even_pair, 2);
const float odd_sum = odd_pair + simd_shuffle_xor(odd_pair, 2);
float result = even_sum + odd_sum;
if (block < blocks) {
    const device uchar* p = packed + (row * blocks + block) * 34;
    int dot = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = part * 4 + k;
        dot += int(as_type<char>(p[2+j])) * int(qs[block*32+j]);
        dot += int(as_type<char>(p[18+j])) * int(qs[block*32+16+j]);
    }
    const int pair = dot + simd_shuffle_xor(dot, 1);
    const int total = pair + simd_shuffle_xor(pair, 2);
    const float weight_scale = float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
    const float activation_scale = float(as_type<half>(scale_bits[block]));
    const float scale = weight_scale * activation_scale;
    result = fma(float(total), scale, result);
}
if (part == 0) out[row] = result;
