const uint lane = thread_position_in_grid.x % 32;
const uint row = thread_position_in_grid.x / 32;
const uint expert = thread_position_in_grid.y;
const uint rows = packed_shape[1];
const uint blocks = WIDTH / 256;
const uint activation = ACTIVATIONS == 1 ? 0 : expert;
const uint group = lane / LANES;
const uint local = lane % LANES;
if (row >= rows) return;
float result = 0.0f;
for (uint first = 0; first < blocks; first += GROUPS) {
    const uint b = first + group;
    int dot = 0, minimum = 0;
    float dd = 0.0f, md = 0.0f;
    if (b < blocks) {
        const device uint8_t* p = packed + ((expert * rows + row) * blocks + b) * 84;
        for (uint k = local; k < 256; k += LANES) {
            const uint sc = p[k / 16];
            const uint pos = 16 + (k / 128) * 32 + k % 32;
            const uint shift = ((k % 128) / 32) * 2;
            const int q = int((p[pos] >> shift) & 3);
            const int act = int(qs[activation * WIDTH + b * 256 + k]);
            dot += int(sc & 15) * q * act;
            minimum += int(sc >> 4) * act;
        }
        const float d = float(as_type<half>(ushort(ushort(p[80]) | (ushort(p[81]) << 8))));
        const float dm = float(as_type<half>(ushort(ushort(p[82]) | (ushort(p[83]) << 8))));
        dd = scales[activation * blocks + b] * d;
        md = -scales[activation * blocks + b] * dm;
    }
    for (ushort delta = LANES / 2; delta > 0; delta /= 2) {
        dot += simd_shuffle_xor(dot, delta);
        minimum += simd_shuffle_xor(minimum, delta);
    }
    for (ushort g = 0; g < GROUPS; ++g) {
        const int sum_dot = simd_shuffle(dot, ushort(g * LANES));
        const int sum_min = simd_shuffle(minimum, ushort(g * LANES));
        const float d = simd_shuffle(dd, ushort(g * LANES));
        const float m = simd_shuffle(md, ushort(g * LANES));
        if (lane == 0 && first + g < blocks) {
            result = fma(m, float(sum_min), result);
            result = fma(d, float(sum_dot), result);
        }
    }
}
if (lane == 0) out[expert * rows + row] = result;
