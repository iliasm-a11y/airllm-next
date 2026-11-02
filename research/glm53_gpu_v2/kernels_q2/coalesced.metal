const uint lane = thread_position_in_grid.x % 32;
const uint row0 = (thread_position_in_grid.x / 32) * TILE;
const uint expert = thread_position_in_grid.y;
const uint rows = packed_shape[1];
const uint blocks = WIDTH / 256;
const uint activation = ACTIVATIONS == 1 ? 0 : expert;
if (row0 >= rows) return;
float result[TILE] = {0.0f};
for (uint b = 0; b < blocks; ++b) {
    int act[8];
    for (uint j = 0; j < 8; ++j) act[j] = int(qs[activation * WIDTH + b * 256 + lane + j * 32]);
    for (uint r = 0; r < TILE; ++r) {
        const uint row = row0 + r;
        int dot = 0, minimum = 0;
        float d = 0.0f, dm = 0.0f;
        if (row < rows) {
            const device uint8_t* p = packed + ((expert * rows + row) * blocks + b) * 84;
            const uint low = p[16 + lane];
            const uint high = p[48 + lane];
            for (uint j = 0; j < 8; ++j) {
                const uint k = lane + j * 32;
                const uint sc = p[k / 16];
                const int q = int(((j < 4 ? low : high) >> (2 * (j % 4))) & 3);
                dot += int(sc & 15) * q * act[j];
                minimum += int(sc >> 4) * act[j];
            }
            d = float(as_type<half>(ushort(ushort(p[80]) | (ushort(p[81]) << 8))));
            dm = float(as_type<half>(ushort(ushort(p[82]) | (ushort(p[83]) << 8))));
        }
        dot = simd_sum(dot);
        minimum = simd_sum(minimum);
        if (lane == 0) {
            result[r] = fma(-scales[activation * blocks + b] * dm, float(minimum), result[r]);
            result[r] = fma( scales[activation * blocks + b] * d, float(dot), result[r]);
        }
    }
}
if (lane == 0) {
    for (uint r = 0; r < TILE && row0 + r < rows; ++r)
        out[expert * rows + row0 + r] = result[r];
}
