const uint lane = thread_position_in_grid.x % 32;
const uint row = thread_position_in_grid.x / 32;
const uint b = thread_position_in_grid.y;
const uint expert = thread_position_in_grid.z;
const uint rows = packed_shape[1];
const uint blocks = WIDTH / 256;
const uint activation = ACTIVATIONS == 1 ? 0 : expert;
if (row >= rows) return;
const device uint8_t* p = packed + ((expert * rows + row) * blocks + b) * 84;
int dot = 0, minimum = 0;
for (uint k = lane; k < 256; k += 32) {
    const uint sc = p[k / 16];
    const uint pos = 16 + (k / 128) * 32 + k % 32;
    const uint shift = ((k % 128) / 32) * 2;
    const int q = int((p[pos] >> shift) & 3);
    const int act = int(qs[activation * WIDTH + b * 256 + k]);
    dot += int(sc & 15) * q * act;
    minimum += int(sc >> 4) * act;
}
dot = simd_sum(dot);
minimum = simd_sum(minimum);
if (lane == 0) {
    dots[(expert * rows + row) * blocks + b] = dot;
    mins[(expert * rows + row) * blocks + b] = minimum;
}
