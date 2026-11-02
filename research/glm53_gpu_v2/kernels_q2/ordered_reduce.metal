const uint row = thread_position_in_grid.x;
const uint expert = thread_position_in_grid.y;
const uint rows = packed_shape[1];
const uint blocks = WIDTH / 256;
const uint activation = ACTIVATIONS == 1 ? 0 : expert;
if (row >= rows) return;
float result = 0.0f;
for (uint b = 0; b < blocks; ++b) {
    const device uint8_t* p = packed + ((expert * rows + row) * blocks + b) * 84;
    const float d = float(as_type<half>(ushort(ushort(p[80]) | (ushort(p[81]) << 8))));
    const float dm = float(as_type<half>(ushort(ushort(p[82]) | (ushort(p[83]) << 8))));
    const uint pos = (expert * rows + row) * blocks + b;
    result = fma(-scales[activation * blocks + b] * dm, float(mins[pos]), result);
    result = fma( scales[activation * blocks + b] * d, float(dots[pos]), result);
}
out[expert * rows + row] = result;
