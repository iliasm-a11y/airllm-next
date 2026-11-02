// Match pinned ARM quantize_row_q8_0: FP32 reciprocal, ties-even, half scale.
const uint block = thread_position_in_grid.x / 32;
const uint lane = thread_position_in_grid.x % 32;
if (block >= x_shape[0] / 32) return;
const float value = x[block * 32 + lane];
const float maximum = simd_max(abs(value));
const float d = maximum / 127.0f;
const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
if (lane == 0) scale_bits[block] = as_type<ushort>(half(d));
qs[block * 32 + lane] = char(rint(value * inverse));
