// Experimental GGML Q2_K decode matvec. GGML authors' layout is MIT licensed.
// MLX supplies declarations. Exact GGUF bytes: scales[16], qs[64], half d,dmin.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_position_in_grid.x % 32;
const uint width = WIDTH;
const uint blocks = width / 256;
if (row >= packed_shape[0]) return;
float result = 0.0f;
for (uint b = 0; b < blocks; ++b) {
    const device uint8_t* p = packed + (row * blocks + b) * 84;
    const float d = float(as_type<half>(ushort(ushort(p[80]) | (ushort(p[81]) << 8))));
    const float dm = float(as_type<half>(ushort(ushort(p[82]) | (ushort(p[83]) << 8))));
    DOT_BODY
}
if (lane == 0) out[row] = result;
