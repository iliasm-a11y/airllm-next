// Fused activation quantization + exact ARM nrc=1 dot, without scratch buffers.
const uint row = thread_position_in_grid.x / 4;
const uint part = thread_position_in_grid.x % 4;
const uint blocks = WIDTH / 32;
if (row >= packed_shape[0]) return;
float even = 0.0f;
float odd = 0.0f;
uint block = 0;
for (; block + 1 < blocks; block += 2) {
    const device uchar* p0 = packed + (row * blocks + block) * 34;
    const float2 v0 = airllm_q8_cpu_partial(p0, x + block*32, part);
    const float2 v1 = airllm_q8_cpu_partial(p0 + 34, x + (block+1)*32, part);
    even = fma(v0.x, v0.y, even);
    odd = fma(v1.x, v1.y, odd);
}
const float even_pair = even + simd_shuffle_xor(even, 1);
const float odd_pair = odd + simd_shuffle_xor(odd, 1);
const float even_sum = even_pair + simd_shuffle_xor(even_pair, 2);
const float odd_sum = odd_pair + simd_shuffle_xor(odd_pair, 2);
float result = even_sum + odd_sum;
if (block < blocks) {
    const device uchar* p = packed + (row * blocks + block) * 34;
    const float2 last = airllm_q8_cpu_partial(p, x + block*32, part);
    const int pair = int(last.x) + simd_shuffle_xor(int(last.x), 1);
    const int total = pair + simd_shuffle_xor(pair, 2);
    result = fma(float(total), last.y, result);
}
if (part == 0) out[row] = result;
