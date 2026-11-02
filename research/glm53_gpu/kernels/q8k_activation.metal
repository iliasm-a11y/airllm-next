// Quantize one original FP32 activation block as GGML quantize_row_q8_K_ref.
// This is the CPU's temporary activation quantization, not weight requantization.
const uint b = thread_position_in_grid.x / 32;
const uint lane = thread_position_in_grid.x % 32;
if (b >= x_shape[0] / 256) return;
float amax = 0.0f;
for (uint k = lane; k < 256; k += 32) amax = max(amax, abs(x[b * 256 + k]));
amax = simd_max(amax);
uint first = 256;
for (uint k = lane; k < 256; k += 32) {
    if (abs(x[b * 256 + k]) == amax) first = min(first, k);
}
first = simd_min(first);
const float signed_max = x[b * 256 + first];
const float iscale = amax == 0.0f ? 0.0f : -127.0f / signed_max;
if (lane == 0) scales[b] = amax == 0.0f ? 0.0f : 1.0f / iscale;
for (uint k = lane; k < 256; k += 32) {
    // GGML nearest_int uses this FP32 add, then masks the mantissa (ties even).
    float rounded = iscale * x[b * 256 + k] + 12582912.0f;
    int q = int(as_type<uint>(rounded) & 0x007fffff) - 0x00400000;
    qs[b * 256 + k] = char(min(127, q));
}
