inline float2 airllm_q8_cpu_partial(device const uchar* p, device const float* x, uint part) {
    float maximum = 0.0f;
    for (uint k = 0; k < 4; ++k) {
        const uint j = part * 4 + k;
        maximum = max(maximum, abs(x[j]));
        maximum = max(maximum, abs(x[j+16]));
    }
    maximum = max(maximum, simd_shuffle_xor(maximum, 1));
    maximum = max(maximum, simd_shuffle_xor(maximum, 2));
    const float d = maximum / 127.0f;
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int dot = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = part * 4 + k;
        const int q0 = int(rint(x[j] * inverse));
        const int q1 = int(rint(x[j+16] * inverse));
        dot += int(as_type<char>(p[2+j])) * q0;
        dot += int(as_type<char>(p[18+j])) * q1;
    }
    const float weight_scale = float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
    const float scale = weight_scale * float(half(d));
    return float2(float(dot), scale);
}
