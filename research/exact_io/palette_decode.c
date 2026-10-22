/* Exact inverse of palette16. Fast path is portable scalar or ARM NEON.
 * No model floating-point operations occur here. No compiler fast-math flags.
 */
#include <stdint.h>
#include <stddef.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

int palette_decode(const uint8_t *src, size_t size, uint8_t *out, size_t n) {
    const size_t nc = (n + 1) / 2;
    if (size < 15 + n + nc) return -1;
    const uint8_t *low = src + 15, *codes = low + n;
    const uint8_t *esc = codes + nc, *end = src + size;
    uint8_t table[16];
    for (int j = 0; j < 15; ++j) table[j] = src[j];
    table[15] = 0;
    size_t i = 0;
#if defined(__ARM_NEON)
    const uint8x16_t lut = vld1q_u8(table), mask = vdupq_n_u8(15);
    for (; i + 32 <= n; i += 32) {
        uint8x16_t packed = vld1q_u8(codes + i / 2);
        uint8x16_t even = vandq_u8(packed, mask), odd = vshrq_n_u8(packed, 4);
        uint8x16_t idx0 = vzip1q_u8(even, odd), idx1 = vzip2q_u8(even, odd);
        uint8x16x2_t first = { { vld1q_u8(low+i), vqtbl1q_u8(lut, idx0) } };
        uint8x16x2_t second = { { vld1q_u8(low+i+16), vqtbl1q_u8(lut, idx1) } };
        vst2q_u8(out+2*i, first); vst2q_u8(out+2*i+32, second);
        /* Avoid scalar escape scan for the overwhelmingly common zero-escape tile. */
        uint8x16_t all = vorrq_u8(vceqq_u8(even, mask), vceqq_u8(odd, mask));
        if (vmaxvq_u8(all)) {
            for (size_t j = i; j < i+32; ++j) {
                unsigned code = (codes[j/2] >> ((j%2)*4)) & 15;
                if (code == 15) {
                    if (esc == end) return -2;
                    out[2*j+1] = *esc++;
                }
            }
        }
    }
#endif
    for (; i < n; ++i) {
        unsigned code = (codes[i/2] >> ((i%2)*4)) & 15;
        out[2*i] = low[i];
        if (code == 15) {
            if (esc == end) return -2;
            out[2*i+1] = *esc++;
        } else out[2*i+1] = table[code];
    }
    return esc == end ? 0 : -3;
}
