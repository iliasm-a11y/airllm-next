/* Independent GGML C oracle; see manifest.json and LICENSE for provenance.
 * Only the ABI and portable IEEE binary16 conversion below are local code.
 * The decoder functions and all lookup tables are unmodified upstream C.
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static float oracle_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t fraction = h & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((fraction & 0x400u) == 0) {
                fraction <<= 1;
                --e;
            }
            bits = sign | ((uint32_t) (e + 127) << 23) |
                   ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

#define GGML_FP16_TO_FP32 oracle_fp16_to_fp32
#define GGML_RESTRICT restrict
#include "ggml-iq-dequant.c.inc"

/* Values copied from the pinned ggml_type enum, checked by Python tests. */
enum { ORACLE_IQ3_XXS = 18, ORACLE_IQ3_S = 21, ORACLE_IQ2_S = 22 };

size_t oracle_block_bytes(int type) {
    switch (type) {
        case ORACLE_IQ2_S: return sizeof(block_iq2_s);
        case ORACLE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case ORACLE_IQ3_S: return sizeof(block_iq3_s);
        default: return 0;
    }
}

int oracle_decode_iq(int type, const void * raw, size_t raw_bytes,
                     float * output, size_t output_elements) {
    const uint16_t endian = 1;
    if (*(const uint8_t *) &endian != 1 || sizeof(float) != 4) return -1;
    const size_t block_bytes = oracle_block_bytes(type);
    if (!block_bytes || raw_bytes % block_bytes) return -2;
    const size_t blocks = raw_bytes / block_bytes;
    if (blocks > INT64_MAX / QK_K || output_elements != blocks * QK_K) return -3;
    if ((!raw && raw_bytes) || (!output && output_elements)) return -4;
    switch (type) {
        case ORACLE_IQ2_S:
            dequantize_row_iq2_s(raw, output, (int64_t) output_elements);
            break;
        case ORACLE_IQ3_XXS:
            dequantize_row_iq3_xxs(raw, output, (int64_t) output_elements);
            break;
        case ORACLE_IQ3_S:
            dequantize_row_iq3_s(raw, output, (int64_t) output_elements);
            break;
        default: return -5;
    }
    return 0;
}

/* Test the local conversion separately rather than assuming it is correct. */
void oracle_convert_all_half(float * output) {
    for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
        output[bits] = oracle_fp16_to_fp32((uint16_t) bits);
    }
}
