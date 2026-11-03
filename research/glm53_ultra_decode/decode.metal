// MIT. Single-token experiments. Original packed Q8_0/Q2_K weights are read
// directly; the CPU's quantize_row_* supplies the activation blocks.
// Integer sums may be regrouped, but FP32 FMA/reduction order is preserved.
#include <metal_stdlib>
using namespace metal;

inline float half_at(const device uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

template<uint TILE>
inline void q8_dot(const device uchar* weights, const device uchar* activation,
                   device float* output, uint width, uint rows, uint gid) {
    const uint part = gid % 4;
    const uint first = (gid / 4) * TILE;
    if (first >= rows) return;
    const uint blocks = width / 32;
    float even[TILE] = {}, odd[TILE] = {};
    uint block = 0;
    for (; block + 1 < blocks; block += 2) {
        const device uchar* a0 = activation + block * 34;
        const device uchar* a1 = a0 + 34;
        const float ad0 = half_at(a0), ad1 = half_at(a1);
        // Reuse quantized activations and their metadata across TILE rows.
        int av0[8], av1[8];
        for (uint j = 0; j < 4; ++j) {
            const uint k = part * 4 + j;
            av0[j] = int(as_type<char>(a0[2+k]));
            av0[j+4] = int(as_type<char>(a0[18+k]));
            av1[j] = int(as_type<char>(a1[2+k]));
            av1[j+4] = int(as_type<char>(a1[18+k]));
        }
        for (uint r = 0; r < TILE; ++r) {
            if (first+r >= rows) continue;
            const device uchar* w0 = weights + ((first+r)*blocks+block)*34;
            const device uchar* w1 = w0 + 34;
            int d0 = 0, d1 = 0;
            for (uint j = 0; j < 4; ++j) {
                const uint k = part*4+j;
                d0 += int(as_type<char>(w0[2+k]))*av0[j];
                d0 += int(as_type<char>(w0[18+k]))*av0[j+4];
                d1 += int(as_type<char>(w1[2+k]))*av1[j];
                d1 += int(as_type<char>(w1[18+k]))*av1[j+4];
            }
            const float s0 = half_at(w0)*ad0, s1 = half_at(w1)*ad1;
            even[r] = fma(float(d0), s0, even[r]);
            odd[r] = fma(float(d1), s1, odd[r]);
        }
    }
    for (uint r = 0; r < TILE; ++r) {
        // Four lanes, matching the ARM nrc=1 even/odd reduction tree.
        float e = even[r]+simd_shuffle_xor(even[r],1);
        float o = odd[r]+simd_shuffle_xor(odd[r],1);
        e = e+simd_shuffle_xor(e,2); o = o+simd_shuffle_xor(o,2);
        float result = e+o;
        if (block < blocks && first+r < rows) {
            const device uchar* w = weights+((first+r)*blocks+block)*34;
            const device uchar* a = activation+block*34;
            int d = 0;
            for (uint j=0;j<4;++j) {
                const uint k=part*4+j;
                d += int(as_type<char>(w[2+k]))*int(as_type<char>(a[2+k]));
                d += int(as_type<char>(w[18+k]))*int(as_type<char>(a[18+k]));
            }
            d = d+simd_shuffle_xor(d,1); d = d+simd_shuffle_xor(d,2);
            const float scale=half_at(w)*half_at(a);
            result=fma(float(d),scale,result);
        }
        if (!part && first+r < rows) output[first+r]=result;
    }
}

template<uint TILE>
inline void q2_dot(const device uchar* weights, const device uchar* activation,
                   device float* output, uint width, uint rows, uint gid) {
    const uint lane=gid%32, first=(gid/32)*TILE, blocks=width/256;
    if (first>=rows) return;
    float result[TILE]={};
    for(uint b=0;b<blocks;++b) {
        // Native block_q8_K = float d, int8 qs[256], int16 bsums[16].
        const device uchar* a=activation+b*292;
        const float ad=as_type<float>(uint(a[0])|(uint(a[1])<<8)|(uint(a[2])<<16)|(uint(a[3])<<24));
        int av[8];
        for(uint j=0;j<8;++j) av[j]=int(as_type<char>(a[4+lane+j*32]));
        for(uint r=0;r<TILE;++r) {
            int dot=0, minimum=0;
            float d=0, dm=0;
            if(first+r<rows) {
                const device uchar* w=weights+((first+r)*blocks+b)*84;
                for(uint j=0;j<8;++j) {
                    const uint k=lane+j*32, sc=w[k/16];
                    const uint pos=16+(k/128)*32+k%32, shift=((k%128)/32)*2;
                    dot+=int(sc&15)*int((w[pos]>>shift)&3)*av[j];
                    minimum+=int(sc>>4)*av[j];
                }
                d=half_at(w+80);dm=half_at(w+82);
            }
            dot=simd_sum(dot);minimum=simd_sum(minimum);
            if(!lane) {
                result[r]=fma(-ad*dm,float(minimum),result[r]);
                result[r]=fma(ad*d,float(dot),result[r]);
            }
        }
    }
    if(!lane) for(uint r=0;r<TILE&&first+r<rows;++r) output[first+r]=result[r];
}

// Two original buffers share one activation; bank2 groups a gate/up-like pair
// in one launch without concatenating/copying the weight payloads.
#define ENTRY(FORMAT,TILE) \
kernel void FORMAT##_tile##TILE(const device uchar* w0 [[buffer(0)]], \
 const device uchar* w1 [[buffer(1)]], const device uchar* a [[buffer(2)]], \
 device float* out [[buffer(3)]], constant uint* shape [[buffer(4)]], \
 uint3 gid [[thread_position_in_grid]]) { \
 if(gid.z>=shape[2]) return; \
 FORMAT##_dot<TILE>(gid.z ? w1 : w0,a,out+gid.z*shape[1],shape[0],shape[1],gid.x); }
ENTRY(q8,1)
ENTRY(q8,2)
ENTRY(q8,4)
ENTRY(q2,1)
ENTRY(q2,2)
ENTRY(q2,4)
