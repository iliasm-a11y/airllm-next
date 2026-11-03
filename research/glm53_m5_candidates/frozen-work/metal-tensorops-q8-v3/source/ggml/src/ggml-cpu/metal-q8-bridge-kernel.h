#pragma once
// Pipeline-compiled source; numerical/speed qualification remains separate.
#define AIRLLM_CPU_METAL_Q8_KERNEL_SHA256 "b69930d559ef2601e4f384e343bcf6a2b24f30a6b28840f631154335714b82b9"
static const char * const airllm_cpu_metal_q8_source = R"Q8BRIDGE(// Experimental source only. Exactness, compilation and speed are unmeasured.
// Each32-term integer dot is followed by one explicitly ordered FP32 FMA.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#pragma clang fp contract(off)

kernel void q8_batch_dot(
    device const uchar * packed [[buffer(0)]],
    device const char * qs [[buffer(1)]],
    device const ushort * scales [[buffer(2)]],
    device float * output [[buffer(3)]],
    constant uint * shape [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint width=shape[0], rows=shape[1], tokens=shape[2], blocks=width/32;
    // All128 participants enter every operation, including partial edge tiles.
    // Tensor views retain the real matrix bounds; the final stores are guarded.
    constexpr auto descriptor=matmul2d_descriptor(32,8,32,false,true,false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor,execution_simdgroups<4>> operation;
    threadgroup int dot_values[32*8];
    float sums[2]={0.0f,0.0f};
    for(uint block=0;block<blocks;++block) {
        auto weights=tensor((device int8_t *)(packed+block*34+2),
            dextents<int,2>{32,int(rows)},array<int,2>{1,int(blocks*34)});
        auto activations=tensor((device int8_t *)(qs+block*32),
            dextents<int,2>{32,int(tokens)},array<int,2>{1,int(width)});
        auto integer_result=tensor(dot_values,dextents<int,2>{8,32},array<int,2>{1,8});
        auto weight_slice=weights.slice(0,int(group.x*32));
        auto activation_slice=activations.slice(0,int(group.y*8));
        operation.run(weight_slice,activation_slice,integer_result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint part=0;part<2;++part) {
            const uint local=tid+part*128, row=group.x*32+local/8, token=group.y*8+local%8;
            if(row<rows && token<tokens) {
                const uint location=(row*blocks+block)*34;
                const ushort bits=ushort(packed[location])|(ushort(packed[location+1])<<8);
                const float scale=float(as_type<half>(bits))*float(as_type<half>(scales[token*blocks+block]));
                // Integer magnitude is at most32*128*128=524288, within int32.
                // Do not merge adjacent blocks: their scales and FMA order differ.
                sums[part]=fma(float(dot_values[local]),scale,sums[part]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint part=0;part<2;++part) {
        const uint local=tid+part*128, row=group.x*32+local/8, token=group.y*8+local%8;
        if(row<rows && token<tokens)output[token*rows+row]=sums[part];
    }
}
)Q8BRIDGE";
