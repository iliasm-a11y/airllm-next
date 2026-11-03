// Compile only: no buffers, queue, command buffer, encoder or GPU dispatch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
int main(int argc,const char **argv) { @autoreleasepool {
    if(argc!=2)return 64;
    NSError *error=nil;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device || ![device supportsFamily:MTLGPUFamilyApple10])return 65;
    NSString *source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if(!source || error)return 66;
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;options.languageVersion=MTLLanguageVersion4_0;
    const auto started=std::chrono::steady_clock::now();
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library || error){fprintf(stderr,"SHADER_COMPILE_FAILED %s\n",error.localizedDescription.UTF8String ?: "unknown");return 67;}
    id<MTLFunction> function=[library newFunctionWithName:@"q8_batch_dot"];
    id<MTLComputePipelineState> pipeline=function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if(!pipeline || error){fprintf(stderr,"PIPELINE_COMPILE_FAILED %s\n",error.localizedDescription.UTF8String ?: "unknown");return 68;}
    if(pipeline.threadExecutionWidth!=32 || pipeline.maxTotalThreadsPerThreadgroup<128)return 69;
    printf("{\"status\":\"shader_and_pipeline_compiled_only\",\"simd_width\":%lu,\"maximum_threads\":%lu,\"device_allocated_bytes\":%lu,\"compile_seconds\":%.9f,\"matrix_dispatches\":0,\"no_command_queue_created\":true}\n",
        (unsigned long)pipeline.threadExecutionWidth,(unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
        (unsigned long)device.currentAllocatedSize,std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
    return 0;
}}
