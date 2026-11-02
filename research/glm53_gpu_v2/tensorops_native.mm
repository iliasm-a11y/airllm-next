// MIT. Fixed-size MPP integer probe. Compile-only never creates a command queue.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstring>

static void emit(NSDictionary * report) {
    NSData * data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    if (data) { fwrite(data.bytes, 1, data.length, stdout); fputc('\n', stdout); }
}
static int fail(NSMutableDictionary * report, NSString * reason) {
    report[@"error"] = reason ?: @"unknown failure";
    emit(report);
    return 2;
}
static NSData * readExact(NSString * path, NSUInteger bytes) {
    NSDictionary * attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
    if (!attributes || [attributes fileSize] != bytes) return nil;
    return [NSData dataWithContentsOfFile:path];
}
int main(int argc, const char * argv[]) { @autoreleasepool {
    NSMutableDictionary * report = [@{
        @"stage": @"arguments", @"shader_library_compiled": @NO, @"pipeline_compiled": @NO,
        @"device_execution_attempted": @NO, @"commands_completed": @0,
        @"hardware_neural_accelerator_utilization_measured": @NO
    } mutableCopy];
    if (argc != 4 || (strcmp(argv[1], "--compile-only") && strcmp(argv[1], "--execute")))
        return fail(report, @"usage: tensorops-native --compile-only|--execute source.metal job.json");
    const bool execute = strcmp(argv[1], "--execute") == 0;
    report[@"mode"] = execute ? @"execute" : @"compile-only";
    report[@"stage"] = @"device_capability";
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fail(report, @"Metal device unavailable");
    report[@"device"] = device.name;
    report[@"supports_metal4_family"] = @([device supportsFamily:MTLGPUFamilyMetal4]);
    if (![device supportsFamily:MTLGPUFamilyMetal4]) return fail(report, @"Metal 4 GPU family unavailable");
    NSDictionary * attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:@(argv[2]) error:nil];
    if (!attrs || [attrs fileSize] > 131072) return fail(report, @"shader source absent or exceeds 128KiB");
    NSError * error = nil;
    NSString * source = [NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&error];
    if (!source) return fail(report, error.localizedDescription);
    MTLCompileOptions * options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_1;
    options.mathMode = MTLMathModeSafe;
    report[@"language_version"] = @"4.1";
    report[@"math_mode"] = @"safe";
    report[@"stage"] = @"shader_compile";
    const auto compile_begin = std::chrono::steady_clock::now();
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    report[@"shader_compile_wall_seconds"] = @(std::chrono::duration<double>(std::chrono::steady_clock::now() - compile_begin).count());
    if (!library) return fail(report, error.localizedDescription ?: @"MPP int8 shader library compilation failed");
    report[@"shader_library_compiled"] = @YES;
    report[@"stage"] = @"pipeline_compile";
    id<MTLFunction> function = [library newFunctionWithName:@"tensorops_int8_partials"];
    if (!function) return fail(report, @"integer entry point absent");
    error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return fail(report, error.localizedDescription ?: @"MPP int8 pipeline compilation failed");
    report[@"pipeline_compiled"] = @YES;
    report[@"thread_execution_width"] = @(pipeline.threadExecutionWidth);
    report[@"static_threadgroup_memory_bytes"] = @(pipeline.staticThreadgroupMemoryLength);
    if (pipeline.staticThreadgroupMemoryLength > (1 << 20) - 9312)
        return fail(report, @"fixture plus pipeline static threadgroup payload exceeds 1MiB");
    if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 32)
        return fail(report, @"prototype requires one complete 32-thread SIMD group");
    if (!execute) {
        report[@"stage"] = @"compile_complete_no_command_queue";
        emit(report);
        return 0;
    }
    report[@"stage"] = @"fixture_admission";
    attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:@(argv[3]) error:nil];
    if (!attrs || [attrs fileSize] > 16384) return fail(report, @"manifest absent or exceeds 16KiB");
    NSData * manifest = [NSData dataWithContentsOfFile:@(argv[3])];
    error = nil;
    NSDictionary * job = [NSJSONSerialization JSONObjectWithData:manifest options:0 error:&error];
    if (![job isKindOfClass:[NSDictionary class]] || error) return fail(report, @"manifest must be a JSON object");
    NSArray * cases = job[@"cases"];
    if (![cases isKindOfClass:[NSArray class]] || cases.count != 3) return fail(report, @"exactly three bounded fixtures required");
    NSMutableArray * measurements = [NSMutableArray array];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return fail(report, @"command queue unavailable");
    NSUInteger completed = 0;
    for (NSDictionary * fixture in cases) { @autoreleasepool {
        if (![fixture isKindOfClass:[NSDictionary class]] || ![fixture[@"name"] isKindOfClass:[NSString class]] ||
            ![fixture[@"weights"] isKindOfClass:[NSString class]] || ![fixture[@"masked"] isKindOfClass:[NSString class]] ||
            ![fixture[@"output"] isKindOfClass:[NSString class]]) return fail(report, @"invalid fixture fields");
        NSData * weights = readExact(fixture[@"weights"], 16 * 32);
        NSData * masked = readExact(fixture[@"masked"], 32 * 8);
        if (!weights || !masked) return fail(report, @"fixed int8 matrix byte count mismatch");
        id<MTLBuffer> a = [device newBufferWithBytes:weights.bytes length:weights.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [device newBufferWithBytes:masked.bytes length:masked.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> c = [device newBufferWithLength:16 * 8 * sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!a || !b || !c) return fail(report, @"bounded Metal buffer allocation failed");
        // Fixed total device payload = 512 + 256 + 512 = 1280 bytes per command.
        memset(c.contents, 0x5a, c.length);
        report[@"stage"] = @"device_execute";
        report[@"device_execution_attempted"] = @YES;
        const auto wall_begin = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return fail(report, @"command encoding unavailable");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:c offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted || command.error)
            return fail(report, command.error.localizedDescription ?: @"GPU command failed");
        completed++;
        report[@"commands_completed"] = @(completed);
        const double begin = command.GPUStartTime, end = command.GPUEndTime;
        const bool valid = std::isfinite(begin) && std::isfinite(end) && begin > 0 && end >= begin;
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_begin).count();
        [measurements addObject:@{@"name": fixture[@"name"], @"valid_gpu_timestamps": @(valid),
            @"gpu_seconds": valid ? @(end - begin) : (id)[NSNull null], @"command_wall_seconds": @(wall)}];
        report[@"stage"] = @"output_write";
        NSData * output = [NSData dataWithBytes:c.contents length:c.length];
        error = nil;
        if (![output writeToFile:fixture[@"output"] options:NSDataWritingAtomic error:&error])
            return fail(report, error.localizedDescription ?: @"output write failed");
    }}
    report[@"stage"] = @"device_complete_reference_not_yet_checked";
    report[@"device_payload_bytes_per_command"] = @1280;
    report[@"commands"] = measurements;
    report[@"timing_scope"] = @"single cold/warm mixed command per fixture, no benchmark or speed ratio";
    emit(report);
    return 0;
}}
