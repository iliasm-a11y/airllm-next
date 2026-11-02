// MIT. Bounded standalone Metal timing adapter for generated research kernels.
// Reports command-buffer GPU timestamps; no CPU duration is labelled GPU time.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>

static void fail(NSString* text) { fprintf(stderr, "%s\n", text.UTF8String); exit(2); }
static MTLSize size3(NSArray* a) {
    if (a.count != 3) fail(@"dispatch dimensions must have three entries");
    NSUInteger x=[a[0] unsignedIntegerValue], y=[a[1] unsignedIntegerValue], z=[a[2] unsignedIntegerValue];
    if (!x || !y || !z || x>(1<<23) || y>(1<<16) || z>8) fail(@"dispatch outside timer bound");
    return MTLSizeMake(x,y,z);
}
int main(int argc, const char* argv[]) { @autoreleasepool {
    if (argc!=2) fail(@"usage: q2-metal-timer manifest.json");
    NSError* error=nil;
    NSData* configuration=[NSData dataWithContentsOfFile:@(argv[1])];
    NSDictionary* job=[NSJSONSerialization JSONObjectWithData:configuration options:0 error:&error];
    if (!job || error) fail(@"invalid timing manifest");
    NSUInteger repetitions=[job[@"repetitions"] unsignedIntegerValue];
    NSUInteger warmups=job[@"warmups"] ? [job[@"warmups"] unsignedIntegerValue] : 4;
    if (warmups<1 || warmups>30) fail(@"warmups outside1..30");
    if (repetitions<1 || repetitions>30) fail(@"repetitions outside1..30");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if (!device) fail(@"Metal unavailable");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    NSString* source=[NSString stringWithContentsOfFile:job[@"source"] encoding:NSUTF8StringEncoding error:&error];
    if (!source || error) fail(@"kernel source absent");
    MTLCompileOptions* options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library || error) fail(error.localizedDescription ?: @"Metal library compile failed");
    NSMutableArray<id<MTLBuffer>>* buffers=[NSMutableArray array];
    NSUInteger total=0;
    for (NSDictionary* b in job[@"buffers"]) { @autoreleasepool {
        NSUInteger bytes=[b[@"bytes"] unsignedIntegerValue];
        if (!bytes || bytes>(64<<20) || total>(64<<20)-bytes) fail(@"buffer payload exceeds64MiB");
        total+=bytes;
        id<MTLBuffer> buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) fail(@"Metal allocation failed");
        memset(buffer.contents,0,bytes);
        if (b[@"input"]) {
            NSData* data=[NSData dataWithContentsOfFile:b[@"input"] options:NSDataReadingMappedIfSafe error:&error];
            if (!data || data.length!=bytes) fail(@"input payload size mismatch");
            memcpy(buffer.contents,data.bytes,bytes);
        }
        [buffers addObject:buffer];
    }}
    const bool suite=job[@"experiments"]!=nil;
    NSArray* experiments=suite ? job[@"experiments"] : @[@{@"name":@"single",@"stages":job[@"stages"]}];
    if (experiments.count<1 || experiments.count>16) fail(@"experiment count outside1..16");
    NSMutableArray* all_pipelines=[NSMutableArray array];
    NSMutableDictionary* samples=[NSMutableDictionary dictionary];
    NSMutableDictionary* wall_samples=[NSMutableDictionary dictionary];
    for (NSDictionary* experiment in experiments) {
        NSString* name=experiment[@"name"];
        if (!name || samples[name]) fail(@"missing/duplicate experiment name");
        samples[name]=[NSMutableArray array]; wall_samples[name]=[NSMutableArray array];
        NSArray* stages=experiment[@"stages"];
        if (stages.count<1 || stages.count>2) fail(@"stage count outside1..2");
        NSMutableArray* pipelines=[NSMutableArray array];
        for (NSDictionary* stage in stages) {
            id<MTLFunction> function=[library newFunctionWithName:stage[@"entry"]];
            if (!function) fail(@"kernel entry absent");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline || error) fail(error.localizedDescription ?: @"pipeline compile failed");
            [pipelines addObject:pipeline];
        }
        [all_pipelines addObject:pipelines];
    }
    std::mt19937 random(5305);
    std::vector<NSUInteger> order(experiments.count);std::iota(order.begin(),order.end(),0);
    NSMutableArray* recorded_orders=[NSMutableArray array];
    for (NSUInteger r=0; r<repetitions+warmups; ++r) {
        std::shuffle(order.begin(),order.end(),random);
        NSMutableArray* recorded_order=[NSMutableArray array];
        for (NSUInteger ei : order) { @autoreleasepool {
            NSDictionary* experiment=experiments[ei];
            NSString* name=experiment[@"name"];
            NSArray* stages=experiment[@"stages"];
            NSArray<id<MTLComputePipelineState>>* pipelines=all_pipelines[ei];
            const auto wall_begin=std::chrono::steady_clock::now();
            id<MTLCommandBuffer> command=[queue commandBuffer];
            for (NSUInteger si=0; si<stages.count; ++si) {
                NSDictionary* stage=stages[si];
                id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                [encoder setComputePipelineState:pipelines[si]];
                NSArray* bindings=stage[@"bindings"];
                for (NSUInteger i=0;i<bindings.count;++i) {
                    NSUInteger bi=[bindings[i] unsignedIntegerValue];
                    if (bi>=buffers.count) fail(@"invalid buffer binding");
                    [encoder setBuffer:buffers[bi] offset:0 atIndex:i];
                }
                MTLSize grid=size3(stage[@"grid"]), group=size3(stage[@"threadgroup"]);
                if (group.width*group.height*group.depth>pipelines[si].maxTotalThreadsPerThreadgroup)
                    fail(@"threadgroup exceeds pipeline capability");
                [encoder dispatchThreads:grid threadsPerThreadgroup:group];[encoder endEncoding];
            }
            [command commit];[command waitUntilCompleted];
            if (command.status!=MTLCommandBufferStatusCompleted || command.error)
                fail(command.error.localizedDescription ?: @"Metal command failed");
            double begin=command.GPUStartTime,end=command.GPUEndTime;
            double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_begin).count();
            if (!std::isfinite(begin)||!std::isfinite(end)||begin<=0||end<begin)
                fail(@"valid GPU timestamps unavailable; refusing CPU timing substitute");
            if (r>=warmups) {
                [samples[name] addObject:@(end-begin)];[wall_samples[name] addObject:@(wall)];
                [recorded_order addObject:name];
            }
        }}
        if(r>=warmups) [recorded_orders addObject:recorded_order];
    }
    for (NSUInteger i=0;i<buffers.count;++i) {
        NSDictionary* spec=job[@"buffers"][i];
        if (spec[@"output"]) {
            NSData* data=[NSData dataWithBytes:buffers[i].contents length:buffers[i].length];
            if (![data writeToFile:spec[@"output"] options:NSDataWritingAtomic error:&error])
                fail(@"output write failed");
        }
    }
    NSMutableDictionary* results=[NSMutableDictionary dictionary];
    auto median=[](NSArray* input) {
        NSArray* v=[input sortedArrayUsingSelector:@selector(compare:)];NSUInteger n=v.count;
        return n%2 ? [v[n/2] doubleValue] : ([v[n/2-1] doubleValue]+[v[n/2] doubleValue])*0.5;
    };
    for (NSDictionary* experiment in experiments) {
        NSString* name=experiment[@"name"];
        results[name]=@{@"device":device.name,@"clock":@"MTLCommandBuffer.GPUEndTime-GPUStartTime",
            @"gpu_seconds":samples[name],@"median_gpu_seconds":@(median(samples[name])),
            @"command_wall_seconds":wall_samples[name],@"median_command_wall_seconds":@(median(wall_samples[name])),
            @"payload_bytes":@(total),@"warmup_commands":@(warmups),
            @"stages_per_command":@([experiment[@"stages"] count])};
    }
    NSDictionary* report=suite ? @{@"experiments":results,@"randomized_order":recorded_orders,
                                    @"device":device.name,@"payload_bytes":@(total)} : results[@"single"];
    NSData* json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    fwrite(json.bytes,1,json.length,stdout); fputc('\n',stdout);
    return 0;
}}
