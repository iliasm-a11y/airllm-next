// MIT. Persistent shared buffers/pipelines; synchronous single-token costs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>
#include <vector>
extern "C" void quantize_row_q8_0(const float*,void*,int64_t);
extern "C" void quantize_row_q8_K_ref(const float*,void*,int64_t);
static void fail(NSString* s){fprintf(stderr,"%s\n",s.UTF8String);exit(2);}
static double elapsed(std::chrono::steady_clock::time_point start){
 return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
static double median(NSArray* a){
 NSArray* v=[a sortedArrayUsingSelector:@selector(compare:)];NSUInteger n=v.count;
 return n%2?[v[n/2] doubleValue]:([v[n/2-1] doubleValue]+[v[n/2] doubleValue])*.5;}
static MTLSize dimensions(NSArray* a){
 if(a.count!=3) fail(@"three dimensions required");
 NSUInteger x=[a[0] unsignedIntegerValue],y=[a[1] unsignedIntegerValue],z=[a[2] unsignedIntegerValue];
 if(!x||!y||!z||x>(1<<20)||y!=1||z>2) fail(@"dispatch exceeds component bounds");
 return MTLSizeMake(x,y,z);}

int main(int argc,const char* argv[]){@autoreleasepool{
 if(argc!=2) fail(@"usage: decode-timer job.json");
 NSError* error=nil;
 NSData* data=[NSData dataWithContentsOfFile:@(argv[1])];
 if(!data||data.length>(1<<20)) fail(@"missing or oversized job");
 NSDictionary* job=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
 if(!job||error) fail(@"invalid job");
 NSUInteger reps=[job[@"repetitions"] unsignedIntegerValue];
 if(reps<3||reps>30) fail(@"repetitions outside3..30");
 const NSUInteger warmups=3;
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)fail(@"Metal unavailable");
 id<MTLCommandQueue> queue=[device newCommandQueue];
 NSString* source=[NSString stringWithContentsOfFile:job[@"source"] encoding:NSUTF8StringEncoding error:&error];
 if(!source||error)fail(@"source unavailable");
 MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
 if(!library||error)fail(error.localizedDescription?:@"shader compilation failed");
 NSArray* specs=job[@"buffers"];
 if(specs.count<5||specs.count>32)fail(@"buffer count exceeds bound");
 NSMutableArray<id<MTLBuffer>>* buffers=[NSMutableArray array];NSUInteger total=0;
 for(NSDictionary* spec in specs){@autoreleasepool{
  NSUInteger bytes=[spec[@"bytes"] unsignedIntegerValue];
  if(!bytes||bytes>(64<<20)||total>(64<<20)-bytes)fail(@"payload exceeds64MiB");
  total+=bytes;
  id<MTLBuffer> buffer=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if(!buffer)fail(@"buffer allocation failed");memset(buffer.contents,0,bytes);
  if(spec[@"input"]){
   NSData* input=[NSData dataWithContentsOfFile:spec[@"input"] options:NSDataReadingMappedIfSafe error:&error];
   if(!input||error||input.length!=bytes)fail(@"input size mismatch");
   memcpy(buffer.contents,input.bytes,bytes);
  }
  [buffers addObject:buffer];
 }}
 NSDictionary* quant=job[@"cpu_quantize"];
 NSUInteger qi=[quant[@"input"] unsignedIntegerValue],qo=[quant[@"output"] unsignedIntegerValue];
 NSUInteger width=[quant[@"width"] unsignedIntegerValue];int format=[quant[@"format"] intValue];
 if((format!=2&&format!=8)||width<256||width>16384||width%256||qi>=buffers.count||qo>=buffers.count||qi==qo||
    buffers[qi].length!=width*4||buffers[qo].length!=width/(format==2?256:32)*(format==2?292:34))
  fail(@"invalid native quantizer layout");
 NSArray* experiments=job[@"experiments"];
 if(experiments.count<1||experiments.count>12)fail(@"experiment count exceeds bound");
 NSMutableArray* pipelines=[NSMutableArray array];
 NSMutableDictionary* results=[NSMutableDictionary dictionary];
 for(NSDictionary* experiment in experiments){
  NSString* name=experiment[@"name"];NSArray* stages=experiment[@"stages"];
  if(!name||results[name]||stages.count<1||stages.count>2)fail(@"invalid experiment");
  results[name]=[@{@"gpu_seconds":[NSMutableArray array],@"total_wall_seconds":[NSMutableArray array],
                   @"cpu_quantize_seconds":[NSMutableArray array]} mutableCopy];
  NSMutableArray* ps=[NSMutableArray array];
  for(NSDictionary* stage in stages){
   id<MTLFunction> f=[library newFunctionWithName:stage[@"entry"]];if(!f)fail(@"missing entry");
   id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&error];
   if(!p||error)fail(error.localizedDescription?:@"pipeline failed");
   [ps addObject:p];
  }
  [pipelines addObject:ps];
 }
 std::mt19937 random(53031);std::vector<NSUInteger> order(experiments.count);
 std::iota(order.begin(),order.end(),0);NSMutableArray* orders=[NSMutableArray array];
 for(NSUInteger rep=0;rep<reps+warmups;++rep){
  std::shuffle(order.begin(),order.end(),random);NSMutableArray* recorded=[NSMutableArray array];
  for(NSUInteger ei:order){@autoreleasepool{
   NSDictionary* ex=experiments[ei];NSArray* stages=ex[@"stages"];
   bool separate=[ex[@"separate_commands"] boolValue];
   const auto begin=std::chrono::steady_clock::now();
   const auto qb=std::chrono::steady_clock::now();
   if(format==8)quantize_row_q8_0((const float*)buffers[qi].contents,buffers[qo].contents,width);
   else quantize_row_q8_K_ref((const float*)buffers[qi].contents,buffers[qo].contents,width);
   const double quant_seconds=elapsed(qb);double gpu_seconds=0;
   id<MTLCommandBuffer> command=[queue commandBuffer];
   for(NSUInteger si=0;si<stages.count;++si){
    NSDictionary* s=stages[si];id<MTLComputePipelineState> p=pipelines[ei][si];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:p];
    NSArray* bindings=s[@"bindings"];if(bindings.count!=5)fail(@"five shader bindings required");
    for(NSUInteger i=0;i<bindings.count;++i){
     NSUInteger bi=[bindings[i] unsignedIntegerValue];if(bi>=buffers.count)fail(@"invalid binding");
     [encoder setBuffer:buffers[bi] offset:0 atIndex:i];
    }
    MTLSize grid=dimensions(s[@"grid"]),group=dimensions(s[@"threadgroup"]);
    if(group.width*group.height*group.depth>p.maxTotalThreadsPerThreadgroup||group.width%p.threadExecutionWidth)
     fail(@"invalid SIMD group");
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];[encoder endEncoding];
    if(separate||si+1==stages.count){
     [command commit];[command waitUntilCompleted];
     const double b=command.GPUStartTime,e=command.GPUEndTime;
     if(command.status!=MTLCommandBufferStatusCompleted||command.error||!std::isfinite(b)||!std::isfinite(e)||b<=0||e<b)
      fail(command.error.localizedDescription?:@"command/timestamps failed");
     gpu_seconds+=e-b;
     if(si+1<stages.count)command=[queue commandBuffer];
    }
   }
   const double wall=elapsed(begin);
   if(rep>=warmups){
    NSMutableDictionary* r=results[ex[@"name"]];
    [r[@"gpu_seconds"] addObject:@(gpu_seconds)];[r[@"total_wall_seconds"] addObject:@(wall)];
    [r[@"cpu_quantize_seconds"] addObject:@(quant_seconds)];[recorded addObject:ex[@"name"]];
   }
  }}
  if(rep>=warmups)[orders addObject:recorded];
 }
 for(NSUInteger i=0;i<specs.count;++i)if(specs[i][@"output"]){
  NSData* out=[NSData dataWithBytes:buffers[i].contents length:buffers[i].length];
  if(![out writeToFile:specs[i][@"output"] options:NSDataWritingAtomic error:&error])fail(@"output failed");
 }
 for(NSDictionary* ex in experiments){
  NSMutableDictionary* r=results[ex[@"name"]];
  r[@"median_gpu_seconds"]=@(median(r[@"gpu_seconds"]));
  r[@"median_total_wall_seconds"]=@(median(r[@"total_wall_seconds"]));
  r[@"median_cpu_quantize_seconds"]=@(median(r[@"cpu_quantize_seconds"]));
  r[@"command_buffers_per_invocation"]=@([ex[@"separate_commands"] boolValue]?[ex[@"stages"] count]:1);
  r[@"dispatches_per_invocation"]=@([ex[@"stages"] count]);
 }
 NSDictionary* report=@{@"device":device.name,@"experiments":results,@"randomized_order":orders,
  @"payload_bytes":@(total),@"warmups":@(warmups),@"math_mode":@"safe",
  @"scope":@"resident original-format synthetic component; total wall includes native CPU quantization, submission and wait; setup/uploads excluded"};
 NSData* json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
 fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);return 0;
}}
