#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(void) {
    @autoreleasepool {
        id<MTLDevice> d=MTLCreateSystemDefaultDevice();
        if (!d) { puts("{\"metal_device_available\":false}"); return 0; }
        NSDictionary *data=@{@"metal_device_available":@YES,@"name":d.name,
            @"unified_memory":@(d.hasUnifiedMemory),
            @"recommended_working_set_bytes":@(d.recommendedMaxWorkingSetSize)};
        NSData *json=[NSJSONSerialization dataWithJSONObject:data options:0 error:nil];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
    }
    return 0;
}
