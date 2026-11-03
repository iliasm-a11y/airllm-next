#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOKit/ps/IOPowerSources.h>
#import <IOKit/ps/IOPSKeys.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include "ggml.h"
struct sample { bool ac; int pressure; uint64_t rss, swap, available; };
sample resources() {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    CFStringRef source = info ? IOPSGetProvidingPowerSourceType(info) : nullptr;
    const bool ac = source && CFEqual(source, CFSTR(kIOPSACPowerValue));
    if (info) CFRelease(info);
    int pressure = 0; size_t bytes = sizeof(pressure);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &bytes, nullptr, 0) != 0 || bytes != sizeof(pressure))
        GGML_ABORT("AIRLLM_CPU_METAL_Q8 resource pressure query failed");
    struct xsw_usage swap = {}; bytes = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &bytes, nullptr, 0) != 0 || bytes != sizeof(swap))
        GGML_ABORT("AIRLLM_CPU_METAL_Q8 resource swap query failed");
    mach_task_basic_info_data_t task = {}; mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &task, &count) != KERN_SUCCESS)
        GGML_ABORT("AIRLLM_CPU_METAL_Q8 resource RSS query failed");
    vm_statistics64_data_t vm = {}; count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    const kern_return_t result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t) &vm, &count);
    mach_port_deallocate(mach_task_self(), host);
    const long page = sysconf(_SC_PAGESIZE);
    if (result != KERN_SUCCESS || page <= 0)
        GGML_ABORT("AIRLLM_CPU_METAL_Q8 available-memory query failed");
    // Matches the installed psutil7.2.2 resource_snapshot branch: available is
    // inactive + free (before psutil adjusts its separate display of free).
    const uint64_t pages = uint64_t(vm.free_count) + uint64_t(vm.inactive_count);
    if (pages > UINT64_MAX / uint64_t(page))
        GGML_ABORT("AIRLLM_CPU_METAL_Q8 available-memory counter overflow");
    return { ac, pressure, (uint64_t) task.resident_size, swap.xsu_used, pages * uint64_t(page) };
}


int main(){std::vector<double> samples; bool ac=true; for(int i=0;i<22;++i){const auto begin=std::chrono::steady_clock::now();const auto value=resources();const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();if(i>=2)samples.push_back(elapsed);ac=ac&&value.ac;}std::sort(samples.begin(),samples.end());printf("{\"status\":\"resource_queries_only\",\"sample_count\":20,\"median_s\":%.9f,\"min_s\":%.9f,\"max_s\":%.9f,\"ac_all\":%s,\"matrix_dispatches\":0}\n",(samples[9]+samples[10])/2,samples.front(),samples.back(),ac?"true":"false");return ac?0:2;}
