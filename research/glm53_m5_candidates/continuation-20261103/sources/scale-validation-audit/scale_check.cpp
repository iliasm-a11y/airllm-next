// Resident CPU predicate audit only; no model, GPU, dispatch or weight changes.
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using Clock=std::chrono::steady_clock;
static bool original(const void *data) {
    ggml_fp16_t bits;std::memcpy(&bits,data,2);
    const float value=ggml_fp16_to_fp32(bits);
    return std::isfinite(value)&&(value==0||std::abs(value)>=0x1p-14f);
}
static bool bit_predicate(const void *data) {
    uint16_t bits;std::memcpy(&bits,data,2);
    const unsigned exponent=bits&0x7c00u;
    return exponent!=0x7c00u && (exponent!=0 || (bits&0x03ffu)==0);
}
template<bool fast> static uint64_t scan(const uint8_t *data,size_t entries) {
    uint64_t accepted=0;
    for(size_t i=0;i<entries;++i)accepted+=fast?bit_predicate(data+34*i):original(data+34*i);
    return accepted;
}
int main() {
    uint64_t mismatch=0;
    for(uint32_t value=0;value<65536;++value) {
        const uint16_t bits=uint16_t(value);
        mismatch+=original(&bits)!=bit_predicate(&bits);
    }
    if(mismatch){std::printf("{\"status\":\"predicate_mismatch\",\"patterns\":65536,\"mismatches\":%llu}\n",(unsigned long long)mismatch);return 2;}
    constexpr size_t entries=8192*(4096/32),bytes=34*entries;
    static_assert(bytes<64u*1024*1024,"bounded resident synthetic scale payload");
    std::vector<uint8_t> data(bytes,0);
    uint32_t random=5305;
    for(size_t i=0;i<entries;++i) {
        random=random*1664525u+1013904223u;
        const uint16_t bits=uint16_t((random&0x83ffu)|(((random>>16)%30+1)<<10));
        std::memcpy(data.data()+34*i,&bits,2);
    }
    std::vector<double> old_times,new_times;volatile uint64_t sink=0;
    for(int repeat=0;repeat<12;++repeat)for(int order=0;order<2;++order) {
        const bool fast=(repeat+order)%2;
        const auto begin=Clock::now();
        const uint64_t count=fast?scan<true>(data.data(),entries):scan<false>(data.data(),entries);
        const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
        if(count!=entries)return 3;sink=sink+count;
        if(repeat>=2)(fast?new_times:old_times).push_back(elapsed);
    }
    auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return (values[4]+values[5])/2;};
    std::printf("{\"status\":\"predicate_exhaustive_equal_cpu_scan_only\",\"patterns\":65536,\"mismatches\":0,\"resident_bytes\":%zu,\"scale_entries\":%zu,\"timed_repetitions_each\":10,\"warmup_repetitions_each\":2,\"original_median_s\":%.9f,\"bits_median_s\":%.9f,\"original_samples_s\":[",bytes,entries,median(old_times),median(new_times));
    for(size_t i=0;i<old_times.size();++i)std::printf("%s%.9f",i?",":"",old_times[i]);
    std::printf("],\"bits_samples_s\":[");
    for(size_t i=0;i<new_times.size();++i)std::printf("%s%.9f",i?",":"",new_times[i]);
    std::printf("],\"checksum\":%llu,\"matrix_dispatches\":0,\"model_loaded\":false}\n",(unsigned long long)sink);
}
