// Private, default-off CPU mmap advice; no weight copies or arithmetic changes.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

namespace airllm_dense {
constexpr size_t max_window = size_t(128) << 20;
constexpr size_t default_window = size_t(32) << 20;
struct entry { int node; void * address; size_t bytes; };
struct stats {
    size_t queued=0, calls=0, advised_bytes=0, failures=0, busy_skips=0, admission_skips=0, cancelled=0;
    int first_error=0;
    double syscall_seconds=0, join_seconds=0;
    uint64_t baseline_compressor=0,max_compressor_growth=0,min_available=UINT64_MAX;
    int last_pressure=0;
};
struct hooks { // Test injection never bypasses immutable-buffer/range checks.
    bool (*admit)(size_t,void *)=nullptr;
    int (*advise)(void *,size_t,void *)=nullptr;
    void * data=nullptr;
};
class scope;
inline std::mutex & registry_mutex() { static std::mutex m; return m; }
inline scope * & active() { static scope * value=nullptr; return value; }
inline std::atomic<bool> & any_active() {static std::atomic<bool> value{false};return value;}

inline bool readonly_range(const void * data,size_t bytes) {
    uintptr_t cursor=reinterpret_cast<uintptr_t>(data);
    if(!cursor||!bytes||bytes>(size_t(1)<<30)||cursor>UINTPTR_MAX-bytes)return false;
    const uintptr_t end=cursor+bytes;
    for(int i=0;cursor<end&&i<64;++i){
        mach_vm_address_t address=cursor;mach_vm_size_t length=0;
        vm_region_basic_info_data_64_t info{};mach_msg_type_number_t count=VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object=MACH_PORT_NULL;
        const auto rc=mach_vm_region(mach_task_self(),&address,&length,VM_REGION_BASIC_INFO_64,
            reinterpret_cast<vm_region_info_t>(&info),&count,&object);
        if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
        if(rc!=KERN_SUCCESS||address>cursor||!length||address>UINTPTR_MAX-length||address+length<=cursor||
           !(info.protection&VM_PROT_READ)||(info.protection&VM_PROT_WRITE))return false;
        cursor=std::min<uintptr_t>(end,address+length);
    }
    return cursor==end;
}
inline bool window_for(const ggml_tensor * w,size_t limit,entry & out) {
    if(!w||!w->data||!w->buffer||!ggml_is_contiguous(w)||!limit||limit>max_window||
       ggml_backend_buffer_get_usage(w->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS||
       std::strcmp(ggml_backend_buffer_name(w->buffer),"CPU_Mapped"))return false;
    const ggml_tensor * leaf=w;
    for(int depth=0;depth<16;++depth){
        if(leaf->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT|GGML_TENSOR_FLAG_COMPUTE))return false;
        if(leaf->op==GGML_OP_NONE)break;
        if(!leaf->view_src||(leaf->op!=GGML_OP_VIEW&&leaf->op!=GGML_OP_RESHAPE&&
           leaf->op!=GGML_OP_TRANSPOSE&&leaf->op!=GGML_OP_PERMUTE))return false;
        leaf=leaf->view_src;
    }
    if(leaf->op!=GGML_OP_NONE||(leaf->flags&(GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT|GGML_TENSOR_FLAG_COMPUTE)))return false;
    const uintptr_t base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(w->buffer));
    const uintptr_t data=reinterpret_cast<uintptr_t>(w->data);
    const size_t bytes=ggml_nbytes(w),size=ggml_backend_buffer_get_size(w->buffer);
    const long page_value=sysconf(_SC_PAGESIZE);
    if(!base||data<base||data-base>size||!bytes||bytes>size-(data-base)||page_value<=0||
       !readonly_range(w->data,bytes))return false;
    const size_t page=size_t(page_value);
    const uintptr_t aligned=data-data%page;
    if(limit<page)return false;
    const size_t cap=limit-limit%page;
    if(data-aligned>=cap)return false;
    const size_t wanted=std::min(bytes,cap-size_t(data-aligned));
    if(data>UINTPTR_MAX-wanted)return false;
    const uintptr_t end=data+wanted;
    const size_t pad=(page-end%page)%page;
    if(end>UINTPTR_MAX-pad||end+pad-aligned>cap||!readonly_range(reinterpret_cast<void *>(aligned),end+pad-aligned))return false;
    out.address=reinterpret_cast<void *>(aligned);out.bytes=end+pad-aligned;return true;
}
struct memory_sample {bool valid=false;int pressure=0;uint64_t available=0,compressor=0;};
inline memory_sample memory() {
    memory_sample sample;
    int pressure=0;size_t length=sizeof(pressure);
    if(sysctlbyname("kern.memorystatus_vm_pressure_level",&pressure,&length,nullptr,0))return sample;
    vm_statistics64_data_t vm{};mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    mach_port_t host=mach_host_self();
    const auto rc=host_statistics64(host,HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&vm),&count);
    mach_port_deallocate(mach_task_self(),host);
    const long page=sysconf(_SC_PAGESIZE);
    if(rc!=KERN_SUCCESS||page<=0)return sample;
    sample.valid=true;sample.pressure=pressure;
    sample.available=(uint64_t(vm.free_count)+vm.inactive_count)*uint64_t(page);
    sample.compressor=uint64_t(vm.compressor_page_count)*uint64_t(page);
    return sample;
}

class scope {
    const ggml_cgraph * graph_;
    hooks hooks_;
    std::vector<entry> entries_;
    std::mutex mutex_;
    std::mutex cancel_mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool enabled_=false,stopping_=false,pending_=false,busy_=false;
    entry job_{};
    int last_attempt_=-1;
    size_t limit_=default_window;
    stats stats_;
    void work() {
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;){
            cv_.wait(lock,[this]{return stopping_||pending_;});
            if(stopping_)return;
            const entry current=job_;pending_=false;busy_=true;lock.unlock();
            const auto start=std::chrono::steady_clock::now();
            const int result=hooks_.advise?hooks_.advise(current.address,current.bytes,hooks_.data):
                posix_madvise(current.address,current.bytes,POSIX_MADV_WILLNEED);
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            lock.lock();++stats_.calls;stats_.advised_bytes+=current.bytes;stats_.syscall_seconds+=elapsed;
            if(result){++stats_.failures;if(!stats_.first_error)stats_.first_error=result;}
            busy_=false;
        }
    }
public:
    explicit scope(ggml_cgraph * graph,hooks injected={},size_t limit=0):graph_(graph),hooks_(injected) {
        const char * flag=std::getenv("AIRLLM_DENSE_LOOKAHEAD");
        if(!flag||std::strcmp(flag,"1")||!graph)return;
        if(!limit){
            limit=default_window;
            const char * mib=std::getenv("AIRLLM_DENSE_LOOKAHEAD_MIB");
            if(mib){
                if(!std::strcmp(mib,"16"))limit=size_t(16)<<20;
                else if(!std::strcmp(mib,"32"))limit=size_t(32)<<20;
                else if(!std::strcmp(mib,"64"))limit=size_t(64)<<20;
                else if(!std::strcmp(mib,"128"))limit=max_window;
                else return;
            }
        }
        if(limit>max_window)return;limit_=limit;
        try {
            if(!hooks_.admit){const auto sample=memory();if(!sample.valid)return;stats_.baseline_compressor=sample.compressor;}
            const int n=ggml_graph_n_nodes(graph);if(n<1||n>65536)return;
            entries_.reserve(1024);
            for(int i=0;i<n;++i){
                const auto * node=ggml_graph_node(graph,i);
                if(node->op!=GGML_OP_MUL_MAT||!node->src[1]||node->src[1]->ne[1]!=1||
                   !(node->flags&GGML_TENSOR_FLAG_COMPUTE))continue;
                entry e{i,nullptr,0};if(!window_for(node->src[0],limit,e))continue;
                if(entries_.size()==1024){entries_.clear();return;}entries_.push_back(e);
            }
            if(entries_.size()<2)return;
            std::lock_guard<std::mutex> registry(registry_mutex());
            if(active()){std::fprintf(stderr,"AIRLLM_DENSE_LOOKAHEAD status=concurrent_graph_skipped\n");return;}
            active()=this;
            try {worker_=std::thread(&scope::work,this);enabled_=true;any_active().store(true,std::memory_order_release);}
            catch(...){active()=nullptr;}
        } catch(...) {} // Advisory failure never changes model arithmetic.
    }
    scope(const scope &)=delete;scope & operator=(const scope &)=delete;
    bool enabled() const {return enabled_;}
    const ggml_cgraph * graph() const {return graph_;}
    void advance(int node) {
        if(!enabled_)return;
        auto it=std::upper_bound(entries_.begin(),entries_.end(),node,[](int value,const entry & e){return value<e.node;});
        if(it==entries_.end())return;
        std::lock_guard<std::mutex> lock(mutex_);
        if(stopping_||it->node<=last_attempt_)return;
        if(busy_||pending_){++stats_.busy_skips;return;}
        last_attempt_=it->node;
        bool admitted=false;
        if(hooks_.admit)admitted=hooks_.admit(it->bytes,hooks_.data);
        else {
            const auto sample=memory();stats_.last_pressure=sample.pressure;
            if(sample.valid){
                stats_.min_available=std::min(stats_.min_available,sample.available);
                const uint64_t growth=sample.compressor>stats_.baseline_compressor?sample.compressor-stats_.baseline_compressor:0;
                stats_.max_compressor_growth=std::max(stats_.max_compressor_growth,growth);
                admitted=sample.pressure==1&&sample.available>=(uint64_t(2)<<30)+it->bytes&&growth<(uint64_t(128)<<20);
            }
        }
        if(!admitted){++stats_.admission_skips;return;}
        job_=*it;pending_=true;++stats_.queued;cv_.notify_one();
    }
    void cancel_and_join() {
        if(!enabled_)return;
        std::lock_guard<std::mutex> cancellation(cancel_mutex_);
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;if(pending_){++stats_.cancelled;pending_=false;}cv_.notify_all();}
        const auto start=std::chrono::steady_clock::now();
        if(worker_.joinable())worker_.join();
        {std::lock_guard<std::mutex> lock(mutex_);stats_.join_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
        {std::lock_guard<std::mutex> registry(registry_mutex());if(active()==this){active()=nullptr;any_active().store(false,std::memory_order_release);}}
    }
    stats snapshot() {std::lock_guard<std::mutex> lock(mutex_);return stats_;}
    ~scope() {
        if(!enabled_)return;cancel_and_join();
        std::fprintf(stderr,"AIRLLM_DENSE_LOOKAHEAD planned=%zu queued=%zu calls=%zu advised_bytes=%zu failures=%zu first_error=%d busy_skips=%zu admission_skips=%zu cancelled=%zu syscall_seconds=%.9f join_seconds=%.9f window_cap=%zu hard_cap=%zu baseline_compressor=%llu compressor_growth=%llu min_available=%llu pressure=%d headroom_floor=%llu\n",
            entries_.size(),stats_.queued,stats_.calls,stats_.advised_bytes,stats_.failures,stats_.first_error,
            stats_.busy_skips,stats_.admission_skips,stats_.cancelled,stats_.syscall_seconds,stats_.join_seconds,limit_,max_window,
            (unsigned long long)stats_.baseline_compressor,(unsigned long long)stats_.max_compressor_growth,
            (unsigned long long)(stats_.min_available==UINT64_MAX?0:stats_.min_available),stats_.last_pressure,2ULL<<30);
    }
};
inline void notify(const ggml_cgraph * graph,int node) noexcept {
    if(!any_active().load(std::memory_order_acquire))return;
    try {std::lock_guard<std::mutex> lock(registry_mutex());if(active()&&active()->graph()==graph)active()->advance(node);}catch(...){}
}
} // namespace airllm_dense
