// Diagnostic only: mincore observes mapping residency without touching weight contents.
#include <algorithm>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <limits>
#include <utility>
#if defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef AIRLLM_PROFILE_RESIDENCY_TEST
static bool test_force_mincore_error=false;
#endif

struct page_snapshot {
    std::string status = "not_inspected", reason;
    std::string weight_buffer_name, weight_buffer_type, weight_device;
    int weight_buffer_usage = -1;
    uint64_t logical_bytes = 0, pages = 0, resident = 0, nonresident = 0;
    uint64_t page_size = 0, selected_experts = 0;
    int error_number = 0;
    double inspection_seconds = 0;
};

static page_snapshot inspect_page_ranges(const std::vector<std::pair<uintptr_t,uint64_t>>& spans,
                                        uint64_t logical_bytes) {
    page_snapshot result;result.logical_bytes=logical_bytes;
#if defined(__APPLE__)
    const long page_size=sysconf(_SC_PAGESIZE);
    if(page_size<=0){result.status="error";result.reason="page_size_unavailable";return result;}
    result.page_size=uint64_t(page_size);
    if(logical_bytes>(1ULL<<30)){result.status="error";result.reason="logical_range_exceeds_1GiB";return result;}
    std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
    for(const auto& span:spans){
        if(!span.first||!span.second||span.second>(1ULL<<30)||span.first>UINTPTR_MAX-span.second||
           span.first+span.second>UINTPTR_MAX-(result.page_size-1)){
            result.status="error";result.reason="invalid_or_oversized_range";return result;
        }
        const uintptr_t start=span.first/result.page_size*result.page_size;
        const uintptr_t end=(span.first+span.second+result.page_size-1)/result.page_size*result.page_size;
        ranges.emplace_back(start,end);
    }
    std::sort(ranges.begin(),ranges.end());
    std::vector<std::pair<uintptr_t,uintptr_t>> merged;
    for(const auto& r:ranges){
        if(!merged.empty()&&r.first<=merged.back().second)merged.back().second=std::max(merged.back().second,r.second);
        else merged.push_back(r);
    }
    uint64_t page_count=0;
    for(const auto& r:merged){
        const uint64_t pages=(r.second-r.first)/result.page_size;
        if(pages>(1ULL<<20)-page_count){result.status="error";result.reason="page_vector_exceeds_1MiB";return result;}
        page_count+=pages;
    }
    // Separate spans may touch rounded boundary pages, so the merged page
    // count is authoritative; logical bytes remain the unrounded selected data.
    if(page_count*result.page_size>(1ULL<<30)){
        result.status="error";result.reason="inspected_range_exceeds_1GiB";return result;
    }
    for(const auto& r:merged){
        const size_t pages=(r.second-r.first)/result.page_size;
        std::vector<char> states(pages,0);
        int probe_result=0;
#ifdef AIRLLM_PROFILE_RESIDENCY_TEST
        if(test_force_mincore_error){errno=ENOSYS;probe_result=-1;}else
#endif
        probe_result=mincore(reinterpret_cast<void*>(r.first),r.second-r.first,states.data());
        if(probe_result!=0){
            result.status="error";result.reason="mincore_failed";result.error_number=errno;
            result.pages=result.resident=result.nonresident=0;return result;
        }
        result.pages+=pages;
        for(const auto value:states)result.resident+=(static_cast<unsigned char>(value)&MINCORE_INCORE)!=0;
    }
    result.nonresident=result.pages-result.resident;result.status="ok";
#else
    (void)spans;result.status="unsupported";result.reason="Darwin_mincore_unavailable";
#endif
    return result;
}

static std::string snapshot_json(const page_snapshot& r){
    std::ostringstream out;
    out<<"{\"status\":\""<<r.status<<"\",\"reason\":\""<<r.reason
       <<"\",\"logical_selected_weight_bytes\":"<<r.logical_bytes
       <<",\"selected_expert_count\":"<<r.selected_experts<<",\"page_size\":"<<r.page_size
       <<",\"inspection_seconds\":"<<r.inspection_seconds
       <<",\"weight_buffer_name\":"<<std::quoted(r.weight_buffer_name)
       <<",\"weight_buffer_type\":"<<std::quoted(r.weight_buffer_type)
       <<",\"weight_device\":"<<std::quoted(r.weight_device)
       <<",\"weight_buffer_usage\":"<<r.weight_buffer_usage;
    if(r.status=="ok")out<<",\"inspected_pages\":"<<r.pages<<",\"resident_pages\":"<<r.resident
                        <<",\"nonresident_pages\":"<<r.nonresident;
    else out<<",\"inspected_pages\":null,\"resident_pages\":null,\"nonresident_pages\":null";
    out<<",\"errno\":"<<r.error_number<<"}";return out.str();
}

static ggml_backend_buffer_t tensor_buffer(ggml_tensor* t){
    for(int i=0;t&&i<16;++i){if(t->buffer)return t->buffer;t=t->view_src;}
    return nullptr;
}

static bool unregistered_cpu_name(const char* name){
    return name&&(std::strcmp(name,"CPU")==0||std::strcmp(name,"CPU_Mapped")==0);
}

static ggml_backend_buffer_t host_buffer(ggml_tensor* t){
    const auto buffer=tensor_buffer(t);
    if(!buffer||!ggml_backend_buffer_is_host(buffer))return nullptr;
    const auto type=ggml_backend_buffer_get_type(buffer);
    const auto device=ggml_backend_buft_get_device(type);
    // Pinned GGML defines both CPU and CPU_Mapped with device=NULL (FIXME in
    // ggml-backend.cpp). Only those exact known host types get this exception.
    const bool eligible=device?ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU:
                               unregistered_cpu_name(ggml_backend_buft_name(type));
    return eligible?buffer:nullptr;
}

static page_snapshot tensor_weight_pages(ggml_tensor* operation){
    page_snapshot result;
    if(operation->op!=GGML_OP_MUL_MAT&&operation->op!=GGML_OP_MUL_MAT_ID){
        result.status="not_applicable";result.reason="not_weight_matmul";return result;
    }
    auto* weight=operation->src[0];
    if(const auto raw=tensor_buffer(weight)){
        const auto type=ggml_backend_buffer_get_type(raw);
        const auto device=ggml_backend_buft_get_device(type);
        result.weight_buffer_name=ggml_backend_buffer_name(raw);
        result.weight_buffer_type=ggml_backend_buft_name(type);
        result.weight_device=device?ggml_backend_dev_name(device):"unregistered";
        result.weight_buffer_usage=int(ggml_backend_buffer_get_usage(raw));
    }
    auto buffer=host_buffer(weight);
    if(!weight||!buffer||!weight->data){result.status="not_applicable";result.reason="not_CPU_host_weight";return result;}
    if(ggml_backend_buffer_get_usage(buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS){
        result.status="not_applicable";result.reason="source_buffer_not_model_weights";return result;
    }
    const uintptr_t base=reinterpret_cast<uintptr_t>(weight->data);
    const uint64_t total=ggml_nbytes(weight);
    const uintptr_t buffer_base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const uint64_t buffer_bytes=ggml_backend_buffer_get_size(buffer);
    if(total>(1ULL<<30)||base<buffer_base||base-buffer_base>buffer_bytes||total>buffer_bytes-(base-buffer_base)){
        result.status="error";result.reason="weight_outside_bounded_host_buffer";return result;
    }
    std::vector<std::pair<uintptr_t,uint64_t>> spans;
    uint64_t logical=0,selected_experts=0;
    if(operation->op==GGML_OP_MUL_MAT_ID){
        auto* ids=operation->src[2];
        if(!ids||ids->type!=GGML_TYPE_I32||!host_buffer(ids)||!ids->data||
           ggml_nbytes(ids)>(1ULL<<20)||ggml_nelements(ids)<1||ggml_nelements(ids)>65536||weight->ne[2]<1||!weight->nb[2]){
            result.status="error";result.reason="selected_IDs_unavailable_or_unbounded";return result;
        }
        const size_t id_bytes=ggml_nbytes(ids);
        std::vector<unsigned char> data(id_bytes);
        // src2 has already been computed. Reading its tiny host I32 tensor
        // observes actual IDs and never touches a weight bank.
        ggml_backend_tensor_get(ids,data.data(),0,id_bytes);
        std::vector<int32_t> chosen;
        for(int64_t i3=0;i3<ids->ne[3];++i3)for(int64_t i2=0;i2<ids->ne[2];++i2)
        for(int64_t i1=0;i1<ids->ne[1];++i1)for(int64_t i0=0;i0<ids->ne[0];++i0){
            const size_t offset=i0*ids->nb[0]+i1*ids->nb[1]+i2*ids->nb[2]+i3*ids->nb[3];
            if(offset>id_bytes||sizeof(int32_t)>id_bytes-offset){result.status="error";result.reason="invalid_ID_stride";return result;}
            int32_t id;std::memcpy(&id,data.data()+offset,sizeof(id));
            if(id<0||id>=weight->ne[2]){result.status="error";result.reason="selected_expert_outside_bank";return result;}
            chosen.push_back(id);
        }
        std::sort(chosen.begin(),chosen.end());chosen.erase(std::unique(chosen.begin(),chosen.end()),chosen.end());
        const uint64_t expert_bytes=weight->nb[2];
        for(const auto id:chosen){
            const uint64_t offset=uint64_t(id)*expert_bytes;
            if(offset>total||expert_bytes>total-offset){result.status="error";result.reason="selected_expert_outside_tensor";return result;}
            spans.emplace_back(base+offset,expert_bytes);logical+=expert_bytes;
        }
        selected_experts=chosen.size();
    }else{spans.emplace_back(base,total);logical=total;}
    auto inspected=inspect_page_ranges(spans,logical);
    inspected.selected_experts=selected_experts;
    inspected.weight_buffer_name=result.weight_buffer_name;
    inspected.weight_buffer_type=result.weight_buffer_type;
    inspected.weight_device=result.weight_device;
    inspected.weight_buffer_usage=result.weight_buffer_usage;
    return inspected;
}


