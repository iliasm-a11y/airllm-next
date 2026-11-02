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

#ifdef AIRLLM_PROFILE_RESIDENCY_TEST
int main(){
#if defined(__APPLE__)
    const size_t page=sysconf(_SC_PAGESIZE);char path[]="/tmp/airllm-mincore-test-XXXXXX";
    const int fd=mkstemp(path);if(fd<0||ftruncate(fd,3*page)!=0)return 2;
    void* mapping=mmap(nullptr,3*page,PROT_NONE,MAP_PRIVATE,fd,0);unlink(path);close(fd);
    if(mapping==MAP_FAILED)return 3;
    // PROT_NONE makes an accidental weight read crash, yet mincore must work.
    auto observed=inspect_page_ranges({{uintptr_t(mapping)+1,2*page-2},{uintptr_t(mapping)+page,page}},3*page-2);
    std::puts(snapshot_json(observed).c_str());
    auto rejected=inspect_page_ranges({{uintptr_t(mapping),(1ULL<<30)+1}},(1ULL<<30)+1);
    std::puts(snapshot_json(rejected).c_str());
    munmap(mapping,3*page);
    test_force_mincore_error=true;
    auto unavailable=inspect_page_ranges({{uintptr_t(1),page}},page);
    std::puts(snapshot_json(unavailable).c_str());
    return observed.status=="ok"&&observed.pages==2&&rejected.status=="error"&&unavailable.status=="error"?0:4;
#else
    std::puts(snapshot_json(inspect_page_ranges({},0)).c_str());return 0;
#endif
}
#else
// Experimental placement diagnostic. Captures serialize operations; not a throughput benchmark.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <regex>
#include <cstdlib>
#include <cmath>
#if defined(__APPLE__)
#include <libproc.h>
#include <mach/mach_time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#include <string>
#include <vector>

struct capture {
    std::filesystem::path directory;
    std::ofstream manifest;
    int step = 0;
    int serial = 0;
    bool profile = false;
    bool profile_pages = false;
    page_snapshot weight_pages;
    bool failed = false;
    std::ofstream operations;
    std::chrono::steady_clock::time_point operation_start;
#if defined(__APPLE__)
    rusage_info_v2 before{};
    bool before_valid = false;
#endif
};


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


#ifdef AIRLLM_PROFILE_TENSOR_TEST
int main(){
#if defined(__APPLE__)
    const size_t page=sysconf(_SC_PAGESIZE);
    ggml_init_params params{1<<20,nullptr,true};auto* context=ggml_init(params);
    auto* weights=ggml_new_tensor_3d(context,GGML_TYPE_Q2_K,256,256,4);
    const size_t bytes=ggml_nbytes(weights),mapped=(bytes+page-1)/page*page;
    char path[]="/tmp/airllm-weight-pages-test-XXXXXX";
    const int fd=mkstemp(path);if(fd<0||ftruncate(fd,mapped)!=0)return 2;
    void* data=mmap(nullptr,mapped,PROT_NONE,MAP_PRIVATE,fd,0);unlink(path);close(fd);
    if(data==MAP_FAILED)return 3;
    auto buffer=ggml_backend_cpu_buffer_from_ptr(data,mapped);
    ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weights->buffer=buffer;weights->data=data;
    auto* ids=ggml_new_tensor_2d(context,GGML_TYPE_I32,2,2);
    ids->nb[1]=16;ids->nb[2]=ids->nb[3]=32;
    auto id_buffer=ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),64);
    ids->buffer=id_buffer;ids->data=ggml_backend_buffer_get_base(id_buffer);
    const int32_t chosen[]={1,3,0,0,3,1};
    ggml_backend_tensor_set(ids,chosen,0,sizeof(chosen));
    ggml_tensor operation{};operation.op=GGML_OP_MUL_MAT_ID;
    operation.src[0]=weights;operation.src[2]=ids;
    const auto selected=tensor_weight_pages(&operation);std::puts(snapshot_json(selected).c_str());
    operation.op=GGML_OP_MUL_MAT;
    const auto dense=tensor_weight_pages(&operation);std::puts(snapshot_json(dense).c_str());
    operation.op=GGML_OP_MUL_MAT_ID;
    const int32_t bad=4;ggml_backend_tensor_set(ids,&bad,0,sizeof(bad));
    const auto invalid=tensor_weight_pages(&operation);std::puts(snapshot_json(invalid).c_str());
    const bool passed=selected.status=="ok"&&selected.selected_experts==2&&selected.logical_bytes==2*weights->nb[2]&&
        dense.status=="ok"&&dense.logical_bytes==bytes&&selected.pages<dense.pages&&
        selected.weight_buffer_type=="CPU_Mapped"&&selected.weight_device=="unregistered"&&
        invalid.status=="error"&&invalid.reason=="selected_expert_outside_bank"&&
        unregistered_cpu_name("CPU")&&unregistered_cpu_name("CPU_Mapped")&&
        !unregistered_cpu_name("MTL0_Mapped")&&!unregistered_cpu_name("CPU_Evil");
    ggml_backend_buffer_free(id_buffer);ggml_backend_buffer_free(buffer);munmap(data,mapped);ggml_free(context);
    return passed?0:4;
#else
    return 0;
#endif
}
#else
static bool evaluate(ggml_tensor * tensor, bool ask, void * user) {
    const std::string name = ggml_get_name(tensor);
    const char * trace_detail = std::getenv("AIRLLM_PROFILE_DETAIL");
    const bool detail = trace_detail && std::strcmp(trace_detail, "1") == 0 &&
        (name.find("-40") != std::string::npos || name == "result_norm");
    const bool selected = (detail && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32)) || name.find("ffn_moe_topk-") == 0 ||
                          name.find("ffn_moe_logits-") == 0 ||
                          name.find("kda_scan_out-") == 0 || name == "result_output";
    auto & out = *static_cast<capture *>(user);
    if (out.profile) {
        if (ask) {
            if(out.profile_pages){
                const auto inspection_begin=std::chrono::steady_clock::now();
                out.weight_pages=tensor_weight_pages(tensor);
                out.weight_pages.inspection_seconds=std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-inspection_begin).count();
            }
#if defined(__APPLE__)
            out.before_valid = proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                reinterpret_cast<rusage_info_t *>(&out.before)) == 0;
#endif
            out.operation_start = std::chrono::steady_clock::now();
        } else {
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now-out.operation_start).count();
            out.operations << "{\"step\":" << out.step << ",\"name\":\"" << name
                << "\",\"op\":\"" << ggml_op_name(tensor->op) << "\",\"seconds\":" << seconds;
#if defined(__APPLE__)
            rusage_info_v2 after{};
            if (out.before_valid && proc_pid_rusage(getpid(), RUSAGE_INFO_V2,
                    reinterpret_cast<rusage_info_t *>(&after)) == 0) {
                mach_timebase_info_data_t scale{};
                mach_timebase_info(&scale);
                const double tick_seconds = double(scale.numer)/scale.denom/1e9;
                out.operations << ",\"cpu_seconds\":" << (after.ri_user_time+after.ri_system_time-
                    out.before.ri_user_time-out.before.ri_system_time)*tick_seconds
                    << ",\"os_read_bytes\":" << after.ri_diskio_bytesread-out.before.ri_diskio_bytesread
                    << ",\"pageins\":" << after.ri_pageins-out.before.ri_pageins;
            }
#endif
            if(out.profile_pages) out.operations << ",\"weight_pages_before\":" << snapshot_json(out.weight_pages);
            out.operations << "}\n";
        }
    }
    if (ask) {
        return selected || out.profile;
    }
    if (!selected) {
        return true;
    }
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes > 64 * 1024 * 1024 || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32)) {
        std::fprintf(stderr, "Unsupported diagnostic tensor: %s\n", name.c_str());
        out.failed = true;
        return false;
    }
    std::vector<unsigned char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    const std::string file = std::to_string(out.step) + "-" + std::to_string(out.serial++) + ".bin";
    std::ofstream stream(out.directory / file, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(reinterpret_cast<const char *>(data.data()), data.size());
    out.manifest << "{\"step\":" << out.step << ",\"name\":\"" << name << "\",\"file\":\"" << file
                 << "\",\"op\":\"" << ggml_op_name(tensor->op) << "\",\"type\":" << int(tensor->type) << ",\"bytes\":" << bytes << ",\"shape\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.manifest << (i ? "," : "") << tensor->ne[i];
    }
    out.manifest << "],\"strides\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.manifest << (i ? "," : "") << tensor->nb[i];
    }
    out.manifest << "]}\n";
    out.manifest.flush();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 9) {
        std::fprintf(stderr, "usage: native_fidelity model.gguf output_dir threads gpu_layers tokens [prompt [forced_tokens.jsonl [cpu|head-only|weights-only|legacy]]]\n");
        return 2;
    }
    const int threads = std::stoi(argv[3]);
    const int gpu_layers = std::stoi(argv[4]);
    const int count = std::stoi(argv[5]);
    if (threads < 1 || threads > 10 || gpu_layers < 0 || gpu_layers > 46 || count < 1 || count > 64) {
        return 2;
    }
    const std::string policy = argc == 9 ? argv[8] : (gpu_layers ? "weights-only" : "cpu");
    const bool head_only = policy == "head-only";
    const bool legacy = policy == "legacy";
    if ((policy != "cpu" && !head_only && policy != "weights-only" && !legacy) ||
        (policy == "cpu" && gpu_layers != 0) || (head_only && gpu_layers != 0) ||
        ((policy == "weights-only" || legacy) && gpu_layers == 0)) {
        throw std::runtime_error("Policy and GPU layers disagree");
    }
    int cache_mib = gpu_layers > 0 ? 256 : 0;
    if (const char * value = std::getenv("AIRLLM_GPU_CACHE_MIB")) {
        if (!std::regex_match(value, std::regex("0|128|256|384|512"))) {
            throw std::runtime_error("Invalid bounded GPU cache budget");
        }
        cache_mib = std::stoi(value);
        if (!gpu_layers && cache_mib) {
            throw std::runtime_error("GPU expert cache requires GPU weight layers");
        }
    }
    const std::string prompt = argc >= 7 ? argv[6] : "La capitale de la France est";
    std::vector<llama_token> forced;
    if (argc >= 8 && std::strcmp(argv[7], "-") != 0) {
        std::ifstream source(argv[7]);
        if (!source || std::filesystem::file_size(argv[7]) > 65536) {
            throw std::runtime_error("Invalid forced-token capture");
        }
        const std::regex pattern("\"token\"\\s*:\\s*([0-9]+)");
        std::string line;
        while (std::getline(source, line)) {
            std::smatch match;
            if (!std::regex_search(line, match, pattern)) {
                throw std::runtime_error("Invalid forced-token row");
            }
            forced.push_back(std::stoi(match[1]));
        }
        if (forced.size() != size_t(count)) {
            throw std::runtime_error("Forced-token count differs from requested count");
        }
    }
    const std::filesystem::path root(argv[2]);
    if (std::filesystem::exists(root)) {
        throw std::runtime_error("Refusing to overwrite an existing capture");
    }
    std::filesystem::create_directories(root);
    std::ofstream(root / "run.json") << "{\"threads\":" << threads << ",\"gpu_layers\":" << gpu_layers
        << ",\"count\":" << count << ",\"context\":512,\"batch\":8,\"moe_cache_mib\":"
        << cache_mib << ",\"policy\":\"" << policy << "\",\"op_offload\":" << (legacy ? "true" : "false")
        << ",\"offload_kqv\":" << (legacy ? "true" : "false")
        << ",\"forced\":" << (forced.empty() ? "false" : "true") << "}\n";
    std::ofstream knobs(root / "environment.json");
    knobs << "{";
    bool comma = false;
    for (const char * name : {"AIRLLM_MMAP_DEMAND", "AIRLLM_EXPERT_PREFETCH", "AIRLLM_DENSE_PREFETCH",
            "AIRLLM_DENSE_PREFETCH_MAX_COLS", "AIRLLM_GPU_CACHE_MIB", "AIRLLM_GLM_HC_POST", "AIRLLM_GLM_LAYER_LOCAL_HC",
            "AIRLLM_GLM_CACHE_SPLIT", "AIRLLM_PROFILE_OPS", "AIRLLM_PROFILE_PAGES"}) {
        const char * value = std::getenv(name);
        if (value) {
            if (!std::regex_match(value, std::regex("[0-9]+"))) {
                throw std::runtime_error("Non-numeric experimental knob");
            }
            knobs << (comma ? "," : "") << "\"" << name << "\":\"" << value << "\"";
            comma = true;
        }
    }
    knobs << "}\n";
    knobs.close();
    capture captured{root, std::ofstream(root / "tensors.jsonl")};
    captured.manifest.exceptions(std::ios::failbit | std::ios::badbit);
    const char * profile = std::getenv("AIRLLM_PROFILE_OPS");
    const char* pages=std::getenv("AIRLLM_PROFILE_PAGES");
    for(const auto* value:{profile,pages}) if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
        throw std::runtime_error("Profile switches require 0 or 1");
    captured.profile_pages=pages&&std::strcmp(pages,"1")==0;
    captured.profile = (profile && std::strcmp(profile, "1") == 0)||captured.profile_pages;
    if (captured.profile) {
        captured.operations.open(root / "operations.jsonl");
        captured.operations.exceptions(std::ios::failbit | std::ios::badbit);
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    ggml_backend_dev_t gpu = head_only ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
    if (head_only && (!gpu || std::string(ggml_backend_dev_name(gpu)).find("MTL") != 0)) {
        throw std::runtime_error("head-only requires a Metal device");
    }
    ggml_backend_dev_t devices[] = { gpu, nullptr };
    if (gpu_layers == 0) {
        mp.devices = devices;
    }
    llama_model_tensor_buft_override overrides[] = {
        {".*ffn_(gate|up|down)_exps.*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
    };
    if (gpu_layers > 0) {
        mp.tensor_buft_overrides = overrides;
    }
    llama_model_tensor_buft_override head_overrides[] = {
        {"^output\\.weight$", head_only ? ggml_backend_dev_buffer_type(gpu) : nullptr}, {nullptr, nullptr}
    };
    if (head_only) {
        mp.tensor_buft_overrides = head_overrides;
    }
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 3;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.op_offload = legacy;
    cp.offload_kqv = legacy;
    cp.moe_cache_size = size_t(cache_mib) * 1024 * 1024;
    cp.cb_eval = evaluate;
    cp.cb_eval_user_data = &captured;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 4;
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (n < 1 || n > 400) {
        throw std::runtime_error("Invalid diagnostic prompt length");
    }
    std::vector<llama_token> input(n);
    n = llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true);
    if (n < 1 || n > 400) {
        throw std::runtime_error("Invalid diagnostic prompt length");
    }
    std::ofstream prompt_tokens(root / "prompt_tokens.json");
    prompt_tokens << "[";
    for (int i = 0; i < n; ++i) {
        prompt_tokens << (i ? "," : "") << input[i];
    }
    prompt_tokens << "]\n";
    prompt_tokens.close();
    for (const auto token : forced) {
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("Forced token outside vocabulary");
        }
    }
    std::ofstream tokens(root / "tokens.jsonl");
    tokens.exceptions(std::ios::failbit | std::ios::badbit);
    int captured_steps = 0;
    for (int offset = 0; offset < n;) {
        const int take = std::min(8, n-offset);
        auto batch = llama_batch_get_one(input.data()+offset, take);
        if (llama_decode(ctx, batch) != 0 || captured.failed) {
            return 5;
        }
        offset += take;
    }
    for (int step = 0; step < count; ++step) {
        captured.step = step;
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            throw std::runtime_error("Missing native logits");
        }
        if (!std::all_of(logits, logits+n_vocab, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("Non-finite native logits");
        }
        const llama_token token = std::max_element(logits, logits+n_vocab)-logits;
        std::ofstream binary(root / ("logits-" + std::to_string(step) + ".f32"), std::ios::binary);
        binary.exceptions(std::ios::failbit | std::ios::badbit);
        binary.write(reinterpret_cast<const char *>(logits), n_vocab*sizeof(float));
        const auto next_input = forced.empty() ? token : forced[step];
        tokens << "{\"step\":" << step << ",\"token\":" << token << ",\"next_input_token\":"
               << next_input << ",\"vocab\":" << n_vocab << "}\n";
        tokens.flush();
        captured_steps++;
        if ((forced.empty() && llama_vocab_is_eog(vocab, token)) || step+1 == count) {
            break;
        }
        auto next = next_input;
        captured.step = step+1;
        auto batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0 || captured.failed) {
            return 6;
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::ofstream(root / "complete.json") << "{\"completed\":true,\"captured_steps\":" << captured_steps << "}\n";
    return 0;
}

#endif // AIRLLM_PROFILE_TENSOR_TEST
#endif // AIRLLM_PROFILE_RESIDENCY_TEST
