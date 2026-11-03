// Isolated native graph fixture, not a model. Never use use_ref as CPU oracle.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <CommonCrypto/CommonDigest.h>
#include <nlohmann/json.hpp>
#include <mach-o/dyld.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <poll.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
namespace {
constexpr size_t LIMIT=64u<<20, META=1u<<20, GPU=4751360;
constexpr int THREADS=10;
std::filesystem::path fatal_path;
const unsigned char * fatal_dst=nullptr;
size_t fatal_bytes=0;
void fatal_observer(const char * message) {
    // Synchronous existing GGML fatal callback, not a signal handler. Only the
    // deliberate late-scale error is intercepted; genuine other failures keep
    // the library's original abort behavior and external guard classification.
    if(message)std::fprintf(stderr,"FIXTURE_NATIVE_FATAL %s\n",message);
    if (!message || !std::strstr(message,"unsupported weight scale after completed tile")) return;
    bool unchanged=fatal_dst && fatal_bytes;
    for(size_t i=0;i<fatal_bytes && unchanged;++i) unchanged=fatal_dst[i]==0xa5;
    std::ofstream out(fatal_path);out<<json({{"expected_error","unsupported weight scale after completed tile"},
        {"destination_unpublished",unchanged},{"destination_bytes_checked",fatal_bytes},
        {"controlled_fixture_exit",86},{"cpu_fallback_performed",false}}).dump()<<'\n';out.close();
    std::_Exit(86); // terminal fixture failure; no unwind/fallback while workers wait
}
std::string digest(const void * data,size_t size) {
    if(size>UINT32_MAX) throw std::runtime_error("bounded checksum input required");
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data,CC_LONG(size),bytes);
    const char hex[]="0123456789abcdef";std::string value;
    for(unsigned char byte:bytes){value+=hex[byte>>4];value+=hex[byte&15];}return value;
}
json linkage() {
    json images=json::array();
    for(uint32_t i=0;i<_dyld_image_count();++i) {
        const char * value=_dyld_get_image_name(i);if(!value)continue;
        std::filesystem::path path=value;auto name=path.filename().string();
        if(name.rfind("libggml",0)!=0 && name.rfind("libllama",0)!=0)continue;
        auto canonical=std::filesystem::weakly_canonical(path).string();
        if(std::find(images.begin(),images.end(),canonical)==images.end())images.push_back(canonical);
    }
    return {{"loaded_native_images",images}};
}
bool compiled_feature() {
    auto reg=ggml_backend_reg_by_name("CPU");
    auto features=reg ? (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_get_features") : nullptr;
    if(features) for(auto f=features(reg);f && f->name;++f)
        if(std::strcmp(f->name,"CPU_METAL_Q8_BRIDGE_COMPILED")==0 && f->value && std::strcmp(f->value,"1")==0)return true;
    return false;
}
int integer(const char * value,int low,int high) {
    char * end=nullptr;long n=std::strtol(value,&end,10);
    if(!value || !*value || !end || *end || n<low || n>high)throw std::runtime_error("invalid bounded shape");return int(n);
}
}
int main(int argc,char ** argv) {
    try {
        if(argc!=2) {
            const char * raw=std::getenv("AIRLLM_FIXTURE_START_FD");
            if(!raw)throw std::runtime_error("owned fixture start pipe required");
            int fd=integer(raw,3,1000000);struct stat state={};
            if(fstat(fd,&state)!=0 || !S_ISFIFO(state.st_mode))throw std::runtime_error("start descriptor is not an inherited pipe");
            pollfd p={fd,POLLIN,0};char token=0;
            if(poll(&p,1,10000)!=1 || read(fd,&token,1)!=1 || token!='G')throw std::runtime_error("owner did not admit fixture before10s");
            close(fd);
        }
        ggml_backend_load_all();
        if(argc==2 && std::string(argv[1])=="--describe") {
            std::cout<<json({{"schema",1},{"fixture","cpu-mapped-q8-graph-warm-v2"},{"threads",THREADS},
                {"cpu_metal_q8_bridge_compiled",compiled_feature()},{"native_linkage",linkage()},
                {"uses_reference_mode",false},{"quantizer","native_cpu_Q8_0_from_float"}}).dump()<<'\n';return 0;
        }
        if(argc!=7)throw std::runtime_error("usage: FIXTURE OUTDIR M K N normal|subnormal-x|subnormal-w|late-subnormal-w off|on");
        const std::filesystem::path root=argv[1];
        if(!std::filesystem::is_directory(root) || std::filesystem::is_symlink(root))throw std::runtime_error("fresh owned output directory required");
        const int m=integer(argv[2],8,8192),k=integer(argv[3],2048,4096),n=integer(argv[4],1,128);
        const std::string mode=argv[5],wanted=argv[6];const char * flag=std::getenv("AIRLLM_CPU_METAL_Q8_BRIDGE");
        const bool on=flag && std::strcmp(flag,"1")==0;
        if((wanted!="off" && wanted!="on") || on!=(wanted=="on") || !compiled_feature())throw std::runtime_error("bridge request/actual feature mismatch");
        if((k!=2048 && k!=4096)||m%8 || (mode!="normal" && mode!="subnormal-x" && mode!="subnormal-w" && mode!="late-subnormal-w"))throw std::runtime_error("unadmitted fixture case");
        if(mode=="late-subnormal-w" && m<=1536)throw std::runtime_error("late refusal needs a second tile");
        const size_t row_bytes=ggml_row_size(GGML_TYPE_Q8_0,k),weight_bytes=row_bytes*m,x_bytes=size_t(k)*n*4,y_bytes=size_t(m)*n*4;
        if(row_bytes!=size_t(k/32)*34 || weight_bytes+x_bytes+y_bytes+META>LIMIT)throw std::runtime_error("fixture payload exceeds64MiB");
        ggml_init_params init={META,nullptr,true};auto ctx=ggml_init(init);if(!ctx)throw std::runtime_error("context allocation failed");
        auto backend=ggml_backend_cpu_init();if(!backend)throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend,THREADS);ggml_backend_cpu_set_use_ref(backend,false);
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,m);auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        ggml_set_name(w,"fixture.q8.weight");ggml_set_name(x,"fixture.input");auto y=ggml_mul_mat(ctx,w,x);ggml_set_name(y,"fixture.output");
        const size_t page=size_t(sysconf(_SC_PAGESIZE)),weight_offset=64;
        const size_t mapping_bytes=(weight_bytes+weight_offset+page-1)/page*page;
        void * mapping=mmap(nullptr,mapping_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
        if(mapping==MAP_FAILED)throw std::runtime_error("owned mmap allocation failed");
        void * weights=(unsigned char *)mapping+weight_offset;
        auto mapped=ggml_backend_cpu_buffer_from_ptr(weights,weight_bytes);ggml_backend_buffer_set_usage(mapped,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(ggml_backend_tensor_alloc(mapped,w,weights)!=GGML_STATUS_SUCCESS)throw std::runtime_error("mapped tensor placement failed");
        auto buft=ggml_backend_cpu_buffer_type();const size_t alignment=ggml_backend_buft_get_alignment(buft);
        const size_t y_offset=(x_bytes+alignment-1)/alignment*alignment,owned_bytes=y_offset+y_bytes;
        auto owned=ggml_backend_buft_alloc_buffer(buft,owned_bytes);if(!owned)throw std::runtime_error("owned input/output allocation failed");
        auto base=(unsigned char *)ggml_backend_buffer_get_base(owned);
        if(ggml_backend_tensor_alloc(owned,x,base)!=GGML_STATUS_SUCCESS || ggml_backend_tensor_alloc(owned,y,base+y_offset)!=GGML_STATUS_SUCCESS)throw std::runtime_error("owned tensor placement failed");
        std::vector<float> row(k);auto quantize=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
        if(!quantize)throw std::runtime_error("native quantizer missing");
        const auto prepare_begin=std::chrono::steady_clock::now();
        for(int r=0;r<m;++r) {
            for(int c=0;c<k;++c)row[c]=float((r*17+c*13)%257-128)/128.0f;
            if(mode=="subnormal-w" || (mode=="late-subnormal-w" && r==1536))for(float & value:row)value*=0x1p-9f;
            quantize(row.data(),(unsigned char *)weights+size_t(r)*row_bytes,k);
        }
        for(int t=0;t<n;++t)for(int c=0;c<k;++c) {
            float value=float((t*11+c*7)%199-99)/64.0f;if(mode=="subnormal-x")value*=0x1p-9f;
            ((float *)x->data)[size_t(t)*k+c]=value;
        }
        std::memset(y->data,0xa5,y_bytes);
        if(mprotect(mapping,mapping_bytes,PROT_READ)!=0)throw std::runtime_error("read-only mapping protection failed");
        auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
        // Only inspect the normal plan. Execution below uses the ordinary CPU
        // backend dispatcher with use_ref unchanged (false), in both processes.
        auto plan=ggml_graph_plan(graph,THREADS,nullptr);
        if(plan.use_ref || plan.n_threads!=THREADS)throw std::runtime_error("ordinary CPU10 plan required");
        const size_t logical=mapping_bytes+owned_bytes+META+row.size()*sizeof(float)+plan.work_size+(on ? GPU : 0);
        if(logical>LIMIT)throw std::runtime_error("whole fixture logical allocation exceeds64MiB");
        json report={{"schema",1},{"fixture","cpu-mapped-q8-graph-warm-v2"},{"M",m},{"K",k},{"N",n},{"case",mode},
            {"threads",THREADS},{"bridge_requested",on},{"cpu_metal_q8_bridge_compiled",compiled_feature()},
            {"execution_backend",on ? "native_cpu_with_metal_q8_bridge" : "native_cpu"},
            {"weight_buffer_type",ggml_backend_buffer_name(mapped)},{"weight_buffer_usage",int(ggml_backend_buffer_get_usage(mapped))},
            {"weight_original_plain_layout",ggml_backend_cpu_buffer_has_plain_layout(mapped)},{"uses_reference_mode",false},
            {"weight_sha256",digest(weights,weight_bytes)},{"input_sha256",digest(x->data,x_bytes)},
            {"weight_bytes",weight_bytes},{"input_output_owned_bytes",owned_bytes},{"metadata_bytes",META},{"weight_mapping_read_only",true},{"weight_mapping_bytes",mapping_bytes},{"weight_offset_in_page",weight_offset},
            {"row_scratch_bytes",row.size()*sizeof(float)},{"planned_graph_workspace_bytes",plan.work_size},
            {"planned_metal_payload_bytes",on ? GPU : 0},{"logical_payload_bound_bytes",logical},{"logical_payload_limit_bytes",LIMIT},
            {"device_allocated_observed_peak_bytes",nullptr},{"peak_rss_bytes",nullptr},{"native_linkage",linkage()},
            {"whole_model_qualified",false},{"physical_ssd_reads",nullptr}};
        report["prepare_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-prepare_begin).count();
        {std::ofstream out(root/"prepared.json");out<<report.dump(2)<<'\n';}
        if(on && mode=="late-subnormal-w") {
            fatal_dst=(const unsigned char *)y->data;fatal_bytes=y_bytes;fatal_path=root/"fatal.json";
            ggml_set_abort_callback(fatal_observer);
        }
        json iteration_seconds=json::array(),iteration_sha256=json::array();
        const auto begin=std::chrono::steady_clock::now();
        for(int iteration=0;iteration<5;++iteration) {
            const auto iteration_start=std::chrono::steady_clock::now();
            const auto status=ggml_backend_graph_compute(backend,graph);
        iteration_seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-iteration_start).count());
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("normal CPU backend graph status failed");
        const float * values=(const float *)y->data;
        for(size_t i=0;i<y_bytes/4;++i)if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite output");
        iteration_sha256.push_back(digest(y->data,y_bytes));
            if(iteration && iteration_sha256.back()!=iteration_sha256.front())throw std::runtime_error("graph reuse changed output bits");
        }
        report["compute_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        report["iteration_seconds"]=iteration_seconds;report["iteration_sha256"]=iteration_sha256;
        rusage lifetime={};if(getrusage(RUSAGE_SELF,&lifetime)!=0)throw std::runtime_error("lifetime RSS unavailable");
        report["peak_rss_bytes"]=lifetime.ru_maxrss;
        report["repetitions"]=5;report["weight_residency_scope"]="same synthetic resident weights; no SSD or GLM speed claim";
        {std::ofstream out(root/"output.f32",std::ios::binary);out.write((const char *)y->data,y_bytes);if(!out)throw std::runtime_error("output write failed");}
        report.update({{"status","completed"},{"all_outputs_finite",true},{"output_bytes",y_bytes},{"output_sha256",digest(y->data,y_bytes)}});
        {std::ofstream out(root/"result.json");out<<report.dump(2)<<'\n';}std::cout<<report.dump()<<'\n';
        ggml_backend_free(backend);ggml_backend_buffer_free(owned);ggml_backend_buffer_free(mapped);munmap(mapping,mapping_bytes);ggml_free(ctx);return 0;
    } catch(const std::exception & error) {std::cerr<<"FIXTURE_ERROR "<<error.what()<<'\n';return 65;}
}
