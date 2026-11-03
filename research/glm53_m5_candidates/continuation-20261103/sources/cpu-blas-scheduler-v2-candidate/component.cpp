// Synthetic scheduler mechanism only. No model, GPU backend or GPU dispatch.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-blas.h"
#include <nlohmann/json.hpp>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <poll.h>
#include <unistd.h>
#include <vector>
using json=nlohmann::json;
static uint32_t random_state;
static float next_float(){random_state=random_state*1664525u+1013904223u;return float(int32_t(random_state>>8)-8388608)*0x1.000002p-23f;}
static std::string digest(const void *data,size_t bytes){unsigned char h[32];CC_SHA256(data,CC_LONG(bytes),h);const char hex[]="0123456789abcdef";std::string s;for(auto c:h){s+=hex[c>>4];s+=hex[c&15];}return s;}
static json compare(const float *a,const float *b,size_t count){size_t diff=0;double error=0;size_t first=count;
    for(size_t i=0;i<count;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite output");
        if(std::memcmp(a+i,b+i,4)){++diff;if(first==count)first=i;}error=std::max(error,std::abs(double(a[i])-double(b[i])));}
    return {{"outputs",count},{"different_float32_bits",diff},{"cpu_bitexact",diff==0},{"all_finite",true},{"max_abs_error",error},
        {"first_different_output",first==count?json(nullptr):json(first)},{"reference_sha256",digest(a,count*4)},{"candidate_sha256",digest(b,count*4)}};}
static void raw(const std::filesystem::path &p,const void *data,size_t bytes){std::ofstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write((const char *)data,bytes);}
int main(int argc,char **argv){try{
    if(argc!=2)throw std::runtime_error("usage: COMPONENT NEW_OUTPUT_DIRECTORY");
    const char * gate_fd=std::getenv("AIRLLM_FIXTURE_START_FD");
    if(!gate_fd || !*gate_fd || std::strspn(gate_fd,"0123456789")!=std::strlen(gate_fd) || std::strlen(gate_fd)>6)throw std::runtime_error("owned start pipe required");
    const int fd=std::stoi(gate_fd);struct stat state={};
    if(fd<3 || fstat(fd,&state)!=0 || !S_ISFIFO(state.st_mode))throw std::runtime_error("start descriptor is not an inherited pipe");
    pollfd p={fd,POLLIN,0};char go=0;
    if(poll(&p,1,10000)!=1 || read(fd,&go,1)!=1 || go!='G')throw std::runtime_error("owner did not admit native child within10s");
    close(fd);
    const std::filesystem::path root=argv[1];if(std::filesystem::exists(root))throw std::runtime_error("fresh output directory required");std::filesystem::create_directories(root);
    // Explicitly initialize only CPU and the CPU accelerator BLAS. Do not call
    // ggml_backend_load_all, which also initializes unrelated GPU backends.
    auto cpu=ggml_backend_cpu_init();auto blas=ggml_backend_blas_init();if(!cpu||!blas)throw std::runtime_error("CPU/BLAS init failed");
    ggml_backend_cpu_set_n_threads(cpu,10);ggml_backend_cpu_set_use_ref(cpu,false);ggml_backend_blas_set_n_threads(blas,10);
    const int k=4096,m=128;const size_t weight_bytes=ggml_row_size(GGML_TYPE_Q8_0,k)*m;json rows=json::array();
    for(int n:{8,16,32})for(unsigned seed:{5305u,713962u}){
        auto ctx=ggml_init({1u<<20,nullptr,true});auto sc=ggml_init({1u<<20,nullptr,true});if(!ctx||!sc)throw std::runtime_error("metadata allocation");
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,m),x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
        ggml_set_name(w,"fixture.q8.weight");ggml_set_name(x,"fixture.input");
        auto full=ggml_mul_mat(ctx,w,x);ggml_set_name(full,"fixture.cpu_full");auto fg=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(fg,full);
        void *weights=nullptr;if(posix_memalign(&weights,64,weight_bytes))throw std::runtime_error("weight allocation");
        auto mapped=ggml_backend_cpu_buffer_from_ptr(weights,weight_bytes);ggml_backend_buffer_set_usage(mapped,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(ggml_backend_tensor_alloc(mapped,w,weights)!=GGML_STATUS_SUCCESS)throw std::runtime_error("mapped placement");
        const size_t xb=size_t(k)*n*4,yb=size_t(m)*n*4;auto owned=ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),xb+2*yb);
        if(!owned)throw std::runtime_error("input/output allocation");auto base=(unsigned char *)ggml_backend_buffer_get_base(owned);
        if(ggml_backend_tensor_alloc(owned,x,base)!=GGML_STATUS_SUCCESS||ggml_backend_tensor_alloc(owned,full,base+xb)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU placement");
        std::vector<ggml_tensor *>parts;std::vector<ggml_cgraph *>pg;
        for(int start=0;start<n;start+=8){auto view=ggml_view_2d(ctx,x,k,8,x->nb[1],size_t(start)*x->nb[1]);auto y=ggml_mul_mat(ctx,w,view);
            auto g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,y);
            if(ggml_backend_tensor_alloc(owned,y,base+xb+yb+size_t(start)*m*4)!=GGML_STATUS_SUCCESS)throw std::runtime_error("split placement");parts.push_back(y);pg.push_back(g);}
        random_state=seed;std::vector<float>line(k);auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
        if(!quant)throw std::runtime_error("native quantizer absent");
        for(int r=0;r<m;++r){for(float &v:line)v=next_float()*.05f;quant(line.data(),(unsigned char *)weights+size_t(r)*ggml_row_size(GGML_TYPE_Q8_0,k),k);}
        for(size_t i=0;i<size_t(k)*n;++i)((float *)x->data)[i]=next_float();
        if(ggml_backend_graph_compute(cpu,fg)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU full compute");
        for(auto g:pg)if(ggml_backend_graph_compute(cpu,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU8 compute");
        auto sw=ggml_new_tensor_2d(sc,GGML_TYPE_Q8_0,k,m),sx=ggml_new_tensor_2d(sc,GGML_TYPE_F32,k,n);
        ggml_set_name(sw,"fixture.q8.weight");ggml_set_name(sx,"fixture.input");
        if(ggml_backend_tensor_alloc(mapped,sw,weights)!=GGML_STATUS_SUCCESS||ggml_backend_tensor_alloc(owned,sx,base)!=GGML_STATUS_SUCCESS)throw std::runtime_error("scheduler leaf placement");
        auto sy=ggml_mul_mat(sc,sw,sx);ggml_set_name(sy,"fixture.scheduled");auto sg=ggml_new_graph_custom(sc,32,false);ggml_build_forward_expand(sg,sy);
        ggml_backend_t backends[]={blas,cpu};auto sched=ggml_backend_sched_new(backends,nullptr,2,64,false,false);if(!sched)throw std::runtime_error("scheduler init");
        if(!ggml_backend_sched_alloc_graph(sched,sg))throw std::runtime_error("scheduler allocation");
        auto selected=ggml_backend_sched_get_tensor_backend(sched,sy);if(!selected)throw std::runtime_error("actual selected backend absent");
        const std::string selected_name=ggml_backend_name(selected);
        const bool blas_supports=ggml_backend_dev_supports_op(ggml_backend_get_device(blas),sy);
        if(ggml_backend_sched_graph_compute(sched,sg)!=GGML_STATUS_SUCCESS)throw std::runtime_error("scheduler compute");ggml_backend_sched_synchronize(sched);
        std::vector<float>scheduled(size_t(m)*n);ggml_backend_tensor_get(sy,scheduled.data(),0,yb);
        const float *split=(const float *)(base+xb+yb);const auto prefix="n"+std::to_string(n)+"-seed"+std::to_string(seed);
        raw(root/(prefix+"-cpu8.f32"),split,yb);raw(root/(prefix+"-cpu-full.f32"),full->data,yb);raw(root/(prefix+"-scheduled.f32"),scheduled.data(),yb);
        rows.push_back({{"K",k},{"M",m},{"N",n},{"seed",seed},{"weight_type","q8_0"},{"weight_buffer_name",ggml_backend_buffer_name(mapped)},
            {"weight_usage",int(ggml_backend_buffer_get_usage(mapped))},{"weight_bytes",weight_bytes},{"weight_sha256",digest(weights,weight_bytes)},
            {"input_sha256",digest(x->data,xb)},{"scheduler_backends",{"BLAS","CPU"}},{"scheduler_op_offload",false},
            {"actual_selected_backend",selected_name},{"blas_supports_op",blas_supports},{"scheduler_splits",ggml_backend_sched_get_n_splits(sched)},
            {"expected_backend",n>=32?"BLAS":"CPU"},{"selected_backend_matches_threshold",selected_name==(n>=32?"BLAS":"CPU")},
            {"expected_blas_weight_dequant_workspace_bytes",n>=32?size_t(k)*m*4:0},
            {"cpu_full_vs_cpu8",compare(split,(const float *)full->data,size_t(m)*n)},
            {"scheduled_vs_cpu8",compare(split,scheduled.data(),scheduled.size())},
            {"source_scope","small actual scheduler mechanism; not a numerical proof for full M8192 or GLM"}});
        ggml_backend_sched_free(sched);ggml_free(sc);ggml_free(ctx);ggml_backend_buffer_free(owned);ggml_backend_buffer_free(mapped);std::free(weights);
    }
    rusage u={};if(getrusage(RUSAGE_SELF,&u)||u.ru_maxrss>(128u<<20))throw std::runtime_error("native lifetime128MiB cap");
    ggml_backend_free(blas);ggml_backend_free(cpu);
    json result={{"status","completed_cpu_blas_scheduler_mechanism_only"},{"model_loaded",false},{"gpu_backend_initialized",false},{"gpu_matrix_dispatches",0},
        {"uses_reference_mode",false},{"cpu_threads",10},{"requested_blas_threads",10},{"lifetime_peak_rss_bytes",u.ru_maxrss},
        {"rows",rows},{"whole_model_causality_proven",false},{"full_model_qualified",false},{"performance_qualified",false}};
    {std::ofstream f(root/"result.json");f<<result.dump(2)<<'\n';}std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}}
